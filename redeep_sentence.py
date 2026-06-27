"""
redeep_sentence.py
==================
Sentence-level ReDeEP analysis: forward-pass ECS/PKS computation on
pre-generated SOAP notes, evaluated against LUQ uncertainty scores.

For each sample in luq_out/generations/:
  1. Load the generated note (notes[0] by default).
  2. Load per-sentence LUQ uncertainty scores from luq_out/sentences/.
  3. Run a single forward pass with TransformerLens, caching attention
     patterns and residual-stream states.
  4. Compute chunk-level (sentence-level) ECS and PKS per layer.
  5. Accumulate across samples/generations, then produce separate plots:
       Fig 1 — ECS & PKS mean ± std across layers
       Fig 2 — Cohen's d with uncertain sentences across layers
       Fig 3 — AUROC across layers
       Fig 4 — Top-5 most discriminative layers for ECS and PKS

Outputs
-------
  activations/sample_NNN_gen_KK.npz  — raw (n_layers, n_sents) arrays per gen
  per_gen_auroc.csv                  — AUROC per sample, gen, layer
  per_gen_cohens_d.csv               — Cohen's d per sample, gen, layer
  layer_metrics.csv                  — pooled per-layer metrics
  sentence_scores.csv                — all sentence-level ECS/PKS/LUQ
  fig1_ecs_pks_mean_std.png
  fig2_cohens_d.png
  fig3_auroc.png
  fig4_top5_layers.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from prompts import build_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

_PERIOD_SPLIT = re.compile(r"(?<=\.)\s+(?=[A-Z])")

_SOAP_HEADER_RE = re.compile(
    r"^\s*(?:subjective|objective|assessment\s*/\s*problem\s+list|assessment|plan)\s*:\s*$",
    re.IGNORECASE,
)


def is_soap_header(sentence: str) -> bool:
    return bool(_SOAP_HEADER_RE.match(sentence.strip()))


def split_sentences(text: str) -> List[str]:
    results = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-•*#+").strip()
        if not line:
            continue
        for part in _PERIOD_SPLIT.split(line):
            part = part.strip()
            if part:
                results.append(part)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    from transformer_lens import HookedTransformer
    print(f"Loading {model_name} on {device} …")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.bfloat16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_content(transcript: str) -> str:
    return build_prompt(transcript)


def tokenize_prompt_and_note(
    model,
    transcript: str,
    note: str,
    device: str,
) -> Tuple[torch.Tensor, int]:
    tokenizer = model.tokenizer
    user_content = _build_user_content(transcript)

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": note},
    ]
    prompt_messages = messages[:1]

    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        full_text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{note}<|eot_id|>"
        )
        prompt_text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    full_ids   = tokenizer.encode(full_text,   return_tensors="pt").to(device)
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    transcript_len = prompt_ids.shape[1]
    note_len = full_ids.shape[1] - transcript_len

    if note_len <= 0:
        raise ValueError(
            f"note_len={note_len} ≤ 0: the note does not appear after the prompt."
        )

    return full_ids, transcript_len


# ─────────────────────────────────────────────────────────────────────────────
# Sentence → token span mapping
# ─────────────────────────────────────────────────────────────────────────────

def find_sentence_token_spans(
    model,
    full_tokens: torch.Tensor,
    transcript_len: int,
    sentences: List[str],
) -> List[Tuple[int, int]]:
    tokenizer = model.tokenizer
    note_token_ids = full_tokens[0, transcript_len:].tolist()
    n_note_tokens  = len(note_token_ids)

    cumulative_len = []
    note_text = ""
    for tid in note_token_ids:
        note_text += tokenizer.decode([tid])
        cumulative_len.append(len(note_text))

    def char_to_tok(char_pos: int) -> int:
        for i, clen in enumerate(cumulative_len):
            if clen > char_pos:
                return i
        return n_note_tokens - 1

    spans: List[Tuple[int, int]] = []
    search_from = 0

    for sent in sentences:
        if not sent:
            last = spans[-1][1] if spans else 0
            spans.append((last, min(last + 1, n_note_tokens)))
            continue

        idx = note_text.find(sent, search_from)
        if idx == -1:
            idx = note_text.find(sent[:20], search_from)
        if idx == -1:
            last = spans[-1][1] if spans else 0
            spans.append((last, min(last + 1, n_note_tokens)))
            continue

        char_start = idx
        char_end   = idx + len(sent)
        tok_start  = char_to_tok(max(0, char_start - 1))
        tok_end    = char_to_tok(char_end - 1) + 1

        tok_start = max(0, min(tok_start, n_note_tokens - 1))
        tok_end   = max(tok_start + 1, min(tok_end, n_note_tokens))
        spans.append((tok_start, tok_end))
        search_from = idx

    return spans


# ─────────────────────────────────────────────────────────────────────────────
# LogitLens helper
# ─────────────────────────────────────────────────────────────────────────────

def _logit_lens(model, x: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(0)
    x = model.ln_final(x)
    x = model.unembed(x)
    return x.squeeze(0).float()


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass — ECS + PKS in one pass
# ─────────────────────────────────────────────────────────────────────────────

def compute_ecs_pks_single_pass(
    model,
    tokens: torch.Tensor,
    transcript_len: int,
    sent_spans: List[Tuple[int, int]],
    device: str,
    top_k_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single forward pass returning ECS (n_layers, n_sents) and PKS (n_layers, n_sents).
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    n_sents  = len(sent_spans)
    T        = transcript_len
    k_top    = max(1, int(T * top_k_frac))

    chunk_attn: List[Optional[List]] = [None] * n_layers
    pks_all    = np.zeros((n_layers, n_sents), dtype=np.float64)
    x_last_ref = [None]
    _mid_tmp   = [None]

    def make_pattern_hook(l: int):
        def fn(value, hook):
            attn    = value[0].to(torch.float32).cpu().numpy()
            seq_len = attn.shape[-1]
            chunks  = []
            for s_start, s_end in sent_spans:
                abs_s = T + s_start
                abs_e = min(T + s_end, seq_len)
                if abs_s < abs_e:
                    w = attn[:, abs_s:abs_e, :T].mean(axis=1)
                else:
                    w = np.zeros((n_heads, T), dtype=np.float32)
                chunks.append(w)
            chunk_attn[l] = chunks
            return value
        return fn

    def make_resid_mid_hook(l: int):
        def fn(value, hook):
            _mid_tmp[0] = value[0, T:].to(torch.float32).cpu()
            return value
        return fn

    def make_resid_post_hook(l: int):
        def fn(value, hook):
            if l == n_layers - 1:
                x_last_ref[0] = value[0].to(torch.float32).cpu().numpy()

            if _mid_tmp[0] is not None:
                x_mid  = _mid_tmp[0].to(device)
                x_post = value[0, T:].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    q_mid  = torch.softmax(_logit_lens(model, x_mid),  dim=-1).cpu().numpy()
                    q_post = torch.softmax(_logit_lens(model, x_post), dim=-1).cpu().numpy()
                m   = 0.5 * (q_mid + q_post)
                eps = 1e-10
                kl1 = np.sum(q_mid  * (np.log(q_mid  + eps) - np.log(m + eps)), axis=-1)
                kl2 = np.sum(q_post * (np.log(q_post + eps) - np.log(m + eps)), axis=-1)
                tok_jsd = np.clip(0.5 * kl1 + 0.5 * kl2, 0.0, 1.0)
                for si, (s_start, s_end) in enumerate(sent_spans):
                    s_e = min(s_end, len(tok_jsd))
                    if s_start < s_e:
                        pks_all[l, si] = tok_jsd[s_start:s_e].mean()
                _mid_tmp[0] = None
            return value
        return fn

    fwd_hooks = []
    for l in range(n_layers):
        fwd_hooks += [
            (f"blocks.{l}.attn.hook_pattern", make_pattern_hook(l)),
            (f"blocks.{l}.hook_resid_mid",    make_resid_mid_hook(l)),
            (f"blocks.{l}.hook_resid_post",   make_resid_post_hook(l)),
        ]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    x_arr   = x_last_ref[0]
    x_trans = x_arr[:T]
    D       = x_arr.shape[-1]

    sent_vecs = np.zeros((n_sents, D), dtype=np.float64)
    for si, (s_start, s_end) in enumerate(sent_spans):
        abs_s = T + s_start
        abs_e = min(T + s_end, x_arr.shape[0])
        if abs_s < abs_e:
            sent_vecs[si] = x_arr[abs_s:abs_e].mean(axis=0)

    ecs_all = np.zeros((n_layers, n_sents), dtype=np.float64)
    for l in range(n_layers):
        if chunk_attn[l] is None:
            continue
        for si in range(n_sents):
            w_chunk = chunk_attn[l][si]
            s_vec   = sent_vecs[si]
            norm_s  = np.linalg.norm(s_vec)
            cos_acc = 0.0
            for h in range(n_heads):
                w_h     = w_chunk[h]
                top_idx = (np.argpartition(w_h, -k_top)[-k_top:]
                           if k_top < T else np.arange(T))
                w_top = w_h[top_idx]
                w_sum = w_top.sum()
                if w_sum < 1e-10:
                    continue
                w_top = w_top / w_sum
                e     = (x_trans[top_idx] * w_top[:, None]).sum(axis=0)
                norm_e = np.linalg.norm(e)
                denom  = norm_e * norm_s
                if denom > 1e-8:
                    cos_acc += float(np.dot(e, s_vec) / denom)
            ecs_all[l, si] = cos_acc / n_heads

    return np.clip(ecs_all, -1.0, 1.0), pks_all


# ─────────────────────────────────────────────────────────────────────────────
# Per-gen metrics: AUROC and Cohen's d
# ─────────────────────────────────────────────────────────────────────────────

def compute_auroc_per_layer(
    ecs_l: np.ndarray,    # (n_layers, n_sents)
    pks_l: np.ndarray,
    labels: np.ndarray,   # (n_sents,) binary int
) -> Tuple[np.ndarray, np.ndarray]:
    """AUROC per layer for 1−ECS and PKS. Returns (ecs_auroc, pks_auroc) each (n_layers,)."""
    from sklearn.metrics import roc_auc_score
    n_layers = ecs_l.shape[0]
    ecs_auroc = np.full(n_layers, 0.5)
    pks_auroc = np.full(n_layers, 0.5)
    y = labels.astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return ecs_auroc, pks_auroc
    for l in range(n_layers):
        try:
            ecs_auroc[l] = float(roc_auc_score(y, -ecs_l[l]))
        except Exception:
            pass
        try:
            pks_auroc[l] = float(roc_auc_score(y, pks_l[l]))
        except Exception:
            pass
    return ecs_auroc, pks_auroc


def compute_cohens_d_per_layer(
    ecs_l: np.ndarray,    # (n_layers, n_sents)
    pks_l: np.ndarray,
    labels: np.ndarray,   # (n_sents,) binary int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cohen's d per layer: (mean_hallu − mean_non_hallu) / pooled_std.
    For ECS this will typically be negative (hallucinated sentences have lower ECS).
    Returns (ecs_d, pks_d) each (n_layers,).
    """
    n_layers = ecs_l.shape[0]
    ecs_d = np.zeros(n_layers)
    pks_d = np.zeros(n_layers)
    y = labels.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 2 or n0 < 2:
        return ecs_d, pks_d
    for l in range(n_layers):
        for scores, out in [(ecs_l[l], ecs_d), (pks_l[l], pks_d)]:
            g1, g0 = scores[y], scores[~y]
            pooled = np.sqrt(
                ((n1 - 1) * g1.var(ddof=1) + (n0 - 1) * g0.var(ddof=1)) / (n1 + n0 - 2)
            )
            out[l] = (g1.mean() - g0.mean()) / (pooled + 1e-10)
    return ecs_d, pks_d


# ─────────────────────────────────────────────────────────────────────────────
# Pooled metrics (for layer_metrics.csv compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def per_layer_metrics(
    ecs_layers: np.ndarray,
    pks_layers: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    n_layers = ecs_layers.shape[0]
    keys = ["ecs_auroc", "pks_auroc", "comb_auroc",
            "ecs_auprc", "pks_auprc", "comb_auprc"]
    result = {k: np.full(n_layers, 0.5 if "auroc" in k else labels.mean())
              for k in keys}
    result["baseline_auprc"] = float(labels.mean())
    y = labels.astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return result
    for l in range(n_layers):
        ecs  = ecs_layers[l]
        pks  = pks_layers[l]
        comb = pks - ecs
        for score, auc_k, apr_k in [
            (-ecs,  "ecs_auroc",  "ecs_auprc"),
            ( pks,  "pks_auroc",  "pks_auprc"),
            ( comb, "comb_auroc", "comb_auprc"),
        ]:
            try:
                result[auc_k][l] = float(roc_auc_score(y, score))
            except ValueError:
                pass
            try:
                result[apr_k][l] = float(average_precision_score(y, score))
            except ValueError:
                pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ECS & PKS mean ± std across layers
# ─────────────────────────────────────────────────────────────────────────────

def plot_ecs_pks_mean_std(
    ecs_all: np.ndarray,   # (n_layers, total_sents)
    pks_all: np.ndarray,
    out_dir: Path,
) -> None:
    """Separate figures: mean ± std of ECS and PKS across layers (pooled over all sentences)."""
    n_layers = ecs_all.shape[0]
    layers   = np.arange(n_layers)

    ecs_mean = ecs_all.mean(axis=1)
    ecs_std  = ecs_all.std(axis=1)
    pks_mean = pks_all.mean(axis=1)
    pks_std  = pks_all.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(layers, ecs_mean, color="steelblue", lw=2, label="ECS mean")
    ax.fill_between(layers, ecs_mean - ecs_std, ecs_mean + ecs_std,
                    color="steelblue", alpha=0.2, label="±1 std")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("ECS")
    ax.set_title("ECS across layers (mean ± std, all sentences)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(layers, pks_mean, color="tomato", lw=2, label="PKS mean")
    ax.fill_between(layers, pks_mean - pks_std, pks_mean + pks_std,
                    color="tomato", alpha=0.2, label="±1 std")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("PKS")
    ax.set_title("PKS across layers (mean ± std, all sentences)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / "fig1_ecs_pks_mean_std.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Cohen's d per layer (averaged over samples × generations)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cohens_d_layers(
    avg_ecs_d: np.ndarray,   # (n_layers,) mean across gens
    avg_pks_d: np.ndarray,
    std_ecs_d: np.ndarray,   # (n_layers,) std across gens
    std_pks_d: np.ndarray,
    out_dir: Path,
) -> None:
    """Cohen's d (hallucinated vs non-hallucinated) per layer, averaged over sample×gen pairs."""
    n_layers = len(avg_ecs_d)
    layers   = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, avg_ecs_d, color="steelblue", lw=2,
            marker="o", ms=3, label="ECS Cohen's d")
    ax.fill_between(layers,
                    avg_ecs_d - std_ecs_d, avg_ecs_d + std_ecs_d,
                    color="steelblue", alpha=0.15)

    ax.plot(layers, avg_pks_d, color="tomato", lw=2,
            marker="s", ms=3, label="PKS Cohen's d")
    ax.fill_between(layers,
                    avg_pks_d - std_pks_d, avg_pks_d + std_pks_d,
                    color="tomato", alpha=0.15)

    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cohen's d  (hallucinated − non-hallucinated)")
    ax.set_title(
        "Cohen's d — ECS & PKS vs uncertain sentences\n"
        "(mean ± std across samples × generations, shaded)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / "fig2_cohens_d.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — AUROC per layer (averaged over samples × generations)
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc_layers(
    avg_ecs_auroc: np.ndarray,
    avg_pks_auroc: np.ndarray,
    std_ecs_auroc: np.ndarray,
    std_pks_auroc: np.ndarray,
    out_dir: Path,
) -> None:
    """AUROC per layer for 1−ECS and PKS, averaged over sample×gen pairs."""
    n_layers = len(avg_ecs_auroc)
    layers   = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, avg_ecs_auroc, color="steelblue", lw=2,
            marker="o", ms=3, label="1−ECS  AUROC")
    ax.fill_between(layers,
                    avg_ecs_auroc - std_ecs_auroc, avg_ecs_auroc + std_ecs_auroc,
                    color="steelblue", alpha=0.15)

    ax.plot(layers, avg_pks_auroc, color="tomato", lw=2,
            marker="s", ms=3, label="PKS  AUROC")
    ax.fill_between(layers,
                    avg_pks_auroc - std_pks_auroc, avg_pks_auroc + std_pks_auroc,
                    color="tomato", alpha=0.15)

    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance (0.5)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUROC")
    ax.set_ylim([0.3, 1.0])
    ax.set_title(
        "AUROC — ECS & PKS vs hallucinated sentences\n"
        "(mean ± std across samples × generations, shaded)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / "fig3_auroc.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Top-5 discriminative layers
# ─────────────────────────────────────────────────────────────────────────────

def plot_top5_layers(
    avg_ecs_auroc: np.ndarray,
    avg_pks_auroc: np.ndarray,
    out_dir: Path,
) -> None:
    """Bar chart of top-5 layers for ECS and PKS by AUROC."""
    top5_ecs = np.argsort(avg_ecs_auroc)[-5:][::-1]
    top5_pks = np.argsort(avg_pks_auroc)[-5:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, top5, vals, color, title in [
        (axes[0], top5_ecs, avg_ecs_auroc, "steelblue", "Top 5 ECS layers  (1−ECS AUROC)"),
        (axes[1], top5_pks, avg_pks_auroc, "tomato",    "Top 5 PKS layers  (PKS AUROC)"),
    ]:
        bars = ax.bar(range(5), vals[top5], color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"L{l}" for l in top5], fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.set_ylim([0.4, min(1.02, vals[top5].max() + 0.08)])
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
        for bar, v in zip(bars, vals[top5]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    p = out_dir / "fig4_top5_layers.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentence-level ReDeEP ECS/PKS analysis on pre-generated SOAP notes"
    )
    parser.add_argument("--model",
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device",
                        default=(
                            "cuda" if torch.cuda.is_available() else
                            "mps"  if torch.backends.mps.is_available() else
                            "cpu"
                        ))
    parser.add_argument("--generations",
                        default="luq_out/llama/generations")
    parser.add_argument("--sentences",
                        default="luq_out/llama/sentences")
    parser.add_argument("--out",
                        default="sae_experiments/redeep_out")
    parser.add_argument("--hallu-thresh", type=float, default=0.5)
    parser.add_argument("--top-k-frac",   type=float, default=0.10)
    parser.add_argument("--note-idx",     type=int,   default=None,
                        help="Restrict to one note index (default: all K)")
    parser.add_argument("--samples",      type=int,   default=None,
                        help="Max number of samples to process")
    parser.add_argument("--no-cache",     action="store_true")
    args = parser.parse_args()

    gen_dir  = Path(args.generations)
    sent_dir = Path(args.sentences)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = out_dir / "sample_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    model    = load_model(args.model, args.device)
    n_layers = model.cfg.n_layers
    print(f"  n_layers={n_layers}  n_heads={model.cfg.n_heads}  d_model={model.cfg.d_model}")

    gen_files = sorted(gen_dir.glob("sample_*_generations.json"))
    if args.samples is not None:
        gen_files = gen_files[:args.samples]
    print(f"  Processing {len(gen_files)} sample(s) …\n")

    # Pooled accumulators (concatenated over all sentences)
    all_ecs: List[np.ndarray] = []
    all_pks: List[np.ndarray] = []
    all_luq: List[np.ndarray] = []

    # Per-gen records for averaged metrics (items 2, 3, 6)
    per_gen_ecs_auroc: List[np.ndarray] = []
    per_gen_pks_auroc: List[np.ndarray] = []
    per_gen_ecs_d:     List[np.ndarray] = []
    per_gen_pks_d:     List[np.ndarray] = []
    per_gen_auroc_rows:  List[Dict] = []
    per_gen_cohens_rows: List[Dict] = []

    # ── Per-sample, per-note loop ─────────────────────────────────────────────
    for gen_path in gen_files:
        m = re.search(r"sample_(\d+)", gen_path.stem)
        si = int(m.group(1)) if m else -1

        with open(gen_path) as f:
            gen_data = json.load(f)

        transcript = gen_data["transcript"]
        notes      = gen_data["notes"]
        K          = len(notes)

        note_indices = [args.note_idx] if args.note_idx is not None else range(K)

        for k in note_indices:
            if k >= K:
                print(f"[sample_{si:03d}] note_idx {k} >= K={K} — skip")
                continue

            note_name = f"sample_{si:03d}_note_{k:02d}"
            csv_path  = sent_dir / f"{note_name}_sentences.csv"

            if not csv_path.exists():
                print(f"[{note_name}] No sentences CSV — skip")
                continue

            sent_df = pd.read_csv(csv_path)
            if sent_df.empty:
                print(f"[{note_name}] Empty sentences CSV — skip")
                continue

            note       = notes[k]
            sentences  = sent_df["sentence"].tolist()
            luq_scores = sent_df["uncertainty"].values.astype(np.float64)

            keep       = [not is_soap_header(s) for s in sentences]
            sentences  = [s for s, m in zip(sentences, keep) if m]
            luq_scores = luq_scores[keep]

            if not sentences:
                print(f"[{note_name}] No content sentences after header filter — skip")
                continue

            n_sents = len(sentences)
            print(f"[{note_name}]  {n_sents} sentences …", end="  ", flush=True)

            try:
                tokens, transcript_len = tokenize_prompt_and_note(
                    model, transcript, note, args.device
                )
            except Exception as exc:
                print(f"tokenise ERROR: {exc}")
                continue

            note_len = tokens.shape[1] - transcript_len
            print(f"T={transcript_len} N={note_len}", end="  ", flush=True)

            try:
                spans = find_sentence_token_spans(
                    model, tokens, transcript_len, sentences
                )
            except Exception as exc:
                print(f"spans ERROR: {exc}")
                continue

            # ── Load from disk cache if available ─────────────────────────────
            ecs_cache = cache_dir / f"{note_name}_ecs.npy"
            pks_cache = cache_dir / f"{note_name}_pks.npy"

            if not args.no_cache and ecs_cache.exists() and pks_cache.exists():
                ecs_l = np.load(str(ecs_cache))
                pks_l = np.load(str(pks_cache))
                n_valid = min(ecs_l.shape[1], len(luq_scores))
                ecs_l = ecs_l[:, :n_valid]
                pks_l = pks_l[:, :n_valid]
                luq_v = luq_scores[:n_valid]
                print(f"ECS={ecs_l.mean():.3f}  PKS={pks_l.mean():.3f}  (cached)")
            else:
                try:
                    ecs_l, pks_l = compute_ecs_pks_single_pass(
                        model, tokens, transcript_len, spans,
                        device=args.device, top_k_frac=args.top_k_frac,
                    )
                except Exception as exc:
                    print(f"forward ERROR: {exc}")
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    continue

                np.save(str(ecs_cache), ecs_l)
                np.save(str(pks_cache), pks_l)

                n_valid = min(ecs_l.shape[1], len(luq_scores))
                ecs_l = ecs_l[:, :n_valid]
                pks_l = pks_l[:, :n_valid]
                luq_v = luq_scores[:n_valid]
                print(f"ECS={ecs_l.mean():.3f}  PKS={pks_l.mean():.3f}  OK")

            # ── Item 5: save per-gen activation file (all layers, backtrackable) ──
            np.savez_compressed(
                act_dir / f"sample_{si:03d}_gen_{k:02d}.npz",
                ecs    = ecs_l.astype(np.float32),   # (n_layers, n_sents)
                pks    = pks_l.astype(np.float32),
                luq    = luq_v.astype(np.float32),
                labels = (luq_v > args.hallu_thresh).astype(np.int8),
            )

            # ── Accumulate pooled data ─────────────────────────────────────────
            all_ecs.append(ecs_l)
            all_pks.append(pks_l)
            all_luq.append(luq_v)

            # ── Item 6: per-gen AUROC and Cohen's d ───────────────────────────
            y_gen = (luq_v > args.hallu_thresh).astype(int)

            auroc_e, auroc_p = compute_auroc_per_layer(ecs_l, pks_l, y_gen)
            cd_e,    cd_p    = compute_cohens_d_per_layer(ecs_l, pks_l, y_gen)

            if y_gen.sum() > 0 and (1 - y_gen).sum() > 0:
                per_gen_ecs_auroc.append(auroc_e)
                per_gen_pks_auroc.append(auroc_p)

            per_gen_ecs_d.append(cd_e)
            per_gen_pks_d.append(cd_p)

            for l in range(n_layers):
                per_gen_auroc_rows.append({
                    "sample": si, "gen": k, "layer": l,
                    "ecs_auroc": round(float(auroc_e[l]), 5),
                    "pks_auroc": round(float(auroc_p[l]), 5),
                })
                per_gen_cohens_rows.append({
                    "sample": si, "gen": k, "layer": l,
                    "ecs_cohens_d": round(float(cd_e[l]), 5),
                    "pks_cohens_d": round(float(cd_p[l]), 5),
                })

    if not all_ecs:
        print("\nNo samples processed. Check paths and model loading.")
        sys.exit(1)

    # ── Stack pooled arrays ───────────────────────────────────────────────────
    ecs_all = np.concatenate(all_ecs, axis=1)   # (n_layers, total_sents)
    pks_all = np.concatenate(all_pks, axis=1)
    luq_all = np.concatenate(all_luq)
    labels  = (luq_all > args.hallu_thresh).astype(int)

    print(f"\nTotal sentences : {luq_all.shape[0]}")
    print(f"Hallucinated    : {labels.sum()}  ({100 * labels.mean():.1f} %)")
    print(f"ECS overall mean: {ecs_all.mean():.4f}")
    print(f"PKS overall mean: {pks_all.mean():.4f}")

    # ── Averaged per-gen metrics ──────────────────────────────────────────────
    # Cohen's d: average over ALL gens (including ones with only one class present)
    avg_ecs_d   = np.mean(per_gen_ecs_d, axis=0)
    std_ecs_d   = np.std(per_gen_ecs_d,  axis=0)
    avg_pks_d   = np.mean(per_gen_pks_d, axis=0)
    std_pks_d   = np.std(per_gen_pks_d,  axis=0)

    # AUROC: average over gens where both classes are present
    if per_gen_ecs_auroc:
        avg_ecs_auroc = np.mean(per_gen_ecs_auroc, axis=0)
        std_ecs_auroc = np.std(per_gen_ecs_auroc,  axis=0)
        avg_pks_auroc = np.mean(per_gen_pks_auroc, axis=0)
        std_pks_auroc = np.std(per_gen_pks_auroc,  axis=0)
    else:
        print("  Warning: no per-gen pair had both classes — AUROC plots will be uninformative.")
        avg_ecs_auroc = std_ecs_auroc = avg_pks_auroc = std_pks_auroc = np.full(n_layers, 0.5)

    top5_ecs = np.argsort(avg_ecs_auroc)[-5:][::-1]
    top5_pks = np.argsort(avg_pks_auroc)[-5:][::-1]
    print(f"\n  Top-5 ECS layers (AUROC): {list(top5_ecs)}")
    print(f"  Top-5 PKS layers (AUROC): {list(top5_pks)}")

    # ── Pooled metrics (for CSV) ──────────────────────────────────────────────
    pooled = per_layer_metrics(ecs_all, pks_all, labels)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame(per_gen_auroc_rows).to_csv(
        out_dir / "per_gen_auroc.csv", index=False
    )
    print(f"\n  Saved {out_dir / 'per_gen_auroc.csv'}")

    pd.DataFrame(per_gen_cohens_rows).to_csv(
        out_dir / "per_gen_cohens_d.csv", index=False
    )
    print(f"  Saved {out_dir / 'per_gen_cohens_d.csv'}")

    pd.DataFrame({
        "layer":       np.arange(n_layers),
        "avg_ecs_auroc": avg_ecs_auroc,
        "avg_pks_auroc": avg_pks_auroc,
        "std_ecs_auroc": std_ecs_auroc,
        "std_pks_auroc": std_pks_auroc,
        "avg_ecs_d":     avg_ecs_d,
        "avg_pks_d":     avg_pks_d,
        "std_ecs_d":     std_ecs_d,
        "std_pks_d":     std_pks_d,
        "pooled_ecs_auroc":  pooled["ecs_auroc"],
        "pooled_pks_auroc":  pooled["pks_auroc"],
        "pooled_comb_auroc": pooled["comb_auroc"],
        "pooled_ecs_auprc":  pooled["ecs_auprc"],
        "pooled_pks_auprc":  pooled["pks_auprc"],
    }).to_csv(out_dir / "layer_metrics.csv", index=False)
    print(f"  Saved {out_dir / 'layer_metrics.csv'}")

    pd.DataFrame({
        **{f"ecs_l{l}": ecs_all[l] for l in range(n_layers)},
        **{f"pks_l{l}": pks_all[l] for l in range(n_layers)},
        "luq_score": luq_all,
        "label":     labels,
    }).to_csv(out_dir / "sentence_scores.csv", index=False)
    print(f"  Saved {out_dir / 'sentence_scores.csv'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nPlotting …")
    plot_ecs_pks_mean_std(ecs_all, pks_all, out_dir)
    plot_cohens_d_layers(avg_ecs_d, avg_pks_d, std_ecs_d, std_pks_d, out_dir)
    plot_auroc_layers(avg_ecs_auroc, avg_pks_auroc, std_ecs_auroc, std_pks_auroc, out_dir)
    plot_top5_layers(avg_ecs_auroc, avg_pks_auroc, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
