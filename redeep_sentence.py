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
from typing import List, Optional, Tuple

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

_SOAP_HEADER_RE = re.compile(
    r"^\s*(?:subjective|objective|assessment\s*/\s*problem\s+list|assessment|plan)\s*:\s*$",
    re.IGNORECASE,
)


def is_soap_header(sentence: str) -> bool:
    return bool(_SOAP_HEADER_RE.match(sentence.strip()))


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


# Exact Llama 3.1 chat scaffold matching generation: a SINGLE user message,
# NO system role. We build the string by hand rather than using
# tokenizer.apply_chat_template(), because the official Llama 3.1 template
# UNCONDITIONALLY injects a
#     <|start_header_id|>system<|end_header_id|>\n\n
#     Cutting Knowledge Date: December 2023\n
#     Today Date: <date>\n\n<|eot_id|>
# block even when no system message is supplied. Those ~24 tokens would be
# counted inside T (the external-context region) and pollute ECS attention.
# See memory: project-prompt-format.
_PROMPT_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
_PROMPT_MIDDLE = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def tokenize_prompt_and_note(
    model,
    transcript: str,
    note: str,
    device: str,
) -> Tuple[torch.Tensor, int]:
    """
    Returns (full_ids, transcript_len) where transcript_len == T is the number
    of tokens BEFORE the note begins, i.e. everything the note attends back to.

    transcript_len is derived from the longest common token prefix between the
    prompt-only encoding and the full encoding. This is robust to BPE boundary
    merging at the prompt→note seam: whatever token first diverges marks the
    start of the note, so we never silently mis-count T by ±1.
    """
    tokenizer = model.tokenizer
    user_content = _build_user_content(transcript)

    prompt_text = f"{_PROMPT_PREFIX}{user_content}{_PROMPT_MIDDLE}"
    full_text   = f"{prompt_text}{note}<|eot_id|>"

    # add_special_tokens=False — the <|begin_of_text|> BOS is already in the string.
    full_ids   = tokenizer.encode(full_text,   return_tensors="pt",
                                  add_special_tokens=False).to(device)
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt",
                                  add_special_tokens=False).to(device)

    # Longest common token prefix → start of the note.
    f0 = full_ids[0]
    p0 = prompt_ids[0]
    max_cmp = min(f0.shape[0], p0.shape[0])
    if max_cmp == 0:
        raise ValueError("Empty tokenisation.")
    eq = (f0[:max_cmp] == p0[:max_cmp])
    diverge = torch.nonzero(~eq, as_tuple=False)
    transcript_len = int(diverge[0].item()) if diverge.numel() > 0 else max_cmp

    if transcript_len != p0.shape[0]:
        # Boundary merge between the assistant header and the note's first token.
        print(f"    [warn] prompt/full token prefix diverged at {transcript_len} "
              f"(prompt_ids len={p0.shape[0]}) — boundary merge at note seam.")

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
) -> Tuple[List[Tuple[int, int]], int, int]:
    """
    Map each sentence string to a half-open token span [tok_start, tok_end)
    in the NOTE region (token indices relative to transcript_len).

    Returns (spans, n_exact_fail, n_fuzzy) where:
      n_exact_fail — sentences that could not be located at all (degenerate span)
      n_fuzzy      — sentences located only via a prefix/suffix anchor, not exact

    Robustness fixes:
      * char→token map built by cumulative PREFIX decoding, which reconstructs
        multi-byte (byte-fallback) tokens correctly — per-token decode does not.
      * trailing special tokens (e.g. <|eot_id|>) are excluded from the searched
        text so they cannot inflate char offsets.
      * search cursor advances to char_end (monotonic, non-overlapping spans).
      * prefix-anchored fallback clamps char_end via a suffix anchor so it cannot
        overrun into the following sentence.
    """
    tokenizer = model.tokenizer
    note_token_ids = full_tokens[0, transcript_len:].tolist()

    # Drop trailing special tokens (eos / eot) from the searchable region so the
    # literal "<|eot_id|>" rendering does not pollute char offsets.
    special_ids = set(tokenizer.all_special_ids or [])
    n_note_tokens = len(note_token_ids)
    n_search = n_note_tokens
    while n_search > 0 and note_token_ids[n_search - 1] in special_ids:
        n_search -= 1

    # Cumulative char length after each token, via prefix decoding.
    cumulative_len: List[int] = []
    note_text = ""
    for i in range(n_search):
        note_text = tokenizer.decode(note_token_ids[: i + 1],
                                     skip_special_tokens=False)
        cumulative_len.append(len(note_text))

    def char_to_tok(char_pos: int) -> int:
        """First token whose decoded text extends past char_pos (token containing it)."""
        for i, clen in enumerate(cumulative_len):
            if clen > char_pos:
                return i
        return max(0, n_search - 1)

    def locate(sent: str, start: int) -> Tuple[int, int, str]:
        """Return (char_start, char_end, mode) — mode in {'exact','fuzzy','fail'}."""
        idx = note_text.find(sent, start)
        if idx != -1:
            return idx, idx + len(sent), "exact"
        # Anchor on a leading slice; bound the end with a trailing slice.
        for plen in (40, 24, 16):
            if len(sent) < plen:
                continue
            j = note_text.find(sent[:plen], start)
            if j == -1:
                continue
            k = note_text.find(sent[-plen:], j)
            end = (k + plen) if k != -1 else (j + len(sent))
            return j, min(end, len(note_text)), "fuzzy"
        return -1, -1, "fail"

    spans: List[Tuple[int, int]] = []
    search_from = 0
    n_exact_fail = 0
    n_fuzzy = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            last = spans[-1][1] if spans else search_from
            tok = min(max(last, 0), max(0, n_search - 1))
            spans.append((tok, min(tok + 1, n_search)))
            n_exact_fail += 1
            continue

        char_start, char_end, mode = locate(sent, search_from)
        if mode == "fail":
            # Could not anchor anywhere ahead of the cursor: degenerate span just
            # after the previous sentence. Do NOT rewind the cursor.
            tok = spans[-1][1] if spans else 0
            tok = min(max(tok, 0), max(0, n_search - 1))
            spans.append((tok, min(tok + 1, n_search)))
            n_exact_fail += 1
            continue
        if mode == "fuzzy":
            n_fuzzy += 1

        tok_start = char_to_tok(char_start)
        tok_end   = char_to_tok(max(char_start, char_end - 1)) + 1

        tok_start = max(0, min(tok_start, n_search - 1))
        tok_end   = max(tok_start + 1, min(tok_end, n_search))
        spans.append((tok_start, tok_end))

        # Monotonic, non-overlapping: advance past this match.
        search_from = char_end

    return spans, n_exact_fail, n_fuzzy


# ─────────────────────────────────────────────────────────────────────────────
# LogitLens helper
# ─────────────────────────────────────────────────────────────────────────────

def _logit_lens(model, x: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(0)
    x = model.ln_final(x)
    x = model.unembed(x)
    return x.squeeze(0).float()


# ─────────────────────────────────────────────────────────────────────────────
# Copying-head identification (static, from model weights)
# ─────────────────────────────────────────────────────────────────────────────

def compute_copying_head_scores(model) -> np.ndarray:
    """
    Per-head copying score from the OV circuit (Elhage et al. 2021; ReDeEP App. B).

    A head exhibits copying behaviour when its OV map W_OV = W_V W_O has eigenvalues
    with positive real part: content written into the residual stream reinforces the
    same direction, i.e. the head copies the attended token's information forward.
    We score each head by the fraction of OV eigenvalues with positive real part.

    Tractable form: W_OV is (d_model × d_model) but rank ≤ d_head, and the nonzero
    spectrum of W_V W_O equals that of W_O W_V (d_head × d_head). We therefore
    eigendecompose the small d_head × d_head matrix per head.

    NOTE on faithfulness: this is the OV-matrix copying statistic in the residual
    basis. ReDeEP's exact procedure scores the full vocabulary circuit
    M = W_E W_OV W_U via a trace + Gershgorin-circle + IQR approximation (App. B),
    which the paper itself adopts only because the V×V eigenproblem is intractable.
    The residual-basis OV spectrum is the standard, tractable proxy for the same
    copying property and is documented here as a deliberate, justified simplification.

    Returns (n_layers, n_heads) array of scores in [0, 1].
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    W_V = model.W_V  # (n_layers, n_heads, d_model, d_head)
    W_O = model.W_O  # (n_layers, n_heads, d_head, d_model)
    with torch.no_grad():
        for l in range(n_layers):
            for h in range(n_heads):
                wv = W_V[l, h].to(torch.float32)            # (d_model, d_head)
                wo = W_O[l, h].to(torch.float32)            # (d_head, d_model)
                m_small = (wo @ wv).cpu().numpy()           # (d_head, d_head)
                eig = np.linalg.eigvals(m_small)
                scores[l, h] = float(np.mean(eig.real > 0.0))
    return scores


def select_copying_heads(scores: np.ndarray, thresh: float) -> np.ndarray:
    """Boolean (n_layers, n_heads) mask: heads with copying score > thresh."""
    return scores > thresh


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
    copying_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single forward pass. Returns (ecs_mean, ecs_copy, pks), each (n_layers, n_sents).

    ECS — faithful to ReDeEP Eq 3 at sentence (chunk) granularity:
      * For layer l, head h: chunk attention a^{l,h} = mean over the sentence's note
        tokens of attention onto the T context tokens (chunk-level pooling, §4.2).
      * Attended set I^{l,h} = top-k% context tokens by a^{l,h}.
      * e^{l,h} = PLAIN MEAN of the FINAL-LAYER (x^L) hidden states of I^{l,h}
        (Eq 3 uses mean-pooling, and x^L — the final layer — not the per-layer
        residual; the layer index l enters ONLY through the attention).
      * ECS^{l,h} = cos(e^{l,h}, sentvec^L), sentvec^L = mean final-layer hidden
        states of the sentence's tokens.
      * ecs_mean  = mean over all heads (complete per-layer curve).
      * ecs_copy  = mean over Copying Heads at that layer (NaN if a layer has none);
        this is the faithful ReDeEP external-context signal.

    PKS — faithful to ReDeEP Eq 4-5: per-layer JSD between logit-lens(resid_mid) and
    logit-lens(resid_post), averaged over the sentence's note tokens.
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    n_sents  = len(sent_spans)
    T        = transcript_len
    k_top    = max(1, int(T * top_k_frac))

    chunk_attn: List[Optional[List]] = [None] * n_layers
    pks_all = np.zeros((n_layers, n_sents), dtype=np.float64)
    x_last  = [None]
    _mid_tmp = [None]

    def make_pattern_hook(l: int):
        def fn(value, hook):
            attn    = value[0].to(torch.float32).cpu().numpy()
            seq_len = attn.shape[-1]
            chunks  = []
            for s_start, s_end in sent_spans:
                abs_s = T + s_start
                abs_e = min(T + s_end, seq_len)
                if abs_s < abs_e:
                    w = attn[:, abs_s:abs_e, :T].mean(axis=1)   # (n_heads, T)
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
            # ── PKS: JSD between resid_mid and resid_post logit-lens (note tokens) ──
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

            # ── Capture FINAL-LAYER hidden states (x^L) for ECS (Eq 3) ──
            if l == n_layers - 1:
                x_last[0] = value[0].to(torch.float32).cpu().numpy()
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

    # ── ECS post-pass: final-layer (x^L) vectors, per-layer attention ──
    x_arr   = x_last[0]
    x_trans = x_arr[:T]                                   # final-layer context vectors
    seq_len = x_arr.shape[0]

    sent_vecs = np.full((n_sents, x_arr.shape[1]), np.nan, dtype=np.float64)
    sent_norm = np.zeros(n_sents, dtype=np.float64)
    for si, (s_start, s_end) in enumerate(sent_spans):
        abs_s = T + s_start
        abs_e = min(T + s_end, seq_len)
        if abs_s < abs_e:
            v = x_arr[abs_s:abs_e].mean(axis=0)
            sent_vecs[si] = v
            sent_norm[si] = np.linalg.norm(v)

    ecs_lh = np.full((n_layers, n_heads, n_sents), np.nan, dtype=np.float64)
    for l in range(n_layers):
        if chunk_attn[l] is None:
            continue
        for si in range(n_sents):
            s_vec  = sent_vecs[si]
            norm_s = sent_norm[si]
            if norm_s < 1e-8 or not np.isfinite(norm_s):
                continue
            w_chunk = chunk_attn[l][si]
            for h in range(n_heads):
                w_h     = w_chunk[h]
                top_idx = (np.argpartition(w_h, -k_top)[-k_top:]
                           if k_top < T else np.arange(T))
                e      = x_trans[top_idx].mean(axis=0)       # PLAIN mean-pool (Eq 3)
                norm_e = np.linalg.norm(e)
                denom  = norm_e * norm_s
                if denom > 1e-8:
                    ecs_lh[l, h, si] = float(np.dot(e, s_vec) / denom)

    # Aggregate over heads.
    ecs_mean = np.nanmean(ecs_lh, axis=1)                     # (n_layers, n_sents)
    if copying_mask is not None and copying_mask.any():
        ecs_copy = np.full((n_layers, n_sents), np.nan, dtype=np.float64)
        for l in range(n_layers):
            heads = np.where(copying_mask[l])[0]
            if heads.size:
                ecs_copy[l] = np.nanmean(ecs_lh[l, heads], axis=0)
    else:
        ecs_copy = np.full((n_layers, n_sents), np.nan, dtype=np.float64)

    return (np.clip(ecs_mean, -1.0, 1.0),
            np.clip(ecs_copy, -1.0, 1.0),
            pks_all)


# ─────────────────────────────────────────────────────────────────────────────
# Per-gen metrics: AUROC and Cohen's d
# ─────────────────────────────────────────────────────────────────────────────

def auroc_per_layer_single(
    score_l: np.ndarray,    # (n_layers, n_sents)
    labels: np.ndarray,     # (n_sents,) binary int
    hallu_high: bool,       # True if HIGHER score ⇒ more likely hallucinated (PKS); False for ECS
) -> np.ndarray:
    """
    AUROC per layer for one signal. NaN at layers whose score is not finite
    (e.g. a copying-head ECS layer with no copying heads) or when only one class
    is present. Returns (n_layers,).
    """
    from sklearn.metrics import roc_auc_score
    n_layers = score_l.shape[0]
    out = np.full(n_layers, np.nan)
    y = labels.astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return out
    sgn = 1.0 if hallu_high else -1.0
    for l in range(n_layers):
        s = score_l[l]
        finite = np.isfinite(s)                 # drop degenerate (NaN) sentences only
        if finite.sum() < 2:
            continue                            # whole layer undefined (e.g. no copying heads)
        yy, ss = y[finite], s[finite]
        if yy.sum() == 0 or (1 - yy).sum() == 0:
            continue
        try:
            out[l] = float(roc_auc_score(yy, sgn * ss))
        except Exception:
            pass
    return out


def cohens_d_per_layer_single(
    score_l: np.ndarray,    # (n_layers, n_sents)
    labels: np.ndarray,     # (n_sents,) binary int
) -> np.ndarray:
    """
    Cohen's d per layer: (mean_hallu − mean_non_hallu) / pooled_std, for one signal.
    For ECS this is typically negative (hallucinated sentences have lower ECS); for
    PKS positive. NaN at non-finite layers or when a class has < 2 members.
    Returns (n_layers,).
    """
    n_layers = score_l.shape[0]
    out = np.full(n_layers, np.nan)
    y_all = labels.astype(bool)
    for l in range(n_layers):
        s = score_l[l]
        finite = np.isfinite(s)                 # drop degenerate (NaN) sentences only
        y = y_all & finite
        not_y = (~y_all) & finite
        n1, n0 = int(y.sum()), int(not_y.sum())
        if n1 < 2 or n0 < 2:
            continue
        g1, g0 = s[y], s[not_y]
        pooled = np.sqrt(
            ((n1 - 1) * g1.var(ddof=1) + (n0 - 1) * g0.var(ddof=1)) / (n1 + n0 - 2)
        )
        out[l] = (g1.mean() - g0.mean()) / (pooled + 1e-10)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ECS & PKS mean ± std across layers
# ─────────────────────────────────────────────────────────────────────────────

def plot_ecs_pks_mean_std(
    ecs_all: np.ndarray,        # (n_layers, total_sents) all-heads ECS
    ecs_copy_all: np.ndarray,   # (n_layers, total_sents) copying-head ECS (may be NaN)
    pks_all: np.ndarray,
    out_dir: Path,
) -> None:
    """mean ± std of ECS (all heads), ECS (copying heads), and PKS across layers."""
    n_layers = ecs_all.shape[0]
    layers   = np.arange(n_layers)

    panels = [
        (ecs_all,      "steelblue", "ECS (all heads)"),
        (ecs_copy_all, "seagreen",  "ECS (copying heads)"),
        (pks_all,      "tomato",    "PKS"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, (arr, color, title) in zip(axes, panels):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(arr, axis=1)
            std  = np.nanstd(arr, axis=1)
        ax.plot(layers, mean, color=color, lw=2, label=f"{title} mean")
        ax.fill_between(layers, mean - std, mean + std,
                        color=color, alpha=0.2, label="±1 std")
        ax.set_xlabel("Layer index")
        ax.set_ylabel(title.split(" (")[0])
        ax.set_title(f"{title} across layers (mean ± std)")
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

def _plot_layer_lines(series, out_path, ylabel, title, hline, ylim=None):
    """series: list of (avg, std, color, marker, label). avg/std are (n_layers,) w/ NaN ok."""
    n_layers = len(series[0][0])
    layers   = np.arange(n_layers)
    fig, ax  = plt.subplots(figsize=(10, 5))
    for avg, std, color, marker, label in series:
        ax.plot(layers, avg, color=color, lw=2, marker=marker, ms=3, label=label)
        if std is not None:
            ax.fill_between(layers, avg - std, avg + std, color=color, alpha=0.15)
    if hline is not None:
        ax.axhline(hline, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_cohens_d_layers(
    avg_ecs_d, std_ecs_d,
    avg_ecscopy_d, std_ecscopy_d,
    avg_pks_d, std_pks_d,
    out_dir: Path,
) -> None:
    """Cohen's d (hallucinated vs non-hallucinated) per layer, averaged over sample×gen."""
    _plot_layer_lines(
        [
            (avg_ecs_d,     std_ecs_d,     "steelblue", "o", "ECS (all heads)"),
            (avg_ecscopy_d, std_ecscopy_d, "seagreen",  "^", "ECS (copying heads)"),
            (avg_pks_d,     std_pks_d,     "tomato",    "s", "PKS"),
        ],
        out_dir / "fig2_cohens_d.png",
        ylabel="Cohen's d  (hallucinated − non-hallucinated)",
        title="Cohen's d — ECS & PKS vs uncertain sentences\n"
              "(mean ± std across samples × generations, shaded)",
        hline=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — AUROC per layer (averaged over samples × generations)
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc_layers(
    avg_ecs_auroc, std_ecs_auroc,
    avg_ecscopy_auroc, std_ecscopy_auroc,
    avg_pks_auroc, std_pks_auroc,
    out_dir: Path,
) -> None:
    """AUROC per layer for 1−ECS (all heads), 1−ECS (copying heads), and PKS."""
    _plot_layer_lines(
        [
            (avg_ecs_auroc,     std_ecs_auroc,     "steelblue", "o", "1−ECS (all heads)"),
            (avg_ecscopy_auroc, std_ecscopy_auroc, "seagreen",  "^", "1−ECS (copying heads)"),
            (avg_pks_auroc,     std_pks_auroc,     "tomato",    "s", "PKS"),
        ],
        out_dir / "fig3_auroc.png",
        ylabel="AUROC",
        title="AUROC — ECS & PKS vs hallucinated sentences\n"
              "(mean ± std across samples × generations, shaded)",
        hline=0.5,
        ylim=[0.3, 1.0],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Top-5 discriminative layers
# ─────────────────────────────────────────────────────────────────────────────

def _top5(vals: np.ndarray) -> np.ndarray:
    """Indices of the 5 highest values, NaN treated as -inf."""
    safe = np.where(np.isfinite(vals), vals, -np.inf)
    return np.argsort(safe)[-5:][::-1]


def plot_top5_layers(
    avg_ecs_auroc: np.ndarray,
    avg_ecscopy_auroc: np.ndarray,
    avg_pks_auroc: np.ndarray,
    out_dir: Path,
) -> None:
    """Bar chart of top-5 layers for ECS (all heads), ECS (copying heads), and PKS by AUROC."""
    panels = [
        (avg_ecs_auroc,     "steelblue", "Top 5 ECS layers (all heads)"),
        (avg_ecscopy_auroc, "seagreen",  "Top 5 ECS layers (copying heads)"),
        (avg_pks_auroc,     "tomato",    "Top 5 PKS layers"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (vals, color, title) in zip(axes, panels):
        top5 = _top5(vals)
        heights = np.where(np.isfinite(vals[top5]), vals[top5], 0.0)
        bars = ax.bar(range(5), heights, color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"L{l}" for l in top5], fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        top_max = np.nanmax(heights) if np.isfinite(heights).any() else 1.0
        ax.set_ylim([0.4, min(1.02, top_max + 0.08)])
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
        for bar, v in zip(bars, vals[top5]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}" if np.isfinite(v) else "n/a",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    p = out_dir / "fig4_top5_layers.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Incremental refresh helper — called after every gen and once at the end
# ─────────────────────────────────────────────────────────────────────────────

def _nan_mean_std(stack: List[np.ndarray], n_layers: int):
    """nanmean / nanstd across a list of (n_layers,) arrays, NaN where all entries NaN."""
    if not stack:
        return np.full(n_layers, np.nan), np.full(n_layers, np.nan)
    arr = np.vstack(stack)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _refresh_outputs(
    all_ecs, all_ecscopy, all_pks, all_luq,
    pg_ecs_auroc, pg_ecscopy_auroc, pg_pks_auroc,
    pg_ecs_d, pg_ecscopy_d, pg_pks_d,
    n_layers: int,
    out_dir: Path,
    hallu_thresh: float,
) -> None:
    """Recompute averaged metrics (NaN-aware) and overwrite plots + layer_metrics.csv."""
    ecs_all     = np.concatenate(all_ecs,     axis=1)
    ecscopy_all = np.concatenate(all_ecscopy, axis=1)
    pks_all     = np.concatenate(all_pks,     axis=1)
    luq_all     = np.concatenate(all_luq)
    labels      = (luq_all > hallu_thresh).astype(int)

    avg_ecs_d,     std_ecs_d     = _nan_mean_std(pg_ecs_d,     n_layers)
    avg_ecscopy_d, std_ecscopy_d = _nan_mean_std(pg_ecscopy_d, n_layers)
    avg_pks_d,     std_pks_d     = _nan_mean_std(pg_pks_d,     n_layers)

    avg_ecs_auroc,     std_ecs_auroc     = _nan_mean_std(pg_ecs_auroc,     n_layers)
    avg_ecscopy_auroc, std_ecscopy_auroc = _nan_mean_std(pg_ecscopy_auroc, n_layers)
    avg_pks_auroc,     std_pks_auroc     = _nan_mean_std(pg_pks_auroc,     n_layers)

    pd.DataFrame({
        "layer":                 np.arange(n_layers),
        "avg_ecs_auroc":         avg_ecs_auroc,
        "std_ecs_auroc":         std_ecs_auroc,
        "avg_ecs_copy_auroc":    avg_ecscopy_auroc,
        "std_ecs_copy_auroc":    std_ecscopy_auroc,
        "avg_pks_auroc":         avg_pks_auroc,
        "std_pks_auroc":         std_pks_auroc,
        "avg_ecs_d":             avg_ecs_d,
        "std_ecs_d":             std_ecs_d,
        "avg_ecs_copy_d":        avg_ecscopy_d,
        "std_ecs_copy_d":        std_ecscopy_d,
        "avg_pks_d":             avg_pks_d,
        "std_pks_d":             std_pks_d,
    }).to_csv(out_dir / "layer_metrics.csv", index=False)

    pd.DataFrame({
        **{f"ecs_l{l}":      ecs_all[l]     for l in range(n_layers)},
        **{f"ecs_copy_l{l}": ecscopy_all[l] for l in range(n_layers)},
        **{f"pks_l{l}":      pks_all[l]     for l in range(n_layers)},
        "luq_score": luq_all,
        "label":     labels,
    }).to_csv(out_dir / "sentence_scores.csv", index=False)

    plot_ecs_pks_mean_std(ecs_all, ecscopy_all, pks_all, out_dir)
    plot_cohens_d_layers(avg_ecs_d, std_ecs_d, avg_ecscopy_d, std_ecscopy_d,
                         avg_pks_d, std_pks_d, out_dir)
    plot_auroc_layers(avg_ecs_auroc, std_ecs_auroc, avg_ecscopy_auroc, std_ecscopy_auroc,
                      avg_pks_auroc, std_pks_auroc, out_dir)
    plot_top5_layers(avg_ecs_auroc, avg_ecscopy_auroc, avg_pks_auroc, out_dir)


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
    parser.add_argument("--copying-thresh", type=float, default=0.5,
                        help="A head is a Copying Head if its OV positive-real-eigenvalue "
                             "fraction exceeds this (default 0.5 = majority positive).")
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

    # ── Copying-head identification (static; computed once, cached to disk) ──
    copy_score_path = out_dir / "copying_head_scores.npy"
    if copy_score_path.exists() and not args.no_cache:
        copying_scores = np.load(str(copy_score_path))
        print(f"  Loaded copying-head scores from {copy_score_path}")
    else:
        print("  Computing copying-head scores (OV eigenvalues) …", flush=True)
        copying_scores = compute_copying_head_scores(model)
        np.save(str(copy_score_path), copying_scores)
    copying_mask = select_copying_heads(copying_scores, args.copying_thresh)
    n_copy = int(copying_mask.sum())
    layers_with_copy = int((copying_mask.any(axis=1)).sum())
    print(f"  Copying heads (score>{args.copying_thresh}): {n_copy}/{copying_scores.size} "
          f"across {layers_with_copy}/{n_layers} layers")
    np.save(str(out_dir / "copying_head_mask.npy"), copying_mask)

    gen_files = sorted(gen_dir.glob("sample_*_generations.json"))
    if args.samples is not None:
        gen_files = gen_files[:args.samples]
    print(f"  Processing {len(gen_files)} sample(s) …\n")

    # Pooled accumulators (concatenated over all sentences)
    all_ecs:     List[np.ndarray] = []   # all-heads ECS
    all_ecscopy: List[np.ndarray] = []   # copying-heads ECS (may contain NaN layers)
    all_pks:     List[np.ndarray] = []
    all_luq:     List[np.ndarray] = []

    # Per-gen records for averaged metrics (NaN where a signal/layer is undefined)
    pg_ecs_auroc:     List[np.ndarray] = []
    pg_ecscopy_auroc: List[np.ndarray] = []
    pg_pks_auroc:     List[np.ndarray] = []
    pg_ecs_d:         List[np.ndarray] = []
    pg_ecscopy_d:     List[np.ndarray] = []
    pg_pks_d:         List[np.ndarray] = []

    # Incremental CSV paths — written after every gen so a crash doesn't lose progress
    auroc_csv_path   = out_dir / "per_gen_auroc.csv"
    cohens_csv_path  = out_dir / "per_gen_cohens_d.csv"
    _auroc_header_written  = auroc_csv_path.exists()
    _cohens_header_written = cohens_csv_path.exists()

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
                spans, n_fail, n_fuzzy = find_sentence_token_spans(
                    model, tokens, transcript_len, sentences
                )
            except Exception as exc:
                print(f"spans ERROR: {exc}")
                continue

            # Token-accounting guard: a sentence that cannot be located gets a
            # degenerate 1-token span, so its ECS/PKS are meaningless. Warn on any
            # failure and skip the note entirely if too many sentences are unlocatable.
            if n_fail or n_fuzzy:
                print(f"[unmatched: {n_fail} fail, {n_fuzzy} fuzzy]",
                      end="  ", flush=True)
            if n_fail > max(2, int(0.20 * n_sents)):
                print(f"SKIP — {n_fail}/{n_sents} sentences unlocatable "
                      f"(>20%); spans unreliable")
                continue

            # ── Load from disk cache if available ─────────────────────────────
            cache_npz = cache_dir / f"{note_name}.npz"

            cached_ok = False
            if not args.no_cache and cache_npz.exists():
                d = np.load(str(cache_npz))
                # ecs_copy depends on --copying-thresh; invalidate if it changed.
                cached_thresh = float(d["copying_thresh"]) if "copying_thresh" in d else None
                if cached_thresh == args.copying_thresh:
                    cached_ok = True
                    ecs_l, ecscopy_l, pks_l = d["ecs"], d["ecs_copy"], d["pks"]
                    n_valid = min(ecs_l.shape[1], len(luq_scores))
                    ecs_l, ecscopy_l, pks_l = (ecs_l[:, :n_valid],
                                               ecscopy_l[:, :n_valid],
                                               pks_l[:, :n_valid])
                    luq_v = luq_scores[:n_valid]
                    print(f"ECS={np.nanmean(ecs_l):.3f}  PKS={pks_l.mean():.3f}  (cached)")
            if not cached_ok:
                try:
                    ecs_l, ecscopy_l, pks_l = compute_ecs_pks_single_pass(
                        model, tokens, transcript_len, spans,
                        device=args.device, top_k_frac=args.top_k_frac,
                        copying_mask=copying_mask,
                    )
                except Exception as exc:
                    print(f"forward ERROR: {exc}")
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    continue

                np.savez_compressed(str(cache_npz),
                                    ecs=ecs_l, ecs_copy=ecscopy_l, pks=pks_l,
                                    copying_thresh=np.float64(args.copying_thresh))

                n_valid = min(ecs_l.shape[1], len(luq_scores))
                ecs_l, ecscopy_l, pks_l = (ecs_l[:, :n_valid],
                                           ecscopy_l[:, :n_valid],
                                           pks_l[:, :n_valid])
                luq_v = luq_scores[:n_valid]
                print(f"ECS={np.nanmean(ecs_l):.3f}  PKS={pks_l.mean():.3f}  OK")

            # ── Save per-gen activation file (all layers + heads, backtrackable) ──
            np.savez_compressed(
                act_dir / f"sample_{si:03d}_gen_{k:02d}.npz",
                ecs      = ecs_l.astype(np.float32),      # (n_layers, n_sents) all heads
                ecs_copy = ecscopy_l.astype(np.float32),  # (n_layers, n_sents) copying heads
                pks      = pks_l.astype(np.float32),
                luq      = luq_v.astype(np.float32),
                labels   = (luq_v > args.hallu_thresh).astype(np.int8),
            )

            # ── Accumulate pooled data ─────────────────────────────────────────
            all_ecs.append(ecs_l)
            all_ecscopy.append(ecscopy_l)
            all_pks.append(pks_l)
            all_luq.append(luq_v)

            # ── Per-gen AUROC and Cohen's d (per signal; NaN-safe) ────────────
            y_gen = (luq_v > args.hallu_thresh).astype(int)

            a_ecs  = auroc_per_layer_single(ecs_l,     y_gen, hallu_high=False)
            a_ecsc = auroc_per_layer_single(ecscopy_l, y_gen, hallu_high=False)
            a_pks  = auroc_per_layer_single(pks_l,     y_gen, hallu_high=True)
            d_ecs  = cohens_d_per_layer_single(ecs_l,     y_gen)
            d_ecsc = cohens_d_per_layer_single(ecscopy_l, y_gen)
            d_pks  = cohens_d_per_layer_single(pks_l,     y_gen)

            pg_ecs_auroc.append(a_ecs)
            pg_ecscopy_auroc.append(a_ecsc)
            pg_pks_auroc.append(a_pks)
            pg_ecs_d.append(d_ecs)
            pg_ecscopy_d.append(d_ecsc)
            pg_pks_d.append(d_pks)

            # Flush this gen's rows to CSV immediately
            layers_arr = np.arange(n_layers)
            pd.DataFrame({
                "sample":         si,
                "gen":            k,
                "layer":          layers_arr,
                "ecs_auroc":      np.round(a_ecs,  5),
                "ecs_copy_auroc": np.round(a_ecsc, 5),
                "pks_auroc":      np.round(a_pks,  5),
            }).to_csv(auroc_csv_path, mode="a",
                      header=not _auroc_header_written, index=False)
            _auroc_header_written = True

            pd.DataFrame({
                "sample":            si,
                "gen":               k,
                "layer":             layers_arr,
                "ecs_cohens_d":      np.round(d_ecs,  5),
                "ecs_copy_cohens_d": np.round(d_ecsc, 5),
                "pks_cohens_d":      np.round(d_pks,  5),
            }).to_csv(cohens_csv_path, mode="a",
                      header=not _cohens_header_written, index=False)
            _cohens_header_written = True

            # Refresh plots and summary CSVs after every gen
            _refresh_outputs(
                all_ecs, all_ecscopy, all_pks, all_luq,
                pg_ecs_auroc, pg_ecscopy_auroc, pg_pks_auroc,
                pg_ecs_d, pg_ecscopy_d, pg_pks_d,
                n_layers, out_dir, args.hallu_thresh,
            )

    if not all_ecs:
        print("\nNo samples processed. Check paths and model loading.")
        sys.exit(1)

    # ── Final summary ─────────────────────────────────────────────────────────
    ecs_all = np.concatenate(all_ecs, axis=1)
    pks_all = np.concatenate(all_pks, axis=1)
    luq_all = np.concatenate(all_luq)
    labels  = (luq_all > args.hallu_thresh).astype(int)

    print(f"\nTotal sentences : {luq_all.shape[0]}")
    print(f"Hallucinated    : {labels.sum()}  ({100 * labels.mean():.1f} %)")
    print(f"ECS overall mean: {np.nanmean(ecs_all):.4f}")
    print(f"PKS overall mean: {pks_all.mean():.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_ecs_auroc     = np.nanmean(np.vstack(pg_ecs_auroc),     axis=0)
        avg_ecscopy_auroc = np.nanmean(np.vstack(pg_ecscopy_auroc), axis=0)
        avg_pks_auroc     = np.nanmean(np.vstack(pg_pks_auroc),     axis=0)
    print(f"  Top-5 ECS layers   (all heads, 1−ECS AUROC): {list(_top5(avg_ecs_auroc))}")
    print(f"  Top-5 ECS layers (copying heads, 1−ECS AUROC): {list(_top5(avg_ecscopy_auroc))}")
    print(f"  Top-5 PKS layers   (PKS AUROC): {list(_top5(avg_pks_auroc))}")

    print("\nDone. Outputs in", out_dir)


if __name__ == "__main__":
    main()
