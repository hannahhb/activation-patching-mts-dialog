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
  5. Accumulate across samples, then plot AUROC / AUPRC vs layer.

Chunk-level ECS (ReDeEP §4.2)
------------------------------
For sentence chunk C = {n_start, …, n_end-1} (note-local token indices):
  w[l,h,t] = mean_{n ∈ C} attn[l,h, n+T, t]          (T = transcript offset)
  top-k transcript tokens by w (k = top_k_frac * T)
  e[l,h]   = Σ_t w_norm[t] * x_last[t]                (last-layer representation)
  s_vec    = mean_{n ∈ C} x_last[T + n]
  ECS[l,s] = mean_h cosine(e[l,h], s_vec)

Chunk-level PKS
---------------
PKS[l,s] = mean_{n ∈ C} JSD( q(x_mid[l,n]) ‖ q(x_post[l,n]) )
where q(x) = softmax(LogitLens(x)) and JSD is Jensen-Shannon divergence.

Hallucination label
-------------------
A sentence is labelled hallucinated if its LUQ uncertainty U > --hallu-thresh
(default 0.5 = coin-flip entailment score).

Usage
-----
    conda run -n curebench python sae_experiments/redeep_sentence.py \\
        --model  meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --device cuda \\
        --generations sae_experiments/luq_out/generations \\
        --sentences   sae_experiments/luq_out/sentences \\
        --out         sae_experiments/redeep_out

Requirements
------------
    transformer_lens, torch, numpy, pandas, matplotlib, scikit-learn
    ~40 GB GPU VRAM for Llama 3.1 8B with attention cache.
    Reduce --samples or run on CPU (slow) if memory is limited.
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

# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants — must match luq_sentence.py exactly so the tokenised
# input sequence is identical to what was used during generation.
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a medical office assistant drafting documentation for a physician. "
    "DO NOT ADD any content that isn't specifically mentioned IN THE TRANSCRIPT. "
    "From the attached transcript generate a SOAP note based on the below template "
    "format for the physician to review, include all the relevant information and "
    "do not include any information that isn't explicitly mentioned in the transcript. "
    "If nothing is mentioned just return [NOT MENTIONED].\n\n"
    "It is VITAL that all the information in the note is as accurate as possible. "
    "Avoid repeating the same information in different sections where possible. "
    "Write the note from the perspective of the physician. "
    "Only include any section of the template if there is information from the "
    "transcript, otherwise omit it. "
    "Begin your response directly with 'Subjective:' — do not add any preamble, "
    "introduction, or heading before the note."
)

_SOAP_TEMPLATE = """Template for Clinical SOAP Note Format:

Subjective:
- HPI: [include here any mentioned symptoms, chronological narrative of patients \
complaints, information obtained from other sources (always identify source if not \
the patient).]
- Past medical history: [include here all of the patients past conditions, treatments \
and encounters, also include relevant social history here including smoking, alcohol, \
drug use and occupation/travel history]
- Review of systems: [include here any additional symptoms in other organs that is \
relevant to the initial presentation]
- Current medications: [list medicines each on a separate line in the format: \
[DRUG NAME] [DRUG DOSE] [DRUG FREQUENCY] [INDICATION]]

Objective:
- Vital signs: [including any mentioned blood pressure, pulse rate, oxygen saturation, \
temperature]
- Physical exam: [the examination findings from the physical exam, if mentioned]
- Test Results: [include in this section any lab test results or imaging reports]

Assessment / Problem List:
- Assessment: [A one-sentence description of the patient and major problem as described \
by the physician, including the diagnosis the physician has identified]
- Problem list: [List clinical problems inline, separated by semicolons, on a single line. \
Format each as [Condition] [Status: active/suspected/confirmed/past/unknown]. \
Leave status as unknown if not mentioned in the transcript. \
Do not use numbered lists or line breaks between problems.]

Plan:
[include here any management plan mentioned in the transcript, including patient \
education, prescriptions, tests, referrals or other plans.]

Follow-up: [include here any plan mentioned to see the patient again, or to be \
discharged.]"""

_STYLE_GUIDELINES = """Please adhere to the following style guidelines:
- Write from the perspective of the physician (first person)
- Write ONLY in complete, grammatical sentences. Do NOT use bullet points, hyphens, \
numbered lists, or any other list formatting anywhere in the note.
- Be ultra-precise, do not use generalising terms
- Be highly detailed
- Include ALL important negations (e.g. "The patient denies fever.") as well as all \
positive findings, written as full sentences.
- List medications as a sentence: "I prescribed [drug] [dose] [frequency] for [indication]."
- Always document if drug allergies are present or not
- Examination findings always refer to physical exam signs only, not symptoms
- Preserve quantities if mentioned in the text"""


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting — identical to luq_sentence.py so indices match the CSV
# ─────────────────────────────────────────────────────────────────────────────

_PERIOD_SPLIT = re.compile(r"(?<=\.)\s+(?=[A-Z])")


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
    return f"{transcript}\n\n{_SOAP_TEMPLATE}\n\n{_STYLE_GUIDELINES}"


def tokenize_prompt_and_note(
    model,
    transcript: str,
    note: str,
    device: str,
) -> Tuple[torch.Tensor, int]:
    """
    Tokenise [system | user: transcript+template | assistant: note] and
    return (full_token_ids, transcript_len).

    transcript_len is the number of tokens that precede the note text so
    that full_token_ids[:, transcript_len:] is exactly the note.

    Uses the HF tokenizer's apply_chat_template when available (Llama 3.1
    instruct). Falls back to a manual Llama-3-style format.
    """
    tokenizer = model.tokenizer
    user_content = _build_user_content(transcript)

    messages = [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": note},
    ]
    prompt_messages = messages[:2]  # without the note

    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Manual Llama 3.1 format
        full_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{_SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{note}<|eot_id|>"
        )
        prompt_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{_SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    full_ids   = tokenizer.encode(full_text,   return_tensors="pt").to(device)
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    transcript_len = prompt_ids.shape[1]
    note_len = full_ids.shape[1] - transcript_len

    if note_len <= 0:
        raise ValueError(
            f"note_len={note_len} ≤ 0: the note does not appear after the prompt. "
            "Check that the chat template is matching generation format."
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
    """
    Map each sentence string to a (start, end) token range in note-local
    index space (0 = first note token = full_tokens[:, transcript_len]).
    end is exclusive.

    Strategy:
      1. Decode each note token individually to build cumulative text.
      2. Find each sentence's character range in that text via str.find.
      3. Map character positions to token indices.
    """
    tokenizer = model.tokenizer
    note_token_ids = full_tokens[0, transcript_len:].tolist()
    n_note_tokens  = len(note_token_ids)

    # Cumulative decoded text after each token
    cumulative_len = []   # length of decoded string after token i
    note_text = ""
    for tid in note_token_ids:
        note_text += tokenizer.decode([tid])
        cumulative_len.append(len(note_text))

    # Map char offset → note token index (first token that covers that char)
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
            # Fallback: find best-matching position by first 20 chars
            idx = note_text.find(sent[:20], search_from)
        if idx == -1:
            # Last resort: append just after previous span
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
        search_from = idx   # allow slight overlap; don't advance past start

    return spans


# ─────────────────────────────────────────────────────────────────────────────
# LogitLens helper
# ─────────────────────────────────────────────────────────────────────────────

def _logit_lens(model, x: torch.Tensor) -> torch.Tensor:
    """Apply final LayerNorm + unembedding to an intermediate residual state."""
    x = x.unsqueeze(0)       # (1, N, d_model)
    x = model.ln_final(x)
    x = model.unembed(x)
    return x.squeeze(0).float()   # (N, d_vocab)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-level ECS
# ─────────────────────────────────────────────────────────────────────────────

def compute_chunk_ecs(
    model,
    cache,
    transcript_len: int,
    sent_spans: List[Tuple[int, int]],
    top_k_frac: float = 0.10,
) -> np.ndarray:
    """
    Chunk-level External Context Score, shape (n_layers, n_sentences).

    For each sentence chunk and each layer:
      1. Average attention weights from chunk tokens → transcript tokens.
      2. Keep top top_k_frac of transcript tokens (sparsify, as in ReDeEP).
      3. Build attended vector as re-normalised weighted sum of last-layer
         transcript hidden states.
      4. Build sentence vector as mean of last-layer note hidden states in chunk.
      5. ECS = mean over heads of cosine(attended_vec, sentence_vec).

    Values in [-1, 1].  High → sentence is semantically grounded in transcript.
    Negative → model generated content pointing away from what it attended to.
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    n_sents  = len(sent_spans)
    T        = transcript_len

    # Last-layer residual-stream states used as semantic representations
    x_last       = cache["resid_post", n_layers - 1][0].float().cpu().numpy()
    x_transcript = x_last[:T]   # (T, D)
    k_top        = max(1, int(T * top_k_frac))

    # Precompute sentence vectors (mean of last-layer hidden states in chunk)
    D = x_last.shape[-1]
    sent_vecs = np.zeros((n_sents, D), dtype=np.float64)
    for si, (s_start, s_end) in enumerate(sent_spans):
        abs_start = T + s_start
        abs_end   = min(T + s_end, x_last.shape[0])
        if abs_start < abs_end:
            sent_vecs[si] = x_last[abs_start:abs_end].mean(axis=0)

    ecs_layers = np.zeros((n_layers, n_sents), dtype=np.float64)
    seq_len = cache["pattern", 0][0].shape[-1]   # full sequence length

    for layer in range(n_layers):
        attn = cache["pattern", layer][0].float().cpu().numpy()  # (H, S, S)

        for si, (s_start, s_end) in enumerate(sent_spans):
            abs_start = T + s_start
            abs_end   = min(T + s_end, seq_len)
            if abs_start >= abs_end:
                continue

            # Mean attention from chunk tokens to each transcript position: (H, T)
            w_chunk = attn[:, abs_start:abs_end, :T].mean(axis=1)  # (H, T)

            s_vec   = sent_vecs[si]
            norm_s  = np.linalg.norm(s_vec)
            cos_acc = 0.0

            for h in range(n_heads):
                w_h = w_chunk[h]   # (T,)

                # Top-k transcript tokens
                if k_top < T:
                    top_idx = np.argpartition(w_h, -k_top)[-k_top:]
                else:
                    top_idx = np.arange(T)

                w_top = w_h[top_idx]
                w_sum = w_top.sum()
                if w_sum < 1e-10:
                    continue
                w_top = w_top / w_sum   # renormalise

                # Attended context vector: (D,)
                e = (x_transcript[top_idx] * w_top[:, None]).sum(axis=0)

                norm_e = np.linalg.norm(e)
                denom  = norm_e * norm_s
                if denom > 1e-8:
                    cos_acc += float(np.dot(e, s_vec) / denom)

            ecs_layers[layer, si] = cos_acc / n_heads

    return np.clip(ecs_layers, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-level PKS
# ─────────────────────────────────────────────────────────────────────────────

def compute_chunk_pks(
    model,
    cache,
    transcript_len: int,
    sent_spans: List[Tuple[int, int]],
    device: str,
) -> np.ndarray:
    """
    Chunk-level Parametric Knowledge Score, shape (n_layers, n_sentences).

    PKS[l, s] = mean_{n ∈ chunk} JSD( q(x_mid^l_n) ‖ q(x_post^l_n) )

    High → FFNs at layer l injected substantial parametric knowledge when
    generating this sentence.  Mean (rather than sum) normalises for
    sentence length to keep scores comparable across sentences.
    """
    n_layers = model.cfg.n_layers
    n_sents  = len(sent_spans)
    pks_layers = np.zeros((n_layers, n_sents), dtype=np.float64)

    for layer in range(n_layers):
        x_mid  = cache["resid_mid",  layer][0, transcript_len:].to(
            device=device, dtype=torch.float32)
        x_post = cache["resid_post", layer][0, transcript_len:].to(
            device=device, dtype=torch.float32)

        with torch.no_grad():
            q_mid  = torch.softmax(_logit_lens(model, x_mid),  dim=-1).cpu().numpy()
            q_post = torch.softmax(_logit_lens(model, x_post), dim=-1).cpu().numpy()

        # Token-level JSD: (note_len,)
        m   = 0.5 * (q_mid + q_post)
        eps = 1e-10
        kl1 = np.sum(q_mid  * (np.log(q_mid  + eps) - np.log(m + eps)), axis=-1)
        kl2 = np.sum(q_post * (np.log(q_post + eps) - np.log(m + eps)), axis=-1)
        token_jsd = np.clip(0.5 * kl1 + 0.5 * kl2, 0.0, 1.0)   # (note_len,)

        for si, (s_start, s_end) in enumerate(sent_spans):
            s_end = min(s_end, len(token_jsd))
            if s_start < s_end:
                pks_layers[layer, si] = token_jsd[s_start:s_end].mean()

    return pks_layers


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_pass(model, tokens: torch.Tensor, device: str):
    """Single forward pass, caching only what ECS/PKS need."""
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: (
                "pattern"    in name or   # attention weights → ECS
                "resid_mid"  in name or   # residual before FFN → PKS
                "resid_post" in name      # residual after FFN; last layer → ECS
            ),
        )
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer AUROC / AUPRC
# ─────────────────────────────────────────────────────────────────────────────

def per_layer_metrics(
    ecs_layers: np.ndarray,   # (n_layers, n_total_sents)
    pks_layers: np.ndarray,
    labels: np.ndarray,       # (n_total_sents,) binary int
) -> Dict:
    """
    AUROC and AUPRC per layer for ECS (inverted), PKS, and their combination.

    ECS is inverted (1 - ECS used as score) because lower ECS → more hallucinated.
    Combined score = PKS - ECS: positive direction for both signals.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    n_layers = ecs_layers.shape[0]
    keys = ["ecs_auroc", "pks_auroc", "comb_auroc",
            "ecs_auprc", "pks_auprc", "comb_auprc"]
    result = {k: np.full(n_layers, 0.5 if "auroc" in k else labels.mean())
              for k in keys}
    result["baseline_auprc"] = float(labels.mean())

    y = labels.astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        print("  Warning: only one class present — AUROC/AUPRC will be uninformative.")
        return result

    for l in range(n_layers):
        ecs  = ecs_layers[l]
        pks  = pks_layers[l]
        comb = pks - ecs   # high → parametric > extractive → hallucination risk

        for score, auc_k, apr_k in [
            (-ecs,  "ecs_auroc",  "ecs_auprc"),   # inverted: lower ECS = worse
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
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_curves(metrics: Dict, out_dir: Path, hallu_thresh: float) -> None:
    n_layers = len(metrics["ecs_auroc"])
    layers   = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(layers, metrics["ecs_auroc"],  "b-o", ms=4, label="1 − ECS")
    ax.plot(layers, metrics["pks_auroc"],  "r-s", ms=4, label="PKS")
    ax.plot(layers, metrics["comb_auroc"], "g-^", ms=4, label="PKS − ECS")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUROC")
    ax.set_title(f"AUROC vs Layer  (U > {hallu_thresh} → hallucinated)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    base = metrics["baseline_auprc"]
    ax.plot(layers, metrics["ecs_auprc"],  "b-o", ms=4, label="1 − ECS")
    ax.plot(layers, metrics["pks_auprc"],  "r-s", ms=4, label="PKS")
    ax.plot(layers, metrics["comb_auprc"], "g-^", ms=4, label="PKS − ECS")
    ax.axhline(base, color="grey", ls="--", lw=0.8,
               label=f"baseline ({base:.2f})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUPRC")
    ax.set_title(f"AUPRC vs Layer  (U > {hallu_thresh} → hallucinated)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / "layer_auroc_auprc.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Saved {p}")


def plot_roc_pr_at_layer(
    ecs_layers: np.ndarray,
    pks_layers: np.ndarray,
    labels: np.ndarray,
    layer: int,
    out_dir: Path,
    tag: str = "",
) -> None:
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    y    = labels.astype(int)
    ecs  = ecs_layers[layer]
    pks  = pks_layers[layer]
    comb = pks - ecs

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax = axes[0]
    for score, label, invert in [
        (ecs,  "1 − ECS", True),
        (pks,  "PKS",     False),
        (comb, "PKS − ECS", False),
    ]:
        s = -score if invert else score
        try:
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            ax.plot(fpr, tpr, label=f"{label}  AUC={auc:.3f}")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "--k", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — Layer {layer}{tag}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # PR
    ax = axes[1]
    for score, label, invert in [
        (ecs,  "1 − ECS", True),
        (pks,  "PKS",     False),
        (comb, "PKS − ECS", False),
    ]:
        s = -score if invert else score
        try:
            prec, rec, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            ax.plot(rec, prec, label=f"{label}  AP={ap:.3f}")
        except Exception:
            pass
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR — Layer {layer}{tag}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"roc_pr_layer{layer}{tag}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Saved {p}")


def plot_ecs_pks_scatter(
    ecs_layers: np.ndarray,
    pks_layers: np.ndarray,
    labels: np.ndarray,
    layer: int,
    out_dir: Path,
) -> None:
    """ECS vs PKS scatter coloured by hallucination label at a given layer."""
    ecs  = ecs_layers[layer]
    pks  = pks_layers[layer]
    y    = labels.astype(bool)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ecs[~y], pks[~y], c="steelblue", alpha=0.5, s=20, label="Non-hallucinated")
    ax.scatter(ecs[ y], pks[ y], c="tomato",    alpha=0.6, s=20, label="Hallucinated")
    ax.set_xlabel("ECS (chunk-level)")
    ax.set_ylabel("PKS (chunk-level)")
    ax.set_title(f"ECS vs PKS — Layer {layer}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"scatter_layer{layer}.png"
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
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace/TransformerLens model name")
    parser.add_argument("--device",
                        default=(
                            "cuda" if torch.cuda.is_available() else
                            "mps"  if torch.backends.mps.is_available() else
                            "cpu"
                        ))
    parser.add_argument("--generations",
                        default="sae_experiments/luq_out/generations",
                        help="Dir with sample_NNN_generations.json")
    parser.add_argument("--sentences",
                        default="sae_experiments/luq_out/sentences",
                        help="Dir with sample_NNN_sentences.csv (LUQ output)")
    parser.add_argument("--out",
                        default="sae_experiments/redeep_out",
                        help="Output directory for plots and CSVs")
    parser.add_argument("--hallu-thresh", type=float, default=0.5,
                        help="LUQ uncertainty threshold to label a sentence hallucinated")
    parser.add_argument("--top-k-frac", type=float, default=0.10,
                        help="Fraction of transcript tokens to keep for ECS (sparsity)")
    parser.add_argument("--note-idx", type=int, default=0,
                        help="Which of the K generations to analyse (default 0 = reference)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max number of samples to process (None = all)")
    args = parser.parse_args()

    gen_dir  = Path(args.generations)
    sent_dir = Path(args.sentences)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.model, args.device)
    n_layers = model.cfg.n_layers
    print(f"  n_layers={n_layers}  n_heads={model.cfg.n_heads}  "
          f"d_model={model.cfg.d_model}")

    # ── Gather sample files ───────────────────────────────────────────────────
    gen_files = sorted(gen_dir.glob("sample_*_generations.json"))
    if args.samples is not None:
        gen_files = gen_files[:args.samples]
    print(f"  Processing {len(gen_files)} sample(s) …\n")

    # Accumulators: one list per sample, concatenated later
    all_ecs: List[np.ndarray] = []
    all_pks: List[np.ndarray] = []
    all_luq: List[np.ndarray] = []

    # ── Per-sample loop ───────────────────────────────────────────────────────
    for gen_path in gen_files:
        name = gen_path.stem.replace("_generations", "")  # e.g. sample_000
        csv_path = sent_dir / f"{name}_sentences.csv"

        if not csv_path.exists():
            print(f"[{name}] No sentences CSV — skip")
            continue

        with open(gen_path) as f:
            gen_data = json.load(f)

        sent_df = pd.read_csv(csv_path)
        if sent_df.empty:
            print(f"[{name}] Empty sentences CSV — skip")
            continue

        transcript  = gen_data["transcript"]
        note_idx    = min(args.note_idx, len(gen_data["notes"]) - 1)
        note        = gen_data["notes"][note_idx]
        sentences   = sent_df["sentence"].tolist()
        luq_scores  = sent_df["uncertainty"].values.astype(np.float64)

        n_sents = len(sentences)
        print(f"[{name}]  {n_sents} sentences …", end="  ", flush=True)

        # Tokenise
        try:
            tokens, transcript_len = tokenize_prompt_and_note(
                model, transcript, note, args.device
            )
        except Exception as exc:
            print(f"tokenise ERROR: {exc}")
            continue

        note_len = tokens.shape[1] - transcript_len
        print(f"T={transcript_len} N={note_len}", end="  ", flush=True)

        # Sentence → token span mapping
        try:
            spans = find_sentence_token_spans(
                model, tokens, transcript_len, sentences
            )
        except Exception as exc:
            print(f"spans ERROR: {exc}")
            continue

        # Forward pass
        try:
            cache = run_forward_pass(model, tokens, args.device)
        except Exception as exc:
            print(f"forward ERROR: {exc}")
            continue

        # Chunk-level ECS and PKS
        try:
            ecs_l = compute_chunk_ecs(
                model, cache, transcript_len, spans,
                top_k_frac=args.top_k_frac,
            )
            pks_l = compute_chunk_pks(
                model, cache, transcript_len, spans,
                device=args.device,
            )
        except Exception as exc:
            print(f"ECS/PKS ERROR: {exc}")
            continue

        del cache
        if args.device == "cuda":
            torch.cuda.empty_cache()

        # Trim to the number of sentences with LUQ scores
        n_valid = min(ecs_l.shape[1], len(luq_scores))
        all_ecs.append(ecs_l[:, :n_valid])
        all_pks.append(pks_l[:, :n_valid])
        all_luq.append(luq_scores[:n_valid])

        print(f"ECS={ecs_l.mean():.3f}  PKS={pks_l.mean():.3f}  OK")

    if not all_ecs:
        print("\nNo samples processed. Check paths and model loading.")
        sys.exit(1)

    # ── Stack and summarise ───────────────────────────────────────────────────
    ecs_all  = np.concatenate(all_ecs, axis=1)   # (n_layers, total_sents)
    pks_all  = np.concatenate(all_pks, axis=1)
    luq_all  = np.concatenate(all_luq)            # (total_sents,)
    labels   = (luq_all > args.hallu_thresh).astype(int)

    print(f"\nTotal sentences: {luq_all.shape[0]}")
    print(f"Hallucinated (U > {args.hallu_thresh}): "
          f"{labels.sum()} ({100 * labels.mean():.1f} %)")
    print(f"ECS overall mean: {ecs_all.mean():.4f}")
    print(f"PKS overall mean: {pks_all.mean():.4f}")

    # ── AUROC / AUPRC per layer ───────────────────────────────────────────────
    print("\nComputing per-layer metrics …")
    metrics = per_layer_metrics(ecs_all, pks_all, labels)

    # Best layers
    best_pks_layer  = int(np.argmax(metrics["pks_auroc"]))
    best_comb_layer = int(np.argmax(metrics["comb_auroc"]))
    best_ecs_layer  = int(np.argmax(metrics["ecs_auroc"]))

    print(f"  Best ECS  layer: {best_ecs_layer:2d}  "
          f"AUROC={metrics['ecs_auroc'][best_ecs_layer]:.4f}")
    print(f"  Best PKS  layer: {best_pks_layer:2d}  "
          f"AUROC={metrics['pks_auroc'][best_pks_layer]:.4f}")
    print(f"  Best COMB layer: {best_comb_layer:2d}  "
          f"AUROC={metrics['comb_auroc'][best_comb_layer]:.4f}")

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame({
        "layer":       np.arange(n_layers),
        "ecs_auroc":   metrics["ecs_auroc"],
        "pks_auroc":   metrics["pks_auroc"],
        "comb_auroc":  metrics["comb_auroc"],
        "ecs_auprc":   metrics["ecs_auprc"],
        "pks_auprc":   metrics["pks_auprc"],
        "comb_auprc":  metrics["comb_auprc"],
    })
    csv_path = out_dir / "layer_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # Save raw sentence-level scores for further analysis
    sents_df = pd.DataFrame({
        **{f"ecs_l{l}": ecs_all[l] for l in range(n_layers)},
        **{f"pks_l{l}": pks_all[l] for l in range(n_layers)},
        "luq_score": luq_all,
        "label":     labels,
    })
    sents_df.to_csv(out_dir / "sentence_scores.csv", index=False)
    print(f"  Saved {out_dir / 'sentence_scores.csv'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nPlotting …")
    plot_layer_curves(metrics, out_dir, args.hallu_thresh)
    plot_roc_pr_at_layer(ecs_all, pks_all, labels, best_comb_layer, out_dir,
                         tag=f"_best_comb")
    plot_roc_pr_at_layer(ecs_all, pks_all, labels, best_pks_layer, out_dir,
                         tag=f"_best_pks")
    plot_ecs_pks_scatter(ecs_all, pks_all, labels, best_comb_layer, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
