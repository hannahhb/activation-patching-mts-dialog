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

# SOAP section headers — standalone lines that carry no clinical content
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

_USER_TEMPLATE = (
    "{system}\n\n"
    "{template}\n\n"
    "{style}\n\n"
    "Transcript:\n{transcript}"
)


def _build_user_content(transcript: str) -> str:
    return _USER_TEMPLATE.format(
        system=_SYSTEM_PROMPT,
        template=_SOAP_TEMPLATE,
        style=_STYLE_GUIDELINES,
        transcript=transcript.strip(),
    )


def tokenize_prompt_and_note(
    model,
    transcript: str,
    note: str,
    device: str,
) -> Tuple[torch.Tensor, int]:
    """
    Tokenise [user: full_prompt | assistant: note] and return
    (full_token_ids, prompt_len).

    prompt_len is the number of tokens before the note, so
    full_token_ids[:, prompt_len:] is exactly the note.

    The user message exactly matches what luq_sentence.py sends to Bedrock
    (single user turn, no system role): system_prompt + template + style +
    "Transcript:\\n" + transcript.
    """
    tokenizer = model.tokenizer
    user_content = _build_user_content(transcript)

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": note},
    ]
    prompt_messages = messages[:1]  # without the note

    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Manual Llama 3.1 format — single user turn, no system header
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
# Forward pass — single pass with hooks (memory-efficient)
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
    Single forward pass with hooks to compute ECS and PKS for all layers.

    Each hook immediately compresses its activation and moves the result to CPU,
    so the GPU never holds more than one layer's activations at a time on top of
    the model weights.

    - pattern hook  → per-chunk mean attention weights (H, T), stored on CPU
    - resid_mid hook → stashed temporarily on CPU for PKS
    - resid_post hook → PKS computed inline; x_last captured from last layer
    - After pass   → ECS computed from compressed chunk attention + x_last

    Returns
    -------
    ecs_layers : (n_layers, n_sents) float64, clipped to [-1, 1]
    pks_layers : (n_layers, n_sents) float64
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    n_sents  = len(sent_spans)
    T        = transcript_len
    k_top    = max(1, int(T * top_k_frac))

    # CPU accumulators
    chunk_attn: List[Optional[List]] = [None] * n_layers   # [l][si] = (H, T) numpy
    pks_all    = np.zeros((n_layers, n_sents), dtype=np.float64)
    x_last_ref = [None]    # (S, D) numpy, set by last-layer resid_post hook
    _mid_tmp   = [None]    # temp: note-token resid_mid for current layer (CPU tensor)

    # ── Hook factories ────────────────────────────────────────────────────────

    def make_pattern_hook(l: int):
        def fn(value, hook):
            # value: (1, H, S, S) — compress to per-chunk means immediately
            attn    = value[0].to(torch.float32).cpu().numpy()  # (H, S, S)
            seq_len = attn.shape[-1]
            chunks  = []
            for s_start, s_end in sent_spans:
                abs_s = T + s_start
                abs_e = min(T + s_end, seq_len)
                if abs_s < abs_e:
                    w = attn[:, abs_s:abs_e, :T].mean(axis=1)  # (H, T)
                else:
                    w = np.zeros((n_heads, T), dtype=np.float32)
                chunks.append(w)
            chunk_attn[l] = chunks
            return value
        return fn

    def make_resid_mid_hook(l: int):
        def fn(value, hook):
            # Stash note-token slice on CPU; freed after the paired resid_post hook
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

    # ── Single forward pass ───────────────────────────────────────────────────
    fwd_hooks = []
    for l in range(n_layers):
        fwd_hooks += [
            (f"blocks.{l}.attn.hook_pattern", make_pattern_hook(l)),
            (f"blocks.{l}.hook_resid_mid",    make_resid_mid_hook(l)),
            (f"blocks.{l}.hook_resid_post",   make_resid_post_hook(l)),
        ]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    # ── ECS: now we have x_last and all chunk attention weights ───────────────
    x_arr   = x_last_ref[0]   # (S, D)
    x_trans = x_arr[:T]       # (T, D)
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
            w_chunk = chunk_attn[l][si]   # (H, T)
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


def bootstrap_layer_metrics(
    ecs_layers: np.ndarray,
    pks_layers: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap 95 % confidence intervals for per-layer AUROC and AUPRC.

    Resamples sentences (rows) with replacement n_boot times, recomputes
    per_layer_metrics on each resample, and returns the (alpha/2, 1-alpha/2)
    percentile bounds across bootstrap iterations.

    Returns
    -------
    Dict with keys  ecs_auroc_lo/hi, pks_auroc_lo/hi, comb_auroc_lo/hi,
                    ecs_auprc_lo/hi, pks_auprc_lo/hi, comb_auprc_lo/hi
    each a 1-D array of length n_layers.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    rng      = np.random.default_rng(seed)
    n_sents  = labels.shape[0]
    n_layers = ecs_layers.shape[0]
    alpha    = (1.0 - ci) / 2.0   # e.g. 0.025 for 95 % CI

    metrics_keys = [
        "ecs_auroc", "pks_auroc", "comb_auroc",
        "ecs_auprc", "pks_auprc", "comb_auprc",
    ]
    # boot_vals[key] → (n_boot, n_layers)
    boot_vals: Dict[str, np.ndarray] = {
        k: np.full((n_boot, n_layers), 0.5 if "auroc" in k else labels.mean())
        for k in metrics_keys
    }

    print(f"  Bootstrapping ({n_boot} resamples) …", end="  ", flush=True)
    for b in range(n_boot):
        idx = rng.integers(0, n_sents, size=n_sents)
        y_b = labels[idx].astype(int)
        if y_b.sum() == 0 or (1 - y_b).sum() == 0:
            continue   # degenerate resample — keep default (uninformative)

        for l in range(n_layers):
            ecs_b  = ecs_layers[l][idx]
            pks_b  = pks_layers[l][idx]
            comb_b = pks_b - ecs_b

            for score, auc_k, apr_k in [
                (-ecs_b,  "ecs_auroc",  "ecs_auprc"),
                ( pks_b,  "pks_auroc",  "pks_auprc"),
                ( comb_b, "comb_auroc", "comb_auprc"),
            ]:
                try:
                    boot_vals[auc_k][b, l] = roc_auc_score(y_b, score)
                except ValueError:
                    pass
                try:
                    boot_vals[apr_k][b, l] = average_precision_score(y_b, score)
                except ValueError:
                    pass

        if (b + 1) % 200 == 0:
            print(f"{b + 1}", end=" ", flush=True)

    print("done")

    ci_out: Dict[str, np.ndarray] = {}
    for k, vals in boot_vals.items():
        ci_out[f"{k}_lo"] = np.percentile(vals, 100 * alpha,       axis=0)
        ci_out[f"{k}_hi"] = np.percentile(vals, 100 * (1 - alpha), axis=0)

    return ci_out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_curves(
    metrics: Dict,
    out_dir: Path,
    hallu_thresh: float,
    ci: Optional[Dict] = None,
) -> None:
    """
    Plot AUROC and AUPRC vs layer with optional shaded bootstrap CI bands.

    Parameters
    ----------
    ci : output of bootstrap_layer_metrics, or None to skip CI bands.
    """
    n_layers = len(metrics["ecs_auroc"])
    layers   = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    specs = [
        # (metric_key, colour, marker, label)
        ("ecs_auroc",  "ecs_auprc",  "steelblue", "o", "1 − ECS"),
        ("pks_auroc",  "pks_auprc",  "tomato",    "s", "PKS"),
        ("comb_auroc", "comb_auprc", "seagreen",  "^", "PKS − ECS"),
    ]

    for auc_k, apr_k, colour, marker, label in specs:
        # AUROC panel
        ax = axes[0]
        ax.plot(layers, metrics[auc_k], color=colour,
                marker=marker, ms=4, lw=1.5, label=label)
        if ci is not None:
            ax.fill_between(layers, ci[f"{auc_k}_lo"], ci[f"{auc_k}_hi"],
                            color=colour, alpha=0.15)

        # AUPRC panel
        ax = axes[1]
        ax.plot(layers, metrics[apr_k], color=colour,
                marker=marker, ms=4, lw=1.5, label=label)
        if ci is not None:
            ax.fill_between(layers, ci[f"{apr_k}_lo"], ci[f"{apr_k}_hi"],
                            color=colour, alpha=0.15)

    ci_label = " (shaded = 95 % bootstrap CI)" if ci is not None else ""

    axes[0].axhline(0.5, color="grey", ls="--", lw=0.8, label="chance")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title(f"AUROC vs Layer  (U > {hallu_thresh} → hallucinated){ci_label}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    base = metrics["baseline_auprc"]
    axes[1].axhline(base, color="grey", ls="--", lw=0.8,
                    label=f"baseline ({base:.2f})")
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_title(f"AUPRC vs Layer  (U > {hallu_thresh} → hallucinated){ci_label}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

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
                        help="Dir with sample_NNN_note_KK_sentences.csv (LUQ output)")
    parser.add_argument("--out",
                        default="sae_experiments/redeep_out",
                        help="Output directory for plots and CSVs")
    parser.add_argument("--hallu-thresh", type=float, default=0.5,
                        help="LUQ uncertainty threshold to label a sentence hallucinated")
    parser.add_argument("--top-k-frac", type=float, default=0.10,
                        help="Fraction of transcript tokens to keep for ECS (sparsity)")
    parser.add_argument("--note-idx", type=int, default=None,
                        help="Restrict to one note index (default: all K notes per sample)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max number of samples to process (None = all)")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples for CI bands (0 = skip)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached per-sample ECS/PKS arrays and recompute")
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

    cache_dir = out_dir / "sample_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators: one list per sample, concatenated later
    all_ecs: List[np.ndarray] = []
    all_pks: List[np.ndarray] = []
    all_luq: List[np.ndarray] = []

    # ── Per-sample, per-note loop ─────────────────────────────────────────────
    for gen_path in gen_files:
        name = gen_path.stem.replace("_generations", "")  # e.g. sample_000

        with open(gen_path) as f:
            gen_data = json.load(f)

        transcript = gen_data["transcript"]
        notes      = gen_data["notes"]
        K          = len(notes)

        note_indices = [args.note_idx] if args.note_idx is not None else range(K)

        for k in note_indices:
            if k >= K:
                print(f"[{name}] note_idx {k} >= K={K} — skip")
                continue

            note_name = f"{name}_note_{k:02d}"
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

            # Drop standalone SOAP section headers
            keep       = [not is_soap_header(s) for s in sentences]
            sentences  = [s for s, m in zip(sentences, keep) if m]
            luq_scores = luq_scores[keep]

            if not sentences:
                print(f"[{note_name}] No content sentences after header filter — skip")
                continue

            n_sents = len(sentences)
            print(f"[{note_name}]  {n_sents} sentences …", end="  ", flush=True)

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

            # ── Load from disk cache if available ─────────────────────────────
            ecs_cache_path = cache_dir / f"{note_name}_ecs.npy"
            pks_cache_path = cache_dir / f"{note_name}_pks.npy"

            if (not args.no_cache
                    and ecs_cache_path.exists()
                    and pks_cache_path.exists()):
                ecs_l = np.load(str(ecs_cache_path))
                pks_l = np.load(str(pks_cache_path))
                n_valid = min(ecs_l.shape[1], len(luq_scores))
                all_ecs.append(ecs_l[:, :n_valid])
                all_pks.append(pks_l[:, :n_valid])
                all_luq.append(luq_scores[:n_valid])
                print(f"ECS={ecs_l.mean():.3f}  PKS={pks_l.mean():.3f}  (cached)")
                continue

            # ── Layer-by-layer forward passes (ECS + PKS) ─────────────────────
            try:
                ecs_l, pks_l = compute_ecs_pks_single_pass(
                    model, tokens, transcript_len, spans,
                    device=args.device, top_k_frac=args.top_k_frac,
                )
            except Exception as exc:
                print(f"forward/ECS/PKS ERROR: {exc}")
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                continue

            # ── Save to disk cache ─────────────────────────────────────────────
            np.save(str(ecs_cache_path), ecs_l)
            np.save(str(pks_cache_path), pks_l)

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

    ci_bands: Optional[Dict] = None
    if args.n_boot > 0:
        ci_bands = bootstrap_layer_metrics(
            ecs_all, pks_all, labels, n_boot=args.n_boot
        )
        # Save CI to CSV alongside point estimates
        ci_df = pd.DataFrame(
            {k: v for k, v in ci_bands.items()},
            index=pd.RangeIndex(n_layers, name="layer"),
        ).reset_index()
        ci_df.to_csv(out_dir / "layer_metrics_ci.csv", index=False)
        print(f"  Saved {out_dir / 'layer_metrics_ci.csv'}")

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
    plot_layer_curves(metrics, out_dir, args.hallu_thresh, ci=ci_bands)
    plot_roc_pr_at_layer(ecs_all, pks_all, labels, best_comb_layer, out_dir,
                         tag=f"_best_comb")
    plot_roc_pr_at_layer(ecs_all, pks_all, labels, best_pks_layer, out_dir,
                         tag=f"_best_pks")
    plot_ecs_pks_scatter(ecs_all, pks_all, labels, best_comb_layer, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
