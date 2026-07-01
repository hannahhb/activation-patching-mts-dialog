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


def select_copying_heads(
    scores: np.ndarray,
    top_frac: Optional[float] = None,
    thresh: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """
    Boolean (n_layers, n_heads) mask of Copying Heads, plus a signature string.

    Two selection modes (top_frac takes precedence if given):
      * top_frac: keep the globally top `top_frac` fraction of heads by copying
        score. Robust — copying heads are a minority, so a fixed score threshold
        (e.g. >0.5) over-selects because real OV matrices skew positive-real.
      * thresh: keep heads with copying score > thresh.
    """
    if top_frac is not None:
        n_total = scores.size
        n_keep  = max(1, int(round(top_frac * n_total)))
        cutoff  = np.sort(scores, axis=None)[::-1][n_keep - 1]
        mask = scores >= cutoff
        return mask, f"topfrac={top_frac}"
    return scores > thresh, f"thresh={thresh}"


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
    word_spans: Optional[np.ndarray] = None,   # (n_words, 2) int32 token spans in note coords
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Single forward pass.

    Returns (ecs_mean, ecs_copy, pks, pks_tok, ecs_word, ecs_word_copy):
      ecs_mean, ecs_copy, pks  — (n_layers, n_sents)        sentence-level
      pks_tok                  — (n_layers, n_note_tokens)  token-level PKS
      ecs_word, ecs_word_copy  — (n_layers, n_words)        word-level ECS
                                 (all-heads mean / copying-heads mean; empty if
                                 word_spans is None). Faithful to Eq 3 using x^L.

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
    n_note   = tokens.shape[1] - T          # note token count (incl. EOS)

    n_words  = len(word_spans) if word_spans is not None else 0
    chunk_attn:    List[Optional[List]]        = [None] * n_layers
    word_attn_all: List[Optional[np.ndarray]]  = [None] * n_layers  # (n_heads, n_words, k_top) int32
    pks_all  = np.zeros((n_layers, n_sents), dtype=np.float64)
    pks_tok  = np.full((n_layers, n_note),   np.nan, dtype=np.float64)
    x_last   = [None]
    _mid_tmp = [None]
    _pool_P  = [None]   # (n_words, n_note) word-mean pooling matrix, built lazily on device

    def make_pattern_hook(l: int):
        def fn(value, hook):
            attn_gpu = value[0].to(torch.float32)              # (n_heads, seq_len, seq_len) on GPU
            seq_len  = attn_gpu.shape[-1]
            ctx_end  = min(T, seq_len)

            # ── Word-level: mean-pool attention per word via a single matmul ──
            # (precomputed pooling matrix P) then topk — all on GPU, 2 kernels
            # instead of one per word. Move only (n_heads, n_words, k_top) to CPU.
            if word_spans is not None and n_words > 0:
                if _pool_P[0] is None:
                    P = torch.zeros(n_words, n_note,
                                    device=attn_gpu.device, dtype=torch.float32)
                    for wi, (ws, we) in enumerate(word_spans):
                        ws_i, we_c = int(ws), min(int(we), n_note)
                        if ws_i < we_c:
                            P[wi, ws_i:we_c] = 1.0 / (we_c - ws_i)
                    _pool_P[0] = P
                note_attn = attn_gpu[:, T:T + n_note, :ctx_end]      # (n_heads, n_note', T)
                nn = note_attn.shape[1]
                P  = _pool_P[0][:, :nn]                             # (n_words, n_note')
                word_tensor = torch.einsum("wn,hnt->hwt", P, note_attn)  # (n_heads, n_words, T)
                top_idx = torch.topk(word_tensor,
                                     min(k_top, ctx_end), dim=-1).indices  # on GPU
                word_attn_all[l] = top_idx.cpu().numpy().astype(np.int32)

            # ── Sentence-level: existing logic (move full attn to CPU once) ──
            attn = attn_gpu.cpu().numpy()
            chunks = []
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
                # sentence-level aggregation (existing)
                for si, (s_start, s_end) in enumerate(sent_spans):
                    s_e = min(s_end, len(tok_jsd))
                    if s_start < s_e:
                        pks_all[l, si] = tok_jsd[s_start:s_e].mean()
                # token-level: store raw JSD per note token
                n_jsd = len(tok_jsd)
                pks_tok[l, :n_jsd] = tok_jsd
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

    # Aggregate over heads — sentence level.
    ecs_mean = np.nanmean(ecs_lh, axis=1)                     # (n_layers, n_sents)
    if copying_mask is not None and copying_mask.any():
        ecs_copy = np.full((n_layers, n_sents), np.nan, dtype=np.float64)
        for l in range(n_layers):
            heads = np.where(copying_mask[l])[0]
            if heads.size:
                ecs_copy[l] = np.nanmean(ecs_lh[l, heads], axis=0)
    else:
        ecs_copy = np.full((n_layers, n_sents), np.nan, dtype=np.float64)

    # ── Word-level ECS post-pass (faithful to Eq 3, word granularity) ─────────
    # For each word: word_vec = mean x^L over the word's note tokens;
    # e^{l,h} = mean x^L over the top-k context tokens attended by the word
    # (indices already computed on GPU in the hook). ECS = cos(e, word_vec).
    # Uses LLM final-layer hidden states — no embedding model.
    if n_words > 0:
        word_vecs = np.full((n_words, x_arr.shape[1]), np.nan, dtype=np.float64)
        word_norm = np.zeros(n_words, dtype=np.float64)
        for wi, (w_start, w_end) in enumerate(word_spans):
            abs_s = T + int(w_start)
            abs_e = min(T + int(w_end), seq_len)
            if abs_s < abs_e:
                v = x_arr[abs_s:abs_e].mean(axis=0)
                word_vecs[wi] = v
                word_norm[wi] = np.linalg.norm(v)

        ecs_word_lh = np.full((n_layers, n_heads, n_words), np.nan, dtype=np.float64)
        for l in range(n_layers):
            if word_attn_all[l] is None:
                continue
            idx_lh = word_attn_all[l]                          # (n_heads, n_words, k_top) int32
            for h in range(n_heads):
                e_h    = x_trans[idx_lh[h]].mean(axis=1)       # (n_words, d_model)
                num    = (e_h * word_vecs).sum(axis=-1)        # (n_words,)
                ne     = np.linalg.norm(e_h, axis=-1)
                denom  = ne * word_norm
                ecs_word_lh[l, h] = np.where(denom > 1e-8, num / denom, np.nan)

        ecs_word = np.nanmean(ecs_word_lh, axis=1)             # (n_layers, n_words)
        if copying_mask is not None and copying_mask.any():
            ecs_word_copy = np.full((n_layers, n_words), np.nan, dtype=np.float64)
            for l in range(n_layers):
                heads = np.where(copying_mask[l])[0]
                if heads.size:
                    ecs_word_copy[l] = np.nanmean(ecs_word_lh[l, heads], axis=0)
        else:
            ecs_word_copy = np.full((n_layers, n_words), np.nan, dtype=np.float64)
    else:
        ecs_word      = np.zeros((n_layers, 0), dtype=np.float64)
        ecs_word_copy = np.zeros((n_layers, 0), dtype=np.float64)

    return (np.clip(ecs_mean,      -1.0, 1.0),
            np.clip(ecs_copy,      -1.0, 1.0),
            pks_all,
            pks_tok,
            np.clip(ecs_word,      -1.0, 1.0),
            np.clip(ecs_word_copy, -1.0, 1.0))


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
# Re-label mode — rebuild metrics/plots from saved activations, no model needed
# ─────────────────────────────────────────────────────────────────────────────

def run_metrics_only(args, act_dir: Path, out_dir: Path) -> None:
    """
    Recompute all AUROC/Cohen's d CSVs and plots from saved activations/*.npz,
    using a (possibly different) per-sentence hallucination metric. No model load.

    Labels for activation sample_NNN_gen_KK.npz are taken, in this priority:
      1. --labels-dir/sample_NNN_note_KK_sentences.csv, column --label-col,
         aligned by the activation's stored sent_idx; else
      2. the activation's own stored `luq` scores (the metric used at compute time).
    A sentence is hallucinated iff its score > --hallu-thresh.
    """
    labels_dir = Path(args.labels_dir) if args.labels_dir else Path(args.sentences)
    act_files  = sorted(act_dir.glob("sample_*_gen_*.npz"))
    if not act_files:
        print(f"No activations found in {act_dir}. Run the full pass first.")
        sys.exit(1)
    print(f"[metrics-only] {len(act_files)} activation files; "
          f"label_col='{args.label_col}' from {labels_dir}")

    n_layers = None
    all_ecs, all_ecscopy, all_pks, all_luq = [], [], [], []
    pg_ecs_auroc, pg_ecscopy_auroc, pg_pks_auroc = [], [], []
    pg_ecs_d, pg_ecscopy_d, pg_pks_d = [], [], []

    auroc_csv_path  = out_dir / "per_gen_auroc.csv"
    cohens_csv_path = out_dir / "per_gen_cohens_d.csv"
    auroc_csv_path.unlink(missing_ok=True)      # full rebuild — don't append to stale rows
    cohens_csv_path.unlink(missing_ok=True)
    auroc_hdr = cohens_hdr = False

    for ap in act_files:
        m = re.search(r"sample_(\d+)_gen_(\d+)", ap.stem)
        si, k = int(m.group(1)), int(m.group(2))
        d = np.load(str(ap), allow_pickle=True)
        ecs_l, ecscopy_l, pks_l = d["ecs"], d["ecs_copy"], d["pks"]
        if n_layers is None:
            n_layers = ecs_l.shape[0]

        # Resolve new labels — text-based matching so judge sentence_idx
        # (LLM-assigned, 0-based) doesn't have to agree with LUQ sentence_idx.
        # Pipeline:
        #   sent_idx (stored in NPZ) → sentence text (via LUQ sentences CSV)
        #   sentence text → label score (via --labels-dir CSV)
        # Categorical label columns (e.g. "Faithful"/"Fabrication") are
        # converted: faithful_label → 0.0, anything else → 1.0.
        score = None
        luq_csv = Path(args.sentences) / f"sample_{si:03d}_note_{k:02d}_sentences.csv"
        lab_csv = labels_dir / f"sample_{si:03d}_note_{k:02d}{args.labels_suffix}"
        if "sent_idx" in d and lab_csv.exists() and luq_csv.exists():
            ldf = pd.read_csv(lab_csv)
            luq_df = pd.read_csv(luq_csv)
            if (args.label_col in ldf.columns
                    and "sentence" in ldf.columns
                    and "sentence" in luq_df.columns
                    and "sentence_idx" in luq_df.columns):
                # Build text → score from label CSV
                raw = ldf[args.label_col].values
                if raw.dtype == object or raw.dtype.kind in ("U", "S"):
                    # categorical: faithful_label=0, else=1
                    scores_arr = np.where(
                        raw == args.faithful_label, 0.0, 1.0
                    ).astype(np.float64)
                else:
                    scores_arr = raw.astype(np.float64)
                text_lut = {
                    str(t).strip(): s
                    for t, s in zip(ldf["sentence"].values, scores_arr)
                }
                # Build sent_idx → text from LUQ sentences CSV
                idx_to_text = {
                    int(i): str(t).strip()
                    for i, t in zip(luq_df["sentence_idx"].values,
                                    luq_df["sentence"].values)
                }
                score = np.array([
                    text_lut.get(idx_to_text.get(int(i), ""), np.nan)
                    for i in d["sent_idx"]
                ])
        if score is None:
            score = d["luq"].astype(np.float64)

        n_valid = min(ecs_l.shape[1], score.shape[0])
        ecs_l, ecscopy_l, pks_l = ecs_l[:, :n_valid], ecscopy_l[:, :n_valid], pks_l[:, :n_valid]
        score = score[:n_valid]

        # Drop sentences with no label (NaN) from this gen.
        ok = np.isfinite(score)
        if ok.sum() == 0:
            continue
        ecs_l, ecscopy_l, pks_l, score = ecs_l[:, ok], ecscopy_l[:, ok], pks_l[:, ok], score[ok]
        y = (score > args.hallu_thresh).astype(int)

        all_ecs.append(ecs_l); all_ecscopy.append(ecscopy_l)
        all_pks.append(pks_l); all_luq.append(score)

        a_ecs  = auroc_per_layer_single(ecs_l,     y, hallu_high=False)
        a_ecsc = auroc_per_layer_single(ecscopy_l, y, hallu_high=False)
        a_pks  = auroc_per_layer_single(pks_l,     y, hallu_high=True)
        d_ecs  = cohens_d_per_layer_single(ecs_l,     y)
        d_ecsc = cohens_d_per_layer_single(ecscopy_l, y)
        d_pks  = cohens_d_per_layer_single(pks_l,     y)
        pg_ecs_auroc.append(a_ecs); pg_ecscopy_auroc.append(a_ecsc); pg_pks_auroc.append(a_pks)
        pg_ecs_d.append(d_ecs); pg_ecscopy_d.append(d_ecsc); pg_pks_d.append(d_pks)

        layers_arr = np.arange(n_layers)
        pd.DataFrame({"sample": si, "gen": k, "layer": layers_arr,
                      "ecs_auroc": np.round(a_ecs, 5),
                      "ecs_copy_auroc": np.round(a_ecsc, 5),
                      "pks_auroc": np.round(a_pks, 5)}).to_csv(
            auroc_csv_path, mode="a", header=not auroc_hdr, index=False)
        auroc_hdr = True
        pd.DataFrame({"sample": si, "gen": k, "layer": layers_arr,
                      "ecs_cohens_d": np.round(d_ecs, 5),
                      "ecs_copy_cohens_d": np.round(d_ecsc, 5),
                      "pks_cohens_d": np.round(d_pks, 5)}).to_csv(
            cohens_csv_path, mode="a", header=not cohens_hdr, index=False)
        cohens_hdr = True

    if not all_ecs:
        print("No labelled sentences found; nothing to plot.")
        sys.exit(1)

    _refresh_outputs(
        all_ecs, all_ecscopy, all_pks, all_luq,
        pg_ecs_auroc, pg_ecscopy_auroc, pg_pks_auroc,
        pg_ecs_d, pg_ecscopy_d, pg_pks_d,
        n_layers, out_dir, args.hallu_thresh,
    )
    luq_all = np.concatenate(all_luq)
    print(f"\n[metrics-only] {luq_all.shape[0]} sentences, "
          f"{int((luq_all > args.hallu_thresh).sum())} hallucinated. Outputs in {out_dir}")


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
    parser.add_argument("--copying-top-frac", type=float, default=0.15,
                        help="Fraction of heads (globally, by OV copying score) to treat as "
                             "Copying Heads (default 0.15 = top 15%%). Copying heads are a "
                             "minority; a fixed score cut over-selects.")
    parser.add_argument("--copying-thresh", type=float, default=None,
                        help="Alternative to --copying-top-frac: absolute OV positive-real-"
                             "eigenvalue-fraction cut. If set, overrides --copying-top-frac.")
    parser.add_argument("--note-idx",     type=int,   default=None,
                        help="Restrict to one note index (default: all K)")
    parser.add_argument("--notes",        type=int,   default=None, metavar="N",
                        help="Process first N generations per sample (default: all K). "
                             "Overridden by --note-idx.")
    parser.add_argument("--samples",      type=int,   default=None,
                        help="Max number of samples to process")
    parser.add_argument("--no-cache",     action="store_true")
    parser.add_argument("--label-col",      default="uncertainty",
                        help="Column in the sentences CSV used as the hallucination score "
                             "(default 'uncertainty' = LUQ). Swap to relabel with another metric.")
    parser.add_argument("--faithful-label", default="Faithful",
                        help="String value in --label-col that means faithful (converted to 0.0); "
                             "all other values become 1.0. Only used for categorical label columns "
                             "(default: 'Faithful').")
    parser.add_argument("--labels-suffix",  default="_sentences.csv",
                        help="Filename suffix for label CSVs in --labels-dir. "
                             "Default '_sentences.csv' matches LUQ output. "
                             "Use '_sentence_judge.csv' for llm_judge sentence/span output.")
    parser.add_argument("--labels-dir",   default=None,
                        help="Directory of per-sentence label CSVs for --metrics-only "
                             "(default: --sentences dir). Files: {sample}_note_{k}_sentences.csv "
                             "with columns sentence_idx + --label-col.")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Rebuild all metrics/plots from saved activations/*.npz using "
                             "--label-col (optionally from --labels-dir) WITHOUT loading the "
                             "model. Use to re-label with a different uncertainty metric.")
    args = parser.parse_args()

    gen_dir  = Path(args.generations)
    sent_dir = Path(args.sentences)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = out_dir / "sample_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_only:
        run_metrics_only(args, act_dir, out_dir)
        return

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
    if args.copying_thresh is not None:
        copying_mask, copying_sig = select_copying_heads(copying_scores, thresh=args.copying_thresh)
    else:
        copying_mask, copying_sig = select_copying_heads(copying_scores, top_frac=args.copying_top_frac)
    n_copy = int(copying_mask.sum())
    layers_with_copy = int((copying_mask.any(axis=1)).sum())
    print(f"  Copying heads ({copying_sig}): {n_copy}/{copying_scores.size} "
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

        if args.note_idx is not None:
            note_indices = [args.note_idx]
        elif args.notes is not None:
            note_indices = range(min(args.notes, K))
        else:
            note_indices = range(K)

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
            col = args.label_col if args.label_col in sent_df.columns else "uncertainty"
            if col not in sent_df.columns:
                print(f"[{note_name}] column '{col}' not in sentences CSV — skip")
                continue
            luq_scores = sent_df[col].values.astype(np.float64)
            sent_idx   = (sent_df["sentence_idx"].values
                          if "sentence_idx" in sent_df.columns
                          else np.arange(len(sentences)))

            keep       = [not is_soap_header(s) for s in sentences]
            sentences  = [s for s, m in zip(sentences, keep) if m]
            luq_scores = luq_scores[keep]
            sent_idx   = sent_idx[keep]

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

            # ── Pre-compute note token strings and word spans (for tokens NPZ) ──
            tokenizer      = model.tokenizer
            note_token_ids = tokens[0, transcript_len:].tolist()

            # Prefix-decode trick: handles multi-byte BPE tokens correctly.
            tok_strs, _prev, _dec = [], 0, ""
            for i in range(len(note_token_ids)):
                _dec = tokenizer.decode(note_token_ids[:i + 1], skip_special_tokens=False)
                tok_strs.append(_dec[_prev:])
                _prev = len(_dec)

            # Group note tokens into words: new word when token starts with ▁ or space.
            word_spans_list: List[List[int]] = []
            for i, tok in enumerate(tok_strs):
                if i == 0 or tok.startswith("▁") or tok.startswith(" ") or tok == "\n":
                    word_spans_list.append([i, i + 1])
                else:
                    if word_spans_list:
                        word_spans_list[-1][1] = i + 1
                    else:
                        word_spans_list.append([i, i + 1])
            word_spans_arr = np.array(word_spans_list, dtype=np.int32)  # (n_words, 2)
            word_strs = [
                "".join(tok_strs[ws:we]) for ws, we in word_spans_arr
            ]

            # ── Load from disk cache if available ─────────────────────────────
            cache_npz  = cache_dir / f"{note_name}.npz"
            tokens_npz = act_dir   / f"sample_{si:03d}_gen_{k:02d}_tokens.npz"

            cached_ok = False
            if not args.no_cache and cache_npz.exists() and tokens_npz.exists():
                d = np.load(str(cache_npz))
                # ecs_copy depends on the copying-head selection; invalidate if it changed.
                cached_sig = str(d["copying_sig"]) if "copying_sig" in d else None
                if cached_sig == copying_sig:
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
                    ecs_l, ecscopy_l, pks_l, pks_tok_l, ecs_word_l, ecs_word_copy_l = \
                        compute_ecs_pks_single_pass(
                            model, tokens, transcript_len, spans,
                            device=args.device, top_k_frac=args.top_k_frac,
                            copying_mask=copying_mask,
                            word_spans=word_spans_arr,
                        )
                except Exception as exc:
                    print(f"forward ERROR: {exc}")
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    continue

                np.savez_compressed(str(cache_npz),
                                    ecs=ecs_l, ecs_copy=ecscopy_l, pks=pks_l,
                                    copying_sig=np.array(copying_sig))

                # Word-level PKS: SUM token PKS within each word (ReDeEP chunk-level
                # PKS sums token scores, §4.2 — not mean).
                if len(word_spans_arr):
                    pks_word_l = np.stack([
                        np.nansum(pks_tok_l[:, ws:we], axis=1)
                        for ws, we in word_spans_arr
                    ], axis=1)                                  # (n_layers, n_words)
                else:
                    pks_word_l = np.zeros((pks_tok_l.shape[0], 0), dtype=np.float32)

                # Token/word-level NPZ. ECS + PKS are precomputed (faithful to Eq 3
                # and chunk-level PKS); strings/spans kept for label alignment.
                np.savez_compressed(
                    str(tokens_npz),
                    pks_tok       = pks_tok_l.astype(np.float32),       # (n_layers, n_note_tokens)
                    pks_word      = pks_word_l.astype(np.float32),      # (n_layers, n_words) SUM
                    ecs_word      = ecs_word_l.astype(np.float32),      # (n_layers, n_words)
                    ecs_word_copy = ecs_word_copy_l.astype(np.float32), # (n_layers, n_words)
                    token_strs    = np.array(tok_strs, dtype=object),
                    word_spans    = word_spans_arr,
                    word_strs     = np.array(word_strs, dtype=object),
                )

                n_valid = min(ecs_l.shape[1], len(luq_scores))
                ecs_l, ecscopy_l, pks_l = (ecs_l[:, :n_valid],
                                           ecscopy_l[:, :n_valid],
                                           pks_l[:, :n_valid])
                luq_v = luq_scores[:n_valid]
                print(f"ECS={np.nanmean(ecs_l):.3f}  PKS={pks_l.mean():.3f}  OK")

            sidx_v = sent_idx[:luq_v.shape[0]]

            # ── Save per-gen activation file (all layers + heads, backtrackable) ──
            # sent_idx + label_col let the analysis be RE-LABELLED later (--metrics-only)
            # with a different hallucination metric, without re-running the model.
            np.savez_compressed(
                act_dir / f"sample_{si:03d}_gen_{k:02d}.npz",
                ecs       = ecs_l.astype(np.float32),      # (n_layers, n_sents) all heads
                ecs_copy  = ecscopy_l.astype(np.float32),  # (n_layers, n_sents) copying heads
                pks       = pks_l.astype(np.float32),
                luq       = luq_v.astype(np.float32),
                labels    = (luq_v > args.hallu_thresh).astype(np.int8),
                sent_idx  = sidx_v.astype(np.int64),
                label_col = np.array(args.label_col),
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
