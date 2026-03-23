"""
Experiment 1 — Per-token mechanistic profiling.

For each generated token computes:
  - Direct logit attribution (DLA) per layer, split into attn/MLP contributions
  - Lookback ratio: fraction of attention directed at source (transcript) positions
  - Source attention entropy: Shannon entropy over source positions (upper layers)
  - Extractive score: longest verbatim n-gram match with source (no model needed)

All contributions use the linear approximation of the final layer norm following
the TransformerLens DLA convention: scale each component output by the cached
LN scale factor at that position, then dot with the unembedding vector.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from config import (
    EARLY_LAYERS, MID_LAYERS, LATE_LAYERS, UPPER_LAYERS, N_LAYERS,
)
from data import extractive_scores


# ── Hook names ─────────────────────────────────────────────────────────────────

def _attn_out(L: int) -> str:  return f"blocks.{L}.hook_attn_out"
def _mlp_out(L: int)  -> str:  return f"blocks.{L}.hook_mlp_out"
def _pattern(L: int)  -> str:  return f"blocks.{L}.attn.hook_pattern"


def cache_filter(name: str) -> bool:
    """Names filter for run_with_cache — store only what we need."""
    return (
        name.endswith("hook_attn_out")
        or name.endswith("hook_mlp_out")
        or "attn.hook_pattern" in name
        or name == "ln_final.hook_scale"
    )


# ── DLA helpers ────────────────────────────────────────────────────────────────

def _dla_for_position(
    model:    HookedTransformer,
    cache,
    pred_pos: int,
    token_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DLA for attn_out and mlp_out at prediction position `pred_pos`
    for the token `token_id`.

    Returns:
        attn_dla : [n_layers]  signed logit contribution of attn_out per layer
        mlp_dla  : [n_layers]  signed logit contribution of mlp_out per layer
    """
    n_layers = model.cfg.n_layers

    # Final LN scale at this position: [1]
    # cache["ln_final.hook_scale"] shape: [batch, seq, 1]
    ln_scale = cache["ln_final.hook_scale"][0, pred_pos, 0]  # scalar tensor

    # Unembedding direction for this token: [d_model]
    # model.W_U: [d_model, d_vocab]
    # Also fold in ln_final.w (element-wise gain) if the model has it
    w_u_col = model.W_U[:, token_id]                          # [d_model]
    if hasattr(model, "ln_final") and hasattr(model.ln_final, "w"):
        w_u_col = model.ln_final.w * w_u_col                  # fold gain

    # Effective direction scaled by LN denominator
    direction = w_u_col / (ln_scale + 1e-8)                   # [d_model]

    attn_dla = np.zeros(n_layers)
    mlp_dla  = np.zeros(n_layers)

    for L in range(n_layers):
        attn_out = cache[_attn_out(L)][0, pred_pos, :]        # [d_model]
        mlp_out  = cache[_mlp_out(L)][0, pred_pos, :]         # [d_model]

        attn_dla[L] = (attn_out @ direction).item()
        mlp_dla[L]  = (mlp_out  @ direction).item()

    return attn_dla, mlp_dla


def _band_aggregate(
    per_layer: np.ndarray,
    absolute:  bool = True,
) -> Tuple[float, float, float]:
    """Aggregate per-layer array into (early, mid, late) band means."""
    fn = np.abs if absolute else lambda x: x
    return (
        float(fn(per_layer[EARLY_LAYERS]).mean()),
        float(fn(per_layer[MID_LAYERS]).mean()),
        float(fn(per_layer[LATE_LAYERS]).mean()),
    )


# ── Lookback ratio ─────────────────────────────────────────────────────────────

def _lookback_ratio(
    cache,
    pred_pos:   int,
    prompt_len: int,
    layers:     List[int],
) -> float:
    """
    Mean fraction of attention weight directed at source (transcript) positions
    across all heads in the given layer set, at query position `pred_pos`.
    """
    ratios = []
    for L in layers:
        # hook_pattern: [batch, n_heads, seq_q, seq_k]
        pat = cache[_pattern(L)][0, :, pred_pos, :]  # [n_heads, seq_k]
        src_weight  = pat[:, :prompt_len].sum(dim=-1)   # [n_heads]
        total_weight = pat.sum(dim=-1).clamp(min=1e-8)  # [n_heads]
        ratios.append((src_weight / total_weight).mean().item())
    return float(np.mean(ratios)) if ratios else 0.0


# ── Source attention entropy ───────────────────────────────────────────────────

def _source_attention_entropy(
    cache,
    pred_pos:   int,
    prompt_len: int,
    layers:     List[int],
) -> float:
    """
    Mean Shannon entropy of attention over source positions (upper layers only).
    Entropy is computed after renormalising attention to sum to 1 over source only.
    """
    entropies = []
    for L in layers:
        pat = cache[_pattern(L)][0, :, pred_pos, :prompt_len]  # [n_heads, prompt_len]
        # Renormalise to get a proper distribution over source positions
        pat_norm = pat / (pat.sum(dim=-1, keepdim=True) + 1e-8)
        # Shannon entropy: H = -sum(p log p)
        log_p    = (pat_norm + 1e-10).log()
        H        = -(pat_norm * log_p).sum(dim=-1)  # [n_heads]
        entropies.append(H.mean().item())
    return float(np.mean(entropies)) if entropies else 0.0


# ── Main profiling function ────────────────────────────────────────────────────

def profile_example(
    model:          HookedTransformer,
    cache,
    full_tokens:    torch.Tensor,
    prompt_len:     int,
    section_labels: List[str],
) -> pd.DataFrame:
    """
    Build the per-token mechanistic DataFrame for one ACI-Bench example.

    Args:
        model          : loaded HookedTransformer (Gemma 2 9B-IT)
        cache          : ActivationCache from run_with_cache
        full_tokens    : [1, total_seq_len] integer token IDs (prompt + generated)
        prompt_len     : number of tokens in the prompt (before the generated note)
        section_labels : SOAP section label for each generated token

    Returns:
        DataFrame with one row per generated token.
    """
    n_gen = full_tokens.shape[1] - prompt_len
    ids   = full_tokens[0].tolist()

    # Extractive scores (no model needed)
    src_ids = ids[:prompt_len]
    gen_ids = ids[prompt_len:]
    ext_scores = extractive_scores(gen_ids, src_ids)

    rows = []
    for i in range(n_gen):
        pred_pos = prompt_len - 1 + i   # position making the prediction
        token_id = ids[prompt_len + i]  # the token that was generated

        attn_dla, mlp_dla = _dla_for_position(model, cache, pred_pos, token_id)

        attn_early, attn_mid, attn_late = _band_aggregate(attn_dla)
        mlp_early,  mlp_mid,  mlp_late  = _band_aggregate(mlp_dla)

        total_abs = attn_early + attn_mid + attn_late + mlp_early + mlp_mid + mlp_late + 1e-8
        attn_frac_total = (attn_early + attn_mid + attn_late) / total_abs
        mlp_frac_total  = (mlp_early  + mlp_mid  + mlp_late)  / total_abs

        lookback   = _lookback_ratio(cache, pred_pos, prompt_len, UPPER_LAYERS)
        src_entropy = _source_attention_entropy(cache, pred_pos, prompt_len, UPPER_LAYERS)

        rows.append({
            "token_idx":                i,
            "token_id":                 token_id,
            "token":                    model.tokenizer.decode([token_id]),
            "section":                  section_labels[i] if i < len(section_labels) else "unknown",
            "extractive_score":         ext_scores[i],
            # Absolute mean DLA per band × component
            "attn_contribution_early":  attn_early,
            "attn_contribution_mid":    attn_mid,
            "attn_contribution_late":   attn_late,
            "mlp_contribution_early":   mlp_early,
            "mlp_contribution_mid":     mlp_mid,
            "mlp_contribution_late":    mlp_late,
            # Signed DLA (useful for Experiment 3 candidate selection)
            "attn_dla_total":           float(attn_dla.sum()),
            "mlp_dla_total":            float(mlp_dla.sum()),
            # Derived ratios
            "attn_fraction":            attn_frac_total,
            "mlp_fraction":             mlp_frac_total,
            "lookback_ratio":           lookback,
            "source_attention_entropy": src_entropy,
        })

    return pd.DataFrame(rows)


# ── Per-encounter aggregation ─────────────────────────────────────────────────

def aggregate_encounter_features(token_df: pd.DataFrame) -> Dict[str, float]:
    """
    Collapse the per-token DataFrame into scalar features used for
    Experiment 2 correlation analysis.
    """
    numeric = token_df.select_dtypes(include="number")
    means   = numeric.mean().to_dict()

    # Also compute per-section means for attn/mlp fractions
    for section in token_df["section"].unique():
        sub = token_df[token_df["section"] == section]
        means[f"attn_frac_{section}"] = sub["attn_fraction"].mean()
        means[f"mlp_frac_{section}"]  = sub["mlp_fraction"].mean()
        means[f"lookback_{section}"]  = sub["lookback_ratio"].mean()

    return means
