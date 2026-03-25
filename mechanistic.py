"""
Per-token mechanistic analysis: DLA and activation patching.

DLA (Direct Logit Attribution)
  For each generated token, decompose the final logit into per-layer
  attn_out and mlp_out contributions using the linear LN approximation.
  Result: DataFrame with columns attn_dla_L{i} and mlp_dla_L{i}.

Activation Patching (zero-ablation)
  For each layer, zero out attn_out (then mlp_out) and measure the drop
  in logit for every generated token. Requires 2×N_LAYERS forward passes.
  Result: two [n_layers, n_gen] numpy arrays (attn_patch, mlp_patch).
  A large positive value = that layer's component is important for that token.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer


# ── Hook name helpers ───────────────────────────────────────────────────────────

def _attn_out(L: int) -> str: return f"blocks.{L}.hook_attn_out"
def _mlp_out(L: int)  -> str: return f"blocks.{L}.hook_mlp_out"


def cache_filter(name: str) -> bool:
    """Keep only what DLA needs: per-layer component outputs + final LN scale."""
    return (
        name.endswith("hook_attn_out")
        or name.endswith("hook_mlp_out")
        or name == "ln_final.hook_scale"
    )


# ── DLA ────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _dla_for_position(
    model:    HookedTransformer,
    cache,
    pred_pos: int,
    token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-layer DLA at one prediction position for one token.

    Uses the linear approximation:
        logit_contribution(component) ≈ component_output · (W_U[:, token] / ln_scale)

    Returns:
        attn_dla : [n_layers]  signed contribution of each attn_out
        mlp_dla  : [n_layers]  signed contribution of each mlp_out
    """
    n_layers = model.cfg.n_layers

    ln_scale = cache["ln_final.hook_scale"][0, pred_pos, 0]
    w_u_col  = model.W_U[:, token_id]
    if hasattr(model, "ln_final") and hasattr(model.ln_final, "w"):
        w_u_col = model.ln_final.w * w_u_col
    direction = w_u_col / (ln_scale + 1e-8)   # [d_model]

    attn_dla = np.zeros(n_layers)
    mlp_dla  = np.zeros(n_layers)
    for L in range(n_layers):
        attn_dla[L] = (cache[_attn_out(L)][0, pred_pos, :] @ direction).item()
        mlp_dla[L]  = (cache[_mlp_out(L)][0, pred_pos, :] @ direction).item()

    return attn_dla, mlp_dla


def profile_example(
    model:       HookedTransformer,
    cache,
    full_tokens: torch.Tensor,
    prompt_len:  int,
) -> pd.DataFrame:
    """
    Per-token DLA DataFrame for one example.

    One row per generated token. Columns:
      token_idx, token_id, token,
      attn_dla_L{0..N-1},  mlp_dla_L{0..N-1},
      attn_fraction, mlp_fraction,      (share of total absolute DLA)
      top_attn_layer, top_mlp_layer     (argmax of abs DLA per component)
    """
    n_layers = model.cfg.n_layers
    n_gen    = full_tokens.shape[1] - prompt_len
    ids      = full_tokens[0].tolist()

    rows = []
    for i in range(n_gen):
        pred_pos = prompt_len - 1 + i
        token_id = ids[prompt_len + i]

        attn_dla, mlp_dla = _dla_for_position(model, cache, pred_pos, token_id)

        row = {
            "token_idx": i,
            "token_id":  token_id,
            "token":     model.tokenizer.decode([token_id]),
        }
        for L in range(n_layers):
            row[f"attn_dla_L{L}"] = float(attn_dla[L])
            row[f"mlp_dla_L{L}"]  = float(mlp_dla[L])

        abs_attn = float(np.abs(attn_dla).sum())
        abs_mlp  = float(np.abs(mlp_dla).sum())
        total    = abs_attn + abs_mlp + 1e-8
        row["attn_fraction"]  = abs_attn / total
        row["mlp_fraction"]   = abs_mlp  / total
        row["top_attn_layer"] = int(np.abs(attn_dla).argmax())
        row["top_mlp_layer"]  = int(np.abs(mlp_dla).argmax())

        rows.append(row)

    return pd.DataFrame(rows)


# ── Activation patching ────────────────────────────────────────────────────────

@torch.no_grad()
def run_activation_patching(
    model:       HookedTransformer,
    full_tokens: torch.Tensor,
    prompt_len:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero-ablation sweep over all layers for attn_out and mlp_out.

    For each layer L and each component (attn/mlp):
      1. Replace that component's output with zeros for all positions.
      2. Measure logit drop for each generated token:
             patch_score = logit_original - logit_ablated
         Positive = the component was pushing the model toward the actual token.

    Requires 2 × N_LAYERS forward passes.

    Returns:
        attn_patch : [n_layers, n_gen]
        mlp_patch  : [n_layers, n_gen]
    """
    n_layers = model.cfg.n_layers
    n_gen    = full_tokens.shape[1] - prompt_len

    # Baseline: logit of the actual generated token at each position
    orig_logits = model(full_tokens)   # [1, seq, vocab]
    orig_vals   = np.array([
        orig_logits[0, prompt_len - 1 + i, full_tokens[0, prompt_len + i].item()].item()
        for i in range(n_gen)
    ])
    del orig_logits
    torch.cuda.empty_cache()

    attn_patch = np.zeros((n_layers, n_gen))
    mlp_patch  = np.zeros((n_layers, n_gen))

    def _zero_hook(value, hook):
        return torch.zeros_like(value)

    for L in range(n_layers):
        for patch_arr, hook_name in (
            (attn_patch, _attn_out(L)),
            (mlp_patch,  _mlp_out(L)),
        ):
            ablated = model.run_with_hooks(
                full_tokens,
                fwd_hooks=[(hook_name, _zero_hook)],
            )
            for i in range(n_gen):
                pred_pos = prompt_len - 1 + i
                tok      = full_tokens[0, prompt_len + i].item()
                patch_arr[L, i] = orig_vals[i] - ablated[0, pred_pos, tok].item()
            del ablated
            torch.cuda.empty_cache()

        print(f"    patching layer {L}/{n_layers - 1}", end="\r")

    print()
    return attn_patch, mlp_patch


# ── Lookback Ratio ─────────────────────────────────────────────────────────────

def lookback_ratio(token_df: pd.DataFrame, early_layers: List[int]) -> float:
    """
    Fraction of total attention DLA that comes from early layers.

    High (→1) = model relies on early-layer attention (direct source retrieval).
    Low (→0)  = attention attribution concentrated in later layers.
    """
    early_cols  = [f"attn_dla_L{L}" for L in early_layers
                   if f"attn_dla_L{L}" in token_df.columns]
    all_attn    = [c for c in token_df.columns if c.startswith("attn_dla_L")]
    if not all_attn:
        return 0.0
    early_mass = float(token_df[early_cols].abs().values.sum()) if early_cols else 0.0
    total_mass = float(token_df[all_attn].abs().values.sum()) + 1e-8
    return round(early_mass / total_mass, 4)


# ── Source Attention Entropy ───────────────────────────────────────────────────

def source_attention_entropy(
    token_df:     pd.DataFrame,
    upper_layers: List[int],
) -> float:
    """
    Normalised Shannon entropy of per-layer attention DLA magnitudes in the
    upper half of the network.

    High = attribution spread across many upper layers (diffuse processing).
    Low  = concentrated on a few upper layers (focused retrieval).
    """
    upper_cols = [f"attn_dla_L{L}" for L in upper_layers
                  if f"attn_dla_L{L}" in token_df.columns]
    if not upper_cols:
        return 0.0

    mean_abs = token_df[upper_cols].abs().mean(axis=0).values
    total    = mean_abs.sum() + 1e-10
    p        = mean_abs / total
    entropy  = -float(np.sum(p * np.log(p + 1e-10)))
    max_ent  = float(np.log(len(upper_cols)))
    return round(entropy / max_ent, 4) if max_ent > 0 else 0.0


# ── Extractive Score ───────────────────────────────────────────────────────────

def extractive_score(
    generated_note:  str,
    source_dialogue: str,
    n:               int = 4,
) -> float:
    """
    Fraction of n-grams in the generated note that appear in the source dialogue.

    1.0 = fully extractive (every n-gram copied from source).
    0.0 = fully abstractive (no n-gram overlap).
    """
    def _ngrams(text: str, k: int):
        words = text.lower().split()
        return {tuple(words[i: i + k]) for i in range(len(words) - k + 1)}

    gen_grams = _ngrams(generated_note,  n)
    src_grams = _ngrams(source_dialogue, n)
    if not gen_grams:
        return 0.0
    return round(len(gen_grams & src_grams) / len(gen_grams), 4)


# ── Aggregate Encounter Features ───────────────────────────────────────────────

def aggregate_encounter_features(
    token_df:        pd.DataFrame,
    attn_patch:      np.ndarray,
    mlp_patch:       np.ndarray,
    generated_note:  str,
    source_dialogue: str,
    tokenizer,
    early_layers:    List[int],
    mid_layers:      List[int],
    late_layers:     List[int],
    upper_layers:    List[int],
) -> Dict:
    """
    Collapse per-token DLA and patching arrays into a single encounter-level
    feature dict compatible with ``analysis.build_signature_matrix``.

    Band DLA contributions are mean absolute DLA across tokens and layers in
    that band.  Patching importance uses mean positive patch score (importance
    of the component for the actual generated tokens).
    """
    def _band_mean(prefix: str, layers: List[int]) -> float:
        cols = [f"{prefix}_L{L}" for L in layers if f"{prefix}_L{L}" in token_df.columns]
        return float(token_df[cols].abs().mean().mean()) if cols else 0.0

    def _patch_band_mean(patch_arr: np.ndarray, layers: List[int]) -> float:
        rows = [L for L in layers if L < patch_arr.shape[0]]
        if not rows:
            return 0.0
        return float(patch_arr[rows, :].clip(min=0).mean())

    feat: Dict = {
        # ── DLA band contributions ────────────────────────────────────────────
        "attn_contribution_early": _band_mean("attn_dla", early_layers),
        "attn_contribution_mid":   _band_mean("attn_dla", mid_layers),
        "attn_contribution_late":  _band_mean("attn_dla", late_layers),
        "mlp_contribution_early":  _band_mean("mlp_dla",  early_layers),
        "mlp_contribution_mid":    _band_mean("mlp_dla",  mid_layers),
        "mlp_contribution_late":   _band_mean("mlp_dla",  late_layers),
        # ── Patching importance (mean positive drop when ablated) ─────────────
        "attn_patch_early": _patch_band_mean(attn_patch, early_layers),
        "attn_patch_mid":   _patch_band_mean(attn_patch, mid_layers),
        "attn_patch_late":  _patch_band_mean(attn_patch, late_layers),
        "mlp_patch_early":  _patch_band_mean(mlp_patch,  early_layers),
        "mlp_patch_mid":    _patch_band_mean(mlp_patch,  mid_layers),
        "mlp_patch_late":   _patch_band_mean(mlp_patch,  late_layers),
        # ── Aggregate ratios ──────────────────────────────────────────────────
        "attn_fraction":            round(float(token_df["attn_fraction"].mean()), 4),
        "mlp_fraction":             round(float(token_df["mlp_fraction"].mean()),  4),
        "lookback_ratio":           lookback_ratio(token_df, early_layers),
        "source_attention_entropy": source_attention_entropy(token_df, upper_layers),
        "extractive_score":         extractive_score(generated_note, source_dialogue),
    }

    # ── Per-section mlp fractions (requires 'section' column) ─────────────────
    if "section" in token_df.columns:
        for sec_name, sub in token_df.groupby("section"):
            feat[f"mlp_frac_{sec_name}"] = round(float(sub["mlp_fraction"].mean()), 4)

    # ── Input complexity features ─────────────────────────────────────────────
    from data import compute_complexity_features
    feat.update(compute_complexity_features(source_dialogue, tokenizer))

    return feat
