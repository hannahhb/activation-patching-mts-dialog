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
