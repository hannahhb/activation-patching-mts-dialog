"""
Activation patching utilities using TransformerLens hooks.

Supports:
  - Layer-level sweeps   : resid_pre, attn_out, mlp_out
  - Per-head sweeps      : attn hook_z decomposed by head
  - Token-position sweep : which positions carry the key signal
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Callable

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from config import N_LAYERS, N_HEADS, HEAD_SWEEP_LAYERS
from metrics import normalised_recovery, logit_diff


# ── Hook name templates ────────────────────────────────────────────────────

HOOK_RESID_PRE = "blocks.{}.hook_resid_pre"
HOOK_ATTN_OUT  = "blocks.{}.hook_attn_out"
HOOK_MLP_OUT   = "blocks.{}.hook_mlp_out"
HOOK_Z         = "blocks.{}.attn.hook_z"      # [batch, seq, n_heads, d_head]


# ── Hook factory ───────────────────────────────────────────────────────────

def _patch_all_positions(clean_act: torch.Tensor) -> Callable:
    """Replace the full activation tensor with the clean version."""
    def hook_fn(value: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return clean_act.to(value.device, value.dtype)
    return hook_fn


def _patch_single_head(clean_z: torch.Tensor, head: int) -> Callable:
    """Replace a single head's output in hook_z."""
    def hook_fn(value: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        value = value.clone()
        value[:, :, head, :] = clean_z[:, :, head, :].to(value.device, value.dtype)
        return value
    return hook_fn


def _patch_single_position(clean_act: torch.Tensor, pos: int) -> Callable:
    """Patch a single token position (for position-level analysis)."""
    def hook_fn(value: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        if pos < value.shape[1]:
            value = value.clone()
            value[:, pos, :] = clean_act[:, pos, :].to(value.device, value.dtype)
        return value
    return hook_fn


# ── Core patching functions ────────────────────────────────────────────────

def patch_layer(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: dict,
    hook_template: str,
    layer: int,
    clean_ld: float,
    corrupted_ld: float,
    correct_id: int,
    wrong_id: int,
) -> float:
    """Patch one layer's activation and return the normalised recovery score."""
    hook_name  = hook_template.format(layer)
    clean_act  = clean_cache[hook_name]
    patched    = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, _patch_all_positions(clean_act))],
    )
    return normalised_recovery(patched, clean_ld, corrupted_ld, correct_id, wrong_id)


def sweep_all_layers(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: dict,
    hook_template: str,
    clean_ld: float,
    corrupted_ld: float,
    correct_id: int,
    wrong_id: int,
    verbose: bool = True,
) -> np.ndarray:
    """Sweep all layers for a given hook type. Returns [n_layers] recovery array."""
    scores = np.zeros(model.cfg.n_layers)
    for layer in range(model.cfg.n_layers):
        score = patch_layer(
            model, corrupted_tokens, clean_cache,
            hook_template, layer,
            clean_ld, corrupted_ld, correct_id, wrong_id,
        )
        scores[layer] = score
        if verbose:
            print(f"    L{layer:02d}  {score:+.4f}")
    return scores


def sweep_heads(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: dict,
    layers: List[int],
    clean_ld: float,
    corrupted_ld: float,
    correct_id: int,
    wrong_id: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Per-head patching for the given layer window.
    Returns [len(layers), n_heads] recovery array.
    """
    scores = np.zeros((len(layers), model.cfg.n_heads))
    for i, layer in enumerate(layers):
        hook_name = HOOK_Z.format(layer)
        clean_z   = clean_cache[hook_name]
        for head in range(model.cfg.n_heads):
            patched = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, _patch_single_head(clean_z, head))],
            )
            scores[i, head] = normalised_recovery(
                patched, clean_ld, corrupted_ld, correct_id, wrong_id
            )
        if verbose:
            print(f"    L{layer:02d}  {[f'{s:.3f}' for s in scores[i]]}")
    return scores


def sweep_positions(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: dict,
    hook_template: str,
    layer: int,
    clean_ld: float,
    corrupted_ld: float,
    correct_id: int,
    wrong_id: int,
    clean_tokens: torch.Tensor,
) -> Tuple[List[str], np.ndarray]:
    """
    Token-position sweep for a single layer.
    Patches one position at a time from the clean run into the corrupted run.

    Returns:
        token_strs : List of token strings for the clean prompt.
        scores     : [seq_len] recovery array.
    """
    hook_name  = hook_template.format(layer)
    clean_act  = clean_cache[hook_name]
    seq_len    = clean_tokens.shape[1]
    token_strs = [model.to_string([t.item()]) for t in clean_tokens[0]]

    scores = np.zeros(seq_len)
    for pos in range(seq_len):
        patched = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, _patch_single_position(clean_act, pos))],
        )
        scores[pos] = normalised_recovery(
            patched, clean_ld, corrupted_ld, correct_id, wrong_id
        )
    return token_strs, scores


# ── Per-example runner ─────────────────────────────────────────────────────

def run_patching_for_example(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    correct_id: int,
    wrong_id: int,
    attn_target: int,
    mlp_target: int,
    head_layers: List[int],
    verbose: bool = True,
) -> Dict:
    """
    Full patching suite for one (clean, corrupted) token pair.

    Returns a dict with:
      resid_scores, attn_scores, mlp_scores  : [n_layers]
      head_scores                             : [len(head_layers), n_heads]
      attn_pos_tokens, attn_pos_scores        : position sweep at attn_target
      mlp_pos_tokens,  mlp_pos_scores         : position sweep at mlp_target
      clean_ld, corrupted_ld
    """
    # ── Baseline forward passes ────────────────────────────────────────────
    clean_logits,     clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, _           = model.run_with_cache(corrupted_tokens)

    clean_ld     = logit_diff(clean_logits,     correct_id, wrong_id)
    corrupted_ld = logit_diff(corrupted_logits, correct_id, wrong_id)

    if verbose:
        print(f"  Clean LD={clean_ld:+.3f}  Corrupted LD={corrupted_ld:+.3f}  "
              f"Gap={clean_ld - corrupted_ld:+.3f}")

    kwargs = dict(
        model=model,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        clean_ld=clean_ld,
        corrupted_ld=corrupted_ld,
        correct_id=correct_id,
        wrong_id=wrong_id,
    )

    # ── Layer sweeps ───────────────────────────────────────────────────────
    if verbose:
        print("  [resid_pre sweep]")
    resid_scores = sweep_all_layers(**kwargs, hook_template=HOOK_RESID_PRE, verbose=verbose)

    if verbose:
        print("  [attn_out sweep]")
    attn_scores  = sweep_all_layers(**kwargs, hook_template=HOOK_ATTN_OUT,  verbose=verbose)

    if verbose:
        print("  [mlp_out sweep]")
    mlp_scores   = sweep_all_layers(**kwargs, hook_template=HOOK_MLP_OUT,   verbose=verbose)

    # ── Per-head sweep ─────────────────────────────────────────────────────
    if verbose:
        print(f"  [head sweep L{head_layers[0]}–L{head_layers[-1]}]")
    head_scores  = sweep_heads(**kwargs, layers=head_layers, verbose=verbose)

    # ── Position sweeps ────────────────────────────────────────────────────
    if verbose:
        print(f"  [position sweep — attn_out L{attn_target}]")
    attn_pos_tokens, attn_pos_scores = sweep_positions(
        **kwargs, hook_template=HOOK_ATTN_OUT, layer=attn_target,
        clean_tokens=clean_tokens,
    )

    if verbose:
        print(f"  [position sweep — mlp_out L{mlp_target}]")
    mlp_pos_tokens, mlp_pos_scores = sweep_positions(
        **kwargs, hook_template=HOOK_MLP_OUT, layer=mlp_target,
        clean_tokens=clean_tokens,
    )

    return dict(
        clean_ld          = clean_ld,
        corrupted_ld      = corrupted_ld,
        resid_scores      = resid_scores,
        attn_scores       = attn_scores,
        mlp_scores        = mlp_scores,
        head_scores       = head_scores,
        head_layers       = head_layers,
        attn_pos_tokens   = attn_pos_tokens,
        attn_pos_scores   = attn_pos_scores,
        mlp_pos_tokens    = mlp_pos_tokens,
        mlp_pos_scores    = mlp_pos_scores,
    )
