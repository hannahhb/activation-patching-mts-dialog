"""
Visualisation helpers for activation patching results.
All plots are saved to `results/` and optionally displayed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from config import ATTN_TARGET_LAYER, MLP_TARGET_LAYER, RESULTS_DIR


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Per-example plots ──────────────────────────────────────────────────────

def plot_layer_comparison(result: Dict, example_idx: int, show: bool = False):
    """
    Side-by-side bars:
      Left  — residual stream (all layers, full picture)
      Right — attn_out vs mlp_out (all layers)
    """
    _ensure_results_dir()
    layers       = np.arange(len(result["resid_scores"]))
    resid_scores = result["resid_scores"]
    attn_scores  = result["attn_scores"]
    mlp_scores   = result["mlp_scores"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    # ── Left: residual stream ──────────────────────────────────────────────
    ax = axes[0]
    ax.bar(layers, resid_scores, color="steelblue", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="green",  linewidth=0.8, linestyle="--", label="Full recovery")
    ax.axvline(ATTN_TARGET_LAYER, color="orange", linewidth=1.5, linestyle=":",
               label=f"Attn target (L{ATTN_TARGET_LAYER})")
    ax.axvline(MLP_TARGET_LAYER,  color="red",    linewidth=1.5, linestyle=":",
               label=f"MLP target (L{MLP_TARGET_LAYER})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalised Recovery")
    ax.set_title("Residual Stream Patching")
    ax.legend(fontsize=8)

    # ── Right: attn-out vs mlp-out ─────────────────────────────────────────
    ax = axes[1]
    w  = 0.38
    ax.bar(layers - w/2, attn_scores, width=w, color="orange", alpha=0.85, label="Attention-Out")
    ax.bar(layers + w/2, mlp_scores,  width=w, color="crimson", alpha=0.85, label="MLP-Out")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="green",  linewidth=0.8, linestyle="--")
    ax.axvline(ATTN_TARGET_LAYER, color="orange", linewidth=1.5, linestyle=":")
    ax.axvline(MLP_TARGET_LAYER,  color="red",    linewidth=1.5, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_title("Attention-Out vs MLP-Out Patching")
    ax.legend()

    plt.suptitle(
        f"MTS Dialog — Activation Patching Recovery  (example #{example_idx})\n"
        f"Clean LD={result['clean_ld']:+.2f}  Corrupted LD={result['corrupted_ld']:+.2f}",
        fontsize=10
    )
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"layer_patching_ex{example_idx:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  Saved: {path}")


def plot_head_heatmap(result: Dict, example_idx: int, show: bool = False):
    """Per-head recovery heatmap for the attention window."""
    _ensure_results_dir()
    head_scores = result["head_scores"]
    head_layers = result["head_layers"]
    n_heads     = head_scores.shape[1]

    fig, ax = plt.subplots(figsize=(max(10, n_heads * 1.2), max(4, len(head_layers) * 0.6)))
    sns.heatmap(
        head_scores,
        ax=ax,
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in head_layers],
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=1.0,
        linewidths=0.4, linecolor="grey",
    )
    ax.set_title(
        f"Per-Head Patching Recovery — example #{example_idx}\n"
        f"Layers {head_layers[0]}–{head_layers[-1]}  "
        f"(high = head carries clean 'copying' signal)"
    )
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"head_heatmap_ex{example_idx:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  Saved: {path}")


def plot_position_scores(result: Dict, example_idx: int, top_n: int = 50, show: bool = False):
    """Two-row plot: position recovery for attn_out and mlp_out target layers."""
    _ensure_results_dir()

    fig, axes = plt.subplots(2, 1, figsize=(20, 9))

    for ax, label, tokens, scores in [
        (axes[0], f"Attention-Out (L{ATTN_TARGET_LAYER})",
         result["attn_pos_tokens"], result["attn_pos_scores"]),
        (axes[1], f"MLP-Out (L{MLP_TARGET_LAYER})",
         result["mlp_pos_tokens"],  result["mlp_pos_scores"]),
    ]:
        n       = min(top_n, len(scores))
        vals    = scores[-n:]
        labels  = tokens[-n:]
        colours = ["steelblue" if v >= 0 else "tomato" for v in vals]

        ax.bar(range(n), vals, color=colours, alpha=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(1, color="green", linewidth=0.8, linestyle="--", label="Full recovery")
        ax.set_ylabel("Normalised Recovery")
        ax.set_title(f"{label} — example #{example_idx} (last {n} positions)")
        ax.legend(fontsize=8)

    plt.suptitle("Which dialogue tokens drive the prediction?", fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"position_patching_ex{example_idx:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  Saved: {path}")


# ── Aggregate plots (across all examples) ─────────────────────────────────

def plot_aggregate(all_results: List[Dict], show: bool = False):
    """
    Mean ± std recovery curves across all examples.
    One panel for attn_out, one for mlp_out.
    """
    _ensure_results_dir()
    if not all_results:
        return

    attn_mat = np.stack([r["attn_scores"] for r in all_results])   # [N, L]
    mlp_mat  = np.stack([r["mlp_scores"]  for r in all_results])

    layers = np.arange(attn_mat.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, mat, colour, label in [
        (axes[0], attn_mat,  "orange",  "Attention-Out"),
        (axes[1], mlp_mat,   "crimson", "MLP-Out"),
    ]:
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        ax.bar(layers, mean, color=colour, alpha=0.7, label=label)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.25, color=colour)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(1, color="green", linewidth=0.8, linestyle="--", label="Full recovery")
        ax.axvline(ATTN_TARGET_LAYER, color="orange", linewidth=1.5, linestyle=":")
        ax.axvline(MLP_TARGET_LAYER,  color="red",    linewidth=1.5, linestyle=":")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Normalised Recovery")
        ax.set_title(f"{label} — mean ± std (N={len(all_results)} examples)")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"MTS Dialog Summarisation — Aggregate Activation Patching  (N={len(all_results)})",
        fontsize=11
    )
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "aggregate_layer_patching.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  Saved: {path}")


def plot_aggregate_head_heatmap(all_results: List[Dict], show: bool = False):
    """Mean head recovery heatmap across all examples."""
    _ensure_results_dir()
    if not all_results:
        return

    head_mat  = np.stack([r["head_scores"] for r in all_results]).mean(axis=0)
    head_layers = all_results[0]["head_layers"]
    n_heads     = head_mat.shape[1]

    fig, ax = plt.subplots(figsize=(max(10, n_heads * 1.2), max(4, len(head_layers) * 0.6)))
    sns.heatmap(
        head_mat,
        ax=ax,
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in head_layers],
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.8,
        linewidths=0.4, linecolor="grey",
    )
    ax.set_title(
        f"Per-Head Patching Recovery — mean over {len(all_results)} MTS examples\n"
        f"Layers {head_layers[0]}–{head_layers[-1]}"
    )
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "aggregate_head_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  Saved: {path}")
