"""
plots.py
========
All matplotlib/seaborn visualisation helpers.

Nothing in this module touches model inference or file I/O beyond saving figures.

Provides
--------
  plot_scatter              ECS vs PKS quadrant scatter
  plot_heatmap              two-row ECS / PKS heatmap across token positions
  plot_risk_bar             per-token hallucination risk bar chart
  plot_layer_discriminability  two-panel AUROC + Cohen's d per layer
"""

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import Q_COLORS
from metrics import hallucination_risk

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _token_color(e: float, p: float, em: float, pm: float) -> str:
    if   e >= em and p <  pm: return Q_COLORS["extractive"]
    elif e <  em and p >= pm: return Q_COLORS["parametric"]
    elif e >= em and p >= pm: return Q_COLORS["synthesized"]
    else:                      return Q_COLORS["hallucinatory"]


# ─────────────────────────────────────────────
# Public plot functions
# ─────────────────────────────────────────────

def plot_scatter(
    ecs: np.ndarray,
    pks: np.ndarray,
    tokens: List[str],
    title: str,
    ax: Optional[plt.Axes] = None,
    highlight: Optional[List[int]] = None,
    annotate_stride: int = 5,
) -> plt.Axes:
    """
    ECS vs PKS quadrant scatter.  Each point is one note token, coloured by
    quadrant.  Optional `highlight` list marks specific positions with a star.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    em = np.median(ecs)
    pm = np.median(pks)

    colors = [_token_color(e, p, em, pm) for e, p in zip(ecs, pks)]
    ax.scatter(ecs, pks, c=colors, s=55, alpha=0.75, edgecolors="white", linewidth=0.4)

    for i, (e, p, tok) in enumerate(zip(ecs, pks, tokens)):
        if i % annotate_stride == 0:
            label = tok.replace("▁", "").replace("Ġ", "").strip()[:10]
            ax.annotate(label, (e, p), fontsize=5.5, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")

    if highlight:
        hx = [ecs[i] for i in highlight if i < len(ecs)]
        hy = [pks[i] for i in highlight if i < len(pks)]
        ax.scatter(hx, hy, s=200, c="black", marker="*", zorder=6,
                   label="Hallucinated span")

    ax.axvline(em, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(pm, color="gray", ls="--", lw=0.8, alpha=0.5)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    pad = 0.02
    ax.text(xhi - pad*(xhi-xlo), ylo + pad*(yhi-ylo), "Extractive",
            ha="right", va="bottom", fontsize=7, color=Q_COLORS["extractive"], style="italic")
    ax.text(xlo + pad*(xhi-xlo), yhi - pad*(yhi-ylo), "Parametric",
            ha="left",  va="top",    fontsize=7, color=Q_COLORS["parametric"],  style="italic")
    ax.text(xhi - pad*(xhi-xlo), yhi - pad*(yhi-ylo), "Synthesized",
            ha="right", va="top",    fontsize=7, color=Q_COLORS["synthesized"], style="italic")
    ax.text(xlo + pad*(xhi-xlo), ylo + pad*(yhi-ylo), "Hallucination\nRisk",
            ha="left",  va="bottom", fontsize=7, color=Q_COLORS["hallucinatory"], style="italic")

    legend_patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in Q_COLORS.items()]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right")

    ax.set_xlabel("External Context Score (ECS)", fontsize=10)
    ax.set_ylabel("Parametric Knowledge Score (PKS)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return ax


def plot_heatmap(
    ecs: np.ndarray,
    pks: np.ndarray,
    tokens: List[str],
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Two-row heatmap: top row = ECS, bottom row = PKS, columns = note tokens."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(14, len(tokens) // 3), 3))

    disp = [t.replace("▁", "").replace("Ġ", "").strip()[:7] for t in tokens]
    data = np.stack([ecs, pks])   # (2, N)

    sns.heatmap(data, ax=ax, xticklabels=disp, yticklabels=["ECS", "PKS"],
                cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.25, linecolor="white", cbar_kws={"shrink": 0.6})
    ax.tick_params(axis="x", labelsize=5.5, rotation=60)
    ax.tick_params(axis="y", labelsize=9,   rotation=0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    return ax


def plot_risk_bar(
    ecs: np.ndarray,
    pks: np.ndarray,
    tokens: List[str],
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of hallucination risk per token, coloured by risk magnitude."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(14, len(tokens) // 3), 3))

    risk   = hallucination_risk(ecs, pks)
    disp   = [t.replace("▁", "").replace("Ġ", "").strip()[:7] for t in tokens]
    colors = plt.cm.RdYlGn_r(risk)

    ax.bar(range(len(risk)), risk, color=colors, width=0.85, edgecolor="none")
    ax.set_xticks(range(len(disp)))
    ax.set_xticklabels(disp, rotation=60, ha="right", fontsize=5.5)
    ax.axhline(0.5, color="#B71C1C", ls="--", lw=1.2, alpha=0.7, label="Risk threshold 0.5")
    ax.set_ylabel("Hallucination Risk", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    return ax


def plot_layer_discriminability(
    disc: Dict,
    title: str,
    axes: Optional[Tuple] = None,
    metric_a: str = "ecs",
    metric_b: str = "pks",
    label_a: str = "ECS",
    label_b: str = "PKS",
    color_a: str = "#2196F3",
    color_b: str = "#FF9800",
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Two-panel figure: AUROC per layer (top) and Cohen's d per layer (bottom).

    Parameters
    ----------
    disc     : dict from layer_discriminability() or dla_discriminability().
    title    : suptitle string.
    axes     : optional (ax_auroc, ax_d) tuple; created if None.
    metric_a : key prefix for first metric  (default "ecs").
    metric_b : key prefix for second metric (default "pks").
    """
    n_layers = len(disc[f"{metric_a}_auroc"])
    layers   = np.arange(n_layers)
    width    = 0.35

    if axes is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n_layers // 2), 8),
                                        sharex=True)
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        ax1, ax2 = axes

    auroc_a = disc[f"{metric_a}_auroc"]
    auroc_b = disc[f"{metric_b}_auroc"]
    d_a     = disc[f"{metric_a}_cohens_d"]
    d_b     = disc[f"{metric_b}_cohens_d"]

    # AUROC panel
    ax1.plot(layers, auroc_a, marker="s", color=color_a,
             label=f"{label_a} AUROC", lw=1.8, markersize=4)
    ax1.plot(layers, auroc_b, marker="o", color=color_b,
             label=f"{label_b} AUROC", lw=1.8, markersize=4)
    ax1.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.7, label="Chance (0.5)")
    ax1.fill_between(layers, 0.5, auroc_a, where=auroc_a > 0.5, alpha=0.12, color=color_a)
    ax1.fill_between(layers, 0.5, auroc_b, where=auroc_b > 0.5, alpha=0.12, color=color_b)
    ax1.set_ylabel("AUROC", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_title("Layer-wise AUROC — hallucinated vs. clean tokens", fontsize=10)

    # Cohen's d panel
    ax2.bar(layers - width / 2, d_a, width, color=color_a, alpha=0.75,
            label=f"{label_a} Cohen's d")
    ax2.bar(layers + width / 2, d_b, width, color=color_b, alpha=0.75,
            label=f"{label_b} Cohen's d")
    ax2.axhline(0, color="gray", ls="-", lw=0.8, alpha=0.6)
    ax2.set_ylabel("Cohen's d  (halluc − clean)", fontsize=10)
    ax2.set_xlabel("Layer", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_title(
        "Layer-wise Cohen's d  (negative = hallucinated tokens score lower)", fontsize=10
    )

    return ax1, ax2
