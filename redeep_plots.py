"""
redeep_plots.py
===============
Standalone plotting script for ReDeEP results.

Reads from a redeep_out directory produced by redeep_sentence.py and generates:

  1. mean_ecs_pks.png  — mean ECS and PKS per layer, averaged over all sentences,
                         with split lines for uncertain vs certain sentences.

  2. auroc_layers.png  — per-layer AUROC for ECS (inverted) and PKS.
                         No PKS−ECS combined metric.

Usage:
  python redeep_plots.py --redeep-dir sae_experiments/redeep_out
  python redeep_plots.py --redeep-dir redeep_out --hallu-thresh 0.5 --out figs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# Plotting style
# ─────────────────────────────────────────────────────────────────────────────

ECS_COLOR  = "#2171b5"   # blue
PKS_COLOR  = "#cb181d"   # red
CERT_ALPHA = 0.55
UNC_ALPHA  = 1.0
LINEWIDTH  = 1.8


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: mean ECS and PKS per layer
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_ecs_pks(
    df: pd.DataFrame,
    hallu_thresh: float,
    out_path: Path,
) -> None:
    """
    Layer-wise mean ECS and PKS, averaged over all sentences.
    Solid lines = uncertain sentences (U > hallu_thresh).
    Dashed lines = certain sentences.
    """
    labels = (df["luq_score"].values > hallu_thresh)

    ecs_cols = sorted([c for c in df.columns if c.startswith("ecs_l")],
                      key=lambda c: int(c[5:]))
    pks_cols = sorted([c for c in df.columns if c.startswith("pks_l")],
                      key=lambda c: int(c[5:]))

    n_layers = len(ecs_cols)
    layers   = np.arange(n_layers)

    def layer_means(cols, mask):
        return np.array([df.loc[mask, c].mean() for c in cols])

    ecs_unc  = layer_means(ecs_cols,  labels)
    ecs_cert = layer_means(ecs_cols, ~labels)
    pks_unc  = layer_means(pks_cols,  labels)
    pks_cert = layer_means(pks_cols, ~labels)
    ecs_all  = layer_means(ecs_cols, np.ones(len(df), dtype=bool))
    pks_all  = layer_means(pks_cols, np.ones(len(df), dtype=bool))

    n_unc  = int(labels.sum())
    n_cert = int((~labels).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    # Left panel: ECS by group
    ax = axes[0]
    ax.plot(layers, ecs_unc,  color=ECS_COLOR, lw=LINEWIDTH,
            label=f"Uncertain (n={n_unc})")
    ax.plot(layers, ecs_cert, color=ECS_COLOR, lw=LINEWIDTH, ls="--", alpha=0.6,
            label=f"Certain (n={n_cert})")
    ax.plot(layers, ecs_all,  color=ECS_COLOR, lw=1.0, ls=":", alpha=0.4,
            label="All sentences")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean ECS", fontsize=11)
    ax.set_title("External Context Score (ECS) by Layer", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right panel: PKS by group
    ax = axes[1]
    ax.plot(layers, pks_unc,  color=PKS_COLOR, lw=LINEWIDTH,
            label=f"Uncertain (n={n_unc})")
    ax.plot(layers, pks_cert, color=PKS_COLOR, lw=LINEWIDTH, ls="--", alpha=0.6,
            label=f"Certain (n={n_cert})")
    ax.plot(layers, pks_all,  color=PKS_COLOR, lw=1.0, ls=":", alpha=0.4,
            label="All sentences")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean PKS", fontsize=11)
    ax.set_title("Parametric Knowledge Score (PKS) by Layer", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"Layer-wise ECS and PKS  (uncertain threshold U > {hallu_thresh})",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: AUROC per layer — ECS and PKS only
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc_layers(
    df: pd.DataFrame,
    hallu_thresh: float,
    out_path: Path,
    metrics_csv: Path | None = None,
) -> None:
    """
    Per-layer AUROC for ECS (inverted: low ECS -> hallucinated) and PKS.
    Reads from sentence_scores.csv so it can recompute if metrics_csv is absent.
    """
    labels = (df["luq_score"].values > hallu_thresh).astype(int)

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        print("Warning: only one class — AUROC is uninformative.")
        return

    ecs_cols = sorted([c for c in df.columns if c.startswith("ecs_l")],
                      key=lambda c: int(c[5:]))
    pks_cols = sorted([c for c in df.columns if c.startswith("pks_l")],
                      key=lambda c: int(c[5:]))

    n_layers = len(ecs_cols)
    layers   = np.arange(n_layers)

    # Recompute AUROC from raw scores (ignore cached metrics_csv for PKS-ECS)
    ecs_auroc = np.zeros(n_layers)
    pks_auroc = np.zeros(n_layers)

    for i, (ec, pk) in enumerate(zip(ecs_cols, pks_cols)):
        ecs_score = -df[ec].values   # invert: lower ECS -> higher hallucination risk
        pks_score =  df[pk].values
        try:
            ecs_auroc[i] = roc_auc_score(labels, ecs_score)
        except ValueError:
            ecs_auroc[i] = 0.5
        try:
            pks_auroc[i] = roc_auc_score(labels, pks_score)
        except ValueError:
            pks_auroc[i] = 0.5

    best_ecs_l = int(np.argmax(ecs_auroc))
    best_pks_l = int(np.argmax(pks_auroc))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(layers, ecs_auroc, color=ECS_COLOR, lw=LINEWIDTH, marker="o", ms=3,
            label=f"ECS  (best layer {best_ecs_l}: {ecs_auroc[best_ecs_l]:.3f})")
    ax.plot(layers, pks_auroc, color=PKS_COLOR, lw=LINEWIDTH, marker="s", ms=3,
            label=f"PKS  (best layer {best_pks_l}: {pks_auroc[best_pks_l]:.3f})")

    ax.axhline(0.5, color="grey", ls="--", lw=0.9, label="Chance (0.5)")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(
        f"Layer-wise AUROC — uncertain vs certain  (U > {hallu_thresh})",
        fontsize=11,
    )
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ReDeEP ECS/PKS plots")
    p.add_argument("--redeep-dir", required=True,
                   help="redeep_out directory containing sentence_scores.csv")
    p.add_argument("--hallu-thresh", type=float, default=0.5,
                   help="LUQ uncertainty threshold for uncertain label (default 0.5)")
    p.add_argument("--out", default=None,
                   help="Output directory for figures (default: same as --redeep-dir)")
    args = p.parse_args()

    redeep_dir = Path(args.redeep_dir)
    out_dir    = Path(args.out) if args.out else redeep_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = redeep_dir / "sentence_scores.csv"
    if not scores_csv.exists():
        raise FileNotFoundError(
            f"sentence_scores.csv not found in {redeep_dir}. "
            "Run redeep_sentence.py first."
        )

    print(f"Loading {scores_csv} …")
    df = pd.read_csv(scores_csv)
    n_ecs = sum(1 for c in df.columns if c.startswith("ecs_l"))
    print(f"  {len(df):,} sentences  |  {n_ecs} layers")
    print(f"  Uncertain (U > {args.hallu_thresh}): "
          f"{(df['luq_score'] > args.hallu_thresh).sum():,} "
          f"({100*(df['luq_score'] > args.hallu_thresh).mean():.1f}%)")

    plot_mean_ecs_pks(df, args.hallu_thresh, out_dir / "mean_ecs_pks.png")
    plot_auroc_layers(df, args.hallu_thresh, out_dir / "auroc_layers.png",
                      metrics_csv=redeep_dir / "layer_metrics.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
