"""
Visualisation utilities for all three experiments.
All functions save to RESULTS_DIR and return the output file path.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, PDSQI9_ATTRIBUTES, N_LAYERS

_OUT = Path(RESULTS_DIR)
_OUT.mkdir(parents=True, exist_ok=True)


# ── Experiment 1 — per-token mechanistic profile ─────────────────────────────

def plot_token_profile(
    token_df: pd.DataFrame,
    encounter_idx: int,
    out_dir: Path = _OUT,
) -> str:
    """
    Stacked area chart: attn vs MLP contribution across generated tokens,
    coloured by SOAP section.  Overlaid: lookback ratio (line).
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    x = token_df["token_idx"].values

    # Top: DLA band contributions
    ax = axes[0]
    ax.fill_between(x, 0, token_df["attn_contribution_late"],
                    alpha=0.7, label="Attn late",  color="#2196F3")
    ax.fill_between(x, 0, token_df["attn_contribution_mid"],
                    alpha=0.5, label="Attn mid",   color="#64B5F6")
    ax.fill_between(x, 0, token_df["mlp_contribution_mid"],
                    alpha=0.7, label="MLP mid",    color="#FF7043")
    ax.fill_between(x, 0, token_df["mlp_contribution_late"],
                    alpha=0.5, label="MLP late",   color="#FFAB91")
    ax.set_ylabel("Mean |DLA|")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Encounter {encounter_idx} — Per-token DLA Profile")

    # Bottom: lookback ratio + extractive score
    ax2 = axes[1]
    ax2.plot(x, token_df["lookback_ratio"],      label="Lookback ratio",    color="#4CAF50")
    ax2.plot(x, token_df["extractive_score"],    label="Extractive score",  color="#9C27B0", linestyle="--")
    ax2.plot(x, token_df["source_attention_entropy"] / (token_df["source_attention_entropy"].max() + 1e-8),
             label="Src attn entropy (norm)", color="#FF9800", linestyle=":")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score / Ratio")
    ax2.set_xlabel("Generated token index")
    ax2.legend(loc="upper right", fontsize=8)

    # Section background shading
    if "section" in token_df.columns:
        _shade_sections(axes[0], token_df)
        _shade_sections(axes[1], token_df)

    plt.tight_layout()
    path = out_dir / f"token_profile_enc{encounter_idx}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _shade_sections(ax, token_df: pd.DataFrame):
    palette = {
        "chief complaint":             "#FFECB3",
        "history of present illness":  "#E8F5E9",
        "review of systems":           "#E3F2FD",
        "physical examination":        "#FCE4EC",
        "assessment":                  "#EDE7F6",
        "plan":                        "#E0F7FA",
    }
    sections = token_df["section"].values
    prev_sec = sections[0]
    start    = 0
    for i, sec in enumerate(sections):
        if sec != prev_sec or i == len(sections) - 1:
            color = palette.get(prev_sec, "#F5F5F5")
            ax.axvspan(start, i, alpha=0.15, color=color, lw=0)
            prev_sec = sec
            start    = i


# ── Experiment 2 — Mechanistic Signature Matrix ───────────────────────────────

def plot_signature_matrix(
    corr_df:   pd.DataFrame,
    pval_df:   Optional[pd.DataFrame] = None,
    out_dir:   Path = _OUT,
    tag:       str  = "",
) -> str:
    """
    Heatmap of Pearson r between PDSQI-9 attributes (rows) and mechanistic
    features (columns).  Significant cells (p < 0.05) are annotated with *.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(corr_df.columns) * 0.9), 5))

    annot = corr_df.round(2).astype(str)
    if pval_df is not None:
        for r in corr_df.index:
            for c in corr_df.columns:
                if r in pval_df.index and c in pval_df.columns:
                    p = pval_df.loc[r, c]
                    if isinstance(p, float) and p < 0.05:
                        annot.loc[r, c] = annot.loc[r, c] + "*"

    sns.heatmap(
        corr_df.astype(float),
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(
        "Mechanistic Signature Matrix\n"
        "(* = p < 0.05; rows = PDSQI-9 attributes, cols = mechanistic features)"
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    fname = f"signature_matrix{'_' + tag if tag else ''}.png"
    path  = out_dir / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_section_anova(
    anova_df: pd.DataFrame,
    out_dir:  Path = _OUT,
) -> str:
    """Bar chart of ANOVA eta-squared per mechanistic feature."""
    fig, ax = plt.subplots(figsize=(10, 4))
    anova_df = anova_df.sort_values("eta_sq", ascending=True)
    colors = ["#FF7043" if p < 0.05 else "#90A4AE"
              for p in anova_df["p_value"]]
    ax.barh(anova_df.index, anova_df["eta_sq"], color=colors)
    ax.set_xlabel("η² (effect size)")
    ax.set_title("Section-type ANOVA: variance explained by SOAP section\n(red = p < 0.05)")
    plt.tight_layout()
    path = out_dir / "section_anova.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_complexity_scatter(
    encounter_features: List[Dict],
    pdsqi9_scores:      List[Dict],
    x_key:              str = "entity_density",
    y_key:              str = "mlp_contribution_mid",
    colour_by_attr:     str = "accurate",
    out_dir:            Path = _OUT,
) -> str:
    """Scatter: complexity feature vs mechanistic feature, coloured by PDSQI-9 score."""
    feat_df  = pd.DataFrame(encounter_features)
    score_df = pd.DataFrame(pdsqi9_scores)

    if x_key not in feat_df.columns or y_key not in feat_df.columns:
        return ""

    x = feat_df[x_key].values
    y = feat_df[y_key].values
    c = pd.to_numeric(score_df.get(colour_by_attr, pd.Series()), errors="coerce").values

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y, c=c, cmap="RdYlGn", vmin=1, vmax=5, s=60, edgecolors="k", lw=0.5)
    plt.colorbar(sc, ax=ax, label=f"PDSQI-9: {colour_by_attr}")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{x_key} vs {y_key}\n(colour = {colour_by_attr} score)")
    plt.tight_layout()
    path = out_dir / f"complexity_scatter_{x_key}_{y_key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# ── Experiment 3 — Patching sweep plots ───────────────────────────────────────

def plot_patching_sweep(
    candidate,
    out_dir: Path = _OUT,
) -> str:
    """
    Three-line plot of normalised recovery across layers for resid/attn/mlp.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    layers = list(range(N_LAYERS))

    if candidate.resid_scores:
        ax.plot(layers, candidate.resid_scores, label="resid_pre", color="#607D8B", lw=1.5)
    if candidate.attn_scores:
        ax.plot(layers, candidate.attn_scores,  label="attn_out",  color="#2196F3", lw=2.0)
    if candidate.mlp_scores:
        ax.plot(layers, candidate.mlp_scores,   label="mlp_out",   color="#FF7043", lw=2.0)

    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.axhline(1, color="green", lw=0.8, linestyle="--", label="Full recovery")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalised recovery")
    ax.set_title(
        f"Activation Patching — Encounter {candidate.encounter_idx} "
        f"(Cat {candidate.category})\n"
        f"Fact: {candidate.target_fact[:60]}"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, N_LAYERS - 1)
    plt.tight_layout()

    path = out_dir / f"patching_enc{candidate.encounter_idx}_cat{candidate.category}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_blind_spot_summary(
    report_rows: List[Dict],
    out_dir: Path = _OUT,
) -> str:
    """
    Grouped bar chart: max_attn_restore vs max_mlp_restore per candidate,
    grouped by category (A/B/C).
    """
    df = pd.DataFrame(report_rows)
    if df.empty:
        return ""

    cat_order = ["A", "B", "C"]
    fig, axes = plt.subplots(1, len(cat_order), figsize=(14, 5), sharey=True)

    for ax, cat in zip(axes, cat_order):
        sub = df[df["category"] == cat].reset_index(drop=True)
        if sub.empty:
            ax.set_title(f"Category {cat}\n(no candidates)")
            continue
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["max_attn_restore"].fillna(0), 0.35,
               label="Max attn restore", color="#2196F3", alpha=0.8)
        ax.bar(x + 0.2, sub["max_mlp_restore"].fillna(0),  0.35,
               label="Max MLP restore",  color="#FF7043", alpha=0.8)
        ax.axhline(0.3, color="red", lw=1, linestyle="--", label="Threshold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"enc{r}" for r in sub["encounter_idx"]], rotation=45, fontsize=8)
        ax.set_title(f"Category {cat}")
        if cat == "A":
            ax.set_ylabel("Normalised recovery")
        ax.legend(fontsize=7)

    plt.suptitle(
        "Blind Spot Detection: Patching Restoration by Category\n"
        "A=Coincidental Correct, B=Penalised Inference, C=Undetected Fragility",
        fontsize=11,
    )
    plt.tight_layout()
    path = out_dir / "blind_spot_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
