"""
Entity DLA Analysis — visualise results from exp1/entity_dla.json.

Produces 5 figures saved to results/entity_dla_analysis/:

  1. attn_mlp_by_entity_type.png  — violin: attn/MLP fraction split by entity type
  2. head_importance_heatmap.png  — [n_layers × n_heads] aggregated |contribution|
  3. top_mlp_layers.png           — bar chart of most important MLP layers
  4. copy_heads.png               — scatter: head contribution vs source attention,
                                    copy-head candidates highlighted
  5. source_attn_distribution.png — per-head source attention score distribution
                                    for top contributing heads

Usage:
    python analyse_entity_dla.py
    python analyse_entity_dla.py --input results/exp1/entity_dla.json --out results/entity_dla_analysis
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

N_LAYERS = 42
N_HEADS  = 16

# Contribution threshold to be considered a "copy head candidate"
COPY_HEAD_CONTRIB_THRESHOLD  = 0.3   # mean |contribution| across appearances
COPY_HEAD_SOURCE_ATTN_THRESHOLD = 0.05  # mean source attention fraction


# ── Load ───────────────────────────────────────────────────────────────────────

def load(path: Path):
    with open(path) as f:
        return json.load(f)


# ── Flatten to DataFrames ──────────────────────────────────────────────────────

def build_entity_df(data: list) -> pd.DataFrame:
    """One row per entity with scalar features."""
    rows = []
    for enc in data:
        enc_idx = enc["encounter_idx"]
        for ent in enc["entities"]:
            rows.append({
                "encounter_idx":            enc_idx,
                "entity_text":              ent["entity_text"],
                "entity_type":              ent["entity_type"],
                "n_output_positions":       len(ent["output_positions"]),
                "attn_fraction":            ent["attn_fraction"],
                "mlp_fraction":             ent["mlp_fraction"],
                "top_heads_attend_to_source": ent["top_heads_attend_to_source"],
                "mean_source_attn":         np.mean(list(ent["source_attention_scores"].values()))
                                            if ent["source_attention_scores"] else 0.0,
            })
    return pd.DataFrame(rows)


def build_head_matrix(data: list) -> np.ndarray:
    """Aggregate |contribution| into [n_layers, n_heads] across all entities."""
    mat = np.zeros((N_LAYERS, N_HEADS), dtype=np.float32)
    counts = np.zeros((N_LAYERS, N_HEADS), dtype=np.int32)
    for enc in data:
        for ent in enc["entities"]:
            for h in ent["top_heads"]:
                mat[h["layer"], h["head"]] += abs(h["contribution"])
                counts[h["layer"], h["head"]] += 1
    # Mean |contribution| per (layer, head) — zero where never in top-k
    with np.errstate(invalid="ignore"):
        mat = np.where(counts > 0, mat / counts, 0.0)
    return mat


def build_mlp_series(data: list) -> pd.Series:
    """Mean |contribution| per MLP layer across all entities."""
    totals = defaultdict(float)
    counts = defaultdict(int)
    for enc in data:
        for ent in enc["entities"]:
            for m in ent["top_mlp_layers"]:
                totals[m["layer"]] += abs(m["contribution"])
                counts[m["layer"]] += 1
    layers = sorted(totals)
    vals   = [totals[l] / counts[l] for l in layers]
    return pd.Series(vals, index=layers, name="mean_abs_contribution")


def build_head_scatter_df(data: list) -> pd.DataFrame:
    """One row per (entity, head) for the copy-head scatter."""
    rows = []
    for enc in data:
        for ent in enc["entities"]:
            sa = ent["source_attention_scores"]
            for h in ent["top_heads"]:
                key = f"L{h['layer']}H{h['head']}"
                rows.append({
                    "layer":           h["layer"],
                    "head":            h["head"],
                    "contribution":    abs(h["contribution"]),
                    "source_attn":     sa.get(key, 0.0),
                    "entity_type":     ent["entity_type"],
                })
    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_attn_mlp_by_type(entity_df: pd.DataFrame, out_dir: Path):
    types = entity_df["entity_type"].value_counts()
    types = types[types >= 3].index.tolist()   # skip rare types
    df = entity_df[entity_df["entity_type"].isin(types)].copy()

    melted = df.melt(
        id_vars="entity_type",
        value_vars=["attn_fraction", "mlp_fraction"],
        var_name="component", value_name="fraction",
    )
    melted["component"] = melted["component"].map(
        {"attn_fraction": "Attention", "mlp_fraction": "MLP"}
    )

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 1.4), 5))
    sns.violinplot(
        data=melted, x="entity_type", y="fraction", hue="component",
        split=True, inner="box", palette={"Attention": "#4C9BE8", "MLP": "#E8834C"},
        ax=ax,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Fraction of Total Logit Contribution")
    ax.set_title("Attention vs MLP Contribution by Clinical Entity Type")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = out_dir / "attn_mlp_by_entity_type.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_head_heatmap(mat: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean |Contribution| (across entity appearances)")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Attention Head Importance for Clinical Entity Generation\n(aggregated over all entities & encounters)")
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(0, N_LAYERS, 5))
    ax.set_yticklabels(range(0, N_LAYERS, 5))
    # Annotate top 5 heads
    flat = mat.flatten()
    top5 = np.argpartition(flat, -5)[-5:]
    for idx in top5:
        l, h = divmod(idx, N_HEADS)
        ax.add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1,
                                   fill=False, edgecolor="blue", linewidth=1.5))
    plt.tight_layout()
    path = out_dir / "head_importance_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_top_mlp_layers(mlp_series: pd.Series, out_dir: Path, top_n: int = 20):
    top = mlp_series.nlargest(top_n).sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#E8834C" if v == top.max() else "#F5C48A" for v in top.values]
    ax.bar(top.index.astype(str), top.values, color=colors, edgecolor="white")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |MLP Contribution|")
    ax.set_title(f"Top {top_n} MLP Layers by Mean Absolute Contribution to Entity Generation")
    plt.tight_layout()
    path = out_dir / "top_mlp_layers.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_copy_heads(scatter_df: pd.DataFrame, out_dir: Path):
    # Aggregate to per-(layer, head) means
    agg = (
        scatter_df.groupby(["layer", "head"])
        .agg(mean_contrib=("contribution", "mean"),
             mean_src_attn=("source_attn",  "mean"),
             n=("contribution", "count"))
        .reset_index()
    )

    copy_mask = (
        (agg["mean_contrib"]  >= COPY_HEAD_CONTRIB_THRESHOLD) &
        (agg["mean_src_attn"] >= COPY_HEAD_SOURCE_ATTN_THRESHOLD)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(agg.loc[~copy_mask, "mean_contrib"],
               agg.loc[~copy_mask, "mean_src_attn"],
               s=agg.loc[~copy_mask, "n"] * 4, alpha=0.4, color="#999999", label="Other heads")
    ax.scatter(agg.loc[copy_mask, "mean_contrib"],
               agg.loc[copy_mask, "mean_src_attn"],
               s=agg.loc[copy_mask, "n"] * 6, alpha=0.85, color="#E84C4C", label="Copy-head candidates")

    for _, row in agg[copy_mask].iterrows():
        ax.annotate(f"L{int(row['layer'])}H{int(row['head'])}",
                    (row["mean_contrib"], row["mean_src_attn"]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4))

    ax.axvline(COPY_HEAD_CONTRIB_THRESHOLD,  color="red",  linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(COPY_HEAD_SOURCE_ATTN_THRESHOLD, color="blue", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Mean |Contribution| to Entity Logits")
    ax.set_ylabel("Mean Source Attention Fraction")
    ax.set_title("Copy-Head Identification\n(high contribution + attends to source dialogue)")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "copy_heads.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # Print copy-head summary
    if copy_mask.any():
        print("\n  Copy-head candidates:")
        for _, row in agg[copy_mask].sort_values("mean_contrib", ascending=False).iterrows():
            print(f"    L{int(row['layer']):02d}H{int(row['head']):02d}  "
                  f"contrib={row['mean_contrib']:.3f}  src_attn={row['mean_src_attn']:.3f}  n={int(row['n'])}")
    else:
        print("  No heads met both thresholds — consider lowering COPY_HEAD_CONTRIB_THRESHOLD")


def plot_source_attn_distribution(scatter_df: pd.DataFrame, out_dir: Path, top_n_heads: int = 10):
    # Pick the top-n heads by mean |contribution|
    top_heads = (
        scatter_df.groupby(["layer", "head"])["contribution"]
        .mean()
        .nlargest(top_n_heads)
        .index
    )
    labels = [f"L{l}H{h}" for l, h in top_heads]
    df_top = scatter_df[
        scatter_df.apply(lambda r: (r["layer"], r["head"]) in top_heads, axis=1)
    ].copy()
    df_top["head_label"] = df_top.apply(lambda r: f"L{int(r['layer'])}H{int(r['head'])}", axis=1)
    df_top["head_label"] = pd.Categorical(df_top["head_label"], categories=labels, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df_top, x="head_label", y="source_attn",
                order=labels, color="#4C9BE8", ax=ax)
    ax.axhline(COPY_HEAD_SOURCE_ATTN_THRESHOLD, color="red", linestyle="--",
               linewidth=0.9, label=f"Threshold ({COPY_HEAD_SOURCE_ATTN_THRESHOLD})")
    ax.set_xlabel("Head (ordered by mean |contribution|)")
    ax.set_ylabel("Source Attention Fraction")
    ax.set_title(f"Source Attention Distribution for Top-{top_n_heads} Contributing Heads")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "source_attn_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(entity_df: pd.DataFrame, mat: np.ndarray, mlp_series: pd.Series):
    print(f"\n{'─'*60}")
    print(f"  Total entities analysed : {len(entity_df)}")
    print(f"  Encounters              : {entity_df['encounter_idx'].nunique()}")
    print(f"  Entity types            : {sorted(entity_df['entity_type'].unique())}")
    print(f"\n  Mean attn_fraction      : {entity_df['attn_fraction'].mean():.3f}")
    print(f"  Mean mlp_fraction       : {entity_df['mlp_fraction'].mean():.3f}")
    print(f"  Entities w/ copy head   : "
          f"{entity_df['top_heads_attend_to_source'].sum()} / {len(entity_df)}")

    print(f"\n  Top-5 attention heads (mean |contribution|):")
    flat = mat.flatten()
    top5 = np.argpartition(flat, -5)[-5:]
    for idx in sorted(top5, key=lambda i: flat[i], reverse=True):
        l, h = divmod(idx, N_HEADS)
        print(f"    L{l:02d}H{h:02d}  {flat[idx]:.4f}")

    print(f"\n  Top-5 MLP layers:")
    for layer, val in mlp_series.nlargest(5).items():
        print(f"    Layer {layer:02d}  {val:.4f}")
    print(f"{'─'*60}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default="results/exp1/entity_dla.json")
    p.add_argument("--out",     default="results/entity_dla_analysis")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    data = load(Path(args.input))

    entity_df  = build_entity_df(data)
    head_mat   = build_head_matrix(data)
    mlp_series = build_mlp_series(data)
    scatter_df = build_head_scatter_df(data)

    print_summary(entity_df, head_mat, mlp_series)

    print("Generating plots ...")
    plot_attn_mlp_by_type(entity_df, out_dir)
    plot_head_heatmap(head_mat, out_dir)
    plot_top_mlp_layers(mlp_series, out_dir)
    plot_copy_heads(scatter_df, out_dir)
    plot_source_attn_distribution(scatter_df, out_dir)

    # Save flat entity table for further inspection
    csv_path = out_dir / "entity_dla_flat.csv"
    entity_df.to_csv(csv_path, index=False)
    print(f"\n  Flat entity table → {csv_path}")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
