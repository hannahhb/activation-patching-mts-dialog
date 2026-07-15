"""
redeep_word_plots.py
====================
Word-level ReDeEP discrimination plots: per-layer AUROC and Cohen's d for
word-level ECS / PKS against hallucinated words.

Inputs (no model / GPU needed — reads precomputed activations):
  activations/sample_NNN_gen_KK_tokens.npz   (from redeep_sentence.py)
      ecs_word       (n_layers, n_words)   all-heads word ECS
      ecs_word_copy  (n_layers, n_words)   copying-head word ECS
      pks_word       (n_layers, n_words)   per-token-mean word PKS (mean of
                                            token-level JSD within the word,
                                            NOT the paper's chunk-level SUM --
                                            SUM confounds with token count at
                                            word granularity, see redeep_sentence.py)
      word_spans     (n_words, 2)          note-token span per word
      word_strs      (n_words,)            decoded word text
  spans/sample_NNN_note_KK_span_judge.csv  (from llm_judge.py --mode span)
      sentence_idx, sentence, label, note_span

Word labels:
  The span judge marks each non-Faithful sentence with the verbatim erroneous
  `note_span`.  We reconstruct the note text from `word_strs`, locate every
  note_span in it (whitespace-tolerant), and label a word HALLUCINATED (1) iff
  its character range overlaps any matched span; every other content word is
  FAITHFUL (0).  Sentence indices are NOT used — matching is purely by text, so
  the judge's own sentence segmentation need not agree with anything.

Estimator:
  Words are POOLED across all (sample, gen) pairs, giving one AUROC / Cohen's d
  per layer over the full word population.  Word granularity yields hundreds of
  units per note, so the pooled per-layer estimate is stable even with few
  samples (unlike sentence-level, where per-gen AUROC is very noisy).

Usage:
    python redeep_word_plots.py
    python redeep_word_plots.py --act-dir sae_experiments/redeep_out/activations \\
                                --span-dir sae_experiments/luq_out/llama_judge/spans \\
                                --out sae_experiments/redeep_out/word_plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from hallucination_content_type import classify_content_type
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ACT_DIR  = "sae_experiments/redeep_out/activations"
DEFAULT_SPAN_DIR = "sae_experiments/luq_out/llama_judge/spans"
DEFAULT_OUT_DIR  = "sae_experiments/redeep_out/word_plots"

# English function words to drop for the "content words only" AUROC view.
# Small hand-rolled list (no nltk dependency, keeps this script self-contained) --
# articles, pronouns, auxiliaries/copulas, prepositions, conjunctions, and a few
# common discourse fillers. Numbers, punctuation-attached tokens (e.g. "98.6,")
# are NOT stripped as noise -- only stopwords are dropped; a token surviving
# punctuation-stripping is kept unless it IS a stopword.
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "if", "then",
    "than", "as", "that", "this", "these", "those", "there", "here",
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "who", "whom", "whose", "which", "what",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "once", "not", "no", "own", "same", "just", "also",
})


def is_content_word(word: str) -> bool:
    """True iff `word` (raw token text, may carry leading whitespace / attached
    punctuation, e.g. ' presented', ' healthy.') is a content word -- i.e. it
    has an alphanumeric core and that core is not an English stopword."""
    core = word.strip().strip(".,;:!?()[]{}\"'`~-").lower()
    if not core or not any(c.isalnum() for c in core):
        return False
    return core not in STOPWORDS


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (self-contained copies of redeep_sentence.py's, so this script
# has no torch / transformer_lens dependency)
# ─────────────────────────────────────────────────────────────────────────────

def auroc_per_layer_single(score_l: np.ndarray, labels: np.ndarray,
                           hallu_high: bool) -> np.ndarray:
    """AUROC per layer. hallu_high=True ⇒ higher score means more hallucinated
    (PKS); False ⇒ lower means more hallucinated (ECS). NaN where undefined."""
    from sklearn.metrics import roc_auc_score
    n_layers = score_l.shape[0]
    out = np.full(n_layers, np.nan)
    y = labels.astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return out
    sgn = 1.0 if hallu_high else -1.0
    for l in range(n_layers):
        s = score_l[l]
        finite = np.isfinite(s)
        if finite.sum() < 2:
            continue
        yy, ss = y[finite], s[finite]
        if yy.sum() == 0 or (1 - yy).sum() == 0:
            continue
        try:
            out[l] = float(roc_auc_score(yy, sgn * ss))
        except Exception:
            pass
    return out


def cohens_d_per_layer_single(score_l: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cohen's d per layer: (mean_hallu − mean_faithful) / pooled_std."""
    n_layers = score_l.shape[0]
    out = np.full(n_layers, np.nan)
    y_all = labels.astype(bool)
    for l in range(n_layers):
        s = score_l[l]
        finite = np.isfinite(s)
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


def _mean_std_per_layer(score_l: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-layer (mean, std) of score_l[:, mask], ignoring NaNs."""
    sub = score_l[:, mask]
    return np.nanmean(sub, axis=1), np.nanstd(sub, axis=1)


def _plot_mean_std_by_layer(metric_all, lab_all, out_path, ylabel,
                            show_title=True, title=None, fontsize=12):
    """Standalone mean±std-by-layer line plot, Hallucinated vs Faithful
    (equivalent to panel 2 of _plot_cohens_d_with_metric_bar, on its own)."""
    n_layers = metric_all.shape[0]
    layers   = np.arange(n_layers)
    hallu_mask = lab_all.astype(bool)
    hallu_mean, hallu_std = _mean_std_per_layer(metric_all, hallu_mask)
    faith_mean, faith_std = _mean_std_per_layer(metric_all, ~hallu_mask)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, hallu_mean, color="tomato", lw=2, marker="s", ms=3, label="Hallucinated")
    ax.fill_between(layers, hallu_mean - hallu_std, hallu_mean + hallu_std, color="tomato", alpha=0.15)
    ax.plot(layers, faith_mean, color="grey", lw=2, marker="o", ms=3, label="Faithful")
    ax.fill_between(layers, faith_mean - faith_std, faith_mean + faith_std, color="grey", alpha=0.15)
    ax.set_xlabel("Layer index", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if show_title and title:
        ax.set_title(title)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.legend(fontsize=fontsize - 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def _plot_cohens_d_with_metric_bar(metric_d, metric_all, lab_all,
                                   out_path, title_suffix, metric_name="PKS",
                                   metric_color="tomato", granularity="Word-level",
                                   show_title=True, only_cohens_d=False, fontsize=12):
    """Figure showing:
       1) metric-only Cohen's d as a uni-color bar chart (ReDeEP-paper style,
          standardized: mean difference / pooled std)
       2) actual (raw, unstandardized) metric scores per layer, mean±std,
          hallucinated vs faithful units  [skipped if only_cohens_d]
       3) raw (unstandardized) mean difference per layer, hallucinated −
          faithful, as a bar chart -- same units as panel 2, no std scaling
          [skipped if only_cohens_d]
    """
    n_layers = len(metric_d)
    layers   = np.arange(n_layers)

    if only_cohens_d:
        fig, ax1 = plt.subplots(figsize=(10, 5))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                                            gridspec_kw={"height_ratios": [1, 1, 1]})

    ax1.bar(layers, metric_d, color=metric_color, width=0.8)
    ax1.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax1.set_ylabel(f"{metric_name} Cohen's d", fontsize=fontsize)
    if show_title:
        ax1.set_title(f"{granularity} {metric_name} Cohen's d by layer\n{title_suffix}")
    if only_cohens_d:
        ax1.set_xlabel("Layer index", fontsize=fontsize)
    ax1.tick_params(axis="both", labelsize=fontsize - 1)
    ax1.grid(alpha=0.3, axis="y")

    if not only_cohens_d:
        hallu_mask = lab_all.astype(bool)
        metric_hallu_mean, metric_hallu_std = _mean_std_per_layer(metric_all, hallu_mask)
        metric_faith_mean, metric_faith_std = _mean_std_per_layer(metric_all, ~hallu_mask)
        ax2.plot(layers, metric_hallu_mean, color="tomato", lw=2, marker="s", ms=3, label="Hallucinated")
        ax2.fill_between(layers, metric_hallu_mean - metric_hallu_std, metric_hallu_mean + metric_hallu_std,
                         color="tomato", alpha=0.15)
        ax2.plot(layers, metric_faith_mean, color="grey", lw=2, marker="o", ms=3, label="Faithful")
        ax2.fill_between(layers, metric_faith_mean - metric_faith_std, metric_faith_mean + metric_faith_std,
                         color="grey", alpha=0.15)
        ax2.set_ylabel(metric_name, fontsize=fontsize)
        if show_title:
            ax2.set_title(f"{granularity} actual {metric_name} scores by layer (mean ± std)")
        ax2.tick_params(axis="both", labelsize=fontsize - 1)
        ax2.legend(fontsize=fontsize - 1)
        ax2.grid(alpha=0.3)

        diff = metric_hallu_mean - metric_faith_mean
        avg_diff = np.nanmean(diff)
        ax3.bar(layers, diff, color=metric_color, width=0.8)
        ax3.axhline(0.0, color="grey", lw=0.8)
        ax3.axhline(avg_diff, color="red", ls="--", lw=1.2, label=f"Average: {avg_diff:.3f}")
        ax3.set_xlabel("Layer index", fontsize=fontsize)
        ax3.set_ylabel(f"{metric_name} mean difference\n(Hallucinated − Faithful)", fontsize=fontsize)
        if show_title:
            ax3.set_title(f"{granularity} raw {metric_name} mean difference by layer (unstandardized)")
        ax3.tick_params(axis="both", labelsize=fontsize - 1)
        ax3.legend(fontsize=fontsize - 1)
        ax3.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def _plot_pks_paper_style(pks_all: np.ndarray, lab_all: np.ndarray, out_path,
                          bar_color="steelblue"):
    """Reproduce the ReDeEP-paper PKS figure style (Fig. 4 d/e):
       (left)  raw per-layer mean difference, hallucinated − faithful
               (NOT standardized by std, unlike Cohen's d), with a dashed
               line at the average difference across layers.
       (right) per-layer Pearson correlation between the raw PKS score and
               the binary hallucination label (point-biserial correlation).
    """
    n_layers = pks_all.shape[0]
    layers   = np.arange(n_layers)
    hallu_mask = lab_all.astype(bool)
    y = lab_all.astype(float)

    diffs     = np.full(n_layers, np.nan)
    pearson_r = np.full(n_layers, np.nan)
    for l in range(n_layers):
        s = pks_all[l]
        finite = np.isfinite(s)
        if finite.sum() < 2:
            continue
        s_f, y_f, h_f = s[finite], y[finite], hallu_mask[finite]
        if h_f.sum() >= 1 and (~h_f).sum() >= 1:
            diffs[l] = s_f[h_f].mean() - s_f[~h_f].mean()
        if np.std(s_f) > 0 and np.std(y_f) > 0:
            pearson_r[l] = np.corrcoef(s_f, y_f)[0, 1]

    avg_diff = np.nanmean(diffs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(layers, diffs, color=bar_color, width=0.8)
    ax1.axhline(avg_diff, color="red", ls="--", lw=1.2, label=f"Average: {avg_diff:.3f}")
    ax1.axhline(0.0, color="grey", lw=0.8)
    ax1.set_xlabel("Layers")
    ax1.set_ylabel("Difference in Parametric Knowledge Scores")
    ax1.set_title("Difference in Parametric Knowledge Scores\n(Hallucination − Truth), per-token mean")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(layers, pearson_r, color=bar_color, width=0.8)
    ax2.axhline(0.0, color="grey", lw=0.8)
    ax2.set_xlabel("Layers")
    ax2.set_ylabel("Pearson Correlation")
    ax2.set_title("Pearson's r: Parametric Knowledge Score\nvs. Hallucination Label, per-token mean")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def _plot_layer_lines(series, out_path, ylabel, title, hline, ylim=None,
                      show_title=True, fontsize=12):
    """series: list of (avg, std|None, color, marker, label). (n_layers,) each."""
    n_layers = len(series[0][0])
    layers   = np.arange(n_layers)
    fig, ax  = plt.subplots(figsize=(10, 5))
    for avg, std, color, marker, label in series:
        ax.plot(layers, avg, color=color, lw=2, marker=marker, ms=3, label=label)
        if std is not None:
            ax.fill_between(layers, avg - std, avg + std, color=color, alpha=0.15)
    if hline is not None:
        ax.axhline(hline, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Layer index", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if show_title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.legend(fontsize=fontsize - 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def _plot_auroc_by_group(group_all: np.ndarray, ecs_all: np.ndarray, ecscopy_all: np.ndarray,
                         pks_all: np.ndarray, n_layers: int, fig_path, csv_path, suptitle: str,
                         faithful_label: str = "Faithful", show_suptitle: bool = True,
                         include_raw_ecs: bool = True, fontsize: int = 12) -> None:
    """Grid of per-layer AUROC subplots, one per non-faithful value of
    `group_all` (a str label per word), each vs `faithful_label` words.
    `ecs_all`/`ecscopy_all`/`pks_all` must already be subset to match
    `group_all` (e.g. content-words-only)."""
    groups_present = sorted(g for g in np.unique(group_all) if g and g != faithful_label)
    if not groups_present:
        print(f"  No non-'{faithful_label}' groups found — skipping {fig_path.name}.")
        return

    faithful_mask = (group_all == faithful_label)
    n_cols = 2
    n_rows = int(np.ceil(len(groups_present) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows),
                             sharey=True, squeeze=False)
    layers = np.arange(n_layers)
    rows = []
    for idx, group in enumerate(groups_present):
        ax = axes[idx // n_cols][idx % n_cols]
        mask = (group_all == group) | faithful_mask
        y = (group_all[mask] == group).astype(int)
        n_group, n_faith_cmp = int(y.sum()), int((1 - y).sum())
        if n_group < 2 or n_faith_cmp < 2:
            ax.text(0.5, 0.5, f"{group}\n(n={n_group}, insufficient for AUROC)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ecs_auroc     = auroc_per_layer_single(ecs_all[:, mask],     y, hallu_high=True)
        ecscopy_auroc = auroc_per_layer_single(ecscopy_all[:, mask], y, hallu_high=True)
        pks_auroc     = auroc_per_layer_single(pks_all[:, mask],     y, hallu_high=True)
        if include_raw_ecs:
            ax.plot(layers, ecs_auroc, color="steelblue", lw=2, marker="o", ms=3, label="ECS (all heads)")
        ax.plot(layers, ecscopy_auroc, color="seagreen",  lw=2, marker="^", ms=3, label="ECS (copying heads)")
        ax.plot(layers, pks_auroc,     color="tomato",    lw=2, marker="s", ms=3, label="PKS (per-token mean)")
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.set_title(f"{group}  (n={n_group} vs {n_faith_cmp} {faithful_label.lower()})", fontsize=fontsize)
        ax.set_xlabel("Layer index", fontsize=fontsize)
        ax.set_ylabel("AUROC", fontsize=fontsize)
        ax.set_ylim(0.3, 1.0)
        ax.tick_params(axis="both", labelsize=fontsize - 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=fontsize - 2)
        for layer_i in range(n_layers):
            rows.append({
                "group":          group,
                "layer":          layer_i,
                "ecs_auroc":      ecs_auroc[layer_i],
                "ecs_copy_auroc": ecscopy_auroc[layer_i],
                "pks_auroc":      pks_auroc[layer_i],
                "n_group":        n_group,
                "n_faithful_cmp": n_faith_cmp,
            })
    for j in range(len(groups_present), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    if show_suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")

    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")


def _plot_ecs_vs_pks_scatter(ecs_all: np.ndarray, pks_all: np.ndarray, lab_all: np.ndarray,
                             layers: List[int], out_path, n_total: int,
                             ecs_label: str = "ECS (copying heads)",
                             pks_label: str = "PKS (per-token mean)",
                             unit: str = "word",
                             max_faithful_points: int = 8000, seed: int = 0) -> None:
    """Grid of ECS-vs-PKS scatter plots, one per layer in `layers`, points =
    words/tokens (per `unit`) colored by hallucinated (red) vs faithful (grey).
    Faithful points are randomly subsampled to `max_faithful_points` for
    legibility/file size (hallucinated points are shown in full, they're
    always the minority)."""
    n_cols = min(2, len(layers))
    n_rows = int(np.ceil(len(layers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.5 * n_rows), squeeze=False)
    rng = np.random.default_rng(seed)

    hallu_mask = lab_all.astype(bool)
    faith_idx_all = np.where(~hallu_mask)[0]
    if faith_idx_all.size > max_faithful_points:
        faith_idx = rng.choice(faith_idx_all, size=max_faithful_points, replace=False)
    else:
        faith_idx = faith_idx_all
    hallu_idx = np.where(hallu_mask)[0]

    for idx, layer in enumerate(layers):
        ax = axes[idx // n_cols][idx % n_cols]
        fin_faith = np.isfinite(ecs_all[layer, faith_idx]) & np.isfinite(pks_all[layer, faith_idx])
        fin_hallu = np.isfinite(ecs_all[layer, hallu_idx]) & np.isfinite(pks_all[layer, hallu_idx])
        if not fin_faith.any() and not fin_hallu.any():
            ax.text(0.5, 0.5, f"Layer {layer}\n(no finite ECS/PKS values --\n"
                    f"likely no copying heads at this layer)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.scatter(ecs_all[layer, faith_idx][fin_faith], pks_all[layer, faith_idx][fin_faith],
                   s=6, color="grey", alpha=0.25, linewidths=0, label="Faithful")
        ax.scatter(ecs_all[layer, hallu_idx][fin_hallu], pks_all[layer, hallu_idx][fin_hallu],
                   s=8, color="tomato", alpha=0.45, linewidths=0, label="Hallucinated")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel(ecs_label)
        ax.set_ylabel(pks_label)
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.3)
    for j in range(len(layers), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"ECS vs PKS by {unit}, content words only\n"
                 f"(pooled: {n_total} content {unit}s, {int(hallu_mask.sum())} hallucinated; "
                 f"faithful subsampled to {len(faith_idx)} points for display)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_metric_strip_all_layers(metric_all: np.ndarray, lab_all: np.ndarray,
                                  out_path, metric_name: str, n_total: int,
                                  color: str = "seagreen",
                                  max_faithful: int = 3000, seed: int = 0) -> None:
    """Single-metric jittered strip plot across ALL layers at once (x=layer,
    y=metric value, points jittered horizontally within each layer). Same
    fixed random subsample of faithful words is used at every layer (so
    you're tracking the same words' values across depth); all hallucinated
    words are shown. This is the single-metric analog of
    _plot_ecs_vs_pks_scatter, generalized to all layers like the percentile
    heatmap rather than 4 hand-picked panels."""
    n_layers = metric_all.shape[0]
    hallu_mask = lab_all.astype(bool)
    rng = np.random.default_rng(seed)

    faith_idx_all = np.where(~hallu_mask)[0]
    faith_idx = (rng.choice(faith_idx_all, size=max_faithful, replace=False)
                 if faith_idx_all.size > max_faithful else faith_idx_all)
    hallu_idx = np.where(hallu_mask)[0]

    fig, ax = plt.subplots(figsize=(0.35 * n_layers + 3, 6))
    jitter_f = rng.uniform(-0.35, 0.35, size=faith_idx.size)
    jitter_h = rng.uniform(-0.35, 0.35, size=hallu_idx.size)

    for l in range(n_layers):
        yf = metric_all[l, faith_idx]
        yh = metric_all[l, hallu_idx]
        finite_f = np.isfinite(yf)
        finite_h = np.isfinite(yh)
        ax.scatter(l + jitter_f[finite_f], yf[finite_f], s=4, color="grey", alpha=0.2, linewidths=0)
        ax.scatter(l + jitter_h[finite_h], yh[finite_h], s=6, color=color, alpha=0.35, linewidths=0)

    ax.scatter([], [], s=12, color="grey", label="Faithful")
    ax.scatter([], [], s=12, color=color, label="Hallucinated")
    ax.axhline(0.0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(metric_name)
    ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 16)))
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.3)
    ax.set_title(f"{metric_name} by layer, hallucinated vs faithful words\n"
                 f"(pooled: {n_total} content words, {int(hallu_mask.sum())} hallucinated; "
                 f"faithful subsampled to {len(faith_idx)} points, same words shown at every layer)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_ecs_vs_pks_kde(ecs_all: np.ndarray, pks_all: np.ndarray, lab_all: np.ndarray,
                         out_path, n_total: int,
                         ecs_label: str = "ECS (copying heads)",
                         pks_label: str = "PKS (per-token mean)",
                         max_faithful: int = 50000, grid_size: int = 100, seed: int = 0) -> None:
    """2D kernel density estimate, ECS vs PKS, hallucinated vs faithful --
    ALL layers pooled into one plot (every (word, layer) pair is one sample,
    not one panel per layer). All finite hallucinated (word, layer) pairs are
    used (no subsampling); faithful is subsampled to `max_faithful` purely for
    KDE tractability -- with ~1.2M faithful (word, layer) pairs, fitting a KDE
    on all of them would be prohibitively slow, and the density estimate
    doesn't need every point to converge given how numerous faithful words
    already are."""
    from scipy.stats import gaussian_kde

    n_layers = ecs_all.shape[0]
    # Pool all layers: flatten (n_layers, n_words) -> (n_layers * n_words,).
    # Row-major flatten puts layer 0's words first, then layer 1's, etc., so
    # tiling the per-word label array n_layers times lines up with it exactly.
    ecs_flat = ecs_all.reshape(-1)
    pks_flat = pks_all.reshape(-1)
    lab_flat = np.tile(lab_all, n_layers)

    finite = np.isfinite(ecs_flat) & np.isfinite(pks_flat)
    ecs_flat, pks_flat, lab_flat = ecs_flat[finite], pks_flat[finite], lab_flat[finite]
    hallu_mask = lab_flat.astype(bool)

    rng = np.random.default_rng(seed)
    hallu_idx = np.where(hallu_mask)[0]                    # ALL hallucinated (word, layer) pairs, no subsampling
    faith_idx_all = np.where(~hallu_mask)[0]
    faith_idx = (rng.choice(faith_idx_all, size=max_faithful, replace=False)
                 if faith_idx_all.size > max_faithful else faith_idx_all)

    x_all = np.concatenate([ecs_flat[hallu_idx], ecs_flat[faith_idx]])
    y_all = np.concatenate([pks_flat[hallu_idx], pks_flat[faith_idx]])
    xmin, xmax = np.percentile(x_all, [0.5, 99.5])
    ymin, ymax = np.percentile(y_all, [0.5, 99.5])
    xx, yy = np.mgrid[xmin:xmax:complex(grid_size), ymin:ymax:complex(grid_size)]
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=(9, 7))

    kde_faith = gaussian_kde(np.vstack([ecs_flat[faith_idx], pks_flat[faith_idx]]))
    zf = kde_faith(grid_pts).reshape(xx.shape)
    ax.contour(xx, yy, zf, levels=8, colors="grey", linewidths=1.2)

    kde_hallu = gaussian_kde(np.vstack([ecs_flat[hallu_idx], pks_flat[hallu_idx]]))
    zh = kde_hallu(grid_pts).reshape(xx.shape)
    ax.contour(xx, yy, zh, levels=8, colors="tomato", linewidths=1.2)

    ax.plot([], [], color="grey", label=f"Faithful (n={faith_idx.size}, subsampled)")
    ax.plot([], [], color="tomato", label=f"Hallucinated (n={hallu_idx.size}, all)")
    ax.set_xlabel(ecs_label)
    ax.set_ylabel(pks_label)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(f"ECS vs PKS density, content words only, all {n_layers} layers pooled\n"
                 f"(pooled: {n_total} content words × {n_layers} layers; "
                 f"contour lines = equal-density levels per class)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_cdf_by_layer(metric_all: np.ndarray, lab_all: np.ndarray, layers: List[int],
                       out_path, metric_name: str, n_total: int, color: str = "tomato") -> None:
    """Grid of empirical-CDF plots, one per layer: hallucinated vs faithful.
    AUROC is exactly P(hallu > faithful) under random draws, i.e. the area
    between these two CDFs -- this shows that separation directly, without
    the visual noise a scatter/raw-value view gets from skewed, heavy-tailed
    data burying the shift in outliers."""
    n_cols = min(2, len(layers))
    n_rows = int(np.ceil(len(layers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5 * n_rows), squeeze=False)
    hallu_mask = lab_all.astype(bool)

    for idx, layer in enumerate(layers):
        ax = axes[idx // n_cols][idx % n_cols]
        h = metric_all[layer, hallu_mask]
        f = metric_all[layer, ~hallu_mask]
        h = np.sort(h[np.isfinite(h)])
        f = np.sort(f[np.isfinite(f)])
        if h.size < 2 or f.size < 2:
            ax.text(0.5, 0.5, f"Layer {layer}\n(insufficient finite values)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.plot(h, np.arange(1, len(h) + 1) / len(h), color=color, lw=2, label="Hallucinated")
        ax.plot(f, np.arange(1, len(f) + 1) / len(f), color="grey", lw=2, label="Faithful")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Cumulative proportion")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(len(layers), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Empirical CDF by layer — {metric_name}, hallucinated vs faithful\n"
                 f"(pooled: {n_total} content words; a rightward/leftward shift between the "
                 f"two curves IS the AUROC separation)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_percentile_hallu_rate(metric_all: np.ndarray, lab_all: np.ndarray, layers: List[int],
                                out_path, metric_name: str, n_total: int,
                                n_bins: int = 10, color: str = "tomato") -> None:
    """Grid of bar charts, one per layer: words binned into `metric` percentile
    buckets (within-layer), plotting fraction hallucinated per bucket. Binning
    averages away point-level noise, making a monotonic trend visible even
    when individual points overlap heavily in a scatter."""
    n_cols = min(2, len(layers))
    n_rows = int(np.ceil(len(layers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.5 * n_rows), squeeze=False)
    overall_rate = lab_all.mean()

    for idx, layer in enumerate(layers):
        ax = axes[idx // n_cols][idx % n_cols]
        s = metric_all[layer]
        finite = np.isfinite(s)
        if finite.sum() < n_bins * 2:
            ax.text(0.5, 0.5, f"Layer {layer}\n(insufficient finite values)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        s_f, y_f = s[finite], lab_all[finite]
        bin_edges = np.nanpercentile(s_f, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-9  # include the max value in the last bin
        bin_idx = np.clip(np.digitize(s_f, bin_edges[1:-1]), 0, n_bins - 1)
        rates = [y_f[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(n_bins)]
        ax.bar(np.arange(1, n_bins + 1), rates, color=color, width=0.8)
        ax.axhline(overall_rate, color="grey", ls="--", lw=1.0, label=f"Overall rate: {overall_rate:.1%}")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel(f"{metric_name} decile (1=lowest, {n_bins}=highest)")
        ax.set_ylabel("Fraction hallucinated")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    for j in range(len(layers), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Hallucination rate by {metric_name} percentile bucket\n"
                 f"(pooled: {n_total} content words; a monotonic trend across bars is the "
                 f"AUROC/Cohen's d effect, with point-level noise binned away)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_percentile_hallu_rate_heatmap(metric_all: np.ndarray, lab_all: np.ndarray,
                                        out_path, metric_name: str, n_total: int,
                                        n_bins: int = 10, cmap: str = "RdBu_r") -> None:
    """Single heatmap: rows = metric decile (1=lowest .. n_bins=highest),
    columns = EVERY layer, color = fraction hallucinated in that (layer,
    decile) bucket. Shows the same information as _plot_percentile_hallu_rate
    but for all layers at once instead of 4 hand-picked panels -- lets you see
    where a decile trend is increasing, decreasing, or flips sign across depth."""
    n_layers = metric_all.shape[0]
    overall_rate = lab_all.mean()
    grid = np.full((n_bins, n_layers), np.nan)

    for l in range(n_layers):
        s = metric_all[l]
        finite = np.isfinite(s)
        if finite.sum() < n_bins * 2:
            continue
        s_f, y_f = s[finite], lab_all[finite]
        bin_edges = np.nanpercentile(s_f, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-9
        bin_idx = np.clip(np.digitize(s_f, bin_edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            if (bin_idx == b).any():
                grid[b, l] = y_f[bin_idx == b].mean()

    finite_vals = grid[np.isfinite(grid)]
    vmin, vmax = (float(finite_vals.min()), float(finite_vals.max())) if finite_vals.size else (0.0, 1.0)
    vcenter = float(np.clip(overall_rate, vmin + 1e-6, vmax - 1e-6)) if vmax > vmin else overall_rate
    norm = (matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            if vmax > vmin else None)

    fig, ax = plt.subplots(figsize=(0.35 * n_layers + 3, 0.4 * n_bins + 2))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                   extent=[-0.5, n_layers - 0.5, 0.5, n_bins + 0.5])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction hallucinated")
    ax.set_xlabel("Layer index")
    ax.set_ylabel(f"{metric_name} decile (1=lowest, {n_bins}=highest)")
    ax.set_yticks(np.arange(1, n_bins + 1))
    ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 16)))
    ax.set_title(f"Hallucination rate by {metric_name} decile, all layers\n"
                 f"(pooled: {n_total} content words; overall rate {overall_rate:.1%}, "
                 f"colorbar centered there -- red=enriched, blue=depleted)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Whitespace-tolerant span → char-range matching
# ─────────────────────────────────────────────────────────────────────────────

def _compact(s: str) -> Tuple[str, List[int]]:
    """Collapse whitespace runs to a single space; return (compact, orig_idx)
    where orig_idx[i] is the index in `s` of the i-th compact char."""
    out_chars: List[str] = []
    orig_idx:  List[int] = []
    prev_space = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            out_chars.append(" ")
            orig_idx.append(i)
            prev_space = True
        else:
            out_chars.append(ch)
            orig_idx.append(i)
            prev_space = False
    return "".join(out_chars), orig_idx


def find_span_char_range(recon: str, span: str) -> Optional[Tuple[int, int]]:
    """Locate `span` in `recon`, tolerant to whitespace and case differences.
    Returns (char_start, char_end) in `recon`, or None if not found."""
    span = span.strip()
    if not span:
        return None
    c_recon, idx_map = _compact(recon)
    c_span, _        = _compact(span)
    c_span = c_span.strip()
    if not c_span:
        return None

    pos = c_recon.find(c_span)
    if pos < 0:
        pos = c_recon.lower().find(c_span.lower())   # case-insensitive fallback
    if pos < 0:
        return None
    start_orig = idx_map[pos]
    end_orig   = idx_map[pos + len(c_span) - 1] + 1
    return start_orig, end_orig


def build_word_labels(word_strs: np.ndarray,
                      note_spans: List[str],
                      note_category_sets: Optional[dict] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, int, int, dict]:
    """
    Return (labels, valid_mask, n_spans, n_matched, categories):
      labels     (n_words,) int   1 = hallucinated (word overlaps a matched span)
      valid_mask (n_words,) bool  False for empty / whitespace-only words
      n_spans    number of note_span strings attempted
      n_matched  number located in the reconstructed note text
      categories dict[str, (n_words,) object array] -- one entry per key in
                 `note_category_sets` (e.g. "judge_label" -> the LLM judge's
                 per-sentence label; "content_type" -> the rule-based
                 condition/procedure/medication/numerical/name/word bucket).
                 Each value list must be the same length as `note_spans`
                 (one category per span). Each output array defaults to
                 "Faithful", overwritten with the matching span's category for
                 words that overlap it. If a word overlaps spans of more than
                 one category (rare), the last matching span wins -- same
                 overlap semantics as `labels`. Empty dict if
                 `note_category_sets` is None/empty.
    """
    words = [str(w) for w in word_strs]
    recon = "".join(words)

    # Per-word character range [start, end) in `recon`.
    starts, ends, cum = [], [], 0
    for w in words:
        starts.append(cum)
        cum += len(w)
        ends.append(cum)
    starts = np.array(starts); ends = np.array(ends)

    labels = np.zeros(len(words), dtype=int)
    note_category_sets = note_category_sets or {}
    categories = {name: np.full(len(words), "Faithful", dtype=object)
                  for name in note_category_sets}
    n_spans = 0
    n_matched = 0
    for i, span in enumerate(note_spans):
        if not str(span).strip():
            continue
        n_spans += 1
        rng = find_span_char_range(recon, str(span))
        if rng is None:
            continue
        n_matched += 1
        s0, s1 = rng
        # A word overlaps the span iff word_start < span_end and word_end > span_start.
        hit = (starts < s1) & (ends > s0)
        labels[hit] = 1
        for name, cat_list in note_category_sets.items():
            categories[name][hit] = cat_list[i]

    valid_mask = np.array([bool(w.strip()) for w in words])
    return labels, valid_mask, n_spans, n_matched, categories


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Word-level ReDeEP AUROC / Cohen's d plots (pooled across gens, "
                    "optionally across multiple dataset splits)")
    p.add_argument("--act-dir",  nargs="+", default=[DEFAULT_ACT_DIR],
                   help="One or more directories of *_tokens.npz files "
                        "(one per split; paired positionally with --span-dir)")
    p.add_argument("--span-dir", nargs="+", default=[DEFAULT_SPAN_DIR],
                   help="One or more directories of *_span_judge.csv files "
                        "(same count/order as --act-dir)")
    p.add_argument("--out",      default=DEFAULT_OUT_DIR,
                   help="Output directory for figures / CSVs")
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.act_dir) != len(args.span_dir):
        raise SystemExit(
            f"--act-dir ({len(args.act_dir)}) and --span-dir ({len(args.span_dir)}) "
            f"must have the same number of entries, given positionally per split.")
    splits   = list(zip(args.act_dir, args.span_dir))
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers = None
    pooled_ecs, pooled_ecscopy, pooled_pks, pooled_lab = [], [], [], []
    pooled_content = []
    pooled_cat = []
    pooled_ctype = []
    coverage_rows = []
    pooled_resp_pks, pooled_resp_lab = [], []
    pooled_tok_pks, pooled_tok_ecs, pooled_tok_ecscopy, pooled_tok_lab = [], [], [], []
    pooled_pks_centered, pooled_ecs_centered, pooled_ecscopy_centered, pooled_lab_centered = [], [], [], []
    per_gen_records = []

    for act_dir_s, span_dir_s in splits:
        act_dir  = Path(act_dir_s)
        span_dir = Path(span_dir_s)
        split_tag = act_dir.parent.name if act_dir.name == "activations" else act_dir.name

        tok_files = sorted(act_dir.glob("sample_*_gen_*_tokens.npz"))
        if not tok_files:
            print(f"[{split_tag}] No *_tokens.npz files in {act_dir}. Skipping.")
            continue
        print(f"[{split_tag}] Found {len(tok_files)} token-level activation files")

        for tp in tok_files:
            m = re.search(r"sample_(\d+)_gen_(\d+)_tokens", tp.stem)
            if not m:
                continue
            si, k = int(m.group(1)), int(m.group(2))

            span_csv = span_dir / f"sample_{si:03d}_note_{k:02d}_span_judge.csv"
            if not span_csv.exists():
                print(f"  [{split_tag}] [skip] sample {si} gen {k}: no span judge CSV")
                continue

            d = np.load(str(tp), allow_pickle=True)
            for key in ("ecs_word", "ecs_word_copy", "pks_word", "word_strs", "word_spans"):
                if key not in d:
                    print(f"  [{split_tag}] [skip] sample {si} gen {k}: {tp.name} missing '{key}' "
                          f"(re-run redeep_sentence.py to regenerate)")
                    break
            else:
                ecs_w     = d["ecs_word"]           # (n_layers, n_words)
                ecscopy_w = d["ecs_word_copy"]
                pks_w     = d["pks_word"]
                word_strs = d["word_strs"]
                if n_layers is None:
                    n_layers = ecs_w.shape[0]
                elif ecs_w.shape[0] != n_layers:
                    print(f"  [{split_tag}] [skip] sample {si} gen {k}: "
                          f"{ecs_w.shape[0]} layers != expected {n_layers}")
                    continue

                sdf = pd.read_csv(span_csv)
                span_col  = sdf.get("note_span",  pd.Series([], dtype=str)).fillna("")
                label_col = sdf.get("label",      pd.Series([], dtype=str)).fillna("")
                span_mask   = span_col.str.strip().astype(bool)
                note_spans  = span_col[span_mask].tolist()
                note_cats   = label_col[span_mask].tolist()   # LLM-judge label per span, e.g. Fabrication
                note_ctypes = [classify_content_type(s) for s in note_spans]  # rule-based content type

                labels, valid, n_spans, n_matched, cats = build_word_labels(
                    word_strs, note_spans,
                    note_category_sets={"judge_label": note_cats, "content_type": note_ctypes})

                # Token-level pooling (no averaging across a word's sub-tokens),
                # restricted to content-word tokens. PKS is genuinely per-token
                # (pks_tok, one raw JSD value per note token -- no word-level
                # mean-pooling). ECS has no saved per-token form in this pipeline
                # (Eq. 3's word-vector IS the mean over the word's tokens, so a
                # single sub-word token has no ECS of its own) -- so each token
                # inherits its own word's ECS (copying heads) value, broadcast
                # across all of that word's tokens via word_spans.
                if "pks_tok" in d:
                    pks_tok        = d["pks_tok"]                    # (n_layers, n_note_tokens)
                    word_spans_all = d["word_spans"]                 # (n_words_full, 2) token-idx ranges
                    ecs_w_full     = d["ecs_word"]                   # (n_layers, n_words_full), pre-slice
                    ecscopy_w_full = d["ecs_word_copy"]              # (n_layers, n_words_full), pre-slice
                    n_words_full   = len(word_strs)
                    content_full   = np.array([is_content_word(str(w)) for w in word_strs])
                    keep_full      = valid[:n_words_full] & content_full
                    labels_full    = labels[:n_words_full]
                    for wi in np.where(keep_full)[0]:
                        s, e = int(word_spans_all[wi, 0]), int(word_spans_all[wi, 1])
                        if e <= s or e > pks_tok.shape[1]:
                            continue
                        n_tok_word = e - s
                        pooled_tok_pks.append(pks_tok[:, s:e])
                        pooled_tok_ecs.append(np.tile(ecs_w_full[:, wi:wi + 1], (1, n_tok_word)))
                        pooled_tok_ecscopy.append(np.tile(ecscopy_w_full[:, wi:wi + 1], (1, n_tok_word)))
                        pooled_tok_lab.append(np.full(n_tok_word, labels_full[wi]))

                # Keep only non-empty words (drop empty / special-token artifacts).
                n_words = ecs_w.shape[1]
                keep = valid[:n_words]
                ecs_w     = ecs_w[:, keep]
                ecscopy_w = ecscopy_w[:, keep]
                pks_w     = pks_w[:, keep]
                lab       = labels[:n_words][keep]
                cat       = cats["judge_label"][:n_words][keep]
                ctype     = cats["content_type"][:n_words][keep]
                content   = np.array([is_content_word(str(w))
                                      for w in np.asarray(word_strs)[:n_words][keep]])

                pooled_ecs.append(ecs_w)
                pooled_ecscopy.append(ecscopy_w)
                pooled_pks.append(pks_w)
                pooled_lab.append(lab)
                pooled_cat.append(cat)
                pooled_ctype.append(ctype)
                pooled_content.append(content)

                # Per-generation-centered content-word PKS/ECS: subtract THIS
                # generation's own mean (over its content words) before pooling
                # across generations. Removes the between-generation nuisance
                # variance (note length/topic/model calibration drift) that
                # inflates the pooled std bands without being about hallucination --
                # targets the clustering/independence issue noted earlier, rather
                # than just re-styling the same numbers.
                if content.any():
                    pks_cw_g     = pks_w[:, content]
                    ecs_cw_g     = ecs_w[:, content]
                    ecscopy_cw_g = ecscopy_w[:, content]
                    pooled_pks_centered.append(pks_cw_g - np.nanmean(pks_cw_g, axis=1, keepdims=True))
                    pooled_ecs_centered.append(ecs_cw_g - np.nanmean(ecs_cw_g, axis=1, keepdims=True))
                    pooled_ecscopy_centered.append(
                        ecscopy_cw_g - np.nanmean(ecscopy_cw_g, axis=1, keepdims=True))
                    pooled_lab_centered.append(lab[content])

                    # Per-file content-word record, tagged by (split, generation
                    # index) -- lets us later group by generation index within a
                    # single split (e.g. aci_test1) instead of pooling everything.
                    per_gen_records.append({
                        "split": split_tag, "gen_idx": k,
                        "ecs": ecs_cw_g, "ecscopy": ecscopy_cw_g, "pks": pks_cw_g,
                        "lab": lab[content],
                    })

                # Response-level aggregate (mirrors ReDeEP Eq. 6: mean PKS over
                # ALL tokens in the response, not just content words -- one
                # PKS-per-layer value and one binary hallucination label per
                # (sample, gen) response, for a direct comparison against the
                # paper's response-level correlation rather than our word-level one).
                if pks_w.shape[1] > 0:
                    pooled_resp_pks.append(np.nanmean(pks_w, axis=1))   # (n_layers,)
                    pooled_resp_lab.append(int(lab.sum() > 0))          # any hallucinated word in this response

                coverage_rows.append({
                    "split":       split_tag,
                    "sample_idx":  si,
                    "gen_idx":     k,
                    "n_words":     int(keep.sum()),
                    "n_hallu":     int(lab.sum()),
                    "n_spans":     n_spans,
                    "n_matched":   n_matched,
                    "match_rate":  round(n_matched / n_spans, 3) if n_spans else np.nan,
                })

    if not pooled_lab:
        print("No usable (tokens + span) pairs found.")
        return

    # ── Pool all words across gens ────────────────────────────────────────────
    ecs_all     = np.concatenate(pooled_ecs,     axis=1)   # (n_layers, total_words)
    ecscopy_all = np.concatenate(pooled_ecscopy, axis=1)
    pks_all     = np.concatenate(pooled_pks,     axis=1)
    lab_all     = np.concatenate(pooled_lab)               # (total_words,)
    cat_all     = np.concatenate(pooled_cat)               # (total_words,) str -- LLM-judge label
    ctype_all   = np.concatenate(pooled_ctype)             # (total_words,) str -- rule-based content type
    content_all = np.concatenate(pooled_content)            # (total_words,) bool

    n_total = lab_all.shape[0]
    n_hallu = int(lab_all.sum())
    print(f"\nPooled words: {n_total}  |  hallucinated: {n_hallu} "
          f"({n_hallu / n_total:.1%})  |  faithful: {n_total - n_hallu}")

    cov = pd.DataFrame(coverage_rows)
    cov.to_csv(out_dir / "word_label_coverage.csv", index=False)
    tot_spans   = int(cov["n_spans"].sum())
    tot_matched = int(cov["n_matched"].sum())
    print(f"Span match rate: {tot_matched}/{tot_spans} "
          f"({(tot_matched / tot_spans if tot_spans else 0):.1%}) "
          f"→ word_label_coverage.csv")

    if n_hallu == 0:
        print("No hallucinated words after span matching — cannot compute AUROC. "
              "Check that span_judge CSVs contain non-Faithful note_span entries.")
        return

    # ── Per-layer AUROC / Cohen's d over the pooled word population ────────────
    # ECS scored with hallu_high=True ⇒ AUROC = P(ECS_hallucinated > ECS_faithful),
    # i.e. we report RAW ECS, not 1−ECS. NOTE: an AUROC > 0.5 here therefore means
    # hallucinated words have HIGHER ECS than faithful ones — the OPPOSITE of
    # ReDeEP's hypothesis (low ECS → hallucination). Equivalently this line is the
    # mirror (1 − x) of the ReDeEP-oriented 1−ECS curve. PKS keeps the ReDeEP
    # direction (high PKS → hallucination).
    ecs_auroc     = auroc_per_layer_single(ecs_all,     lab_all, hallu_high=True)
    ecscopy_auroc = auroc_per_layer_single(ecscopy_all, lab_all, hallu_high=True)
    pks_auroc     = auroc_per_layer_single(pks_all,     lab_all, hallu_high=True)

    ecs_d     = cohens_d_per_layer_single(ecs_all,     lab_all)
    ecscopy_d = cohens_d_per_layer_single(ecscopy_all, lab_all)
    pks_d     = cohens_d_per_layer_single(pks_all,     lab_all)

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_layer_lines(
        [
            (ecs_auroc,     None, "steelblue", "o", "ECS (all heads)"),
            (ecscopy_auroc, None, "seagreen",  "^", "ECS (copying heads)"),
            (pks_auroc,     None, "tomato",    "s", "PKS (per-token mean)"),
        ],
        out_dir / "fig_word_auroc.png",
        ylabel="AUROC",
        title=f"Word-level AUROC — ECS & PKS (per-token mean) vs hallucinated words\n"
              f"(pooled: {n_total} words, {n_hallu} hallucinated)",
        hline=0.5,
        ylim=[0.3, 1.0],
    )

    # ── Content-words-only AUROC (drop articles/pronouns/prepositions/etc.) ────
    n_content       = int(content_all.sum())
    n_hallu_content = int(lab_all[content_all].sum())
    if n_content == 0 or n_hallu_content == 0 or (n_content - n_hallu_content) == 0:
        print("Skipping content-words-only AUROC plot: no usable class balance "
              f"({n_content} content words, {n_hallu_content} hallucinated).")
    else:
        ecs_auroc_c     = auroc_per_layer_single(ecs_all[:, content_all],     lab_all[content_all], hallu_high=True)
        ecscopy_auroc_c = auroc_per_layer_single(ecscopy_all[:, content_all], lab_all[content_all], hallu_high=True)
        pks_auroc_c     = auroc_per_layer_single(pks_all[:, content_all],     lab_all[content_all], hallu_high=True)
        _plot_layer_lines(
            [
                (ecs_auroc_c,     None, "steelblue", "o", "ECS (all heads)"),
                (ecscopy_auroc_c, None, "seagreen",  "^", "ECS (copying heads)"),
                (pks_auroc_c,     None, "tomato",    "s", "PKS (per-token mean)"),
            ],
            out_dir / "fig_word_auroc_content_only.png",
            ylabel="AUROC",
            title=f"Word-level AUROC (content words only) — ECS & PKS vs hallucinated words\n"
                  f"(pooled: {n_content} content words, {n_hallu_content} hallucinated; "
                  f"stopwords/punctuation excluded)",
            hline=0.5,
            ylim=[0.3, 1.0],
        )
        pd.DataFrame({
            "layer":           np.arange(n_layers),
            "ecs_auroc":       ecs_auroc_c,
            "ecs_copy_auroc":  ecscopy_auroc_c,
            "pks_auroc":       pks_auroc_c,
        }).to_csv(out_dir / "layer_word_metrics_content_only.csv", index=False)
        print(f"  Saved {out_dir / 'layer_word_metrics_content_only.csv'}")

    _plot_layer_lines(
        [
            (ecs_d,     None, "steelblue", "o", "ECS (all heads)"),
            (ecscopy_d, None, "seagreen",  "^", "ECS (copying heads)"),
            (pks_d,     None, "tomato",    "s", "PKS (per-token mean)"),
        ],
        out_dir / "fig_word_cohens_d.png",
        ylabel="Cohen's d  (hallucinated − faithful)",
        title=f"Word-level Cohen's d — ECS & PKS (per-token mean) vs hallucinated words\n"
              f"(pooled: {n_total} words, {n_hallu} hallucinated)",
        hline=0.0,
    )
    _plot_cohens_d_with_metric_bar(
        pks_d, pks_all, lab_all,
        out_dir / "fig_word_cohens_d_pks_bar.png",
        title_suffix=f"(pooled: {n_total} words, {n_hallu} hallucinated)",
        metric_name="PKS (per-token mean)",
    )
    if n_content > 0 and n_hallu_content > 0 and (n_content - n_hallu_content) > 0:
        pks_d_c = cohens_d_per_layer_single(pks_all[:, content_all], lab_all[content_all])
        _plot_cohens_d_with_metric_bar(
            pks_d_c, pks_all[:, content_all], lab_all[content_all],
            out_dir / "fig_word_cohens_d_pks_bar_content_only.png",
            title_suffix=f"content words only (pooled: {n_content} content words, "
                         f"{n_hallu_content} hallucinated)",
            metric_name="PKS (per-token mean)",
            show_title=False, only_cohens_d=True, fontsize=16,
        )
        _plot_mean_std_by_layer(
            pks_all[:, content_all], lab_all[content_all],
            out_dir / "fig_word_actual_pks_content_only.png",
            ylabel="PKS (per-token mean)",
            show_title=False, fontsize=16,
        )
    _plot_pks_paper_style(
        pks_all, lab_all,
        out_dir / "fig_word_pks_paper_style.png",
    )

    # Word-level ECS (copying heads), mirroring the PKS mean±std/Cohen's d panel above.
    _plot_cohens_d_with_metric_bar(
        ecscopy_d, ecscopy_all, lab_all,
        out_dir / "fig_word_cohens_d_ecs_bar.png",
        title_suffix=f"(pooled: {n_total} words, {n_hallu} hallucinated)",
        metric_name="ECS (copying heads)",
        metric_color="seagreen",
        show_title=False, only_cohens_d=True, fontsize=16,
    )
    _plot_mean_std_by_layer(
        ecscopy_all, lab_all,
        out_dir / "fig_word_actual_ecs.png",
        ylabel="ECS (copying heads)",
        show_title=False, fontsize=16,
    )
    if n_content > 0 and n_hallu_content > 0 and (n_content - n_hallu_content) > 0:
        ecscopy_d_c = cohens_d_per_layer_single(ecscopy_all[:, content_all], lab_all[content_all])
        _plot_cohens_d_with_metric_bar(
            ecscopy_d_c, ecscopy_all[:, content_all], lab_all[content_all],
            out_dir / "fig_word_cohens_d_ecs_bar_content_only.png",
            title_suffix=f"content words only (pooled: {n_content} content words, "
                         f"{n_hallu_content} hallucinated)",
            metric_name="ECS (copying heads)",
            metric_color="seagreen",
            show_title=False, only_cohens_d=True, fontsize=16,
        )
        pd.DataFrame({
            "layer": np.arange(n_layers),
            "pks_cohens_d_content_only": pks_d_c,
            "ecs_copy_cohens_d_content_only": ecscopy_d_c,
        }).to_csv(out_dir / "layer_word_cohens_d_content_only.csv", index=False)
        print(f"  Saved {out_dir / 'layer_word_cohens_d_content_only.csv'}")
        _plot_mean_std_by_layer(
            ecscopy_all[:, content_all], lab_all[content_all],
            out_dir / "fig_word_actual_ecs_content_only.png",
            ylabel="ECS (copying heads)",
            show_title=False, fontsize=16,
        )

    # ── Per-group AUROC, content words only (stopwords/punctuation excluded
    #    via content_all). Two groupings, same words: the LLM judge's own
    #    sentence-level label (Fabrication/Negation/Causality/Contextual),
    #    and the rule-based content-type bucket (condition/procedure/
    #    medication/numerical/name/word) from hallucination_content_type.py ──
    ecs_all_cw     = ecs_all[:, content_all]
    ecscopy_all_cw = ecscopy_all[:, content_all]
    pks_all_cw     = pks_all[:, content_all]
    n_content = int(content_all.sum())

    # All-words (not content-filtered) version, styled for the paper: no
    # descriptive suptitle, raw ECS (all heads) curve dropped, larger axis text.
    _plot_auroc_by_group(
        cat_all, ecs_all, ecscopy_all, pks_all, n_layers,
        out_dir / "fig_word_auroc_by_category.png",
        out_dir / "layer_word_metrics_by_category.csv",
        suptitle=f"Word-level AUROC by LLM-judge category vs Faithful — ECS & PKS\n"
                 f"(pooled: {n_total} words)",
        show_suptitle=False, include_raw_ecs=False, fontsize=14,
    )
    _plot_auroc_by_group(
        cat_all[content_all], ecs_all_cw, ecscopy_all_cw, pks_all_cw, n_layers,
        out_dir / "fig_word_auroc_by_category_content_only.png",
        out_dir / "layer_word_metrics_by_category_content_only.csv",
        suptitle=f"Word-level AUROC by LLM-judge category vs Faithful — ECS & PKS\n"
                 f"content words only (pooled: {n_content} content words)",
    )
    _plot_auroc_by_group(
        ctype_all[content_all], ecs_all_cw, ecscopy_all_cw, pks_all_cw, n_layers,
        out_dir / "fig_word_auroc_by_content_type_content_only.png",
        out_dir / "layer_word_metrics_by_content_type_content_only.csv",
        suptitle=f"Word-level AUROC by rule-based content type vs Faithful — ECS & PKS\n"
                 f"content words only (pooled: {n_content} content words)",
    )

    # ── ECS vs PKS scatter, content words only, hallucinated vs faithful ──────
    lab_cw = lab_all[content_all]
    ecscopy_auroc_cw = auroc_per_layer_single(ecscopy_all_cw, lab_cw, hallu_high=True)
    pks_auroc_cw     = auroc_per_layer_single(pks_all_cw,     lab_cw, hallu_high=True)
    # Earliest layer with any finite ECS (copying heads) value -- layer 0 is
    # typically empty since copying heads are rare/absent that early.
    layers_with_ecscopy = np.where(np.isfinite(ecscopy_all_cw).any(axis=1))[0]
    earliest_layer = int(layers_with_ecscopy.min()) if layers_with_ecscopy.size else 0
    scatter_layers = sorted(set([
        earliest_layer,
        int(np.nanargmax(ecscopy_auroc_cw)),
        int(np.nanargmax(pks_auroc_cw)),
        n_layers - 1,
    ]))
    _plot_ecs_vs_pks_scatter(
        ecscopy_all_cw, pks_all_cw, lab_cw, scatter_layers,
        out_dir / "fig_ecs_vs_pks_scatter_content_only.png",
        n_content,
    )
    _plot_ecs_vs_pks_kde(
        ecscopy_all_cw, pks_all_cw, lab_cw,
        out_dir / "fig_ecs_vs_pks_kde_all_layers_content_only.png",
        n_content,
    )

    # ── Per-generation-centered CDF + percentile-binned hallucination rate ────
    # Centering removes each generation's own baseline PKS/ECS before pooling,
    # targeting the between-generation clustering confound noted earlier
    # (66,738 words come from only 200 independent generations) rather than
    # just re-styling the same raw numbers.
    if pooled_lab_centered:
        pks_c_all     = np.concatenate(pooled_pks_centered,     axis=1)
        ecs_c_all     = np.concatenate(pooled_ecs_centered,     axis=1)
        ecscopy_c_all = np.concatenate(pooled_ecscopy_centered, axis=1)
        lab_c_all     = np.concatenate(pooled_lab_centered)
        n_c = lab_c_all.shape[0]
        n_c_hallu = int(lab_c_all.sum())
        print(f"\nPer-generation-centered content words pooled: {n_c}  |  hallucinated: {n_c_hallu} "
              f"({n_c_hallu / n_c:.1%})")

        if n_c_hallu >= 2 and (n_c - n_c_hallu) >= 2:
            ecs_c_auroc     = auroc_per_layer_single(ecs_c_all,     lab_c_all, hallu_high=True)
            ecscopy_c_auroc = auroc_per_layer_single(ecscopy_c_all, lab_c_all, hallu_high=True)
            pks_c_auroc     = auroc_per_layer_single(pks_c_all,     lab_c_all, hallu_high=True)
            layers_with_ecscopy_c = np.where(np.isfinite(ecscopy_c_all).any(axis=1))[0]
            earliest_c = int(layers_with_ecscopy_c.min()) if layers_with_ecscopy_c.size else 0
            centered_layers = sorted(set([
                earliest_c,
                int(np.nanargmax(ecscopy_c_auroc)),
                int(np.nanargmax(pks_c_auroc)),
                n_layers - 1,
            ]))

            # AUROC per layer on the centered data, directly comparable to
            # fig_word_auroc_content_only.png (raw, uncentered) -- same 3
            # series, same axes/style, so centering's effect on AUROC itself
            # (not just visual clarity) is a straight side-by-side comparison.
            _plot_layer_lines(
                [
                    (ecscopy_c_auroc, None, "seagreen",  "^", "ECS (copying heads)"),
                    (pks_c_auroc,     None, "tomato",    "s", "PKS (per-token mean)"),
                ],
                out_dir / "fig_word_auroc_centered_content_only.png",
                ylabel="AUROC",
                title=f"Word-level AUROC, per-generation-centered (content words only)\n"
                      f"(pooled: {n_c} content words, {n_c_hallu} hallucinated; "
                      f"compare against fig_word_auroc_content_only.png)",
                hline=0.5,
                ylim=[0.3, 1.0],
                show_title=False, fontsize=16,
            )
            pd.DataFrame({
                "layer":           np.arange(n_layers),
                "ecs_auroc":       ecs_c_auroc,
                "ecs_copy_auroc":  ecscopy_c_auroc,
                "pks_auroc":       pks_c_auroc,
            }).to_csv(out_dir / "layer_word_metrics_centered_content_only.csv", index=False)
            print(f"  Saved {out_dir / 'layer_word_metrics_centered_content_only.csv'}")

            def _best_c(name, arr):
                safe = np.where(np.isfinite(arr), arr, -np.inf)
                l = int(np.argmax(safe))
                return f"  {name}: best layer {l} = {arr[l]:.3f}"
            print("Centered peak discrimination (compare to raw 'Peak discrimination' below):")
            print(_best_c("ECS (all heads)     AUROC", ecs_c_auroc))
            print(_best_c("ECS (copying)       AUROC", ecscopy_c_auroc))
            print(_best_c("PKS (per-token mean) AUROC", pks_c_auroc))

            _plot_cdf_by_layer(
                pks_c_all, lab_c_all, centered_layers,
                out_dir / "fig_pks_cdf_centered_content_only.png",
                "PKS (per-generation-centered)", n_c, color="tomato",
            )
            _plot_cdf_by_layer(
                ecscopy_c_all, lab_c_all, centered_layers,
                out_dir / "fig_ecs_cdf_centered_content_only.png",
                "ECS copying heads (per-generation-centered)", n_c, color="seagreen",
            )
            _plot_percentile_hallu_rate(
                pks_c_all, lab_c_all, centered_layers,
                out_dir / "fig_pks_percentile_hallu_rate_centered_content_only.png",
                "PKS (per-generation-centered)", n_c, color="tomato",
            )
            _plot_percentile_hallu_rate(
                ecscopy_c_all, lab_c_all, centered_layers,
                out_dir / "fig_ecs_percentile_hallu_rate_centered_content_only.png",
                "ECS copying heads (per-generation-centered)", n_c, color="seagreen",
            )
            _plot_percentile_hallu_rate_heatmap(
                pks_c_all, lab_c_all,
                out_dir / "fig_pks_percentile_hallu_rate_heatmap_centered_content_only.png",
                "PKS (per-generation-centered)", n_c,
            )
            _plot_percentile_hallu_rate_heatmap(
                ecscopy_c_all, lab_c_all,
                out_dir / "fig_ecs_percentile_hallu_rate_heatmap_centered_content_only.png",
                "ECS copying heads (per-generation-centered)", n_c,
            )
            _plot_ecs_vs_pks_scatter(
                ecscopy_c_all, pks_c_all, lab_c_all, centered_layers,
                out_dir / "fig_ecs_vs_pks_scatter_centered_content_only.png",
                n_c,
                ecs_label="ECS (copying heads, per-generation-centered)",
                pks_label="PKS (per-generation-centered)",
            )
            _plot_metric_strip_all_layers(
                ecscopy_c_all, lab_c_all,
                out_dir / "fig_ecs_strip_all_layers_centered_content_only.png",
                "ECS (copying heads, per-generation-centered)", n_c,
                color="seagreen",
            )
        else:
            print("Skipping centered CDF/percentile plots: insufficient class balance.")

    # ── Token-level ECS vs PKS scatter, content words only ────────────────────
    # PKS here is genuinely per-token (no mean-pooling across a word's
    # sub-tokens); ECS is the word-level value broadcast to each of that
    # word's tokens (see note at the pooling site above -- ECS has no finer
    # granularity than "word" in this pipeline).
    if pooled_tok_lab:
        tok_pks_all     = np.concatenate(pooled_tok_pks,     axis=1)
        tok_ecs_all     = np.concatenate(pooled_tok_ecs,     axis=1)
        tok_ecscopy_all = np.concatenate(pooled_tok_ecscopy, axis=1)
        tok_lab_all     = np.concatenate(pooled_tok_lab)
        n_tok = tok_lab_all.shape[0]
        n_tok_hallu = int(tok_lab_all.sum())
        print(f"\nPooled content-word tokens: {n_tok}  |  hallucinated: {n_tok_hallu} "
              f"({n_tok_hallu / n_tok:.1%})  |  faithful: {n_tok - n_tok_hallu}")

        if n_tok_hallu >= 2 and (n_tok - n_tok_hallu) >= 2:
            tok_ecs_auroc     = auroc_per_layer_single(tok_ecs_all,     tok_lab_all, hallu_high=True)
            tok_ecscopy_auroc = auroc_per_layer_single(tok_ecscopy_all, tok_lab_all, hallu_high=True)
            tok_pks_auroc     = auroc_per_layer_single(tok_pks_all,     tok_lab_all, hallu_high=True)

            # AUROC per layer, token granularity (PKS raw per-token, not
            # word-averaged; ECS word-level broadcast to tokens -- see note above).
            _plot_layer_lines(
                [
                    (tok_ecs_auroc,     None, "steelblue", "o", "ECS (all heads, word-level broadcast)"),
                    (tok_ecscopy_auroc, None, "seagreen",  "^", "ECS (copying heads, word-level broadcast)"),
                    (tok_pks_auroc,     None, "tomato",    "s", "PKS (raw per-token, not averaged)"),
                ],
                out_dir / "fig_token_auroc_content_only.png",
                ylabel="AUROC",
                title=f"Token-level AUROC (content words only) — ECS & PKS vs hallucinated tokens\n"
                      f"(pooled: {n_tok} content tokens, {n_tok_hallu} hallucinated; "
                      f"PKS per-token, ECS word-level broadcast)",
                hline=0.5,
                ylim=[0.3, 1.0],
            )
            pd.DataFrame({
                "layer":           np.arange(n_layers),
                "ecs_auroc":       tok_ecs_auroc,
                "ecs_copy_auroc":  tok_ecscopy_auroc,
                "pks_auroc":       tok_pks_auroc,
            }).to_csv(out_dir / "layer_token_metrics_content_only.csv", index=False)
            print(f"  Saved {out_dir / 'layer_token_metrics_content_only.csv'}")

            # ECS mean±std by layer, hallucinated vs faithful tokens (mirrors the
            # PKS mean±std/Cohen's d panel, now for ECS at token granularity).
            tok_ecscopy_d = cohens_d_per_layer_single(tok_ecscopy_all, tok_lab_all)
            _plot_cohens_d_with_metric_bar(
                tok_ecscopy_d, tok_ecscopy_all, tok_lab_all,
                out_dir / "fig_token_ecs_cohens_d_bar.png",
                title_suffix=f"content words only (pooled: {n_tok} content tokens, "
                             f"{n_tok_hallu} hallucinated; word-level value broadcast to tokens)",
                metric_name="ECS (copying heads)",
                metric_color="seagreen",
                granularity="Token-level",
            )

            layers_with_tok_ecscopy = np.where(np.isfinite(tok_ecscopy_all).any(axis=1))[0]
            tok_earliest = int(layers_with_tok_ecscopy.min()) if layers_with_tok_ecscopy.size else 0
            tok_scatter_layers = sorted(set([
                tok_earliest,
                int(np.nanargmax(tok_ecscopy_auroc)),
                int(np.nanargmax(tok_pks_auroc)),
                n_layers - 1,
            ]))
            _plot_ecs_vs_pks_scatter(
                tok_ecscopy_all, tok_pks_all, tok_lab_all, tok_scatter_layers,
                out_dir / "fig_ecs_vs_pks_scatter_token_level_content_only.png",
                n_tok,
                ecs_label="ECS (copying heads, word-level value broadcast to tokens)",
                pks_label="PKS (raw per-token, not averaged)",
                unit="token",
            )
        else:
            print("Skipping token-level plots: insufficient class balance.")

    # ── Per-layer metrics CSV ─────────────────────────────────────────────────
    pd.DataFrame({
        "layer":           np.arange(n_layers),
        "ecs_auroc":       ecs_auroc,
        "ecs_copy_auroc":  ecscopy_auroc,
        "pks_auroc":       pks_auroc,
        "ecs_cohens_d":    ecs_d,
        "ecs_copy_cohens_d": ecscopy_d,
        "pks_cohens_d":    pks_d,
    }).to_csv(out_dir / "layer_word_metrics.csv", index=False)
    print(f"  Saved {out_dir / 'layer_word_metrics.csv'}")

    # ── Response-level PKS (mirrors ReDeEP Eq. 6: one mean-PKS-per-layer value
    #    per whole response, correlated against a per-response binary label --
    #    "does this response contain any hallucinated word?" -- rather than our
    #    word-level unit of analysis. Same plotting functions as the word-level
    #    figures above, just fed response-level arrays instead of word-level. ──
    print()
    if len(pooled_resp_lab) < 4:
        print(f"Only {len(pooled_resp_lab)} responses pooled — skipping response-level PKS plots.")
    else:
        pks_resp_all = np.stack(pooled_resp_pks, axis=1)   # (n_layers, n_responses)
        lab_resp_all = np.array(pooled_resp_lab)           # (n_responses,)
        n_resp = lab_resp_all.shape[0]
        n_resp_hallu = int(lab_resp_all.sum())
        print(f"Response-level pooled: {n_resp} responses  |  hallucinated: {n_resp_hallu} "
              f"({n_resp_hallu / n_resp:.1%})  |  faithful: {n_resp - n_resp_hallu}")

        if n_resp_hallu < 2 or (n_resp - n_resp_hallu) < 2:
            print("Skipping response-level PKS plots: insufficient class balance "
                  f"({n_resp_hallu} hallucinated vs {n_resp - n_resp_hallu} faithful responses).")
        else:
            pks_resp_auroc = auroc_per_layer_single(pks_resp_all, lab_resp_all, hallu_high=True)
            pks_resp_d     = cohens_d_per_layer_single(pks_resp_all, lab_resp_all)

            _plot_cohens_d_with_metric_bar(
                pks_resp_d, pks_resp_all, lab_resp_all,
                out_dir / "fig_response_cohens_d_pks_bar.png",
                title_suffix=f"response-level, per Eq. 6 (pooled: {n_resp} responses, "
                             f"{n_resp_hallu} hallucinated)",
                metric_name="PKS (per-token mean)",
                granularity="Response-level",
            )
            _plot_pks_paper_style(
                pks_resp_all, lab_resp_all,
                out_dir / "fig_response_pks_paper_style.png",
            )
            pd.DataFrame({
                "layer":     np.arange(n_layers),
                "pks_auroc": pks_resp_auroc,
                "pks_cohens_d": pks_resp_d,
            }).to_csv(out_dir / "layer_response_pks_metrics.csv", index=False)
            print(f"  Saved {out_dir / 'layer_response_pks_metrics.csv'}")

            best_l = int(np.nanargmax(pks_resp_auroc))
            print(f"  Response-level PKS AUROC: best layer {best_l} = {pks_resp_auroc[best_l]:.3f}"
                  f"  (Cohen's d = {pks_resp_d[best_l]:.3f})")

    # Best layers, for a quick read.
    def _best(name, arr, hi=True):
        safe = np.where(np.isfinite(arr), arr, -np.inf if hi else np.inf)
        l = int((np.argmax if hi else np.argmin)(safe))
        return f"  {name}: best layer {l} = {arr[l]:.3f}"
    print("\nPeak discrimination:")
    print(_best("ECS (all heads)     AUROC", ecs_auroc))
    print(_best("ECS (copying)       AUROC", ecscopy_auroc))
    print(_best("PKS (per-token mean) AUROC", pks_auroc))


if __name__ == "__main__":
    main()
