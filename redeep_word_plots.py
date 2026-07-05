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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ACT_DIR  = "sae_experiments/redeep_out/activations"
DEFAULT_SPAN_DIR = "sae_experiments/luq_out/llama_judge/spans"
DEFAULT_OUT_DIR  = "sae_experiments/redeep_out/word_plots"


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


def _plot_cohens_d_with_pks_bar(pks_d, pks_all, lab_all,
                                out_path, title_suffix, pks_color="tomato"):
    """Two-panel figure:
       1) PKS-only Cohen's d as a uni-color bar chart (ReDeEP-paper style)
       2) actual (raw, unstandardized) PKS scores per layer, mean±std,
          hallucinated vs faithful words
    """
    n_layers = len(pks_d)
    layers   = np.arange(n_layers)
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})

    ax2.bar(layers, pks_d, color=pks_color, width=0.8)
    ax2.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax2.set_ylabel("PKS Cohen's d (per-token mean)")
    ax2.set_title(f"Word-level PKS Cohen's d by layer — per-token mean\n{title_suffix}")
    ax2.grid(alpha=0.3, axis="y")

    hallu_mask = lab_all.astype(bool)
    pks_hallu_mean, pks_hallu_std = _mean_std_per_layer(pks_all, hallu_mask)
    pks_faith_mean, pks_faith_std = _mean_std_per_layer(pks_all, ~hallu_mask)
    ax3.plot(layers, pks_hallu_mean, color="tomato", lw=2, marker="s", ms=3, label="Hallucinated")
    ax3.fill_between(layers, pks_hallu_mean - pks_hallu_std, pks_hallu_mean + pks_hallu_std,
                     color="tomato", alpha=0.15)
    ax3.plot(layers, pks_faith_mean, color="grey", lw=2, marker="o", ms=3, label="Faithful")
    ax3.fill_between(layers, pks_faith_mean - pks_faith_std, pks_faith_mean + pks_faith_std,
                     color="grey", alpha=0.15)
    ax3.set_xlabel("Layer index")
    ax3.set_ylabel("PKS (per-token mean)")
    ax3.set_title("Word-level actual PKS scores by layer (mean ± std, per-token mean)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

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


def _plot_layer_lines(series, out_path, ylabel, title, hline, ylim=None):
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
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
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
                      note_spans: List[str]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Return (labels, valid_mask, n_spans, n_matched):
      labels     (n_words,) int   1 = hallucinated (word overlaps a matched span)
      valid_mask (n_words,) bool  False for empty / whitespace-only words
      n_spans    number of note_span strings attempted
      n_matched  number located in the reconstructed note text
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
    n_spans = 0
    n_matched = 0
    for span in note_spans:
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

    valid_mask = np.array([bool(w.strip()) for w in words])
    return labels, valid_mask, n_spans, n_matched


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Word-level ReDeEP AUROC / Cohen's d plots (pooled across gens)")
    p.add_argument("--act-dir",  default=DEFAULT_ACT_DIR,
                   help="Directory of *_tokens.npz files")
    p.add_argument("--span-dir", default=DEFAULT_SPAN_DIR,
                   help="Directory of *_span_judge.csv files")
    p.add_argument("--out",      default=DEFAULT_OUT_DIR,
                   help="Output directory for figures / CSVs")
    return p.parse_args()


def main():
    args     = parse_args()
    act_dir  = Path(args.act_dir)
    span_dir = Path(args.span_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_files = sorted(act_dir.glob("sample_*_gen_*_tokens.npz"))
    if not tok_files:
        print(f"No *_tokens.npz files in {act_dir}. Run redeep_sentence.py first.")
        return
    print(f"Found {len(tok_files)} token-level activation files")

    n_layers = None
    pooled_ecs, pooled_ecscopy, pooled_pks, pooled_lab = [], [], [], []
    coverage_rows = []

    for tp in tok_files:
        m = re.search(r"sample_(\d+)_gen_(\d+)_tokens", tp.stem)
        if not m:
            continue
        si, k = int(m.group(1)), int(m.group(2))

        span_csv = span_dir / f"sample_{si:03d}_note_{k:02d}_span_judge.csv"
        if not span_csv.exists():
            print(f"  [skip] sample {si} gen {k}: no span judge CSV")
            continue

        d = np.load(str(tp), allow_pickle=True)
        for key in ("ecs_word", "ecs_word_copy", "pks_word", "word_strs", "word_spans"):
            if key not in d:
                print(f"  [skip] sample {si} gen {k}: {tp.name} missing '{key}' "
                      f"(re-run redeep_sentence.py to regenerate)")
                break
        else:
            ecs_w     = d["ecs_word"]           # (n_layers, n_words)
            ecscopy_w = d["ecs_word_copy"]
            pks_w     = d["pks_word"]
            word_strs = d["word_strs"]
            if n_layers is None:
                n_layers = ecs_w.shape[0]

            sdf = pd.read_csv(span_csv)
            note_spans = [s for s in sdf.get("note_span", pd.Series([], dtype=str)).fillna("").tolist()
                          if str(s).strip()]

            labels, valid, n_spans, n_matched = build_word_labels(word_strs, note_spans)

            # Keep only content words (drop empty / special-token words).
            n_words = ecs_w.shape[1]
            keep = valid[:n_words]
            ecs_w     = ecs_w[:, keep]
            ecscopy_w = ecscopy_w[:, keep]
            pks_w     = pks_w[:, keep]
            lab       = labels[:n_words][keep]

            pooled_ecs.append(ecs_w)
            pooled_ecscopy.append(ecscopy_w)
            pooled_pks.append(pks_w)
            pooled_lab.append(lab)

            coverage_rows.append({
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
    _plot_cohens_d_with_pks_bar(
        pks_d, pks_all, lab_all,
        out_dir / "fig_word_cohens_d_pks_bar.png",
        title_suffix=f"(pooled: {n_total} words, {n_hallu} hallucinated)",
    )
    _plot_pks_paper_style(
        pks_all, lab_all,
        out_dir / "fig_word_pks_paper_style.png",
    )

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
