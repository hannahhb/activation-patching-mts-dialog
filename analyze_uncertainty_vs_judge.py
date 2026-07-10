"""
analyze_uncertainty_vs_judge.py
================================
Validates LUQ uncertainty estimation against LLM-as-a-judge faithfulness
labels (CREOLA taxonomy: Faithful/Fabrication/Negation/Causality/Contextual).

Four analyses:
  1. sentence   -- sentence-level uncertainty vs sentence-level judge label.
  2. span       -- atomic-fact uncertainty, provenance-matched to its
                   originating sentence (fact_sentence_match.py output),
                   related back to the judge label of that sentence.
  3. section-u  -- uncertainty distribution by SOAP section (bar plot).
  4. section-j  -- judge faithfulness rate by SOAP section (bar plot).

Metrics: AUROC + AUPRC (Faithful=0 vs Not-Faithful=1), the standard
threshold-independent pair for this literature -- AUROC for overall ranking
quality, AUPRC because faithful sentences dominate (class imbalance).

Data sources (paths overridable via CLI):
  sentence uncertainty : luq_out/llama/generations/<config>/<split>/sentences/sample_NNN_note_KK_sentences.csv
  span/sentence judge  : luq_out/llama_judge/<config>/<split>/spans/sample_NNN_note_KK_span_judge.csv
  fact uncertainty+prov: luq_out/llama_atomic/facts_matched/sample_NNN_note_KK_facts_matched.csv
  note text (for section-j): luq_out/llama/generations/<config>/<split>/sample_NNN_generations.json

The two pipelines split sentences differently (the judge skips preamble/
header-only lines that split_sentences() includes), so rows are joined by
exact normalized sentence TEXT, not sentence_idx -- confirmed on real data
that the text matches byte-for-byte where both pipelines cover the same
sentence, but at different indices.

Usage:
    python analyze_uncertainty_vs_judge.py --task sentence --config aci --split test1 --start 0 --end 44
    python analyze_uncertainty_vs_judge.py --task span --config aci --split test1
    python analyze_uncertainty_vs_judge.py --task section-u --config aci --split test1
    python analyze_uncertainty_vs_judge.py --task section-j --config aci --split test1
    python analyze_uncertainty_vs_judge.py --task all --config aci --split test1
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE = Path(__file__).parent

NOT_FAITHFUL_LABELS = {"Fabrication", "Negation", "Causality", "Contextual"}
VALID_LABELS = NOT_FAITHFUL_LABELS | {"Faithful"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def binary_nonfaithful(label: str) -> Optional[int]:
    """Faithful -> 0, a real hallucination label -> 1, PARSE_ERROR/unknown -> None (dropped)."""
    if label == "Faithful":
        return 0
    if label in NOT_FAITHFUL_LABELS:
        return 1
    return None


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """AUROC + AUPRC for a continuous score predicting a binary outcome."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    n = len(y_true)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == n:
        return {"n": n, "n_pos": n_pos, "auroc": float("nan"), "auprc": float("nan"),
               "note": "only one class present -- AUROC/AUPRC undefined"}
    return {
        "n": n,
        "n_pos": n_pos,
        "pos_rate": round(n_pos / n, 4),
        "auroc": round(float(roc_auc_score(y_true, y_score)), 4),
        "auprc": round(float(average_precision_score(y_true, y_score)), 4),
    }


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def load_sentence_uncertainty(sent_dir: Path, sample_idx: int, note_idx: int) -> Optional[pd.DataFrame]:
    p = sent_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_sentences.csv"
    return pd.read_csv(p) if p.exists() else None


def load_span_judge(judge_dir: Path, sample_idx: int, note_idx: int) -> Optional[pd.DataFrame]:
    p = judge_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_span_judge.csv"
    return pd.read_csv(p) if p.exists() else None


def load_facts_matched(matched_dir: Path, sample_idx: int, note_idx: int) -> Optional[pd.DataFrame]:
    p = matched_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts_matched.csv"
    return pd.read_csv(p) if p.exists() else None


def load_note_text(gen_dir: Path, sample_idx: int, note_idx: int) -> Optional[str]:
    p = gen_dir / f"sample_{sample_idx:03d}_generations.json"
    if not p.exists():
        return None
    notes = json.loads(p.read_text()).get("notes", [])
    return notes[note_idx] if note_idx < len(notes) else None


def iter_available(dir_a: Path, dir_b: Path, suffix_a: str, suffix_b: str,
                   start: int, end: int):
    """Yield (sample_idx, note_idx) pairs present in BOTH dir_a and dir_b."""
    for sample_idx in range(start, end):
        note_idx = 0
        while True:
            pa = dir_a / f"sample_{sample_idx:03d}_note_{note_idx:02d}{suffix_a}"
            pb = dir_b / f"sample_{sample_idx:03d}_note_{note_idx:02d}{suffix_b}"
            if not pa.exists() and note_idx == 0:
                break  # no notes at all for this sample
            if not pa.exists():
                break  # ran past the last note for this sample
            if pb.exists():
                yield sample_idx, note_idx
            note_idx += 1


# -----------------------------------------------------------------------------
# Task 1: sentence-level uncertainty vs sentence-level judge
# -----------------------------------------------------------------------------

def task_sentence(sent_dir: Path, judge_dir: Path, start: int, end: int) -> pd.DataFrame:
    rows = []
    for sample_idx, note_idx in iter_available(sent_dir, judge_dir, "_sentences.csv", "_span_judge.csv", start, end):
        u_df = load_sentence_uncertainty(sent_dir, sample_idx, note_idx)
        j_df = load_span_judge(judge_dir, sample_idx, note_idx)

        u_lookup = {_norm(r["sentence"]): r["uncertainty"] for _, r in u_df.iterrows()}
        for _, jr in j_df.iterrows():
            key = _norm(jr["sentence"])
            if key not in u_lookup:
                continue
            y = binary_nonfaithful(jr["label"])
            if y is None:
                continue
            rows.append({
                "sample_idx": sample_idx, "note_idx": note_idx,
                "sentence": jr["sentence"], "label": jr["label"],
                "uncertainty": u_lookup[key], "nonfaithful": y,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[sentence] no matched rows found")
        return df

    metrics = binary_metrics(df["nonfaithful"].to_numpy(), df["uncertainty"].to_numpy())
    print(f"[sentence] matched {len(df)} sentences across "
         f"{df[['sample_idx', 'note_idx']].drop_duplicates().shape[0]} notes")
    print(f"[sentence] AUROC={metrics.get('auroc')}  AUPRC={metrics.get('auprc')}  "
         f"n={metrics['n']}  n_nonfaithful={metrics.get('n_pos')}  "
         f"pos_rate={metrics.get('pos_rate')}")
    print("[sentence] mean uncertainty by label:")
    print(df.groupby("label")["uncertainty"].agg(["mean", "median", "count"]).round(4))
    return df


# -----------------------------------------------------------------------------
# Task 2: atomic-fact uncertainty, provenance-matched to sentence, vs judge
# -----------------------------------------------------------------------------

def task_span(matched_dir: Path, judge_dir: Path, start: int, end: int) -> pd.DataFrame:
    rows = []
    for sample_idx, note_idx in iter_available(matched_dir, judge_dir, "_facts_matched.csv", "_span_judge.csv", start, end):
        m_df = load_facts_matched(matched_dir, sample_idx, note_idx)
        j_df = load_span_judge(judge_dir, sample_idx, note_idx)

        j_lookup = {_norm(r["sentence"]): r for _, r in j_df.iterrows()}
        for _, fr in m_df.iterrows():
            if not fr.get("clean_match", False):
                continue  # unreliable provenance match, don't feed it into validation
            key = _norm(fr["sentence_text"])
            if key not in j_lookup:
                continue
            jr = j_lookup[key]
            y = binary_nonfaithful(jr["label"])
            if y is None:
                continue
            note_span = str(jr.get("note_span", "") or "")
            span_hit = bool(note_span) and any(
                w.lower() in note_span.lower() for w in str(fr["fact"]).split() if len(w) > 3
            )
            rows.append({
                "sample_idx": sample_idx, "note_idx": note_idx,
                "fact": fr["fact"], "uncertainty": fr["uncertainty"],
                "match_sent_idx": fr["match_sent_idx"], "sentence_label": jr["label"],
                "nonfaithful": y, "note_span": note_span, "span_word_overlap": span_hit,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[span] no matched rows found")
        return df

    metrics = binary_metrics(df["nonfaithful"].to_numpy(), df["uncertainty"].to_numpy())
    print(f"[span] matched {len(df)} facts (via provenance) across "
         f"{df[['sample_idx', 'note_idx']].drop_duplicates().shape[0]} notes")
    print(f"[span] AUROC={metrics.get('auroc')}  AUPRC={metrics.get('auprc')}  "
         f"n={metrics['n']}  n_nonfaithful={metrics.get('n_pos')}  "
         f"pos_rate={metrics.get('pos_rate')}")

    # Stricter check: among HIGH-uncertainty facts (>0.5) whose matched
    # sentence was flagged non-faithful WITH a specific note_span, how often
    # does the fact's own text actually overlap that flagged span (vs. just
    # sharing a sentence with it)?
    high_u_nonfaithful = df[(df["uncertainty"] > 0.5) & (df["nonfaithful"] == 1) & (df["note_span"] != "")]
    if len(high_u_nonfaithful):
        hit_rate = high_u_nonfaithful["span_word_overlap"].mean()
        print(f"[span] of {len(high_u_nonfaithful)} high-uncertainty facts (>0.5) matched to a "
             f"non-faithful sentence with a flagged span, {hit_rate:.1%} overlap the exact flagged span")
    return df


def _shares_content_word(fact_text: str, note_span: str) -> bool:
    """True if fact_text and note_span share >=1 non-trivial (len>3) word."""
    fact_words = {w.lower().strip(".,;:") for w in str(fact_text).split() if len(w) > 3}
    span_words = {w.lower().strip(".,;:") for w in str(note_span).split() if len(w) > 3}
    return bool(fact_words & span_words)


def task_span_strict(matched_dir: Path, judge_dir: Path, start: int, end: int) -> pd.DataFrame:
    """Stricter span-level ground truth than task_span(): task_span labels
    EVERY fact in a non-faithful sentence as positive, which is really a
    sentence-level comparison. Here:
      positive : fact overlaps the judge's flagged note_span by >=1
                 non-trivial content word (it IS the hallucinated content).
      negative : fact is matched to a sentence labeled fully Faithful.
      excluded : fact is matched to a non-faithful sentence but does NOT
                 overlap its note_span (same sentence as a hallucination,
                 but not the hallucinated part -- ambiguous under
                 sentence-level judging, would just add noise either way),
                 or the sentence is non-faithful with no note_span given.
    """
    rows = []
    for sample_idx, note_idx in iter_available(matched_dir, judge_dir, "_facts_matched.csv", "_span_judge.csv", start, end):
        m_df = load_facts_matched(matched_dir, sample_idx, note_idx)
        j_df = load_span_judge(judge_dir, sample_idx, note_idx)

        j_lookup = {_norm(r["sentence"]): r for _, r in j_df.iterrows()}
        for _, fr in m_df.iterrows():
            if not fr.get("clean_match", False):
                continue
            key = _norm(fr["sentence_text"])
            if key not in j_lookup:
                continue
            jr = j_lookup[key]
            label = jr["label"]
            note_span = str(jr.get("note_span", "") or "")

            if label == "Faithful":
                y = 0
            elif label in NOT_FAITHFUL_LABELS and note_span and _shares_content_word(fr["fact"], note_span):
                y = 1
            else:
                continue  # excluded -- ambiguous

            rows.append({
                "sample_idx": sample_idx, "note_idx": note_idx,
                "fact": fr["fact"], "uncertainty": fr["uncertainty"],
                "sentence_label": label, "note_span": note_span, "span_level_positive": y,
            })

    return pd.DataFrame(rows)


def task_span_strict_all_datasets(gen_base_dir: Path, judge_base_dir: Path,
                                  start: int, end: int, out_png: Path) -> pd.DataFrame:
    """Runs task_span_strict() across all 6 datasets, prints per-dataset and
    combined AUROC/AUPRC, and plots a stacked bar of positive vs negative
    fact counts per dataset."""
    all_dfs = []
    for config, split in DATASETS:
        matched_dir = (Path("luq_out/llama_atomic/facts_matched") if (config, split) == ("aci", "test1")
                      else Path("luq_out/llama_atomic/facts_matched") / config / split)
        judge_dir = judge_base_dir / config / split / "spans"
        df = task_span_strict(matched_dir, judge_dir, start, end)
        if df.empty:
            print(f"[span-strict] {config}/{split}: no matched rows found")
            continue
        df["dataset"] = f"{config}/{split}"
        all_dfs.append(df)

        n_pos = int(df["span_level_positive"].sum())
        n_neg = len(df) - n_pos
        metrics = binary_metrics(df["span_level_positive"].to_numpy(), df["uncertainty"].to_numpy())
        print(f"[span-strict] {config}/{split}: n={len(df)}  positive={n_pos}  negative={n_neg}  "
             f"AUROC={metrics.get('auroc')}  AUPRC={metrics.get('auprc')}")

    if not all_dfs:
        print("[span-strict] nothing to plot")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    metrics = binary_metrics(combined["span_level_positive"].to_numpy(), combined["uncertainty"].to_numpy())
    print(f"\n[span-strict] COMBINED across all datasets: n={len(combined)}  "
         f"positive={int(combined['span_level_positive'].sum())}  "
         f"negative={len(combined) - int(combined['span_level_positive'].sum())}  "
         f"AUROC={metrics.get('auroc')}  AUPRC={metrics.get('auprc')}")

    counts = combined.groupby(["dataset", "span_level_positive"]).size().unstack(fill_value=0)
    counts = counts.rename(columns={0: "negative (Faithful)", 1: "positive (overlaps flagged span)"})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = counts.index.tolist()
    neg_vals = counts["negative (Faithful)"].to_numpy()
    pos_vals = counts["positive (overlaps flagged span)"].to_numpy()
    ax.bar(labels, neg_vals, label="negative (Faithful)", color="#4C72B0")
    ax.bar(labels, pos_vals, bottom=neg_vals, label="positive (overlaps flagged span)", color="#C44E52")
    ax.set_ylabel("number of facts")
    ax.set_title("Span-level positive vs negative facts by dataset\n"
                 f"(combined AUROC={metrics.get('auroc')}, AUPRC={metrics.get('auprc')})")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[span-strict] saved {out_png}")
    return combined


# -----------------------------------------------------------------------------
# Task 3: uncertainty distribution by SOAP section (bar plot)
# -----------------------------------------------------------------------------

def task_section_uncertainty(facts_dir: Path, start: int, end: int, out_png: Path) -> pd.DataFrame:
    rows = []
    for sample_idx in range(start, end):
        note_idx = 0
        while True:
            p = facts_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts.csv"
            if not p.exists():
                break
            df = pd.read_csv(p)
            rows.append(df[["section", "uncertainty"]])
            note_idx += 1
    if not rows:
        print("[section-u] no facts CSVs found")
        return pd.DataFrame()

    all_df = pd.concat(rows, ignore_index=True)
    stats = all_df.groupby("section")["uncertainty"].agg(["mean", "std", "count"]).round(4)
    print("[section-u] uncertainty by section:")
    print(stats)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=4)
    ax.set_ylabel("uncertainty (mean ± std)")
    ax.set_title("Atomic-fact uncertainty by SOAP section")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[section-u] saved {out_png}")
    return all_df


# -----------------------------------------------------------------------------
# Task 4: judge faithfulness rate by SOAP section (bar plot)
# -----------------------------------------------------------------------------

def task_section_judge(judge_dir: Path, gen_dir: Path, start: int, end: int, out_png: Path) -> pd.DataFrame:
    import luq_sentence as luq

    rows = []
    for sample_idx in range(start, end):
        note_idx = 0
        while True:
            jp = judge_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_span_judge.csv"
            if not jp.exists():
                if note_idx == 0:
                    break
                break
            note_text = load_note_text(gen_dir, sample_idx, note_idx)
            if note_text is None:
                note_idx += 1
                continue
            j_df = pd.read_csv(jp)
            sentences = luq.split_sentences(note_text)
            sent_sections = luq.assign_sentence_sections(note_text, sentences)
            sec_lookup = {_norm(s): sec for s, sec in zip(sentences, sent_sections)}

            for _, jr in j_df.iterrows():
                sec = sec_lookup.get(_norm(jr["sentence"]), "all")
                rows.append({"section": sec, "label": jr["label"],
                            "nonfaithful": binary_nonfaithful(jr["label"])})
            note_idx += 1

    df = pd.DataFrame(rows).dropna(subset=["nonfaithful"])
    if df.empty:
        print("[section-j] no judge rows found")
        return df

    stats = df.groupby("section")["nonfaithful"].agg(["mean", "count"]).round(4)
    stats = stats.rename(columns={"mean": "nonfaithful_rate"})
    print("[section-j] non-faithful rate by section:")
    print(stats)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats.index, stats["nonfaithful_rate"])
    ax.set_ylabel("non-faithful rate (LLM-judge)")
    ax.set_title("LLM-judge non-faithful rate by SOAP section")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[section-j] saved {out_png}")
    return df


# -----------------------------------------------------------------------------
# Cross-dataset comparison: per-note fact-level uncertainty, and per-note
# uncertainty vs judge, both as box plots across the 6 config/split datasets.
# -----------------------------------------------------------------------------

DATASETS: List[Tuple[str, str]] = [
    ("aci", "test1"), ("aci", "test2"), ("aci", "test3"),
    ("virtscribe", "test1"), ("virtscribe", "test2"), ("virtscribe", "test3"),
]


def per_note_fact_uncertainty(facts_dir: Path, start: int, end: int) -> List[float]:
    """One value per note: mean fact-level uncertainty across that note's facts."""
    vals = []
    for sample_idx in range(start, end):
        note_idx = 0
        while True:
            p = facts_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts.csv"
            if not p.exists():
                break
            df = pd.read_csv(p)
            if len(df):
                vals.append(float(df["uncertainty"].mean()))
            note_idx += 1
    return vals


def per_note_judge_nonfaithful_rate(judge_dir: Path, start: int, end: int) -> List[float]:
    """One value per note: fraction of judged sentences labeled non-Faithful."""
    vals = []
    for sample_idx in range(start, end):
        note_idx = 0
        while True:
            p = judge_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_span_judge.csv"
            if not p.exists():
                break
            df = pd.read_csv(p)
            labels = df["label"].map(binary_nonfaithful).dropna()
            if len(labels):
                vals.append(float(labels.mean()))
            note_idx += 1
    return vals


def task_per_note_uncertainty_boxplot(base_dir: Path, judge_base_dir: Path,
                                       start: int, end: int, out_png: Path) -> Dict[str, List[float]]:
    """Per-note mean fact-level uncertainty, one box per dataset, compared
    across all 6 config/split combinations."""
    data: Dict[str, List[float]] = {}
    for config, split in DATASETS:
        facts_dir = base_dir / config / split / "facts"
        vals = per_note_fact_uncertainty(facts_dir, start, end)
        label = f"{config}/{split}"
        data[label] = vals
        if vals:
            print(f"[per-note-u] {label}: n_notes={len(vals)}  "
                 f"mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
        else:
            print(f"[per-note-u] {label}: no facts data found")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [k for k, v in data.items() if v]
    values = [data[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(values, tick_labels=labels, showmeans=True, meanline=True)
    ax.set_ylabel("per-note mean fact-level uncertainty")
    ax.set_title("Per-note fact-level uncertainty across datasets\n(box=IQR, dashed line=mean, whiskers=1.5xIQR)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[per-note-u] saved {out_png}")
    return data


def task_compare_uncertainty_vs_judge_boxplot(base_dir: Path, judge_base_dir: Path,
                                              start: int, end: int, out_png: Path) -> None:
    """Single plot: for each dataset, two side-by-side boxes -- per-note mean
    fact-level uncertainty vs per-note LLM-judge non-faithful rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unc_vals, judge_vals, labels = [], [], []
    for config, split in DATASETS:
        facts_dir = base_dir / config / split / "facts"
        judge_dir = judge_base_dir / config / split / "spans"
        u = per_note_fact_uncertainty(facts_dir, start, end)
        j = per_note_judge_nonfaithful_rate(judge_dir, start, end)
        if not u and not j:
            continue
        unc_vals.append(u)
        judge_vals.append(j)
        labels.append(f"{config}/{split}")
        print(f"[compare] {config}/{split}: uncertainty n={len(u)} mean={np.mean(u) if u else float('nan'):.4f}  "
             f"judge n={len(j)} mean={np.mean(j) if j else float('nan'):.4f}")

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 5))
    group_width = 2.0
    positions_u = [i * group_width for i in range(n)]
    positions_j = [i * group_width + 0.7 for i in range(n)]

    bp_u = ax.boxplot(unc_vals, positions=positions_u, widths=0.55,
                      patch_artist=True, showmeans=True, meanline=True)
    bp_j = ax.boxplot(judge_vals, positions=positions_j, widths=0.55,
                      patch_artist=True, showmeans=True, meanline=True)
    for patch in bp_u["boxes"]:
        patch.set_facecolor("#4C72B0")
    for patch in bp_j["boxes"]:
        patch.set_facecolor("#DD8452")

    ax.set_xticks([p + 0.35 for p in positions_u])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("per-note score")
    ax.set_title("Per-note uncertainty vs LLM-judge non-faithful rate, by dataset")
    ax.legend([bp_u["boxes"][0], bp_j["boxes"][0]],
             ["fact-level uncertainty (mean per note)", "LLM-judge non-faithful rate (per note)"],
             loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[compare] saved {out_png}")


# -----------------------------------------------------------------------------
# Bucket sentence-level uncertainty scores and report the count distribution.
# -----------------------------------------------------------------------------

def task_bucket_sentence_uncertainty(gen_base_dir: Path, start: int, end: int,
                                     out_dir: Path, n_buckets: int = 10) -> pd.DataFrame:
    """Every sentence-level uncertainty score across all 6 datasets, bucketed
    into n_buckets equal-width bins over [0, 1], with a count distribution
    table + bar plot."""
    rows = []
    for config, split in DATASETS:
        sent_dir = gen_base_dir / config / split / "sentences"
        if not sent_dir.exists():
            continue
        for p in sorted(sent_dir.glob("sample_*_sentences.csv")):
            df = pd.read_csv(p)
            for u in df["uncertainty"]:
                rows.append({"dataset": f"{config}/{split}", "uncertainty": u})

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        print("[bucket-u] no sentence uncertainty data found")
        return all_df

    bins = np.linspace(0, 1, n_buckets + 1)
    bucket_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(n_buckets)]
    all_df["bucket"] = pd.cut(all_df["uncertainty"], bins=bins, labels=bucket_labels, include_lowest=True)

    counts = all_df["bucket"].value_counts().reindex(bucket_labels, fill_value=0)
    print(f"[bucket-u] sentence uncertainty count distribution (all datasets combined, n={len(all_df)}):")
    print(counts.to_string())

    counts_csv = out_dir / "sentence_uncertainty_bucket_counts.csv"
    counts.rename("count").rename_axis("bucket").reset_index().to_csv(counts_csv, index=False)
    print(f"[bucket-u] saved {counts_csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bucket_labels, counts.values)
    ax.set_xlabel("uncertainty bucket")
    ax.set_ylabel("number of sentences")
    ax.set_title(f"Sentence-level uncertainty distribution (n={len(all_df)}, all 6 datasets)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out_png = out_dir / "sentence_uncertainty_bucket_counts.png"
    fig.savefig(out_png, dpi=150)
    print(f"[bucket-u] saved {out_png}")
    return all_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True,
                   choices=["sentence", "span", "span-strict", "section-u", "section-j", "all",
                           "per-note-u", "compare-box", "bucket-u"])
    p.add_argument("--n-buckets", type=int, default=10, help="Number of uncertainty buckets for --task bucket-u")
    p.add_argument("--config", default="aci")
    p.add_argument("--split", default="test1")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=132)
    p.add_argument("--sent-dir", default=None, help="Override sentence-uncertainty dir")
    p.add_argument("--judge-dir", default=None, help="Override span-judge dir")
    p.add_argument("--facts-dir", default=None, help="Override atomic facts dir")
    p.add_argument("--matched-dir", default="luq_out/llama_atomic/facts_matched",
                   help="Override provenance-matched facts dir")
    p.add_argument("--gen-dir", default=None, help="Override generations dir")
    p.add_argument("--gen-base-dir", default="luq_out/llama/generations",
                   help="Parent dir containing <config>/<split>/facts -- used by "
                        "per-note-u/compare-box, which loop over all 6 datasets")
    p.add_argument("--judge-base-dir", default="luq_out/llama_judge",
                   help="Parent dir containing <config>/<split>/spans -- used by compare-box")
    p.add_argument("--out-dir", default="luq_out/analysis", help="Where to save plots/CSVs")
    return p.parse_args()


def main():
    args = parse_args()
    combo = f"{args.config}/{args.split}"
    sent_dir = Path(args.sent_dir or f"luq_out/llama/generations/{combo}/sentences")
    judge_dir = Path(args.judge_dir or f"luq_out/llama_judge/{combo}/spans")
    facts_dir = Path(args.facts_dir or f"luq_out/llama/generations/{combo}/facts")
    matched_dir = Path(args.matched_dir)
    gen_dir = Path(args.gen_dir or f"luq_out/llama/generations/{combo}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task in {"sentence", "all"}:
        df = task_sentence(sent_dir, judge_dir, args.start, args.end)
        if not df.empty:
            df.to_csv(out_dir / f"sentence_vs_judge_{args.config}_{args.split}.csv", index=False)

    if args.task in {"span", "all"}:
        df = task_span(matched_dir, judge_dir, args.start, args.end)
        if not df.empty:
            df.to_csv(out_dir / f"span_provenance_vs_judge_{args.config}_{args.split}.csv", index=False)

    if args.task in {"section-u", "all"}:
        task_section_uncertainty(facts_dir, args.start, args.end,
                                 out_dir / f"section_uncertainty_{args.config}_{args.split}.png")

    if args.task in {"section-j", "all"}:
        task_section_judge(judge_dir, gen_dir, args.start, args.end,
                           out_dir / f"section_judge_{args.config}_{args.split}.png")

    if args.task == "per-note-u":
        task_per_note_uncertainty_boxplot(Path(args.gen_base_dir), Path(args.judge_base_dir),
                                          args.start, args.end,
                                          out_dir / "per_note_uncertainty_by_dataset.png")

    if args.task == "compare-box":
        task_compare_uncertainty_vs_judge_boxplot(Path(args.gen_base_dir), Path(args.judge_base_dir),
                                                  args.start, args.end,
                                                  out_dir / "uncertainty_vs_judge_by_dataset.png")

    if args.task == "span-strict":
        df = task_span_strict_all_datasets(Path(args.gen_base_dir), Path(args.judge_base_dir),
                                           args.start, args.end,
                                           out_dir / "span_strict_positive_vs_negative.png")
        if not df.empty:
            df.to_csv(out_dir / "span_strict_all_datasets.csv", index=False)

    if args.task == "bucket-u":
        task_bucket_sentence_uncertainty(Path(args.gen_base_dir), args.start, args.end,
                                         out_dir, n_buckets=args.n_buckets)


if __name__ == "__main__":
    main()
