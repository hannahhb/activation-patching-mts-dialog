"""
analyze_human_agreement.py
===========================
Resolves the human faithfulness signal across the 3 annotators (hannah,
rashika_bahl, daniel):

  1. Human-human inter-annotator agreement (Cohen's kappa, pairwise) on
     binarized Faithful(0)/Not-Faithful(1) -- the ceiling any automated
     method (LLM-judge, uncertainty) is being compared against.
  2. A per-sentence majority-vote consensus label, with a breakdown of how
     each sentence's consensus was actually reached (unanimous / resolved by
     majority / needed the adjudicated consensus file / unresolved tie /
     only one annotator available).

Header/preamble sentences ("Subjective:", "Here is the SOAP note based on
the transcript:", etc.) are excluded throughout, via
fact_sentence_match._is_header_sentence.

Output: analysis/human_consensus_sentence_labels.csv
  sample_idx, note_idx, sentence_idx, sentence,
  hannah_faithful, rashika_bahl_faithful, daniel_faithful   (binary 0/1, blank if that annotator didn't label this sentence)
  n_annotators, majority_binary, majority_type, resolved_by

Usage:
    python analyze_human_agreement.py
"""

import sys
from collections import Counter
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import annotator as ann
from fact_sentence_match import _is_header_sentence
from analyze_uncertainty_vs_judge import (
    _norm, binary_nonfaithful, binary_metrics, load_span_judge, load_sentence_uncertainty,
    task_sentence, DATASETS,
)

OUT_DIR = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)

# Human annotations only exist for aci/test1 (annotator.py's hardcoded GEN_DIR),
# so the judge/uncertainty data compared against the human majority label must
# come from the same combo.
JUDGE_DIR = Path("luq_out/llama_judge/aci/test1/spans")
SENT_DIR = Path("luq_out/llama/generations/aci/test1/sentences")


def binarize(faithful_str):
    if faithful_str == "Faithful":
        return 0
    if faithful_str == "Not Faithful":
        return 1
    return None


def main():
    combos = ann.annotated_sample_notes()
    print(f"{len(combos)} (sample, note) pairs have >=1 annotator")

    rows = []
    pair_keys = [("hannah", "rashika_bahl"), ("hannah", "daniel"), ("rashika_bahl", "daniel")]
    pair_votes = {k: [] for k in pair_keys}
    reliability_columns = []  # one [hannah, rashika_bahl, daniel] column per sentence, NaN if unlabeled
    resolved_counts = Counter()
    n_header_excluded = 0

    for sid, nid in combos:
        transcript, note, n_notes = ann.load_generation(sid, nid)
        if note is None:
            continue
        sentences = ann.load_sentences(sid, nid, note)

        per_annotator = {}
        for name in ann.ANNOTATORS:
            if ann.annot_path(sid, nid, name).exists():
                per_annotator[name] = ann.load_annotations(sid, nid, name).get("sentence_labels", {})

        consensus_labels = ann.load_annotations(sid, nid, ann.CONSENSUS_NAME).get("sentence_labels", {})

        all_idxs = set()
        for labels in per_annotator.values():
            all_idxs.update(int(k) for k in labels.keys())

        for idx in sorted(all_idxs):
            if idx >= len(sentences):
                continue
            sent_text = sentences[idx]
            if _is_header_sentence(sent_text):
                n_header_excluded += 1
                continue

            key = str(idx)
            votes, types = {}, {}
            for name in ann.ANNOTATORS:
                if name in per_annotator and key in per_annotator[name]:
                    b = binarize(per_annotator[name][key].get("faithful"))
                    if b is not None:
                        votes[name] = b
                        if b == 1:
                            types[name] = per_annotator[name][key].get("type")

            if not votes:
                continue

            for a, b in pair_keys:
                if a in votes and b in votes:
                    pair_votes[(a, b)].append((votes[a], votes[b]))

            # one column per sentence, one row per annotator, NaN if that
            # annotator didn't label this sentence -- krippendorff.alpha
            # handles the missing entries natively.
            reliability_columns.append([votes.get(name, np.nan) for name in ann.ANNOTATORS])

            n = len(votes)
            binary_vals = list(votes.values())
            if n >= 2:
                counts = Counter(binary_vals)
                top_val, top_n = counts.most_common(1)[0]
                if top_n * 2 > n:
                    majority = top_val
                    resolved_by = "unanimous" if top_n == n else "majority"
                elif key in consensus_labels:
                    majority = binarize(consensus_labels[key].get("faithful"))
                    resolved_by = "consensus_file"
                else:
                    majority = None
                    resolved_by = "unresolved_tie"
            else:
                majority = binary_vals[0]
                resolved_by = "single_annotator"

            resolved_counts[resolved_by] += 1

            majority_type = None
            if majority == 1:
                contributing = [types[name] for name in votes if votes[name] == 1 and name in types]
                if contributing:
                    majority_type = Counter(contributing).most_common(1)[0][0]

            rows.append({
                "sample_idx": sid, "note_idx": nid, "sentence_idx": idx, "sentence": sent_text,
                "hannah_faithful": votes.get("hannah"),
                "rashika_bahl_faithful": votes.get("rashika_bahl"),
                "daniel_faithful": votes.get("daniel"),
                "n_annotators": n,
                "majority_binary": majority,
                "majority_type": majority_type,
                "resolved_by": resolved_by,
            })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "human_consensus_sentence_labels.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nexcluded {n_header_excluded} header/preamble sentences")
    print(f"saved {out_csv} ({len(df)} sentences)")

    total = sum(resolved_counts.values())
    print(f"\nconsensus resolution breakdown (n={total}):")
    for k in ["unanimous", "majority", "consensus_file", "unresolved_tie", "single_annotator"]:
        v = resolved_counts.get(k, 0)
        print(f"  {k:<18} {v:>4}  ({v / total:.1%})")

    print("\nhuman-human pairwise Cohen's kappa (ceiling reference):")
    for (a, b), pairs in pair_votes.items():
        if len(pairs) < 2:
            print(f"  {a} vs {b}: n={len(pairs)} -- too few overlapping labels")
            continue
        y1 = [p[0] for p in pairs]
        y2 = [p[1] for p in pairs]
        kappa = cohen_kappa_score(y1, y2)
        agree = sum(1 for x, y in pairs if x == y) / len(pairs)
        print(f"  {a} vs {b}: n={len(pairs)}  kappa={kappa:.3f}  raw_agreement={agree:.1%}")

    avg_pairwise_kappa = np.mean([
        cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs])
        for pairs in pair_votes.values() if len(pairs) >= 2
    ])
    print(f"\n  average pairwise kappa (\"Light's kappa\"): {avg_pairwise_kappa:.3f}  (crude, quick cross-check only)")

    # Genuine 3-way agreement: Krippendorff's alpha handles the uneven
    # coverage (22 sentences with all 3 annotators, 3 with only 2) in one
    # calculation, unlike Cohen's kappa (strictly pairwise) or Fleiss' kappa
    # (assumes a fixed rater count per item).
    reliability_data = np.array(reliability_columns, dtype=float).T  # (n_raters, n_items)
    alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="nominal")
    print(f"\n3-way human agreement, Krippendorff's alpha (nominal, handles missing raters): "
         f"{alpha:.3f}  (n_items={reliability_data.shape[1]}, n_raters={reliability_data.shape[0]})")

    fleiss = fleiss_kappa_fixed_raters(np.array(reliability_columns, dtype=float))
    if fleiss is not None:
        kappa, n_used, n_total = fleiss
        print(f"3-way human agreement, Fleiss' kappa (n=3 raters only, "
             f"{n_used}/{n_total} sentences with all 3 present): {kappa:.3f}")

    three_way_comparison(df)


def fleiss_kappa_fixed_raters(reliability_columns: np.ndarray, n_categories: int = 2):
    """Standard Fleiss' kappa requires a FIXED number of raters per item --
    unlike Krippendorff's alpha, it has no native way to handle the 3
    sentences in our data with only 2 annotators. Restrict to the subset
    with all 3 raters present (drops the incomplete rows) rather than
    misapplying the fixed-n formula to mixed-coverage data."""
    full = reliability_columns[~np.isnan(reliability_columns).any(axis=1)]
    if len(full) == 0:
        return None

    counts = np.zeros((len(full), n_categories))
    for i, row in enumerate(full):
        for val in row:
            counts[i, int(val)] += 1

    N, n = len(full), full.shape[1]
    p_j = counts.sum(axis=0) / (N * n)
    P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e_bar = np.sum(p_j ** 2)
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa, len(full), len(reliability_columns)


def three_way_comparison(human_df: pd.DataFrame) -> pd.DataFrame:
    """Human (majority vote) vs Judge vs Uncertainty, pairwise, all three
    matched to the SAME sentences (the ones with a resolved human majority
    label). Unresolved ties (majority_binary is NaN) are dropped -- there's
    no human ground truth to compare against for those.
    """
    resolved = human_df.dropna(subset=["majority_binary"]).copy()
    resolved["majority_binary"] = resolved["majority_binary"].astype(int)

    rows = []
    for (sid, nid), group in resolved.groupby(["sample_idx", "note_idx"]):
        j_df = load_span_judge(JUDGE_DIR, sid, nid)
        u_df = load_sentence_uncertainty(SENT_DIR, sid, nid)
        if j_df is None or u_df is None:
            continue
        j_lookup = {_norm(r["sentence"]): r["label"] for _, r in j_df.iterrows()}
        u_lookup = {_norm(r["sentence"]): r["uncertainty"] for _, r in u_df.iterrows()}

        for _, hr in group.iterrows():
            key = _norm(hr["sentence"])
            if key not in j_lookup or key not in u_lookup:
                continue
            judge_label = j_lookup[key]
            judge_bin = binary_nonfaithful(judge_label)
            if judge_bin is None:
                continue  # PARSE_ERROR
            rows.append({
                "sample_idx": sid, "note_idx": nid, "sentence": hr["sentence"],
                "human_majority": hr["majority_binary"],
                "judge_label": judge_label, "judge_nonfaithful": judge_bin,
                "uncertainty": u_lookup[key],
            })

    merged = pd.DataFrame(rows)
    if merged.empty:
        print("\n[3-way] no sentences matched across human + judge + uncertainty")
        return merged

    merged.to_csv(OUT_DIR / "three_way_matched_sentences.csv", index=False)
    print(f"\n[3-way] matched {len(merged)} sentences with human + judge + uncertainty all present")

    print("\n[3-way] Human (majority) vs Judge -- Cohen's kappa + raw agreement:")
    kappa = cohen_kappa_score(merged["human_majority"], merged["judge_nonfaithful"])
    agree = (merged["human_majority"] == merged["judge_nonfaithful"]).mean()
    print(f"  n={len(merged)}  kappa={kappa:.3f}  raw_agreement={agree:.1%}")

    # Judge's binary decision (0/1, not continuous) still works as a "score"
    # for roc_auc_score/average_precision_score -- with a single hard
    # decision point the ROC curve degenerates to one point, so this AUROC
    # is mathematically just balanced accuracy. Computing it anyway puts
    # Judge on the same AUROC scale as Uncertainty below, for a direct
    # side-by-side comparison against the same human ground truth.
    print("\n[3-way] Human (majority) vs Judge -- AUROC/AUPRC (judge's binary decision as a 'score'):")
    m = binary_metrics(merged["human_majority"].to_numpy(), merged["judge_nonfaithful"].to_numpy())
    print(f"  n={m['n']}  n_nonfaithful={m.get('n_pos')}  pos_rate={m.get('pos_rate')}  "
         f"AUROC={m.get('auroc')}  AUPRC={m.get('auprc')}")

    print("\n[3-way] Human (majority) vs Uncertainty -- AUROC/AUPRC:")
    m = binary_metrics(merged["human_majority"].to_numpy(), merged["uncertainty"].to_numpy())
    print(f"  n={m['n']}  n_nonfaithful={m.get('n_pos')}  pos_rate={m.get('pos_rate')}  "
         f"AUROC={m.get('auroc')}  AUPRC={m.get('auprc')}")

    print("\n[3-way] Judge vs Uncertainty -- AUROC/AUPRC (same matched subset, for apples-to-apples):")
    m = binary_metrics(merged["judge_nonfaithful"].to_numpy(), merged["uncertainty"].to_numpy())
    print(f"  n={m['n']}  n_nonfaithful={m.get('n_pos')}  pos_rate={m.get('pos_rate')}  "
         f"AUROC={m.get('auroc')}  AUPRC={m.get('auprc')}")

    judge_all = build_full_judge_uncertainty_data()
    plot_faithfulness_vs_uncertainty(merged, judge_all, OUT_DIR / "scatter_faithfulness_vs_uncertainty.png")
    plot_violin_faithfulness_vs_uncertainty(merged, judge_all, OUT_DIR / "violin_faithfulness_vs_uncertainty.png")
    plot_uncertainty_by_error_type(judge_all, OUT_DIR / "uncertainty_by_error_type.png")
    return merged


def build_full_judge_uncertainty_data() -> pd.DataFrame:
    """Judge vs Uncertainty is NOT bottlenecked by human-annotation
    availability (unlike the Human panel, which only exists for aci/test1's
    annotated subset) -- so use every sentence across all 6 datasets where a
    judge label + uncertainty score are both available, via the same
    task_sentence() matching already used in analyze_uncertainty_vs_judge.py."""
    all_dfs = []
    for config, split in DATASETS:
        sent_dir = Path(f"luq_out/llama/generations/{config}/{split}/sentences")
        judge_dir = Path(f"luq_out/llama_judge/{config}/{split}/spans")
        df = task_sentence(sent_dir, judge_dir, 0, 132)
        if not df.empty:
            all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    print(f"\n[3-way] judge-vs-uncertainty panel using ALL 6 datasets: n={len(combined)} "
         f"(vs 511 if restricted to the human-annotated subset)")
    return combined


CREOLA_ERROR_TYPES = ["Fabrication", "Negation", "Causality", "Contextual"]


def plot_uncertainty_by_error_type(judge_df: pd.DataFrame, out_png: Path) -> pd.DataFrame:
    """One violin per CREOLA label (Faithful + the 4 error types), ordered by
    AUROC (each error type vs Faithful) so the "easier to catch" progression
    reads left to right. Each error-type violin is annotated with its AUROC
    and n, so the plot carries the same information as the summary table
    plus the full distribution shape (which the table can't show -- e.g.
    Fabrication's weak AUROC comes from a distribution that barely shifts
    off the Faithful baseline, not from a shifted-but-noisy one)."""
    aurocs = {}
    for t in CREOLA_ERROR_TYPES:
        sub = judge_df[judge_df["label"].isin([t, "Faithful"])]
        y = (sub["label"] == t).astype(int)
        if y.sum() < 3 or y.sum() == len(y):
            continue
        aurocs[t] = roc_auc_score(y, sub["uncertainty"])

    order = ["Faithful"] + sorted(aurocs, key=aurocs.get)
    plot_df = judge_df[judge_df["label"].isin(order)].copy()
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=order, ordered=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    palette = {"Faithful": "#4C72B0", "Fabrication": "#DD8452", "Contextual": "#C44E52",
              "Negation": "#937860", "Causality": "#8C4C4C"}
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=plot_df, x="label", y="uncertainty", ax=ax, order=order,
                   hue="label", legend=False, palette=palette, inner="quartile", cut=0, bw_adjust=0.6)

    means = plot_df.groupby("label", observed=True)["uncertainty"].mean()
    ns = plot_df["label"].value_counts()
    for i, cat in enumerate(order):
        ax.scatter(i, means[cat], color="black", marker="D", s=70, zorder=5)
        label_text = f"n={ns[cat]}"
        if cat != "Faithful":
            label_text += f"\nAUROC={aurocs[cat]:.3f}"
        ax.text(i, 1.05, label_text, ha="center", va="bottom", fontsize=9)

    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("")
    ax.set_ylabel("sentence-level uncertainty")
    ax.set_title("Uncertainty distribution by CREOLA error type\n"
                "(ordered by AUROC vs Faithful, judge labels, all 6 datasets)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\n[error-type] saved {out_png}")
    return plot_df


def plot_faithfulness_vs_uncertainty(human_df: pd.DataFrame, judge_df: pd.DataFrame, out_png: Path) -> None:
    """Jittered strip plot (a raw scatter would just be two overlapping
    vertical lines, since faithfulness is binary) of sentence-level
    uncertainty against Human and Judge faithfulness side by side.
    Human panel is restricted to the human-annotated subset (that's all that
    exists); Judge panel uses the full cross-dataset judge_df, since it isn't
    bottlenecked by annotation availability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    panels = [
        (axes[0], human_df, "human_majority", f"Human (majority) faithfulness (n={len(human_df)})"),
        (axes[1], judge_df, "nonfaithful", f"LLM-judge faithfulness (n={len(judge_df)})"),
    ]
    for i, (ax, data, col, title) in enumerate(panels):
        sns.stripplot(data=data, x=col, y="uncertainty", ax=ax, jitter=0.25, alpha=0.35, size=3,
                     palette={0: "#4C72B0", 1: "#C44E52"}, hue=col, legend=False)
        means = data.groupby(col)["uncertainty"].mean()
        ax.scatter(means.index, means.values, color="black", marker="D", s=80, zorder=5, label="mean")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Faithful", "Not Faithful"])
        ax.set_xlabel(title)
        ax.set_ylabel("sentence-level uncertainty" if i == 0 else "")
        ax.legend(loc="upper left")

    fig.suptitle("Sentence-level uncertainty vs faithfulness\n"
                "(Human: aci/test1 annotated subset; Judge: all 6 datasets)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\n[3-way] saved {out_png}")


def plot_violin_faithfulness_vs_uncertainty(human_df: pd.DataFrame, judge_df: pd.DataFrame, out_png: Path,
                                            add_title: bool = True) -> None:
    """Violin plot version: shows the full distribution shape of uncertainty
    within each faithfulness group, not just point density.
    Human panel uses inner='stick' (n=511 -- individual observations are
    still readable as tick marks, showing the discrete quantization bands
    from the K-way vote). Judge panel uses the full ~10k-row cross-dataset
    data, where individual sticks would just merge into solid smears -- the
    quantization bands show up anyway as bumps in the KDE shape at that
    volume, so inner='quartile' (a cleaner summary) is used instead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    panels = [
        (axes[0], human_df, "human_majority", f"Human (majority) faithfulness (n={len(human_df)})", "stick"),
        (axes[1], judge_df, "nonfaithful", f"LLM-judge faithfulness (n={len(judge_df)})", "quartile"),
    ]
    for i, (ax, data, col, title, inner) in enumerate(panels):
        sns.violinplot(data=data, x=col, y="uncertainty", ax=ax, hue=col, legend=False,
                       palette={0: "#4C72B0", 1: "#C44E52"}, inner=inner, cut=0, bw_adjust=0.6)
        means = data.groupby(col)["uncertainty"].mean()
        ax.scatter(means.index, means.values, color="black", marker="D", s=80, zorder=5, label="mean")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Faithful", "Not Faithful"])
        ax.set_xlabel(title)
        ax.set_ylabel("sentence-level uncertainty" if i == 0 else "")
        ax.legend(loc="upper left")

    if add_title:
        fig.suptitle("Sentence-level uncertainty distribution by faithfulness\n"
                    "(Human: aci/test1 annotated subset; Judge: all 6 datasets)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[3-way] saved {out_png}")


if __name__ == "__main__":
    main()
