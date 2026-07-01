"""
iaa_sentence.py
===============
Inter-annotator agreement between human annotators and the LLM-as-a-judge
at the sentence level.

Sources:
  Human annotations : annotations/sample_NNN_note_00_<annotator>.json
  LLM judge output  : luq_out/llama_judge/sentences/sample_NNN_note_00_sentence_judge.csv

Sentence matching strategy:
  Both sources derive sentences from the same note text.  The human annotator
  (annotator.py split_sentences) includes bare section-header lines
  ("Subjective:", "Objective:", etc.) as separate sentences.  The LLM judge
  may also emit spurious header-only lines.  Any sentence whose full text
  ends with ":" (nothing after the colon) is treated as a structural header
  and stripped before matching.  Remaining sentences are aligned sequentially.

Metrics reported (binary Faithful / Not-Faithful AND 5-class):
  - Cohen's Kappa   — pairwise between every annotator pair
  - Krippendorff's Alpha — across all available raters per sample
  - % agreement    — raw observed agreement
  - Confusion matrix (human-1 vs LLM, collapsed to 2 classes)

Usage:
    python iaa_sentence.py
    python iaa_sentence.py --annot-dir annotations \\
                           --judge-dir luq_out/llama_judge/sentences
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_ANNOT_DIR = "annotations"
DEFAULT_JUDGE_DIR = "luq_out/llama_judge/sentences"
DEFAULT_GEN_DIR   = "luq_out/llama/generations"

LABEL5  = ["Faithful", "Fabrication", "Negation", "Causality", "Contextual"]
LABEL2  = ["Faithful", "Not Faithful"]

# ── Sentence utilities ────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """Replicates annotator.py split_sentences exactly."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sentences = []
    for line in lines:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def is_header(s: str) -> bool:
    """True if sentence is a bare section header (no clinical content)."""
    return s.strip().endswith(":")


def strip_headers(sentences: List[str]) -> List[Tuple[int, str]]:
    """Return (original_idx, sentence) for non-header sentences."""
    return [(i, s) for i, s in enumerate(sentences) if not is_header(s)]


# ── Label extraction ──────────────────────────────────────────────────────────

def human_label5(entry: dict) -> str:
    """Extract 5-class label from a human annotation sentence entry."""
    if not isinstance(entry, dict):
        entry = {"faithful": entry, "type": None}
    faithful = entry.get("faithful", "")
    if faithful == "Faithful":
        return "Faithful"
    error_type = entry.get("type")
    if error_type in {"Fabrication", "Negation", "Causality", "Contextual"}:
        return error_type
    return "Not Faithful"   # Not Faithful but type unset


def human_label2(entry: dict) -> str:
    return "Faithful" if human_label5(entry) == "Faithful" else "Not Faithful"


def llm_label5(label: str) -> str:
    return label if label in LABEL5 else "Not Faithful"


def llm_label2(label: str) -> str:
    return "Faithful" if llm_label5(label) == "Faithful" else "Not Faithful"


# ── Per-sample alignment ──────────────────────────────────────────────────────

def load_human(path: Path) -> Optional[Dict[int, dict]]:
    """Load human annotation sentence labels as {original_idx: entry}."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.get("sentence_labels", {}).items()}


def load_judge(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def align_sample(sample_idx: int, note: str,
                 human_labels: Dict[str, Dict[int, dict]],
                 judge_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Align human and judge labels for one sample.
    Returns (aligned_df, mismatch_messages).

    aligned_df columns: sentence_idx (annotator), sentence_text,
                        + one column per rater with their 5-class label.
    """
    # Annotator sentence list (with original indices)
    all_sents   = split_sentences(note)
    content     = strip_headers(all_sents)           # [(orig_idx, text), ...]

    # Judge sentences after stripping judge-side headers
    judge_content = []
    if judge_df is not None:
        for _, row in judge_df.iterrows():
            s = str(row["sentence"]).strip()
            if not is_header(s):
                judge_content.append((int(row["sentence_idx"]), s, row["label"]))

    mismatches = []
    n_human  = len(content)
    n_judge  = len(judge_content)

    if n_human != n_judge:
        mismatches.append(
            f"sample_{sample_idx:03d}: annotator {n_human} sentences vs "
            f"judge {n_judge} after header stripping"
        )

    rows = []
    n_match = min(n_human, n_judge)

    for pos in range(n_match):
        orig_idx, annot_text = content[pos]
        _, judge_text, judge_lbl = judge_content[pos]

        row = {
            "sentence_idx":  orig_idx,
            "sentence_text": annot_text,
            "judge":         llm_label5(judge_lbl),
        }
        for rater, labels in human_labels.items():
            entry = labels.get(orig_idx)
            row[rater] = human_label5(entry) if entry else None

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, mismatches


# ── IAA computation ───────────────────────────────────────────────────────────

def kappa(a: List[str], b: List[str]) -> float:
    """Cohen's kappa, returns nan if not computable."""
    paired = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(paired) < 2:
        return float("nan")
    xs, ys = zip(*paired)
    try:
        return cohen_kappa_score(xs, ys)
    except Exception:
        return float("nan")


def pct_agree(a: List[str], b: List[str]) -> float:
    paired = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not paired:
        return float("nan")
    return sum(x == y for x, y in paired) / len(paired)


def kripp_alpha(ratings: List[List[Optional[str]]], level_of_measurement: str = "nominal") -> float:
    """Krippendorff's alpha over a list of rater label sequences (with Nones for missing)."""
    label_map = {l: i for i, l in enumerate(LABEL5 + ["Not Faithful"])}
    # Pad all rater sequences to the same length
    max_len = max(len(r) for r in ratings)
    coded = []
    for rater_labels in ratings:
        row = [label_map.get(l, np.nan) if l is not None else np.nan
               for l in rater_labels]
        row += [np.nan] * (max_len - len(row))
        coded.append(row)
    arr = np.array(coded, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    try:
        return krippendorff.alpha(reliability_data=arr,
                                  level_of_measurement=level_of_measurement)
    except Exception:
        return float("nan")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sentence-level IAA: human vs LLM judge")
    p.add_argument("--annot-dir", default=DEFAULT_ANNOT_DIR)
    p.add_argument("--judge-dir", default=DEFAULT_JUDGE_DIR)
    p.add_argument("--gen-dir",   default=DEFAULT_GEN_DIR)
    p.add_argument("--out",       default="luq_out/iaa_sentence.csv")
    return p.parse_args()


def main():
    args      = parse_args()
    annot_dir = Path(args.annot_dir)
    judge_dir = Path(args.judge_dir)
    gen_dir   = Path(args.gen_dir)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover all annotated samples and their raters
    sample_raters: Dict[int, Dict[str, Path]] = {}
    for p in sorted(annot_dir.glob("sample_*_note_00_*.json")):
        parts = p.stem.split("_note_00_")
        sid   = int(parts[0].replace("sample_", ""))
        rater = parts[1]
        sample_raters.setdefault(sid, {})[rater] = p

    print(f"Samples with annotations : {len(sample_raters)}")
    print(f"Raters found             : {sorted({r for v in sample_raters.values() for r in v})}")
    print()

    all_mismatches = []
    per_sample_rows = []

    # Collect pooled label vectors for global IAA
    pool: Dict[str, List[Optional[str]]] = {}   # rater → flat labels

    for sid in sorted(sample_raters):
        gen_path = gen_dir / f"sample_{sid:03d}_generations.json"
        if not gen_path.exists():
            continue
        note = json.loads(gen_path.read_text())["notes"][0]

        human_labels = {}
        for rater, path in sample_raters[sid].items():
            lbl = load_human(path)
            if lbl:
                human_labels[rater] = lbl

        judge_df = load_judge(judge_dir / f"sample_{sid:03d}_note_00_sentence_judge.csv")

        aligned, mismatches = align_sample(sid, note, human_labels, judge_df)
        all_mismatches.extend(mismatches)

        if aligned.empty:
            continue

        rater_cols = [c for c in aligned.columns
                      if c not in ("sentence_idx", "sentence_text")]

        # Per-sample kappas
        row = {"sample_idx": sid, "n_sentences": len(aligned)}
        for i, r1 in enumerate(rater_cols):
            for r2 in rater_cols[i+1:]:
                v1 = aligned[r1].tolist()
                v2 = aligned[r2].tolist()
                row[f"kappa5_{r1}_vs_{r2}"]  = round(kappa(v1, v2), 4)
                row[f"agree5_{r1}_vs_{r2}"]  = round(pct_agree(v1, v2), 4)
                # Binary
                b1 = [("Faithful" if x == "Faithful" else "Not Faithful") if x else None for x in v1]
                b2 = [("Faithful" if x == "Faithful" else "Not Faithful") if x else None for x in v2]
                row[f"kappa2_{r1}_vs_{r2}"]  = round(kappa(b1, b2), 4)
                row[f"agree2_{r1}_vs_{r2}"]  = round(pct_agree(b1, b2), 4)

        per_sample_rows.append(row)

        # Accumulate for global pool
        for rater in rater_cols:
            pool.setdefault(rater, []).extend(aligned[rater].tolist())

    # ── Print mismatch report ─────────────────────────────────────────────────
    print("=" * 60)
    print("SENTENCE COUNT MISMATCHES (after header stripping)")
    print("=" * 60)
    if all_mismatches:
        for m in all_mismatches:
            print(f"  {m}")
    else:
        print("  None — all samples aligned perfectly.")
    print()

    if not per_sample_rows:
        print("No aligned samples found.")
        return

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_df.to_csv(out_path, index=False)
    print(f"Per-sample IAA saved → {out_path}")
    print()

    # ── Global IAA ────────────────────────────────────────────────────────────
    raters = sorted(pool.keys())
    print("=" * 60)
    print("GLOBAL AGREEMENT (pooled across all samples)")
    print("=" * 60)
    print(f"Raters: {raters}")
    print()

    # Pairwise
    print("Pairwise Cohen's Kappa")
    print(f"  {'Pair':<35} {'κ (5-class)':>12}  {'κ (binary)':>10}  {'agree%':>8}")
    print("  " + "-" * 68)
    for i, r1 in enumerate(raters):
        for r2 in raters[i+1:]:
            v1, v2 = pool[r1], pool[r2]
            k5 = kappa(v1, v2)
            b1 = ["Faithful" if x == "Faithful" else "Not Faithful" if x else None for x in v1]
            b2 = ["Faithful" if x == "Faithful" else "Not Faithful" if x else None for x in v2]
            k2 = kappa(b1, b2)
            ag = pct_agree(v1, v2)
            print(f"  {r1} vs {r2:<20} {k5:>12.4f}  {k2:>10.4f}  {ag:>7.1%}")

    print()
    # Krippendorff alpha over all raters
    all_rating_seqs = [pool[r] for r in raters]
    ka = kripp_alpha(all_rating_seqs)
    print(f"Krippendorff's Alpha (nominal, 5-class, all raters): {ka:.4f}")
    # Binary
    bin_seqs = [
        ["Faithful" if x == "Faithful" else "Not Faithful" if x else None for x in pool[r]]
        for r in raters
    ]
    ka2 = kripp_alpha(bin_seqs)
    print(f"Krippendorff's Alpha (nominal, binary,   all raters): {ka2:.4f}")
    print()

    # ── Confusion matrix: first human rater vs LLM ───────────────────────────
    human_raters = [r for r in raters if r != "judge"]
    if human_raters and "judge" in raters:
        ref = human_raters[0]
        v_h = pool[ref]
        v_j = pool["judge"]
        paired = [(h, j) for h, j in zip(v_h, v_j) if h is not None and j is not None]
        if paired:
            hs, js = zip(*paired)
            present = sorted(set(hs) | set(js))
            cm = confusion_matrix(hs, js, labels=present)
            print(f"Confusion matrix: {ref} (rows) vs judge (cols)")
            print(f"  Labels: {present}")
            header = "  " + " " * 15 + "  ".join(f"{l[:8]:>8}" for l in present)
            print(header)
            for i, row_lbl in enumerate(present):
                print(f"  {row_lbl:<15}" + "  ".join(f"{cm[i,j]:>8}" for j in range(len(present))))
            print()

            # Binary confusion
            bh = ["Faithful" if x == "Faithful" else "Not Faithful" for x in hs]
            bj = ["Faithful" if x == "Faithful" else "Not Faithful" for x in js]
            print(f"Binary report: {ref} as ground truth vs judge")
            print(classification_report(bh, bj, target_names=["Faithful", "Not Faithful"]))

    # ── Per-sample summary ────────────────────────────────────────────────────
    print()
    print("Per-sample kappa summary:")
    kappa_cols = [c for c in per_sample_df.columns if c.startswith("kappa5_")]
    if kappa_cols:
        print(per_sample_df[["sample_idx"] + kappa_cols].to_string(index=False))


if __name__ == "__main__":
    main()
