"""
luq_table.py
============
Compute the LUQ results table for the ACL paper.

For each model (--luq-dir), reads every sample_*_note_*_sentences.csv file
(one per generation), assigns each sentence to its SOAP section by tracking
header lines in order, then reports:

  Section | Total Sentences | Q1 [0,.25) | Q2 [.25,.5) | Q3 [.5,.75) | Q4 [.75,1] | Mean U

Usage:
  # luq_dir can be any of:
  #   luq_out/              (contains sentences/ subdirectory)
  #   luq_out/sentences/    (contains CSVs directly)
  #   .                     (if luq_out/ is a subdirectory here)
  python luq_table.py --luq-dir luq_out --model "Llama 3.1 8B"
  python luq_table.py \
      --luq-dir llama_results/luq_out --model "Llama 3.1 8B" \
      --luq-dir gemma_results/luq_out --model "Gemma 3 4B IT"
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SOAP section detection
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_PATTERNS = [
    (re.compile(r"^follow.?up", re.IGNORECASE),        "Follow-up"),
    (re.compile(r"^plan",        re.IGNORECASE),        "Plan"),
    (re.compile(r"^assessment",  re.IGNORECASE),        "Assessment"),
    (re.compile(r"^objective",   re.IGNORECASE),        "Objective"),
    (re.compile(r"^subjective",  re.IGNORECASE),        "Subjective"),
]

_SECTION_ORDER = ["Subjective", "Objective", "Assessment", "Plan", "Follow-up"]

_HEADER_RE = re.compile(
    r"^(subjective|objective|assessment|plan|follow.?up)[:\s]",
    re.IGNORECASE,
)


def detect_section(sentence: str) -> str | None:
    """Return section name if sentence is a SOAP header line, else None."""
    s = sentence.strip().lstrip("-•*#+").strip()
    for pattern, name in _SECTION_PATTERNS:
        if pattern.match(s):
            return name
    return None


def assign_sections(sentences: list[str]) -> list[str]:
    """
    Walk sentences in order; when a header is detected update current section.
    Sentences before any header are labelled 'Preamble'.
    """
    current = "Preamble"
    labels = []
    for sent in sentences:
        detected = detect_section(sent)
        if detected is not None:
            current = detected
        labels.append(current)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Load all per-generation sentence CSVs for one model
# ─────────────────────────────────────────────────────────────────────────────

def load_sentence_data(luq_dir: Path) -> pd.DataFrame:
    """
    Read every sample_*_note_*_sentences.csv.
    Searches in order:
      1. {luq_dir}/sentences/
      2. {luq_dir}/  (CSVs directly inside)
      3. {luq_dir}/luq_out/sentences/  (if luq_dir is an --out root)
    Returns a DataFrame with columns:
        sample_idx, note_idx, sentence, uncertainty, section
    """
    candidates = [
        luq_dir / "sentences",
        luq_dir,
        luq_dir / "luq_out" / "sentences",
    ]
    files = []
    sent_dir = None
    for candidate in candidates:
        if candidate.is_dir():
            found = sorted(candidate.glob("sample_*_note_*_sentences.csv"))
            if found:
                files = found
                sent_dir = candidate
                break

    if not files:
        tried = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"No sample_*_note_*_sentences.csv files found. Tried:\n  {tried}"
        )
    print(f"  Found {len(files)} note files in {sent_dir}")

    records = []
    for f in files:
        # Parse sample and note index from filename
        m = re.search(r"sample_(\d+)_note_(\d+)", f.stem)
        if not m:
            continue
        sample_idx = int(m.group(1))
        note_idx   = int(m.group(2))

        df = pd.read_csv(f)
        sections = assign_sections(df["sentence"].tolist())
        df["section"]    = sections
        df["sample_idx"] = sample_idx
        df["note_idx"]   = note_idx
        records.append(df)

    all_df = pd.concat(records, ignore_index=True)
    print(f"  Loaded {len(files)} note files, {len(all_df):,} total sentence rows")
    return all_df


# ─────────────────────────────────────────────────────────────────────────────
# Compute table rows
# ─────────────────────────────────────────────────────────────────────────────

QUARTILE_EDGES = [0.0, 0.25, 0.50, 0.75, 1.001]  # right edge slightly >1
QUARTILE_LABELS = ["Q1 [0,.25)", "Q2 [.25,.5)", "Q3 [.5,.75)", "Q4 [.75,1]"]


def section_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each SOAP section (plus an 'All' summary row), compute:
        total sentences, Q1–Q4 fractions, mean U
    Excludes Preamble sentences (headers/boilerplate before the SOAP body).
    """
    # Drop preamble and raw header sentences
    df = df[~df["section"].isin(["Preamble"])].copy()

    rows = []
    sections_present = [s for s in _SECTION_ORDER if s in df["section"].unique()]

    for section in sections_present + ["All"]:
        sub = df if section == "All" else df[df["section"] == section]
        u = sub["uncertainty"].values

        q_fracs = []
        for lo, hi in zip(QUARTILE_EDGES[:-1], QUARTILE_EDGES[1:]):
            frac = float(np.mean((u >= lo) & (u < hi))) * 100
            q_fracs.append(f"{frac:.1f}%")

        rows.append({
            "Section":         section,
            "Total Sentences": len(u),
            **dict(zip(QUARTILE_LABELS, q_fracs)),
            "Mean U(s_j)":     f"{np.mean(u):.3f}",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_table(model: str, stats: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print(f"  Model: {model}")
    print(f"{'='*72}")
    print(stats.to_string(index=False))


def save_table(model: str, stats: pd.DataFrame, out_dir: Path) -> None:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    out_path = out_dir / f"luq_table_{slug}.csv"
    stats.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="LUQ results table by SOAP section")
    p.add_argument(
        "--luq-dir",
        dest="luq_dirs",
        action="append",
        required=True,
        metavar="DIR",
        help="Path to luq_out directory (repeat for multiple models)",
    )
    p.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        metavar="NAME",
        help="Model label matching each --luq-dir (same order)",
    )
    p.add_argument(
        "--out",
        default=".",
        help="Directory to write CSV tables (default: current dir)",
    )
    args = p.parse_args()

    if len(args.luq_dirs) != len(args.models):
        p.error("--luq-dir and --model must be provided the same number of times")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for luq_dir, model in zip(args.luq_dirs, args.models):
        print(f"\nProcessing: {model} ({luq_dir})")
        df    = load_sentence_data(Path(luq_dir))
        stats = section_stats(df)
        print_table(model, stats)
        save_table(model, stats, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
