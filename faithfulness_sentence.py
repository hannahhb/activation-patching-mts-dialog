"""
faithfulness_sentence.py
========================
SummaC_ZS-style transcript-grounded faithfulness scorer.

For each generated note sentence, computes the maximum NLI entailment
probability against any transcript sentence (Laban et al., TACL 2022):

  Faithfulness(s_j) = max_i  P(entail | transcript_i, s_j)
  Uncertainty(s_j)  = 1 - Faithfulness(s_j)

The M x N pair matrix (transcript sentences x note sentences) is computed
in micro-batches using the same DeBERTa-v3-large NLI model as LUQ.

Output: luq_out/llama/faithfulness/sample_NNN_note_KK_faithfulness.csv
  sentence_idx, sentence, faithfulness, uncertainty

Usage:
  python faithfulness_sentence.py --start 0 --end 10
  python faithfulness_sentence.py --start 0 --end 132 --no-cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from luq_sentence import (
    DEFAULT_OUT_DIR,
    MICRO_BATCH_PAIRS,
    NLI_BATCH_SIZE,
    assign_sentence_sections,
    clear_memory,
    get_nli,
    predict_pair_supports,
    split_sentences,
)

FAITHFULNESS_OUT_SUBDIR = "faithfulness"

# Speaker-tag prefix pattern for dialogue transcripts like "[doctor]", "[patient]"
import re
SPEAKER_TAG_RE = re.compile(r"^\s*\[(?:doctor|patient|clinician|nurse)[^\]]*\]\s*", re.IGNORECASE)


def split_transcript_sentences(transcript: str) -> List[str]:
    """
    Split a dialogue transcript into sentence units.
    Strips speaker tags so the NLI model sees clean premise text.
    """
    sentences = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip speaker tag, keep the utterance
        utterance = SPEAKER_TAG_RE.sub("", line).strip()
        if utterance:
            # Further split long utterances on periods
            for sent in split_sentences(utterance):
                if sent.strip():
                    sentences.append(sent.strip())
    return sentences


def score_note_faithfulness(
    note: str,
    transcript_sentences: List[str],
) -> List[Dict]:
    """
    Compute SummaC_ZS faithfulness for every sentence in note.

    Returns list of dicts: sentence_idx, sentence, faithfulness, uncertainty
    """
    nli = get_nli()
    note_sentences = split_sentences(note)
    if not note_sentences or not transcript_sentences:
        return []

    M = len(transcript_sentences)
    N = len(note_sentences)

    # Build all (transcript_i, note_j) pairs in column-major order so we can
    # fill the M x N matrix and then take column-wise max.
    pairs: List[tuple] = []
    pair_i: List[int] = []   # transcript sentence index
    pair_j: List[int] = []   # note sentence index

    for j, note_sent in enumerate(note_sentences):
        for i, trans_sent in enumerate(transcript_sentences):
            pairs.append((trans_sent, note_sent))
            pair_i.append(i)
            pair_j.append(j)

    # Score in micro-batches
    all_scores = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), MICRO_BATCH_PAIRS):
        batch = pairs[start : start + MICRO_BATCH_PAIRS]
        all_scores[start : start + len(batch)] = predict_pair_supports(nli, batch)

    # Fill M x N matrix, then take column max (SummaC_ZS)
    pair_matrix = np.full((M, N), 0.0, dtype=np.float64)
    for idx, (i, j) in enumerate(zip(pair_i, pair_j)):
        pair_matrix[i, j] = all_scores[idx]

    # Column max: best-supporting transcript sentence for each note sentence
    faithfulness = pair_matrix.max(axis=0)   # shape (N,)
    uncertainty = 1.0 - faithfulness

    rows = []
    for j, note_sent in enumerate(note_sentences):
        rows.append({
            "sentence_idx": j,
            "sentence": note_sent,
            "faithfulness": round(float(faithfulness[j]), 4),
            "uncertainty": round(float(uncertainty[j]), 4),
        })
    return rows


def score_generation_file(
    gen_path: Path,
    out_dir: Path,
    use_cache: bool,
) -> Dict:
    saved = json.loads(gen_path.read_text())
    sample_idx = int(saved["sample_idx"])
    notes = saved["notes"]
    transcript = saved.get("transcript", "")

    if not transcript:
        return {"sample_idx": sample_idx, "status": "no_transcript"}
    if not notes:
        return {"sample_idx": sample_idx, "status": "no_notes"}

    transcript_sentences = split_transcript_sentences(transcript)
    if not transcript_sentences:
        return {"sample_idx": sample_idx, "status": "empty_transcript"}

    tqdm.write(f"  [faith] transcript: {len(transcript_sentences)} sentences")

    summary_scores = []

    for ref_idx, note in enumerate(notes):
        out_path = out_dir / f"sample_{sample_idx:03d}_note_{ref_idx:02d}_faithfulness.csv"

        if use_cache and out_path.exists():
            tqdm.write(f"  [cache] {out_path.name}")
            df = pd.read_csv(out_path)
            summary_scores.append(float(df["faithfulness"].mean()))
            continue

        rows = score_note_faithfulness(note, transcript_sentences)
        if not rows:
            tqdm.write(f"  [faith] sample {sample_idx} note {ref_idx:02d}: no sentences")
            continue

        pd.DataFrame(rows).to_csv(out_path, index=False)
        mean_f = float(np.mean([r["faithfulness"] for r in rows]))
        summary_scores.append(mean_f)
        tqdm.write(
            f"  [faith] sample {sample_idx} note {ref_idx:02d}: "
            f"{len(rows)} sentences, mean faithfulness={mean_f:.4f}"
        )

    clear_memory()

    if not summary_scores:
        return {"sample_idx": sample_idx, "status": "no_sentences"}

    return {
        "sample_idx": sample_idx,
        "status": "ok",
        "K_actual": len(notes),
        "notes_scored": len(summary_scores),
        "mean_faithfulness": round(float(np.mean(summary_scores)), 4),
        "min_faithfulness": round(float(np.min(summary_scores)), 4),
    }


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def stage_score(
    start: int,
    end: int,
    out_dir: Path,
    use_cache: bool,
) -> pd.DataFrame:
    gen_dir = out_dir / "generations"
    faith_dir = out_dir / FAITHFULNESS_OUT_SUBDIR
    faith_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for sample_idx in tqdm(range(start, end), desc="faithfulness", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            rows.append({"sample_idx": sample_idx, "status": "no_generations"})
            continue

        print(f"\n[faithfulness] sample {sample_idx}")
        rows.append(score_generation_file(gen_path, faith_dir, use_cache))

    df = pd.DataFrame(rows)
    results_path = out_dir / "faithfulness_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n[faithfulness] wrote {results_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SummaC_ZS transcript-grounded faithfulness scorer for ACI-Bench SOAP notes"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=132)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help=f"Root output dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-score even if output CSV already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    print(f"Samples: {args.start} to {args.end - 1}")
    print(f"Output:  {out_dir / FAITHFULNESS_OUT_SUBDIR}")
    print(f"Cache:   {'on' if use_cache else 'off'}")
    print(f"Method:  SummaC_ZS (max entailment over transcript sentences)")

    stage_score(args.start, args.end, out_dir, use_cache)


if __name__ == "__main__":
    main()
