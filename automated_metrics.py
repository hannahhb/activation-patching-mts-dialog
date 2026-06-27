"""
automated_metrics.py
====================
Compute automated metrics for generated SOAP notes.

Reference-based (vs ACI-Bench "note" column):
  BLEU-1, ROUGE-L, METEOR

Reference-free (vs source transcript):
  SummaC-ZS — NLI-based factual consistency of the note against the transcript

For each sample, scores are computed per generated note then averaged across K.

Dependencies:
    pip install rouge-score nltk summac
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Usage:
    python automated_metrics.py
    python automated_metrics.py --generations-dir luq_out/llama/generations --out luq_out/llama
    python automated_metrics.py --start 0 --end 44
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_REPO   = "mkieffer/ACI-Bench-MedARC"
_DATASET_CONFIG = "aci"
_DATASET_SPLIT  = "test1"


def _load_dataset():
    print(f"[data] Loading {_DATASET_REPO} …")
    ds = load_dataset(_DATASET_REPO, _DATASET_CONFIG, split=_DATASET_SPLIT)
    print(f"[data] {len(ds)} rows, columns: {ds.column_names}")
    return ds


def _get_reference(row: dict) -> str:
    for col in ["note", "tgt", "reference", "summary", "output"]:
        if col in row and row[col]:
            return row[col].strip()
    raise KeyError(f"No reference note column found in {list(row.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Reference-based metrics
# ─────────────────────────────────────────────────────────────────────────────

_rouge  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def _reference_metrics(hypothesis: str, reference: str) -> Dict[str, float]:
    hyp_tok = hypothesis.lower().split()
    ref_tok  = reference.lower().split()
    return {
        "bleu1":   sentence_bleu([ref_tok], hyp_tok, weights=(1, 0, 0, 0),
                                  smoothing_function=_smooth),
        "rouge_l": _rouge.score(reference, hypothesis)["rougeL"].fmeasure,
        "meteor":  meteor_score([ref_tok], hyp_tok),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SummaC (reference-free, transcript → note)
# ─────────────────────────────────────────────────────────────────────────────

_summac_model = None


def _get_summac():
    global _summac_model
    if _summac_model is None:
        from summac.model_summac import SummaCZS
        print("[summac] Loading SummaCZS (mnli) …")
        _summac_model = SummaCZS(granularity="sentence", model_name="mnli", device="cpu")
        print("[summac] Ready.")
    return _summac_model


def _summac_score(notes: List[str], transcript: str) -> List[float]:
    """Score each note against the transcript. Returns one score per note."""
    model   = _get_summac()
    result  = model.score([transcript] * len(notes), notes)
    return result["scores"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    start: int,
    end: int,
    generations_dir: Path,
    out_dir: Path,
    gen_idx: int = 0,
) -> pd.DataFrame:
    ds  = _load_dataset()
    end = min(end, len(ds))

    rows: List[Dict] = []

    for si in tqdm(range(start, end), desc="metrics", unit="sample"):
        gen_path = generations_dir / f"sample_{si:03d}_generations.json"
        if not gen_path.exists():
            tqdm.write(f"  [skip] sample {si}: no generation file.")
            continue

        saved      = json.loads(gen_path.read_text())
        notes      = saved["notes"]
        transcript = saved.get("transcript", "")
        if not notes:
            continue

        try:
            reference = _get_reference(ds[si])
        except KeyError as e:
            tqdm.write(f"  [skip] sample {si}: {e}")
            continue

        if gen_idx >= len(notes):
            tqdm.write(f"  [skip] sample {si}: gen_idx {gen_idx} out of range (K={len(notes)}).")
            continue
        note = notes[gen_idx]

        # Reference-based: BLEU-1, ROUGE-L, METEOR
        ref = _reference_metrics(note, reference)

        # Reference-free: SummaC vs transcript
        summac = _summac_score([note], transcript)[0] if transcript else float("nan")

        rows.append({
            "sample_idx": si,
            "bleu1":      round(ref["bleu1"],   4),
            "rouge_l":    round(ref["rouge_l"], 4),
            "meteor":     round(ref["meteor"],  4),
            "summac":     round(summac,          4),
        })

    df = pd.DataFrame(rows)
    out_path = out_dir / "automated_metrics.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[done] {len(df)} samples → {out_path.resolve()}")
    if not df.empty:
        print(f"\n  BLEU-1  : {df['bleu1'].mean():.4f}  (reference-based)")
        print(f"  ROUGE-L : {df['rouge_l'].mean():.4f}  (reference-based)")
        print(f"  METEOR  : {df['meteor'].mean():.4f}  (reference-based)")
        print(f"  SummaC  : {df['summac'].mean():.4f}  (reference-free)\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BLEU-1 / ROUGE-L / METEOR / SummaC for generated SOAP notes"
    )
    p.add_argument("--generations-dir", default="luq_out/llama/generations",
                   help="Directory containing sample_NNN_generations.json files")
    p.add_argument("--out",             default="luq_out/llama",
                   help="Output directory for automated_metrics.csv")
    p.add_argument("--start",           type=int, default=0)
    p.add_argument("--end",             type=int, default=132)
    p.add_argument("--gen-idx",         type=int, default=0,
                   help="Which generation to score (0-indexed, default 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        start=args.start,
        end=args.end,
        generations_dir=Path(args.generations_dir),
        out_dir=Path(args.out),
        gen_idx=args.gen_idx,
    )
