"""
factuality_nli.py
=================
Transcript-grounded factuality scoring for atomic facts.

For each atomic fact extracted from a generated note:
  - Slide a speaker-turn-aligned window over the full transcript
  - Run NLI: premise = window text, hypothesis = atomic fact
  - transcript_support = max softmax entailment prob across all windows
  - factuality_uncertainty = 1 - transcript_support

Window boundaries always start at a [doctor] or [patient] turn so the
NLI model always sees complete speaker turns, never mid-sentence text.

Window size : 1400 chars  (~470 DeBERTa tokens, leaving room for hypothesis)
Stride      : 700 chars   (advance by ~10 turns at avg 68 chars/turn)
Windows/transcript: ~7 on average (range 5–12 across the 44-sample set)
NLI calls/fact:     ~7  →  ~840 calls/sample (120 facts × 7 windows)

Reuses the atomic-fact decomposition cache from atomic_luq.py
(<out>/facts/sample_NNN_decomp.json) — run atomic_luq first, or pass
--decomp-dir to the same directory.

Usage:
    python factuality_nli.py --start 0 --end 44
    python factuality_nli.py --gen-dir luq_out/llama/generations \\
                             --decomp-dir luq_out/llama_atomic \\
                             --out luq_out/llama_factuality
    python factuality_nli.py --nli-model mrm8488/deberta-v3-large-finetuned-mnli
    python factuality_nli.py --no-cache
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.special import softmax as sp_softmax
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import luq_sentence as luq

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_K          = 3
DEFAULT_GEN_DIR    = "luq_out/llama/generations"
DEFAULT_DECOMP_DIR = "luq_out/llama_atomic"
DEFAULT_OUT_DIR    = "luq_out/llama_factuality"

WINDOW_CHARS = 1400
STRIDE_CHARS = 700

TURN_RE = re.compile(r'(?=\[(?:doctor|patient)\])', re.IGNORECASE)


# ── Sliding window over transcript ────────────────────────────────────────────

def transcript_windows(transcript: str,
                        window: int = WINDOW_CHARS,
                        stride: int = STRIDE_CHARS) -> List[str]:
    """
    Split transcript into overlapping windows that always start at a speaker
    turn boundary ([doctor] or [patient]).  Each window is at most `window`
    chars; the start pointer advances by at least `stride` chars between
    consecutive windows.
    """
    turns = [t.strip() for t in TURN_RE.split(transcript) if t.strip()]
    if not turns:
        windows, start = [], 0
        while start < len(transcript):
            windows.append(transcript[start: start + window])
            if start + window >= len(transcript):
                break
            start += stride
        return windows

    n = len(turns)
    windows: List[str] = []
    i = 0
    while i < n:
        buf, total, j = [], 0, i
        while j < n:
            seg_len = len(turns[j]) + 1
            if buf and total + seg_len > window:
                break
            buf.append(turns[j])
            total += seg_len
            j += 1
        if not buf:
            buf = [turns[i]]
        windows.append("\n".join(buf))
        advanced = 0
        while i < n - 1 and advanced < stride:
            advanced += len(turns[i]) + 1
            i += 1
        if i >= n - 1:
            break
    return windows


# ── NLI helper ────────────────────────────────────────────────────────────────

def max_entail_prob(nli, windows: List[str], hypothesis: str,
                    entail_idx: int) -> float:
    """Max softmax entailment probability of hypothesis against any window."""
    pairs = [(w, hypothesis) for w in windows]
    raw = np.asarray(
        nli.predict(pairs,
                    batch_size=len(pairs),
                    apply_softmax=False,
                    show_progress_bar=False)
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    probs = sp_softmax(raw, axis=1)
    return float(probs[:, entail_idx].max())


# ── Per-sample scoring ────────────────────────────────────────────────────────

def score_sample(sample_idx: int,
                 transcript: str,
                 decomp: List[Dict],
                 out_dir: Path,
                 use_cache: bool,
                 nli,
                 entail_idx: int) -> List[dict]:
    facts_dir = out_dir / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)

    windows = transcript_windows(transcript)
    tqdm.write(f"  transcript windows: {len(windows)} "
               f"(chars {[len(w) for w in windows]})")

    rows = []
    for note_idx, note_decomp in enumerate(decomp):
        csv_p = facts_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_factuality.csv"
        if use_cache and csv_p.exists():
            df = pd.read_csv(csv_p)
            tqdm.write(f"  [cache] factuality sample {sample_idx} note {note_idx}")
            rows.extend(df.to_dict("records"))
            continue

        note_rows = []
        for sec, facts in note_decomp.items():
            if not facts:
                continue
            for fact in facts:
                support = max_entail_prob(nli, windows, fact, entail_idx)
                note_rows.append({
                    "fact_idx":               len(note_rows),
                    "section":                sec,
                    "fact":                   fact,
                    "transcript_support":     round(support, 4),
                    "factuality_uncertainty": round(1.0 - support, 4),
                })

        df = pd.DataFrame(note_rows) if note_rows else pd.DataFrame(
            columns=["fact_idx", "section", "fact",
                     "transcript_support", "factuality_uncertainty"])
        df.to_csv(csv_p, index=False)
        mean_s = df["transcript_support"].mean() if len(df) else float("nan")
        tqdm.write(f"  [saved] sample {sample_idx} note {note_idx}: "
                   f"{len(df)} facts, mean_support={mean_s:.3f}")
        rows.extend(df.to_dict("records"))

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Factuality NLI: score atomic facts against the transcript")
    p.add_argument("--start",      type=int, default=0)
    p.add_argument("--end",        type=int, default=132)
    p.add_argument("--K",          type=int, default=DEFAULT_K)
    p.add_argument("--gen-dir",    default=DEFAULT_GEN_DIR)
    p.add_argument("--decomp-dir", default=DEFAULT_DECOMP_DIR)
    p.add_argument("--out",        default=DEFAULT_OUT_DIR)
    p.add_argument("--nli-model",  default=None,
                   help="Override NLI model (default: cross-encoder/nli-deberta-v3-large)")
    p.add_argument("--no-cache",   action="store_true")
    return p.parse_args()


def main():
    args       = parse_args()
    gen_dir    = Path(args.gen_dir).resolve()
    decomp_dir = Path(args.decomp_dir).resolve()
    out_dir    = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache  = not args.no_cache

    print(f"Gen dir    : {gen_dir}")
    print(f"Decomp dir : {decomp_dir}")
    print(f"Out dir    : {out_dir}")
    print(f"Samples    : {args.start} – {args.end - 1}")
    print(f"Notes/sample: {args.K}")
    print(f"Window     : {WINDOW_CHARS} chars / stride {STRIDE_CHARS} chars")
    print(f"Cache      : {'on' if use_cache else 'off'}")

    if args.nli_model:
        luq.NLI_MODEL_NAME = args.nli_model

    nli        = luq.get_nli()
    entail_idx = luq._label_indices[0]

    summary_rows = []
    for sample_idx in tqdm(range(args.start, args.end), desc="samples", unit="sample"):
        gen_path    = gen_dir    / f"sample_{sample_idx:03d}_generations.json"
        decomp_path = decomp_dir / "facts" / f"sample_{sample_idx:03d}_decomp.json"

        if not gen_path.exists():
            tqdm.write(f"  [skip] sample {sample_idx}: no generation file")
            continue
        if not decomp_path.exists():
            tqdm.write(f"  [skip] sample {sample_idx}: no decomp cache "
                       f"(run atomic_luq.py first)")
            continue

        gen        = json.loads(gen_path.read_text())
        transcript = gen["transcript"]
        decomp     = json.loads(decomp_path.read_text())[: args.K]

        tqdm.write(f"\n[sample {sample_idx}] transcript {len(transcript)} chars, "
                   f"{len(decomp)} notes")

        rows = score_sample(sample_idx, transcript, decomp,
                            out_dir, use_cache, nli, entail_idx)

        if rows:
            df = pd.DataFrame(rows)
            summary_rows.append({
                "sample_idx":       sample_idx,
                "n_facts":          len(df),
                "mean_support":     round(float(df["transcript_support"].mean()), 4),
                "low_support_frac": round(float((df["transcript_support"] < 0.5).mean()), 4),
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out_csv = out_dir / "factuality_results.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\n[done] summary → {out_csv}")
        print(summary.describe())


if __name__ == "__main__":
    main()
