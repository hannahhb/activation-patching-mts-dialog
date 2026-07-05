"""
atomic_luq.py
=============
LUQ-ATOMIC: atomic-fact-level uncertainty scoring across K generated notes.

For each sample:
  1. Load all K=10 notes; parse raw sections for every note.
  2. Decompose only the first DECOMP_K=3 notes into atomic facts via LLM
     (cached to disk).
  3. For each fact in a decomposed note, score it against the raw section
     text of every other note (K-1 = 9 reference notes) using NLI:
       - premise  = sliding windows over the reference section text
       - hypothesis = atomic fact
     Soft entailment probability: softmax(logits)[entail_idx], max over
     windows, then averaged across the 9 reference notes.
  4. uncertainty(fact) = 1 - mean_entail_prob
  5. Save per-note CSVs: fact_idx, section, fact, uncertainty
     → <out>/facts/sample_NNN_note_KK_facts.csv

Usage:
    python atomic_luq.py
    python atomic_luq.py --start 0 --end 44
    python atomic_luq.py --gen-dir luq_out/llama/generations --out luq_out/llama_atomic
    python atomic_luq.py --no-cache
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import factmatch_sentence as fm
import luq_sentence as luq

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_K         = 10   # total notes in the NLI reference pool
DEFAULT_DECOMP_K  = 3    # how many notes to decompose into atomic facts
DEFAULT_GEN_DIR   = "luq_out/llama/generations"
DEFAULT_OUT_DIR   = "luq_out/llama_atomic"

# If the same-section entailment probability falls below this, also check
# the reference note's OTHER sections before concluding "unsupported" --
# guards against LLM generations inconsistently filing the same fact under
# different SOAP sections across resamples (see score_sample Step 3).
SECTION_FALLBACK_THRESHOLD = 0.5

# Section groupings — first match wins (checked against lowercased header lines)
SECTION_PATTERNS: List[Tuple[str, str]] = [
    ("subjective",  r"^subjective|^hpi|^history of present|^review of systems|^ros"
                    r"|^past medical|^pmh|^social history|^family history"
                    r"|^current medications|^medications|^allergies"),
    ("objective",   r"^objective|^vital|^physical exam|^pe\b|^test results"
                    r"|^labs|^imaging|^results"),
    ("assessment",  r"^assessment|^problem list|^impression|^diagnosis"),
    ("plan",        r"^plan|^follow.?up|^orders|^disposition"),
]
SECTION_NAMES = [s for s, _ in SECTION_PATTERNS]
OTHER_SECTION = "other"   # preamble / unmatched lines


# ── Section parser ─────────────────────────────────────────────────────────────

def parse_sections(note: str) -> Dict[str, str]:
    """Split a note into section text blocks keyed by section name."""
    blocks: Dict[str, List[str]] = {s: [] for s in SECTION_NAMES}
    blocks[OTHER_SECTION] = []
    current = OTHER_SECTION

    for line in note.split("\n"):
        header = line.lower().strip().rstrip(":").rstrip()
        matched = False
        if len(header) < 50:   # only short lines can be headers
            for sec, pat in SECTION_PATTERNS:
                if re.match(pat, header):
                    current = sec
                    matched = True
                    break
        if not matched:
            blocks[current].append(line)

    return {sec: "\n".join(lines).strip() for sec, lines in blocks.items()}


# ── Decomposition ──────────────────────────────────────────────────────────────

def decompose_section(text: str) -> List[str]:
    if not text.strip():
        return []
    raw = fm._llm_extract(fm._NOTE_PROMPT.format(note=text.strip()))
    return fm.filter_facts(raw)


def dedupe_across_sections(note_decomp: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Facts are decomposed per SOAP section independently (decompose_section
    is called once per section with no visibility into the others), so the
    same real-world fact commonly gets extracted twice under different
    sections -- e.g. a medication mentioned during history-taking
    (subjective: "The patient is on metformin.") and again when the plan
    confirms/continues it (plan: "The patient is taking metformin.").
    filter_facts() only removes malformed fragments within a single
    section's output; it never sees facts from other sections, so this
    cross-section duplication survives untouched today.

    Fix: flatten every section's facts into one list, deduplicate by cosine
    similarity (reusing factmatch_sentence.deduplicate_facts, threshold
    0.92 -- already tuned/battle-tested there for the same kind of
    paraphrase-duplicate problem), then reassign survivors back to
    whichever section they first appeared in."""
    flat_facts, flat_secs = [], []
    for sec in SECTION_NAMES:
        for f in note_decomp.get(sec, []):
            flat_facts.append(f)
            flat_secs.append(sec)
    if len(flat_facts) < 2:
        return note_decomp

    survivors = fm.deduplicate_facts(flat_facts)  # first-occurrence-wins, in original order
    survivor_counts = Counter(survivors)

    out = {sec: [] for sec in SECTION_NAMES}
    for f, sec in zip(flat_facts, flat_secs):
        if survivor_counts[f] > 0:
            out[sec].append(f)
            survivor_counts[f] -= 1  # only re-attach as many copies as survived
    return out


# ── Sliding window NLI ────────────────────────────────────────────────────────

def _max_entail_prob(nli, premise_windows: List[str], hypothesis: str, entail_idx: int) -> float:
    """Softmax entailment probability for a fact against a section, max over windows."""
    from scipy.special import softmax as sp_softmax
    pairs = [(w, hypothesis) for w in premise_windows]
    raw = np.asarray(
        nli.predict(pairs,
                    batch_size=len(pairs),
                    apply_softmax=False,
                    show_progress_bar=False)
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    probs = sp_softmax(raw, axis=1)          # (n_windows, n_labels)
    return float(probs[:, entail_idx].max()) # max entailment prob across windows


def _max_entail_probs_batch(
    nli,
    premise_windows: List[str],
    hypotheses: List[str],
    entail_idx: int,
) -> np.ndarray:
    """Max entailment probability for many facts against one section's windows.

    This keeps the scoring logic identical to repeated _max_entail_prob calls,
    but collapses thousands of tiny CrossEncoder.predict() invocations into
    larger batched calls that actually use the GPU.
    """
    from scipy.special import softmax as sp_softmax

    if not premise_windows or not hypotheses:
        return np.zeros(len(hypotheses), dtype=np.float64)

    n_w = len(premise_windows)
    n_h = len(hypotheses)
    pairs = [(w, h) for h in hypotheses for w in premise_windows]
    raw = np.asarray(
        nli.predict(
            pairs,
            batch_size=min(len(pairs), max(luq.MICRO_BATCH_PAIRS, luq.NLI_BATCH_SIZE)),
            apply_softmax=False,
            show_progress_bar=False,
        )
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    probs = sp_softmax(raw, axis=1)[:, entail_idx].reshape(n_h, n_w)
    return probs.max(axis=1).astype(np.float64)


# ── Paths ──────────────────────────────────────────────────────────────────────

def facts_csv_path(out_dir: Path, sample_idx: int, note_idx: int) -> Path:
    return out_dir / "facts" / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts.csv"


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_sample(sample_idx: int, notes: List[str], decomp_k: int,
                 out_dir: Path, use_cache: bool) -> List[dict]:
    """
    notes     : all K notes (default 10) — used as NLI reference pool.
    decomp_k  : first decomp_k notes are decomposed; their facts are scored
                against the raw section text of every other note (K-1 refs).
    uncertainty = 1 - mean(softmax entailment prob over K-1 reference notes)
    """
    K = len(notes)
    facts_dir = out_dir / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: parse raw sections for all K notes ───────────────────────────
    # all_sections[note_idx][sec] = raw section text (used as NLI premise)
    all_sections: List[Dict[str, str]] = [parse_sections(note) for note in notes]

    # ── Step 2: decompose first decomp_k notes into atomic facts (LLM, cached)
    # Cache is a per-sample list covering however many notes were decomposed
    # the LAST time this ran. If decomp_k grows between runs (e.g. someone
    # reruns with a larger --atomic-decomp-k), only decompose the newly
    # requested notes rather than trusting a shorter cached list as-is --
    # that used to IndexError on the notes past the old cache's length.
    decomp_cache = out_dir / "facts" / f"sample_{sample_idx:03d}_decomp.json"

    all_decomp = []
    if use_cache and decomp_cache.exists():
        all_decomp = json.loads(decomp_cache.read_text())
        tqdm.write(f"  [cache] decomp sample {sample_idx}: {len(all_decomp)}/{decomp_k} notes cached")

    if len(all_decomp) < decomp_k:
        for note_idx in range(len(all_decomp), decomp_k):
            note_decomp = {}
            for sec in SECTION_NAMES:
                text = all_sections[note_idx].get(sec, "")
                facts = decompose_section(text) if text else []
                note_decomp[sec] = facts
                if facts:
                    tqdm.write(f"    note {note_idx} [{sec}]: {len(facts)} facts")
            n_before = sum(len(v) for v in note_decomp.values())
            note_decomp = dedupe_across_sections(note_decomp)
            n_after = sum(len(v) for v in note_decomp.values())
            if n_after < n_before:
                tqdm.write(f"    note {note_idx}: cross-section dedup {n_before} -> {n_after} facts")
            all_decomp.append(note_decomp)
        decomp_cache.write_text(json.dumps(all_decomp, indent=2))

    # ── Step 3: NLI scoring ──────────────────────────────────────────────────
    # For each fact in a decomposed note:
    #   premise    = sliding windows over the matching section of each reference note
    #   hypothesis = atomic fact
    #   score      = max softmax entailment prob across windows (per ref note)
    #   uncertainty = 1 - mean(score over K-1 reference notes)
    nli = luq.get_nli()
    entail_idx = luq._label_indices[0]
    tokenizer = nli.tokenizer

    # Precompute windowed premises once per note/section. This used to be
    # recomputed inside the innermost fact loop, which dominated runtime even
    # before NLI inference.
    windows_cache: List[Dict[str, List[str]]] = []
    for sec_map in all_sections:
        sec_windows = {}
        for sec in SECTION_NAMES:
            text = sec_map.get(sec, "").strip()
            sec_windows[sec] = luq.sentence_windows(text, tokenizer=tokenizer) if text else []
        windows_cache.append(sec_windows)

    rows = []
    for note_idx in range(decomp_k):
        csv_p = facts_csv_path(out_dir, sample_idx, note_idx)
        if use_cache and csv_p.exists():
            df = pd.read_csv(csv_p)
            tqdm.write(f"  [cache] scores sample {sample_idx} note {note_idx}")
            rows.extend(df.to_dict("records"))
            continue

        ref_indices = [j for j in range(K) if j != note_idx]
        note_rows = []

        for sec in SECTION_NAMES:
            my_facts = all_decomp[note_idx].get(sec, [])
            if not my_facts:
                continue

            support_sum = np.zeros(len(my_facts), dtype=np.float64)
            support_cnt = np.zeros(len(my_facts), dtype=np.int32)

            for ref_idx in ref_indices:
                windows = windows_cache[ref_idx].get(sec, [])
                if windows:
                    probs = _max_entail_probs_batch(nli, windows, my_facts, entail_idx)
                    checked = np.ones(len(my_facts), dtype=bool)
                else:
                    probs = np.zeros(len(my_facts), dtype=np.float64)
                    checked = np.zeros(len(my_facts), dtype=bool)

                low_mask = probs < SECTION_FALLBACK_THRESHOLD
                if low_mask.any():
                    # Same-section support is weak. Only for those weak facts,
                    # search the other sections of the same reference note.
                    weak_facts = [my_facts[i] for i in np.where(low_mask)[0]]
                    for other_sec in SECTION_NAMES:
                        if other_sec == sec:
                            continue
                        other_windows = windows_cache[ref_idx].get(other_sec, [])
                        if not other_windows:
                            continue
                        checked[low_mask] = True
                        other_probs = _max_entail_probs_batch(
                            nli, other_windows, weak_facts, entail_idx
                        )
                        probs[low_mask] = np.maximum(probs[low_mask], other_probs)

                support_sum += probs
                support_cnt += checked.astype(np.int32)

            for local_idx, fact in enumerate(my_facts):
                if support_cnt[local_idx] == 0:
                    uncertainty = 1.0
                else:
                    uncertainty = 1.0 - float(support_sum[local_idx] / support_cnt[local_idx])

                note_rows.append({
                    "fact_idx":    len(note_rows),
                    "section":     sec,
                    "fact":        fact,
                    "uncertainty": round(uncertainty, 4),
                })

        df = pd.DataFrame(note_rows) if note_rows else pd.DataFrame(
            columns=["fact_idx", "section", "fact", "uncertainty"])
        df.to_csv(csv_p, index=False)
        mean_u = df["uncertainty"].mean() if len(df) else float("nan")
        tqdm.write(f"  [saved] sample {sample_idx} note {note_idx}: "
                   f"{len(df)} facts, mean_u={mean_u:.3f}")
        rows.extend(df.to_dict("records"))

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LUQ-ATOMIC: section-matched atomic-fact uncertainty scoring")
    p.add_argument("--start",    type=int, default=0)
    p.add_argument("--end",      type=int, default=132)
    p.add_argument("--K",        type=int, default=DEFAULT_K,
                   help="Total note pool per sample used as NLI reference (default 10)")
    p.add_argument("--decomp-k", type=int, default=DEFAULT_DECOMP_K,
                   help="How many notes to decompose into atomic facts (default 3)")
    p.add_argument("--gen-dir",  default=DEFAULT_GEN_DIR)
    p.add_argument("--out",      default=DEFAULT_OUT_DIR)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    gen_dir  = Path(args.gen_dir).resolve()
    out_dir  = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    print(f"Gen dir  : {gen_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"Samples  : {args.start} – {args.end - 1}")
    print(f"K (pool) : {args.K}")
    print(f"Decomp K : {args.decomp_k}")
    print(f"Sections : {SECTION_NAMES}")
    print(f"Cache    : {'on' if use_cache else 'off'}")

    summary_rows = []
    for sample_idx in tqdm(range(args.start, args.end), desc="samples", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            continue
        gen = json.loads(gen_path.read_text())
        notes = gen["notes"][: args.K]
        if len(notes) < 2:
            tqdm.write(f"  [skip] sample {sample_idx}: fewer than 2 notes")
            continue

        decomp_k = min(args.decomp_k, len(notes))
        print(f"\n[sample {sample_idx}] {len(notes)} notes, decomposing first {decomp_k}")
        rows = score_sample(sample_idx, notes, decomp_k, out_dir, use_cache)

        if rows:
            df = pd.DataFrame(rows)
            summary_rows.append({
                "sample_idx":  sample_idx,
                "n_facts":     len(df),
                "mean_u":      round(float(df["uncertainty"].mean()), 4),
                "high_u_frac": round(float((df["uncertainty"] > 0.5).mean()), 4),
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out_csv = out_dir / "atomic_luq_results.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\n[done] summary → {out_csv}")
        print(summary.describe())


if __name__ == "__main__":
    main()
