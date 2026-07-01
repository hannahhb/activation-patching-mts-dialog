"""
factmatch_sentence.py
=====================
Fact-level faithfulness and completeness scorer using an embed+NLI hybrid.

Pipeline
--------
1. Extract atomic facts from transcript via LLM — once per sample, cached.
2. Extract atomic facts from each generated note via LLM.
3. Embed all facts with sentence-transformers (all-mpnet-base-v2).
4. For each note fact: find top-k transcript facts by cosine similarity.
5. Run DeBERTa NLI on those top-k pairs to get entailment / neutral / contradiction.
6. Classify per CREOLA taxonomy:
     any entailment in top-k  → supported
     no entailment + contradiction  → negation-type hallucination
     all neutral / cosine < floor   → fabrication-type hallucination
7. Symmetric completeness pass (transcript facts → note facts).

Outputs per sample:
  <factmatch_dir>/sample_NNN_transcript_facts.json    (cached; shared across notes)

Outputs per note:
  <factmatch_dir>/sample_NNN_note_KK_factmatch.csv    (faithfulness)
      fact_idx, fact, supported, halluc_type, top_cos, top_trans_fact, top_nli
  <factmatch_dir>/sample_NNN_note_KK_completeness.csv (completeness)
      fact_idx, fact, covered, top_cos, top_note_fact, top_nli

Summary CSV: <out>/factmatch_results.csv

Usage
-----
  python factmatch_sentence.py --start 0 --end 10
  python factmatch_sentence.py --start 0 --end 132 --max-notes 5
  python factmatch_sentence.py --start 0 --end 132 --no-cache
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import luq_sentence
from luq_sentence import (
    DEFAULT_OUT_DIR,
    NLI_BATCH_SIZE,
    get_bedrock_client,
    get_nli,
    clear_memory,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FACTMATCH_SUBDIR = "factmatch"
BEDROCK_GEN_MODEL = "us.meta.llama3-3-70b-instruct-v1:0"
# Biomedical encoder — better than all-mpnet-base-v2 for clinical synonyms,
# drug names, and abbreviations (FactEHR uses ClinicalBERT embeddings for the same reason)
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

TOP_K = 8
COSINE_FLOOR = 0.30  # skip NLI entirely if best cosine is below this
NEGATION_COS_MIN  = 0.93  # contradiction only counts as negation if this high-cosine
HIGH_COS_SUPPORT  = 0.95  # cosine this high → treat as supported near-paraphrase

# Post-decomposition quality filters (FactEHR: 12% of GPT-4o facts are non-atomic/non-independent)
DEDUP_THRESHOLD = 0.92        # cosine above which two facts are considered duplicates
MIN_FACT_WORDS  = 3           # facts shorter than this are malformed fragments
_CONJUNCTION_RE = re.compile(
    r"^(and|also|however|additionally|furthermore|moreover|in addition|but|or|nor|so|yet)\b",
    re.IGNORECASE,
)

MAX_RETRIES = 5
RETRY_SLEEP = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embed] Loading {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
        print("[embed] Ready")
    return _embedder


def embed_facts(facts: List[str]) -> np.ndarray:
    """Returns L2-normalised embeddings, shape (N, d)."""
    return np.asarray(
        get_embedder().encode(facts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def filter_facts(facts: List[str]) -> List[str]:
    """
    Remove malformed or non-independent facts post-decomposition.
    FactEHR found ~12% of GPT-4o facts fail atomicity/independence checks;
    higher for smaller models.
    """
    out = []
    for f in facts:
        if len(f.split()) < MIN_FACT_WORDS:
            continue
        if _CONJUNCTION_RE.match(f):  # non-independent fragment
            continue
        out.append(f)
    return out


def deduplicate_facts(facts: List[str]) -> List[str]:
    """
    Remove near-duplicate facts using cosine similarity.
    Decomposition often produces paraphrase pairs that inflate counts
    and bias precision/recall metrics.
    """
    if len(facts) <= 1:
        return facts
    embs = embed_facts(facts)
    sim: np.ndarray = embs @ embs.T
    keep: List[str] = []
    dropped: set = set()
    for i in range(len(facts)):
        if i in dropped:
            continue
        keep.append(facts[i])
        for j in range(i + 1, len(facts)):
            if sim[i, j] >= DEDUP_THRESHOLD:
                dropped.add(j)
    return keep


def clean_facts(facts: List[str]) -> List[str]:
    """Filter then deduplicate — order matters (filter first to avoid embedding junk)."""
    return deduplicate_facts(filter_facts(facts))


# ─────────────────────────────────────────────────────────────────────────────
# NLI — 3-class labels
# ─────────────────────────────────────────────────────────────────────────────

def predict_pair_nli_labels(pairs: List[Tuple[str, str]]) -> List[str]:
    """
    Run DeBERTa-v3-large NLI on (premise, hypothesis) pairs.
    Returns 'entailment', 'neutral', or 'contradiction' for each pair.
    """
    if not pairs:
        return []

    nli = get_nli()  # also sets luq_sentence._label_indices
    entail_idx, contradict_idx = luq_sentence._label_indices
    neutral_idx = next(i for i in range(3) if i not in (entail_idx, contradict_idx))

    idx_to_label: Dict[int, str] = {
        entail_idx:     "entailment",
        neutral_idx:    "neutral",
        contradict_idx: "contradiction",
    }

    raw = np.asarray(
        nli.predict(
            pairs,
            batch_size=NLI_BATCH_SIZE,
            apply_softmax=True,
            show_progress_bar=False,
        )
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    return [idx_to_label[int(np.argmax(row))] for row in raw]


# ─────────────────────────────────────────────────────────────────────────────
# LLM fact extraction — sentence-level
# ─────────────────────────────────────────────────────────────────────────────

_TRANSCRIPT_PROMPT = """\
You are a meticulous physician reviewing a clinical consultation transcript.
Your goal is to write out every fact stated or confirmed in the transcript as separate, independent facts, delimited by " // " (do not use any other format or say "Here is the list...").

At each step, separate out information with multiple modifiers into simpler, more granular facts.
For medications, split drug name, dose, frequency, and indication into separate facts — repeat the drug name in each.
Include patient demographics mentioned in the transcript (name, age, sex, occupation).
Do not include pure questions or back-channel utterances (e.g. "okay", "sure", "mm-hmm").

Always attribute facts to the correct person:
- Use "The patient" for things the patient says, experiences, or reports about themselves.
- Use "The patient's partner" (or other named companion) for things a companion reports.
- Use "The doctor" only for actions the doctor takes (prescribes, examines, diagnoses, orders).
- Do NOT attribute clinical observations about the patient to the doctor (e.g. "The doctor appears healthy" is wrong; "The patient considers themselves healthy" is correct).

Example:
Transcript: "[doctor] hi sarah what brings you in today [patient] i've been having chest pain for about three days [doctor] okay do you have any fever chills nausea or vomiting [patient] no none of those [doctor] any abdominal pain or diarrhea [patient] no stomach pain but i do get diarrhea sometimes after drinking [doctor] any blood in your stool [patient] no [doctor] do you smoke or drink [patient] i used to smoke half a pack a day but i quit two years ago and i drink maybe two drinks a week [doctor] okay i'm going to start you on lisinopril five milligrams once daily for your blood pressure and i want to see you back in four weeks [patient] sounds good"
Atomic facts:
The patient's name is Sarah. // The patient has chest pain. // The chest pain has lasted three days. // The patient denied fever. // The patient denied chills. // The patient denied nausea. // The patient denied vomiting. // The patient denied abdominal pain. // The patient experiences diarrhea. // The diarrhea occurs after drinking. // The patient denied blood in the stool. // The patient used to smoke. // The patient smoked about half a pack per day. // The patient quit smoking two years ago. // The patient drinks alcohol. // The patient drinks about two drinks per week. // The doctor prescribed lisinopril. // The lisinopril dose is 5 milligrams. // Lisinopril is taken once daily. // Lisinopril is prescribed for blood pressure. // The follow-up appointment is in four weeks.

Transcript:
{transcript}

Atomic facts:\
"""

_NOTE_PROMPT = """\
You are a meticulous physician. Extract every clinical fact from this SOAP note as atomic facts delimited by " // ". Output ONLY the facts — no explanation, no preamble.

Rules:
- Split conjunctions: "denied X and Y" → "The patient denied X. // The patient denied Y."
- Split medications: drug name, dose, frequency, indication as separate facts (repeat drug name in each).
- Include all clinical findings, diagnoses, plans, and demographics.

Example 1:
Note: "The patient denied lip or throat swelling."
Facts: The patient denied lip swelling. // The patient denied throat swelling.

Example 2:
Note: "Start Singulair 10 mg once daily for asthma."
Facts: The doctor prescribed Singulair. // The Singulair dose is 10 mg. // Singulair is taken once daily. // Singulair is prescribed for asthma.

Clinical note:
{note}
Facts:\
"""




_COVERED_PROMPT = """\
Note sentences:
{candidates}

Is "{fact}" expressed by any sentence above (even if paraphrased)? Answer "Yes" or "No".\
"""

COVERED_TOP_K = 3  # number of note candidates sent to LLM for completeness


def _llm_covered(fact: str, candidates: List[str]) -> int:
    cand_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    prompt = _COVERED_PROMPT.format(candidates=cand_text, fact=fact.strip())
    client = get_bedrock_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.converse(
                modelId=BEDROCK_GEN_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 4, "temperature": 0.0},
            )
            text = resp["output"]["message"]["content"][0]["text"].strip().lower()
            return 1 if text.startswith("yes") else 0
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                tqdm.write(f"  [covered] error (attempt {attempt + 1}): {exc}")
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                tqdm.write(f"  [covered] failed after {MAX_RETRIES} attempts: {exc}")
                return 0
    return 0


FAITH_TOP_K = 5  # top-k transcript facts retrieved for NLI


def _llm_extract(prompt: str, max_tokens: int = 1024) -> List[str]:
    client = get_bedrock_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.converse(
                modelId=BEDROCK_GEN_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
            text = resp["output"]["message"]["content"][0]["text"].strip()
            facts = [f.strip() for f in text.split("//") if f.strip()]
            return [f for f in facts if len(f) > 4]
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                tqdm.write(f"  [factmatch] LLM error (attempt {attempt + 1}): {exc}")
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                tqdm.write(f"  [factmatch] LLM failed after {MAX_RETRIES} attempts: {exc}")
                return []


def extract_transcript_facts(transcript: str) -> List[str]:
    raw = _llm_extract(_TRANSCRIPT_PROMPT.format(transcript=transcript.strip()))
    cleaned = filter_facts(raw)
    if len(raw) != len(cleaned):
        tqdm.write(f"  [factmatch] transcript facts: {len(raw)} raw → {len(cleaned)} after filter")
    return cleaned


def extract_note_facts(note: str) -> List[str]:
    raw = _llm_extract(_NOTE_PROMPT.format(note=note.strip()))
    cleaned = filter_facts(raw)
    if len(raw) != len(cleaned):
        tqdm.write(f"  [factmatch] note facts: {len(raw)} raw → {len(cleaned)} after filter")
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Core matching
# ─────────────────────────────────────────────────────────────────────────────

def _classify_facts(
    query_facts: List[str],
    cand_facts: List[str],
    query_embs: np.ndarray,
    cand_embs: np.ndarray,
    result_key: str,       # "supported" (faithfulness) or "covered" (completeness)
    other_fact_col: str,   # column name for the best-matching candidate fact
) -> List[Dict]:
    """
    For each query fact, retrieve top-k candidates by cosine, run NLI, classify.

    Faithfulness call:
      query = note facts,       cand = transcript facts
      NLI pair = (transcript_fact, note_fact)   [premise → hypothesis]

    Completeness call:
      query = transcript facts, cand = note facts
      NLI pair = (note_fact, transcript_fact)   [premise → hypothesis]
    """
    if not query_facts or not cand_facts:
        return []

    sim: np.ndarray = query_embs @ cand_embs.T

    # ── Completeness pass: LLM yes/no over top-COVERED_TOP_K note candidates ──
    if result_key == "covered":
        rows: List[Dict] = []
        for i, qfact in enumerate(query_facts):
            k = min(COVERED_TOP_K, len(cand_facts))
            top_idx = np.argsort(sim[i])[::-1][:k]
            top_cos = sim[i][top_idx]
            top_cands = [cand_facts[j] for j in top_idx]

            top_cos_val = round(float(top_cos[0]), 4) if len(top_cos) > 0 else 0.0
            meta = {
                "fact_idx": i,
                "fact": qfact,
                "top_cos": top_cos_val,
                other_fact_col: top_cands[0] if top_cands else "",
            }

            if top_cos_val < COSINE_FLOOR:
                val, llm_ans = 0, "no"
            else:
                val = _llm_covered(qfact, top_cands)
                llm_ans = "yes" if val else "no"

            rows.append({**meta, result_key: val, "top_nli": llm_ans})
        return rows

    # ── Faithfulness pass: NLI over top-FAITH_TOP_K, stop at first entailment ──
    rows: List[Dict] = []
    for i, qfact in enumerate(query_facts):
        k = min(FAITH_TOP_K, len(cand_facts))
        top_idx = np.argsort(sim[i])[::-1][:k]
        top_cos = sim[i][top_idx]
        top_cands = [cand_facts[j] for j in top_idx]

        top_cos_val = round(float(top_cos[0]), 4) if len(top_cos) > 0 else 0.0
        best_fact = top_cands[0] if top_cands else ""

        if top_cos_val < COSINE_FLOOR:
            val, halluc, nli_label = 0, "fabrication", "neutral"
        elif top_cos_val >= HIGH_COS_SUPPORT:
            val, halluc, nli_label = 1, "", "entailment"
        else:
            # Batch all top-k NLI pairs; iterate in cosine order, stop at first entailment
            pairs = [(c, qfact) for c in top_cands]
            labels = predict_pair_nli_labels(pairs)
            nli_label = labels[0]
            val, halluc = 0, "fabrication"
            for cand, label in zip(top_cands, labels):
                if label == "entailment":
                    val, halluc, nli_label, best_fact = 1, "", "entailment", cand
                    break
            # Negation only from top-1 pair to avoid spurious contradictions
            if val == 0 and labels[0] == "contradiction" and top_cos_val >= NEGATION_COS_MIN:
                val, halluc, nli_label = 0, "negation", "contradiction"

        meta = {
            "fact_idx": i,
            "fact": qfact,
            "top_cos": top_cos_val,
            other_fact_col: best_fact,
        }
        rows.append({**meta, result_key: val, "halluc_type": halluc, "top_nli": nli_label})

    return rows


def score_note_facts(
    note_facts: List[str],
    trans_facts: List[str],
    note_embs: np.ndarray,
    trans_embs: np.ndarray,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (faithfulness_rows, completeness_rows).

    faithfulness: note facts checked against transcript facts (precision)
    completeness: transcript facts checked against note facts  (recall)
    """
    faith_rows = _classify_facts(
        note_facts, trans_facts, note_embs, trans_embs,
        result_key="supported", other_fact_col="top_trans_fact",
    )
    return faith_rows, []


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_transcript_facts(
    sample_idx: int,
    transcript: str,
    fact_dir: Path,
    use_cache: bool,
) -> List[str]:
    cache = fact_dir / f"sample_{sample_idx:03d}_transcript_facts.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())["facts"]

    tqdm.write(f"  [factmatch] extracting transcript facts for sample {sample_idx}")
    facts = extract_transcript_facts(transcript)
    cache.write_text(json.dumps({"sample_idx": sample_idx, "facts": facts}, indent=2))
    tqdm.write(f"  [factmatch] {len(facts)} transcript facts extracted")
    return facts


def score_generation_file(
    gen_path: Path,
    out_dir: Path,
    use_cache: bool,
    max_notes: int = 5,
) -> Dict:
    saved      = json.loads(gen_path.read_text())
    sample_idx = int(saved["sample_idx"])
    K_actual   = len(saved["notes"])
    notes      = saved["notes"][:max_notes]
    transcript = saved.get("transcript", "")

    if not transcript or not notes:
        return {"sample_idx": sample_idx, "status": "missing_data", "K_actual": K_actual}

    tqdm.write(f"  [factmatch] {K_actual} generations available, scoring {len(notes)}")

    trans_facts = load_transcript_facts(sample_idx, transcript, out_dir, use_cache)
    if not trans_facts:
        return {"sample_idx": sample_idx, "status": "no_transcript_facts", "K_actual": K_actual}

    # Embed transcript facts once; reuse across all notes for this sample
    trans_embs = embed_facts(trans_facts)

    faith_summary: List[float] = []

    for ref_idx, note in enumerate(notes):
        faith_path = out_dir / f"sample_{sample_idx:03d}_note_{ref_idx:02d}_factmatch.csv"

        if use_cache and faith_path.exists():
            tqdm.write(f"  [cache] {faith_path.name}")
            faith_summary.append(float(pd.read_csv(faith_path)["supported"].mean()))
            continue

        note_facts = extract_note_facts(note)
        if not note_facts:
            tqdm.write(f"  [factmatch] sample {sample_idx} note {ref_idx:02d}: no facts extracted")
            continue

        note_embs = embed_facts(note_facts)

        faith_rows, _ = score_note_facts(
            note_facts, trans_facts, note_embs, trans_embs
        )

        if not faith_rows:
            continue

        pd.DataFrame(faith_rows).to_csv(faith_path, index=False)

        faith_val = float(np.mean([r["supported"] for r in faith_rows]))
        faith_summary.append(faith_val)

        n_neg = sum(1 for r in faith_rows if r.get("halluc_type") == "negation")
        n_fab = sum(1 for r in faith_rows if r.get("halluc_type") == "fabrication")
        tqdm.write(
            f"  [factmatch] sample {sample_idx} note {ref_idx:02d}: "
            f"{len(note_facts)} note facts | {len(trans_facts)} transcript facts | "
            f"faithfulness={faith_val:.2%} negation={n_neg} fabrication={n_fab}"
        )

    if not faith_summary:
        return {"sample_idx": sample_idx, "status": "no_results", "K_actual": K_actual}

    return {
        "sample_idx": sample_idx,
        "status": "ok",
        "K_actual":           K_actual,
        "notes_scored":       len(faith_summary),
        "mean_faithfulness":  round(float(np.mean(faith_summary)), 4),
        "n_transcript_facts": len(trans_facts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage
# ─────────────────────────────────────────────────────────────────────────────

def stage_score(
    start: int,
    end: int,
    out_dir: Path,
    use_cache: bool,
    max_notes: int = 5,
) -> pd.DataFrame:
    gen_dir  = out_dir / "generations"
    fact_dir = out_dir / FACTMATCH_SUBDIR
    fact_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for sample_idx in tqdm(range(start, end), desc="factmatch", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            rows.append({"sample_idx": sample_idx, "status": "no_generations"})
            continue

        print(f"\n[factmatch] sample {sample_idx}")
        rows.append(score_generation_file(gen_path, fact_dir, use_cache, max_notes))
        clear_memory()

    df = pd.DataFrame(rows)
    results_path = out_dir / "factmatch_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n[factmatch] wrote {results_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fact-level faithfulness + completeness scorer (embed+NLI hybrid)"
    )
    p.add_argument("--start",     type=int, default=0)
    p.add_argument("--end",       type=int, default=132)
    p.add_argument("--out",       default=DEFAULT_OUT_DIR,
                   help=f"Root output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--max-notes", type=int, default=3,
                   help="Number of generated notes to score per sample (default: 5)")
    p.add_argument("--no-cache",  action="store_true",
                   help="Re-score even if output CSVs already exist")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Samples:      {args.start} to {args.end - 1}")
    print(f"Max notes:    {args.max_notes} per sample")
    print(f"Output:       {out_dir / FACTMATCH_SUBDIR}")
    print(f"LLM:          {BEDROCK_GEN_MODEL}")
    print(f"Embedder:     {EMBED_MODEL}")
    print(f"Top-k NLI:    {TOP_K}  (cosine floor: {COSINE_FLOOR})")
    print(f"Cache:        {'on' if not args.no_cache else 'off'}")

    stage_score(args.start, args.end, out_dir, not args.no_cache, args.max_notes)


if __name__ == "__main__":
    main()
