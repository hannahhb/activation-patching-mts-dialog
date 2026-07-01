"""
faithfulness_llm_sentence.py
============================
LLM-as-judge transcript-grounded faithfulness scorer (MiniCheck-style).

For each sentence in each generated note, asks Llama 3.1 8B:
  "Is this sentence supported by the transcript? Yes or No."

One call per sentence — no cross-generation comparisons needed.

  Faithfulness(s_j) = 1 if "Yes", 0 if "No"
  Uncertainty(s_j)  = 1 - Faithfulness(s_j)

Output: luq_out/llama/faithfulness_llm/sample_NNN_note_KK_faithfulness_llm.csv
  sentence_idx, sentence, supported, uncertainty

Usage:
  python faithfulness_llm_sentence.py --start 0 --end 10
  python faithfulness_llm_sentence.py --cost-estimate --start 0 --end 132
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from luq_sentence import (
    DEFAULT_OUT_DIR,
    BEDROCK_GEN_MODEL,
    get_bedrock_client,
    split_sentences,
)

BEDROCK_GEN_MODEL = "us.meta.llama3-3-70b-instruct-v1:0"


FAITHFULNESS_LLM_SUBDIR = "faithfulness_llm"

BEDROCK_INPUT_PRICE_PER_1K  = 0.00022
BEDROCK_OUTPUT_PRICE_PER_1K = 0.00022
EST_TRANSCRIPT_TOKENS = 350
EST_SENTENCE_TOKENS   = 20
EST_OVERHEAD_TOKENS   = 40
EST_INPUT_TOKENS      = EST_TRANSCRIPT_TOKENS + EST_SENTENCE_TOKENS + EST_OVERHEAD_TOKENS
EST_OUTPUT_TOKENS     = 2
EST_SENTENCES_PER_NOTE = 17
EST_K = 10

MAX_RETRIES   = 5
RETRY_SLEEP   = 2.0

PROMPT_TEMPLATE = """\
You are checking whether a sentence from a clinical SOAP note is supported by the patient transcript.

Rules:
- Answer "Yes" if the factual content of the sentence is present in the transcript, even if phrased differently.
- The note is written after the encounter, so plans or intentions stated in the transcript (e.g. "I'm going to prescribe") may appear as completed actions in the note (e.g. "I prescribed"). Treat tense differences as acceptable.
- Answer "No" only if the sentence makes a factual claim that contradicts or is absent from the transcript.

Transcript:
{transcript}

Sentence: "{sentence}"

Answer with only "Yes" or "No".\
"""

STRUCTURAL_LABEL_RE = re.compile(
    r"^(Subjective|Objective|Assessment\s*(/\s*Problem\s*List)?|Problem\s*List"
    r"|Plan|Follow[- ]?up|HPI|Current\s*medications?|Review\s*of\s*systems?"
    r"|Vital\s*signs?|Physical\s*exam?|Test\s*Results?)\s*:?\s*$"
    r"|^Here\s+is\s+the\s+SOAP.*:\s*$",
    re.IGNORECASE,
)


def build_prompt(transcript: str, sentence: str) -> str:
    return PROMPT_TEMPLATE.format(
        transcript=transcript.strip(),
        sentence=sentence.strip(),
    )


def query_supported(transcript: str, sentence: str) -> Optional[bool]:
    """Returns True (supported), False (not supported), or None (failed)."""
    client = get_bedrock_client()
    prompt  = build_prompt(transcript, sentence)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.converse(
                modelId=BEDROCK_GEN_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 8, "temperature": 0.0},
            )
            text = response["output"]["message"]["content"][0]["text"].strip().lower()
            if text.startswith("yes"):
                return True
            if text.startswith("no"):
                return False
            tqdm.write(f"  [faith_llm] unexpected response: {text!r} — treating as No")
            return False
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                tqdm.write(f"  [faith_llm] error (attempt {attempt + 1}): {exc}")
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                tqdm.write(f"  [faith_llm] failed after {MAX_RETRIES} attempts: {exc}")
                return None


DECOMPOSE_PROMPT = """\
Break the following sentence into atomic facts — the smallest independent claims that can each be verified separately. Output one fact per line with no bullet points or numbering.

Sentence: "{sentence}"

Atomic facts:\
"""


def decompose_atoms(sentence: str) -> List[str]:
    """Call Llama to split a sentence into atomic facts. Returns list of fact strings."""
    client = get_bedrock_client()
    prompt = DECOMPOSE_PROMPT.format(sentence=sentence.strip())
    for attempt in range(MAX_RETRIES):
        try:
            response = client.converse(
                modelId=BEDROCK_GEN_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 256, "temperature": 0.0},
            )
            text = response["output"]["message"]["content"][0]["text"].strip()
            atoms = [
                line.lstrip("-•*0123456789.) ").strip()
                for line in text.splitlines()
                if line.strip()
            ]
            return [a for a in atoms if len(a) > 4]
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                tqdm.write(f"  [faith_llm] decompose error (attempt {attempt + 1}): {exc}")
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                tqdm.write(f"  [faith_llm] decompose failed: {exc}")
                return [sentence]  # fall back to treating sentence as single atom


def score_note(
    transcript: str,
    note: str,
) -> tuple[List[Dict], List[Dict]]:
    """
    Returns (sentence_rows, atom_rows).

    sentence_rows: one row per sentence with binary supported + atom_faithfulness fraction
    atom_rows:     one row per atom with sentence_idx, atom_idx, atom, supported
    """
    sentences = split_sentences(note)
    sentence_rows: List[Dict] = []
    atom_rows: List[Dict] = []

    for idx, sent in enumerate(sentences):
        if STRUCTURAL_LABEL_RE.match(sent.strip()):
            sentence_rows.append({
                "sentence_idx": idx,
                "sentence": sent,
                "supported": 1,
                "uncertainty": 0,
                "atom_faithfulness": 1.0,
            })
            continue

        # Sentence-level check
        result = query_supported(transcript, sent)
        supported = int(result) if result is not None else 0

        # Atomic decomposition + per-atom check
        atoms = decompose_atoms(sent)
        atom_supported = []
        for atom_idx, atom in enumerate(atoms):
            atom_result = query_supported(transcript, atom)
            a_sup = int(atom_result) if atom_result is not None else 0
            atom_supported.append(a_sup)
            atom_rows.append({
                "sentence_idx": idx,
                "atom_idx": atom_idx,
                "atom": atom,
                "supported": a_sup,
                "uncertainty": 1 - a_sup,
            })

        atom_faithfulness = float(np.mean(atom_supported)) if atom_supported else float(supported)

        sentence_rows.append({
            "sentence_idx": idx,
            "sentence": sent,
            "supported": supported,
            "uncertainty": 1 - supported,
            "atom_faithfulness": round(atom_faithfulness, 4),
        })

    return sentence_rows, atom_rows


def score_generation_file(
    gen_path: Path,
    out_dir: Path,
    use_cache: bool,
    max_notes: int = 5,
) -> Dict:
    saved = json.loads(gen_path.read_text())
    sample_idx = int(saved["sample_idx"])
    notes      = saved["notes"][:max_notes]
    transcript = saved.get("transcript", "")

    if not transcript or not notes:
        return {"sample_idx": sample_idx, "status": "missing_data"}

    summary = []
    for ref_idx, note in enumerate(notes):
        out_path      = out_dir / f"sample_{sample_idx:03d}_note_{ref_idx:02d}_faithfulness_llm.csv"
        out_path_atoms = out_dir / f"sample_{sample_idx:03d}_note_{ref_idx:02d}_atoms_llm.csv"

        if use_cache and out_path.exists() and out_path_atoms.exists():
            tqdm.write(f"  [cache] {out_path.name}")
            df = pd.read_csv(out_path)
            col = "atom_faithfulness" if "atom_faithfulness" in df.columns else "supported"
            summary.append(float(df[col].mean()))
            continue

        sentence_rows, atom_rows = score_note(transcript, note)
        if not sentence_rows:
            continue

        pd.DataFrame(sentence_rows).to_csv(out_path, index=False)
        pd.DataFrame(atom_rows).to_csv(out_path_atoms, index=False)

        mean_s = float(np.mean([r["atom_faithfulness"] for r in sentence_rows]))
        summary.append(mean_s)
        tqdm.write(
            f"  [faith_llm] sample {sample_idx} note {ref_idx:02d}: "
            f"{len(sentence_rows)} sentences, {len(atom_rows)} atoms, "
            f"atom_faithfulness={mean_s:.2%}"
        )

    if not summary:
        return {"sample_idx": sample_idx, "status": "no_sentences"}

    return {
        "sample_idx": sample_idx,
        "status": "ok",
        "K_actual": len(notes),
        "notes_scored": len(summary),
        "mean_atom_faithfulness": round(float(np.mean(summary)), 4),
        "min_atom_faithfulness":  round(float(np.min(summary)), 4),
    }


def stage_score(start: int, end: int, out_dir: Path, use_cache: bool, max_notes: int = 5) -> pd.DataFrame:
    gen_dir  = out_dir / "generations"
    faith_dir = out_dir / FAITHFULNESS_LLM_SUBDIR
    faith_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for sample_idx in tqdm(range(start, end), desc="faithfulness_llm", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            rows.append({"sample_idx": sample_idx, "status": "no_generations"})
            continue

        print(f"\n[faith_llm] sample {sample_idx}")
        rows.append(score_generation_file(gen_path, faith_dir, use_cache, max_notes))

    df = pd.DataFrame(rows)
    results_path = out_dir / "faithfulness_llm_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n[faith_llm] wrote {results_path}")
    return df


def cost_estimate(start: int, end: int, gen_dir: Path, max_notes: int = 5) -> None:
    total_calls = 0
    for sample_idx in range(start, end):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if gen_path.exists():
            saved = json.loads(gen_path.read_text())
            total_calls += sum(len(split_sentences(n)) for n in saved["notes"][:max_notes])
        else:
            total_calls += max_notes * EST_SENTENCES_PER_NOTE

    input_cost  = total_calls * EST_INPUT_TOKENS  / 1000 * BEDROCK_INPUT_PRICE_PER_1K
    output_cost = total_calls * EST_OUTPUT_TOKENS / 1000 * BEDROCK_OUTPUT_PRICE_PER_1K
    total_cost  = input_cost + output_cost

    print(f"\n{'='*52}")
    print(f"  Faithfulness LLM Cost Estimate — Llama 3.1 8B")
    print(f"{'='*52}")
    print(f"  Samples:        {start} to {end - 1} ({end - start} samples)")
    print(f"  Total calls:    {total_calls:,}  (1 per sentence per note)")
    print(f"  Est. input tok: {total_calls * EST_INPUT_TOKENS:,}  (~{EST_INPUT_TOKENS}/call)")
    print(f"  Input cost:     ${input_cost:.4f}")
    print(f"  Output cost:    ${output_cost:.4f}")
    print(f"  TOTAL:          ${total_cost:.4f}")
    print(f"{'='*52}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-as-judge transcript faithfulness scorer for ACI-Bench SOAP notes"
    )
    p.add_argument("--start",     type=int, default=0)
    p.add_argument("--end",       type=int, default=132)
    p.add_argument("--out",       default=DEFAULT_OUT_DIR)
    p.add_argument("--max-notes", type=int, default=5,
                   help="Number of generations to score per sample (default: 5)")
    p.add_argument("--no-cache",      action="store_true")
    p.add_argument("--cost-estimate", action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out).resolve()

    if args.cost_estimate:
        cost_estimate(args.start, args.end, out_dir / "generations", args.max_notes)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Samples:   {args.start} to {args.end - 1}")
    print(f"Max notes: {args.max_notes} per sample")
    print(f"Output:    {out_dir / FAITHFULNESS_LLM_SUBDIR}")
    print(f"Model:     {BEDROCK_GEN_MODEL}")
    print(f"Cache:     {'on' if not args.no_cache else 'off'}")

    stage_score(args.start, args.end, out_dir, not args.no_cache, args.max_notes)


if __name__ == "__main__":
    main()
