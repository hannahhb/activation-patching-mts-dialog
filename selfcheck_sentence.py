"""
selfcheck_sentence.py
=====================
SelfCheckGPT-LLM style sentence-level faithfulness scorer.

For each generated note (as reference) and each of its sentences, prompts
a Llama 3.1 8B to answer Yes/No: "Does the context above support this sentence?"
The context is the matching SOAP section from each of the K-1 other generations.

  Uncertainty(s_j) = fraction of "No" responses across K-1 comparisons

This is orthogonal to LUQ: LUQ uses NLI on cross-generation consistency;
SelfCheckGPT-LLM uses an explicit binary prompt, better for nuanced clinical
phrasing that tricks softmax-based entailment.

Output: luq_out/llama/selfcheck/sample_NNN_note_KK_selfcheck.csv
  sentence_idx, sentence, uncertainty, n_yes, n_no

Usage:
  python selfcheck_sentence.py --stage score --start 0 --end 10
  python selfcheck_sentence.py --cost-estimate --start 0 --end 132
  python selfcheck_sentence.py --stage score --backend hf --start 0 --end 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse text-processing utilities from luq_sentence.py
from luq_sentence import (
    DEFAULT_OUT_DIR,
    BEDROCK_GEN_MODEL,
    BEDROCK_REGION,
    assign_sentence_sections,
    get_bedrock_client,
    get_transcript_and_gold,
    load_aci_bench,
    parse_sections,
    split_sentences,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SELFCHECK_OUT_SUBDIR = "selfcheck"

# Bedrock on-demand pricing for Llama 3.1 8B (us-east-1, 2024)
BEDROCK_LLAMA_INPUT_PRICE_PER_1K = 0.00022   # USD per 1K input tokens
BEDROCK_LLAMA_OUTPUT_PRICE_PER_1K = 0.00022  # USD per 1K output tokens

# Rough token estimates for cost calculation
EST_CONTEXT_TOKENS = 350   # typical SOAP section
EST_PROMPT_OVERHEAD = 80   # instruction text + sentence itself
EST_INPUT_TOKENS = EST_CONTEXT_TOKENS + EST_PROMPT_OVERHEAD
EST_OUTPUT_TOKENS = 2      # "Yes" or "No"
EST_SENTENCES_PER_NOTE = 17
EST_K = 10

RETRY_SLEEP_SEC = 2.0
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SELFCHECK_PROMPT = """\
Context:
{context}

Sentence: "{sentence}"

Does the context above explicitly support this sentence? Answer with only "Yes" or "No".\
"""


def build_selfcheck_prompt(sentence: str, context: str) -> str:
    return SELFCHECK_PROMPT.format(
        context=context.strip(),
        sentence=sentence.strip(),
    )


# ---------------------------------------------------------------------------
# Backend: Bedrock
# ---------------------------------------------------------------------------

def _call_bedrock(sentence: str, context: str) -> Optional[str]:
    """Returns 'yes', 'no', or None on failure."""
    client = get_bedrock_client()
    prompt_text = build_selfcheck_prompt(sentence, context)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.converse(
                modelId=BEDROCK_GEN_MODEL,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt_text}],
                }],
                inferenceConfig={
                    "maxTokens": 8,
                    "temperature": 0.0,
                },
            )
            text = response["output"]["message"]["content"][0]["text"].strip().lower()
            if text.startswith("yes"):
                return "yes"
            if text.startswith("no"):
                return "no"
            # Unexpected response — treat as ambiguous, default to "no"
            tqdm.write(f"  [selfcheck] unexpected response: {text!r} — treating as 'no'")
            return "no"
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                tqdm.write(f"  [selfcheck] bedrock error (attempt {attempt+1}): {exc}")
                time.sleep(RETRY_SLEEP_SEC * (attempt + 1))
            else:
                tqdm.write(f"  [selfcheck] bedrock failed after {MAX_RETRIES} attempts: {exc}")
                return None


# ---------------------------------------------------------------------------
# Backend: HuggingFace
# ---------------------------------------------------------------------------

_hf_pipeline = None


def get_hf_pipeline(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline
        print(f"[hf] Loading {model_name} — requires ~16 GB VRAM")
        _hf_pipeline = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            max_new_tokens=8,
            temperature=0.01,
            do_sample=False,
        )
        print("[hf] Pipeline ready")
    return _hf_pipeline


def _call_hf(sentence: str, context: str, model_name: str) -> Optional[str]:
    pipe = get_hf_pipeline(model_name)
    messages = [{"role": "user", "content": build_selfcheck_prompt(sentence, context)}]
    try:
        output = pipe(messages)
        text = output[0]["generated_text"][-1]["content"].strip().lower()
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        tqdm.write(f"  [selfcheck] unexpected hf response: {text!r} — treating as 'no'")
        return "no"
    except Exception as exc:
        tqdm.write(f"  [selfcheck] hf error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def query_yes_no(
    sentence: str,
    context: str,
    backend: str,
    hf_model: str,
) -> Optional[str]:
    if backend == "bedrock":
        return _call_bedrock(sentence, context)
    return _call_hf(sentence, context, hf_model)


def score_note(
    ref_note: str,
    other_notes: List[str],
    backend: str,
    hf_model: str,
) -> List[Dict]:
    """
    Score every sentence in ref_note against each note in other_notes.

    Returns a list of dicts with keys:
      sentence_idx, sentence, uncertainty, n_yes, n_no
    """
    sentences = split_sentences(ref_note)
    if not sentences:
        return []

    sent_sections = assign_sentence_sections(ref_note, sentences)
    n = len(sentences)
    yes_counts = [0] * n
    no_counts = [0] * n

    for other_note in other_notes:
        other_secs = parse_sections(other_note)

        for sent_idx, (sent, section) in enumerate(zip(sentences, sent_sections)):
            # Section-matched context; fall back to full note
            context = other_secs.get(section) or other_secs.get("all", other_note)
            if not context or not context.strip():
                context = other_note

            answer = query_yes_no(sent, context, backend, hf_model)
            if answer == "yes":
                yes_counts[sent_idx] += 1
            else:
                # None (failed) is conservatively counted as "no"
                no_counts[sent_idx] += 1

    rows = []
    for sent_idx in range(n):
        total = yes_counts[sent_idx] + no_counts[sent_idx]
        uncertainty = no_counts[sent_idx] / total if total > 0 else 1.0
        rows.append({
            "sentence_idx": sent_idx,
            "sentence": sentences[sent_idx],
            "uncertainty": round(uncertainty, 4),
            "n_yes": yes_counts[sent_idx],
            "n_no": no_counts[sent_idx],
        })
    return rows


def score_generation_file(
    gen_path: Path,
    out_dir: Path,
    backend: str,
    hf_model: str,
    use_cache: bool,
) -> Dict:
    saved = json.loads(gen_path.read_text())
    sample_idx = int(saved["sample_idx"])
    notes = saved["notes"]
    K = len(notes)

    if K < 2:
        return {"sample_idx": sample_idx, "status": "insufficient_generations", "K_actual": K}

    summary_rows = []

    for ref_idx, ref_note in enumerate(notes):
        out_path = out_dir / f"sample_{sample_idx:03d}_note_{ref_idx:02d}_selfcheck.csv"
        if use_cache and out_path.exists():
            tqdm.write(f"  [cache] {out_path.name}")
            df = pd.read_csv(out_path)
            summary_rows.append(float(df["uncertainty"].mean()))
            continue

        other_notes = [n for i, n in enumerate(notes) if i != ref_idx]
        tqdm.write(f"  [selfcheck] sample {sample_idx} note {ref_idx:02d}: "
                   f"{len(split_sentences(ref_note))} sentences × {len(other_notes)} others")

        rows = score_note(ref_note, other_notes, backend, hf_model)
        if not rows:
            tqdm.write(f"  [selfcheck] sample {sample_idx} note {ref_idx:02d}: no sentences")
            continue

        pd.DataFrame(rows).to_csv(out_path, index=False)
        summary_rows.append(float(np.mean([r["uncertainty"] for r in rows])))

    if not summary_rows:
        return {"sample_idx": sample_idx, "status": "no_sentences", "K_actual": K}

    return {
        "sample_idx": sample_idx,
        "status": "ok",
        "K_actual": K,
        "notes_scored": len(summary_rows),
        "mean_uncertainty": round(float(np.mean(summary_rows)), 4),
        "max_uncertainty": round(float(np.max(summary_rows)), 4),
    }


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------

def stage_score(
    start: int,
    end: int,
    out_dir: Path,
    backend: str,
    hf_model: str,
    use_cache: bool,
) -> pd.DataFrame:
    gen_dir = out_dir / "generations"
    sc_dir = out_dir / SELFCHECK_OUT_SUBDIR
    sc_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for sample_idx in tqdm(range(start, end), desc="selfcheck", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            rows.append({"sample_idx": sample_idx, "status": "no_generations"})
            continue

        print(f"\n[selfcheck] sample {sample_idx}")
        rows.append(score_generation_file(gen_path, sc_dir, backend, hf_model, use_cache))

    df = pd.DataFrame(rows)
    results_path = out_dir / "selfcheck_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n[selfcheck] wrote {results_path}")
    return df


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

def cost_estimate(start: int, end: int, gen_dir: Path) -> None:
    """
    Estimate Bedrock Llama 3.1 8B API cost for the selfcheck run.
    Uses actual generation files where available, falls back to defaults.
    """
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for sample_idx in range(start, end):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if gen_path.exists():
            saved = json.loads(gen_path.read_text())
            notes = saved["notes"]
            K = len(notes)
            if K < 2:
                continue
            # For each note as reference: count sentences × (K-1) comparisons
            total_sents = sum(len(split_sentences(n)) for n in notes)
            calls_this_sample = total_sents * (K - 1)
        else:
            K = EST_K
            calls_this_sample = K * EST_SENTENCES_PER_NOTE * (K - 1)

        total_calls += calls_this_sample
        total_input_tokens += calls_this_sample * EST_INPUT_TOKENS
        total_output_tokens += calls_this_sample * EST_OUTPUT_TOKENS

    input_cost = total_input_tokens / 1000 * BEDROCK_LLAMA_INPUT_PRICE_PER_1K
    output_cost = total_output_tokens / 1000 * BEDROCK_LLAMA_OUTPUT_PRICE_PER_1K
    total_cost = input_cost + output_cost

    print(f"\n{'='*56}")
    print(f"  SelfCheckGPT Cost Estimate — Bedrock Llama 3.1 8B")
    print(f"{'='*56}")
    print(f"  Samples:           {start} to {end-1} ({end - start} samples)")
    print(f"  Total LLM calls:   {total_calls:,}")
    print(f"  Input tokens:      {total_input_tokens:,}  (~{EST_INPUT_TOKENS} per call)")
    print(f"  Output tokens:     {total_output_tokens:,}  (~{EST_OUTPUT_TOKENS} per call)")
    print(f"  Input cost:        ${input_cost:.4f}")
    print(f"  Output cost:       ${output_cost:.4f}")
    print(f"  TOTAL:             ${total_cost:.4f}")
    print(f"{'='*56}")
    print(f"\n  Note: Bedrock Llama 3.1 8B does not support Anthropic-style")
    print(f"  prompt caching. Each of the K-1 calls per sentence is billed")
    print(f"  in full. The constant instruction prefix (~{EST_PROMPT_OVERHEAD} tokens) is")
    print(f"  re-sent each time.")
    print(f"\n  With an Anthropic model (e.g. Haiku 4.5) and prompt caching,")
    print(f"  the prefix could be cached at a 90% discount, reducing input")
    print(f"  cost to ~${input_cost * (0.1 + 0.9 * EST_PROMPT_OVERHEAD/EST_INPUT_TOKENS):.4f}.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SelfCheckGPT-LLM sentence-level faithfulness scorer for ACI-Bench SOAP notes"
    )
    parser.add_argument("--stage", choices=["score"], default="score",
                        help="Only 'score' is supported (generation reuses luq_sentence.py output)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=132)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help=f"Root output dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--backend", choices=["bedrock", "hf"], default="bedrock")
    parser.add_argument("--hf-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID (only used with --backend hf)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-score even if output CSV already exists")
    parser.add_argument("--cost-estimate", action="store_true",
                        help="Print cost estimate and exit (no API calls made)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()

    if args.cost_estimate:
        gen_dir = out_dir / "generations"
        cost_estimate(args.start, args.end, gen_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    print(f"Stage:   {args.stage}")
    print(f"Samples: {args.start} to {args.end - 1}")
    print(f"Backend: {args.backend}" +
          (f" ({args.hf_model})" if args.backend == "hf" else f" ({BEDROCK_GEN_MODEL})"))
    print(f"Output:  {out_dir / SELFCHECK_OUT_SUBDIR}")
    print(f"Cache:   {'on' if use_cache else 'off'}")

    stage_score(args.start, args.end, out_dir, args.backend, args.hf_model, use_cache)


if __name__ == "__main__":
    main()
