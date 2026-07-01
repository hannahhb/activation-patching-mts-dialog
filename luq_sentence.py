"""
luq_sentence_sectionwise_fast.py
================================
Fast sentence-level LUQ for clinical note generations.

Core behaviour:
  - Generate K notes per transcript, or score cached generations.
  - Score every generated note as the reference once.
  - Compare each reference note against every other sampled note.
  - With K=10, this gives 10 x 9 = 90 note-to-note comparisons.
  - Use section-wise premises: a Subjective sentence is checked against the
    other note's Subjective section, Plan against Plan, etc.
  - Use LUQ-style binary entailment-vs-contradiction normalisation.
  - Chunk long sections with overlapping token windows rather than silent truncation.
  - Use premise-first micro-batching and cached section chunks for speed.

Typical use:
  python luq_sentence_sectionwise_fast.py --stage generate --start 0 --end 10
  python luq_sentence_sectionwise_fast.py --stage score --start 0 --end 10
  python luq_sentence_sectionwise_fast.py --stage all --start 0 --end 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from prompts import build_prompt as build_user_message

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

BEDROCK_GEN_MODEL = "us.meta.llama3-1-8b-instruct-v1:0"
BEDROCK_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

DATASET_REPO = "mkieffer/ACI-Bench-MedARC"
DATASET_CONFIG = "aci"
DATASET_SPLIT = "test1"

DEFAULT_K = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_OUT_DIR = "luq_out/llama"

PAIR_MAX_TOKENS = 512
SPECIAL_TOKEN_RESERVE = 8
MIN_PREMISE_WINDOW = 64
DEFAULT_STRIDE_RATIO = 0.5
DEFAULT_PREMISE_WINDOW = 448
WINDOW_BUCKET = 32


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


CUDA_AVAILABLE = _cuda_available()
NLI_BATCH_SIZE = 16 if CUDA_AVAILABLE else 4
MICRO_BATCH_PAIRS = 128 if CUDA_AVAILABLE else 16
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "cpu"


# -----------------------------------------------------------------------------
# Dataset and generation
# -----------------------------------------------------------------------------

def load_aci_bench(split: str = DATASET_SPLIT):
    from datasets import load_dataset
    print(f"[data] Loading {DATASET_REPO} config={DATASET_CONFIG} split={split}")
    return load_dataset(DATASET_REPO, DATASET_CONFIG, split=split)


def get_transcript_and_gold(row: dict) -> Tuple[str, str]:
    transcript_cols = ["src", "dialogue", "conversation", "transcript", "input"]
    gold_cols = ["tgt", "note", "reference", "summary", "output"]

    transcript = next((row[c] for c in transcript_cols if c in row), None)
    if transcript is None:
        raise KeyError(f"No transcript column found in {list(row.keys())}")

    gold = next((row[c] for c in gold_cols if c in row), "")
    return transcript, gold


_bedrock_client = None


def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        from botocore.config import Config

        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=BEDROCK_REGION,
            config=Config(
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=20,
                read_timeout=120,
            ),
        )
        print(f"[bedrock] Client initialised in {BEDROCK_REGION}")
    return _bedrock_client


def generate_note(transcript: str) -> str:
    client = get_bedrock_client()
    response = client.converse(
        modelId=BEDROCK_GEN_MODEL,
        messages=[{"role": "user", "content": [{"text": build_user_message(transcript)}]}],
        inferenceConfig={
            "maxTokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
        },
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def generate_notes(transcript: str, k: int) -> List[str]:
    notes: List[str] = []
    for i in tqdm(range(k), desc="  generate", unit="note", leave=False):
        try:
            notes.append(generate_note(transcript))
        except Exception as exc:
            tqdm.write(f"  [generate] note {i + 1}/{k} failed: {exc}")
    return notes


# -----------------------------------------------------------------------------
# Sentence, section, and claim processing
# -----------------------------------------------------------------------------

PERIOD_SPLIT = re.compile(r"(?<=\.)\s+(?=[A-Z])")
ABBREV_RE = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
    r"Sep|Oct|Nov|Dec|approx|appt|Dept|No|Vol|Fig)\.",
    re.IGNORECASE,
)
FIELD_PREFIX_RE = re.compile(r"^[A-Za-z /\-]+:\s*")
TEMPLATE_JUNK_RE = re.compile(r"\[unknown\]|\[NOT MENTIONED\]|\[not mentioned\]", re.IGNORECASE)
SEMICOLON_SPLIT_RE = re.compile(r";\s*")
DOT_PLACEHOLDER = "\x00DOT\x00"

SOAP_HEADER_RE = re.compile(
    r"^(Subjective|Objective|Assessment\s*(?:/\s*Problem\s*List)?|Assessment|Problem\s*List|Plan|Follow[- ]?up)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_section(header: str) -> str:
    h = header.strip().lower()
    if h.startswith("subjective"):
        return "subjective"
    if h.startswith("objective"):
        return "objective"
    if h.startswith("assessment") or h.startswith("problem"):
        return "assessment"
    if h.startswith("plan"):
        return "plan"
    if h.startswith("follow"):
        return "followup"
    return "all"


def parse_sections(note: str) -> Dict[str, str]:
    """Return top-level SOAP sections. Falls back to {'all': note}."""
    matches = list(SOAP_HEADER_RE.finditer(note))
    if not matches:
        return {"all": note}

    sections: Dict[str, str] = {}
    for i, match in enumerate(matches):
        name = normalize_section(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        text = note[start:end].strip()
        # If duplicate headers occur, append instead of overwriting.
        sections[name] = (sections.get(name, "") + "\n" + text).strip()

    sections["all"] = note
    return sections


def split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-•*#+").strip()
        if not line:
            continue
        protected = ABBREV_RE.sub(lambda m: m.group(0).replace(".", DOT_PLACEHOLDER), line)
        for part in PERIOD_SPLIT.split(protected):
            sent = part.replace(DOT_PLACEHOLDER, ".").strip()
            if sent:
                sentences.append(sent)
    return sentences


def assign_sentence_sections(note: str, sentences: List[str]) -> List[str]:
    """Assign each reference sentence to a SOAP section by character location."""
    matches = list(SOAP_HEADER_RE.finditer(note))
    if not matches:
        return ["all"] * len(sentences)

    spans = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        spans.append((start, end, normalize_section(match.group(1))))

    out: List[str] = []
    search_pos = 0
    for sentence in sentences:
        pos = note.find(sentence, search_pos)
        section = "all"
        if pos != -1:
            for start, end, name in spans:
                if start <= pos < end:
                    section = name
                    break
            search_pos = pos + len(sentence)
        out.append(section)
    return out


def clean_sentence(sentence: str) -> str:
    sentence = FIELD_PREFIX_RE.sub("", sentence).strip()
    sentence = TEMPLATE_JUNK_RE.sub("", sentence).strip(" ;,.")
    return sentence


def split_subclaims(sentence: str) -> List[str]:
    cleaned = clean_sentence(sentence)
    parts = [p.strip() for p in SEMICOLON_SPLIT_RE.split(cleaned) if p.strip()]
    return parts if len(parts) > 1 else ([cleaned] if cleaned else [])


# -----------------------------------------------------------------------------
# NLI scoring
# -----------------------------------------------------------------------------

_nli = None
_label_indices = None


def get_nli():
    global _nli, _label_indices
    if _nli is None:
        from sentence_transformers import CrossEncoder

        print(f"[nli] Loading {NLI_MODEL_NAME}")
        _nli = CrossEncoder(NLI_MODEL_NAME)
        _label_indices = infer_label_indices(_nli)
        print(f"[nli] Ready. entail={_label_indices[0]}, contradict={_label_indices[1]}")
    return _nli


def infer_label_indices(nli) -> Tuple[int, int]:
    labels = {int(i): str(label).lower() for i, label in nli.model.config.id2label.items()}
    entail = next((i for i, label in labels.items() if "entail" in label), None)
    contradict = next((i for i, label in labels.items() if "contrad" in label), None)
    if entail is None or contradict is None:
        raise ValueError(f"Could not infer entailment/contradiction labels from {labels}")
    return entail, contradict


def chunk_premise_ids(tokenizer, premise_ids: List[int], window_size: int) -> List[str]:
    """Decode overlapping premise windows. No premise tokens are silently discarded."""
    if not premise_ids:
        return [""]
    if len(premise_ids) <= window_size:
        return [tokenizer.decode(premise_ids, skip_special_tokens=True)]

    stride = max(1, int(window_size * DEFAULT_STRIDE_RATIO))
    chunks: List[str] = []
    start = 0
    while start < len(premise_ids):
        end = min(start + window_size, len(premise_ids))
        chunks.append(tokenizer.decode(premise_ids[start:end], skip_special_tokens=True))
        if end == len(premise_ids):
            break
        start += stride
    return chunks


def bucketed_window_size(tokenizer, claim: str) -> int:
    hyp_len = len(tokenizer.encode(claim, add_special_tokens=False))
    available = PAIR_MAX_TOKENS - hyp_len - SPECIAL_TOKEN_RESERVE
    available = max(MIN_PREMISE_WINDOW, available)
    if available >= DEFAULT_PREMISE_WINDOW:
        return DEFAULT_PREMISE_WINDOW
    return max(MIN_PREMISE_WINDOW, (available // WINDOW_BUCKET) * WINDOW_BUCKET)


def binary_luq_support(logits: np.ndarray, entail_idx: int, contradict_idx: int) -> float:
    ec_logits = logits[[entail_idx, contradict_idx]]
    ec_exp = np.exp(ec_logits - np.max(ec_logits))
    return float(ec_exp[0] / np.sum(ec_exp))


def predict_pair_supports(nli, pairs: List[Tuple[str, str]]) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.float64)

    entail_idx, contradict_idx = _label_indices
    raw = np.asarray(
        nli.predict(
            pairs,
            batch_size=NLI_BATCH_SIZE,
            apply_softmax=False,
            show_progress_bar=False,
        )
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    return np.asarray(
        [binary_luq_support(row, entail_idx, contradict_idx) for row in raw],
        dtype=np.float64,
    )


# -----------------------------------------------------------------------------
# LUQ scoring
# -----------------------------------------------------------------------------

def clear_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_reference_claims(notes: List[str], tokenizer):
    """
    Pre-split every generated note once.

    Returns:
      sentences_by_ref[ref_idx]
      claims_by_ref[ref_idx]
      claim_to_sentence_by_ref[ref_idx]
      claim_sections_by_ref[ref_idx]
      window_sizes_by_ref[ref_idx]
    """
    sentences_by_ref: List[List[str]] = []
    claims_by_ref: List[List[str]] = []
    claim_to_sentence_by_ref: List[List[int]] = []
    claim_sections_by_ref: List[List[str]] = []
    window_sizes_by_ref: List[List[int]] = []

    for note in notes:
        sentences = split_sentences(note)
        sent_sections = assign_sentence_sections(note, sentences)
        claims: List[str] = []
        claim_to_sentence: List[int] = []
        claim_sections: List[str] = []
        window_sizes: List[int] = []

        for sent_idx, sentence in enumerate(sentences):
            section = sent_sections[sent_idx] if sent_idx < len(sent_sections) else "all"
            for claim in split_subclaims(sentence):
                claims.append(claim)
                claim_to_sentence.append(sent_idx)
                claim_sections.append(section)
                window_sizes.append(bucketed_window_size(tokenizer, claim))

        sentences_by_ref.append(sentences)
        claims_by_ref.append(claims)
        claim_to_sentence_by_ref.append(claim_to_sentence)
        claim_sections_by_ref.append(claim_sections)
        window_sizes_by_ref.append(window_sizes)

    return (
        sentences_by_ref,
        claims_by_ref,
        claim_to_sentence_by_ref,
        claim_sections_by_ref,
        window_sizes_by_ref,
    )


def compute_luq_all_references(notes: List[str]) -> List[Dict]:
    """
    Compute LUQ for every generated note as reference, preserving all K x (K-1)
    comparisons while using section-wise premise matching for speed.

    For each reference claim, use the matching section of the comparison note as
    the NLI premise. If the comparison note lacks that section, fall back to the
    full note. Each section is chunked with overlap if it exceeds the pair token
    budget, so section-wise scoring is not silent truncation.
    """
    K = len(notes)
    if K < 2:
        return []

    nli = get_nli()
    tokenizer = nli.tokenizer

    (
        sentences_by_ref,
        claims_by_ref,
        claim_to_sentence_by_ref,
        claim_sections_by_ref,
        window_sizes_by_ref,
    ) = build_reference_claims(notes, tokenizer)

    support_sum_by_ref: List[np.ndarray] = [
        np.zeros(len(sentences), dtype=np.float64)
        for sentences in sentences_by_ref
    ]

    # Premise-first: parse/tokenise each sampled note's sections once, then score
    # every other reference against those section premises.
    for premise_idx, sampled_note in enumerate(notes):
        section_text = parse_sections(sampled_note)
        section_ids = {
            sec: tokenizer.encode(text, add_special_tokens=False)
            for sec, text in section_text.items()
        }
        if "all" not in section_ids:
            section_ids["all"] = tokenizer.encode(sampled_note, add_special_tokens=False)

        # Cache chunk lists for this premise by (section, window_size).
        chunk_cache: Dict[Tuple[str, int], List[str]] = {}

        claim_max_by_ref: Dict[int, np.ndarray] = {
            ref_idx: np.full(len(claims_by_ref[ref_idx]), -np.inf, dtype=np.float64)
            for ref_idx in range(K)
            if ref_idx != premise_idx and claims_by_ref[ref_idx]
        }

        pair_buffer: List[Tuple[str, str]] = []
        pair_refs: List[int] = []
        pair_claim_indices: List[int] = []

        def flush_buffer() -> None:
            nonlocal pair_buffer, pair_refs, pair_claim_indices
            if not pair_buffer:
                return
            scores = predict_pair_supports(nli, pair_buffer)
            for score, ref_idx, claim_idx in zip(scores, pair_refs, pair_claim_indices):
                if score > claim_max_by_ref[ref_idx][claim_idx]:
                    claim_max_by_ref[ref_idx][claim_idx] = float(score)
            pair_buffer = []
            pair_refs = []
            pair_claim_indices = []

        for ref_idx in range(K):
            if ref_idx == premise_idx:
                continue

            claims = claims_by_ref[ref_idx]
            sections = claim_sections_by_ref[ref_idx]
            windows = window_sizes_by_ref[ref_idx]
            if not claims:
                continue

            for claim_idx, claim in enumerate(claims):
                requested_sec = sections[claim_idx]
                premise_sec = requested_sec if requested_sec in section_ids else "all"
                window_size = windows[claim_idx]
                cache_key = (premise_sec, window_size)

                if cache_key not in chunk_cache:
                    chunk_cache[cache_key] = chunk_premise_ids(
                        tokenizer,
                        section_ids[premise_sec],
                        window_size,
                    )

                for chunk in chunk_cache[cache_key]:
                    pair_buffer.append((chunk, claim))
                    pair_refs.append(ref_idx)
                    pair_claim_indices.append(claim_idx)
                    if len(pair_buffer) >= MICRO_BATCH_PAIRS:
                        flush_buffer()

        flush_buffer()

        # Convert claim support for this premise into sentence support for each reference.
        for ref_idx, claim_max in claim_max_by_ref.items():
            claim_max[~np.isfinite(claim_max)] = 0.5
            sentence_claim_scores: List[List[float]] = [
                [] for _ in sentences_by_ref[ref_idx]
            ]
            for claim_idx, sent_idx in enumerate(claim_to_sentence_by_ref[ref_idx]):
                sentence_claim_scores[sent_idx].append(float(claim_max[claim_idx]))

            sentence_support = np.zeros(len(sentences_by_ref[ref_idx]), dtype=np.float64)
            for sent_idx, scores in enumerate(sentence_claim_scores):
                sentence_support[sent_idx] = float(np.mean(scores)) if scores else 0.5

            support_sum_by_ref[ref_idx] += sentence_support

        section_names = ",".join(k for k in section_ids.keys() if k != "all") or "all"
        print(
            f"  [luq] premise={premise_idx:02d}: scored against {K - 1} references; "
            f"sections={section_names}; chunk_sets={len(chunk_cache)}"
        )
        clear_memory()

    results: List[Dict] = []
    denom = K - 1
    for ref_idx, sentences in enumerate(sentences_by_ref):
        if not sentences:
            continue
        uncertainty = 1.0 - support_sum_by_ref[ref_idx] / denom
        results.append({
            "sentences": sentences,
            "uncertainty": uncertainty,
            "mean_u": float(uncertainty.mean()),
            "K_actual": K,
            "ref_idx": ref_idx,
        })

    return results


def score_generation_file(gen_path: Path, sent_dir: Path, use_cache: bool) -> Dict:
    saved = json.loads(gen_path.read_text())
    sample_idx = int(saved["sample_idx"])
    notes = saved["notes"]

    if len(notes) < 2:
        return {"sample_idx": sample_idx, "status": "insufficient_generations", "K_actual": len(notes)}

    expected_csvs = [sent_dir / f"sample_{sample_idx:03d}_note_{i:02d}_sentences.csv" for i in range(len(notes))]
    if use_cache and all(path.exists() for path in expected_csvs):
        print(f"  [cache] sample {sample_idx}: all {len(expected_csvs)} note CSVs exist")
        results = []
        for i, path in enumerate(expected_csvs):
            df = pd.read_csv(path)
            results.append({
                "ref_idx": i,
                "mean_u": float(df["uncertainty"].mean()),
                "uncertainty": df["uncertainty"].to_numpy(dtype=float),
                "sentences": df["sentence"].tolist(),
            })
    else:
        print(f"  [luq] scoring all {len(notes) * (len(notes) - 1)} comparisons with section-wise batching")
        results = compute_luq_all_references(notes)
        for result in results:
            ref_idx = int(result["ref_idx"])
            pd.DataFrame({
                "sentence_idx": np.arange(len(result["sentences"])),
                "sentence": result["sentences"],
                "uncertainty": np.round(result["uncertainty"], 4),
            }).to_csv(sent_dir / f"sample_{sample_idx:03d}_note_{ref_idx:02d}_sentences.csv", index=False)
        clear_memory()

    if not results:
        return {"sample_idx": sample_idx, "status": "luq_failed", "K_actual": len(notes)}

    all_uncertainty = np.concatenate([r["uncertainty"] for r in results])
    return {
        "sample_idx": sample_idx,
        "status": "ok",
        "mean_u": round(float(np.mean([r["mean_u"] for r in results])), 4),
        "max_u": round(float(np.max(all_uncertainty)), 4),
        "high_u_frac": round(float(np.mean(all_uncertainty > 0.5)), 4),
        "K_actual": len(notes),
        "notes_scored": len(results),
        "n_sentences_first_ref": len(results[0]["sentences"]),
        "comparisons": len(notes) * (len(notes) - 1),
    }


# -----------------------------------------------------------------------------
# Stages
# -----------------------------------------------------------------------------

def stage_generate(start: int, end: int, k: int, out_dir: Path, use_cache: bool) -> None:
    gen_dir = out_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    ds = load_aci_bench()
    end = min(end, len(ds))

    for sample_idx in tqdm(range(start, end), desc="generate", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if use_cache and gen_path.exists():
            print(f"  [cache] sample {sample_idx}: generations exist")
            continue

        transcript, gold = get_transcript_and_gold(ds[sample_idx])
        notes = generate_notes(transcript, k)
        if not notes:
            print(f"  [generate] sample {sample_idx}: no successful generations")
            continue

        gen_path.write_text(json.dumps({
            "sample_idx": sample_idx,
            "transcript": transcript,
            "gold_note": gold,
            "notes": notes,
            "model": BEDROCK_GEN_MODEL,
            "K": k,
            "temperature": DEFAULT_TEMPERATURE,
        }, indent=2), encoding="utf-8")
        print(f"  [generate] sample {sample_idx}: saved {len(notes)} notes")


def stage_score(start: int, end: int, out_dir: Path, use_cache: bool,
                gen_dir_override: Path = None) -> pd.DataFrame:
    gen_dir = gen_dir_override if gen_dir_override else out_dir / "generations"
    sent_dir = out_dir / "sentences"
    sent_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for sample_idx in tqdm(range(start, end), desc="score", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            rows.append({"sample_idx": sample_idx, "status": "no_generations"})
            continue

        print(f"\n[score] sample {sample_idx}")
        rows.append(score_generation_file(gen_path, sent_dir, use_cache))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "luq_results.csv", index=False)
    print(f"\n[score] wrote {out_dir / 'luq_results.csv'}")
    return df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section-wise all-against-all LUQ scoring for ACI-Bench notes")
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=132)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--nli-model", default=None,
                        help="Override NLI model (e.g. mrm8488/deberta-v3-large-finetuned-mnli)")
    parser.add_argument("--gen-dir", default=None,
                        help="Path to existing generations directory (skips generation stage)")
    return parser.parse_args()


def main() -> None:
    global NLI_MODEL_NAME
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    if args.nli_model:
        NLI_MODEL_NAME = args.nli_model

    gen_dir_override = Path(args.gen_dir).resolve() if args.gen_dir else None

    print(f"Stage: {args.stage}")
    print(f"Samples: {args.start} to {args.end - 1}")
    print(f"Output: {out_dir}")
    print(f"K: {args.K}")
    print(f"Cache: {'on' if use_cache else 'off'}")
    print(f"NLI model: {NLI_MODEL_NAME}")
    print(f"NLI device: {DEVICE_NAME}")
    print(f"NLI batch size: {NLI_BATCH_SIZE}")
    print(f"Micro-batch pairs: {MICRO_BATCH_PAIRS}")
    if gen_dir_override:
        print(f"Generations from: {gen_dir_override}")

    if args.stage in {"generate", "all"} and gen_dir_override is None:
        stage_generate(args.start, args.end, args.K, out_dir, use_cache)

    if args.stage in {"score", "all"}:
        stage_score(args.start, args.end, out_dir, use_cache,
                    gen_dir_override=gen_dir_override)


if __name__ == "__main__":
    main()
