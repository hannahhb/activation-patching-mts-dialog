"""
luq_sentence_streamed_simplified.py
==================================
Memory-safe sentence-level LUQ for clinical note generations.

Core behaviour:
  - Generate K notes per transcript, or score cached generations.
  - Score every generated note as the reference once.
  - Compare each reference note against every other sampled note.
  - With K=10, this gives 10 x 9 = 90 note-to-note comparisons.
  - Stream NLI in small micro-batches to avoid memory spikes while keeping GPU batching.
  - Use LUQ-style binary entailment-vs-contradiction normalisation.
  - Cover long premises with overlapping token windows rather than silent truncation.
  - Cache premise tokenisation/chunks per sampled note for speed.

Typical use:
  python luq_sentence_streamed_simplified.py --stage generate --start 0 --end 10
  python luq_sentence_streamed_simplified.py --stage score --start 0 --end 10
  python luq_sentence_streamed_simplified.py --stage all --start 0 --end 10
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

# Cross-encoder NLI models are usually 512-token models. We dynamically reserve
# space for the hypothesis and special tokens so premise windows are not silently
# truncated by the tokenizer.
PAIR_MAX_TOKENS = 512
SPECIAL_TOKEN_RESERVE = 8
MIN_PREMISE_WINDOW = 64
DEFAULT_STRIDE_RATIO = 0.5

# Small batches preserve the all-90 comparison logic while avoiding the huge
# all_pairs tensor/list that caused memory crashes. Defaults are selected from
# the actual runtime device: CUDA can safely use larger batches; CPU uses smaller
# micro-batches to avoid RAM spikes and excessive tokenisation backlog.
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
# Sentence and claim processing
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
    """
    Decode overlapping premise windows of a fixed token width.
    The caller chooses window_size so each (window, hypothesis) pair fits within
    the NLI token budget. No premise tokens are silently discarded.
    """
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


def window_size_for_claim(tokenizer, claim: str) -> int:
    """
    Reserve enough token budget for the hypothesis so CrossEncoder does not
    silently truncate the premise. Clinical-note sentences should usually leave
    a large premise window; very long hypotheses still get a minimum window.
    """
    hyp_ids = tokenizer.encode(claim, add_special_tokens=False)
    available = PAIR_MAX_TOKENS - len(hyp_ids) - SPECIAL_TOKEN_RESERVE
    return max(MIN_PREMISE_WINDOW, available)


def binary_luq_support(logits: np.ndarray, entail_idx: int, contradict_idx: int) -> float:
    ec_logits = logits[[entail_idx, contradict_idx]]
    ec_exp = np.exp(ec_logits - np.max(ec_logits))
    return float(ec_exp[0] / np.sum(ec_exp))


def predict_pair_supports(nli, pairs: List[Tuple[str, str]]) -> np.ndarray:
    """
    Run the NLI cross-encoder on a small list of pairs and return LUQ binary
    entailment-vs-contradiction support scores.
    """
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

    scores = [binary_luq_support(row, entail_idx, contradict_idx) for row in raw]
    return np.asarray(scores, dtype=np.float64)


def support_for_sentences_against_note(nli, sentences: List[str], sampled_note: str) -> np.ndarray:
    """
    Score all reference sentences against one sampled note.

    This keeps the important implementation details:
      - full sampled note is the premise;
      - long premises are covered by overlapping token windows;
      - each subclaim takes max support across all premise windows;
      - sentence support is the mean over its subclaims;
      - LUQ support uses binary entailment-vs-contradiction normalisation.

    Speed fix:
      - tokenise the sampled note once;
      - cache decoded premise chunks by window size;
      - run NLI in small micro-batches instead of one pair at a time.
    """
    tokenizer = nli.tokenizer
    premise_ids = tokenizer.encode(sampled_note, add_special_tokens=False)
    chunk_cache: Dict[int, List[str]] = {}

    # Flatten sentence -> subclaim structure while keeping mapping back to sentences.
    claims: List[str] = []
    claim_to_sentence: List[int] = []
    empty_sentence_indices = set()

    for sent_idx, sentence in enumerate(sentences):
        subclaims = split_subclaims(sentence)
        if not subclaims:
            empty_sentence_indices.add(sent_idx)
            continue
        for claim in subclaims:
            claims.append(claim)
            claim_to_sentence.append(sent_idx)

    if not claims:
        return np.full(len(sentences), 0.5, dtype=np.float64)

    # Build and score pairs in micro-batches. We do not keep the full cross-product
    # in memory. We only keep max support per claim.
    claim_max = np.full(len(claims), -np.inf, dtype=np.float64)
    pair_buffer: List[Tuple[str, str]] = []
    pair_claim_indices: List[int] = []

    def flush_buffer() -> None:
        nonlocal pair_buffer, pair_claim_indices, claim_max
        if not pair_buffer:
            return
        scores = predict_pair_supports(nli, pair_buffer)
        for score, claim_idx in zip(scores, pair_claim_indices):
            if score > claim_max[claim_idx]:
                claim_max[claim_idx] = float(score)
        pair_buffer = []
        pair_claim_indices = []

    for claim_idx, claim in enumerate(claims):
        window_size = window_size_for_claim(tokenizer, claim)
        if window_size not in chunk_cache:
            chunk_cache[window_size] = chunk_premise_ids(tokenizer, premise_ids, window_size)

        for chunk in chunk_cache[window_size]:
            pair_buffer.append((chunk, claim))
            pair_claim_indices.append(claim_idx)
            if len(pair_buffer) >= MICRO_BATCH_PAIRS:
                flush_buffer()

    flush_buffer()
    claim_max[~np.isfinite(claim_max)] = 0.5

    sentence_claim_scores: List[List[float]] = [[] for _ in sentences]
    for claim_idx, sent_idx in enumerate(claim_to_sentence):
        sentence_claim_scores[sent_idx].append(float(claim_max[claim_idx]))

    sentence_support = np.zeros(len(sentences), dtype=np.float64)
    for sent_idx, scores in enumerate(sentence_claim_scores):
        sentence_support[sent_idx] = float(np.mean(scores)) if scores else 0.5

    return sentence_support


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


def compute_luq_for_reference(notes: List[str], ref_idx: int) -> Dict:
    """
    Compute sentence uncertainty for notes[ref_idx] against all other notes.
    """
    if len(notes) < 2:
        return {}

    nli = get_nli()
    ref_note = notes[ref_idx]
    sentences = split_sentences(ref_note)
    other_indices = [i for i in range(len(notes)) if i != ref_idx]

    if not sentences:
        return {}

    print(f"  [luq] ref={ref_idx:02d}: {len(sentences)} sentences x {len(other_indices)} notes")
    support_sum = np.zeros(len(sentences), dtype=np.float64)

    for other_idx in other_indices:
        sampled_note = notes[other_idx]
        support_sum += support_for_sentences_against_note(nli, sentences, sampled_note)

    # Clear once per reference, not after every note comparison. Calling
    # torch.cuda.empty_cache() inside the inner loop slows scoring a lot.
    clear_memory()

    uncertainty = 1.0 - support_sum / len(other_indices)
    return {
        "sentences": sentences,
        "uncertainty": uncertainty,
        "mean_u": float(uncertainty.mean()),
        "K_actual": len(notes),
        "ref_idx": ref_idx,
    }


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
        results = []
        for ref_idx in range(len(notes)):
            result = compute_luq_for_reference(notes, ref_idx)
            if not result:
                continue

            pd.DataFrame({
                "sentence_idx": np.arange(len(result["sentences"])),
                "sentence": result["sentences"],
                "uncertainty": np.round(result["uncertainty"], 4),
            }).to_csv(sent_dir / f"sample_{sample_idx:03d}_note_{ref_idx:02d}_sentences.csv", index=False)

            results.append(result)
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


def stage_score(start: int, end: int, out_dir: Path, use_cache: bool) -> pd.DataFrame:
    gen_dir = out_dir / "generations"
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
    parser = argparse.ArgumentParser(description="Streamed all-against-all LUQ scoring for ACI-Bench notes")
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=132)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    print(f"Stage: {args.stage}")
    print(f"Samples: {args.start} to {args.end - 1}")
    print(f"Output: {out_dir}")
    print(f"K: {args.K}")
    print(f"Cache: {'on' if use_cache else 'off'}")
    print(f"NLI device: {DEVICE_NAME}")
    print(f"NLI batch size: {NLI_BATCH_SIZE}")
    print(f"Micro-batch pairs: {MICRO_BATCH_PAIRS}")

    if args.stage in {"generate", "all"}:
        stage_generate(args.start, args.end, args.K, out_dir, use_cache)

    if args.stage in {"score", "all"}:
        stage_score(args.start, args.end, out_dir, use_cache)


if __name__ == "__main__":
    main()
