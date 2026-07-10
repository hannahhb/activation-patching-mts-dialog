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
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from prompts import build_prompt as build_user_message

warnings.filterwarnings("ignore")

# When this file is executed directly, its module name is "__main__". Other
# files (notably atomic_luq.py) import it as "luq_sentence". Without this
# alias, Python creates a second module instance on import, so globals like
# the cached CrossEncoder singleton (_nli) are duplicated and the NLI model
# gets loaded twice in one run.
if __name__ == "__main__":
    sys.modules.setdefault("luq_sentence", sys.modules[__name__])

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
DEFAULT_OUT_DIR = "luq_out/llama/generations"

PAIR_MAX_TOKENS = 512
SPECIAL_TOKEN_RESERVE = 8
MIN_PREMISE_WINDOW = 64
DEFAULT_STRIDE_RATIO = 0.5
DEFAULT_PREMISE_WINDOW = 448
WINDOW_BUCKET = 32


def _detect_device() -> str:
    """cuda > mps > cpu. mps (Apple Silicon GPU) was previously undetected
    here -- benchmarked at ~3.7x faster than cpu for this NLI model, so
    running on an Apple Silicon Mac without this check silently left a
    real GPU unused."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE_NAME = _detect_device()
GPU_AVAILABLE = DEVICE_NAME != "cpu"
NLI_BATCH_SIZE = 16 if GPU_AVAILABLE else 4
MICRO_BATCH_PAIRS = 128 if GPU_AVAILABLE else 16


# -----------------------------------------------------------------------------
# Dataset and generation
# -----------------------------------------------------------------------------

def load_aci_bench(split: str = DATASET_SPLIT, config: str = DATASET_CONFIG):
    from datasets import load_dataset
    print(f"[data] Loading {DATASET_REPO} config={config} split={split}")
    return load_dataset(DATASET_REPO, config, split=split)


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


def dedupe_by_transcript(ds) -> List[int]:
    """Row indices to keep: first occurrence of each unique dialogue text.

    test1/test2/test3 each contain 22 encounters x 2 transcript_version rows
    (asr / asrcorr); most pairs share a byte-identical dialogue, and only a
    minority were genuinely ASR-corrected. A corrected pair is NOT an exact
    duplicate (its transcript text differs) so both versions survive dedup.
    """
    seen = set()
    keep = []
    for i, row in enumerate(ds):
        transcript, _ = get_transcript_and_gold(row)
        if transcript not in seen:
            seen.add(transcript)
            keep.append(i)
    return keep


def generate_note(transcript: str) -> str:
    from llm_client import get_llm
    response = get_llm().converse(
        stage="generation",
        model_id=BEDROCK_GEN_MODEL,
        messages=[{"role": "user", "content": [{"text": build_user_message(transcript)}]}],
        inference_config={
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

        print(f"[nli] Loading {NLI_MODEL_NAME} on device={DEVICE_NAME}")
        _nli = CrossEncoder(NLI_MODEL_NAME, device=DEVICE_NAME)
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


SENTENCE_WINDOW_SIZE = 2  # consecutive sentences per NLI premise chunk
SENTENCE_WINDOW_MAX_CHARS = 1800  # above this, fall back to token-based sub-chunking


def sentence_windows(text: str, tokenizer=None,
                     window_size: int = SENTENCE_WINDOW_SIZE) -> List[str]:
    """Sliding window of `window_size` consecutive sentences (stride 1) used
    as NLI premise chunks, instead of one large token-budget-maxed premise
    covering a whole (often multi-topic) section.

    Why: cross-encoder NLI models are trained on short, single-topic
    premise/hypothesis pairs. Verified empirically that the exact same
    supporting sentence scores ~0.99 entailment for a narrow claim when
    passed alone, but ~0.01 for the identical claim when embedded in a full
    multi-sentence section alongside unrelated content (e.g. a demographic
    fact diluted by surrounding symptom/timeline sentences) -- the model
    isn't finding the supporting clause, it's judging the whole passage's
    gist. Small, localized windows (max-pooled over) avoid that dilution.

    Independent of claim length (unlike the old bucketed_window_size), so
    chunk lists can be cached per (premise note, section) alone and reused
    across every claim that section is compared against.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [""]
    if len(sentences) <= window_size:
        chunks = [" ".join(sentences)]
    else:
        chunks = [" ".join(sentences[i:i + window_size])
                 for i in range(len(sentences) - window_size + 1)]

    if tokenizer is None:
        return chunks

    # Safety net: an unusually long chunk (e.g. one run-on clinical sentence)
    # that still overflows the NLI pair budget gets token-chunked via the
    # existing machinery rather than silently truncated by the tokenizer.
    out: List[str] = []
    for c in chunks:
        if len(c) <= SENTENCE_WINDOW_MAX_CHARS:
            out.append(c)
        else:
            ids = tokenizer.encode(c, add_special_tokens=False)
            out.extend(chunk_premise_ids(tokenizer, ids, DEFAULT_PREMISE_WINDOW))
    return out


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
    """
    sentences_by_ref: List[List[str]] = []
    claims_by_ref: List[List[str]] = []
    claim_to_sentence_by_ref: List[List[int]] = []
    claim_sections_by_ref: List[List[str]] = []

    for note in notes:
        sentences = split_sentences(note)
        sent_sections = assign_sentence_sections(note, sentences)
        claims: List[str] = []
        claim_to_sentence: List[int] = []
        claim_sections: List[str] = []

        for sent_idx, sentence in enumerate(sentences):
            section = sent_sections[sent_idx] if sent_idx < len(sent_sections) else "all"
            for claim in split_subclaims(sentence):
                claims.append(claim)
                claim_to_sentence.append(sent_idx)
                claim_sections.append(section)

        sentences_by_ref.append(sentences)
        claims_by_ref.append(claims)
        claim_to_sentence_by_ref.append(claim_to_sentence)
        claim_sections_by_ref.append(claim_sections)

    return (
        sentences_by_ref,
        claims_by_ref,
        claim_to_sentence_by_ref,
        claim_sections_by_ref,
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
    ) = build_reference_claims(notes, tokenizer)

    support_sum_by_ref: List[np.ndarray] = [
        np.zeros(len(sentences), dtype=np.float64)
        for sentences in sentences_by_ref
    ]

    # Premise-first: parse each sampled note's sections once, then score every
    # other reference's claims against those section premises.
    for premise_idx, sampled_note in enumerate(notes):
        section_text = parse_sections(sampled_note)
        if "all" not in section_text:
            section_text["all"] = sampled_note

        # Cache sentence-window chunk lists for this premise by section alone
        # -- unlike the old token-budget windows, sentence windows don't
        # depend on the claim being checked, so one chunk list per section
        # (not per (section, window_size) pair) is reused across every claim
        # that matches it, from every reference note.
        chunk_cache: Dict[str, List[str]] = {}

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
            if not claims:
                continue

            for claim_idx, claim in enumerate(claims):
                requested_sec = sections[claim_idx]
                premise_sec = requested_sec if requested_sec in section_text else "all"

                if premise_sec not in chunk_cache:
                    chunk_cache[premise_sec] = sentence_windows(
                        section_text[premise_sec], tokenizer=tokenizer,
                    )

                for chunk in chunk_cache[premise_sec]:
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

        section_names = ",".join(k for k in section_text.keys() if k != "all") or "all"
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


DEFAULT_SENTENCE_GPTOSS_SCORE_K = 3  # how many notes to score (matches
                                     # atomic_luq's DEFAULT_DECOMP_K
                                     # convention) -- with K=5 notes and
                                     # ~20-28 sentences/note, scoring all K
                                     # would be 5x20x4=~400 calls/sample;
                                     # scoring only 3 gives 3x20x4=~240.


def compute_luq_all_references_gptoss(notes: List[str], reasoning_effort: str = "low",
                                       max_workers: int = None,
                                       score_k: int = None) -> List[Dict]:
    """gpt-oss equivalent of compute_luq_all_references: ONE yes/no call per
    (whole sentence, other note) pair -- no sub-claim splitting (gpt-oss
    doesn't need the cross-encoder's granularity workaround) and no
    sentence-window sub-chunking of the premise (premise = the full matching
    section of the other note directly, 128K context handles it).

    Only the first `score_k` notes are scored (default
    DEFAULT_SENTENCE_GPTOSS_SCORE_K), each checked against all K-1 OTHER
    notes -- mirrors atomic_luq's decomp_k, to keep call volume bounded
    regardless of how many notes were generated.

    Same output shape as compute_luq_all_references (list of dicts with
    sentences/uncertainty/mean_u/K_actual/ref_idx) so score_generation_file
    can write identical CSVs regardless of backend.
    """
    from llm_client import gptoss_yesno, GPTOSS_MAX_WORKERS

    K = len(notes)
    if K < 2:
        return []
    max_workers = max_workers or GPTOSS_MAX_WORKERS
    score_k = min(score_k or DEFAULT_SENTENCE_GPTOSS_SCORE_K, K)

    section_text_by_note: List[Dict[str, str]] = []
    sentences_by_note: List[List[str]] = []
    sentence_sections_by_note: List[List[str]] = []
    for note in notes:
        st = parse_sections(note)
        if "all" not in st:
            st["all"] = note
        section_text_by_note.append(st)

        sentences = split_sentences(note)
        sentences_by_note.append(sentences)
        sentence_sections_by_note.append(assign_sentence_sections(note, sentences))

    # Flatten every (ref_idx, sent_idx, other_idx) job up front instead of
    # nesting per-note batches -- keeps max_workers calls continuously in
    # flight across the whole sample rather than draining/refilling per note.
    jobs: List[Tuple[int, int, int]] = [
        (ref_idx, sent_idx, other_idx)
        for ref_idx in range(score_k)
        for sent_idx in range(len(sentences_by_note[ref_idx]))
        for other_idx in range(K)
        if other_idx != ref_idx
    ]

    vote_sum: Dict[int, np.ndarray] = {
        ref_idx: np.zeros(len(sentences_by_note[ref_idx]), dtype=np.float64) for ref_idx in range(score_k)
    }
    vote_count: Dict[int, np.ndarray] = {
        ref_idx: np.zeros(len(sentences_by_note[ref_idx]), dtype=np.int32) for ref_idx in range(score_k)
    }

    def run_job(ref_idx: int, sent_idx: int, other_idx: int):
        sentence = sentences_by_note[ref_idx][sent_idx]
        requested_sec = sentence_sections_by_note[ref_idx][sent_idx]
        section_text = section_text_by_note[other_idx]
        premise_sec = requested_sec if requested_sec in section_text else "all"
        premise = section_text[premise_sec].strip()
        if not premise:
            return ref_idx, sent_idx, None
        return ref_idx, sent_idx, gptoss_yesno(premise, sentence, reasoning_effort)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_job, *job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="  [gptoss] sentence scoring", unit="call", leave=False):
            ref_idx, sent_idx, vote = fut.result()
            if vote is not None:
                vote_sum[ref_idx][sent_idx] += vote
                vote_count[ref_idx][sent_idx] += 1

    results: List[Dict] = []
    for ref_idx in range(score_k):
        sentences = sentences_by_note[ref_idx]
        if not sentences:
            continue
        counts = vote_count[ref_idx]
        sums = vote_sum[ref_idx]
        support = np.where(counts > 0, sums / np.maximum(counts, 1), 0.5)
        uncertainty = 1.0 - support
        results.append({
            "sentences": sentences,
            "uncertainty": uncertainty,
            "mean_u": float(uncertainty.mean()),
            "K_actual": K,
            "ref_idx": ref_idx,
        })

    return results


def score_generation_file(gen_path: Path, sent_dir: Path, use_cache: bool,
                          backend: str = "cross-encoder", reasoning_effort: str = "low",
                          gptoss_workers: int = None, gptoss_score_k: int = None) -> Dict:
    saved = json.loads(gen_path.read_text())
    sample_idx = int(saved["sample_idx"])
    notes = saved["notes"]

    if len(notes) < 2:
        return {"sample_idx": sample_idx, "status": "insufficient_generations", "K_actual": len(notes)}

    # gpt-oss only scores the first `score_k` notes (see
    # compute_luq_all_references_gptoss), so only that many CSVs ever get
    # written -- checking for all K would always miss cache.
    n_expected = (min(gptoss_score_k or DEFAULT_SENTENCE_GPTOSS_SCORE_K, len(notes))
                 if backend == "gptoss" else len(notes))
    expected_csvs = [sent_dir / f"sample_{sample_idx:03d}_note_{i:02d}_sentences.csv" for i in range(n_expected)]
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
        if backend == "gptoss":
            print(f"  [luq] scoring {n_expected} of {len(notes)} notes x {len(notes) - 1} "
                 f"references (gpt-oss)")
            results = compute_luq_all_references_gptoss(notes, reasoning_effort=reasoning_effort,
                                                         max_workers=gptoss_workers,
                                                         score_k=gptoss_score_k)
        else:
            print(f"  [luq] scoring all {len(notes) * (len(notes) - 1)} comparisons (cross-encoder)")
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
        "comparisons": len(results) * (len(notes) - 1),
    }


# -----------------------------------------------------------------------------
# Stages
# -----------------------------------------------------------------------------

def stage_generate(start: int, end: int, k: int, out_dir: Path, use_cache: bool,
                   split: str = DATASET_SPLIT, config: str = DATASET_CONFIG,
                   dedupe: bool = False) -> None:
    """dedupe=True restricts to rows with a unique (byte-identical) transcript
    -- meaningful for the "aci" config's test1/test2/test3 splits, where most
    asr/asrcorr pairs are exact duplicates. The "virtassist"/"virtscribe"
    configs pair humantrans/asr transcripts that are independently produced
    and never byte-identical, so dedupe is a no-op there (all rows kept) --
    that's expected, not a bug."""
    gen_dir = out_dir 
    gen_dir.mkdir(parents=True, exist_ok=True)

    ds = load_aci_bench(split=split, config=config)
    row_idx = dedupe_by_transcript(ds) if dedupe else list(range(len(ds)))
    if dedupe:
        print(f"[{split}] {len(ds)} rows -> {len(row_idx)} unique-transcript rows kept "
              f"({len(ds) - len(row_idx)} exact duplicates dropped)")
    end = min(end, len(row_idx))

    from llm_client import tracker as llm_tracker
    n_processed = 0
    for sample_idx in tqdm(range(start, end), desc="generate", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if use_cache and gen_path.exists():
            print(f"  [cache] sample {sample_idx}: generations exist")
            continue

        row = ds[row_idx[sample_idx]]
        transcript, gold = get_transcript_and_gold(row)
        notes = generate_notes(transcript, k)
        if not notes:
            print(f"  [generate] sample {sample_idx}: no successful generations")
            continue

        gen_path.write_text(json.dumps({
            "sample_idx": sample_idx,
            "orig_row_idx": row_idx[sample_idx],
            "encounter_id": row.get("encounter_id"),
            "transcript_version": row.get("transcript_version"),
            "transcript": transcript,
            "gold_note": gold,
            "notes": notes,
            "model": BEDROCK_GEN_MODEL,
            "K": k,
            "temperature": DEFAULT_TEMPERATURE,
        }, indent=2), encoding="utf-8")
        print(f"  [generate] sample {sample_idx}: saved {len(notes)} notes")

        n_processed += 1
        if n_processed % 5 == 0:
            print(f"  [cost] after {n_processed} samples:\n{llm_tracker.summary()}")

    print(f"\n[cost] final:\n{llm_tracker.summary()}")


def stage_score(start: int, end: int, out_dir: Path, use_cache: bool,
                gen_dir_override: Path = None, atomic: bool = False,
                atomic_out: str = None, atomic_decomp_k: int = None,
                atomic_only: bool = False, atomic_backend: str = "gptoss",
                atomic_reasoning_effort: str = "low", atomic_gptoss_workers: int = None,
                sentence_backend: str = "cross-encoder", sentence_reasoning_effort: str = "low",
                sentence_gptoss_workers: int = None, sentence_gptoss_score_k: int = None) -> pd.DataFrame:
    """atomic=True additionally runs LUQ-ATOMIC (fact-level uncertainty, see
    atomic_luq.py) for each sample alongside the normal sentence-level scoring.
    atomic_only=True skips the (slower, all-against-all NLI) sentence-level
    scoring entirely and only runs LUQ-ATOMIC -- implies atomic=True.
    Kept as a lazy import here since atomic_luq.py itself imports this module
    (luq_sentence) -- importing it at module level would be circular."""
    atomic = atomic or atomic_only
    gen_dir = gen_dir_override if gen_dir_override else out_dir
    sent_dir = out_dir / "sentences"
    if not atomic_only:
        sent_dir.mkdir(parents=True, exist_ok=True)

    atomic_mod = None
    atomic_out_dir = None
    if atomic:
        import atomic_luq as atomic_mod
        # atomic_luq.score_sample appends "/facts" itself, so passing out_dir
        # directly lands facts at <out_dir>/facts -- a sibling of "sentences",
        # nested under the same <config>/<split> combo dir by default (e.g.
        # luq_out/llama/generations/aci/test1/facts), unless --atomic-out
        # explicitly points somewhere else (e.g. the old shared
        # luq_out/llama_atomic cache).
        atomic_out_dir = Path(atomic_out).resolve() if atomic_out else out_dir
        atomic_out_dir.mkdir(parents=True, exist_ok=True)
        atomic_decomp_k = atomic_decomp_k or atomic_mod.DEFAULT_DECOMP_K

    from llm_client import tracker as llm_tracker

    rows: List[Dict] = []
    n_processed = 0
    for sample_idx in tqdm(range(start, end), desc="score", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            rows.append({"sample_idx": sample_idx, "status": "no_generations"})
            continue

        print(f"\n[score] sample {sample_idx}")
        if not atomic_only:
            rows.append(score_generation_file(gen_path, sent_dir, use_cache,
                                              backend=sentence_backend,
                                              reasoning_effort=sentence_reasoning_effort,
                                              gptoss_workers=sentence_gptoss_workers,
                                              gptoss_score_k=sentence_gptoss_score_k))

        if atomic_mod is not None:
            notes = json.loads(gen_path.read_text()).get("notes", [])
            decomp_k = min(atomic_decomp_k, len(notes))
            if len(notes) >= 2:
                if atomic_backend == "gptoss":
                    workers = atomic_gptoss_workers or atomic_mod.GPTOSS_MAX_WORKERS
                    atomic_mod.score_sample_gptoss(sample_idx, notes, decomp_k, atomic_out_dir,
                                                   use_cache, reasoning_effort=atomic_reasoning_effort,
                                                   max_workers=workers)
                else:
                    atomic_mod.score_sample(sample_idx, notes, decomp_k, atomic_out_dir, use_cache)
            else:
                print(f"  [atomic] sample {sample_idx}: fewer than 2 notes, skipped")

        # Decomposition (stage="decomp") runs for every atomic sample; NLI
        # shows up as stage="nli" whenever either backend is gptoss (the
        # cross-encoder is local/free, never tracked here).
        if atomic_mod is not None or sentence_backend == "gptoss":
            n_processed += 1
            if n_processed % 5 == 0:
                print(f"  [cost] after {n_processed} samples:\n{llm_tracker.summary()}")

    if atomic_mod is not None or sentence_backend == "gptoss":
        print(f"\n[cost] final:\n{llm_tracker.summary()}")

    df = pd.DataFrame(rows)
    if not atomic_only:
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
    parser.add_argument("--split", nargs="+", default=[DATASET_SPLIT],
                        choices=["train", "valid", "test1", "test2", "test3"],
                        help="ACI-Bench split(s) to generate from (default test1). "
                             "Pass multiple space-separated values (e.g. "
                             "--split test1 test2 test3) to run all of them; "
                             "each gets its own subdirectory under --out.")
    parser.add_argument("--config", nargs="+", default=[DATASET_CONFIG],
                        choices=["aci", "virtassist", "virtscribe"],
                        help="ACI-Bench-MedARC config(s): conversation style "
                             "(default aci; virtassist/virtscribe pair "
                             "humantrans/asr transcripts instead of asr/asrcorr). "
                             "Pass multiple values to run all of them.")
    parser.add_argument("--dedupe-transcripts", action="store_true",
                        help="Restrict to unique-transcript rows (drops exact "
                             "asr/asrcorr duplicates; see dedupe_by_transcript. "
                             "No-op for virtassist/virtscribe -- their "
                             "humantrans/asr pairs are never byte-identical.)")
    parser.add_argument("--atomic", action="store_true",
                        help="During the score stage, also run LUQ-ATOMIC "
                             "(fact-level uncertainty via atomic_luq.py) for "
                             "each sample, alongside the normal sentence-level "
                             "scoring.")
    parser.add_argument("--atomic-out", default=None,
                        help="Output dir for LUQ-ATOMIC facts (default: nested "
                             "under this combo's own out_dir, e.g. "
                             "luq_out/llama/generations/aci/test1/facts -- pass "
                             "e.g. luq_out/llama_atomic to use the old shared "
                             "cache location instead)")
    parser.add_argument("--atomic-decomp-k", type=int, default=3,
                        help="How many notes per sample to decompose into atomic "
                             "facts (default: atomic_luq.py's own default, 3)")
    parser.add_argument("--atomic-only", action="store_true",
                        help="Skip the slower sentence-level (all-against-all "
                             "NLI) scoring entirely and only run LUQ-ATOMIC. "
                             "Implies --atomic.")
    parser.add_argument("--atomic-backend", choices=["cross-encoder", "gptoss"],
                        default="gptoss",
                        help="LUQ-ATOMIC Step-3 scoring backend: the local "
                             "cross-encoder (default) or gpt-oss-20b yes/no "
                             "via Bedrock Converse (premise = full matching "
                             "section, no windowing needed at 128K context).")
    parser.add_argument("--atomic-reasoning-effort", choices=["low", "medium", "high"],
                        default="low",
                        help="gpt-oss reasoning_effort (only used with "
                             "--atomic-backend gptoss)")
    parser.add_argument("--atomic-gptoss-workers", type=int, default=None,
                        help="Concurrent Bedrock calls for --atomic-backend "
                             "gptoss (default: atomic_luq.py's own default)")
    parser.add_argument("--sentence-backend", choices=["cross-encoder", "gptoss"],
                        default="gptoss",
                        help="Sentence-level LUQ scoring backend: the local "
                             "cross-encoder (default) or gpt-oss-20b yes/no "
                             "via Bedrock Converse (premise = full matching "
                             "section, no windowing needed at 128K context).")
    parser.add_argument("--sentence-reasoning-effort", choices=["low", "medium", "high"],
                        default="low",
                        help="gpt-oss reasoning_effort (only used with "
                             "--sentence-backend gptoss)")
    parser.add_argument("--sentence-gptoss-workers", type=int, default=None,
                        help="Concurrent Bedrock calls for --sentence-backend "
                             "gptoss (default: llm_client.py's GPTOSS_MAX_WORKERS)")
    parser.add_argument("--sentence-gptoss-score-k", type=int, default=None,
                        help="How many notes per sample to sentence-score with "
                             "--sentence-backend gptoss (default: 3, matching "
                             "atomic_luq's decomp_k convention)")
    return parser.parse_args()


def main() -> None:
    global NLI_MODEL_NAME
    args = parse_args()
    out_root = Path(args.out).resolve()
    use_cache = not args.no_cache

    if args.nli_model:
        NLI_MODEL_NAME = args.nli_model

    gen_dir_override = Path(args.gen_dir).resolve() if args.gen_dir else None

    combos = [(config, split) for config in args.config for split in args.split]

    print(f"Stage: {args.stage}")
    print(f"Samples: {args.start} to {args.end - 1}")
    print(f"K: {args.K}")
    print(f"Cache: {'on' if use_cache else 'off'}")
    print(f"NLI model: {NLI_MODEL_NAME}")
    print(f"NLI device: {DEVICE_NAME}")
    print(f"NLI batch size: {NLI_BATCH_SIZE}")
    print(f"Micro-batch pairs: {MICRO_BATCH_PAIRS}")
    if gen_dir_override:
        print(f"Generations from: {gen_dir_override}")
    print(f"Config x split combinations: {combos}")

    for config, split in combos:
        # Always nest by <config>/<split>, e.g. luq_out/llama/generations/aci/test1 --
        # every downstream tool (annotator.py, llm_hallucination_label.py, etc.)
        # expects this layout even for the single-combo default case.
        out_dir = out_root / f"{config}/{split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== config={config} split={split} -> {out_dir} ===")

        if args.stage in {"generate", "all"} and gen_dir_override is None:
            stage_generate(args.start, args.end, args.K, out_dir, use_cache,
                           split=split, config=config, dedupe=args.dedupe_transcripts)

        if args.stage in {"score", "all"}:
            stage_score(args.start, args.end, out_dir, use_cache,
                       gen_dir_override=gen_dir_override, atomic=args.atomic,
                       atomic_out=args.atomic_out, atomic_decomp_k=args.atomic_decomp_k,
                       atomic_only=args.atomic_only, atomic_backend=args.atomic_backend,
                       atomic_reasoning_effort=args.atomic_reasoning_effort,
                       atomic_gptoss_workers=args.atomic_gptoss_workers,
                       sentence_backend=args.sentence_backend,
                       sentence_reasoning_effort=args.sentence_reasoning_effort,
                       sentence_gptoss_workers=args.sentence_gptoss_workers,
                       sentence_gptoss_score_k=args.sentence_gptoss_score_k)


if __name__ == "__main__":
    main()
