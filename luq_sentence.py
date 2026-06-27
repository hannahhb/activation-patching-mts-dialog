"""
luq_sentence.py
===============
Sentence-level uncertainty quantification using the LUQ method
(Zhang et al., EMNLP 2024) applied to clinical note generation.

For each ACI-Bench transcript:
  1. Generate K notes via Llama 3.1 8B on AWS Bedrock using the
     Asgari et al. (2025) Experiment 8 prompt — the best-performing
     non-function-call configuration on PDSQI-9 criteria.
  2. Treat the first generation (r_a) as the reference note.
  3. Split r_a into sentences on newlines and periods.
  4. For each sentence s_j, compute against every other
     generation r':
         P(entail | s_j, r') = exp(l_e) / (exp(l_e) + exp(l_c))
     using DeBERTa-v3-large NLI (as in the LUQ paper).
  5. Per-sentence uncertainty:
         U(s_j) = 1 − mean_{r'} P(entail | s_j, r')
  6. Per-sample uncertainty:
         U = mean_{s_j} U(s_j)

Outputs (written to --out/luq_out/):
  generations/sample_NNN_generations.json  — K raw generated notes
  sentences/sample_NNN_sentences.csv       — per-sentence uncertainty
  luq_results.csv                          — per-sample summary
  top3_report.html                         — highlighted HTML

Usage:
  python luq_sentence.py
  python luq_sentence.py --start 0 --end 10 --K 5
  python luq_sentence.py --K 10 --bedrock-region us-east-1 --out results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Constants
# ─────────────────────────────────────────────────────────────────────────────

_BEDROCK_GEN_MODEL      = "us.meta.llama3-1-8b-instruct-v1:0"
_BEDROCK_DEFAULT_REGION = "us-east-1"
_NLI_MODEL_NAME         = "cross-encoder/nli-deberta-v3-large"

_DATASET_REPO   = "mkieffer/ACI-Bench-MedARC"
_DATASET_CONFIG = "aci"
_DATASET_SPLIT  = "test1"

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Asgari et al. (2025) Experiment 8 prompt
#     Best-performing structured prompt on PDSQI-9 criteria.
#     Source: Table 1, npj Digital Medicine (2025) 8:274.
# ─────────────────────────────────────────────────────────────────────────────

from prompts import build_prompt as _build_user_message


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_aci_bench(split: str = _DATASET_SPLIT):
    from datasets import load_dataset
    print(f"[data] Loading {_DATASET_REPO} config={_DATASET_CONFIG} split={split} …")
    ds = load_dataset(_DATASET_REPO, _DATASET_CONFIG, split=split)
    print(f"[data] {len(ds)} rows, columns: {ds.column_names}")
    return ds


def _get_transcript_and_note(row: dict) -> Tuple[str, str]:
    for tc in ["src", "dialogue", "conversation", "transcript", "input"]:
        if tc in row:
            transcript = row[tc]
            break
    else:
        raise KeyError(f"No transcript column found in {list(row.keys())}")
    for nc in ["tgt", "note", "reference", "summary", "output"]:
        if nc in row:
            gold_note = row[nc]
            break
    else:
        gold_note = ""
    return transcript, gold_note


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Bedrock generation
# ─────────────────────────────────────────────────────────────────────────────

_bedrock_client = None


def _get_bedrock_client(region: str):
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        from botocore.config import Config as BotoConfig
        cfg = BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=120,
        )
        _bedrock_client = boto3.client(
            "bedrock-runtime", region_name=region, config=cfg
        )
        print(f"[bedrock] Client initialised (region={region})")
    return _bedrock_client


def generate_note(
    transcript: str,
    bedrock_region: str = _BEDROCK_DEFAULT_REGION,
    model_id: str = _BEDROCK_GEN_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Generate one SOAP note from a transcript via Bedrock Converse."""
    client = _get_bedrock_client(bedrock_region)
    user_text = _build_user_message(transcript)

    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def generate_K_notes(
    transcript: str,
    K: int = 10,
    bedrock_region: str = _BEDROCK_DEFAULT_REGION,
    model_id: str = _BEDROCK_GEN_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> List[str]:
    """Generate K independent notes. Returns however many succeed (≥ 1)."""
    notes = []
    for k in tqdm(range(K), desc="  notes", unit="note", leave=False):
        try:
            note = generate_note(
                transcript,
                bedrock_region=bedrock_region,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            notes.append(note)
        except Exception as exc:
            tqdm.write(f"  [gen] {k+1}/{K} failed: {exc}")
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

_PERIOD_SPLIT  = re.compile(r"(?<=\.)\s+(?=[A-Z])")

# Titles / abbreviations whose trailing period must not trigger a sentence split
_ABBREV_RE = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug'
    r'|Sep|Oct|Nov|Dec|approx|appt|Dept|No|Vol|Fig)\.',
    re.IGNORECASE,
)
_DOT_PLACEHOLDER = "\x00DOT\x00"


def _protect_abbreviations(text: str) -> str:
    return _ABBREV_RE.sub(lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER), text)


def _restore_abbreviations(text: str) -> str:
    return text.replace(_DOT_PLACEHOLDER, ".")
_FIELD_PREFIX  = re.compile(r"^[A-Za-z /\-]+:\s*")   # strips "Problem list: ", "Assessment: " etc.
_TEMPLATE_JUNK = re.compile(r"\[unknown\]|\[NOT MENTIONED\]|\[not mentioned\]",
                             re.IGNORECASE)
_SEMICOLON_SPLIT = re.compile(r";\s*")


def _clean_for_nli(sent: str) -> str:
    """Strip structured field prefixes and template placeholders before NLI scoring."""
    s = _FIELD_PREFIX.sub("", sent).strip()
    s = _TEMPLATE_JUNK.sub("", s).strip(" ;,.")
    return s


def _expand_semicolons(sent: str) -> List[str]:
    """
    If a sentence contains semicolons (list of claims), split into sub-claims
    and return each; otherwise return the sentence unchanged.
    Only splits when there are at least 2 non-empty parts.
    """
    parts = [p.strip() for p in _SEMICOLON_SPLIT.split(sent) if p.strip()]
    return parts if len(parts) > 1 else [sent]


def split_sentences(text: str) -> List[str]:
    """Split on newlines and '. [Capital]', preserving known abbreviations (Mr., Dr., etc.)."""
    results = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-•*#+").strip()
        if not line:
            continue
        protected = _protect_abbreviations(line)
        for part in _PERIOD_SPLIT.split(protected):
            part = _restore_abbreviations(part).strip()
            if part:
                results.append(part)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SOAP section parsing
# ─────────────────────────────────────────────────────────────────────────────

_SOAP_HEADER_RE = re.compile(
    r"^(Subjective|Objective|Assessment\s*(?:/\s*Problem\s*List)?|Plan|Follow[- ]?up)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

_SECTION_NORM: Dict[str, str] = {
    "subjective": "subjective",
    "objective":  "objective",
    "assessment": "assessment",
    "plan":       "plan",
    "follow-up":  "followup",
    "follow up":  "followup",
}


def _normalize_section(header: str) -> str:
    h = header.strip().lower()
    # Assessment / Problem List → assessment
    if h.startswith("assessment"):
        return "assessment"
    if h.startswith("follow"):
        return "followup"
    return _SECTION_NORM.get(h, h)


def _parse_soap_sections(note: str) -> Dict[str, str]:
    """
    Split a SOAP note into its top-level sections.
    Returns {section_name: section_text}.
    Falls back to {"all": note} if no headers are found.
    """
    matches = list(_SOAP_HEADER_RE.finditer(note))
    if not matches:
        return {"all": note}
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        name  = _normalize_section(m.group(1))
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        sections[name] = note[start:end].strip()
    return sections


def _assign_sentence_sections(note: str, sentences: List[str]) -> List[str]:
    """
    For each sentence, return which SOAP section it belongs to based on its
    character position in the note. Falls back to "all" if no headers found.
    """
    matches = list(_SOAP_HEADER_RE.finditer(note))
    if not matches:
        return ["all"] * len(sentences)

    spans = [
        (m.start(),
         matches[i + 1].start() if i + 1 < len(matches) else len(note),
         _normalize_section(m.group(1)))
        for i, m in enumerate(matches)
    ]

    result = []
    search_pos = 0
    for sent in sentences:
        pos = note.find(sent, search_pos)
        section = "all"
        if pos != -1:
            for start, end, name in spans:
                if start <= pos < end:
                    section = name
                    break
            search_pos = pos + len(sent)
        result.append(section)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  NLI model
# ─────────────────────────────────────────────────────────────────────────────

_nli_pipeline = None


def _get_nli_pipeline(model_name: str = _NLI_MODEL_NAME):
    global _nli_pipeline
    if _nli_pipeline is None:
        from sentence_transformers import CrossEncoder
        print(f"[nli] Loading {model_name} …")
        _nli_pipeline = CrossEncoder(model_name)
        print("[nli] Ready.")
    return _nli_pipeline


_NLI_WINDOW_TOKENS = 380   # premise tokens per window (leaves ~128 for hypothesis + special tokens)
_NLI_STRIDE_TOKENS = 190   # 50% overlap between windows


def _chunk_premise(tokenizer, premise: str) -> List[str]:
    """
    Split premise into overlapping token windows so no content is lost.
    Each chunk is at most _NLI_WINDOW_TOKENS tokens; windows overlap by _NLI_STRIDE_TOKENS.
    Returns a list of decoded text strings (one per window).
    """
    token_ids = tokenizer.encode(premise, add_special_tokens=False)
    if len(token_ids) <= _NLI_WINDOW_TOKENS:
        return [premise]
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + _NLI_WINDOW_TOKENS, len(token_ids))
        chunks.append(tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
        if end == len(token_ids):
            break
        start += _NLI_STRIDE_TOKENS
    return chunks


def _entailment_probs(
    nli: object,
    hypotheses: List[str],
    premise: str,
) -> np.ndarray:
    """
    For each hypothesis h, compute P(entail | h, premise) as the 3-way softmax
    entailment probability (entail / (entail + neutral + contradict)).

    Using 3-way softmax rather than binary {entail, contradict} renormalisation
    avoids inflating uncertainty when the premise is simply silent on the claim
    (neutral) — a common case for structured list sentences.

    Each hypothesis is cleaned (field prefix and template artifacts stripped)
    and multi-claim semicolon lists are expanded into sub-claims.

    The premise is split into overlapping token windows (_NLI_WINDOW_TOKENS with
    _NLI_STRIDE_TOKENS overlap) so no part of a long note is dropped.
    For each sub-claim the max entailment score across all windows is taken —
    the claim is considered entailed if any part of the note supports it.
    Sub-claim scores are then averaged for the final hypothesis score.
    """
    if not hypotheses:
        return np.array([])

    tokenizer  = nli.tokenizer
    id2label   = nli.model.config.id2label
    entail_idx = next((int(i) for i, l in id2label.items() if "entail" in l.lower()), 2)

    premise_chunks = _chunk_premise(tokenizer, premise)

    # Build all (chunk, sub-claim) pairs in one pass, tracking indices for aggregation
    all_pairs: List[tuple]  = []
    pair_index: List[tuple] = []   # (hyp_idx, sc_idx, chunk_idx)
    empty_hyps: set         = set()

    for h_idx, h in enumerate(hypotheses):
        sub_claims = [sc for sc in _expand_semicolons(_clean_for_nli(h)) if sc]
        if not sub_claims:
            empty_hyps.add(h_idx)
            continue
        for sc_idx, sc in enumerate(sub_claims):
            for c_idx, chunk in enumerate(premise_chunks):
                all_pairs.append((chunk, sc))
                pair_index.append((h_idx, sc_idx, c_idx))

    if not all_pairs:
        return np.full(len(hypotheses), 0.5)

    # Single batched NLI call for all pairs
    raw = np.array(nli.predict(all_pairs, apply_softmax=False, show_progress_bar=False))
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    exp_raw      = np.exp(raw - raw.max(axis=1, keepdims=True))
    probs        = exp_raw / exp_raw.sum(axis=1, keepdims=True)
    entail_flat  = probs[:, entail_idx]   # shape: (n_pairs,)

    # Aggregate: max over chunks per (hyp, sub-claim), then mean over sub-claims per hyp
    hyp_sc_windows: dict = defaultdict(list)
    for flat_idx, (h_idx, sc_idx, _) in enumerate(pair_index):
        hyp_sc_windows[(h_idx, sc_idx)].append(float(entail_flat[flat_idx]))

    hyp_sc_max: dict = defaultdict(list)
    for (h_idx, sc_idx), window_scores in hyp_sc_windows.items():
        hyp_sc_max[h_idx].append(max(window_scores))

    scores = []
    for h_idx in range(len(hypotheses)):
        if h_idx in empty_hyps:
            scores.append(0.5)
        else:
            scores.append(float(np.mean(hyp_sc_max[h_idx])))

    return np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LUQ sentence-level uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def compute_luq_sentence(
    notes: List[str],
    nli_model_name: str = _NLI_MODEL_NAME,
    ref_idx: int = 0,
) -> Dict:
    """
    Compute sentence-level LUQ uncertainty over K generated notes.

    Reference note = notes[ref_idx].
    For each sentence s_j in the reference:
        U(s_j) = 1 − mean_{r'} P(entail | s_j, r')
    where r' ranges over all notes except the reference.

    Sentences are assigned to their SOAP section; each is scored only against
    the matching section of every other note rather than the full note.
    If a section is absent from another note, the full note is used as fallback.
    All sentences in the same (section × other-note) pair are batched into one
    NLI call, so total calls = n_sections × (K-1) instead of n_sentences × (K-1).

    Returns dict with keys:
        sentences   : List[str]
        uncertainty : np.ndarray  — U(s_j) ∈ [0, 1]
        mean_u      : float
        K_actual    : int
        ref_idx     : int
    """
    K = len(notes)
    if K < 2:
        print("  [luq] Need at least 2 notes — skipping.")
        return {}

    nli = _get_nli_pipeline(nli_model_name)

    ref_note      = notes[ref_idx]
    other_notes   = [n for i, n in enumerate(notes) if i != ref_idx]
    sentences     = split_sentences(ref_note)

    if not sentences:
        print("  [luq] No sentences found in reference note.")
        return {}

    sent_sections  = _assign_sentence_sections(ref_note, sentences)
    other_sections = [_parse_soap_sections(n) for n in other_notes]

    # Group sentence indices by section
    sec_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, sec in enumerate(sent_sections):
        sec_to_indices[sec].append(i)

    n_sections = len(sec_to_indices)
    print(f"  [luq] {len(sentences)} sentences across {n_sections} section(s), "
          f"scoring against {len(other_notes)} other generations …")

    entail_sum = np.zeros(len(sentences), dtype=np.float64)

    for other_sec in other_sections:
        entail_scores = np.zeros(len(sentences), dtype=np.float64)

        for sec, indices in sec_to_indices.items():
            # Use matching section; fall back to full note if section absent
            premise = (
                other_sec.get(sec)
                or "\n".join(other_sec.values())
            )
            hyps  = [sentences[i] for i in indices]
            probs = _entailment_probs(nli, hyps, premise)
            for i, p in zip(indices, probs):
                entail_scores[i] = p

        entail_sum += entail_scores

    uncertainty = 1.0 - entail_sum / len(other_notes)
    mean_u      = float(uncertainty.mean())
    print(f"  [luq] mean sentence uncertainty = {mean_u:.4f}")

    return {
        "sentences":   sentences,
        "uncertainty": uncertainty,
        "mean_u":      mean_u,
        "K_actual":    K,
        "ref_idx":     ref_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9.  HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _uncertainty_colour(u: float) -> str:
    """
    0.0 → dark green (certain)
    0.5 → white
    1.0 → deep red   (uncertain)
    """
    if u <= 0.5:
        t = u / 0.5
        r = int(20  + t * (255 - 20))
        g = int(140 + t * (255 - 140))
        b = int(20  + t * (255 - 20))
    else:
        t = (u - 0.5) / 0.5
        r = int(255)
        g = int(255 * (1 - t))
        b = int(255 * (1 - t))
    return f"rgb({r},{g},{b})"


def _colour_legend() -> str:
    stops = ", ".join(
        f"{_uncertainty_colour(i/100)} {i}%" for i in range(0, 101, 5)
    )
    return f"""
<div style="margin:12px 0;">
  <div style="height:18px;background:linear-gradient(to right,{stops});
       border-radius:4px;border:1px solid #ccc;"></div>
  <div style="display:flex;justify-content:space-between;font-size:11px;
       color:#555;margin-top:2px;">
    <span>U = 0.0 (certain)</span>
    <span>U = 0.5</span>
    <span>U = 1.0 (uncertain)</span>
  </div>
</div>"""


def _render_note_html(sentences: List[str], uncertainty: np.ndarray) -> str:
    spans = []
    for sent, u in zip(sentences, uncertainty):
        colour = _uncertainty_colour(float(u))
        esc    = sent.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        spans.append(
            f'<span class="sent" style="background:{colour};padding:2px 4px;'
            f'border-radius:3px;margin:2px 1px;display:inline-block;" '
            f'title="U={u:.3f}">{esc}</span>'
        )
    return "\n".join(spans)


def build_html_report(
    top_samples: List[Dict],
    out_path: Path,
) -> None:
    sections = []
    for rank, item in enumerate(top_samples, start=1):
        si        = item["sample_idx"]
        sentences = item["sentences"]
        uncert    = item["uncertainty"]
        mean_u    = item["mean_u"]
        K         = item["K_actual"]
        transcript_preview = item.get("transcript", "")[:500]

        note_html = _render_note_html(sentences, uncert)
        tr_html   = (transcript_preview
                     .replace("&", "&amp;").replace("<", "&lt;")
                     .replace("\n", "<br>"))

        sections.append(f"""
<div style="margin-bottom:40px;border:1px solid #ddd;border-radius:8px;
     padding:20px;background:#fafafa;">
  <h2 style="margin-top:0;">
    #{rank} — Sample {si}
    <span style="font-size:14px;font-weight:normal;color:#555;">
      Mean U = {mean_u:.4f} &nbsp;|&nbsp; K = {K}
    </span>
  </h2>
  {_colour_legend()}
  <h3>Transcript (excerpt)</h3>
  <div style="font-family:monospace;font-size:12px;background:#f4f4f4;
       border-left:4px solid #aaa;padding:10px;white-space:pre-wrap;
       margin-bottom:16px;">{tr_html}</div>
  <h3>Generated Note — sentence uncertainty</h3>
  <div style="font-family:Georgia,serif;font-size:14px;line-height:2.4;
       background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:16px;">
    {note_html}
  </div>
</div>""")

    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LUQ Sentence Uncertainty — Top {len(top_samples)} Samples</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 24px;
    color: #222; background: #f7f7f7;
  }}
  h1 {{ margin-bottom: 4px; }}
  .sent {{ cursor: default; }}
</style>
</head>
<body>
<h1>LUQ Sentence-Level Uncertainty — Top {len(top_samples)} Most Uncertain Samples</h1>
<p style="color:#555;margin-bottom:28px;">
  Model: <b>Llama 3.1 8B (Bedrock)</b> &nbsp;|&nbsp;
  Prompt: Asgari et al. 2025, Experiment 8 (PDSQI-9 optimised)<br>
  NLI: <b>{_NLI_MODEL_NAME}</b> &nbsp;|&nbsp;
  U = 1 − mean P(entail | s, r') across K−1 other generations.<br>
  Hover over a sentence to see its exact uncertainty score.
</p>
{body}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[html] Written to {out_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Stage 1 — Generation
# ─────────────────────────────────────────────────────────────────────────────

def stage_generate(
    start: int,
    end: int,
    K: int,
    temperature: float,
    max_tokens: int,
    bedrock_region: str,
    bedrock_model: str,
    out_dir: Path,
    use_cache: bool = True,
) -> None:
    """Generate K notes per sample and save to generations/sample_NNN_generations.json."""
    gen_dir = out_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    ds  = load_aci_bench()
    end = min(end, len(ds))

    skipped = generated = failed = 0

    for si in tqdm(range(start, end), desc="generate", unit="sample"):
        print(f"\n{'='*56}")
        print(f"  [generate] Sample {si}")
        print(f"{'='*56}")

        gen_path = gen_dir / f"sample_{si:03d}_generations.json"

        if use_cache and gen_path.exists():
            saved = json.loads(gen_path.read_text())
            print(f"  [cache] {len(saved['notes'])} generations already exist — skipping.")
            skipped += 1
            continue

        row = ds[si]
        try:
            transcript, gold_note = _get_transcript_and_note(row)
        except Exception as exc:
            print(f"  [data] Failed: {exc}")
            failed += 1
            continue

        print(f"  Transcript: {len(transcript)} chars")
        notes = generate_K_notes(
            transcript,
            K=K,
            bedrock_region=bedrock_region,
            model_id=bedrock_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not notes:
            print("  [gen] No notes generated — skipping.")
            failed += 1
            continue

        gen_path.write_text(json.dumps({
            "sample_idx":  si,
            "transcript":  transcript,
            "gold_note":   gold_note,
            "notes":       notes,
            "model":       bedrock_model,
            "K":           K,
            "temperature": temperature,
        }, indent=2))
        print(f"  [gen] Saved {len(notes)} notes → {gen_path.name}")
        generated += 1

    print(f"\n[generate] Done — generated={generated}  skipped={skipped}  failed={failed}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Stage 2 — NLI Scoring
# ─────────────────────────────────────────────────────────────────────────────

def stage_score(
    start: int,
    end: int,
    nli_model_name: str,
    out_dir: Path,
    sent_dir: Path,
    use_cache: bool = True,
    top_n: int = 3,
) -> pd.DataFrame:
    """Load generation JSONs, run NLI scoring, save sentence CSVs and summary."""
    gen_dir = out_dir / "generations"
    sent_dir.mkdir(parents=True, exist_ok=True)

    ds  = load_aci_bench()
    end = min(end, len(ds))

    summary_rows: List[Dict]      = []
    all_results:  Dict[int, Dict] = {}

    for si in tqdm(range(start, end), desc="score", unit="sample"):
        print(f"\n{'='*56}")
        print(f"  [score] Sample {si}")
        print(f"{'='*56}")

        # ── Cache check: all sentence CSVs already exist ──────────────────────
        existing_sent = sorted(sent_dir.glob(f"sample_{si:03d}_note_*_sentences.csv"))
        if use_cache and existing_sent:
            print(f"  [cache] {len(existing_sent)} sentence CSVs found — skipping NLI.")
            all_note_results = []
            for sp in existing_sent:
                df = pd.read_csv(sp)
                k  = int(sp.stem.split("_note_")[1].split("_")[0])
                all_note_results.append({
                    "sentences":   df["sentence"].tolist(),
                    "uncertainty": df["uncertainty"].values.astype(float),
                    "mean_u":      float(df["uncertainty"].mean()),
                    "K_actual":    len(existing_sent),
                    "ref_idx":     k,
                })
            transcript = gold_note = ""
            notes = []
        else:
            # ── Load generation JSON ──────────────────────────────────────────
            gen_path = gen_dir / f"sample_{si:03d}_generations.json"
            if not gen_path.exists():
                print(f"  [score] No generation file — run --stage generate first.")
                summary_rows.append({
                    "sample_idx": si, "status": "no_generations",
                    "mean_u": float("nan"), "K_actual": 0,
                })
                continue

            saved      = json.loads(gen_path.read_text())
            notes      = saved["notes"]
            transcript = saved.get("transcript", "")
            gold_note  = saved.get("gold_note", "")
            print(f"  Loaded {len(notes)} generations.")

            if len(notes) < 2:
                print("  [score] Fewer than 2 notes — skipping.")
                summary_rows.append({
                    "sample_idx": si, "status": "insufficient_generations",
                    "mean_u": float("nan"), "K_actual": len(notes),
                })
                continue

            # ── NLI scoring ───────────────────────────────────────────────────
            all_note_results = []
            for k in range(len(notes)):
                result_k = compute_luq_sentence(notes, nli_model_name=nli_model_name, ref_idx=k)
                if not result_k:
                    print(f"  [score] note {k} failed — skipping.")
                    continue
                sent_df = pd.DataFrame({
                    "sentence_idx": np.arange(len(result_k["sentences"])),
                    "sentence":     result_k["sentences"],
                    "uncertainty":  result_k["uncertainty"].round(4),
                })
                sent_df.to_csv(sent_dir / f"sample_{si:03d}_note_{k:02d}_sentences.csv", index=False)
                all_note_results.append(result_k)

        if not all_note_results:
            summary_rows.append({
                "sample_idx": si, "status": "luq_failed",
                "mean_u": float("nan"), "K_actual": 0,
            })
            continue

        result     = all_note_results[0]
        mean_u_all = float(np.mean([r["mean_u"] for r in all_note_results]))
        max_u_all  = float(np.max([r["uncertainty"].max() for r in all_note_results]))
        all_u      = np.concatenate([r["uncertainty"] for r in all_note_results])

        all_results[si] = {
            "sample_idx": si,
            "transcript": transcript,
            "gold_note":  gold_note,
            "notes":      notes,
            **result,
        }
        summary_rows.append({
            "sample_idx":   si,
            "status":       "ok",
            "mean_u":       round(mean_u_all, 4),
            "max_u":        round(max_u_all, 4),
            "n_sentences":  len(result["sentences"]),
            "K_actual":     result["K_actual"],
            "notes_scored": len(all_note_results),
            "high_u_frac":  round(float((all_u > 0.5).mean()), 4),
        })
        print(f"  mean_U={mean_u_all:.4f}  n_sentences={len(result['sentences'])}  "
              f"K={result['K_actual']}  notes_scored={len(all_note_results)}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "luq_results.csv", index=False)
    print(f"\n[score] {len(summary_rows)} samples → {out_dir / 'luq_results.csv'}")

    # ── Top-N HTML report ─────────────────────────────────────────────────────
    ok_df       = summary_df[summary_df["status"] == "ok"].sort_values("mean_u", ascending=False)
    top_indices = ok_df["sample_idx"].head(top_n).tolist()
    top_data    = [all_results[i] for i in top_indices if i in all_results]
    if top_data:
        build_html_report(top_data, out_dir / "top3_report.html")

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# 12.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentence-level LUQ uncertainty on ACI-Bench — Llama 3.1 8B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  generate  Call Bedrock to produce K notes per sample; saves to <out>/generations/.
  score     Run NLI scoring on cached generations; saves sentence CSVs + summary.
  all       Run both stages in sequence (default).

Examples:
  python luq_sentence.py --stage generate --start 0 --end 132
  python luq_sentence.py --stage generate --start 0 --end 132 --no-cache
  python luq_sentence.py --stage score    --start 0 --end 132
  python luq_sentence.py --stage score    --start 0 --end 132 --no-cache
  python luq_sentence.py --stage all      --start 0 --end 132
""",
    )
    p.add_argument("--stage",          choices=["generate", "score", "all"], default="all",
                   help="Which stage to run (default: all)")
    p.add_argument("--no-cache",       action="store_true",
                   help="Ignore existing cached files for the selected stage and reprocess")
    p.add_argument("--start",          type=int,   default=0,
                   help="First sample index inclusive (default 0)")
    p.add_argument("--end",            type=int,   default=132,
                   help="Last sample index exclusive (default 132)")
    p.add_argument("--out",            default="luq_out/llama",
                   help="Output root directory (default luq_out/llama)")

    # Generation options
    g = p.add_argument_group("generation")
    g.add_argument("--K",              type=int,   default=10,
                   help="Generations per sample (default 10)")
    g.add_argument("--temperature",    type=float, default=0.7)
    g.add_argument("--max-tokens",     type=int,   default=1024)
    g.add_argument("--bedrock-region", default=os.environ.get(
                       "AWS_DEFAULT_REGION", _BEDROCK_DEFAULT_REGION))
    g.add_argument("--bedrock-model",  default=_BEDROCK_GEN_MODEL)

    # Scoring options
    s = p.add_argument_group("scoring")
    s.add_argument("--nli-model",      default=_NLI_MODEL_NAME)
    s.add_argument("--top-n",          type=int,   default=3,
                   help="Samples in HTML report (default 3)")

    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.out).resolve()
    sent_dir = out_dir / "sentences"

    print(f"\n  Stage          : {args.stage}")
    print(f"  Cache          : {'disabled (--no-cache)' if args.no_cache else 'enabled'}")
    print(f"  Samples        : {args.start} – {args.end - 1}")
    print(f"  Output dir     : {out_dir}")
    if args.stage in ("generate", "all"):
        print(f"  Bedrock model  : {args.bedrock_model}")
        print(f"  K generations  : {args.K}  temp={args.temperature}")
    if args.stage in ("score", "all"):
        print(f"  NLI model      : {args.nli_model}")
        print(f"  Sentences dir  : {sent_dir}")
    print()

    if args.stage in ("generate", "all"):
        stage_generate(
            start=args.start, end=args.end,
            K=args.K, temperature=args.temperature, max_tokens=args.max_tokens,
            bedrock_region=args.bedrock_region, bedrock_model=args.bedrock_model,
            out_dir=out_dir, use_cache=not args.no_cache,
        )

    if args.stage in ("score", "all"):
        stage_score(
            start=args.start, end=args.end,
            nli_model_name=args.nli_model,
            out_dir=out_dir, sent_dir=sent_dir,
            use_cache=not args.no_cache, top_n=args.top_n,
        )

    print("\n" + "═" * 56)
    print(f"  Done. Results in: {out_dir}")
    print("═" * 56 + "\n")


if __name__ == "__main__":
    main()
