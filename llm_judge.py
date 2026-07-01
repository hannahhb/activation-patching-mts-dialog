"""
llm_judge.py
============
LLM-as-a-judge faithfulness evaluation using Llama 3.3 70B.

Modes (--mode):

  sentence  — 1 LLM call per note. Splits the note into sentences and
              classifies each against the transcript.
                sentence_idx, sentence, label

  fact      — 1 LLM call per atomic fact. Reads facts from the atomic_luq
              facts CSV.
                fact_idx, section, fact, label, reasoning

  span      — Like sentence but non-Faithful sentences also carry the exact
              verbatim note span responsible for the error.
                sentence_idx, sentence, label, note_span

By default only note_00 is evaluated. Pass --all-notes to evaluate every
generation for each sample.

Categories (all modes):
  Faithful    — Fully supported by the transcript.
  Fabrication — Not mentioned in the transcript at all.
  Negation    — Directly contradicts something stated in the transcript.
  Causality   — Causal or temporal relationship is wrong or unsupported.
  Contextual  — Content present in the transcript but misattributed
                (e.g. said by the doctor, attributed to the patient).

Usage:
    python llm_judge.py --mode sentence --start 0 --end 44
    python llm_judge.py --mode span --start 0 --end 44 --all-notes
    python llm_judge.py --mode fact --start 0 --end 44 --all-notes
    python llm_judge.py --mode sentence --no-cache
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from factmatch_sentence import get_bedrock_client

# ── Constants ─────────────────────────────────────────────────────────────────
BEDROCK_MODEL    = "us.meta.llama3-3-70b-instruct-v1:0"
DEFAULT_GEN_DIR  = "luq_out/llama/generations"
DEFAULT_FACTS_DIR = "luq_out/llama_atomic/facts"
DEFAULT_OUT_DIR  = "luq_out/llama_judge"
MAX_RETRIES      = 3
RETRY_SLEEP      = 4
LABELS           = {"Faithful", "Fabrication", "Negation", "Causality", "Contextual"}

# ── Prompts ───────────────────────────────────────────────────────────────────

# Sentence-level: 1 call per note. The LLM segments the note into sentences
# and classifies each one. Grounded in CREOLA taxonomy (Asagri et al.) and
# BioNLP faithfulness evaluation conventions.
_SENTENCE_PROMPT = """\
You are an expert clinical documentation auditor. Your task is to evaluate \
the faithfulness of a generated SOAP note against the source \
doctor-patient encounter transcript.

The transcript was captured by an ambient recording device and may contain \
automatic speaker diarisation errors (doctor and patient labels occasionally \
swapped). When assessing faithfulness, focus on whether the clinical content \
is present in the transcript regardless of which speaker it is attributed to.

TRANSCRIPT:
{transcript}

GENERATED SOAP NOTE:
{note}

Instructions:
Split the note into individual sentences. For each sentence assign exactly \
one label from the taxonomy below.

TAXONOMY:
  Faithful    — The sentence is fully supported by the transcript.
  Fabrication — The sentence contains information not mentioned anywhere \
in the transcript.
  Negation    — The sentence contradicts something explicitly stated in \
the transcript.
  Causality   — The sentence describes a causal or temporal relationship \
that is absent or incorrect in the transcript.
  Contextual  — The content exists in the transcript but is misattributed \
or taken out of context (e.g. said by the doctor but presented as \
patient-reported, or a conditional stated as definite).

Return a JSON array — one object per sentence, in note order:
[
  {{"sentence_idx": 0, "sentence": "<exact sentence text>", "label": "<label>"}},
  ...
]

Return only the JSON array, no other text."""


# Span-level: 1 call per sample. Like sentence mode but for non-Faithful
# sentences also returns the verbatim note span responsible for the error.
_SPAN_PROMPT = """\
You are an expert clinical documentation auditor. Your task is to evaluate \
the faithfulness of a generated SOAP note against the source \
doctor-patient encounter transcript, and for any errors identify the exact \
words in the note that are problematic.

The transcript was captured by an ambient recording device and may contain \
automatic speaker diarisation errors (doctor and patient labels occasionally \
swapped). When assessing faithfulness, focus on whether the clinical content \
is present in the transcript regardless of which speaker it is attributed to.

TRANSCRIPT:
{transcript}

GENERATED SOAP NOTE:
{note}

Instructions:
Split the note into individual sentences. For each sentence assign exactly \
one label from the taxonomy below. For any sentence that is NOT Faithful, \
also provide the exact verbatim span from that sentence that is erroneous \
(the minimal phrase that causes the error — not the whole sentence).

TAXONOMY:
  Faithful    — The sentence is fully supported by the transcript.
  Fabrication — The sentence contains information not mentioned anywhere \
in the transcript.
  Negation    — The sentence contradicts something explicitly stated in \
the transcript.
  Causality   — The sentence describes a causal or temporal relationship \
that is absent or incorrect in the transcript.
  Contextual  — The content exists in the transcript but is misattributed \
or taken out of context (e.g. said by the doctor but presented as \
patient-reported, or a conditional stated as definite).

Return a JSON array — one object per sentence, in note order.
For Faithful sentences: {{"sentence_idx": 0, "sentence": "...", "label": "Faithful"}}
For non-Faithful sentences: {{"sentence_idx": 1, "sentence": "...", "label": "Fabrication", "note_span": "<exact words from the sentence>"}}

Return only the JSON array, no other text."""


# Fact-level: 1 call per atomic fact.
_FACT_PROMPT = """\
You are an expert clinical documentation auditor evaluating whether an \
atomic fact from a generated SOAP note is faithful to the source \
doctor-patient transcript.

The transcript may contain speaker diarisation errors (doctor/patient labels \
occasionally swapped). Focus on whether the clinical content is present \
regardless of attribution.

TRANSCRIPT:
{transcript}

SOAP SECTION: {section}
ATOMIC FACT: "{fact}"

Classify this fact using exactly one label:

  Faithful    — Clearly supported by the transcript.
  Fabrication — Not mentioned anywhere in the transcript.
  Negation    — Directly contradicts something stated in the transcript.
  Causality   — Causal or temporal relationship is wrong or unsupported.
  Contextual  — Present in the transcript but misattributed or taken out \
of context.

Respond with a JSON object on a single line:
{{"label": "<label>", "reasoning": "<one sentence>"}}"""


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int = 4096) -> Optional[str]:
    client = get_bedrock_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.converse(
                modelId=BEDROCK_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
            return resp["output"]["message"]["content"][0]["text"].strip()
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                tqdm.write(f"  [llm] error (attempt {attempt+1}): {exc}")
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                tqdm.write(f"  [llm] failed: {exc}")
                return None


def _parse_label(text: str) -> Dict[str, str]:
    """Extract label (+optional reasoning) from a single-fact response."""
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            label = obj.get("label", "").strip()
            if label in LABELS:
                return {"label": label,
                        "reasoning": obj.get("reasoning", "").strip()}
        except json.JSONDecodeError:
            pass
    for label in LABELS:
        if label.lower() in text.lower():
            return {"label": label, "reasoning": text[:200]}
    return {"label": "PARSE_ERROR", "reasoning": text[:200]}


def _parse_sentence_array(text: str) -> List[Dict]:
    """Extract JSON array of sentence classifications."""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
        out = []
        for item in items:
            label = item.get("label", "").strip()
            out.append({
                "sentence_idx": item.get("sentence_idx", len(out)),
                "sentence":     item.get("sentence", "").strip(),
                "label":        label if label in LABELS else "PARSE_ERROR",
            })
        return out
    except json.JSONDecodeError:
        return []


def _parse_span_array(text: str) -> List[Dict]:
    """Extract JSON array of sentence + label + optional note_span."""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
        out = []
        for item in items:
            label = item.get("label", "").strip()
            row = {
                "sentence_idx": item.get("sentence_idx", len(out)),
                "sentence":     item.get("sentence", "").strip(),
                "label":        label if label in LABELS else "PARSE_ERROR",
                "note_span":    item.get("note_span", ""),
            }
            out.append(row)
        return out
    except json.JSONDecodeError:
        return []


# ── Sentence-level judge ──────────────────────────────────────────────────────

def judge_sentences(sample_idx: int, transcript: str, note: str,
                    out_dir: Path, use_cache: bool,
                    note_idx: int = 0) -> List[dict]:
    csv_p = out_dir / "sentences" / f"sample_{sample_idx:03d}_note_{note_idx:02d}_sentence_judge.csv"
    csv_p.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and csv_p.exists():
        df = pd.read_csv(csv_p)
        tqdm.write(f"  [cache] sentence judge sample {sample_idx} note {note_idx}")
        return df.to_dict("records")

    prompt = _SENTENCE_PROMPT.format(transcript=transcript, note=note)
    raw = _call_llm(prompt)
    if raw is None:
        return []

    rows = _parse_sentence_array(raw)
    if not rows:
        tqdm.write(f"  [sentence] parse error for sample {sample_idx} note {note_idx}")
        return []

    df = pd.DataFrame(rows)
    df.to_csv(csv_p, index=False)
    counts = df["label"].value_counts().to_dict()
    tqdm.write(f"  [sentence] sample {sample_idx} note {note_idx}: {len(df)} sentences — {counts}")
    return rows


# ── Span-level judge ─────────────────────────────────────────────────────────

def judge_spans(sample_idx: int, transcript: str, note: str,
                out_dir: Path, use_cache: bool,
                note_idx: int = 0) -> List[dict]:
    csv_p = out_dir / "spans" / f"sample_{sample_idx:03d}_note_{note_idx:02d}_span_judge.csv"
    csv_p.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and csv_p.exists():
        df = pd.read_csv(csv_p)
        tqdm.write(f"  [cache] span judge sample {sample_idx} note {note_idx}")
        return df.to_dict("records")

    prompt = _SPAN_PROMPT.format(transcript=transcript, note=note)
    raw = _call_llm(prompt)
    if raw is None:
        return []

    rows = _parse_span_array(raw)
    if not rows:
        tqdm.write(f"  [span] parse error for sample {sample_idx} note {note_idx}")
        return []

    df = pd.DataFrame(rows)
    df.to_csv(csv_p, index=False)
    counts = df["label"].value_counts().to_dict()
    n_spans = (df["note_span"].notna() & (df["note_span"] != "")).sum()
    tqdm.write(f"  [span] sample {sample_idx} note {note_idx}: {len(df)} sentences, "
               f"{n_spans} spans — {counts}")
    return rows


# ── Fact-level judge ──────────────────────────────────────────────────────────

def judge_facts(sample_idx: int, transcript: str, facts_df: pd.DataFrame,
                out_dir: Path, use_cache: bool,
                note_idx: int = 0) -> List[dict]:
    csv_p = out_dir / "facts" / f"sample_{sample_idx:03d}_note_{note_idx:02d}_fact_judge.csv"
    csv_p.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and csv_p.exists():
        df = pd.read_csv(csv_p)
        tqdm.write(f"  [cache] fact judge sample {sample_idx} note {note_idx}")
        return df.to_dict("records")

    rows = []
    for _, row in facts_df.iterrows():
        prompt = _FACT_PROMPT.format(
            transcript=transcript,
            section=row["section"],
            fact=row["fact"],
        )
        raw = _call_llm(prompt, max_tokens=256)
        parsed = _parse_label(raw) if raw else {"label": "ERROR", "reasoning": ""}

        rows.append({
            "fact_idx":  int(row["fact_idx"]),
            "section":   row["section"],
            "fact":      row["fact"],
            "label":     parsed["label"],
            "reasoning": parsed["reasoning"],
        })
        tqdm.write(f"    [{parsed['label']}] {row['fact'][:70]}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_p, index=False)
    counts = df["label"].value_counts().to_dict()
    tqdm.write(f"  [fact] sample {sample_idx} note {note_idx}: {len(df)} facts — {counts}")
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-a-judge faithfulness evaluation")
    p.add_argument("--mode",       choices=["sentence", "fact", "span"], required=True,
                   help="sentence: classify sentences | fact: classify atomic facts | "
                        "span: classify sentences + extract erroneous note spans")
    p.add_argument("--start",      type=int, default=0)
    p.add_argument("--end",        type=int, default=132)
    p.add_argument("--gen-dir",    default=DEFAULT_GEN_DIR)
    p.add_argument("--facts-dir",  default=DEFAULT_FACTS_DIR,
                   help="Directory containing atomic_luq per-note fact CSVs (fact mode)")
    p.add_argument("--out",        default=DEFAULT_OUT_DIR)
    p.add_argument("--notes",      type=int, default=1, metavar="N",
                   help="Number of generations to evaluate per sample (default 1 = note_00 only)")
    p.add_argument("--all-notes",  action="store_true",
                   help="Evaluate every generation per sample (overrides --notes)")
    p.add_argument("--no-cache",   action="store_true")
    return p.parse_args()


def main():
    args      = parse_args()
    gen_dir   = Path(args.gen_dir).resolve()
    facts_dir = Path(args.facts_dir).resolve()
    out_dir   = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    if args.all_notes:
        notes_desc = "all notes"
    elif args.notes == 1:
        notes_desc = "note_00 only"
    else:
        notes_desc = f"first {args.notes} notes"
    print(f"Mode      : {args.mode}")
    print(f"Model     : {BEDROCK_MODEL}")
    print(f"Gen dir   : {gen_dir}")
    print(f"Out dir   : {out_dir}")
    print(f"Samples   : {args.start} – {args.end - 1}  ({notes_desc})")
    print(f"Cache     : {'on' if use_cache else 'off'}")

    summary_rows = []
    for sample_idx in tqdm(range(args.start, args.end), desc="samples", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            tqdm.write(f"  [skip] sample {sample_idx}: no generation file")
            continue

        gen        = json.loads(gen_path.read_text())
        transcript = gen["transcript"]
        all_notes  = gen["notes"]
        if args.all_notes:
            note_indices = list(range(len(all_notes)))
        else:
            note_indices = list(range(min(args.notes, len(all_notes))))

        tqdm.write(f"\n[sample {sample_idx}] {len(all_notes)} notes, "
                   f"evaluating {len(note_indices)}")

        for note_idx in note_indices:
            note = all_notes[note_idx]

            if args.mode == "sentence":
                rows = judge_sentences(sample_idx, transcript, note, out_dir,
                                       use_cache, note_idx)
                if rows:
                    df = pd.DataFrame(rows)
                    counts = df["label"].value_counts().to_dict()
                    summary_rows.append({
                        "sample_idx":  sample_idx,
                        "note_idx":    note_idx,
                        "n_sentences": len(df),
                        "halluc_rate": round(1 - counts.get("Faithful", 0) / len(df), 4),
                        **{f"n_{l.lower()}": counts.get(l, 0) for l in LABELS},
                    })

            elif args.mode == "fact":
                facts_csv = facts_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts.csv"
                if not facts_csv.exists():
                    tqdm.write(f"  [skip] sample {sample_idx} note {note_idx}: no facts CSV "
                               f"(run atomic_luq.py first)")
                    continue
                facts_df = pd.read_csv(facts_csv)
                rows = judge_facts(sample_idx, transcript, facts_df, out_dir,
                                   use_cache, note_idx)
                if rows:
                    df = pd.DataFrame(rows)
                    counts = df["label"].value_counts().to_dict()
                    summary_rows.append({
                        "sample_idx":  sample_idx,
                        "note_idx":    note_idx,
                        "n_facts":     len(df),
                        "halluc_rate": round(1 - counts.get("Faithful", 0) / len(df), 4),
                        **{f"n_{l.lower()}": counts.get(l, 0) for l in LABELS},
                    })

            else:  # span
                rows = judge_spans(sample_idx, transcript, note, out_dir,
                                   use_cache, note_idx)
                if rows:
                    df = pd.DataFrame(rows)
                    counts = df["label"].value_counts().to_dict()
                    n_spans = (df["note_span"].notna() & (df["note_span"] != "")).sum()
                    summary_rows.append({
                        "sample_idx":  sample_idx,
                        "note_idx":    note_idx,
                        "n_sentences": len(df),
                        "n_spans":     int(n_spans),
                        "halluc_rate": round(1 - counts.get("Faithful", 0) / len(df), 4),
                        **{f"n_{l.lower()}": counts.get(l, 0) for l in LABELS},
                    })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out_csv = out_dir / f"judge_{args.mode}_results.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\n[done] summary → {out_csv}")
        print(summary.describe())
        print("\nLabel totals:")
        for l in LABELS:
            col = f"n_{l.lower()}"
            if col in summary:
                print(f"  {l:<15} {summary[col].sum():>5.0f}")


if __name__ == "__main__":
    main()
