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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

_SYSTEM_PROMPT = (
    "You are a medical office assistant drafting documentation for a physician. "
    "DO NOT ADD any content that isn't specifically mentioned IN THE TRANSCRIPT. "
    "From the attached transcript generate a SOAP note based on the below template "
    "format for the physician to review, include all the relevant information and "
    "do not include any information that isn't explicitly mentioned in the transcript. "
    "If nothing is mentioned just return [NOT MENTIONED].\n\n"
    "It is VITAL that all the information in the note is as accurate as possible. "
    "Avoid repeating the same information in different sections where possible. "
    "Write the note from the perspective of the physician. "
    "Only include any section of the template if there is information from the "
    "transcript, otherwise omit it. "
    "Begin your response directly with 'Subjective:' — do not add any preamble, "
    "introduction, or heading before the note."
)

_SOAP_TEMPLATE = """Template for Clinical SOAP Note Format:

Subjective:
- HPI: [include here any mentioned symptoms, chronological narrative of patients \
complaints, information obtained from other sources (always identify source if not \
the patient).]
- Past medical history: [include here all of the patients past conditions, treatments \
and encounters, also include relevant social history here including smoking, alcohol, \
drug use and occupation/travel history]
- Review of systems: [include here any additional symptoms in other organs that is \
relevant to the initial presentation]
- Current medications: [list medicines each on a separate line in the format: \
[DRUG NAME] [DRUG DOSE] [DRUG FREQUENCY] [INDICATION]]

Objective:
- Vital signs: [including any mentioned blood pressure, pulse rate, oxygen saturation, \
temperature]
- Physical exam: [the examination findings from the physical exam, if mentioned]
- Test Results: [include in this section any lab test results or imaging reports]

Assessment / Problem List:
- Assessment: [A one-sentence description of the patient and major problem as described \
by the physician, including the diagnosis the physician has identified]
- Problem list: [List clinical problems inline, separated by semicolons, on a single line. \
Format each as [Condition] [Status: active/suspected/confirmed/past/unknown]. \
Leave status as unknown if not mentioned in the transcript. \
Do not use numbered lists or line breaks between problems.]

Plan:
[include here any management plan mentioned in the transcript, including patient \
education, prescriptions, tests, referrals or other plans.]

Follow-up: [include here any plan mentioned to see the patient again, or to be \
discharged.]"""

_STYLE_GUIDELINES = """Please adhere to the following style guidelines:
- Write from the perspective of the physician (first person)
- Write ONLY in complete, grammatical sentences. Do NOT use bullet points, hyphens, \
numbered lists, or any other list formatting anywhere in the note.
- Be ultra-precise, do not use generalising terms
- Be highly detailed
- Include ALL important negations (e.g. "The patient denies fever.") as well as all \
positive findings, written as full sentences.
- List medications as a sentence: "I prescribed [drug] [dose] [frequency] for [indication]."
- Always document if drug allergies are present or not
- Examination findings always refer to physical exam signs only, not symptoms
- Preserve quantities if mentioned in the text"""

_USER_TEMPLATE = (
    "{system}\n\n"
    "{template}\n\n"
    "{style}\n\n"
    "Transcript:\n{transcript}"
)


def _build_user_message(transcript: str) -> str:
    return _USER_TEMPLATE.format(
        system=_SYSTEM_PROMPT,
        template=_SOAP_TEMPLATE,
        style=_STYLE_GUIDELINES,
        transcript=transcript.strip(),
    )


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
    for k in range(K):
        try:
            note = generate_note(
                transcript,
                bedrock_region=bedrock_region,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            notes.append(note)
            print(f"  [gen] {k+1}/{K} done ({len(note)} chars)")
        except Exception as exc:
            print(f"  [gen] {k+1}/{K} failed: {exc}")
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

_PERIOD_SPLIT = re.compile(r"(?<=\.)\s+(?=[A-Z])")


def split_sentences(text: str) -> List[str]:
    """Split on newlines and '. [Capital]', stripping bullet markers."""
    results = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-•*#+").strip()
        if not line:
            continue
        for part in _PERIOD_SPLIT.split(line):
            part = part.strip()
            if part:
                results.append(part)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  NLI model
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


def _entailment_probs(
    nli: object,
    hypotheses: List[str],
    premise: str,
) -> np.ndarray:
    """
    For each hypothesis h in `hypotheses`, compute
        P(entail | h, premise)  =  exp(l_e) / (exp(l_e) + exp(l_c))
    using raw logits and binary renormalisation over {entail, contra},
    as in the LUQ paper (Zhang et al., 2024).

    DeBERTa NLI cross-encoders expect (premise, hypothesis) order.
    Returns array of shape (len(hypotheses),).
    """
    if not hypotheses:
        return np.array([])

    # NLI convention: first element is premise, second is hypothesis
    pairs = [(premise, h) for h in hypotheses]

    raw = nli.predict(pairs, apply_softmax=False)   # raw logits (N, 3)
    raw = np.array(raw)

    if raw.ndim == 2:
        # Resolve label order from model config
        id2label = nli.model.config.id2label
        entail_idx = next(
            (int(i) for i, l in id2label.items() if "entail" in l.lower()), 2
        )
        contra_idx = next(
            (int(i) for i, l in id2label.items() if "contra" in l.lower()), 0
        )
        l_e = raw[:, entail_idx]
        l_c = raw[:, contra_idx]
        # Binary softmax over {entail, contra} — equivalent to LUQ formula
        exp_e = np.exp(l_e - np.maximum(l_e, l_c))   # numerically stable
        exp_c = np.exp(l_c - np.maximum(l_e, l_c))
        return exp_e / (exp_e + exp_c)
    else:
        # Single-score model — treat as entailment probability directly
        return raw


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

    ref_note    = notes[ref_idx]
    other_notes = [n for i, n in enumerate(notes) if i != ref_idx]
    sentences   = split_sentences(ref_note)

    if not sentences:
        print("  [luq] No sentences found in reference note.")
        return {}

    print(f"  [luq] {len(sentences)} sentences, scoring against "
          f"{len(other_notes)} other generations …")

    uncertainty = np.zeros(len(sentences), dtype=np.float64)

    for i, sent in enumerate(sentences):
        pair_probs = []
        for r_prime in other_notes:
            p = _entailment_probs(nli, [sent], r_prime)
            pair_probs.append(float(p[0]))

        mean_entail    = float(np.mean(pair_probs)) if pair_probs else 0.0
        uncertainty[i] = 1.0 - mean_entail

    mean_u = float(uncertainty.mean())
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
# 10.  Main batch loop
# ─────────────────────────────────────────────────────────────────────────────

def run_luq_batch(
    start: int = 0,
    end: int = 10,
    K: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    bedrock_region: str = _BEDROCK_DEFAULT_REGION,
    bedrock_model: str = _BEDROCK_GEN_MODEL,
    nli_model_name: str = _NLI_MODEL_NAME,
    out: Path = Path("."),
    top_n: int = 3,
    cache_generations: bool = True,
) -> pd.DataFrame:

    out_dir  = out / "luq_out"
    gen_dir  = out_dir / "generations"
    sent_dir = out_dir / "sentences"
    for d in [out_dir, gen_dir, sent_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ds  = load_aci_bench()
    end = min(end, len(ds))

    summary_rows: List[Dict]     = []
    all_results:  Dict[int, Dict] = {}

    for si in range(start, end):
        print(f"\n{'='*56}")
        print(f"  Sample {si}  ({si - start + 1}/{end - start})")
        print(f"{'='*56}")

        row = ds[si]
        try:
            transcript, gold_note = _get_transcript_and_note(row)
        except Exception as exc:
            print(f"  [data] Failed: {exc}")
            continue

        print(f"  Transcript: {len(transcript)} chars")

        # ── Generation (with optional cache) ─────────────────────────────────
        gen_path = gen_dir / f"sample_{si:03d}_generations.json"
        if cache_generations and gen_path.exists():
            with open(gen_path) as f:
                saved = json.load(f)
            notes = saved["notes"]
            print(f"  [cache] Loaded {len(notes)} cached generations.")
        else:
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
                summary_rows.append({
                    "sample_idx": si, "status": "no_generations",
                    "mean_u": float("nan"), "K_actual": 0,
                })
                continue
            with open(gen_path, "w") as f:
                json.dump({
                    "sample_idx":  si,
                    "transcript":  transcript,
                    "gold_note":   gold_note,
                    "notes":       notes,
                    "model":       bedrock_model,
                    "K":           K,
                    "temperature": temperature,
                }, f, indent=2)

        if len(notes) < 2:
            print("  [luq] Fewer than 2 notes — skipping uncertainty.")
            summary_rows.append({
                "sample_idx": si, "status": "insufficient_generations",
                "mean_u": float("nan"), "K_actual": len(notes),
            })
            continue

        # ── LUQ uncertainty for all K notes ──────────────────────────────────
        all_note_results: List[Dict] = []
        for k in range(len(notes)):
            result_k = compute_luq_sentence(
                notes, nli_model_name=nli_model_name, ref_idx=k
            )
            if not result_k:
                print(f"  [luq] note {k} failed — skipping.")
                continue
            sent_df = pd.DataFrame({
                "sentence_idx": np.arange(len(result_k["sentences"])),
                "sentence":     result_k["sentences"],
                "uncertainty":  result_k["uncertainty"].round(4),
            })
            sent_df.to_csv(
                sent_dir / f"sample_{si:03d}_note_{k:02d}_sentences.csv",
                index=False,
            )
            all_note_results.append(result_k)

        if not all_note_results:
            summary_rows.append({
                "sample_idx": si, "status": "luq_failed",
                "mean_u": float("nan"), "K_actual": len(notes),
            })
            continue

        # ── Store note[0] result for HTML report ──────────────────────────────
        result = all_note_results[0]
        all_results[si] = {
            "sample_idx": si,
            "transcript": transcript,
            "gold_note":  gold_note,
            "notes":      notes,
            **result,
        }

        # Aggregate across all successfully scored notes
        mean_u_all    = float(np.mean([r["mean_u"] for r in all_note_results]))
        max_u_all     = float(np.max([r["uncertainty"].max() for r in all_note_results]))
        all_u         = np.concatenate([r["uncertainty"] for r in all_note_results])
        high_u_frac   = float((all_u > 0.5).mean())

        summary_rows.append({
            "sample_idx":   si,
            "status":       "ok",
            "mean_u":       round(mean_u_all, 4),
            "max_u":        round(max_u_all, 4),
            "n_sentences":  len(result["sentences"]),
            "K_actual":     result["K_actual"],
            "notes_scored": len(all_note_results),
            "high_u_frac":  round(high_u_frac, 4),
        })

        print(f"  mean_U (all notes)={mean_u_all:.4f}  "
              f"n_sentences={len(result['sentences'])}  K={result['K_actual']}  "
              f"notes_scored={len(all_note_results)}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "luq_results.csv", index=False)
    print(f"\n[done] {len(summary_rows)} samples → luq_results.csv")

    # ── Top-N HTML report ─────────────────────────────────────────────────────
    ok_df = summary_df[summary_df["status"] == "ok"].sort_values(
        "mean_u", ascending=False
    )
    top_indices = ok_df["sample_idx"].head(top_n).tolist()
    top_data    = [all_results[i] for i in top_indices if i in all_results]

    if top_data:
        build_html_report(top_data, out_dir / "top3_report.html")
        print(f"\n[top-{top_n}] Most uncertain samples: {top_indices}")
        for si in top_indices:
            r = all_results.get(si)
            if r:
                print(f"  Sample {si}: mean_U={r['mean_u']:.4f}  "
                      f"n_sentences={len(r['sentences'])}")

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# 11.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentence-level LUQ uncertainty on ACI-Bench via Bedrock Llama 3.1 8B"
    )
    p.add_argument("--start",          type=int,   default=0,
                   help="First sample index (inclusive, default 0)")
    p.add_argument("--end",            type=int,   default=10,
                   help="Last sample index (exclusive, default 10)")
    p.add_argument("--K",              type=int,   default=10,
                   help="Generations per sample (default 10)")
    p.add_argument("--temperature",    type=float, default=0.7,
                   help="Sampling temperature (default 0.7, as in LUQ paper)")
    p.add_argument("--max-tokens",     type=int,   default=1024,
                   help="Max tokens per generation (default 1024)")
    p.add_argument("--bedrock-region", default=os.environ.get(
                       "AWS_DEFAULT_REGION", _BEDROCK_DEFAULT_REGION),
                   help="AWS region for Bedrock (default us-east-1)")
    p.add_argument("--bedrock-model",  default=_BEDROCK_GEN_MODEL,
                   help=f"Bedrock model ID (default {_BEDROCK_GEN_MODEL})")
    p.add_argument("--nli-model",      default=_NLI_MODEL_NAME,
                   help=f"NLI cross-encoder (default {_NLI_MODEL_NAME})")
    p.add_argument("--out",            default=".",
                   help="Output root directory (default .)")
    p.add_argument("--top-n",          type=int,   default=3,
                   help="Number of samples to include in HTML report (default 3)")
    p.add_argument("--no-cache",       action="store_true",
                   help="Regenerate notes even if cached JSON exists")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"\n  Model          : {args.bedrock_model}")
    print(f"  Region         : {args.bedrock_region}")
    print(f"  Dataset        : {_DATASET_REPO} / {_DATASET_CONFIG} / {_DATASET_SPLIT}")
    print(f"  Samples        : {args.start} – {args.end - 1}")
    print(f"  K generations  : {args.K}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  NLI model      : {args.nli_model}")
    print(f"  Output dir     : {Path(args.out).resolve()}")
    print(f"  Cache          : {'disabled' if args.no_cache else 'enabled'}")
    print()

    run_luq_batch(
        start=args.start,
        end=args.end,
        K=args.K,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        bedrock_region=args.bedrock_region,
        bedrock_model=args.bedrock_model,
        nli_model_name=args.nli_model,
        out=Path(args.out),
        top_n=args.top_n,
        cache_generations=not args.no_cache,
    )

    print("\n" + "═" * 56)
    print(f"  Done. Results in: {(Path(args.out) / 'luq_out').resolve()}")
    print("═" * 56 + "\n")


if __name__ == "__main__":
    main()
