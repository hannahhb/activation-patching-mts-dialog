"""
demo_fact_span_decompose.py
============================
Demo: decompose a SOAP note into atomic facts AND get each fact's span-level
origin (the exact substring of the note it was derived from) in a SINGLE LLM
call, instead of the two-stage pipeline (atomic_luq.py decomposition +
fact_sentence_match.py's NER/lemma-matching _locate_span heuristic).

Why single-call span attribution:
  fact_sentence_match.py's _locate_span() is a POST-HOC heuristic — it never
  sees the LLM's own reasoning about which words support which fact, so it
  can under/over-match (see the "back"/"pain" and "denied"/"deny" stopword
  bugs already fixed there). Asking the LLM to report its own source span
  at extraction time sidesteps that: it already knows which phrase it copied
  the fact from. The tradeoff is the LLM can hallucinate a plausible-looking
  span that ISN'T a verbatim substring — so every returned span is verified
  against the note text before being trusted (see verify_spans below), with
  the same whitespace/case-tolerant matcher already used elsewhere in this
  codebase (find_span_char_range).

Usage:
    python demo_fact_span_decompose.py
    python demo_fact_span_decompose.py --note-file path/to/note.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_llm
from redeep_word_plots import find_span_char_range  # whitespace/case-tolerant substring locator

BEDROCK_MODEL = "us.meta.llama3-3-70b-instruct-v1:0"

# ─────────────────────────────────────────────────────────────────────────────
# The prompt
# ─────────────────────────────────────────────────────────────────────────────
# Rules/voice reused verbatim from factmatch_sentence._NOTE_PROMPT (the
# production decomposition prompt) so facts extracted here stay consistent
# with the rest of the pipeline; only the output format changes (JSON with a
# "span" field per fact, replacing the flat " // "-delimited list).

_SYSTEM_PROMPT = (
    "You are a meticulous physician extracting atomic clinical facts from a "
    "SOAP note, with each fact's exact textual origin. Return ONLY valid "
    "JSON — no explanation, no preamble, no markdown code fences."
)

_USER_TEMPLATE = """\
Extract every clinical fact from the SOAP note below as a JSON array. For \
each fact, also give its "spans": a JSON list of one or more exact minimal \
substrings of the note (copy-pasted verbatim, not paraphrased) that \
together support the fact.

VERBATIM RULE (the rule most often broken — follow it exactly):
- Every string in "spans" MUST be an EXACT, character-for-character, \
case-preserving substring of the note below. Copy it, do not retype or \
improve it.
- Do NOT upgrade informal or colloquial wording into clinical phrasing for \
the span. If the source text says "seems to make it worse", the span must \
contain that exact wording — NOT a rewritten version like "exacerbated by".
    WRONG:  fact: "The patient's pain is exacerbated by walking."
            spans: ["exacerbated by walking"]        <- rewritten, not in the text
    RIGHT:  fact: "The patient's pain is exacerbated by walking."
            spans: ["walking seems to make it worse"] <- copied verbatim
- Before writing a span, check it word-for-word against the note below. If \
you cannot find an exact match, search again for the closest literal \
wording in the note and use that — never invent clinical-sounding text.

MULTIPLE-SPANS RULE:
- Almost every fact has exactly ONE span: "spans": ["<one string>"].
- A fact split off a shared local clause (e.g. "X and Y were normal" split \
into a fact about X and a fact about Y) still has ONE span — give BOTH \
facts the SAME full shared span. This is NOT a multi-span case: the \
evidence is local and contiguous, just shared between two split facts.
- Only use TWO OR MORE entries in "spans" when the fact genuinely requires \
evidence from separate, non-adjacent parts of the note (e.g. a diagnosis \
combining a symptom mentioned in one sentence with a lab value mentioned \
several sentences later). List each supporting substring separately.

Other rules:
- Split conjunctions: "denied X and Y" -> two separate facts, one for X and \
one for Y.
- Split medications: drug name, dose, frequency, indication as separate \
facts (repeat the drug name in each). Each fact's span is the specific \
phrase that supports IT alone, not the whole medication sentence (e.g. the \
dose fact's span is "10 mg", not the full sentence).
- Include all clinical findings, diagnoses, plans, and demographics.

Return this JSON structure and nothing else:
[
  {{"fact": "<atomic fact as a full sentence>", "spans": ["<exact verbatim substring>", ...]}},
  ...
]

Example 1 — single local span shared across a conjunction split:
Note: "Chest X-ray and pulmonary function test were normal."
Output:
[
  {{"fact": "The patient's chest X-ray is normal.", "spans": ["Chest X-ray and pulmonary function test were normal"]}},
  {{"fact": "The patient's pulmonary function test is normal.", "spans": ["Chest X-ray and pulmonary function test were normal"]}}
]

Example 2 — verbatim, not paraphrased (keep the source's own words):
Note: "The patient reports standing and walking seems to make the pain worse; coughing and sneezing make it worse too."
Output:
[
  {{"fact": "The patient's pain is exacerbated by standing.", "spans": ["standing and walking seems to make the pain worse"]}},
  {{"fact": "The patient's pain is exacerbated by walking.", "spans": ["standing and walking seems to make the pain worse"]}},
  {{"fact": "The patient's pain is exacerbated by coughing.", "spans": ["coughing and sneezing make it worse"]}},
  {{"fact": "The patient's pain is exacerbated by sneezing.", "spans": ["coughing and sneezing make it worse"]}}
]

Example 3 — genuinely disjoint spans (two separate, non-adjacent parts):
Note: "HPI: The patient reports chest pain radiating to the left arm. ... Assessment: Troponin returned at 0.8, consistent with myocardial infarction."
Output:
[
  {{"fact": "The patient has chest pain radiating to the left arm.", "spans": ["chest pain radiating to the left arm"]}},
  {{"fact": "The patient's troponin is elevated at 0.8, consistent with myocardial infarction.", "spans": ["Troponin returned at 0.8", "consistent with myocardial infarction"]}}
]

Clinical note:
\"\"\"
{note}
\"\"\"

Output:\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Call + parse + verify
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_array(raw: str) -> Optional[List[dict]]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _extract_spans(item: dict) -> List[str]:
    """Read the "spans" list, with a defensive fallback to a legacy/singular
    "span" string key in case the model doesn't follow the schema exactly."""
    spans = item.get("spans")
    if isinstance(spans, list):
        return [str(s).strip() for s in spans if str(s).strip()]
    single = item.get("span")
    return [str(single).strip()] if single and str(single).strip() else []


def decompose_with_spans(note: str, model_id: str = BEDROCK_MODEL) -> List[Dict]:
    """
    Returns a list of dicts, one per fact:
      fact, spans (list of verbatim strings as the LLM returned them),
      span_offsets (list of (start, end) or None per entry in `spans`,
        None where that particular span isn't a real substring of `note`),
      span_verified (bool — True only if EVERY span in `spans` was located;
        False means at least one is not a real substring and the fact's
        provenance should not be fully trusted for highlighting),
      synthesized (bool, computed as len(spans) > 1 — NOT self-reported by
        the LLM, since that field turned out to be unreliable; see the
        --decomp-mode span design notes for why we stopped trusting it).
    """
    user_msg = _USER_TEMPLATE.format(note=note.strip())
    resp = get_llm().converse(
        stage="fact_span_demo",
        model_id=model_id,
        system=[{"text": _SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_msg}]}],
        inference_config={"maxTokens": 2048, "temperature": 0.0},
    )
    raw = resp["output"]["message"]["content"][0]["text"]
    parsed = _parse_json_array(raw)
    if parsed is None:
        print("[warn] Could not parse JSON from model output:")
        print(raw)
        return []

    results = []
    for item in parsed:
        fact = str(item.get("fact", "")).strip()
        spans = _extract_spans(item)
        if not fact or not spans:
            continue
        offsets = [find_span_char_range(note, s) for s in spans]
        results.append({
            "fact": fact,
            "spans": spans,
            "span_offsets": offsets,
            "span_verified": all(o is not None for o in offsets),
            "synthesized": len(spans) > 1,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_NOTE = """\
Subjective:

HPI: The patient presented with a 2-hour history of low back pain and tingling sensation in the right leg, which started after raking leaves in the yard. The pain was exacerbated by standing and improved with rest and a hot shower. The patient had a similar episode 3 years ago that resolved on its own.

Past medical history: The patient has a history of similar low back pain 3 years ago, which resolved on its own. No other relevant medical history was mentioned.

Review of systems: The patient reports tingling sensation in the right leg.

Current medications: None mentioned.

Objective:

Vital signs: [NOT MENTIONED]

Physical exam: The patient had pain on flexion, palpation at the L5 level, and normal dorsiflexion. The pulses were equal in all extremities.

Test Results: X-ray of the low back showed normal results.

Assessment / Problem List:

Assessment: The patient likely sprained their low back, causing the current symptoms.

Problem list: Low back sprain; Tingling sensation in the right leg; Back pain.

Plan:

I prescribed meloxicam 15 mg orally daily for pain management. I also prescribed physical therapy to help with stretches and exercises to improve the patient's condition. I instructed the patient to rest and follow up with me in two weeks if the symptoms do not improve, at which point we will consider an MRI.

Follow-up: The patient will follow up with me in two weeks to reassess the condition and consider further testing, including an MRI if necessary.\
"""  # real generated note: aci/test1, sample_001, note_02


def main():
    parser = argparse.ArgumentParser(description="Demo: fact decomposition with span-level origin")
    parser.add_argument("--note-file", default=None,
                        help="Path to a .txt file with the note (default: built-in real example)")
    args = parser.parse_args()

    note = Path(args.note_file).read_text() if args.note_file else _DEMO_NOTE

    print(f"[demo] Decomposing note ({len(note)} chars) via {BEDROCK_MODEL} ...\n")
    results = decompose_with_spans(note)

    n_verified = sum(r["span_verified"] for r in results)
    n_synth = sum(r["synthesized"] for r in results)
    print(f"[demo] {len(results)} facts extracted, {n_verified} fully span-verified, "
         f"{len(results) - n_verified} with at least one unverified span, "
         f"{n_synth} genuinely disjoint (multi-span)\n")

    for i, r in enumerate(results, 1):
        syn = "  [disjoint/synthesized]" if r["synthesized"] else ""
        print(f"{i:>2}. {r['fact']}{syn}")
        for span, offset in zip(r["spans"], r["span_offsets"]):
            flag = f"chars [{offset[0]}:{offset[1]}]" if offset else "[UNVERIFIED — not found in note]"
            print(f"    span: {span!r}  {flag}")
        print()


if __name__ == "__main__":
    main()
