"""
demo_gptoss_nli.py
===================
Checks the REAL production gpt-oss-20b path (atomic_luq._gptoss_yesno) against
a handful of real (premise, hypothesis) pairs pulled straight from sample_000
(aci/test2): prints the raw Converse response blocks (reasoningContent vs the
final text answer) alongside the parsed 1.0/0.0 that score_sample_gptoss
would actually record, using the same GPTOSS_MAX_TOKENS=300 cap and
reasoning_effort as production -- so a truncated-mid-reasoning response (no
final text block) would show up here before it shows up in a real run.

Includes the two failure cases the cross-encoder NLI model got wrong
(diagnosed earlier by hand):
  - "severe pain" anaphora dilution   (all notes agree, cross-encoder said no)
  - "Murphy's sign" clinical paraphrase (all notes agree, cross-encoder said no)
plus a clean entailed case, a clean contradicted case, and a genuinely-absent
case, for contrast.

Usage:
    python demo_gptoss_nli.py
    python demo_gptoss_nli.py --reasoning-effort medium
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import luq_sentence as luq
import atomic_luq as al
from llm_client import GPTOSS_MODEL_ID as MODEL_ID, GPTOSS_ANSWER_MAX_TOKENS as GPTOSS_MAX_TOKENS, \
    build_yesno_prompt, gptoss_yesno

GEN_PATH = Path("luq_out/llama/generations/aci/test2/sample_000_generations.json")


def load_cases():
    gen = json.loads(GEN_PATH.read_text())
    notes = gen["notes"]
    sections = [al.parse_sections(n) for n in notes]

    return [
        {
            "name": "clean_entailed",
            "premise": sections[1]["objective"],
            "hypothesis": "The patient's blood pressure is 128/88.",
            "expected": 1.0,
        },
        {
            "name": "clean_contradicted",
            "premise": sections[4]["subjective"],
            "hypothesis": "The patient is a 25-year-old male.",
            "expected": 0.0,
        },
        {
            "name": "genuinely_absent",
            "premise": sections[1]["objective"],
            "hypothesis": "The patient has a regular heart rhythm.",
            "expected": 0.0,
        },
        {
            "name": "anaphora_dilution (cross-encoder said NO, should be YES)",
            "premise": sections[3]["subjective"],
            "hypothesis": "The abdominal pain is severe.",
            "expected": 1.0,
        },
        {
            "name": "clinical_paraphrase (cross-encoder said NO, should be YES)",
            "premise": sections[1]["objective"],
            "hypothesis": "The patient's abdominal pain is consistent with Murphy's sign.",
            "expected": 1.0,
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    args = ap.parse_args()

    client = luq.get_bedrock_client()
    cases = load_cases()
    n_pass = 0

    for case in cases:
        prompt = build_yesno_prompt(case["premise"], case["hypothesis"])
        print("=" * 80)
        print(f"CASE: {case['name']}")
        print(f"HYPOTHESIS: {case['hypothesis']}")
        print("-" * 80)

        try:
            response = client.converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": GPTOSS_MAX_TOKENS, "temperature": 0.0},
                additionalModelRequestFields={"reasoning_effort": args.reasoning_effort},
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        print("RAW response['output']['message']['content'] blocks:")
        for i, block in enumerate(response["output"]["message"]["content"]):
            print(f"  [{i}] keys={list(block.keys())}")
            print(f"      {block}")

        usage = response.get("usage", {})
        stop_reason = response.get("stopReason")
        print(f"stopReason={stop_reason}  usage={usage}  (maxTokens cap={GPTOSS_MAX_TOKENS})")

        # Same extraction path production uses (atomic_luq._gptoss_yesno),
        # run again here so a truncated/unparseable response would surface
        # in this demo before it ever shows up in a real scoring run.
        parsed = gptoss_yesno(case["premise"], case["hypothesis"], args.reasoning_effort)
        ok = parsed == case["expected"]
        n_pass += ok
        print(f"PARSED (production path): {parsed}  expected={case['expected']}  "
              f"{'PASS' if ok else 'FAIL'}")
        print()

    print(f"{n_pass}/{len(cases)} cases matched expected entailment.")


if __name__ == "__main__":
    main()
