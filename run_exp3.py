"""
Experiment 3 — PDSQI-9 Blind Spot Detection via Activation Patching.

Pipeline:
  1. Load Experiment 1+2 outputs
  2. Auto-select candidate encounters for categories A, B, C
  3. For each candidate: build corrupted transcript (fact swap / negation flip)
  4. Run denoising activation patching (resid, attn_out, mlp_out sweeps)
  5. Classify each fact as retrieval-grounded or parametric
  6. Build blind spot report + taxonomy

Usage:
    python run_exp3.py [--exp1-dir results/exp1] [--exp2-dir results/exp2]
                      [--results-dir results] [--n-per-cat 7] [--device cuda]

IMPORTANT — manual step required:
  After candidate_scaffold.json is written, open it and fill in:
    - dialogue_corrupt  (the transcript with the fact changed)
    - target_fact       (human-readable description)
    - correct_token     (the single token the model should generate at the target pos)
    - wrong_token       (what it would generate from the corrupted transcript)
  Then re-run with --skip-scaffold to proceed directly to patching.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from config import (
    MODEL_NAME, DTYPE, SOAP_PROMPT_TEMPLATE, RESULTS_DIR,
)
from data import load_aci_examples, segment_soap
from patching import (
    PatchingCandidate,
    select_candidates,
    run_patching_candidate,
    build_blind_spot_report,
    corrupt_dosage_swap,
    corrupt_negation_flip,
)
from visualise import plot_patching_sweep, plot_blind_spot_summary


# ── Prompt builder (same as Experiment 1) ─────────────────────────────────────

def make_prompt_fn(tokenizer):
    def prompt_fn(dialogue: str) -> str:
        raw = SOAP_PROMPT_TEMPLATE.format(dialogue=dialogue)
        try:
            messages = [{"role": "user", "content": raw}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return raw
    return prompt_fn


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 3: Blind spot detection")
    p.add_argument("--exp1-dir",    type=str, default="results/exp1")
    p.add_argument("--exp2-dir",    type=str, default="results/exp2")
    p.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    p.add_argument("--n-per-cat",   type=int, default=7)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--skip-scaffold", action="store_true",
                   help="Scaffold JSON already filled in; proceed to patching")
    return p.parse_args()


def main():
    args     = parse_args()
    exp1_dir = Path(args.exp1_dir)
    exp2_dir = Path(args.exp2_dir)
    out_dir  = Path(args.results_dir) / "exp3"
    out_dir.mkdir(parents=True, exist_ok=True)

    scaffold_path = out_dir / "candidate_scaffold.json"

    # ── Step 1: Load Exp 1+2 outputs ─────────────────────────────────────────
    with open(exp1_dir / "encounter_features.json") as f:
        encounter_features = json.load(f)

    with open(exp1_dir / "example_meta.json") as f:
        example_meta = json.load(f)

    with open(exp2_dir / "pdsqi9_scores.json") as f:
        pdsqi9_scores = json.load(f)

    token_dfs = []
    for meta in example_meta:
        p = Path(meta["token_df_path"])
        token_dfs.append(pd.read_parquet(p) if p.exists() else pd.DataFrame())

    # ── Step 2: Load/build candidate scaffold ─────────────────────────────────
    if not args.skip_scaffold:
        print("Loading ACI-Bench examples ...")
        examples = load_aci_examples(n=len(example_meta) * 2)
        idx_to_ex = {ex.idx: ex for ex in examples}

        # Attach generated notes and sections to examples
        for meta in example_meta:
            ex = idx_to_ex.get(meta["idx"])
            if ex:
                ex.generated_note = meta["generated_note"]
                ex.soap_sections  = meta.get("soap_sections", {})

        ordered_examples = [
            idx_to_ex[meta["idx"]] for meta in example_meta
            if meta["idx"] in idx_to_ex
        ]

        candidates = select_candidates(
            ordered_examples, token_dfs, pdsqi9_scores,
            n_per_category=args.n_per_cat,
        )

        print(f"\nSelected {len(candidates)} candidates:")
        for c in candidates:
            print(f"  Cat {c.category}  enc={c.encounter_idx}: {c.description}")

        # Serialise scaffold for manual completion
        scaffold = []
        for c in candidates:
            d = asdict(c)
            # Add example dialogue for reference
            ex = idx_to_ex.get(c.encounter_idx)
            d["dialogue_preview"] = (ex.dialogue[:400] if ex else "")
            d["generated_note_preview"] = (
                ex.generated_note[:300] if ex else ""
            )
            scaffold.append(d)

        with open(scaffold_path, "w") as f:
            json.dump(scaffold, f, indent=2)

        print(f"\nCandidate scaffold written → {scaffold_path}")
        print(
            "\n>>> MANUAL STEP REQUIRED <<<\n"
            "Open results/exp3/candidate_scaffold.json and fill in:\n"
            "  - dialogue_corrupt  : transcript with the fact changed\n"
            "  - target_fact       : human description of the fact\n"
            "  - correct_token     : single token model should generate\n"
            "  - wrong_token       : token generated from corrupted transcript\n"
            "\nThen re-run:  python run_exp3.py --skip-scaffold\n"
        )
        return

    # ── Step 3: Load completed scaffold and run patching ─────────────────────
    with open(scaffold_path) as f:
        scaffold = json.load(f)

    candidates = []
    for d in scaffold:
        if not d.get("dialogue_corrupt") or not d.get("correct_token"):
            print(f"  Skipping enc={d['encounter_idx']} Cat {d['category']} "
                  f"— scaffold not filled in.")
            continue
        c = PatchingCandidate(
            encounter_idx    = d["encounter_idx"],
            category         = d["category"],
            description      = d["description"],
            dialogue_clean   = d["dialogue_clean"],
            dialogue_corrupt = d["dialogue_corrupt"],
            target_fact      = d["target_fact"],
            correct_token    = d["correct_token"],
            wrong_token      = d["wrong_token"],
        )
        candidates.append(c)

    if not candidates:
        print("No completed candidates found. Fill in the scaffold and re-run.")
        return

    print(f"\nLoading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        dtype=getattr(torch, DTYPE),
        default_padding_side="left",
        device=args.device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_fn = make_prompt_fn(tokenizer)

    # ── Step 4: Run patching sweeps ───────────────────────────────────────────
    patched_candidates = []
    for c in candidates:
        print(f"\n── Patching enc={c.encounter_idx} Cat {c.category} ──────────────")
        print(f"   {c.description}")
        try:
            c = run_patching_candidate(model, c, prompt_fn, verbose=True)
            patched_candidates.append(c)
            plot_patching_sweep(c, out_dir=out_dir)
        except Exception as e:
            print(f"   ERROR: {e}")

    # ── Step 5: Build report ──────────────────────────────────────────────────
    report = build_blind_spot_report(patched_candidates)

    report_path = out_dir / "blind_spot_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBlind spot report → {report_path}")

    # Summary
    print("\n── Blind Spot Taxonomy ───────────────────────────────────────────")
    for cat in ["A", "B", "C"]:
        cat_rows = [r for r in report if r["category"] == cat]
        if not cat_rows:
            continue
        n_blind = sum(1 for r in cat_rows if r["pdsqi9_blind_spot"])
        label = {
            "A": "Coincidental correctness (PDSQI-9 misses MLP-grounded facts)",
            "B": "Penalised inference (PDSQI-9 penalises retrieval-grounded synthesis)",
            "C": "Undetected fragility (high score but parametric generation)",
        }[cat]
        print(f"\n  {label}")
        print(f"  Confirmed blind spots: {n_blind}/{len(cat_rows)}")
        for r in cat_rows:
            print(
                f"    enc={r['encounter_idx']}  grounding={r['grounding']:<12} "
                f"attn_max={r['max_attn_restore']:.3f}  "
                f"mlp_max={r['max_mlp_restore']:.3f}  "
                f"blind_spot={r['pdsqi9_blind_spot']}"
            )

    # Frequency estimate across full dataset
    n_total = len(example_meta)
    for cat, rate in {
        "A": sum(1 for r in report if r["category"] == "A" and r["pdsqi9_blind_spot"]),
        "B": sum(1 for r in report if r["category"] == "B" and r["pdsqi9_blind_spot"]),
        "C": sum(1 for r in report if r["category"] == "C" and r["pdsqi9_blind_spot"]),
    }.items():
        cat_total = sum(1 for r in report if r["category"] == cat)
        if cat_total:
            freq = rate / cat_total
            print(f"\n  Estimated frequency of Cat {cat} blind spot: "
                  f"{freq:.0%} of encounters ({int(freq * n_total)}/{n_total})")

    plot_blind_spot_summary(report, out_dir=out_dir)
    print(f"\nSummary plot → {out_dir / 'blind_spot_summary.png'}")
    print("\n── Experiment 3 complete ─────────────────────────────────────────")


if __name__ == "__main__":
    main()
