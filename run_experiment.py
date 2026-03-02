"""
Main experiment runner: MTS Dialog Summarisation — Activation Patching

Usage:
    python run_experiment.py
    python run_experiment.py --n_examples 5 --show_plots
    python run_experiment.py --n_examples 1 --example_idx 42

The script:
  1. Loads N examples from har1/MTS_Dialogue-Clinical_Note (HuggingFace).
  2. Applies speaker-swap corruption to each example.
  3. Runs activation patching (resid_pre, attn_out, mlp_out, per-head, per-position).
  4. Saves per-example and aggregate figures + a JSON results file.
"""

import argparse
import json
import os
import sys
import torch

from transformer_lens import HookedTransformer

from config import (
    MODEL_NAME, DEVICE, DTYPE,
    N_EXAMPLES, RANDOM_SEED,
    ATTN_TARGET_LAYER, MLP_TARGET_LAYER,
    HEAD_SWEEP_LAYERS, RESULTS_DIR, CORRUPTION,
)
from data     import load_mts_examples, preview_example
from metrics  import find_answer_tokens, logit_diff, top_k_predictions
from patching import run_patching_for_example
from visualise import (
    plot_layer_comparison,
    plot_head_heatmap,
    plot_position_scores,
    plot_aggregate,
    plot_aggregate_head_heatmap,
)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MTS Dialog Activation Patching")
    p.add_argument("--n_examples",  type=int,  default=N_EXAMPLES,
                   help="Number of MTS examples to run (default: %(default)s)")
    p.add_argument("--seed",        type=int,  default=RANDOM_SEED)
    p.add_argument("--show_plots",  action="store_true",
                   help="Display plots interactively (default: save only)")
    p.add_argument("--skip_position_sweep", action="store_true",
                   help="Skip the per-token-position sweep (faster)")
    p.add_argument("--verbose",     action="store_true",
                   help="Print layer-by-layer scores during sweep")
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────

def load_model() -> HookedTransformer:
    print(f"\nLoading {MODEL_NAME} ...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        center_writing_weights=False,  # preserve Gemma 2 normalisation
        center_unembed=False,
        fold_ln=False,
        dtype=DTYPE,
        device=DEVICE,
    )
    model.eval()
    print(f"  Layers={model.cfg.n_layers}  Heads={model.cfg.n_heads}  "
          f"d_model={model.cfg.d_model}  device={DEVICE}\n")
    return model


# ── Serialise numpy arrays for JSON ───────────────────────────────────────

def _to_serialisable(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    return obj


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_model()

    # ── Load data ──────────────────────────────────────────────────────────
    examples = load_mts_examples(
        n=args.n_examples,
        seed=args.seed,
        corruption=CORRUPTION,
    )

    all_results    = []
    summary_rows   = []

    for ex_num, ex in enumerate(examples):
        print(f"\n{'='*65}")
        print(f"Example {ex_num+1}/{len(examples)}  (dataset idx={ex.idx})")
        print(f"{'='*65}")
        preview_example(ex, char_limit=300)

        # ── Tokenise ───────────────────────────────────────────────────────
        clean_tokens     = model.to_tokens(ex.clean_prompt,     prepend_bos=True)
        corrupted_tokens = model.to_tokens(ex.corrupted_prompt, prepend_bos=True)

        print(f"\n  Clean tokens    : {clean_tokens.shape[1]}")
        print(f"  Corrupted tokens: {corrupted_tokens.shape[1]}")

        # ── Find answer tokens from model predictions ──────────────────────
        print("\n  Finding answer tokens ...")
        with torch.no_grad():
            clean_logits     = model(clean_tokens)
            corrupted_logits = model(corrupted_tokens)

        print("\n  Top-5 clean predictions:")
        for tok, ld in top_k_predictions(model, clean_logits, k=5):
            print(f"    {tok!r:20s}  {ld:.3f}")

        print("\n  Top-5 corrupted predictions:")
        for tok, ld in top_k_predictions(model, corrupted_logits, k=5):
            print(f"    {tok!r:20s}  {ld:.3f}")

        answer_tokens = find_answer_tokens(model, clean_logits, corrupted_logits)

        # Skip examples where the clean/corrupted predictions barely differ
        gap = (
            logit_diff(clean_logits,     answer_tokens.correct_id, answer_tokens.wrong_id) -
            logit_diff(corrupted_logits, answer_tokens.correct_id, answer_tokens.wrong_id)
        )
        if gap < 0.5:
            print(f"\n  ⚠ Logit-diff gap={gap:.3f} is very small — "
                  f"corruption may not have changed predictions much. Skipping.")
            continue

        # ── Run patching suite ─────────────────────────────────────────────
        print("\n  Running patching experiments ...")
        result = run_patching_for_example(
            model            = model,
            clean_tokens     = clean_tokens,
            corrupted_tokens = corrupted_tokens,
            correct_id       = answer_tokens.correct_id,
            wrong_id         = answer_tokens.wrong_id,
            attn_target      = ATTN_TARGET_LAYER,
            mlp_target       = MLP_TARGET_LAYER,
            head_layers      = HEAD_SWEEP_LAYERS,
            verbose          = args.verbose,
        )

        # Skip position sweep if flagged
        if args.skip_position_sweep:
            result["attn_pos_scores"] = []
            result["mlp_pos_scores"]  = []
            result["attn_pos_tokens"] = []
            result["mlp_pos_tokens"]  = []

        result["example_idx"]       = ex.idx
        result["correct_answer"]    = answer_tokens.correct_str
        result["wrong_answer"]      = answer_tokens.wrong_str
        result["clean_prompt"]      = ex.clean_prompt
        result["corrupted_prompt"]  = ex.corrupted_prompt

        # ── Plot per-example figures ───────────────────────────────────────
        print("\n  Generating figures ...")
        plot_layer_comparison(result, ex.idx, show=args.show_plots)
        plot_head_heatmap(result,     ex.idx, show=args.show_plots)
        if not args.skip_position_sweep:
            plot_position_scores(result, ex.idx, show=args.show_plots)

        # ── Summary row ────────────────────────────────────────────────────
        attn_target_recovery = result["attn_scores"][ATTN_TARGET_LAYER]
        mlp_target_recovery  = result["mlp_scores"][MLP_TARGET_LAYER]
        best_attn_layer      = int(result["attn_scores"].argmax())
        best_mlp_layer       = int(result["mlp_scores"].argmax())

        summary_rows.append({
            "example_idx"         : ex.idx,
            "correct_answer"      : answer_tokens.correct_str,
            "wrong_answer"        : answer_tokens.wrong_str,
            "clean_ld"            : result["clean_ld"],
            "corrupted_ld"        : result["corrupted_ld"],
            f"attn_L{ATTN_TARGET_LAYER}_recovery"  : attn_target_recovery,
            f"mlp_L{MLP_TARGET_LAYER}_recovery"    : mlp_target_recovery,
            "best_attn_layer"     : best_attn_layer,
            "best_attn_recovery"  : float(result["attn_scores"].max()),
            "best_mlp_layer"      : best_mlp_layer,
            "best_mlp_recovery"   : float(result["mlp_scores"].max()),
        })

        print(f"\n  Layer {ATTN_TARGET_LAYER} Attn-Out recovery : {attn_target_recovery:+.4f}"
              f"  {'✓ COPYING' if attn_target_recovery > 0.3 else '○ weak'}")
        print(f"  Layer {MLP_TARGET_LAYER} MLP-Out recovery  : {mlp_target_recovery:+.4f}"
              f"  {'✓ CONSTRUCTION' if mlp_target_recovery > 0.3 else '○ weak'}")

        all_results.append(result)

    # ── Aggregate plots ────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print("Generating aggregate plots ...")
        plot_aggregate(all_results,           show=args.show_plots)
        plot_aggregate_head_heatmap(all_results, show=args.show_plots)

    # ── Save results JSON ──────────────────────────────────────────────────
    output = {
        "model"               : MODEL_NAME,
        "n_examples_run"      : len(all_results),
        "corruption"          : CORRUPTION,
        "attn_target_layer"   : ATTN_TARGET_LAYER,
        "mlp_target_layer"    : MLP_TARGET_LAYER,
        "head_sweep_layers"   : HEAD_SWEEP_LAYERS,
        "per_example_results" : [_to_serialisable(r) for r in all_results],
        "summary"             : summary_rows,
    }

    json_path = os.path.join(RESULTS_DIR, "patching_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── Print summary table ────────────────────────────────────────────────
    if summary_rows:
        print(f"\n{'─'*75}")
        print(f"{'Ex':>4}  {'Correct':>12}  {'Wrong':>12}  "
              f"{'Attn-L'+str(ATTN_TARGET_LAYER):>10}  {'MLP-L'+str(MLP_TARGET_LAYER):>9}  "
              f"{'BestAttn':>9}  {'BestMLP':>8}")
        print(f"{'─'*75}")
        for row in summary_rows:
            print(
                f"{row['example_idx']:>4}  "
                f"{row['correct_answer']:>12}  "
                f"{row['wrong_answer']:>12}  "
                f"{row[f'attn_L{ATTN_TARGET_LAYER}_recovery']:>10.4f}  "
                f"{row[f'mlp_L{MLP_TARGET_LAYER}_recovery']:>9.4f}  "
                f"L{row['best_attn_layer']:02d}={row['best_attn_recovery']:.3f}  "
                f"L{row['best_mlp_layer']:02d}={row['best_mlp_recovery']:.3f}"
            )
        print(f"{'─'*75}")


if __name__ == "__main__":
    main()
