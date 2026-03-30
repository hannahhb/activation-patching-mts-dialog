"""
Experiment 2 — PDSQI-9 Scoring and Correlation with Mechanistic Profiles.

Pipeline:
  1. Load Experiment 1 outputs (encounter features + per-token parquets)
  2. Score each generated note with PDSQI-9 via GPT-4o (full note + per-section)
  3. Analysis A: Build Mechanistic Signature Matrix (Pearson r heatmap)
  4. Analysis B: Section-level ANOVA + mechanistic deviation → accuracy correlation
  5. Analysis C: Input complexity interaction (two-way ANOVA)
  6. Save all results as CSV / JSON / PNG

Usage:
    python run_exp2.py [--exp1-dir results/exp1] [--results-dir results]
                      [--judge-model gpt-4o]
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    PDSQI9_ATTRIBUTES, RESULTS_DIR, JUDGE_MODEL,
)
from pdsqi9_judge import PDSQI9Judge
from analysis import (
    build_signature_matrix,
    signature_matrix_pvalues,
    section_anova,
    section_mechanistic_deviations,
    complexity_interaction_anova,
    complexity_correlation_table,
)
from visualise import (
    plot_signature_matrix,
    plot_section_anova,
    plot_complexity_scatter,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_exp1_outputs(exp1_dir: Path):
    """Load encounter features, per-token DataFrames, and example metadata."""
    feat_path = exp1_dir / "encounter_features.json"
    meta_path = exp1_dir / "example_meta.json"

    with open(feat_path) as f:
        encounter_features = json.load(f)

    with open(meta_path) as f:
        example_meta = json.load(f)

    token_dfs = []
    for meta in example_meta:
        p = Path(meta["token_df_path"])
        if p.exists():
            token_dfs.append(pd.read_parquet(p))
        else:
            token_dfs.append(pd.DataFrame())

    return encounter_features, example_meta, token_dfs


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2: PDSQI-9 scoring + correlation")
    p.add_argument("--exp1-dir",      type=str, default="results/exp1")
    p.add_argument("--results-dir",   type=str, default=RESULTS_DIR)
    p.add_argument("--judge-model",   type=str, default=JUDGE_MODEL,
                   help="Model identifier (meaning depends on --judge-backend)")
    p.add_argument("--judge-backend", type=str, default="openai",
                   choices=["openai", "hf", "bedrock"],
                   help=(
                       "Inference backend for the PDSQI-9 judge. "
                       "openai: OpenAI API (needs OPENAI_API_KEY); "
                       "hf: HuggingFace Inference API (needs HF_API_KEY); "
                       "bedrock: AWS Bedrock (needs AWS credentials)"
                   ))
    p.add_argument("--skip-scoring", action="store_true",
                   help="Skip API scoring; load cached scores only")
    return p.parse_args()


def main():
    args    = parse_args()
    exp1_dir = Path(args.exp1_dir)
    out_dir  = Path(args.results_dir) / "exp2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Experiment 1 outputs from {exp1_dir} ...")
    encounter_features, example_meta, token_dfs = load_exp1_outputs(exp1_dir)
    print(f"  {len(example_meta)} encounters loaded.")

    # ── Step 2: PDSQI-9 scoring ───────────────────────────────────────────────
    scores_path = out_dir / "pdsqi9_scores.json"

    if args.skip_scoring and scores_path.exists():
        print(f"Loading cached PDSQI-9 scores from {scores_path}")
        with open(scores_path) as f:
            all_scores = json.load(f)
    else:
        judge = PDSQI9Judge(model=args.judge_model, backend=args.judge_backend)
        all_scores = []

        for i, meta in enumerate(example_meta):
            print(f"\n── Scoring encounter {meta['idx']} ({i+1}/{len(example_meta)}) ──")
            note     = meta["generated_note"]
            dialogue = ""  # reload from ACI-Bench if needed — stored in meta if possible
            sections = meta.get("soap_sections", {})

            # Load dialogue from ACI-Bench if not in meta
            if not dialogue:
                try:
                    from data import load_aci_examples
                    # cache: only load once
                    if not hasattr(main, "_aci_cache"):
                        main._aci_cache = {
                            ex.idx: ex.dialogue
                            for ex in load_aci_examples(n=len(example_meta) * 2)
                        }
                    dialogue = main._aci_cache.get(meta["idx"], "")
                except Exception:
                    dialogue = ""

            result = judge.score_note(
                dialogue=dialogue,
                note=note,
                sections=sections,
            )
            result["encounter_idx"] = meta["idx"]
            all_scores.append(result)

        with open(scores_path, "w") as f:
            json.dump(all_scores, f, indent=2)
        print(f"\nPDSQI-9 scores saved → {scores_path}")

    # ── Flatten full-note scores for correlation analysis ─────────────────────
    flat_scores = [
        {attr: s["full"].get(attr) for attr in PDSQI9_ATTRIBUTES}
        for s in all_scores
    ]

    # ── Analysis A: Mechanistic Signature Matrix ──────────────────────────────
    print("\n── Analysis A: Mechanistic Signature Matrix ──────────────────────")
    corr_df = build_signature_matrix(encounter_features, flat_scores)
    pval_df = signature_matrix_pvalues(encounter_features, flat_scores)

    corr_path = out_dir / "signature_matrix_corr.csv"
    pval_path = out_dir / "signature_matrix_pval.csv"
    corr_df.to_csv(corr_path)
    pval_df.to_csv(pval_path)
    print(f"  Correlation matrix → {corr_path}")
    print(f"  P-value matrix     → {pval_path}")

    plot_path = plot_signature_matrix(corr_df, pval_df, out_dir=out_dir)
    print(f"  Heatmap            → {plot_path}")

    # Print top correlations
    melted = corr_df.stack().dropna()
    melted = melted[melted.abs() > 0.3].sort_values(key=abs, ascending=False)
    if not melted.empty:
        print("\n  Top correlations (|r| > 0.3):")
        for (attr, feat), r in melted.head(10).items():
            p = pval_df.loc[attr, feat] if attr in pval_df.index and feat in pval_df.columns else "?"
            print(f"    {attr:<18} × {feat:<30}  r={r:+.3f}  p={p}")

    # ── Analysis B: Section ANOVA ─────────────────────────────────────────────
    print("\n── Analysis B: Section-level ANOVA ──────────────────────────────")
    valid_dfs = [df for df in token_dfs if not df.empty]
    if valid_dfs:
        anova_df = section_anova(valid_dfs)
        anova_path = out_dir / "section_anova.csv"
        anova_df.to_csv(anova_path)
        print(f"  ANOVA results → {anova_path}")
        print(anova_df.sort_values("eta_sq", ascending=False).head(8).to_string())

        anova_plot = plot_section_anova(anova_df, out_dir=out_dir)
        print(f"  ANOVA plot    → {anova_plot}")

        # Deviation vs accuracy
        dev_df = section_mechanistic_deviations(valid_dfs, all_scores)
        if not dev_df.empty:
            dev_path = out_dir / "section_deviations.csv"
            dev_df.to_csv(dev_path, index=False)
            print(f"  Section deviations → {dev_path}")
            from scipy import stats
            mask = dev_df["accurate"].notna() & dev_df["deviation"].notna()
            if mask.sum() >= 5:
                r, p = stats.pearsonr(dev_df.loc[mask, "deviation"],
                                      dev_df.loc[mask, "accurate"])
                print(f"  Deviation ↔ accuracy: r={r:+.3f}, p={p:.4f}")

    # ── Analysis C: Complexity interaction ────────────────────────────────────
    print("\n── Analysis C: Input Complexity Interaction ──────────────────────")
    anova_result = complexity_interaction_anova(encounter_features, all_scores)
    anova_c_path = out_dir / "complexity_anova.json"
    with open(anova_c_path, "w") as f:
        json.dump(anova_result, f, indent=2)
    print(f"  Two-way ANOVA → {anova_c_path}")
    print(json.dumps(anova_result, indent=4))

    corr_table = complexity_correlation_table(encounter_features, flat_scores)
    corr_table_path = out_dir / "complexity_correlations.csv"
    corr_table.to_csv(corr_table_path)
    print(f"  Complexity correlation table → {corr_table_path}")

    for x_key in ["entity_density", "source_len_tokens"]:
        for y_key in ["mlp_contribution_mid", "lookback_ratio"]:
            plot_complexity_scatter(
                encounter_features, flat_scores,
                x_key=x_key, y_key=y_key,
                out_dir=out_dir,
            )

    print("\n── Experiment 2 complete ─────────────────────────────────────────")
    print(f"All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
