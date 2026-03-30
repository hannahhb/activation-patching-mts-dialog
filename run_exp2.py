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
)
from visualise import (
    plot_signature_matrix,
    plot_section_anova,
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
    p.add_argument("--zero-shot", action="store_true",
                   help="Use zero-shot prompting (no few-shot examples)")
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
        judge = PDSQI9Judge(model=args.judge_model, backend=args.judge_backend, zero_shot=args.zero_shot)
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
        if not anova_df.empty:
            print(anova_df.sort_values("eta_sq", ascending=False).head(8).to_string())
        else:
            print("  (no features had sufficient data for ANOVA)")

        anova_plot = plot_section_anova(anova_df, out_dir=out_dir)
        print(f"  ANOVA plot    → {anova_plot}")

        # Deviation vs each PDSQI-9 attribute
        from scipy import stats as _stats
        dev_rows = []
        for attr in PDSQI9_ATTRIBUTES:
            dev_df = section_mechanistic_deviations(valid_dfs, all_scores, accuracy_key=attr)
            if dev_df.empty:
                continue
            dev_df["pdsqi_attr"] = attr
            dev_path = out_dir / f"section_deviations_{attr}.csv"
            dev_df.to_csv(dev_path, index=False)
            mask = dev_df[attr].notna() & dev_df["deviation"].notna()
            if mask.sum() >= 5:
                r, p = _stats.pearsonr(dev_df.loc[mask, "deviation"], dev_df.loc[mask, attr])
                dev_rows.append({"attribute": attr, "r": round(r, 3), "p": round(p, 4), "n": int(mask.sum())})
                print(f"  Deviation ↔ {attr:<18}: r={r:+.3f}  p={p:.4f}  n={mask.sum()}")
            else:
                print(f"  Deviation ↔ {attr:<18}: insufficient data (n={mask.sum()})")

        if dev_rows:
            dev_summary = pd.DataFrame(dev_rows).set_index("attribute")
            dev_summary_path = out_dir / "section_deviations_summary.csv"
            dev_summary.to_csv(dev_summary_path)
            print(f"\n  Deviation summary → {dev_summary_path}")

    print("\n── Experiment 2 complete ─────────────────────────────────────────")
    print(f"All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
