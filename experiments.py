"""
experiments.py
==============
All experiment runner functions for ECS/PKS-based factuality analysis.

Experiments
-----------
2a  Token-level ECS/PKS mapping on the gold reference SOAP note.
2b  Model-generated note: scatter + HTML report + KDE comparison.
2c  Hallucination screening via synthetic injection.
2d  Direct Logit Attribution (DLA) per layer.
3   REDEEP hallucination scoring (logistic α/β on sets F and A).
4   Layer-wise discriminability statistics across N examples.

Imports from: config, tokenization, metrics, plots, halluc_llm (optional).
"""

import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformer_lens import HookedTransformer
from exp_helpers import *

from config import Config, Q_COLORS, load_aci_sample
from tokenization import generate_note, tokenize_as_generated, tokenize_pair
from metrics import (
    apply_layer_thresholds,
    calibrate_layer_thresholds,
    compute_dla,
    compute_ecs_pks,
    dla_discriminability,
    fit_hallucination_regressor,
    hallucination_risk,
    identify_copy_head_layers,
    identify_knowledge_ffns,
    layer_discriminability,
    quadrant_stats,
)
from plots import (
    plot_heatmap,
    plot_layer_discriminability,
    plot_risk_bar,
    plot_scatter,
)

warnings.filterwarnings("ignore")

# Optional LLM-based hallucination injector (requires huggingface_hub or boto3)
try:
    from halluc_llm import (
        HallucinationGenerationError,
        halluc_token_indices,
        inject_hallucinations,
        inject_hallucinations_llm,
    )
    _LLM_HALLUC_AVAILABLE = True
except ImportError:
    _LLM_HALLUC_AVAILABLE = False


# ─────────────────────────────────────────────
# Experiment 2b helpers
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 9. Experiment 2a
# ─────────────────────────────────────────────

def run_experiment_2a(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    transcript: str,
    gold_note: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Experiment 2a: ECS/PKS token-level mapping for the gold reference SOAP note.

    Outputs
    -------
    exp2a_scatter.png  — quadrant scatter
    exp2a_heatmap.png  — ECS/PKS heatmap across token positions
    exp2a_risk.png     — per-token hallucination risk bar chart
    """
    print("\n" + "═"*54)
    print("  EXPERIMENT 2a — ECS/PKS Token-Level Mapping")
    print("═"*54)

    tokens, t_len, note_toks = tokenize_pair(model, transcript, gold_note)
    tokens = tokens.to(cfg.device)
    print(f"  Transcript : {t_len} tokens")
    print(f"  Note       : {len(note_toks)} tokens")

    print("  Running forward pass + computing ECS / PKS …")
    ecs, pks, _, _ = compute_ecs_pks(model, tokens, t_len, cfg)
    stats          = quadrant_stats(ecs, pks, f"2a — Gold Note (ACI-Bench sample {cfg.sample_idx})")

    fig, ax = plt.subplots(figsize=(9, 8))
    plot_scatter(ecs, pks, note_toks,
                 f"Exp 2a — ECS vs PKS: Clinical Note Tokens\n({cfg.model_name})",
                 ax=ax, annotate_stride=4)
    plt.tight_layout()
    fig.savefig(out / "exp2a_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2a_scatter.png")

    fig, ax = plt.subplots(figsize=(max(16, len(note_toks) // 2), 3))
    plot_heatmap(ecs, pks, note_toks, "Exp 2a — ECS/PKS Heatmap across Note Tokens", ax=ax)
    plt.tight_layout()
    fig.savefig(out / "exp2a_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2a_heatmap.png")

    fig, ax = plt.subplots(figsize=(max(16, len(note_toks) // 2), 3))
    plot_risk_bar(ecs, pks, note_toks,
                  "Exp 2a — Per-Token Hallucination Risk (Low ECS + Low PKS)", ax=ax)
    plt.tight_layout()
    fig.savefig(out / "exp2a_risk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2a_risk.png")

    return ecs, pks, note_toks, stats


# ─────────────────────────────────────────────
# 10. Experiment 2b
# ─────────────────────────────────────────────

def run_experiment_2b(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    transcript: str,
    gold_note: str,
) -> Dict:
    """
    Experiment 2b: ECS/PKS scatter + HTML risk report for the model-generated
    note, with distributional comparison against the gold reference.

    We never subtract scores at matched token positions — the notes may be
    structured differently while both being clinically valid.  Instead we
    compare score *distributions* (KDE), which requires no alignment.

    Outputs
    -------
    exp2b_scatter_gen.png       — ECS/PKS scatter for the generated note
    exp2b_highlighted_note.html — scatter + note with risk-highlighted tokens
    exp2b_distributions.png     — KDE: gold vs generated ECS and PKS
    exp2b_generated_note.txt    — raw generated note text
    """
    print("\n" + "═"*54)
    print("  EXPERIMENT 2b — Generated Note: Scatter + Risk Highlights")
    print("═"*54)

    generated_note = generate_note(model, transcript, cfg)
    (out / "exp2b_generated_note.txt").write_text(generated_note, encoding="utf-8")
    print("  Saved → exp2b_generated_note.txt")

    tok_gen,  tl_gen,  nt_gen  = tokenize_pair(model, transcript, generated_note)
    tok_gold, tl_gold, nt_gold = tokenize_pair(model, transcript, gold_note)
    tok_gen  = tok_gen.to(cfg.device)
    tok_gold = tok_gold.to(cfg.device)

    print(f"  Generated note : {len(nt_gen)} tokens")
    print(f"  Gold note      : {len(nt_gold)} tokens  (distribution baseline only)")

    print("  Computing ECS/PKS for generated note …")
    ecs_gen, pks_gen, _, _   = compute_ecs_pks(model, tok_gen,  tl_gen,  cfg)
    print("  Computing ECS/PKS for gold note …")
    ecs_gold, pks_gold, _, _ = compute_ecs_pks(model, tok_gold, tl_gold, cfg)

    stats_gen  = quadrant_stats(ecs_gen,  pks_gen,  f"2b — Generated Note (sample {cfg.sample_idx})")
    stats_gold = quadrant_stats(ecs_gold, pks_gold, f"2b — Gold Note      (sample {cfg.sample_idx})")

    print(f"\n  Distributional delta (Generated − Gold):")
    print(f"  {'─'*42}")
    for k in ["mean_ecs", "mean_pks", "mean_risk",
              "extractive_frac", "parametric_frac",
              "synthesized_frac", "hallucinatory_frac"]:
        print(f"  {k:<30} {stats_gen[k] - stats_gold[k]:>+8.4f}")

    fig, ax = plt.subplots(figsize=(9, 8))
    plot_scatter(ecs_gen, pks_gen, nt_gen,
                 f"Exp 2b — Model-Generated Note: ECS vs PKS\n({cfg.model_name})",
                 ax=ax, annotate_stride=5)
    plt.tight_layout()
    scatter_path = out / "exp2b_scatter_gen.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2b_scatter_gen.png")

    build_html_report(
        scatter_png=scatter_path,
        tokens=nt_gen,
        ecs=ecs_gen,
        pks=pks_gen,
        generated_note=generated_note,
        model_name=cfg.model_name,
        sample_idx=cfg.sample_idx,
        out_path=out / "exp2b_highlighted_note.html",
    )
    print("  Saved → exp2b_highlighted_note.html")

    plot_distribution_comparison(
        ecs_gold, pks_gold, ecs_gen, pks_gen,
        out / "exp2b_distributions.png",
        cfg.model_name,
    )
    print("  Saved → exp2b_distributions.png")

    return {
        "ecs_gen":  ecs_gen,  "pks_gen":  pks_gen,
        "ecs_gold": ecs_gold, "pks_gold": pks_gold,
        "stats_gen": stats_gen, "stats_gold": stats_gold,
        "generated_note": generated_note,
    }


# ─────────────────────────────────────────────
# 11. Experiment 2c — hallucination validation
# ─────────────────────────────────────────────



def run_experiment_2c(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    transcript: str,
    gold_note: str,
    inject_fn=None,
) -> Dict:
    """
    Experiment 2c: ECS/PKS hallucination screening validation via synthetic
    injection into the gold reference note.

    Procedure
    ---------
    1. Inject up to 3 plausible-but-wrong substitutions (wrong drug, dosage,
       diagnosis, vital) into the gold note, recording exact char spans.
    2. Compute ECS and PKS for every token of the corrupted note against the
       original transcript.
    3. Map injection spans to token indices (ground-truth hallucination labels).
    4. Report quadrant breakdown: injected vs. clean tokens.
    5. Plot scatter with injected tokens starred and labelled.

    Expected finding: injected tokens cluster in Low-ECS + Low-PKS because they
    are neither grounded in the transcript nor supported by in-context parametric
    knowledge.  This validates ECS/PKS as a screening signal.

    Outputs
    -------
    exp2c_scatter.png         — quadrant scatter; ★ = injected tokens
    exp2c_tokens.csv          — per-token ECS, PKS, risk, quadrant, injected flag
    exp2c_corrupted_note.txt  — the hallucinated note (for inspection)
    """
    print("\n" + "═"*54)
    print("  EXPERIMENT 2c — Hallucination Screening Validation")
    print("═"*54)

    _inject = inject_fn or inject_hallucinations
    corrupted_note, injections = _inject(
        gold_note, max_injections=3, seed=cfg.sample_idx + 42
    )

    print(f"\n  Injected {len(injections)} hallucination(s):")
    for inj in injections:
        orig = f"'{inj['original']}'" if inj["original"] else "(nothing — appended)"
        print(f"    [{inj['category']:22s}]  {orig}  →  '{inj['replacement']}'")

    (out / "exp2c_corrupted_note.txt").write_text(corrupted_note, encoding="utf-8")
    print("  Saved → exp2c_corrupted_note.txt")

    tokens, t_len, note_toks = tokenize_as_generated(model, transcript, corrupted_note)
    tokens = tokens.to(cfg.device)
    print(f"\n  Prompt (instruction + transcript) : {t_len} tokens")
    print(f"  Note                              : {len(note_toks)} tokens")

    halluc_idx = halluc_token_indices(model.tokenizer, corrupted_note, injections)
    print(f"\n  Hallucinated token positions ({len(halluc_idx)} tokens):")
    for i in halluc_idx:
        label = note_toks[i].replace("▁", "").replace("Ġ", "").strip() if i < len(note_toks) else "?"
        print(f"    [{i:4d}] '{label}'")

    print("\n  Computing ECS/PKS for corrupted note …")
    ecs, pks, ecs_layers, pks_layers = compute_ecs_pks(model, tokens, t_len, cfg)

    em = float(np.median(ecs))
    pm = float(np.median(pks))

    def _quadrant(e: float, p: float) -> str:
        if   e >= em and p <  pm: return "extractive"
        elif e <  em and p >= pm: return "parametric"
        elif e >= em and p >= pm: return "synthesized"
        else:                      return "hallucinatory"

    risk      = hallucination_risk(ecs, pks)
    is_halluc = [i in set(halluc_idx) for i in range(len(note_toks))]

    rows = [
        {
            "token_idx": i,
            "token":     t.replace("▁", "").replace("Ġ", "").strip(),
            "ecs":       round(float(e), 4),
            "pks":       round(float(p), 4),
            "risk":      round(float(r), 4),
            "injected":  int(h),
            "quadrant":  _quadrant(e, p),
        }
        for i, (t, e, p, r, h) in enumerate(zip(note_toks, ecs, pks, risk, is_halluc))
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp2c_tokens.csv", index=False)
    print("  Saved → exp2c_tokens.csv")

    df_h = df[df["injected"] == 1]
    df_c = df[df["injected"] == 0]
    n_h  = max(len(df_h), 1)
    n_c  = max(len(df_c), 1)

    print(f"\n  ── Quadrant breakdown  (thresholds: ECS≥{em:.3f}, PKS≥{pm:.3f}) ──")
    print(f"  {'quadrant':<18}  {'injected':>14}  {'clean':>14}")
    print("  " + "─" * 50)
    for q in ["hallucinatory", "extractive", "parametric", "synthesized"]:
        ni     = int((df_h["quadrant"] == q).sum()) if len(df_h) else 0
        nc     = int((df_c["quadrant"] == q).sum())
        marker = "  ◄ expected" if q == "hallucinatory" else ""
        print(f"  {q:<18}  {ni:>4} / {len(df_h):>3} ({ni/n_h:>4.0%})  "
              f"{nc:>4} / {len(df_c):>3} ({nc/n_c:>4.0%}){marker}")

    if len(df_h):
        print(f"\n  ── Injected token scores ──")
        print(f"  {'token':<18}  {'ECS':>6}  {'PKS':>6}  {'risk':>6}  {'quadrant'}")
        print("  " + "─" * 60)
        for _, row in df_h.iterrows():
            print(f"  {row['token']:<18}  {row['ecs']:>6.3f}  "
                  f"{row['pks']:>6.3f}  {row['risk']:>6.3f}  {row['quadrant']}")

    fig, ax = plt.subplots(figsize=(9, 8))
    plot_scatter(
        ecs, pks, note_toks,
        (f"Exp 2c — Gold Note + Injected Hallucinations\n"
         f"({cfg.model_name})   ★ = injected hallucination tokens"),
        ax=ax, highlight=halluc_idx, annotate_stride=5,
    )

    for i in halluc_idx:
        if i >= len(note_toks):
            continue
        label = note_toks[i].replace("▁", "").replace("Ġ", "").strip()[:16]
        ax.annotate(label, (ecs[i], pks[i]),
                    fontsize=7, color="#B71C1C", fontweight="bold",
                    xytext=(8, 8), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#B71C1C", lw=0.8))

    if len(df_h):
        n_low_low = int((df_h["quadrant"] == "hallucinatory").sum())
        ax.text(
            0.02, 0.98,
            f"Injected tokens\nin Low-ECS + Low-PKS:\n"
            f"{n_low_low} / {len(df_h)} = {n_low_low/len(df_h):.0%}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                      edgecolor=Q_COLORS["hallucinatory"], linewidth=1.5),
        )

    plt.tight_layout()
    fig.savefig(out / "exp2c_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2c_scatter.png")

    # ── Layer-wise discriminability ───────────────────────────────────────────
    halluc_mask = np.array(is_halluc, dtype=bool)
    disc = layer_discriminability(pks_layers, ecs_layers, halluc_mask)

    if disc is not None:
        n_layers = pks_layers.shape[0]

        # Print summary table
        best_pks_layer = int(np.argmax(np.abs(disc["pks_cohens_d"])))
        best_ecs_layer = int(np.argmax(np.abs(disc["ecs_cohens_d"])))
        print(f"\n  ── Layer-wise discriminability  ({n_layers} layers) ──")
        print(f"  {'layer':>6}  {'PKS AUROC':>10}  {'PKS d':>8}  {'ECS AUROC':>10}  {'ECS d':>8}")
        print("  " + "─" * 50)
        for l in range(n_layers):
            marker = ""
            if l == best_pks_layer: marker += "  ← best PKS"
            if l == best_ecs_layer: marker += "  ← best ECS"
            print(f"  {l:>6}  {disc['pks_auroc'][l]:>10.4f}  {disc['pks_cohens_d'][l]:>8.3f}"
                  f"  {disc['ecs_auroc'][l]:>10.4f}  {disc['ecs_cohens_d'][l]:>8.3f}{marker}")

        # Save CSV (includes Pearson r from updated layer_discriminability)
        disc_df = pd.DataFrame({
            "layer":         np.arange(n_layers),
            "pks_auroc":     disc["pks_auroc"],
            "pks_cohens_d":  disc["pks_cohens_d"],
            "pks_pearson_r": disc["pks_pearson_r"],
            "ecs_auroc":     disc["ecs_auroc"],
            "ecs_cohens_d":  disc["ecs_cohens_d"],
            "ecs_pearson_r": disc["ecs_pearson_r"],
        })
        disc_df.to_csv(out / "exp2c_layer_discriminability.csv", index=False)
        print("  Saved → exp2c_layer_discriminability.csv")

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(max(10, n_layers // 2), 8),
                                 sharex=True)
        fig.suptitle(
            f"Exp 2c — Layer-wise Discriminability  ({cfg.model_name})\n"
            f"Hallucinated ({halluc_mask.sum()}) vs. clean ({(~halluc_mask).sum()}) tokens",
            fontsize=11, fontweight="bold",
        )
        plot_layer_discriminability(disc, title="", axes=(axes[0], axes[1]))
        plt.tight_layout()
        fig.savefig(out / "exp2c_layer_discriminability.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved → exp2c_layer_discriminability.png")
    else:
        print("  [2c] Skipping layer discriminability — need ≥1 token in each class.")

    return {
        "ecs": ecs, "pks": pks,
        "ecs_layers": ecs_layers, "pks_layers": pks_layers,
        "halluc_idx": halluc_idx, "injections": injections, "df": df,
        "disc": disc,
    }


# ─────────────────────────────────────────────
# 12. Experiment 2d — Direct Logit Attribution
# ─────────────────────────────────────────────

def run_experiment_2d(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    transcript: str,
    gold_note: str,
    inject_fn=None,
) -> Dict:
    """
    Experiment 2d: Direct Logit Attribution (DLA) — attention vs MLP contribution
    for hallucinated vs clean tokens.

    Procedure
    ---------
    1. Inject hallucinations into the gold note (same setup as 2c).
    2. Tokenize with the generation prompt (same as 2c).
    3. Decompose each note token's residual stream into per-layer attention and
       MLP contributions via DLA: component_out · W_U[:, token].
    4. Compute layer-wise discriminability (AUROC + Cohen's d) for attn and MLP.
    5. Sum DLA over layers → total attn vs total MLP per token.
    6. Scatter: total attn DLA vs total MLP DLA, hallucinated tokens starred.

    Expected findings
    -----------------
    Hallucinated tokens should show:
      - Lower attention DLA  (attn heads are not copying relevant transcript info)
      - Higher MLP DLA       (parametric memory fires, overriding the transcript)

    Outputs
    -------
    exp2d_scatter.png               — total attn vs total MLP DLA; ★ = injected
    exp2d_layer_discriminability.png — AUROC + Cohen's d per layer
    exp2d_tokens.csv                 — per-token total + per-layer attn/mlp DLA
    exp2d_layer_discriminability.csv — per-layer AUROC and Cohen's d
    exp2d_corrupted_note.txt         — corrupted note text
    """
    print("\n" + "═"*54)
    print("  EXPERIMENT 2d — Direct Logit Attribution")
    print("═"*54)

    # ── 1. Inject hallucinations ─────────────────────────────────────────────
    _inject = inject_fn or inject_hallucinations
    corrupted_note, injections = _inject(
        gold_note, max_injections=3, seed=cfg.sample_idx + 42
    )

    print(f"\n  Injected {len(injections)} hallucination(s):")
    for inj in injections:
        orig = f"'{inj['original']}'" if inj["original"] else "(nothing — appended)"
        print(f"    [{inj['category']:22s}]  {orig}  →  '{inj['replacement']}'")

    (out / "exp2d_corrupted_note.txt").write_text(corrupted_note, encoding="utf-8")
    print("  Saved → exp2d_corrupted_note.txt")

    # ── 2. Tokenise with generation prompt ───────────────────────────────────
    tokens, t_len, note_toks = tokenize_as_generated(model, transcript, corrupted_note)
    tokens = tokens.to(cfg.device)
    note_len = len(note_toks)
    print(f"\n  Prompt (instruction + transcript) : {t_len} tokens")
    print(f"  Note                              : {note_len} tokens")

    # ── 3. Map injections to token indices ───────────────────────────────────
    halluc_idx = halluc_token_indices(model.tokenizer, corrupted_note, injections)
    print(f"\n  Hallucinated token positions ({len(halluc_idx)} tokens):")
    for i in halluc_idx:
        label = note_toks[i].replace("▁", "").replace("Ġ", "").strip() if i < note_len else "?"
        print(f"    [{i:4d}] '{label}'")

    halluc_mask = np.zeros(note_len, dtype=bool)
    halluc_mask[halluc_idx] = True

    # ── 4. Compute DLA ────────────────────────────────────────────────────────
    print("\n  Computing DLA (attention + MLP per layer) …")
    attn_dla, mlp_dla = compute_dla(model, tokens, t_len, cfg)

    # ── 5. Layer-wise discriminability ───────────────────────────────────────
    disc = dla_discriminability(attn_dla, mlp_dla, halluc_mask)

    n_layers = attn_dla.shape[0]

    if disc is not None:
        best_attn = int(np.argmax(np.abs(disc["attn_cohens_d"])))
        best_mlp  = int(np.argmax(np.abs(disc["mlp_cohens_d"])))
        print(f"\n  ── Layer-wise DLA discriminability ({n_layers} layers) ──")
        print(f"  {'layer':>6}  {'Attn AUROC':>11}  {'Attn d':>8}  {'MLP AUROC':>11}  {'MLP d':>8}")
        print("  " + "─" * 54)
        for l in range(n_layers):
            marker = ""
            if l == best_attn: marker += "  ← peak attn"
            if l == best_mlp:  marker += "  ← peak MLP"
            print(f"  {l:>6}  {disc['attn_auroc'][l]:>11.4f}  {disc['attn_cohens_d'][l]:>8.3f}"
                  f"  {disc['mlp_auroc'][l]:>11.4f}  {disc['mlp_cohens_d'][l]:>8.3f}{marker}")

        disc_df = pd.DataFrame({
            "layer":        np.arange(n_layers),
            "attn_auroc":   disc["attn_auroc"],
            "attn_cohens_d":disc["attn_cohens_d"],
            "mlp_auroc":    disc["mlp_auroc"],
            "mlp_cohens_d": disc["mlp_cohens_d"],
        })
        disc_df.to_csv(out / "exp2d_layer_discriminability.csv", index=False)
        print("  Saved → exp2d_layer_discriminability.csv")

        fig, axes = plt.subplots(2, 1, figsize=(max(10, n_layers // 2), 8), sharex=True)
        fig.suptitle(
            f"Exp 2d — DLA Layer-wise Discriminability  ({cfg.model_name})\n"
            f"Hallucinated ({halluc_mask.sum()}) vs. clean ({(~halluc_mask).sum()}) tokens",
            fontsize=11, fontweight="bold",
        )
        plot_layer_discriminability(
            disc, title="",
            axes=(axes[0], axes[1]),
            metric_a="attn", metric_b="mlp",
            label_a="Attention DLA", label_b="MLP DLA",
            color_a="#9C27B0", color_b="#F44336",
        )
        plt.tight_layout()
        fig.savefig(out / "exp2d_layer_discriminability.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved → exp2d_layer_discriminability.png")
    else:
        print("  [2d] Skipping layer discriminability — need ≥1 token in each class.")

    # ── 6. Total DLA and per-token CSV ───────────────────────────────────────
    total_attn = attn_dla.sum(axis=0)   # (note_len,)
    total_mlp  = mlp_dla.sum(axis=0)

    rows = [
        {
            "token_idx":  i,
            "token":      note_toks[i].replace("▁", "").replace("Ġ", "").strip(),
            "injected":   int(halluc_mask[i]),
            "total_attn": round(float(total_attn[i]), 5),
            "total_mlp":  round(float(total_mlp[i]),  5),
            **{f"attn_l{l}": round(float(attn_dla[l, i]), 5) for l in range(n_layers)},
            **{f"mlp_l{l}":  round(float(mlp_dla [l, i]), 5) for l in range(n_layers)},
        }
        for i in range(note_len)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp2d_tokens.csv", index=False)
    print("  Saved → exp2d_tokens.csv")

    if len(halluc_idx):
        df_h = df[df["injected"] == 1]
        print(f"\n  ── Injected token DLA summary ──")
        print(f"  {'token':<18}  {'total_attn':>11}  {'total_mlp':>11}")
        print("  " + "─" * 46)
        for _, row in df_h.iterrows():
            print(f"  {row['token']:<18}  {row['total_attn']:>11.4f}  {row['total_mlp']:>11.4f}")
        print(f"\n  Mean (hallucinated) — attn: {df_h['total_attn'].mean():.4f}"
              f"  mlp: {df_h['total_mlp'].mean():.4f}")
        df_c = df[df["injected"] == 0]
        print(f"  Mean (clean)        — attn: {df_c['total_attn'].mean():.4f}"
              f"  mlp: {df_c['total_mlp'].mean():.4f}")

    # ── 7. Scatter: total attn DLA vs total MLP DLA ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))

    em = float(np.median(total_attn))
    pm = float(np.median(total_mlp))

    colors = []
    for a, m in zip(total_attn, total_mlp):
        if   a >= em and m <  pm: colors.append(Q_COLORS["extractive"])    # high attn, low mlp
        elif a <  em and m >= pm: colors.append(Q_COLORS["parametric"])    # low attn, high mlp
        elif a >= em and m >= pm: colors.append(Q_COLORS["synthesized"])   # both high
        else:                     colors.append(Q_COLORS["hallucinatory"]) # both low

    ax.scatter(total_attn, total_mlp, c=colors, s=55, alpha=0.75,
               edgecolors="white", linewidth=0.4)

    # Annotate every 5th token
    for i, (a, m, tok) in enumerate(zip(total_attn, total_mlp, note_toks)):
        if i % 5 == 0:
            label = tok.replace("▁", "").replace("Ġ", "").strip()[:10]
            ax.annotate(label, (a, m), fontsize=5.5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    # Star hallucinated tokens
    if halluc_idx:
        hx = [total_attn[i] for i in halluc_idx if i < note_len]
        hy = [total_mlp [i] for i in halluc_idx if i < note_len]
        ax.scatter(hx, hy, s=220, c="black", marker="*", zorder=6,
                   label="Hallucinated token")
        for i in halluc_idx:
            if i >= note_len: continue
            label = note_toks[i].replace("▁", "").replace("Ġ", "").strip()[:16]
            ax.annotate(label, (total_attn[i], total_mlp[i]),
                        fontsize=7, color="#B71C1C", fontweight="bold",
                        xytext=(8, 8), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#B71C1C", lw=0.8))

    ax.axvline(em, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(pm, color="gray", ls="--", lw=0.8, alpha=0.5)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=Q_COLORS["extractive"],    label="High Attn, Low MLP"),
        mpatches.Patch(color=Q_COLORS["parametric"],    label="Low Attn, High MLP"),
        mpatches.Patch(color=Q_COLORS["synthesized"],   label="Both high"),
        mpatches.Patch(color=Q_COLORS["hallucinatory"], label="Both low"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right")

    ax.set_xlabel("Total Attention DLA  (Σ layers)", fontsize=10)
    ax.set_ylabel("Total MLP DLA  (Σ layers)", fontsize=10)
    ax.set_title(
        f"Exp 2d — Attention vs MLP DLA: Corrupted Note Tokens\n"
        f"({cfg.model_name})   ★ = injected hallucination tokens",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out / "exp2d_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2d_scatter.png")

    return {
        "attn_dla": attn_dla, "mlp_dla": mlp_dla,
        "total_attn": total_attn, "total_mlp": total_mlp,
        "halluc_idx": halluc_idx, "injections": injections,
        "disc": disc, "df": df,
    }


# ─────────────────────────────────────────────
# 13. Experiment 3 — Calibrated hallucination flagging
# ─────────────────────────────────────────────


def collect_training_data(
    model: HookedTransformer,
    cfg: Config,
    inject_fn,
    sample_indices: List[int],
    n_injections: int = 5,
) -> Dict:
    """
    Accumulate (ecs_layers, pks_layers, halluc_mask) training data for Exp 3
    by running synthetic hallucination injection on multiple ACI-Bench samples.

    Each sample goes through: load → inject → tokenize → ECS/PKS forward pass.
    Samples where injection yields zero hallucinated tokens are skipped.

    Parameters
    ----------
    sample_indices : list of integer row indices into ACI-Bench test1 split.
    n_injections   : maximum hallucinations injected per sample.

    Returns
    -------
    Dict with keys:
      ecs_layers_all  : (n_layers, N_total) — concatenated ECS layers
      pks_layers_all  : (n_layers, N_total) — concatenated PKS layers
      halluc_mask_all : (N_total,) bool
      n_samples       : number of samples successfully processed
      sample_stats    : list of per-sample dicts for logging
    """
    from dataclasses import replace as _dc_replace

    ecs_layers_list:  List[np.ndarray] = []
    pks_layers_list:  List[np.ndarray] = []
    halluc_mask_list: List[np.ndarray] = []
    sample_stats: List[Dict] = []

    for si in sample_indices:
        print(f"\n  [Training] Sample {si} …")
        # Build a per-sample config with the correct sample_idx
        cfg_i = _dc_replace(cfg, sample_idx=si)
        try:
            tr_i, note_i = load_aci_sample(cfg_i)
        except Exception as exc:
            print(f"  [Training] Sample {si}: load failed — {exc}")
            continue

        # Inject hallucinations
        try:
            corrupted, injections = inject_fn(note_i, max_injections=n_injections, seed=si)
        except Exception as exc:
            print(f"  [Training] Sample {si}: injection failed — {exc}")
            continue

        if not injections:
            print(f"  [Training] Sample {si}: no injections produced, skipping.")
            continue

        # Map injected spans → token positions
        halluc_idx = halluc_token_indices(model.tokenizer, corrupted, injections)

        # Tokenise with generation prompt prefix (teacher-forcing equivalent)
        tokens, t_len, note_toks = tokenize_as_generated(model, tr_i, corrupted)
        tokens   = tokens.to(cfg.device)
        note_len = len(note_toks)

        # Build halluc mask
        mask = np.zeros(note_len, dtype=bool)
        for idx in halluc_idx:
            if idx < note_len:
                mask[idx] = True

        if mask.sum() == 0:
            print(f"  [Training] Sample {si}: no hallucinated tokens in note span, skipping.")
            continue

        # Forward pass → per-layer ECS/PKS
        ecs_i, pks_i, ecs_layers_i, pks_layers_i = compute_ecs_pks(model, tokens, t_len, cfg)

        ecs_layers_list.append(ecs_layers_i)
        pks_layers_list.append(pks_layers_i)
        halluc_mask_list.append(mask)

        n_h = int(mask.sum())
        n_c = int((~mask).sum())
        sample_stats.append({
            "sample_idx":   si,
            "note_len":     note_len,
            "n_halluc":     n_h,
            "n_clean":      n_c,
            "n_injections": len(injections),
        })
        print(f"  [Training] Sample {si}: {note_len} tokens  "
              f"({n_h} hallucinated / {n_c} clean)")

    if not ecs_layers_list:
        raise RuntimeError(
            "collect_training_data: no samples collected — "
            "all samples failed during load/inject/tokenize."
        )

    ecs_layers_all  = np.concatenate(ecs_layers_list,  axis=1)
    pks_layers_all  = np.concatenate(pks_layers_list,  axis=1)
    halluc_mask_all = np.concatenate(halluc_mask_list)

    # Per-sample records kept separately so callers can compute per-example stats
    per_sample = [
        {
            "sample_idx":  sample_stats[i]["sample_idx"],
            "ecs_layers":  ecs_layers_list[i],   # (n_layers, n_tokens_i)
            "pks_layers":  pks_layers_list[i],   # (n_layers, n_tokens_i)
            "halluc_mask": halluc_mask_list[i],  # (n_tokens_i,)
        }
        for i in range(len(ecs_layers_list))
    ]

    n_h_total = int(halluc_mask_all.sum())
    n_c_total = int((~halluc_mask_all).sum())
    print(f"\n  Training set: {len(ecs_layers_list)} samples  |  "
          f"{halluc_mask_all.shape[0]} tokens total  "
          f"({n_h_total} hallucinated / {n_c_total} clean)")

    return {
        "ecs_layers_all":  ecs_layers_all,
        "pks_layers_all":  pks_layers_all,
        "halluc_mask_all": halluc_mask_all,
        "per_sample":      per_sample,
        "n_samples":       len(ecs_layers_list),
        "sample_stats":    sample_stats,
    }


def _load_exp4_data(exp4_out: Path) -> Optional[Dict]:
    """
    Load all reusable data from a previous Exp 4 run for use in Exp 3.

    Reads:
      - ``exp4_layer_stats.csv``          — per-example × per-layer discriminability
      - ``activations/sample_*_activations.npz`` — raw token-level ecs_layers,
            pks_layers, halluc_mask saved during Exp 4

    Returns a dict with keys:
        pks_pearson_r_per, ecs_pearson_r_per,
        pks_auroc_per,     ecs_auroc_per,
        pks_cohens_d_per,  ecs_cohens_d_per  — lists of (n_layers,) arrays
        per_sample    — list of dicts {sample_idx, ecs_layers, pks_layers, halluc_mask}
                        populated only for examples whose .npz file exists
        ecs_layers_all, pks_layers_all, halluc_mask_all
                      — concatenated activations across all loaded .npz files
        n_layers      — int
        sample_indices — list of int (from CSV)
    Returns None if exp4_layer_stats.csv is missing or malformed.
    """
    csv_path = exp4_out / "exp4_layer_stats.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  [Exp 3] Could not read {csv_path}: {exc}")
        return None

    required = {"sample_idx", "layer", "pks_pearson_r", "ecs_pearson_r",
                "pks_auroc", "ecs_auroc", "pks_cohens_d", "ecs_cohens_d"}
    if not required.issubset(df.columns):
        print(f"  [Exp 3] exp4_layer_stats.csv missing columns: {required - set(df.columns)}")
        return None

    sample_indices = sorted(df["sample_idx"].unique().tolist())
    n_layers = int(df["layer"].max()) + 1

    pks_pearson_r_per: List[np.ndarray] = []
    ecs_pearson_r_per: List[np.ndarray] = []
    pks_auroc_per:     List[np.ndarray] = []
    ecs_auroc_per:     List[np.ndarray] = []
    pks_cohens_d_per:  List[np.ndarray] = []
    ecs_cohens_d_per:  List[np.ndarray] = []

    for si in sample_indices:
        rows = df[df["sample_idx"] == si].sort_values("layer")
        if len(rows) != n_layers:
            print(f"  [Exp 3] Sample {si}: {len(rows)} layers in CSV (expected {n_layers}), skipping.")
            continue
        pks_pearson_r_per.append(rows["pks_pearson_r"].to_numpy())
        ecs_pearson_r_per.append(rows["ecs_pearson_r"].to_numpy())
        pks_auroc_per.append(rows["pks_auroc"].to_numpy())
        ecs_auroc_per.append(rows["ecs_auroc"].to_numpy())
        pks_cohens_d_per.append(rows["pks_cohens_d"].to_numpy())
        ecs_cohens_d_per.append(rows["ecs_cohens_d"].to_numpy())

    if not pks_pearson_r_per:
        return None

    # Load raw activations from .npz files
    act_dir = exp4_out / "activations"
    per_sample: List[Dict] = []
    ecs_list, pks_list, mask_list = [], [], []

    for si in sample_indices:
        npz_path = act_dir / f"sample_{si:04d}_activations.npz"
        if not npz_path.exists():
            print(f"  [Exp 3] No activations file for sample {si} — skipping for regressor.")
            continue
        try:
            npz = np.load(npz_path)
            ecs_l = npz["ecs_layers"].astype(np.float64)   # (n_layers, n_tokens)
            pks_l = npz["pks_layers"].astype(np.float64)
            mask  = npz["halluc_mask"].astype(bool)
        except Exception as exc:
            print(f"  [Exp 3] Could not load {npz_path}: {exc}")
            continue

        per_sample.append({
            "sample_idx":  si,
            "ecs_layers":  ecs_l,
            "pks_layers":  pks_l,
            "halluc_mask": mask,
        })
        ecs_list.append(ecs_l)
        pks_list.append(pks_l)
        mask_list.append(mask)

    if ecs_list:
        ecs_layers_all  = np.concatenate(ecs_list,  axis=1)
        pks_layers_all  = np.concatenate(pks_list,  axis=1)
        halluc_mask_all = np.concatenate(mask_list)
    else:
        ecs_layers_all = pks_layers_all = halluc_mask_all = None

    n_act = len(per_sample)
    print(f"  [Exp 3] Loaded Exp 4 data: {len(pks_pearson_r_per)} examples (discriminability), "
          f"{n_act} examples (activations)")

    return {
        "pks_pearson_r_per": pks_pearson_r_per,
        "ecs_pearson_r_per": ecs_pearson_r_per,
        "pks_auroc_per":     pks_auroc_per,
        "ecs_auroc_per":     ecs_auroc_per,
        "pks_cohens_d_per":  pks_cohens_d_per,
        "ecs_cohens_d_per":  ecs_cohens_d_per,
        "per_sample":        per_sample,
        "ecs_layers_all":    ecs_layers_all,
        "pks_layers_all":    pks_layers_all,
        "halluc_mask_all":   halluc_mask_all,
        "n_layers":          n_layers,
        "sample_indices":    sample_indices,
    }


def run_experiment_3(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    transcript: str,
    gold_note: str,
    inject_fn=None,
    n_train_samples: int = 3,
    n_injections: int = 5,
    n_ffn_layers: int = 5,
    n_copy_layers: int = 5,
    halluc_threshold: float = 0.5,
    exp4_out: Optional[Path] = None,
) -> Dict:
    """
    Experiment 3: REDEEP hallucination scoring on a real generated note.

    Implements the paper formula:
        Ht(t) = Σ_{l∈F} α·P^l_t  −  Σ_{l∈A} β·E^{l,h}_t

    where α, β > 0 are learned via logistic regression on a multi-sample
    training set with synthetic hallucination injection.

    Updated version:
    ----------------
    Layers are selected using AUROC rather than Pearson r.

    F = top-k layers by highest PKS AUROC
        because higher PKS should indicate hallucination.

    A = top-k layers by lowest raw ECS AUROC
        because lower ECS should indicate hallucination.

    Pipeline
    --------
    1. Collect training data — run ECS/PKS with hallucination injection on
       `n_train_samples` ACI-Bench samples, not the target sample.
       Skipped for layer-selection if ``exp4_out`` is given and
       ``exp4_layer_stats.csv`` exists there.
    2. Layer-wise discriminability — compute Pearson r, AUROC, Cohen's d.
    3. Identify set F — top-`n_ffn_layers` by PKS AUROC.
    4. Identify set A — top-`n_copy_layers` by lowest raw ECS AUROC.
    5. Fit α, β — logistic regression on [PKS_F | ECS_A] features.
    6. Score the generated note — compute halluc_prob via clf.predict_proba.
    7. Validate — gold n-gram coverage + optional NLI cross-encoder.
    8. Outputs — CSV, scatter, HTML report.
    """
    print("\n" + "═" * 54)
    print("  EXPERIMENT 3 — REDEEP Hallucination Scoring")
    print("═" * 54)

    _inject = inject_fn or inject_hallucinations

    # ── 1. Training data — from Exp 4 cache or fresh forward passes ──────────
    exp4_data = None
    if exp4_out is not None:
        print(f"\n  Step 1/8 — Loading training data from Exp 4 output: {exp4_out} …")
        exp4_data = _load_exp4_data(exp4_out)
        if exp4_data is None:
            print("  [Exp 3] Exp 4 data could not be loaded — falling back to fresh collection.")

    if exp4_data is not None:
        # ── Path A: reuse Exp 4 discriminability stats + saved activations ──
        pks_pearson_r_per = exp4_data["pks_pearson_r_per"]
        ecs_pearson_r_per = exp4_data["ecs_pearson_r_per"]
        pks_auroc_per = exp4_data["pks_auroc_per"]
        ecs_auroc_per = exp4_data["ecs_auroc_per"]
        pks_cohens_d_per = exp4_data["pks_cohens_d_per"]
        ecs_cohens_d_per = exp4_data["ecs_cohens_d_per"]
        n_layers = exp4_data["n_layers"]

        ecs_layers_all = exp4_data["ecs_layers_all"]
        pks_layers_all = exp4_data["pks_layers_all"]
        halluc_mask_all = exp4_data["halluc_mask_all"]

        if ecs_layers_all is not None:
            n_h_total = int(halluc_mask_all.sum())
            n_c_total = int((~halluc_mask_all).sum())
            print(
                f"  Loaded activations: {halluc_mask_all.shape[0]} tokens total  "
                f"({n_h_total} hallucinated / {n_c_total} clean)"
            )
        else:
            print("  No activations found in Exp 4 output — regressor step will re-run forward passes.")
            n_h_total = n_c_total = 0

        per_example_rows: List[Dict] = []
        for i, si in enumerate(exp4_data["sample_indices"][:len(pks_pearson_r_per)]):
            for l in range(n_layers):
                per_example_rows.append({
                    "sample_idx": si,
                    "layer": l,
                    "pks_pearson_r": round(float(pks_pearson_r_per[i][l]), 6),
                    "ecs_pearson_r": round(float(ecs_pearson_r_per[i][l]), 6),
                    "pks_auroc": round(float(pks_auroc_per[i][l]), 6),
                    "ecs_auroc": round(float(ecs_auroc_per[i][l]), 6),
                    "pks_cohens_d": round(float(pks_cohens_d_per[i][l]), 6),
                    "ecs_cohens_d": round(float(ecs_cohens_d_per[i][l]), 6),
                })

    else:
        # ── Path B: fresh collection ─────────────────────────────────────────
        print(
            f"\n  Step 1/8 — Collecting training data "
            f"({n_train_samples} samples, {n_injections} injections each) …"
        )

        dataset_size = 250
        all_indices = [i for i in range(dataset_size) if i != cfg.sample_idx]
        train_indices = all_indices[:n_train_samples]

        try:
            train_data = collect_training_data(
                model, cfg, _inject, train_indices, n_injections=n_injections
            )
        except RuntimeError as exc:
            print(f"\n  [Exp 3] Training data collection failed: {exc}")
            print("  Falling back to single-sample calibration from Exp 2c …")
            calib_2c = run_experiment_2c(
                model, cfg, out, transcript, gold_note, inject_fn=_inject
            )
            ecs_layers_all = calib_2c["ecs_layers"]
            pks_layers_all = calib_2c["pks_layers"]
            note_len_cal = ecs_layers_all.shape[1]
            halluc_mask_all = np.array(
                [i in set(calib_2c["halluc_idx"]) for i in range(note_len_cal)],
                dtype=bool,
            )
            train_data = {
                "ecs_layers_all": ecs_layers_all,
                "pks_layers_all": pks_layers_all,
                "halluc_mask_all": halluc_mask_all,
                "per_sample": [],
                "n_samples": 1,
                "sample_stats": [],
            }

        ecs_layers_all = train_data["ecs_layers_all"]
        pks_layers_all = train_data["pks_layers_all"]
        halluc_mask_all = train_data["halluc_mask_all"]
        n_layers = ecs_layers_all.shape[0]
        n_h_total = int(halluc_mask_all.sum())
        n_c_total = int((~halluc_mask_all).sum())

        if train_data["sample_stats"]:
            pd.DataFrame(train_data["sample_stats"]).to_csv(
                out / "exp3_training_summary.csv", index=False
            )
            print("  Saved → exp3_training_summary.csv")

        print(f"\n  Step 2/8 — Computing per-example layer-wise discriminability …")

        pks_pearson_r_per: List[np.ndarray] = []
        ecs_pearson_r_per: List[np.ndarray] = []
        pks_auroc_per: List[np.ndarray] = []
        ecs_auroc_per: List[np.ndarray] = []
        pks_cohens_d_per: List[np.ndarray] = []
        ecs_cohens_d_per: List[np.ndarray] = []
        per_example_rows: List[Dict] = []

        for rec in train_data.get("per_sample", []):
            si = rec["sample_idx"]
            disc_i = layer_discriminability(
                rec["pks_layers"], rec["ecs_layers"], rec["halluc_mask"]
            )

            if disc_i is None:
                print(f"  [Step 2] Sample {si}: skipped (single class in mask).")
                continue

            pks_pearson_r_per.append(disc_i["pks_pearson_r"])
            ecs_pearson_r_per.append(disc_i["ecs_pearson_r"])
            pks_auroc_per.append(disc_i["pks_auroc"])
            ecs_auroc_per.append(disc_i["ecs_auroc"])
            pks_cohens_d_per.append(disc_i["pks_cohens_d"])
            ecs_cohens_d_per.append(disc_i["ecs_cohens_d"])

            for l in range(n_layers):
                per_example_rows.append({
                    "sample_idx": si,
                    "layer": l,
                    "pks_pearson_r": round(float(disc_i["pks_pearson_r"][l]), 6),
                    "ecs_pearson_r": round(float(disc_i["ecs_pearson_r"][l]), 6),
                    "pks_auroc": round(float(disc_i["pks_auroc"][l]), 6),
                    "ecs_auroc": round(float(disc_i["ecs_auroc"][l]), 6),
                    "pks_cohens_d": round(float(disc_i["pks_cohens_d"][l]), 6),
                    "ecs_cohens_d": round(float(disc_i["ecs_cohens_d"][l]), 6),
                })

            print(
                f"  Sample {si}: "
                f"PKS r mean={disc_i['pks_pearson_r'].mean():+.4f}, "
                f"PKS AUROC mean={disc_i['pks_auroc'].mean():.4f}  |  "
                f"ECS r mean={disc_i['ecs_pearson_r'].mean():+.4f}, "
                f"ECS AUROC mean={disc_i['ecs_auroc'].mean():.4f}"
            )

    # ── end of Path A / Path B branching ────────────────────────────────────

    if not pks_pearson_r_per:
        raise RuntimeError(
            "Exp 3: no valid per-example discriminability results — "
            "all training samples had single-class masks."
        )

    n_valid_samples = len(pks_pearson_r_per)
    print(f"\n  Valid samples for aggregation: {n_valid_samples}")

    # Aggregate: mean across examples → shape (n_layers,)
    pks_pearson_r = np.stack(pks_pearson_r_per, axis=0).mean(axis=0)
    ecs_pearson_r = np.stack(ecs_pearson_r_per, axis=0).mean(axis=0)
    pks_pearson_r_std = np.stack(pks_pearson_r_per, axis=0).std(axis=0)
    ecs_pearson_r_std = np.stack(ecs_pearson_r_per, axis=0).std(axis=0)

    pks_auroc = np.stack(pks_auroc_per, axis=0).mean(axis=0)
    ecs_auroc = np.stack(ecs_auroc_per, axis=0).mean(axis=0)
    pks_auroc_std = np.stack(pks_auroc_per, axis=0).std(axis=0)
    ecs_auroc_std = np.stack(ecs_auroc_per, axis=0).std(axis=0)

    pks_cohens_d = np.stack(pks_cohens_d_per, axis=0).mean(axis=0)
    ecs_cohens_d = np.stack(ecs_cohens_d_per, axis=0).mean(axis=0)

    # Save: per-example detail + aggregate summary
    pd.DataFrame(per_example_rows).to_csv(
        out / "exp3_training_correlations_per_example.csv", index=False
    )
    print("  Saved → exp3_training_correlations_per_example.csv")

    corr_df = pd.DataFrame({
        "layer": np.arange(n_layers),
        "pks_pearson_r_mean": pks_pearson_r,
        "pks_pearson_r_std": pks_pearson_r_std,
        "ecs_pearson_r_mean": ecs_pearson_r,
        "ecs_pearson_r_std": ecs_pearson_r_std,
        "pks_auroc_mean": pks_auroc,
        "pks_auroc_std": pks_auroc_std,
        "ecs_auroc_mean": ecs_auroc,
        "ecs_auroc_std": ecs_auroc_std,
        "ecs_auroc_reversed_mean": 1.0 - ecs_auroc,
        "ecs_auroc_reversed_std": ecs_auroc_std,
        "pks_cohens_d_mean": pks_cohens_d,
        "ecs_cohens_d_mean": ecs_cohens_d,
    })
    corr_df.to_csv(out / "exp3_training_correlations.csv", index=False)
    print("  Saved → exp3_training_correlations.csv")

    # ── Band plot: AUROC, Cohen's d, Pearson r per layer ─────────────────────
    _layers_ax = np.arange(n_layers)
    _colour_ecs = "#2196F3"
    _colour_pks = "#FF9800"

    fig3, axes3 = plt.subplots(3, 2, figsize=(max(14, n_layers // 2), 12), sharex=True)
    fig3.suptitle(
        f"Exp 3 — Training Discriminability per Layer  ({cfg.model_name})\n"
        f"Mean ± 1 SD across {n_valid_samples} training example(s)  "
        f"({n_injections} injections each)",
        fontsize=12,
        fontweight="bold",
    )

    _band_specs3 = [
        (ecs_auroc_per, axes3[0, 0], "ECS — raw AUROC", "AUROC", _colour_ecs, 0.5, "Chance"),
        (pks_auroc_per, axes3[0, 1], "PKS — AUROC", "AUROC", _colour_pks, 0.5, "Chance"),
        (ecs_cohens_d_per, axes3[1, 0], "ECS — Cohen's d", "Cohen's d", _colour_ecs, 0.0, "Zero"),
        (pks_cohens_d_per, axes3[1, 1], "PKS — Cohen's d", "Cohen's d", _colour_pks, 0.0, "Zero"),
        (ecs_pearson_r_per, axes3[2, 0], "ECS — Pearson r", "Pearson r", _colour_ecs, 0.0, "r = 0"),
        (pks_pearson_r_per, axes3[2, 1], "PKS — Pearson r", "Pearson r", _colour_pks, 0.0, "r = 0"),
    ]

    for per_list, ax, title, ylabel, colour, hline, hline_label in _band_specs3:
        mat = np.stack(per_list, axis=0)
        mean_v = mat.mean(axis=0)
        std_v = mat.std(axis=0)

        for row in mat:
            ax.plot(_layers_ax, row, color=colour, lw=0.7, alpha=0.25)

        ax.fill_between(
            _layers_ax,
            mean_v - std_v,
            mean_v + std_v,
            color=colour,
            alpha=0.22,
            label="±1 SD",
        )
        ax.plot(_layers_ax, mean_v, color=colour, lw=2.2, label="Mean")
        ax.axhline(hline, color="gray", ls="--", lw=1.0, alpha=0.6, label=hline_label)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls=":", alpha=0.4)

    for ax in axes3[2]:
        ax.set_xlabel("Layer", fontsize=10)

    plt.tight_layout()
    training_stats_path = out / "exp3_training_layer_stats.png"
    fig3.savefig(training_stats_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved → exp3_training_layer_stats.png")

    # Print top-5 by aggregated AUROC for each metric
    print(f"\n  ── Top layers by mean PKS AUROC (highest → set F) ──")
    top_pks = np.argsort(pks_auroc)[::-1][:5]
    for l in top_pks:
        print(
            f"    layer {l:>3}  "
            f"PKS AUROC={pks_auroc[l]:.4f} ±{pks_auroc_std[l]:.4f}  "
            f"r={pks_pearson_r[l]:+.4f} ±{pks_pearson_r_std[l]:.4f}  "
            f"d={pks_cohens_d[l]:+.3f}"
        )

    print(f"\n  ── Top layers by mean raw ECS AUROC (lowest → set A) ──")
    top_ecs = np.argsort(ecs_auroc)[:5]
    for l in top_ecs:
        print(
            f"    layer {l:>3}  "
            f"raw ECS AUROC={ecs_auroc[l]:.4f} ±{ecs_auroc_std[l]:.4f}  "
            f"reversed={1.0 - ecs_auroc[l]:.4f}  "
            f"r={ecs_pearson_r[l]:+.4f} ±{ecs_pearson_r_std[l]:.4f}  "
            f"d={ecs_cohens_d[l]:+.3f}"
        )

    # ── 3. Identify set F by PKS AUROC ───────────────────────────────────────
    print(
        f"\n  Step 3/8 — Identifying set F "
        f"(top-{n_ffn_layers} Knowledge FFN layers by PKS AUROC) …"
    )

    F = np.argsort(pks_auroc)[::-1][:n_ffn_layers].tolist()

    print(
        f"  Set F = {F}  "
        f"(PKS AUROC values: {[round(float(pks_auroc[l]), 4) for l in F]})"
    )

    # ── 4. Identify set A by ECS AUROC ───────────────────────────────────────
    print(
        f"\n  Step 4/8 — Identifying set A "
        f"(top-{n_copy_layers} Copying Head layers by lowest raw ECS AUROC) …"
    )

    # Raw ECS AUROC is low when clean tokens tend to have higher ECS than hallucinated tokens.
    # Equivalently, these are the highest AUROC layers for -ECS.
    A = np.argsort(ecs_auroc)[:n_copy_layers].tolist()

    print(
        f"  Set A = {A}  "
        f"(raw ECS AUROC values: {[round(float(ecs_auroc[l]), 4) for l in A]}, "
        f"reversed: {[round(float(1.0 - ecs_auroc[l]), 4) for l in A]})"
    )

    # ── 5. Fit logistic regression α, β ──────────────────────────────────────
    print(f"\n  Step 5/8 — Fitting REDEEP logistic regressor (|F|={len(F)}, |A|={len(A)}) …")

    if ecs_layers_all is None:
        print("  No cached activations — running fresh forward passes for regressor …")
        dataset_size = 250
        all_indices = [i for i in range(dataset_size) if i != cfg.sample_idx]
        train_indices = all_indices[:n_train_samples]
        train_data = collect_training_data(
            model, cfg, _inject, train_indices, n_injections=n_injections
        )
        ecs_layers_all = train_data["ecs_layers_all"]
        pks_layers_all = train_data["pks_layers_all"]
        halluc_mask_all = train_data["halluc_mask_all"]

    clf, scaler, alpha, beta = fit_hallucination_regressor(
        pks_layers_all, ecs_layers_all, halluc_mask_all, F, A
    )

    print(f"  α (PKS weight) = {alpha:.6f}")
    print(f"  β (ECS weight) = {beta:.6f}")

    try:
        from sklearn.metrics import roc_auc_score as _auc

        pks_feats_tr = pks_layers_all[F].T
        ecs_feats_tr = ecs_layers_all[A].T
        X_tr = np.concatenate([pks_feats_tr, ecs_feats_tr], axis=1)
        X_tr_sc = scaler.transform(X_tr)
        prob_tr = clf.predict_proba(X_tr_sc)[:, 1]
        auc_tr = _auc(halluc_mask_all.astype(int), prob_tr)
        print(f"  Training AUROC = {auc_tr:.4f}")

    except Exception as exc:
        print(f"  [Exp 3] Training AUROC skipped: {exc}")

    sel_txt = (
        f"Set F (Knowledge FFNs, top-{n_ffn_layers} by mean PKS AUROC): {F}\n"
        f"Set A (Copying Heads,  top-{n_copy_layers} by lowest mean raw ECS AUROC): {A}\n"
        f"alpha (PKS coefficient): {alpha:.8f}\n"
        f"beta  (ECS coefficient): {beta:.8f}\n"
        f"\nPKS AUROC at F layers:\n"
        + "\n".join(
            f"  layer {l}: AUROC={pks_auroc[l]:.6f}, "
            f"Pearson r={pks_pearson_r[l]:+.6f} ± {pks_pearson_r_std[l]:.6f}, "
            f"Cohen's d={pks_cohens_d[l]:+.6f}"
            for l in F
        )
        + "\n"
        f"\nECS AUROC at A layers:\n"
        + "\n".join(
            f"  layer {l}: raw AUROC={ecs_auroc[l]:.6f}, "
            f"reversed AUROC={1.0 - ecs_auroc[l]:.6f}, "
            f"Pearson r={ecs_pearson_r[l]:+.6f} ± {ecs_pearson_r_std[l]:.6f}, "
            f"Cohen's d={ecs_cohens_d[l]:+.6f}"
            for l in A
        )
        + "\n"
        f"\nTraining samples: {train_data['n_samples'] if 'train_data' in locals() else n_valid_samples}  |  "
        f"Tokens: {halluc_mask_all.shape[0]}  "
        f"({n_h_total} hallucinated / {n_c_total} clean)\n"
    )

    (out / "exp3_selected_layers.txt").write_text(sel_txt, encoding="utf-8")
    print("  Saved → exp3_selected_layers.txt")

    # ── 6. Score the generated note ──────────────────────────────────────────
    print(f"\n  Step 6/8 — Scoring the generated note …")

    generated_note = generate_note(model, transcript, cfg)

    tokens, t_len, note_toks = tokenize_as_generated(model, transcript, generated_note)
    tokens = tokens.to(cfg.device)
    note_len = len(note_toks)

    print(f"  Prompt  : {t_len} tokens  |  Note: {note_len} tokens")

    ecs, pks, ecs_layers, pks_layers = compute_ecs_pks(model, tokens, t_len, cfg)

    pks_feats_gen = pks_layers[F].T
    ecs_feats_gen = ecs_layers[A].T
    X_gen = np.concatenate([pks_feats_gen, ecs_feats_gen], axis=1)
    X_gen_sc = scaler.transform(X_gen)

    halluc_prob = clf.predict_proba(X_gen_sc)[:, 1]
    halluc_prob = np.clip(halluc_prob, 0.0, 1.0)
    flagged = halluc_prob >= halluc_threshold

    print(
        f"  Tokens flagged (prob ≥ {halluc_threshold}): "
        f"{int(flagged.sum())} / {note_len}  ({100 * flagged.mean():.1f}%)"
    )
    print(
        f"  Mean halluc_prob : {halluc_prob.mean():.4f}  "
        f"Median: {np.median(halluc_prob):.4f}"
    )

    # ── 7. Outputs ───────────────────────────────────────────────────────────
    print(f"\n  Step 7/8 — Writing outputs …")

    rows = [
        {
            "token_idx":   i,
            "token":       note_toks[i].replace("▁", "").replace("Ġ", "").strip(),
            "ecs":         round(float(ecs[i]),         4),
            "pks":         round(float(pks[i]),         4),
            "halluc_prob": round(float(halluc_prob[i]), 4),
            "flagged":     int(flagged[i]),
        }
        for i in range(note_len)
    ]

    df = pd.DataFrame(rows)
    df.to_csv(out / "exp3_tokens.csv", index=False)
    print("  Saved → exp3_tokens.csv")

    # Scatter: ECS vs PKS coloured by halluc_prob
    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        ecs,
        pks,
        c=halluc_prob,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
        s=55,
        alpha=0.80,
        edgecolors="white",
        linewidth=0.4,
    )

    plt.colorbar(
        sc,
        ax=ax,
        label="Hallucination probability  (REDEEP logistic)",
        shrink=0.75,
    )

    top_idx = np.argsort(halluc_prob)[-10:]
    ax.scatter(
        ecs[top_idx],
        pks[top_idx],
        s=180,
        c="black",
        marker="*",
        zorder=6,
        label="Top-10 risk tokens",
    )

    for i in top_idx:
        lbl = note_toks[i].replace("▁", "").replace("Ġ", "").strip()[:14]
        ax.annotate(
            lbl,
            (ecs[i], pks[i]),
            fontsize=6.5,
            color="#B71C1C",
            fontweight="bold",
            xytext=(6, 6),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#B71C1C", lw=0.7),
        )

    em = float(np.median(ecs))
    pm = float(np.median(pks))

    ax.axvline(em, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(pm, color="gray", ls="--", lw=0.8, alpha=0.5)

    ax.legend(fontsize=8)
    ax.set_xlabel("External Context Score (ECS)", fontsize=10)
    ax.set_ylabel("Parametric Knowledge Score (PKS)", fontsize=10)
    ax.set_title(
        f"Exp 3 — Generated Note: ECS vs PKS  ({cfg.model_name})\n"
        f"Colour = REDEEP hallucination probability  "
        f"(F={F[:3]}…, A={A[:3]}…, α={alpha:.3f}, β={beta:.3f})",
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    scatter_path = out / "exp3_scatter_calibrated.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp3_scatter_calibrated.png")

    # ── Inline HTML report ───────────────────────────────────────────────────
    def _risk_colour(p: float) -> str:
        r = int(255 * p)
        g = int(255 * (1 - p))
        return f"rgb({r},{g},60)"

    token_spans = "".join(
        f'<span title="prob={halluc_prob[i]:.3f} | ecs={ecs[i]:.3f} | pks={pks[i]:.3f}" '
        f'style="background:{_risk_colour(float(halluc_prob[i]))}; '
        f'opacity:{0.35 + 0.65 * float(halluc_prob[i]):.2f}; '
        f'border-radius:3px; padding:1px 2px; margin:1px;">'
        f'{note_toks[i].replace("<", "&lt;").replace(">", "&gt;")}'
        f'</span>'
        for i in range(note_len)
    )

    layer_rows = "".join(
        f"<tr><td>{l}</td>"
        f"<td>{pks_auroc[l]:.4f}</td><td>{pks_pearson_r[l]:+.4f}</td><td>{pks_cohens_d[l]:+.4f}</td>"
        f"<td>{ecs_auroc[l]:.4f}</td><td>{ecs_pearson_r[l]:+.4f}</td><td>{ecs_cohens_d[l]:+.4f}</td>"
        f"<td>{'★ F' if l in F else ''}{'★ A' if l in A else ''}</td></tr>"
        for l in range(n_layers)
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exp 3 Report — {cfg.model_name}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; color: #222; }}
  h1 {{ font-size: 1.3em; }} h2 {{ font-size: 1.1em; margin-top: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
  th {{ background: #f0f0f0; text-align: center; }}
  .note {{ line-height: 2.2; font-size: 0.9em; background: #fafafa;
           border: 1px solid #ddd; padding: 1em; border-radius: 6px; }}
  img {{ max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }}
  .meta {{ font-size: 0.85em; color: #555; }}
</style></head><body>
<h1>Experiment 3 — REDEEP Hallucination Report</h1>
<p class="meta">Model: <b>{cfg.model_name}</b> &nbsp;|&nbsp;
Sample: <b>{cfg.sample_idx}</b> &nbsp;|&nbsp;
Threshold: <b>{halluc_threshold}</b> &nbsp;|&nbsp;
Flagged: <b>{int(flagged.sum())} / {note_len}</b> tokens &nbsp;|&nbsp;
α={alpha:.4f} &nbsp; β={beta:.4f}<br>
Set F (PKS layers): {F}<br>
Set A (ECS layers): {A}</p>

<h2>Generated Note — colour = hallucination probability</h2>
<div class="note">{token_spans}</div>

<h2>ECS vs PKS Scatter</h2>
<img src="{scatter_path.name}" alt="scatter">

<h2>Layer-wise Discriminability (mean across training examples)</h2>
<table>
<tr><th>Layer</th>
<th>PKS AUROC</th><th>PKS Pearson r</th><th>PKS Cohen d</th>
<th>ECS AUROC</th><th>ECS Pearson r</th><th>ECS Cohen d</th>
<th>Selected</th></tr>
{layer_rows}
</table>
</body></html>"""

    html_path = out / "exp3_report.html"
    html_path.write_text(html, encoding="utf-8")
    print("  Saved → exp3_report.html")

    return {
        "halluc_prob":    halluc_prob,
        "flagged":        flagged,
        "ecs":            ecs,
        "pks":            pks,
        "ecs_layers":     ecs_layers,
        "pks_layers":     pks_layers,
        "F":              F,
        "A":              A,
        "alpha":          alpha,
        "beta":           beta,
        "pks_pearson_r":  pks_pearson_r,
        "ecs_pearson_r":  ecs_pearson_r,
        "pks_auroc":      pks_auroc,
        "ecs_auroc":      ecs_auroc,
        "df":             df,
        "generated_note": generated_note,
        "clf":            clf,
        "scaler":         scaler,
    }

# ─────────────────────────────────────────────
# 14. Experiment 4 — Layer-wise statistics across examples
# ─────────────────────────────────────────────

def run_experiment_4(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    inject_fn,
    n_examples: int = 10,
    n_injections: int = 5,
    sample_start: int = 0,
) -> Dict:
    """
    Experiment 4: Layer-wise ECS/PKS discriminability statistics aggregated
    across multiple ACI-Bench examples with synthetic hallucination injection.

    For each example the pipeline is:
      load → inject → tokenize (generation prompt) → ECS/PKS forward pass
      → layer_discriminability (AUROC, Cohen's d, Pearson r per layer)

    Results are collected across all valid examples, then summarised as:
      • Mean ± std of AUROC and Cohen's d per layer   (band plots)
      • Per-example Pearson r curves per layer         (spaghetti + mean)

    This reveals which layers are *consistently* discriminative across
    different notes, vs. which are noisy or note-specific.

    Parameters
    ----------
    n_examples   : number of ACI-Bench examples to process.
    n_injections : max hallucinations injected per example.
    sample_start : first ACI-Bench row index (inclusive).

    Outputs
    -------
    exp4_layer_stats.png  — 3×2 grid: AUROC, Cohen's d, Pearson r (ECS|PKS),
                            each panel shows mean ± 1 SD band + thin per-example traces
    exp4_layer_stats.csv  — tidy per-example × per-layer table
    exp4_summary.csv      — per-example summary (n_halluc, n_clean, mean metrics)
    """
    from dataclasses import replace as _dc_replace

    print("\n" + "═"*54)
    print("  EXPERIMENT 4 — Layer-wise Statistics Across Examples")
    print("═"*54)
    print(f"\n  Examples  : {sample_start} → {sample_start + n_examples - 1}")
    print(f"  Injections: {n_injections} per example")

    _inject = inject_fn or inject_hallucinations

    # ── 1. Per-example data collection ──────────────────────────────────────
    # Storage: list of (n_layers,) arrays, one entry per valid example
    pks_auroc_all:    List[np.ndarray] = []
    pks_cohens_d_all: List[np.ndarray] = []
    pks_pearson_r_all: List[np.ndarray] = []
    ecs_auroc_all:    List[np.ndarray] = []
    ecs_cohens_d_all: List[np.ndarray] = []
    ecs_pearson_r_all: List[np.ndarray] = []

    summary_rows: List[Dict] = []
    csv_rows:     List[Dict] = []
    valid_sample_indices: List[int] = []

    # Per-example output folders
    auroc_dir       = out / "auroc"
    cohens_d_dir    = out / "cohens_d"
    pearson_r_dir   = out / "pearson_r"
    activations_dir = out / "activations"
    for _d in (auroc_dir, cohens_d_dir, pearson_r_dir, activations_dir):
        _d.mkdir(parents=True, exist_ok=True)

    for si in range(sample_start, sample_start + n_examples):
        print(f"\n  ── Sample {si} ──────────────────────────────────────")
        cfg_i = _dc_replace(cfg, sample_idx=si)

        try:
            transcript, gold_note = load_aci_sample(cfg_i)
        except Exception as exc:
            print(f"  [4] Sample {si}: load failed — {exc}")
            continue

        try:
            corrupted, injections = _inject(gold_note, max_injections=n_injections, seed=si)
        except Exception as exc:
            print(f"  [4] Sample {si}: injection failed — {exc}")
            continue

        if not injections:
            print(f"  [4] Sample {si}: no injections produced, skipping.")
            continue

        halluc_idx = halluc_token_indices(model.tokenizer, corrupted, injections)

        tokens, t_len, note_toks = tokenize_as_generated(model, transcript, corrupted)
        tokens   = tokens.to(cfg.device)
        note_len = len(note_toks)

        halluc_mask = np.zeros(note_len, dtype=bool)
        for idx in halluc_idx:
            if idx < note_len:
                halluc_mask[idx] = True

        n_h = int(halluc_mask.sum())
        n_c = int((~halluc_mask).sum())

        if n_h == 0:
            print(f"  [4] Sample {si}: no hallucinated tokens in note span, skipping.")
            continue

        print(f"  Tokens: {note_len}  |  Hallucinated: {n_h}  |  Clean: {n_c}")

        ecs, pks, ecs_layers, pks_layers = compute_ecs_pks(model, tokens, t_len, cfg)

        disc = layer_discriminability(pks_layers, ecs_layers, halluc_mask)
        if disc is None:
            print(f"  [4] Sample {si}: layer_discriminability returned None, skipping.")
            continue

        n_layers = pks_layers.shape[0]

        # Accumulate per-metric arrays
        pks_auroc_all.append(disc["pks_auroc"])
        pks_cohens_d_all.append(disc["pks_cohens_d"])
        pks_pearson_r_all.append(disc["pks_pearson_r"])
        ecs_auroc_all.append(disc["ecs_auroc"])
        ecs_cohens_d_all.append(disc["ecs_cohens_d"])
        ecs_pearson_r_all.append(disc["ecs_pearson_r"])
        valid_sample_indices.append(si)

        # ── Save per-example CSVs ────────────────────────────────────────────
        _layers_col = list(range(n_layers))

        pd.DataFrame({
            "layer":     _layers_col,
            "ecs_auroc": [round(float(v), 6) for v in disc["ecs_auroc"]],
            "pks_auroc": [round(float(v), 6) for v in disc["pks_auroc"]],
        }).to_csv(auroc_dir / f"sample_{si:04d}_auroc.csv", index=False)

        pd.DataFrame({
            "layer":        _layers_col,
            "ecs_cohens_d": [round(float(v), 6) for v in disc["ecs_cohens_d"]],
            "pks_cohens_d": [round(float(v), 6) for v in disc["pks_cohens_d"]],
        }).to_csv(cohens_d_dir / f"sample_{si:04d}_cohens_d.csv", index=False)

        pd.DataFrame({
            "layer":         _layers_col,
            "ecs_pearson_r": [round(float(v), 6) for v in disc["ecs_pearson_r"]],
            "pks_pearson_r": [round(float(v), 6) for v in disc["pks_pearson_r"]],
        }).to_csv(pearson_r_dir / f"sample_{si:04d}_pearson_r.csv", index=False)

        print(f"  Saved per-example CSVs → auroc/ | cohens_d/ | pearson_r/")

        # ── Save raw token-level activations ────────────────────────────────
        # ecs_layers / pks_layers : (n_layers, note_len)  float32
        # halluc_mask             : (note_len,)            bool
        # These are required by Exp 3 to fit the logistic regressor without
        # re-running forward passes.
        np.savez_compressed(
            activations_dir / f"sample_{si:04d}_activations.npz",
            ecs_layers  = ecs_layers.astype(np.float32),
            pks_layers  = pks_layers.astype(np.float32),
            halluc_mask = halluc_mask,
            sample_idx  = np.array(si),
        )
        print(f"  Saved activations → activations/sample_{si:04d}_activations.npz"
              f"  ({ecs_layers.nbytes / 1024:.1f} KB)")

        # Summary row
        summary_rows.append({
            "sample_idx":    si,
            "note_len":      note_len,
            "n_halluc":      n_h,
            "n_clean":       n_c,
            "n_injections":  len(injections),
            "mean_pks_auroc": float(disc["pks_auroc"].mean()),
            "mean_ecs_auroc": float(disc["ecs_auroc"].mean()),
            "mean_pks_pearson_r": float(disc["pks_pearson_r"].mean()),
            "mean_ecs_pearson_r": float(disc["ecs_pearson_r"].mean()),
        })

        # Tidy CSV rows (one per layer)
        for l in range(n_layers):
            csv_rows.append({
                "sample_idx":   si,
                "layer":        l,
                "pks_auroc":    round(float(disc["pks_auroc"][l]),    5),
                "pks_cohens_d": round(float(disc["pks_cohens_d"][l]), 5),
                "pks_pearson_r":round(float(disc["pks_pearson_r"][l]),5),
                "ecs_auroc":    round(float(disc["ecs_auroc"][l]),    5),
                "ecs_cohens_d": round(float(disc["ecs_cohens_d"][l]), 5),
                "ecs_pearson_r":round(float(disc["ecs_pearson_r"][l]),5),
            })

        # Progress print
        print(f"  PKS AUROC — mean: {disc['pks_auroc'].mean():.4f}  "
              f"peak: {disc['pks_auroc'].max():.4f} @ layer {disc['pks_auroc'].argmax()}")
        print(f"  ECS AUROC — mean: {disc['ecs_auroc'].mean():.4f}  "
              f"peak: {disc['ecs_auroc'].max():.4f} @ layer {disc['ecs_auroc'].argmax()}")

    n_valid = len(valid_sample_indices)
    if n_valid == 0:
        print("\n  [Exp 4] No valid examples — all failed. Aborting.")
        return {}

    print(f"\n  Valid examples: {n_valid} / {n_examples}")

    # ── 2. Save CSVs ─────────────────────────────────────────────────────────
    pd.DataFrame(csv_rows).to_csv(out / "exp4_layer_stats.csv", index=False)
    print("  Saved → exp4_layer_stats.csv")
    pd.DataFrame(summary_rows).to_csv(out / "exp4_summary.csv", index=False)
    print("  Saved → exp4_summary.csv")

    # ── 3. Stack arrays: (n_valid, n_layers) ─────────────────────────────────
    def _stack(lst):
        return np.stack(lst, axis=0)   # (n_valid, n_layers)

    pks_auroc_mat     = _stack(pks_auroc_all)
    pks_cohens_d_mat  = _stack(pks_cohens_d_all)
    pks_pearson_r_mat = _stack(pks_pearson_r_all)
    ecs_auroc_mat     = _stack(ecs_auroc_all)
    ecs_cohens_d_mat  = _stack(ecs_cohens_d_all)
    ecs_pearson_r_mat = _stack(ecs_pearson_r_all)

    layers = np.arange(n_layers)

    # ── 4. Plot: AUROC, Cohen's d, Pearson r (mean ± std) ───────────────────
    fig, axes = plt.subplots(3, 2, figsize=(max(14, n_layers // 2), 13), sharex=True)
    fig.suptitle(
        f"Exp 4 — Layer-wise ECS/PKS Discriminability  ({cfg.model_name})\n"
        f"Mean ± 1 SD across {n_valid} examples  ({n_injections} injections each)",
        fontsize=12, fontweight="bold",
    )

    _band_specs = [
        # (matrix, ax,         title,                  ylabel,      colour,    hline, hline_label)
        (ecs_auroc_mat,    axes[0, 0], "ECS — AUROC",      "AUROC",     "#2196F3", 0.5, "Chance"),
        (pks_auroc_mat,    axes[0, 1], "PKS — AUROC",      "AUROC",     "#FF9800", 0.5, "Chance"),
        (ecs_cohens_d_mat, axes[1, 0], "ECS — Cohen's d",  "Cohen's d", "#2196F3", 0.0, "Zero"),
        (pks_cohens_d_mat, axes[1, 1], "PKS — Cohen's d",  "Cohen's d", "#FF9800", 0.0, "Zero"),
        (ecs_pearson_r_mat,axes[2, 0], "ECS — Pearson r",  "Pearson r", "#2196F3", 0.0, "r = 0"),
        (pks_pearson_r_mat,axes[2, 1], "PKS — Pearson r",  "Pearson r", "#FF9800", 0.0, "r = 0"),
    ]

    for mat, ax, title, ylabel, colour, hline, hline_label in _band_specs:
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        # Individual example traces (thin, semi-transparent)
        for row in mat:
            ax.plot(layers, row, color=colour, lw=0.6, alpha=0.20)

        # Mean ± std band
        ax.fill_between(layers, mean - std, mean + std,
                        color=colour, alpha=0.22, label="±1 SD")
        ax.plot(layers, mean, color=colour, lw=2.2, label="Mean")

        ax.axhline(hline, color="gray", ls="--", lw=1.0, alpha=0.6, label=hline_label)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls=":", alpha=0.4)

    for ax in axes[2]:
        ax.set_xlabel("Layer", fontsize=10)

    plt.tight_layout()
    stats_path = out / "exp4_layer_stats.png"
    fig.savefig(stats_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp4_layer_stats.png")

    # ── 6. Console summary ────────────────────────────────────────────────────
    print(f"\n  ── Aggregate summary ({n_valid} examples) ──")
    print(f"  {'':6}  {'PKS AUROC':>12}  {'ECS AUROC':>12}  "
          f"{'PKS r':>10}  {'ECS r':>10}")
    print("  " + "─" * 56)
    for l in range(n_layers):
        print(
            f"  layer {l:>2}  "
            f"{pks_auroc_mat[:,l].mean():>6.4f}±{pks_auroc_mat[:,l].std():.3f}  "
            f"{ecs_auroc_mat[:,l].mean():>6.4f}±{ecs_auroc_mat[:,l].std():.3f}  "
            f"{pks_pearson_r_mat[:,l].mean():>+7.4f}  "
            f"{ecs_pearson_r_mat[:,l].mean():>+7.4f}"
        )

    return {
        "pks_auroc_mat":     pks_auroc_mat,
        "pks_cohens_d_mat":  pks_cohens_d_mat,
        "pks_pearson_r_mat": pks_pearson_r_mat,
        "ecs_auroc_mat":     ecs_auroc_mat,
        "ecs_cohens_d_mat":  ecs_cohens_d_mat,
        "ecs_pearson_r_mat": ecs_pearson_r_mat,
        "valid_sample_indices": valid_sample_indices,
        "n_valid":           n_valid,
        "n_layers":          n_layers,
    }


# ─────────────────────────────────────────────
# 15. Entry point
# ─────────────────────────────────────────────
