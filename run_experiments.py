"""
run_experiments.py
==================
Experiment runner for ECS/PKS-based factuality analysis of clinical notes.
Imports all metrics and helpers from ecs_pks.py.

Experiments
-----------
2a  Token-level ECS/PKS mapping on the gold reference SOAP note.
    Outputs: scatter, heatmap, per-token risk bar chart.

2b  Model-generated note: scatter + HTML report with highlighted risk tokens
    + distributional comparison (KDE) against the gold note.
    Outputs: scatter, highlighted HTML, KDE distribution plots, raw note text.

2c  Hallucination screening validation via synthetic injection.
    Injects plausible-but-wrong terms into the gold note, runs ECS/PKS, and
    reports where the known-bad tokens fall in the quadrant space.
    Outputs: scatter with injected tokens starred, per-token CSV, corrupted note.

3   REDEEP hallucination scoring on real generated text.
    Collects multi-sample training data, computes Pearson r per layer,
    identifies Knowledge FFNs (set F) and Copying Head layers (set A),
    fits logistic α/β coefficients, scores the generated note.
    Outputs: training_correlations.csv, selected_layers.txt, tokens.csv,
             scatter_calibrated.png, report.html.

4   Layer-wise discriminability statistics across N examples.
    Runs ECS/PKS with hallucination injection on a range of ACI-Bench
    examples, computes per-layer AUROC/Cohen's d/Pearson r for each, and
    plots mean ± std band (AUROC, Cohen's d) and per-example spaghetti
    lines (Pearson r) to show which layers are consistently discriminative.
    Outputs: exp4_layer_stats.png, exp4_layer_stats.csv, exp4_summary.csv.

Usage
-----
    python run_experiments.py                              # 2a + 2b, gemma, sample 0
    python run_experiments.py --exp 2c                    # only 2c
    python run_experiments.py --exp 3                     # Exp 3 (REDEEP scoring)
    python run_experiments.py --exp 3 --n-train-samples 5 --n-ffn-layers 3
    python run_experiments.py --exp all                   # 2a + 2b + 2c + 2d + 3
    python run_experiments.py --model llama               # Llama 3 8B
    python run_experiments.py --sample 3                  # ACI-Bench row 3
    python run_experiments.py --max-new-tokens 300
"""

import argparse
import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformer_lens import HookedTransformer

from ecs_pks import (
    Config,
    Q_COLORS,
    apply_layer_thresholds,
    calibrate_layer_thresholds,
    compute_dla,
    compute_ecs_pks,
    dla_discriminability,
    fit_hallucination_regressor,
    generate_note,
    hallucination_risk,
    identify_copy_head_layers,
    identify_knowledge_ffns,
    layer_discriminability,
    load_aci_sample,
    plot_heatmap,
    plot_layer_discriminability,
    plot_risk_bar,
    plot_scatter,
    quadrant_stats,
    tokenize_as_generated,
    tokenize_pair,
)

warnings.filterwarnings("ignore")

# Optional LLM-based hallucination generator (requires huggingface_hub or boto3)
try:
    from halluc_llm import HallucinationGenerationError, inject_hallucinations_llm, inject_hallucinations, halluc_token_indices
    _LLM_HALLUC_AVAILABLE = True
except ImportError:
    _LLM_HALLUC_AVAILABLE = False


# ─────────────────────────────────────────────
# 8. Experiment 2b helpers
# ─────────────────────────────────────────────

def plot_distribution_comparison(
    ecs_gold: np.ndarray,
    pks_gold: np.ndarray,
    ecs_gen: np.ndarray,
    pks_gen: np.ndarray,
    out_path: Path,
    model_name: str,
) -> None:
    """
    Compare the *distributions* of ECS and PKS between the gold and generated
    notes via KDE.

    Per-token positional subtraction is invalid when the two notes have
    different structures (both may be clinically correct but worded differently).
    Distributional comparison requires no alignment and is always valid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, vals_g, vals_m, label in [
        (axes[0], ecs_gold, ecs_gen, "ECS"),
        (axes[1], pks_gold, pks_gen, "PKS"),
    ]:
        sns.kdeplot(vals_g, ax=ax, label="Gold reference",
                    color="#4CAF50", fill=True, alpha=0.25, linewidth=1.8)
        sns.kdeplot(vals_m, ax=ax, label="Model generated",
                    color="#F44336", fill=True, alpha=0.25, linewidth=1.8)
        ax.axvline(np.median(vals_g), color="#4CAF50", ls="--", lw=1.2, alpha=0.8)
        ax.axvline(np.median(vals_m), color="#F44336", ls="--", lw=1.2, alpha=0.8)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{label} distribution: Gold vs Generated",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle(f"Exp 2b — Distributional Comparison  ({model_name})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_html_report(
    scatter_png: Path,
    tokens: List[str],
    ecs: np.ndarray,
    pks: np.ndarray,
    generated_note: str,
    model_name: str,
    sample_idx: int,
    out_path: Path,
) -> None:
    """
    Self-contained HTML report containing:
      1. ECS/PKS scatter (embedded as base64 PNG).
      2. Full generated note with hallucination-risk tokens highlighted.

    Risk tokens = those below the median on BOTH ECS and PKS (Low-Low quadrant).
    Token spacing is reconstructed from SentencePiece / BPE leading-space markers.
    """
    import base64

    with open(scatter_png, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    em        = float(np.median(ecs))
    pm        = float(np.median(pks))
    risk_mask = (ecs < em) & (pks < pm)

    html_parts: List[str] = []
    for tok, is_risky in zip(tokens, risk_mask):
        space = " " if (tok.startswith("▁") or tok.startswith("Ġ")) else ""
        word  = tok[1:] if space else tok
        word_esc = (word
                    .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace("\n", "<br>\n"))
        if is_risky:
            html_parts.append(f'{space}<mark class="risk">{word_esc}</mark>')
        else:
            html_parts.append(f"{space}{word_esc}")
    note_html = "".join(html_parts)

    n         = len(ecs)
    hi_ecs    = ecs >= em
    hi_pks    = pks >= pm
    pct_extr  = 100 * float(np.mean( hi_ecs & ~hi_pks))
    pct_param = 100 * float(np.mean(~hi_ecs &  hi_pks))
    pct_synth = 100 * float(np.mean( hi_ecs &  hi_pks))
    pct_risk  = 100 * float(np.mean(~hi_ecs & ~hi_pks))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exp 2b — Hallucination Risk Report</title>
  <style>
    body      {{ font-family: Georgia, serif; max-width: 960px;
                margin: 40px auto; padding: 0 24px; color: #222; }}
    h2        {{ border-bottom: 2px solid #eee; padding-bottom: 8px; color: #333; }}
    h3        {{ color: #555; margin-top: 32px; }}
    img       {{ max-width: 100%; border: 1px solid #ddd;
                border-radius: 6px; margin: 12px 0; }}
    .meta     {{ font-size: 13px; color: #777; margin-bottom: 24px; }}
    .legend   {{ display: flex; gap: 20px; flex-wrap: wrap;
                margin: 12px 0 20px; font-size: 13px; }}
    .leg-item {{ display: flex; align-items: center; gap: 8px; }}
    .swatch   {{ width: 18px; height: 18px; border-radius: 3px;
                border: 1px solid rgba(0,0,0,.15); flex-shrink: 0; }}
    .note-box {{ background: #fafafa; border: 1px solid #ddd;
                border-radius: 6px; padding: 22px 26px;
                line-height: 2.0; font-size: 14px; white-space: pre-wrap; }}
    mark.risk {{ background: #FFCDD2; color: #B71C1C;
                border-radius: 3px; padding: 1px 3px; font-style: normal; }}
    table     {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td    {{ border: 1px solid #ddd; padding: 7px 12px; text-align: left; }}
    th        {{ background: #f5f5f5; }}
    .risk-row {{ background: #FFEBEE; font-weight: bold; }}
  </style>
</head>
<body>
<h2>Experiment 2b — ECS/PKS Hallucination Risk Report</h2>
<p class="meta">
  Model: <strong>{model_name}</strong> &nbsp;|&nbsp;
  ACI-Bench sample: <strong>{sample_idx}</strong> &nbsp;|&nbsp;
  Note tokens: <strong>{n}</strong><br>
  ECS threshold (median): <strong>{em:.4f}</strong> &nbsp;|&nbsp;
  PKS threshold (median): <strong>{pm:.4f}</strong>
</p>

<h3>ECS vs PKS Scatter — Model-Generated Note</h3>
<img src="data:image/png;base64,{img_b64}" alt="ECS/PKS scatter">

<h3>Quadrant Distribution</h3>
<table>
  <tr><th>Quadrant</th><th>Condition</th><th>% of tokens</th></tr>
  <tr><td>Extractive</td>
      <td>High ECS, Low PKS — copied from transcript</td>
      <td>{pct_extr:.1f}%</td></tr>
  <tr><td>Parametric</td>
      <td>Low ECS, High PKS — drawn from medical knowledge</td>
      <td>{pct_param:.1f}%</td></tr>
  <tr><td>Synthesized</td>
      <td>High ECS, High PKS — grounded reasoning</td>
      <td>{pct_synth:.1f}%</td></tr>
  <tr class="risk-row"><td>Hallucination Risk</td>
      <td>Low ECS, Low PKS — grounded in neither source</td>
      <td>{pct_risk:.1f}%</td></tr>
</table>

<h3>Generated Note — Highlighted Tokens</h3>
<div class="legend">
  <div class="leg-item">
    <div class="swatch" style="background:#FFCDD2;"></div>
    <span>Hallucination risk (Low ECS + Low PKS)</span>
  </div>
  <div class="leg-item">
    <div class="swatch" style="background:#fff;"></div>
    <span>Extractive, parametric, or synthesized</span>
  </div>
</div>
<div class="note-box">{note_html}</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


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

def _gold_coverage_labels(
    generated_note: str,
    gold_note: str,
    note_toks: List[str],
    ngram: int = 3,
) -> np.ndarray:
    """
    Token-level pseudo-labels derived from n-gram coverage of the gold note.

    For each token position, reconstruct overlapping n-grams of decoded text
    centred on that token.  If none of the n-grams appear in the (normalised)
    gold note, the token is flagged as potentially hallucinated (label = 1).

    Returns
    -------
    labels : (note_len,) float in {0.0, 1.0}
    """
    import re as _re

    def _normalise(text: str) -> str:
        return _re.sub(r"\s+", " ", text.lower().strip())

    gold_norm = _normalise(gold_note)
    # Pre-tokenise gold into word n-grams for fast lookup
    gold_words = gold_norm.split()
    gold_ngrams: set = set()
    for i in range(len(gold_words) - ngram + 1):
        gold_ngrams.add(" ".join(gold_words[i: i + ngram]))

    # Decode note tokens to plain words
    cleaned = [t.replace("▁", " ").replace("Ġ", " ").strip() for t in note_toks]
    n = len(cleaned)
    labels = np.ones(n, dtype=np.float64)   # default: not covered

    for i in range(n):
        # Build n-gram window centred roughly on position i
        start = max(0, i - ngram + 1)
        end   = min(n, i + ngram)
        window = " ".join(cleaned[start:end])
        window_norm = _normalise(window)
        window_words = window_norm.split()
        for j in range(len(window_words) - ngram + 1):
            cand = " ".join(window_words[j: j + ngram])
            if cand in gold_ngrams:
                labels[i] = 0.0   # covered by gold → not flagged
                break

    return labels


def _nli_sentence_labels(
    transcript: str,
    generated_note: str,
    note_toks: List[str],
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> Optional[np.ndarray]:
    """
    Sentence-level NLI pseudo-labels mapped to token positions.

    For each sentence in the generated note, queries an NLI cross-encoder to
    compute P(entailment | transcript, sentence).  Low entailment probability
    → the sentence makes a claim unsupported by the transcript → tokens in that
    sentence get label ≈ 1.

    Requires:  pip install sentence-transformers

    Returns
    -------
    labels : (note_len,) float in [0, 1] — 1 = likely hallucinated.
             Returns None if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        warnings.warn(
            "[Exp 3] sentence-transformers not installed — skipping NLI validation. "
            "Install with:  pip install sentence-transformers"
        )
        return None

    import re as _re

    # Split generated note into sentences (simple split on . ? !)
    sentence_spans: List[Tuple[int, int]] = []   # (char_start, char_end)
    sentences: List[str] = []
    for m in _re.finditer(r"[^.!?\n]+[.!?\n]?", generated_note):
        s = m.group(0).strip()
        if len(s) > 10:
            sentence_spans.append((m.start(), m.end()))
            sentences.append(s)

    if not sentences:
        return None

    print(f"  [NLI] Loading {nli_model_name} …")
    nli = CrossEncoder(nli_model_name)

    # NLI: premise = transcript, hypothesis = each generated sentence
    pairs   = [(transcript[:2000], sent) for sent in sentences]   # truncate transcript
    logits  = nli.predict(pairs, apply_softmax=True)               # (S, 3): contra/neut/entail
    entail_prob = logits[:, 2]                                      # entailment column

    # Map sentence-level score to token-level via character offsets
    n = len(note_toks)
    token_labels = np.zeros(n, dtype=np.float64)

    # Build cumulative char position for each token
    cursor = 0
    tok_char_starts = []
    for tok in note_toks:
        piece = tok.replace("▁", " ").replace("Ġ", " ")
        tok_char_starts.append(cursor)
        cursor += len(piece)

    for (cs, ce), ep in zip(sentence_spans, entail_prob):
        halluc_score = float(1.0 - ep)
        for i, tc in enumerate(tok_char_starts):
            if cs <= tc < ce:
                token_labels[i] = halluc_score

    return token_labels


def _build_exp3_html(
    scatter_png: Path,
    note_toks: List[str],
    halluc_prob: np.ndarray,
    ecs: np.ndarray,
    pks: np.ndarray,
    gold_labels: Optional[np.ndarray],
    nli_labels: Optional[np.ndarray],
    calib_summary: Dict,
    generated_note: str,
    model_name: str,
    sample_idx: int,
    threshold: float,
    out_path: Path,
) -> None:
    """
    Self-contained HTML report for Experiment 3.

    Sections
    --------
    1. Calibration summary table (selected layers, J-stat, AUROC, threshold).
    2. ECS/PKS scatter of the generated note, points coloured by halluc_prob.
    3. Validation metrics (gold coverage, NLI) if available.
    4. Generated note with gradient highlighting (white → red ∝ halluc_prob).
    """
    import base64

    with open(scatter_png, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    # ── Token HTML ────────────────────────────────────────────────────────────
    def _prob_to_style(p: float) -> str:
        if p < 0.15:
            return ""
        # Interpolate white → #FFCDD2 (light red) → #F44336 (red)
        g = int(255 - (255 - 67)  * p)
        b = int(255 - (255 - 54)  * p)
        r = 255
        return f"background:rgb({r},{g},{b});border-radius:3px;padding:1px 2px;"

    html_toks: List[str] = []
    for tok, p in zip(note_toks, halluc_prob):
        space = " " if (tok.startswith("▁") or tok.startswith("Ġ")) else ""
        word  = tok[1:] if space else tok
        word_esc = (word
                    .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace("\n", "<br>"))
        style = _prob_to_style(float(p))
        title = f'title="p={p:.3f}"'
        if style:
            html_toks.append(f'{space}<span style="{style}" {title}>{word_esc}</span>')
        else:
            html_toks.append(f"{space}{word_esc}")
    note_html = "".join(html_toks)

    # ── Calibration table rows ────────────────────────────────────────────────
    cal_rows = ""
    for metric, layers_info in calib_summary.items():
        if not layers_info:
            continue
        for layer, info in sorted(layers_info.items(), key=lambda x: -x[1]["j_stat"]):
            dir_sym = "↓ low" if info["direction"] == -1 else "↑ high"
            cal_rows += (
                f"<tr><td>{metric}</td><td>{layer}</td>"
                f"<td>{info['auroc']:.3f}</td><td>{info['j_stat']:.3f}</td>"
                f"<td>{info['threshold']:.4f}</td><td>{dir_sym}</td></tr>\n"
            )

    # ── Validation rows ────────────────────────────────────────────────────────
    val_rows = ""
    flagged  = halluc_prob >= threshold
    if gold_labels is not None:
        from sklearn.metrics import roc_auc_score as _auc
        try:
            auc_gold = _auc(gold_labels, halluc_prob)
            prec = float(np.mean(gold_labels[flagged])) if flagged.any() else float("nan")
            recall = float(np.sum(flagged & (gold_labels > 0.5)) / (gold_labels > 0.5).sum()) if (gold_labels > 0.5).any() else float("nan")
            val_rows += (
                f"<tr><td>Gold n-gram coverage</td>"
                f"<td>{auc_gold:.3f}</td><td>{prec:.3f}</td><td>{recall:.3f}</td></tr>"
            )
        except Exception:
            pass
    if nli_labels is not None:
        from sklearn.metrics import roc_auc_score as _auc
        try:
            auc_nli = _auc(nli_labels > 0.5, halluc_prob)
            prec = float(np.mean((nli_labels > 0.5)[flagged])) if flagged.any() else float("nan")
            val_rows += (
                f"<tr><td>NLI (low entailment)</td>"
                f"<td>{auc_nli:.3f}</td><td>{prec:.3f}</td><td>—</td></tr>"
            )
        except Exception:
            pass

    n_flagged = int(flagged.sum())
    n_total   = len(halluc_prob)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exp 3 — Calibrated Hallucination Flagging</title>
  <style>
    body      {{ font-family: Georgia, serif; max-width: 1000px;
                margin: 40px auto; padding: 0 24px; color: #222; }}
    h2        {{ border-bottom: 2px solid #eee; padding-bottom: 8px; }}
    h3        {{ color: #555; margin-top: 32px; }}
    img       {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 12px 0; }}
    .meta     {{ font-size: 13px; color: #777; margin-bottom: 24px; }}
    .note-box {{ background: #fafafa; border: 1px solid #ddd; border-radius: 6px;
                padding: 22px 26px; line-height: 2.2; font-size: 14px;
                white-space: pre-wrap; word-break: break-word; }}
    table     {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }}
    th, td    {{ border: 1px solid #ddd; padding: 7px 12px; text-align: left; }}
    th        {{ background: #f5f5f5; }}
    .legend   {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0;
                font-size: 13px; align-items: center; }}
    .grad     {{ width: 180px; height: 16px; border-radius: 4px;
                background: linear-gradient(to right, #ffffff, #f44336);
                border: 1px solid #ddd; }}
  </style>
</head>
<body>
<h2>Experiment 3 — Calibrated Token-Level Hallucination Flagging</h2>
<p class="meta">
  Model: <strong>{model_name}</strong> &nbsp;|&nbsp;
  Sample: <strong>{sample_idx}</strong> &nbsp;|&nbsp;
  Tokens: <strong>{n_total}</strong> &nbsp;|&nbsp;
  Flagged (prob ≥ {threshold:.2f}): <strong>{n_flagged}</strong>
  ({100*n_flagged/n_total:.1f}%)
</p>

<h3>ECS vs PKS — Generated Note (coloured by hallucination probability)</h3>
<img src="data:image/png;base64,{img_b64}" alt="Exp 3 scatter">

<h3>Calibration — Selected Layers from Exp 2c</h3>
<table>
  <tr><th>Metric</th><th>Layer</th><th>AUROC</th><th>Youden J</th>
      <th>Threshold</th><th>Flag when</th></tr>
  {cal_rows}
</table>

{"<h3>Validation</h3><table><tr><th>Signal</th><th>AUROC vs halluc_prob</th><th>Precision@flagged</th><th>Recall@flagged</th></tr>" + val_rows + "</table>" if val_rows else ""}

<h3>Generated Note — Hallucination Probability Highlights</h3>
<div class="legend">
  <span>Low risk</span>
  <div class="grad"></div>
  <span>High risk</span>
  &nbsp; (hover token for exact probability)
</div>
<div class="note-box">{note_html}</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


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

    n_h_total = int(halluc_mask_all.sum())
    n_c_total = int((~halluc_mask_all).sum())
    print(f"\n  Training set: {len(ecs_layers_list)} samples  |  "
          f"{halluc_mask_all.shape[0]} tokens total  "
          f"({n_h_total} hallucinated / {n_c_total} clean)")

    return {
        "ecs_layers_all":  ecs_layers_all,
        "pks_layers_all":  pks_layers_all,
        "halluc_mask_all": halluc_mask_all,
        "n_samples":       len(ecs_layers_list),
        "sample_stats":    sample_stats,
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
    nli_model: Optional[str] = None,
) -> Dict:
    """
    Experiment 3: REDEEP hallucination scoring on a real generated note.

    Implements the paper formula:
        Ht(t) = Σ_{l∈F} α·P^l_t  −  Σ_{l∈A} β·E^{l,h}_t

    where α, β > 0 are learned via logistic regression on a multi-sample
    training set with synthetic hallucination injection.

    Pipeline
    --------
    1. Collect training data — run ECS/PKS with hallucination injection on
       `n_train_samples` ACI-Bench samples (not the target sample).
    2. Layer-wise Pearson r — compute point-biserial correlation between
       per-layer ECS/PKS and the binary hallucination labels.
    3. Identify set F (Knowledge FFNs) — top-`n_ffn_layers` by PKS Pearson r.
    4. Identify set A (Copying Head layers) — top-`n_copy_layers` by most
       negative ECS Pearson r.
    5. Fit α, β — logistic regression on [PKS_F | ECS_A] features with
       class_weight="balanced" to handle heavy class imbalance.
    6. Score the generated note — compute halluc_prob via clf.predict_proba.
    7. Validate — gold n-gram coverage + optional NLI cross-encoder.
    8. Outputs — CSV, scatter, HTML report.

    Parameters
    ----------
    n_train_samples  : number of ACI-Bench samples to collect training data from.
                       Samples are drawn from the same split, skipping the
                       target sample index.
    n_injections     : max hallucinations injected per training sample.
    n_ffn_layers     : size of set F (Knowledge FFN layers).
    n_copy_layers    : size of set A (Copying Head layers).
    halluc_threshold : probability cutoff for binary flag in HTML/CSV.
    nli_model        : HuggingFace NLI cross-encoder ID (optional).

    Outputs
    -------
    exp3_training_correlations.csv — per-layer Pearson r, AUROC, Cohen's d
    exp3_training_summary.csv      — per training sample stats
    exp3_selected_layers.txt       — F layers, A layers, α, β
    exp3_tokens.csv                — per-token halluc_prob, ecs, pks, flag
    exp3_scatter_calibrated.png    — ECS/PKS scatter coloured by halluc_prob
    exp3_report.html               — gradient-highlighted generated note
    """
    print("\n" + "═"*54)
    print("  EXPERIMENT 3 — REDEEP Hallucination Scoring")
    print("═"*54)

    _inject = inject_fn or inject_hallucinations

    # ── 1. Collect training data ─────────────────────────────────────────────
    print(f"\n  Step 1/8 — Collecting training data "
          f"({n_train_samples} samples, {n_injections} injections each) …")

    # Use samples adjacent to the target (skip cfg.sample_idx)
    dataset_size = 250   # ACI-Bench test1 has ~250 rows; safe upper bound
    all_indices  = [i for i in range(dataset_size) if i != cfg.sample_idx]
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
        ecs_layers_all  = calib_2c["ecs_layers"]
        pks_layers_all  = calib_2c["pks_layers"]
        note_len_cal    = ecs_layers_all.shape[1]
        halluc_mask_all = np.array(
            [i in set(calib_2c["halluc_idx"]) for i in range(note_len_cal)],
            dtype=bool,
        )
        train_data = {
            "ecs_layers_all":  ecs_layers_all,
            "pks_layers_all":  pks_layers_all,
            "halluc_mask_all": halluc_mask_all,
            "n_samples":       1,
            "sample_stats":    [],
        }

    ecs_layers_all  = train_data["ecs_layers_all"]
    pks_layers_all  = train_data["pks_layers_all"]
    halluc_mask_all = train_data["halluc_mask_all"]
    n_layers        = ecs_layers_all.shape[0]
    n_h_total       = int(halluc_mask_all.sum())
    n_c_total       = int((~halluc_mask_all).sum())

    # Save training summary CSV
    if train_data["sample_stats"]:
        pd.DataFrame(train_data["sample_stats"]).to_csv(
            out / "exp3_training_summary.csv", index=False
        )
        print("  Saved → exp3_training_summary.csv")

    # ── 2. Layer-wise Pearson r (and AUROC/Cohen's d) ────────────────────────
    print(f"\n  Step 2/8 — Computing layer-wise discriminability "
          f"({n_layers} layers, {n_h_total} hallucinated / {n_c_total} clean tokens) …")

    disc = layer_discriminability(pks_layers_all, ecs_layers_all, halluc_mask_all)
    if disc is None:
        raise RuntimeError(
            "Exp 3: layer_discriminability returned None — "
            "both classes must be present in training data."
        )

    pks_pearson_r = disc["pks_pearson_r"]
    ecs_pearson_r = disc["ecs_pearson_r"]

    # Save correlations CSV
    corr_df = pd.DataFrame({
        "layer":         np.arange(n_layers),
        "pks_pearson_r": pks_pearson_r,
        "ecs_pearson_r": ecs_pearson_r,
        "pks_auroc":     disc["pks_auroc"],
        "ecs_auroc":     disc["ecs_auroc"],
        "pks_cohens_d":  disc["pks_cohens_d"],
        "ecs_cohens_d":  disc["ecs_cohens_d"],
    })
    corr_df.to_csv(out / "exp3_training_correlations.csv", index=False)
    print("  Saved → exp3_training_correlations.csv")

    # Print top-5 by Pearson r for each metric
    print(f"\n  ── Top layers by PKS Pearson r (most positive → set F) ──")
    top_pks = np.argsort(pks_pearson_r)[::-1][:5]
    for l in top_pks:
        print(f"    layer {l:>3}  r={pks_pearson_r[l]:+.4f}  "
              f"AUROC={disc['pks_auroc'][l]:.4f}  d={disc['pks_cohens_d'][l]:+.3f}")

    print(f"\n  ── Top layers by ECS Pearson r (most negative → set A) ──")
    top_ecs = np.argsort(ecs_pearson_r)[:5]
    for l in top_ecs:
        print(f"    layer {l:>3}  r={ecs_pearson_r[l]:+.4f}  "
              f"AUROC={disc['ecs_auroc'][l]:.4f}  d={disc['ecs_cohens_d'][l]:+.3f}")

    # ── 3. Identify set F (Knowledge FFNs) ───────────────────────────────────
    print(f"\n  Step 3/8 — Identifying set F (top-{n_ffn_layers} Knowledge FFN layers) …")
    F = identify_knowledge_ffns(pks_pearson_r, top_k=n_ffn_layers)
    print(f"  Set F = {F}  (r values: {[round(float(pks_pearson_r[l]),4) for l in F]})")

    # ── 4. Identify set A (Copying Head layers) ───────────────────────────────
    print(f"\n  Step 4/8 — Identifying set A (top-{n_copy_layers} Copying Head layers) …")
    A = identify_copy_head_layers(ecs_pearson_r, top_k=n_copy_layers)
    print(f"  Set A = {A}  (r values: {[round(float(ecs_pearson_r[l]),4) for l in A]})")

    # ── 5. Fit logistic regression (α, β) ────────────────────────────────────
    print(f"\n  Step 5/8 — Fitting REDEEP logistic regressor (|F|={len(F)}, |A|={len(A)}) …")
    clf, scaler, alpha, beta = fit_hallucination_regressor(
        pks_layers_all, ecs_layers_all, halluc_mask_all, F, A
    )
    print(f"  α (PKS weight) = {alpha:.6f}")
    print(f"  β (ECS weight) = {beta:.6f}")

    # Training-set AUROC
    try:
        from sklearn.metrics import roc_auc_score as _auc
        pks_feats_tr = pks_layers_all[F].T
        ecs_feats_tr = ecs_layers_all[A].T
        X_tr = np.concatenate([pks_feats_tr, ecs_feats_tr], axis=1)
        X_tr_sc = scaler.transform(X_tr)
        prob_tr  = clf.predict_proba(X_tr_sc)[:, 1]
        auc_tr   = _auc(halluc_mask_all.astype(int), prob_tr)
        print(f"  Training AUROC = {auc_tr:.4f}")
    except Exception as exc:
        print(f"  [Exp 3] Training AUROC skipped: {exc}")

    # Save selected layers info
    sel_txt = (
        f"Set F (Knowledge FFNs, top-{n_ffn_layers} by PKS Pearson r): {F}\n"
        f"Set A (Copying Heads,  top-{n_copy_layers} by ECS Pearson r): {A}\n"
        f"alpha (PKS coefficient): {alpha:.8f}\n"
        f"beta  (ECS coefficient): {beta:.8f}\n"
        f"\nPKS Pearson r at F layers: {[round(float(pks_pearson_r[l]),6) for l in F]}\n"
        f"ECS Pearson r at A layers: {[round(float(ecs_pearson_r[l]),6) for l in A]}\n"
        f"\nTraining samples: {train_data['n_samples']}  |  "
        f"Tokens: {halluc_mask_all.shape[0]}  "
        f"({n_h_total} hallucinated / {n_c_total} clean)\n"
    )
    (out / "exp3_selected_layers.txt").write_text(sel_txt, encoding="utf-8")
    print("  Saved → exp3_selected_layers.txt")

    # ── 6. Score the generated note ───────────────────────────────────────────
    print(f"\n  Step 6/8 — Scoring the generated note …")

    gen_txt_path = out / "exp2b_generated_note.txt"
    if gen_txt_path.exists():
        generated_note = gen_txt_path.read_text(encoding="utf-8")
        print(f"  Loaded generated note from {gen_txt_path}  "
              f"({len(generated_note)} chars)")
    else:
        print("  Generating note (no cached exp2b output found) …")
        generated_note = generate_note(model, transcript, cfg)
        gen_txt_path.write_text(generated_note, encoding="utf-8")

    tokens, t_len, note_toks = tokenize_as_generated(model, transcript, generated_note)
    tokens   = tokens.to(cfg.device)
    note_len = len(note_toks)
    print(f"  Prompt  : {t_len} tokens  |  Note: {note_len} tokens")

    ecs, pks, ecs_layers, pks_layers = compute_ecs_pks(model, tokens, t_len, cfg)

    # Build feature matrix for generated note
    pks_feats_gen = pks_layers[F].T   # (note_len, |F|)
    ecs_feats_gen = ecs_layers[A].T   # (note_len, |A|)
    X_gen   = np.concatenate([pks_feats_gen, ecs_feats_gen], axis=1)
    X_gen_sc = scaler.transform(X_gen)
    halluc_prob = clf.predict_proba(X_gen_sc)[:, 1]   # (note_len,)
    halluc_prob = np.clip(halluc_prob, 0.0, 1.0)
    flagged     = halluc_prob >= halluc_threshold

    print(f"  Tokens flagged (prob ≥ {halluc_threshold}): "
          f"{int(flagged.sum())} / {note_len}  ({100*flagged.mean():.1f}%)")
    print(f"  Mean halluc_prob : {halluc_prob.mean():.4f}  "
          f"Median: {np.median(halluc_prob):.4f}")

    # ── 7. Validation signals ─────────────────────────────────────────────────
    print(f"\n  Step 7/8 — Validation …")
    gold_labels = _gold_coverage_labels(generated_note, gold_note, note_toks, ngram=3)
    n_uncovered = int((gold_labels > 0.5).sum())
    print(f"  Gold n-gram coverage: {n_uncovered} / {note_len} tokens not in gold note "
          f"({100*n_uncovered/note_len:.1f}%)")

    nli_labels: Optional[np.ndarray] = None
    if nli_model:
        print(f"  NLI validation with {nli_model} …")
        nli_labels = _nli_sentence_labels(transcript, generated_note, note_toks, nli_model)
        if nli_labels is not None:
            print(f"  NLI: {int((nli_labels > 0.5).sum())} / {note_len} tokens in "
                  f"low-entailment sentences")

    try:
        from sklearn.metrics import roc_auc_score as _auc
        print(f"\n  ── Validation AUROC (halluc_prob vs pseudo-labels) ──")
        auc_gold = _auc(gold_labels, halluc_prob)
        print(f"  Gold n-gram coverage   AUROC = {auc_gold:.4f}")
        if nli_labels is not None:
            auc_nli = _auc((nli_labels > 0.5).astype(int), halluc_prob)
            print(f"  NLI low-entailment     AUROC = {auc_nli:.4f}")
    except Exception as exc:
        print(f"  [validation] AUROC could not be computed: {exc}")

    # ── 8. Outputs ────────────────────────────────────────────────────────────
    print(f"\n  Step 8/8 — Writing outputs …")

    # Per-token CSV
    rows = [
        {
            "token_idx":   i,
            "token":       note_toks[i].replace("▁", "").replace("Ġ", "").strip(),
            "ecs":         round(float(ecs[i]),         4),
            "pks":         round(float(pks[i]),         4),
            "halluc_prob": round(float(halluc_prob[i]), 4),
            "flagged":     int(flagged[i]),
            "gold_label":  round(float(gold_labels[i]), 4),
            "nli_label":   round(float(nli_labels[i]),  4) if nli_labels is not None else None,
        }
        for i in range(note_len)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp3_tokens.csv", index=False)
    print("  Saved → exp3_tokens.csv")

    # Scatter: ECS vs PKS coloured by halluc_prob
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        ecs, pks,
        c=halluc_prob,
        cmap="RdYlGn_r",
        vmin=0.0, vmax=1.0,
        s=55, alpha=0.80, edgecolors="white", linewidth=0.4,
    )
    plt.colorbar(sc, ax=ax, label="Hallucination probability  (REDEEP logistic)", shrink=0.75)

    top_idx = np.argsort(halluc_prob)[-10:]
    ax.scatter(ecs[top_idx], pks[top_idx], s=180, c="black", marker="*",
               zorder=6, label="Top-10 risk tokens")
    for i in top_idx:
        lbl = note_toks[i].replace("▁", "").replace("Ġ", "").strip()[:14]
        ax.annotate(lbl, (ecs[i], pks[i]),
                    fontsize=6.5, color="#B71C1C", fontweight="bold",
                    xytext=(6, 6), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#B71C1C", lw=0.7))

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
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    scatter_path = out / "exp3_scatter_calibrated.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp3_scatter_calibrated.png")

    # HTML report (reuse existing helper)
    calib_summary = {
        "F (Knowledge FFNs)":   {l: {"pearson_r": float(pks_pearson_r[l])} for l in F},
        "A (Copying Heads)":    {l: {"pearson_r": float(ecs_pearson_r[l])} for l in A},
    }
    _build_exp3_html(
        scatter_png=scatter_path,
        note_toks=note_toks,
        halluc_prob=halluc_prob,
        ecs=ecs,
        pks=pks,
        gold_labels=gold_labels,
        nli_labels=nli_labels,
        calib_summary=calib_summary,
        generated_note=generated_note,
        model_name=cfg.model_name,
        sample_idx=cfg.sample_idx,
        threshold=halluc_threshold,
        out_path=out / "exp3_report.html",
    )
    print("  Saved → exp3_report.html")

    return {
        "halluc_prob":      halluc_prob,
        "flagged":          flagged,
        "ecs":              ecs,
        "pks":              pks,
        "ecs_layers":       ecs_layers,
        "pks_layers":       pks_layers,
        "F":                F,
        "A":                A,
        "alpha":            alpha,
        "beta":             beta,
        "pks_pearson_r":    pks_pearson_r,
        "ecs_pearson_r":    ecs_pearson_r,
        "gold_labels":      gold_labels,
        "nli_labels":       nli_labels,
        "df":               df,
        "generated_note":   generated_note,
        "clf":              clf,
        "scaler":           scaler,
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

def _resolve_inject_fn(args):
    """
    Return the injection callable to use for Experiment 2c.

    Priority:
      1. --halluc-backend regex  → always use the built-in regex injector
      2. --halluc-backend hf/bedrock  → use inject_hallucinations_llm if the
         halluc_llm module loaded successfully, else warn and fall back to regex
    """
    backend = args.halluc_backend

    if backend == "regex":
        print("  [2c] Hallucination backend: regex (rule-based)")
        return inject_hallucinations   # defined in this file

    if not _LLM_HALLUC_AVAILABLE:
        warnings.warn(
            f"[2c] --halluc-backend={backend} requested but halluc_llm could not "
            f"be imported (missing huggingface_hub or boto3).  "
            f"Falling back to regex-based injection."
        )
        return inject_hallucinations

    # Build kwargs for inject_hallucinations_llm
    llm_kwargs: Dict = {"backend": backend}
    if backend == "hf" and args.hf_model:
        llm_kwargs["hf_model"] = args.hf_model
    if backend == "bedrock":
        if args.bedrock_model:
            llm_kwargs["bedrock_model"] = args.bedrock_model
        if args.bedrock_region:
            llm_kwargs["bedrock_region"] = args.bedrock_region

    def _llm_inject(note: str, max_injections: int = 10, seed: int = 42):
        """Thin wrapper so inject_hallucinations_llm matches the (note, max, seed) signature."""
        try:
            return inject_hallucinations_llm(note, max_injections=max_injections, **llm_kwargs)
        except HallucinationGenerationError as exc:
            warnings.warn(
                f"[2c] LLM injection failed ({exc}).  "
                f"Falling back to regex-based injection."
            )
            return inject_hallucinations(note, max_injections=max_injections, seed=seed)

    model_id = llm_kwargs.get("hf_model") or llm_kwargs.get("bedrock_model") or "(default)"
    print(f"  [2c] Hallucination backend: {backend}  model: {model_id}")
    return _llm_inject


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REDEEP ECS/PKS — clinical note experiments")
    p.add_argument("--model", choices=["gemma", "llama"], default="gemma",
                   help="gemma → google/gemma-2-2b-it  |  llama → meta-llama/Meta-Llama-3-8B-instruct")
    p.add_argument("--exp", choices=["2a", "2b", "2c", "2d", "3", "4", "both", "all"], default="both",
                   help="Which experiment(s) to run  (both=2a+2b [default]  |  all=2a+2b+2c+2d+3+4)")
    p.add_argument("--sample", type=int, default=0,
                   help="Row index from ACI-Bench test1 split (default: 0)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens for generated note in Exp 2b (default: 512)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for generation (0.0 = greedy)")
    p.add_argument("--halluc-backend", choices=["regex", "hf", "bedrock"], default="bedrock",
                   help="Hallucination injection method for Exp 2c  "
                        "(regex=rule-based fallback  |  hf=HuggingFace API  |  bedrock=AWS Bedrock)")
    p.add_argument("--hf-model", default=None,
                   help="HuggingFace model ID for --halluc-backend hf  "
                        "(default: Qwen/Qwen2.5-72B-Instruct)")
    p.add_argument("--bedrock-model", default=None,
                   help="Bedrock model ID for --halluc-backend bedrock  "
                        "(default: anthropic.claude-3-haiku-20240307-v1:0)")
    p.add_argument("--bedrock-region", default=None,
                   help="AWS region for --halluc-backend bedrock  "
                        "(default: AWS_DEFAULT_REGION env var or us-east-1)")
    p.add_argument("--out", default=".", help="Output directory for plots and CSV")
    p.add_argument("--n-train-samples", type=int, default=3,
                   help="Number of ACI-Bench samples used to build Exp 3 training set (default: 3)")
    p.add_argument("--n-injections", type=int, default=5,
                   help="Max hallucinations injected per training sample in Exp 3 (default: 5)")
    p.add_argument("--n-ffn-layers", type=int, default=5,
                   help="Size of set F (Knowledge FFN layers selected by PKS Pearson r) in Exp 3 "
                        "(default: 5)")
    p.add_argument("--n-copy-layers", type=int, default=5,
                   help="Size of set A (Copying Head layers selected by ECS Pearson r) in Exp 3 "
                        "(default: 5)")
    p.add_argument("--halluc-threshold", type=float, default=0.5,
                   help="Probability cutoff for binary hallucination flag in Exp 3 (default: 0.5)")
    p.add_argument("--nli-model", default=None,
                   help="HuggingFace NLI cross-encoder for Exp 3 validation, e.g. "
                        "cross-encoder/nli-deberta-v3-small  (requires sentence-transformers)")
    p.add_argument("--n-examples", type=int, default=10,
                   help="Number of ACI-Bench examples to process in Exp 4 (default: 10)")
    p.add_argument("--sample-start", type=int, default=0,
                   help="First ACI-Bench row index for Exp 4 range (default: 0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        model_name=(
            "google/gemma-2-2b-it"
            if args.model == "gemma"
            else "meta-llama/Meta-Llama-3-8B-instruct"
        ),
        sample_idx=args.sample,
        max_new_tokens=args.max_new_tokens,
        gen_temperature=args.temperature,
        output_dir=args.out,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}  |  dtype: {cfg.dtype}")
    print(f"  Dataset        : {cfg.dataset_repo}  [{cfg.dataset_config} / {cfg.dataset_split}]")
    print(f"  Sample index   : {cfg.sample_idx}")
    print(f"  Max new tokens : {cfg.max_new_tokens}  |  temperature: {cfg.gen_temperature}")
    print(f"  Output dir     : {out.resolve()}")

    transcript, gold_note = load_aci_sample(cfg)

    print(f"\nLoading {cfg.model_name} via TransformerLens …")
    model = HookedTransformer.from_pretrained(
        cfg.model_name,
        dtype=cfg.dtype,
        default_padding_side="right",
    )
    model.eval()
    model.to(cfg.device)
    print(f"  Layers: {model.cfg.n_layers}  |  Heads: {model.cfg.n_heads}"
          f"  |  d_model: {model.cfg.d_model}")

    if args.exp in ("2a", "all"):
        run_experiment_2a(model, cfg, out, transcript, gold_note)

    if args.exp in ("2b", "all"):
        run_experiment_2b(model, cfg, out, transcript, gold_note)

    if args.exp in ("2c", "both", "all"):
        inject_fn = _resolve_inject_fn(args)
        run_experiment_2c(model, cfg, out, transcript, gold_note, inject_fn=inject_fn)

    if args.exp in ("2d", "both", "all"):
        inject_fn = _resolve_inject_fn(args)
        run_experiment_2d(model, cfg, out, transcript, gold_note, inject_fn=inject_fn)

    if args.exp in ("3", "all"):
        inject_fn = _resolve_inject_fn(args)
        run_experiment_3(
            model, cfg, out, transcript, gold_note,
            inject_fn=inject_fn,
            n_train_samples=args.n_train_samples,
            n_injections=args.n_injections,
            n_ffn_layers=args.n_ffn_layers,
            n_copy_layers=args.n_copy_layers,
            halluc_threshold=args.halluc_threshold,
            nli_model=args.nli_model,
        )

    if args.exp in ("4", "all"):
        inject_fn = _resolve_inject_fn(args)
        run_experiment_4(
            model, cfg, out,
            inject_fn=inject_fn,
            n_examples=args.n_examples,
            n_injections=args.n_injections,
            sample_start=args.sample_start,
        )

    print("\n" + "═"*54)
    print("  Done.  All outputs written to:", out.resolve())
    print("═"*54 + "\n")


if __name__ == "__main__":
    main()
