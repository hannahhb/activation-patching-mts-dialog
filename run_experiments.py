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

Usage
-----
    python run_experiments.py                        # 2a + 2b, gemma, sample 0
    python run_experiments.py --exp 2c               # only 2c
    python run_experiments.py --exp all              # 2a + 2b + 2c
    python run_experiments.py --model llama          # Llama 3 8B
    python run_experiments.py --sample 3             # ACI-Bench row 3
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
    compute_dla,
    compute_ecs_pks,
    dla_discriminability,
    generate_note,
    hallucination_risk,
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

        # Save CSV
        disc_df = pd.DataFrame({
            "layer":        np.arange(n_layers),
            "pks_auroc":    disc["pks_auroc"],
            "pks_cohens_d": disc["pks_cohens_d"],
            "ecs_auroc":    disc["ecs_auroc"],
            "ecs_cohens_d": disc["ecs_cohens_d"],
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
# 13. Entry point
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
    p.add_argument("--exp", choices=["2a", "2b", "2c", "2d", "both", "all"], default="both",
                   help="Which experiment(s) to run  (both=2a+2b [default]  |  all=2a+2b+2c+2d)")
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

    print("\n" + "═"*54)
    print("  Done.  All outputs written to:", out.resolve())
    print("═"*54 + "\n")


if __name__ == "__main__":
    main()
