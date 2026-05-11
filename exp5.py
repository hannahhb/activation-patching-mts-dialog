"""
exp5.py
=======
Experiment 5 — Activation Patching / Causal Tracing for hallucination
localisation via ROME-style restoration scoring.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformer_lens import HookedTransformer

from config import Config, load_aci_sample
from metrics import compute_causal_patch_scores
from tokenization import tokenize_as_generated

warnings.filterwarnings("ignore")

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
    inject_hallucinations = None


# ─────────────────────────────────────────────
# 15. Experiment 5 — Activation Patching / Causal Tracing
# ─────────────────────────────────────────────

def run_experiment_5(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    transcript: str,
    gold_note: str,
    inject_fn=None,
    n_injections: int = 5,
    components: Optional[List[str]] = None,
    n_examples: int = 1,
    sample_start: int = 0,
) -> Dict:
    """
    Experiment 5: Activation Patching / Causal Tracing for hallucination localisation.

    For each example the pipeline is:
      1. Inject hallucinations into the gold note  →  corrupted note + halluc_idx
      2. Tokenise both gold (clean) and corrupted notes with the generation prompt
      3. For each (layer, component) pair:
           • Run the corrupted forward pass with the clean activation patched in
             at all hallucinated token positions
           • Measure restoration = (P_patched − P_corrupt) / (P_clean − P_corrupt)
      4. Aggregate restoration matrices across examples (mean ± std)
      5. Plot heatmap + bar charts + save CSV

    The restoration score tells you *which layers and components are causally
    responsible* for hallucinated tokens — a much stronger signal than the
    correlation-based ECS/PKS discriminability from Experiments 3/4.

    Interpretation
    ──────────────
    High restoration at layer l, component c → clean information from that
    component is sufficient to prevent the hallucination.  This pins the causal
    locus of the error to that specific circuit element.

    Parameters
    ----------
    n_injections : max hallucinations per example.
    components   : list of activation types to patch.
                   Default: ["resid_pre", "attn_out", "mlp_out"]
    n_examples   : number of ACI-Bench examples to run (>1 averages results).
    sample_start : first ACI-Bench row index (inclusive).

    Outputs
    -------
    exp5_restoration.csv      — per-layer × component restoration scores (mean ± std)
    exp5_heatmap.png          — (n_layers × components) restoration heatmap
    exp5_components.png       — bar chart of mean restoration per component
    exp5_peak_layers.png      — restoration vs layer for each component (line plot)
    """
    from dataclasses import replace as _dc_replace

    print("\n" + "═" * 54)
    print("  EXPERIMENT 5 — Activation Patching / Causal Tracing")
    print("═" * 54)

    if components is None:
        components = ["resid_pre", "attn_out", "mlp_out"]

    _inject = inject_fn or inject_hallucinations

    # Accumulate restoration matrices across examples
    restoration_all: List[np.ndarray] = []   # each (n_layers, n_components)
    example_meta:   List[Dict]        = []

    for si in range(sample_start, sample_start + n_examples):
        print(f"\n  ── Sample {si} ──────────────────────────────────────")
        cfg_i = _dc_replace(cfg, sample_idx=si)

        try:
            tr_i, gold_i = load_aci_sample(cfg_i)
        except Exception as exc:
            print(f"  [5] Sample {si}: load failed — {exc}")
            continue

        # ── Inject hallucinations ────────────────────────────────────────────
        try:
            corrupted, injections = _inject(gold_i, max_injections=n_injections, seed=si)
        except Exception as exc:
            print(f"  [5] Sample {si}: injection failed — {exc}")
            continue

        if not injections:
            print(f"  [5] Sample {si}: no injections produced, skipping.")
            continue

        halluc_idx = halluc_token_indices(model.tokenizer, corrupted, injections)

        # ── Tokenise clean (gold) and corrupted notes ────────────────────────
        tokens_corr, t_len_corr, note_toks_corr = tokenize_as_generated(
            model, tr_i, corrupted
        )
        tokens_clean, t_len_clean, _ = tokenize_as_generated(
            model, tr_i, gold_i
        )

        # Both use the same prompt so transcript_len is identical
        assert t_len_corr == t_len_clean, (
            f"Prompt lengths differ ({t_len_corr} vs {t_len_clean}) — "
            "transcript changed between runs."
        )
        transcript_len = t_len_corr
        note_len_corr  = len(note_toks_corr)

        # Convert note-space halluc_idx → sequence-space positions
        halluc_seq = [
            transcript_len + idx
            for idx in halluc_idx
            if idx < note_len_corr
        ]

        if not halluc_seq:
            print(f"  [5] Sample {si}: no hallucinated positions in note span, skipping.")
            continue

        n_h = len(halluc_seq)
        print(f"  Hallucinated sequence positions: {n_h}  ({halluc_seq[:5]}{'…' if n_h>5 else ''})")

        # ── Run causal patching ──────────────────────────────────────────────
        result = compute_causal_patch_scores(
            model,
            tokens_clean     = tokens_clean.to(cfg.device),
            tokens_corrupted = tokens_corr.to(cfg.device),
            halluc_seq_positions = halluc_seq,
            components       = components,
            device           = cfg.device,
        )

        if result is None:
            print(f"  [5] Sample {si}: no valid patch positions, skipping.")
            continue

        restoration_all.append(result["restoration"])   # (n_layers, n_components)

        example_meta.append({
            "sample_idx":       si,
            "n_injections":     len(injections),
            "n_halluc_seq":     n_h,
            "n_valid_patched":  len(result["valid_positions"]),
            "baseline_clean":   round(result["baseline_clean"],   4),
            "baseline_corrupt": round(result["baseline_corrupt"], 4),
            **{
                f"peak_layer_{c}": int(np.argmax(result["restoration"][:, ci]))
                for ci, c in enumerate(components)
            },
        })

        print(f"  Baseline  clean P(correct) : {result['baseline_clean']:.4f}")
        print(f"  Baseline corrupt P(correct) : {result['baseline_corrupt']:.4f}")
        for ci, c in enumerate(components):
            peak_l = int(np.argmax(result["restoration"][:, ci]))
            peak_v = float(result["restoration"][peak_l, ci])
            print(f"  {c:12s} — peak restoration {peak_v:+.4f} @ layer {peak_l}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    n_valid = len(restoration_all)
    if n_valid == 0:
        print("\n  [Exp 5] No valid examples — aborting.")
        return {}

    print(f"\n  Valid examples: {n_valid} / {n_examples}")

    rest_stack = np.stack(restoration_all, axis=0)   # (n_valid, n_layers, n_components)
    rest_mean  = rest_stack.mean(axis=0)              # (n_layers, n_components)
    rest_std   = rest_stack.std(axis=0)

    n_layers = rest_mean.shape[0]
    layers   = np.arange(n_layers)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_rows = []
    for l in range(n_layers):
        row = {"layer": l}
        for ci, c in enumerate(components):
            row[f"{c}_mean"] = round(float(rest_mean[l, ci]), 6)
            row[f"{c}_std"]  = round(float(rest_std[l, ci]),  6)
        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(out / "exp5_restoration.csv", index=False)
    pd.DataFrame(example_meta).to_csv(out / "exp5_example_meta.csv", index=False)
    print("  Saved → exp5_restoration.csv")
    print("  Saved → exp5_example_meta.csv")

    # ── Plot 1: Heatmap (n_layers × n_components) ────────────────────────────
    fig_h, ax_h = plt.subplots(figsize=(max(6, len(components) * 2), max(8, n_layers // 3)))

    sns.heatmap(
        rest_mean,
        ax=ax_h,
        xticklabels=components,
        yticklabels=layers,
        cmap="RdYlGn",
        center=0.0,
        vmin=-0.5, vmax=1.0,
        annot=(n_layers <= 32),
        fmt=".2f",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Mean restoration score", "shrink": 0.7},
    )
    ax_h.set_xlabel("Component", fontsize=11)
    ax_h.set_ylabel("Layer",     fontsize=11)
    ax_h.set_title(
        f"Exp 5 — Causal Tracing: Restoration per Layer × Component\n"
        f"{cfg.model_name}  |  {n_valid} example(s)  |  {n_injections} injections each",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig_h.savefig(out / "exp5_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    print("  Saved → exp5_heatmap.png")

    # ── Plot 2: Per-layer line plot (one line per component) ─────────────────
    _colours = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
    fig_l, ax_l = plt.subplots(figsize=(max(12, n_layers // 2), 5))

    for ci, c in enumerate(components):
        col = _colours[ci % len(_colours)]
        mean_c = rest_mean[:, ci]
        std_c  = rest_std[:, ci]
        ax_l.plot(layers, mean_c, color=col, lw=2.2, marker="o", markersize=4, label=c)
        ax_l.fill_between(layers, mean_c - std_c, mean_c + std_c,
                          color=col, alpha=0.15)

    ax_l.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.6, label="No effect")
    ax_l.axhline(1.0, color="black", ls=":",  lw=0.8, alpha=0.4, label="Full restoration")
    ax_l.set_xlabel("Layer", fontsize=11)
    ax_l.set_ylabel("Mean restoration score", fontsize=11)
    ax_l.set_title(
        f"Exp 5 — Restoration vs Layer  ({cfg.model_name})\n"
        f"Mean ± 1 SD across {n_valid} example(s)",
        fontsize=11, fontweight="bold",
    )
    ax_l.legend(fontsize=9)
    ax_l.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    fig_l.savefig(out / "exp5_peak_layers.png", dpi=150, bbox_inches="tight")
    plt.close(fig_l)
    print("  Saved → exp5_peak_layers.png")

    # ── Plot 3: Bar chart — mean restoration per component ───────────────────
    comp_means = rest_mean.mean(axis=0)   # (n_components,)
    comp_stds  = rest_std.mean(axis=0)

    fig_b, ax_b = plt.subplots(figsize=(max(5, len(components) * 1.5), 4))
    bars = ax_b.bar(
        components, comp_means,
        color=[_colours[i % len(_colours)] for i in range(len(components))],
        edgecolor="white", alpha=0.85,
    )
    ax_b.errorbar(
        components, comp_means, yerr=comp_stds,
        fmt="none", color="black", capsize=5, lw=1.5,
    )
    for bar, val in zip(bars, comp_means):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_b.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_b.set_ylabel("Mean restoration (across all layers)", fontsize=10)
    ax_b.set_title(
        f"Exp 5 — Overall Causal Contribution per Component\n"
        f"{cfg.model_name}  |  {n_valid} example(s)",
        fontsize=11, fontweight="bold",
    )
    ax_b.set_ylim(min(0, comp_means.min() - 0.1), comp_means.max() + 0.15)
    plt.tight_layout()
    fig_b.savefig(out / "exp5_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print("  Saved → exp5_components.png")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n  ── Peak restoration layers ──")
    for ci, c in enumerate(components):
        peak_l = int(np.argmax(rest_mean[:, ci]))
        print(f"  {c:12s}  peak layer={peak_l:>3}  "
              f"restoration={rest_mean[peak_l, ci]:+.4f} ± {rest_std[peak_l, ci]:.4f}")

    return {
        "restoration_mean": rest_mean,
        "restoration_std":  rest_std,
        "rest_stack":       rest_stack,
        "components":       components,
        "n_layers":         n_layers,
        "n_valid":          n_valid,
    }

