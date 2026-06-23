"""
exp4.py
=======
Experiment 4 — Layer-wise ECS/PKS discriminability statistics aggregated
across multiple ACI-Bench examples with synthetic hallucination injection.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

from sae_experiments.old.config import Config, load_aci_sample
from metrics import (
    compute_ecs_pks,
    layer_discriminability,
)
from sae_experiments.old.tokenization import tokenize_as_generated

warnings.filterwarnings("ignore")

try:
    from sae_experiments.old.halluc_llm import (
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

        ecs, pks, ecs_layers, pks_layers, _ = compute_ecs_pks(model, tokens, t_len, cfg)

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

