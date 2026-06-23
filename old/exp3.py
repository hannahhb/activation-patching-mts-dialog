"""
exp3.py
=======
Experiment 3 — REDEEP hallucination scoring via logistic regression on
layer-wise ECS/PKS features derived from synthetic training data.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

from sae_experiments.old.config import Config, load_aci_sample
from sae_experiments.old.experiments import _load_exp4_data, collect_training_data
from sae_experiments.old.exp2 import run_experiment_2c
from metrics import (
    compute_ecs_pks,
    fit_hallucination_regressor,
    layer_discriminability,
)
from sae_experiments.old.tokenization import generate_note, tokenize_as_generated

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

    # ── 3. Identify set F — layers where high PKS predicts hallucination ────
    # Use most positive PKS Cohen's d: hallucinated tokens score higher PKS
    # (model over-relying on parametric memory).  Cohen's d sign is unambiguous
    # and robust to class imbalance.
    print(
        f"\n  Step 3/8 — Identifying set F "
        f"(top-{n_ffn_layers} Knowledge FFN layers by most positive PKS Cohen's d) …"
    )

    F = np.argsort(pks_cohens_d)[::-1][:n_ffn_layers].tolist()

    print(
        f"  Set F = {F}  "
        f"(PKS Cohen's d: {[round(float(pks_cohens_d[l]), 4) for l in F]}, "
        f"AUROC: {[round(float(pks_auroc[l]), 4) for l in F]})"
    )

    # ── 4. Identify set A — layers where low ECS predicts hallucination ──────
    # Use most negative ECS Cohen's d: hallucinated tokens have lower ECS
    # (copying heads fail to attend to the transcript).
    print(
        f"\n  Step 4/8 — Identifying set A "
        f"(top-{n_copy_layers} Copying Head layers by most negative ECS Cohen's d) …"
    )

    A = np.argsort(ecs_cohens_d)[:n_copy_layers].tolist()

    print(
        f"  Set A = {A}  "
        f"(ECS Cohen's d: {[round(float(ecs_cohens_d[l]), 4) for l in A]}, "
        f"AUROC: {[round(float(ecs_auroc[l]), 4) for l in A]})"
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
        f"Set F (Knowledge FFNs, top-{n_ffn_layers} by most positive PKS Cohen's d): {F}\n"
        f"Set A (Copying Heads,  top-{n_copy_layers} by most negative ECS Cohen's d): {A}\n"
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

    ecs, pks, ecs_layers, pks_layers, _ = compute_ecs_pks(model, tokens, t_len, cfg)

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
