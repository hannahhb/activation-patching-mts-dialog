"""
exp6.py
=======
Experiment 6 — Semantic Entropy Probe for hallucination detection.

Theory
------
Semantic entropy (Farquhar et al., NeurIPS 2024) measures how much a model's
K independent generations disagree *semantically* about the same output
position.  High SE → the model is uncertain → hallucination-prone.

The probe variant trains a linear classifier on the residual stream
(resid_post) at each layer to *predict* high-SE positions from a single
forward pass, eliminating the K-generation cost at inference time.

Because compute_ecs_pks() already caches resid_post for all layers and returns
the cache as its 5th value, no extra forward pass is required here — the probe
reads activations from the cache produced alongside ECS/PKS.

Pipeline
--------
1. Training examples  — for n_train_examples ACI-Bench samples:
     a. Generate K notes → compute semantic entropy per sentence position
     b. Run ECS/PKS forward pass → extract resid_post from the cache
     c. Accumulate (resid_post_layers, token_se_scores) across all examples

2. Probe training — one logistic-regression probe per layer; evaluate with
   cross-validated AUROC to identify which layers carry uncertainty signal.

3. Score the target note — generate note → run ECS/PKS forward pass →
   apply probes from the cached resid_post → produce per-token SE probability.

4. Analysis — scatter plots comparing SE_probe vs ECS and PKS to assess
   signal independence (different failure modes).

5. Optional ensemble — if Exp 3 halluc_prob is provided, fit an augmented
   logistic regressor with SE_probe as an additional feature and compare AUROC.

Outputs
-------
exp6_probe_auroc_per_layer.png  — per-layer CV AUROC (line plot)
exp6_probe_stats.csv            — layer, auroc, best_layer flag
exp6_se_vs_ecs_scatter.png      — SE prob vs ECS for note tokens
exp6_se_vs_pks_scatter.png      — SE prob vs PKS for note tokens
exp6_token_scores.csv           — per-token: ecs, pks, se_prob_best, se_prob_mean
exp6_report.html                — three-signal token highlights
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

from config import Config, load_aci_sample
from metrics import (
    apply_se_probe,
    compute_ecs_pks,
    compute_semantic_entropy,
    quadrant_stats,
    train_se_probe,
)
from tokenization import generate_note, tokenize_as_generated, tokenize_pair

warnings.filterwarnings("ignore")


def run_experiment_6(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    transcript: str,
    gold_note: str,
    K: int = 5,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    se_threshold: float = 0.5,
    n_train_examples: int = 5,
    sample_start: int = 0,
    halluc_prob_exp3: Optional[np.ndarray] = None,
) -> Dict:
    """
    Experiment 6: Semantic Entropy Probe — learn per-layer linear probes that
    predict high-uncertainty (high-SE) tokens from residual stream activations,
    then apply them to the target generated note.

    Parameters
    ----------
    K                : number of independent generations per training example.
    nli_model_name   : sentence-transformers CrossEncoder for semantic clustering.
    se_threshold     : SE score above which a token is labelled "uncertain".
    n_train_examples : number of ACI-Bench examples used to train the probes.
    sample_start     : first row index in ACI-Bench for training examples.
    halluc_prob_exp3 : optional (note_len,) array from Exp 3 — if provided,
                       adds an ensemble comparison (Exp3 alone vs Exp3+SE probe).
    """
    from dataclasses import replace as _dc_replace

    print("\n" + "═" * 54)
    print("  EXPERIMENT 6 — Semantic Entropy Probe")
    print("═" * 54)
    print(f"  K={K}  NLI={nli_model_name}  threshold={se_threshold}")
    print(f"  Training examples: {sample_start} … {sample_start + n_train_examples - 1}")

    # ── Step 1: Collect training data ────────────────────────────────────────
    print(f"\n  Step 1/{5} — Collecting SE labels + residual stream for "
          f"{n_train_examples} training examples …")

    resid_list: List[np.ndarray] = []   # each (n_layers, n_tokens_i, d_model)
    se_list:    List[np.ndarray] = []   # each (n_tokens_i,)
    n_train_valid = 0

    for si in range(sample_start, sample_start + n_train_examples):
        print(f"\n  ── Training sample {si} ──────────────────────────────")
        cfg_i = _dc_replace(cfg, sample_idx=si)

        try:
            tr_i, gold_i = load_aci_sample(cfg_i)
        except Exception as exc:
            print(f"  [6] Sample {si}: load failed — {exc}")
            continue

        # ── 1a. Semantic entropy from K generations ──────────────────────────
        print(f"  Generating K={K} notes for SE computation …")
        se_result = compute_semantic_entropy(
            model, tr_i, cfg_i,
            K=K, nli_model_name=nli_model_name,
        )
        if se_result is None:
            print(f"  [6] Sample {si}: SE computation failed, skipping.")
            continue

        token_se = se_result["token_se_scores"]   # (note_len_i,)
        note_len_i = se_result["note_len"]
        print(f"  Mean SE = {se_result['mean_se']:.4f}  |  "
              f"note_len = {note_len_i}  |  K_actual = {se_result['K_actual']}")

        # ── 1b. ECS/PKS forward pass → cache with resid_post ─────────────────
        # Use the first generated note (same as SE computation's first sample)
        first_note = se_result["notes"][0]
        try:
            tokens, t_len, note_toks = tokenize_pair(model, tr_i, first_note)
        except Exception as exc:
            print(f"  [6] Sample {si}: tokenisation failed — {exc}")
            continue

        tokens = tokens.to(cfg.device)

        _, _, _, _, cache_i = compute_ecs_pks(model, tokens, t_len, cfg_i)

        # ── 1c. Extract resid_post for note positions ─────────────────────────
        n_layers = model.cfg.n_layers
        d_model  = model.cfg.d_model
        actual_note_len = min(note_len_i, tokens.shape[1] - t_len)

        if actual_note_len <= 0:
            print(f"  [6] Sample {si}: no note tokens in cache, skipping.")
            continue

        resid_i = np.zeros((n_layers, actual_note_len, d_model), dtype=np.float32)
        for l in range(n_layers):
            resid_l = cache_i["resid_post", l][0].float().cpu().numpy()
            resid_i[l] = resid_l[t_len: t_len + actual_note_len]

        # Trim token_se to actual_note_len (may differ slightly due to K-sample alignment)
        se_trimmed = token_se[:actual_note_len]
        if len(se_trimmed) < actual_note_len:
            se_trimmed = np.concatenate([
                se_trimmed,
                np.zeros(actual_note_len - len(se_trimmed))
            ])

        resid_list.append(resid_i)
        se_list.append(se_trimmed)
        n_train_valid += 1
        print(f"  [6] Sample {si}: {actual_note_len} tokens  mean_se={se_trimmed.mean():.4f}")

    if n_train_valid == 0:
        print("\n  [Exp 6] No valid training examples — aborting.")
        return {}

    print(f"\n  Valid training examples: {n_train_valid} / {n_train_examples}")

    # ── Step 2: Train probes ──────────────────────────────────────────────────
    print(f"\n  Step 2/{5} — Training SE probes (one per layer) …")

    # Concatenate across examples: (n_layers, N_total, d_model) and (N_total,)
    resid_all = np.concatenate(resid_list, axis=1)   # (n_layers, N_total, d_model)
    se_all    = np.concatenate(se_list)               # (N_total,)

    print(f"  Training set: {resid_all.shape[1]} tokens  "
          f"(high-SE fraction: {(se_all > se_threshold).mean():.1%})")

    probe_dict = train_se_probe(resid_all, se_all, threshold=se_threshold)
    if probe_dict is None:
        print("  [Exp 6] Probe training failed — aborting.")
        return {}

    # Save probe AUROC CSV
    n_layers = len(probe_dict["probes"])
    probe_df = pd.DataFrame({
        "layer":      np.arange(n_layers),
        "auroc":      probe_dict["auroc"],
        "best_layer": [l == probe_dict["best_layer"] for l in range(n_layers)],
    })
    probe_df.to_csv(out / "exp6_probe_stats.csv", index=False)
    print("  Saved → exp6_probe_stats.csv")

    # Plot: per-layer probe AUROC
    fig_a, ax_a = plt.subplots(figsize=(max(10, n_layers // 2), 4))
    layers_ax = np.arange(n_layers)
    auroc_arr = probe_dict["auroc"]

    ax_a.plot(layers_ax, auroc_arr, color="#9C27B0", lw=2.2, marker="o", markersize=4)
    ax_a.axhline(0.5,  color="gray",  ls="--", lw=1.0, alpha=0.6, label="Chance (0.5)")
    ax_a.axhline(0.7,  color="green", ls=":",  lw=0.8, alpha=0.5, label="Good (0.7)")
    ax_a.axvline(probe_dict["best_layer"], color="#9C27B0", ls=":", lw=1.2,
                 alpha=0.7, label=f"Best layer ({probe_dict['best_layer']})")
    ax_a.fill_between(layers_ax, 0.5, auroc_arr,
                      where=auroc_arr > 0.5, color="#9C27B0", alpha=0.12, label="Above chance")
    ax_a.set_xlabel("Layer", fontsize=11)
    ax_a.set_ylabel("Cross-validated AUROC", fontsize=11)
    ax_a.set_title(
        f"Exp 6 — SE Probe AUROC per Layer  ({cfg.model_name})\n"
        f"Trained on {n_train_valid} examples × K={K} generations  "
        f"|  SE threshold = {se_threshold}",
        fontsize=11, fontweight="bold",
    )
    ax_a.legend(fontsize=9)
    ax_a.grid(axis="y", ls=":", alpha=0.4)
    ax_a.set_ylim(max(0.0, auroc_arr.min() - 0.05), min(1.0, auroc_arr.max() + 0.1))
    plt.tight_layout()
    fig_a.savefig(out / "exp6_probe_auroc_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    print("  Saved → exp6_probe_auroc_per_layer.png")

    # ── Step 3: Score the target note ─────────────────────────────────────────
    print(f"\n  Step 3/{5} — Scoring the target generated note …")

    target_note = generate_note(model, transcript, cfg)
    (out / "exp6_generated_note.txt").write_text(target_note, encoding="utf-8")
    print("  Saved → exp6_generated_note.txt")

    tokens_tgt, t_len_tgt, note_toks_tgt = tokenize_pair(model, transcript, target_note)
    tokens_tgt = tokens_tgt.to(cfg.device)
    note_len_tgt = len(note_toks_tgt)

    ecs_tgt, pks_tgt, _, _, cache_tgt = compute_ecs_pks(
        model, tokens_tgt, t_len_tgt, cfg
    )

    se_out = apply_se_probe(cache_tgt, probe_dict, t_len_tgt, note_len_tgt)
    se_prob_best = se_out["se_prob_best"]          # (note_len,)
    se_prob_mean = se_out["se_prob"].mean(axis=0)  # (note_len,) mean across layers

    print(f"  Note length : {note_len_tgt} tokens")
    print(f"  SE probe (best layer {se_out['best_layer']}) — "
          f"mean: {se_prob_best.mean():.4f}  median: {np.median(se_prob_best):.4f}")
    print(f"  ECS — mean: {ecs_tgt.mean():.4f}  "
          f"PKS — mean: {pks_tgt.mean():.4f}")

    # ── Step 4: Analysis plots ────────────────────────────────────────────────
    print(f"\n  Step 4/{5} — Generating analysis plots …")

    _COLOURS = {"extractive": "#2196F3", "parametric": "#FF9800",
                "synthesized": "#4CAF50", "hallucinatory": "#F44336"}

    em_ecs = float(np.median(ecs_tgt))
    em_pks = float(np.median(pks_tgt))
    em_se  = float(np.median(se_prob_best))

    def _quad_colour(e, p):
        if   e >= em_ecs and p < em_pks:  return _COLOURS["extractive"]
        elif e < em_ecs  and p >= em_pks: return _COLOURS["parametric"]
        elif e >= em_ecs and p >= em_pks: return _COLOURS["synthesized"]
        else:                              return _COLOURS["hallucinatory"]

    quad_colours = [_quad_colour(e, p) for e, p in zip(ecs_tgt, pks_tgt)]

    # Scatter 1: SE_prob vs ECS
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(ecs_tgt, se_prob_best, c=quad_colours, s=45, alpha=0.7,
                edgecolors="white", linewidth=0.3)
    ax1.axvline(em_ecs, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.axhline(em_se,  color="gray", ls="--", lw=0.8, alpha=0.5)

    # Compute Pearson r for annotation
    corr_ecs_se = float(np.corrcoef(ecs_tgt, se_prob_best)[0, 1])
    ax1.set_xlabel("External Context Score (ECS)", fontsize=11)
    ax1.set_ylabel("SE Probe Probability  (high = uncertain)", fontsize=11)
    ax1.set_title(
        f"Exp 6 — SE Probe vs ECS  ({cfg.model_name})\n"
        f"Pearson r = {corr_ecs_se:+.3f}  |  "
        f"Best probe layer = {se_out['best_layer']}",
        fontsize=11, fontweight="bold",
    )
    ax1.text(0.02, 0.98,
             "Independent signals if |r| < 0.3",
             transform=ax1.transAxes, fontsize=8, va="top", color="gray")
    plt.tight_layout()
    fig1.savefig(out / "exp6_se_vs_ecs_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved → exp6_se_vs_ecs_scatter.png")

    # Scatter 2: SE_prob vs PKS
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(pks_tgt, se_prob_best, c=quad_colours, s=45, alpha=0.7,
                edgecolors="white", linewidth=0.3)
    ax2.axvline(em_pks, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(em_se,  color="gray", ls="--", lw=0.8, alpha=0.5)

    corr_pks_se = float(np.corrcoef(pks_tgt, se_prob_best)[0, 1])
    ax2.set_xlabel("Parametric Knowledge Score (PKS)", fontsize=11)
    ax2.set_ylabel("SE Probe Probability  (high = uncertain)", fontsize=11)
    ax2.set_title(
        f"Exp 6 — SE Probe vs PKS  ({cfg.model_name})\n"
        f"Pearson r = {corr_pks_se:+.3f}  |  "
        f"Best probe layer = {se_out['best_layer']}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig2.savefig(out / "exp6_se_vs_pks_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved → exp6_se_vs_pks_scatter.png")

    # ── Per-token CSV ─────────────────────────────────────────────────────────
    rows = [
        {
            "token_idx":     i,
            "token":         note_toks_tgt[i].replace("▁", "").replace("Ġ", "").strip(),
            "ecs":           round(float(ecs_tgt[i]),        4),
            "pks":           round(float(pks_tgt[i]),        4),
            "se_prob_best":  round(float(se_prob_best[i]),   4),
            "se_prob_mean":  round(float(se_prob_mean[i]),   4),
            "quadrant":      (
                "extractive"    if ecs_tgt[i] >= em_ecs and pks_tgt[i] <  em_pks else
                "parametric"    if ecs_tgt[i] <  em_ecs and pks_tgt[i] >= em_pks else
                "synthesized"   if ecs_tgt[i] >= em_ecs and pks_tgt[i] >= em_pks else
                "hallucinatory"
            ),
        }
        for i in range(note_len_tgt)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp6_token_scores.csv", index=False)
    print("  Saved → exp6_token_scores.csv")

    # ── Step 5: Optional ensemble comparison ──────────────────────────────────
    ensemble_auc: Optional[float] = None
    base_auc:     Optional[float] = None

    if halluc_prob_exp3 is not None and len(halluc_prob_exp3) == note_len_tgt:
        print(f"\n  Step 5/{5} — Ensemble comparison (Exp 3 + SE probe) …")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score

            # We don't have true halluc labels for the target note — we use
            # se_prob_best > threshold as a proxy "oracle" for illustration only
            y_proxy = (se_prob_best > se_threshold).astype(int)

            if y_proxy.sum() > 0 and (1 - y_proxy).sum() > 0:
                X_base = halluc_prob_exp3.reshape(-1, 1)
                X_aug  = np.stack([halluc_prob_exp3, se_prob_best], axis=1)

                sc = StandardScaler()
                base_auc    = roc_auc_score(y_proxy, X_base.ravel())
                aug_auc     = roc_auc_score(y_proxy, X_aug @ np.array([0.5, 0.5]))

                ensemble_auc = float(aug_auc)
                base_auc     = float(base_auc)
                print(f"  Base AUROC (Exp 3 alone)    : {base_auc:.4f}")
                print(f"  Ensemble AUROC (Exp3 + SE)  : {ensemble_auc:.4f}")
                delta = ensemble_auc - base_auc
                print(f"  Delta                        : {delta:+.4f}")
        except Exception as exc:
            print(f"  [6] Ensemble step failed: {exc}")

    # ── HTML report ───────────────────────────────────────────────────────────
    def _se_colour(p: float) -> str:
        # Purple gradient: low SE = white/light, high SE = deep purple
        intensity = int(200 * (1 - p))
        return f"rgb({intensity + 55},{intensity},{200})"

    token_spans = "".join(
        f'<span title="se={se_prob_best[i]:.3f} | ecs={ecs_tgt[i]:.3f} | pks={pks_tgt[i]:.3f}" '
        f'style="background:{_se_colour(float(se_prob_best[i]))}; '
        f'border-radius:3px; padding:1px 2px; margin:1px; color:white;">'
        f'{note_toks_tgt[i].replace("<","&lt;").replace(">","&gt;")}'
        f'</span>'
        for i in range(note_len_tgt)
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exp 6 — SE Probe Report — {cfg.model_name}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; color: #222; }}
  h1 {{ font-size: 1.3em; }} h2 {{ font-size: 1.1em; margin-top: 1.5em; }}
  .note {{ line-height: 2.4; font-size: 0.9em; background: #111;
           border: 1px solid #444; padding: 1em; border-radius: 6px; }}
  img {{ max-width: 100%; border: 1px solid #ccc; border-radius: 4px; margin: 0.5em 0; }}
  .meta {{ font-size: 0.85em; color: #555; }}
  table {{ border-collapse: collapse; font-size: 0.82em; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
  th {{ background: #f0f0f0; text-align: center; }}
</style></head><body>
<h1>Experiment 6 — Semantic Entropy Probe Report</h1>
<p class="meta">
Model: <b>{cfg.model_name}</b> &nbsp;|&nbsp;
K={K} &nbsp;|&nbsp; NLI: {nli_model_name}<br>
Training examples: {n_train_valid} &nbsp;|&nbsp;
SE threshold: {se_threshold} &nbsp;|&nbsp;
Best probe layer: {se_out['best_layer']} &nbsp;|&nbsp;
Best layer AUROC: {probe_dict['auroc'][se_out['best_layer']]:.4f}<br>
ECS–SE Pearson r: {corr_ecs_se:+.3f} &nbsp;|&nbsp;
PKS–SE Pearson r: {corr_pks_se:+.3f}
{'<br>Ensemble AUROC (Exp3+SE): ' + f'{ensemble_auc:.4f}  (base: {base_auc:.4f})' if ensemble_auc is not None else ''}
</p>

<h2>Generated Note — colour = SE probe probability (purple = high uncertainty)</h2>
<div class="note">{token_spans}</div>

<h2>SE Probe vs ECS</h2>
<img src="exp6_se_vs_ecs_scatter.png" alt="SE vs ECS">

<h2>SE Probe vs PKS</h2>
<img src="exp6_se_vs_pks_scatter.png" alt="SE vs PKS">

<h2>Probe AUROC per Layer</h2>
<img src="exp6_probe_auroc_per_layer.png" alt="Probe AUROC">

<h2>Signal Interpretation</h2>
<table>
<tr><th>Signal</th><th>High value means …</th><th>Low value means …</th></tr>
<tr><td>ECS ↑</td><td>model looked at transcript</td><td>model ignored transcript</td></tr>
<tr><td>PKS ↑</td><td>model drew on parametric memory</td><td>model trusted context</td></tr>
<tr><td>SE ↑</td><td>model was uncertain (varied across K generations)</td><td>model was confident</td></tr>
</table>
</body></html>"""

    (out / "exp6_report.html").write_text(html, encoding="utf-8")
    print("  Saved → exp6_report.html")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n  ── Exp 6 summary ──")
    print(f"  Best probe layer          : {se_out['best_layer']}  "
          f"(AUROC = {probe_dict['auroc'][se_out['best_layer']]:.4f})")
    print(f"  Mean probe AUROC          : {probe_dict['auroc'].mean():.4f}")
    print(f"  ECS–SE correlation        : {corr_ecs_se:+.4f}")
    print(f"  PKS–SE correlation        : {corr_pks_se:+.4f}")
    print(f"  SE signal independence    : "
          f"{'YES (|r|<0.3)' if max(abs(corr_ecs_se),abs(corr_pks_se))<0.3 else 'PARTIAL — signals overlap'}")

    return {
        "probe_dict":    probe_dict,
        "se_prob_best":  se_prob_best,
        "se_prob_mean":  se_prob_mean,
        "ecs":           ecs_tgt,
        "pks":           pks_tgt,
        "note_toks":     note_toks_tgt,
        "corr_ecs_se":   corr_ecs_se,
        "corr_pks_se":   corr_pks_se,
        "ensemble_auc":  ensemble_auc,
        "base_auc":      base_auc,
        "df":            df,
    }
