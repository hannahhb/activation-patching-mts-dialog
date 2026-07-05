"""
redeep_word_classifier.py
==========================
Trains word-level hallucination classifiers on the full per-word ReDeEP
feature vector (all-layer ECS, ECS-copy, PKS) and compares them against
single-feature (best-layer) baselines.

CV scheme: leave-N-samples-out (default N=3). The unique samples (notes) are
shuffled once with a fixed seed and chunked into groups of N; each fold holds
out one chunk's words as the test set and trains on the rest. A sample's
words are never split across train/test. This is a relaxation of strict
leave-one-sample-out CV, chosen because a single held-out sample averages
only ~15-16 hallucinated words (too few for a stable per-fold AUROC/PR-AUC);
leave-3-out gives ~45-50 held-out hallucinated words per fold.

Models:
  - full_logreg  L2 logistic regression on all layers x {ECS, ECS-copy, PKS}
  - full_hgb     HistGradientBoostingClassifier on the same feature vector
  - baseline_ecs / baseline_ecscopy / baseline_pks
                 1-feature logistic regression on the single best layer for
                 that feature (best layer chosen once from the pooled data,
                 NOT re-selected per fold — this is a mild optimism in the
                 baselines' favour, flagged in the printed output).

Reported per fold: AUROC, PR-AUC (average precision), Brier score.
Reported pooled (out-of-fold): AUROC, PR-AUC, Brier, plus ROC / PR /
reliability-diagram figures.

Usage:
    python redeep_word_classifier.py
    python redeep_word_classifier.py --leave-n 3 --seed 0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redeep_word_plots import build_word_labels, auroc_per_layer_single  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
)
from sklearn.calibration import calibration_curve

DEFAULT_ACT_DIR  = "sae_experiments/redeep_out/activations"
DEFAULT_SPAN_DIR = "sae_experiments/luq_out/llama_judge/spans"
DEFAULT_OUT_DIR  = "sae_experiments/redeep_out/word_classifier"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (mirrors redeep_word_plots.py's loop, but keeps a per-word
# sample-id for CV grouping instead of pooling immediately).
# ─────────────────────────────────────────────────────────────────────────────

def load_all_words(act_dir: Path, span_dir: Path):
    tok_files = sorted(act_dir.glob("sample_*_gen_*_tokens.npz"))
    if not tok_files:
        raise SystemExit(f"No *_tokens.npz files in {act_dir}")

    ecs_chunks, ecscopy_chunks, pks_chunks, lab_chunks, grp_chunks = [], [], [], [], []
    n_layers = None

    for tp in tok_files:
        m = re.search(r"sample_(\d+)_gen_(\d+)_tokens", tp.stem)
        if not m:
            continue
        si, k = int(m.group(1)), int(m.group(2))

        span_csv = span_dir / f"sample_{si:03d}_note_{k:02d}_span_judge.csv"
        if not span_csv.exists():
            continue

        d = np.load(str(tp), allow_pickle=True)
        needed = ("ecs_word", "ecs_word_copy", "pks_word", "word_strs")
        if any(key not in d for key in needed):
            continue

        ecs_w     = d["ecs_word"]
        ecscopy_w = d["ecs_word_copy"]
        pks_w     = d["pks_word"]
        word_strs = d["word_strs"]
        if n_layers is None:
            n_layers = ecs_w.shape[0]

        sdf = pd.read_csv(span_csv)
        note_spans = [s for s in sdf.get("note_span", pd.Series([], dtype=str)).fillna("").tolist()
                      if str(s).strip()]
        labels, valid, _, _ = build_word_labels(word_strs, note_spans)

        n_words = ecs_w.shape[1]
        keep = valid[:n_words]
        if keep.sum() == 0:
            continue

        ecs_chunks.append(ecs_w[:, keep])
        ecscopy_chunks.append(ecscopy_w[:, keep])
        pks_chunks.append(pks_w[:, keep])
        lab_chunks.append(labels[:n_words][keep])
        grp_chunks.append(np.full(int(keep.sum()), si, dtype=int))

    if not lab_chunks:
        raise SystemExit("No usable (tokens + span) pairs found.")

    ecs_all     = np.concatenate(ecs_chunks,     axis=1)   # (n_layers, N)
    ecscopy_all = np.concatenate(ecscopy_chunks, axis=1)
    pks_all     = np.concatenate(pks_chunks,     axis=1)
    lab_all     = np.concatenate(lab_chunks)                # (N,)
    grp_all     = np.concatenate(grp_chunks)                # (N,) sample idx

    return ecs_all, ecscopy_all, pks_all, lab_all, grp_all, n_layers


def build_feature_matrix(ecs_all, ecscopy_all, pks_all):
    """Concatenate [ECS layers | ECS-copy layers (non-all-NaN only) | PKS layers]
    into an (N, n_features) matrix, with column names for reference."""
    n_layers = ecs_all.shape[0]
    ecscopy_keep = ~np.all(np.isnan(ecscopy_all), axis=1)   # drop layers w/ no copying heads

    cols = [f"ecs_L{l}" for l in range(n_layers)]
    blocks = [ecs_all.T]

    cols += [f"ecscopy_L{l}" for l in range(n_layers) if ecscopy_keep[l]]
    blocks.append(ecscopy_all[ecscopy_keep].T)

    cols += [f"pks_L{l}" for l in range(n_layers)]
    blocks.append(pks_all.T)

    X = np.concatenate(blocks, axis=1)
    return X, cols, ecscopy_keep


# ─────────────────────────────────────────────────────────────────────────────
# CV
# ─────────────────────────────────────────────────────────────────────────────

def make_leave_n_out_folds(groups: np.ndarray, n_leave: int, seed: int) -> List[np.ndarray]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    return [perm[i:i + n_leave] for i in range(0, len(perm), n_leave)]


def fit_predict(model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def make_full_logreg():
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)),
    ])


def make_full_hgb():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, class_weight="balanced", random_state=0,
    )


def make_baseline_logreg():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])


def safe_metrics(y_true, p) -> Dict[str, float]:
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return {"auroc": np.nan, "pr_auc": np.nan, "brier": np.nan,
                "accuracy": np.nan, "balanced_accuracy": np.nan, "majority_baseline_acc": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan}
    pred = (p >= 0.5).astype(int)
    return {
        "auroc":  float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier":  float(brier_score_loss(y_true, p)),
        "accuracy":              float(accuracy_score(y_true, pred)),
        "balanced_accuracy":     float(balanced_accuracy_score(y_true, pred)),
        "majority_baseline_acc": float(max(y_true.mean(), 1 - y_true.mean())),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall":    float(recall_score(y_true, pred, zero_division=0)),
        "f1":        float(f1_score(y_true, pred, zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Word-level hallucination classifier, leave-N-sample-out CV")
    p.add_argument("--act-dir",  default=DEFAULT_ACT_DIR)
    p.add_argument("--span-dir", default=DEFAULT_SPAN_DIR)
    p.add_argument("--out",      default=DEFAULT_OUT_DIR)
    p.add_argument("--leave-n",  type=int, default=3, help="samples held out per fold")
    p.add_argument("--seed",     type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    act_dir, span_dir, out_dir = Path(args.act_dir), Path(args.span_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading words from {act_dir} / labels from {span_dir} ...")
    ecs_all, ecscopy_all, pks_all, lab_all, grp_all, n_layers = load_all_words(act_dir, span_dir)
    n_total, n_hallu = lab_all.shape[0], int(lab_all.sum())
    n_samples = len(np.unique(grp_all))
    print(f"Words: {n_total}  hallucinated: {n_hallu} ({n_hallu/n_total:.1%})  "
          f"samples: {n_samples}  layers: {n_layers}")

    X, feat_names, ecscopy_keep = build_feature_matrix(ecs_all, ecscopy_all, pks_all)
    print(f"Full feature matrix: {X.shape}  "
          f"(ecs:{n_layers}  ecscopy:{int(ecscopy_keep.sum())} of {n_layers}  pks:{n_layers})")

    # Best single layer per feature type, chosen ONCE on the pooled data (not
    # nested in CV) -- this is what defines the single-feature baselines.
    ecs_auroc     = auroc_per_layer_single(ecs_all,     lab_all, hallu_high=True)
    ecscopy_auroc = auroc_per_layer_single(ecscopy_all, lab_all, hallu_high=True)
    pks_auroc     = auroc_per_layer_single(pks_all,     lab_all, hallu_high=True)
    best_ecs_l     = int(np.nanargmax(ecs_auroc))
    best_ecscopy_l = int(np.nanargmax(ecscopy_auroc))
    best_pks_l     = int(np.nanargmax(pks_auroc))
    print(f"\nBaseline layers (picked on pooled data, NOT nested in CV -- mildly "
          f"optimistic for the baselines):")
    print(f"  ECS      layer {best_ecs_l}  (pooled AUROC {ecs_auroc[best_ecs_l]:.3f})")
    print(f"  ECS-copy layer {best_ecscopy_l}  (pooled AUROC {ecscopy_auroc[best_ecscopy_l]:.3f})")
    print(f"  PKS      layer {best_pks_l}  (pooled AUROC {pks_auroc[best_pks_l]:.3f})")

    baseline_feats = {
        "baseline_ecs":     ecs_all[best_ecs_l].reshape(-1, 1),
        "baseline_ecscopy": ecscopy_all[best_ecscopy_l].reshape(-1, 1),
        "baseline_pks":     pks_all[best_pks_l].reshape(-1, 1),
    }

    folds = make_leave_n_out_folds(grp_all, args.leave_n, args.seed)
    print(f"\n{len(folds)} folds, leave-{args.leave_n}-samples-out "
          f"(seed={args.seed}, {n_samples} unique samples)")

    model_names = ["full_logreg", "full_hgb", "baseline_ecs", "baseline_ecscopy", "baseline_pks"]
    oof_pred = {m: np.full(n_total, np.nan) for m in model_names}
    fold_rows = []

    for fi, test_samples in enumerate(folds):
        test_mask  = np.isin(grp_all, test_samples)
        train_mask = ~test_mask
        y_tr, y_te = lab_all[train_mask], lab_all[test_mask]
        if y_tr.sum() == 0 or y_te.sum() == 0:
            print(f"  fold {fi}: skipped (no positives in train or test)")
            continue

        for name in model_names:
            if name == "full_logreg":
                model = make_full_logreg()
                Xf = X
            elif name == "full_hgb":
                model = make_full_hgb()
                Xf = X
            else:
                model = make_baseline_logreg()
                Xf = baseline_feats[name]

            p_te = fit_predict(model, Xf[train_mask], y_tr, Xf[test_mask])
            oof_pred[name][test_mask] = p_te
            m = safe_metrics(y_te, p_te)
            fold_rows.append({
                "fold": fi, "model": name,
                "n_test_samples": len(test_samples), "n_test_words": int(test_mask.sum()),
                "n_test_hallu": int(y_te.sum()), **m,
            })

        n_te = int(test_mask.sum())
        print(f"  fold {fi:2d}: test samples {list(test_samples)}  "
              f"n_words={n_te}  n_hallu={int(y_te.sum())}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "word_classifier_fold_metrics.csv", index=False)

    # ── Summary: mean±std across folds, plus pooled out-of-fold metrics ───────
    summary_rows = []
    for name in model_names:
        sub = fold_df[fold_df.model == name]
        p_oof = oof_pred[name]
        valid = np.isfinite(p_oof)
        pooled = safe_metrics(lab_all[valid], p_oof[valid])
        summary_rows.append({
            "model": name,
            "auroc_mean":  sub.auroc.mean(),  "auroc_std":  sub.auroc.std(),
            "pr_auc_mean": sub.pr_auc.mean(), "pr_auc_std": sub.pr_auc.std(),
            "brier_mean":  sub.brier.mean(),  "brier_std":  sub.brier.std(),
            "accuracy_mean": sub.accuracy.mean(), "accuracy_std": sub.accuracy.std(),
            "balanced_accuracy_mean": sub.balanced_accuracy.mean(),
            "balanced_accuracy_std":  sub.balanced_accuracy.std(),
            "majority_baseline_acc_mean": sub.majority_baseline_acc.mean(),
            "precision_mean": sub.precision.mean(), "precision_std": sub.precision.std(),
            "recall_mean":    sub.recall.mean(),    "recall_std":    sub.recall.std(),
            "f1_mean":        sub.f1.mean(),        "f1_std":        sub.f1.std(),
            "pooled_auroc":  pooled["auroc"],
            "pooled_pr_auc": pooled["pr_auc"],
            "pooled_brier":  pooled["brier"],
            "pooled_accuracy": pooled["accuracy"],
            "pooled_balanced_accuracy": pooled["balanced_accuracy"],
            "pooled_precision": pooled["precision"],
            "pooled_recall":    pooled["recall"],
            "pooled_f1":        pooled["f1"],
            "n_folds": len(sub),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "word_classifier_summary.csv", index=False)

    base_rate = n_hallu / n_total
    print(f"\n=== Summary (base rate = {base_rate:.3f}) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Figures (pooled out-of-fold predictions) ──────────────────────────────
    colors = {"full_logreg": "tab:purple", "full_hgb": "tab:red",
              "baseline_ecs": "steelblue", "baseline_ecscopy": "seagreen", "baseline_pks": "tomato"}

    fig, ax = plt.subplots(figsize=(6, 6))
    for name in model_names:
        p = oof_pred[name]
        valid = np.isfinite(p)
        y = lab_all[valid]
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, p[valid])
        auc = summary_df.loc[summary_df.model == name, "pooled_auroc"].iloc[0]
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Word-level hallucination detection — ROC (pooled out-of-fold)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "fig_word_classifier_roc.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    for name in model_names:
        p = oof_pred[name]
        valid = np.isfinite(p)
        y = lab_all[valid]
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y, p[valid])
        ap = summary_df.loc[summary_df.model == name, "pooled_pr_auc"].iloc[0]
        ax.plot(rec, prec, color=colors[name], lw=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(base_rate, color="grey", ls="--", lw=0.8, label=f"base rate ({base_rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Word-level hallucination detection — PR (pooled out-of-fold)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "fig_word_classifier_pr.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    for name in model_names:
        p = oof_pred[name]
        valid = np.isfinite(p)
        y = lab_all[valid]
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        frac_pos, mean_pred = calibration_curve(y, p[valid], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, color=colors[name], marker="o", ms=3, lw=1.5, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Word-level hallucination detection — calibration (pooled out-of-fold)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "fig_word_classifier_calibration.png", dpi=150); plt.close()

    # ── Feature importance for the full logistic model (last-fold refit on all data) ──
    final_logreg = make_full_logreg()
    final_logreg.fit(X, lab_all)
    coef = final_logreg.named_steps["clf"].coef_[0]
    imp_df = pd.DataFrame({"feature": feat_names, "coef": coef}).sort_values(
        "coef", key=np.abs, ascending=False)
    imp_df.to_csv(out_dir / "full_logreg_feature_importance.csv", index=False)

    print(f"\nSaved: {out_dir}/word_classifier_fold_metrics.csv, "
          f"word_classifier_summary.csv, fig_word_classifier_{{roc,pr,calibration}}.png, "
          f"full_logreg_feature_importance.csv")


if __name__ == "__main__":
    main()
