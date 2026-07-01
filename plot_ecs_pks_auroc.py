"""
plot_ecs_pks_auroc.py
---------------------
For each transformer layer, computes AUROC and Cohen's d for three signals:
  - ECS       : entropy of copying scores (all heads, mean-pooled across heads)
  - ECS_copy  : ECS restricted to copying heads (via copying_head_mask)
  - PKS       : probability of key sharing

Binary target: labels=1 (uncertain sentence) vs labels=0 (certain sentence)
from the npz files in activations/.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────
ACTIVATIONS_DIR = Path(__file__).parent / "activations"
MASK_PATH       = Path(__file__).parent / "copying_head_mask.npy"
OUT_PATH        = Path(__file__).parent / "ecs_pks_auroc_cohend.png"

# ── Load ───────────────────────────────────────────────────────────────────
mask = np.load(MASK_PATH)          # (n_layers, n_heads) bool

ecs_all      = []
ecs_copy_all = []
pks_all      = []
labels_all   = []

for path in sorted(ACTIVATIONS_DIR.glob("*.npz")):
    try:
        d = np.load(path)
    except Exception:
        print(f"  [skip] corrupt: {path.name}")
        continue
    ecs_all.append(d["ecs"])           # (n_layers, n_sent)
    ecs_copy_all.append(d["ecs_copy"]) # (n_layers, n_sent)
    pks_all.append(d["pks"])           # (n_layers, n_sent)
    labels_all.append(d["labels"])     # (n_sent,)

# Concatenate across all generations
ecs      = np.concatenate(ecs_all,      axis=1)  # (n_layers, total_sent)
ecs_copy = np.concatenate(ecs_copy_all, axis=1)
pks      = np.concatenate(pks_all,      axis=1)
labels   = np.concatenate(labels_all)             # (total_sent,)

n_layers = ecs.shape[0]
print(f"Total sentences: {len(labels)}  |  uncertain: {labels.sum()}  |  certain: {(labels==0).sum()}")

# ── Helpers ────────────────────────────────────────────────────────────────
def cohen_d(a, b):
    """Cohen's d: (mean_a - mean_b) / pooled std."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    pooled = np.sqrt(((n_a - 1) * a.std(ddof=1)**2 + (n_b - 1) * b.std(ddof=1)**2) / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def per_layer_metrics(scores):
    """
    scores: (n_layers, total_sent)
    Returns (aurocs, cohens_ds) each of length n_layers.

    AUROC is always reported as the discriminability above chance:
    max(auroc, 1-auroc). This is correct here because lower ECS / PKS
    predicts uncertainty (label=1), so the raw AUROC is < 0.5; flipping
    the score gives the equivalent above-chance value.

    Cohen's d is (uncertain − certain): negative values mean uncertain
    sentences have lower ECS/PKS, which is the expected direction.
    """
    aurocs, ds = [], []
    unc = labels == 1
    cer = labels == 0
    for layer in range(scores.shape[0]):
        s = scores[layer]
        try:
            auroc = roc_auc_score(labels, s)
            auroc = max(auroc, 1 - auroc)   # fold to above-chance
        except ValueError:
            auroc = float("nan")
        aurocs.append(auroc)
        ds.append(cohen_d(s[unc], s[cer]))
    return np.array(aurocs), np.array(ds)


# ── Compute ────────────────────────────────────────────────────────────────
auroc_ecs,      cd_ecs      = per_layer_metrics(ecs)
auroc_ecs_copy, cd_ecs_copy = per_layer_metrics(ecs_copy)
auroc_pks,      cd_pks      = per_layer_metrics(pks)

layers = np.arange(n_layers)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ECS / PKS as predictors of sentence-level uncertainty", fontsize=13)

# AUROC
for vals, label, style in [
    (auroc_ecs,      "ECS (all heads)",     "-"),
    (auroc_ecs_copy, "ECS (copying heads)", "--"),
    (auroc_pks,      "PKS",                 ":"),
]:
    ax1.plot(layers, vals, style, linewidth=1.8, label=label)
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="chance")
ax1.set_xlabel("Layer")
ax1.set_ylabel("AUROC")
ax1.set_title("AUROC per layer")
ax1.legend(fontsize=9)
ax1.set_xlim(0, n_layers - 1)
ax1.set_ylim(0.3, 1.0)
ax1.grid(True, alpha=0.3)

# Cohen's d
for vals, label, style in [
    (cd_ecs,      "ECS (all heads)",     "-"),
    (cd_ecs_copy, "ECS (copying heads)", "--"),
    (cd_pks,      "PKS",                 ":"),
]:
    ax2.plot(layers, vals, style, linewidth=1.8, label=label)
ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax2.set_xlabel("Layer")
ax2.set_ylabel("Cohen's d")
ax2.set_title("Cohen's d per layer (uncertain − certain)")
ax2.legend(fontsize=9)
ax2.set_xlim(0, n_layers - 1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved to {OUT_PATH}")
plt.show()
