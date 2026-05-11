"""
experiments.py
==============
Compatibility shim + shared training-data helpers for Exp 3.

The individual experiment runners now live in exp2.py … exp5.py.
This module:
  • Defines the shared helpers collect_training_data() and _load_exp4_data()
    that Exp 3 needs to build its regressor.
  • Re-exports run_experiment_2a … run_experiment_5 so that existing callers
    (run_experiments.py) continue to work without modification.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

from config import Config, load_aci_sample
from metrics import compute_ecs_pks
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
# Shared helpers used by Exp 3
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
        ecs_i, pks_i, ecs_layers_i, pks_layers_i, _ = compute_ecs_pks(model, tokens, t_len, cfg)

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

    # Load raw activations — scan ALL .npz files in the activations directory,
    # not just those referenced by the CSV.  This means a larger Exp 4 run
    # (more .npz files) is fully utilised even if the CSV was written from a
    # smaller run.
    act_dir = exp4_out / "activations"
    per_sample: List[Dict] = []
    ecs_list, pks_list, mask_list = [], [], []

    if act_dir.exists():
        npz_paths = sorted(act_dir.glob("sample_*_activations.npz"))
    else:
        npz_paths = []

    for npz_path in npz_paths:
        try:
            npz = np.load(npz_path)
            si    = int(npz["sample_idx"])
            ecs_l = npz["ecs_layers"].astype(np.float64)   # (n_layers, n_tokens)
            pks_l = npz["pks_layers"].astype(np.float64)
            mask  = npz["halluc_mask"].astype(bool)
        except Exception as exc:
            print(f"  [Exp 3] Could not load {npz_path.name}: {exc}")
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
          f"{n_act} examples (activations — all .npz files in {act_dir.name}/)")

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
