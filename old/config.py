"""
config.py
=========
Central configuration dataclass, shared constants, and ACI-Bench data loading.

This is a leaf module with no internal project dependencies.
"""

import textwrap
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from datasets import load_dataset

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class Config:
    """Central configuration.  Override via CLI in run_experiments.py."""

    # ── Model ──────────────────────────────────────────────────────────────────
    model_name: str = "google/gemma-2-2b-it"

    # ── Device ─────────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # dtype: bfloat16 on GPU halves memory; float32 on CPU for stability
    dtype: torch.dtype = field(
        default_factory=lambda: torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    # ── ECS computation ────────────────────────────────────────────────────────
    ecs_layers: str = "all"   # "all" | "last_half" (kept for compatibility)

    # ── PKS computation ────────────────────────────────────────────────────────
    pks_method: str = "jsd"   # paper-correct: JSD over LogitLens distributions

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset_repo:   str = "mkieffer/ACI-Bench-MedARC"
    dataset_config: str = "aci"
    dataset_split:  str = "test2"
    sample_idx:     int = 1

    # ── Generation (Experiment 2b) ─────────────────────────────────────────────
    max_new_tokens:  int   = 512
    gen_temperature: float = 0.5

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: str = "."


# ─────────────────────────────────────────────
# Shared colour palette (used by all plots)
# ─────────────────────────────────────────────

Q_COLORS = {
    "extractive":    "#2196F3",   # blue
    "parametric":    "#FF9800",   # orange
    "synthesized":   "#4CAF50",   # green
    "hallucinatory": "#F44336",   # red
}


# ─────────────────────────────────────────────
# ACI-Bench data loading
# ─────────────────────────────────────────────

_TRANSCRIPT_CANDIDATES = ["src", "dialogue", "conversation", "transcript", "input"]
_NOTE_CANDIDATES        = ["tgt", "note", "reference", "summary", "output"]


def _pick_column(columns: List[str], candidates: List[str], role: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
        f"Cannot find {role} column in dataset.  "
        f"Available columns: {columns}.  Expected one of: {candidates}"
    )


def load_aci_sample(cfg: Config) -> Tuple[str, str]:
    """
    Load one (transcript, gold_note) pair from ACI-Bench (HuggingFace).

    Returns
    -------
    transcript : patient-clinician dialogue
    gold_note  : reference SOAP note
    """
    print(f"  Loading {cfg.dataset_repo}  "
          f"config='{cfg.dataset_config}'  split='{cfg.dataset_split}'  "
          f"idx={cfg.sample_idx} …")

    ds   = load_dataset(cfg.dataset_repo, cfg.dataset_config, split=cfg.dataset_split)
    cols = ds.column_names
    print(f"  Dataset columns : {cols}")
    print(f"  Rows in split   : {len(ds)}")

    t_col = _pick_column(cols, _TRANSCRIPT_CANDIDATES, "transcript")
    n_col = _pick_column(cols, _NOTE_CANDIDATES,       "note")
    print(f"  Using columns   : transcript='{t_col}'  note='{n_col}'")

    row        = ds[cfg.sample_idx]
    transcript = row[t_col]
    gold_note  = row[n_col]

    print(f"\n  ── Transcript preview (first 300 chars) ──")
    print(textwrap.indent(transcript[:300].strip(), "    "))
    print(f"\n  ── Gold note preview (first 300 chars) ──")
    print(textwrap.indent(gold_note[:300].strip(), "    "))
    print()

    return transcript, gold_note
