"""
redeep_clinical.py
==================
Experiments 2a and 2b: ECS/PKS-based factuality analysis for AI clinical note generation.
Applies the REDEEP framework (External Context Score + Parametric Knowledge Score) to
distinguish extractive from abstractive token generation, and to detect hallucination risk.

Data source:
    mkieffer/ACI-Bench-MedARC  —  config "aci",  split "test1"
    https://huggingface.co/datasets/mkieffer/ACI-Bench-MedARC
    Column auto-detection handles both the standard ACI-Bench schema
    (columns: src / tgt) and any alternate naming used by this HF variant.

Experiment 2a — Token-level ECS/PKS mapping
    For every token in the gold reference SOAP note, compute ECS (how much the
    model drew on the patient transcript) and PKS (how much it drew on parametric
    medical knowledge).  Produces a quadrant scatter, a heatmap, and a per-token
    hallucination risk chart.

Experiment 2b — Contrastive gold reference vs model-generated note
    Generates a SOAP note from the transcript using the loaded model, then
    compares its ECS/PKS profile against the gold reference note.
    Model-generated notes contain natural hallucinations; the hypothesis is
    that tokens unique to the generated note (absent from the reference) will
    cluster in the Low-ECS + Low-PKS quadrant.
    Produces side-by-side scatter plots, line-trace comparisons, a delta
    heatmap, and a token-level CSV.

Supported models (white-box access required):
    "google/gemma-2-2b"          ~5 GB VRAM / ~10 GB RAM
    "meta-llama/Meta-Llama-3-8B" ~16 GB VRAM / ~32 GB RAM
    Any model supported by TransformerLens.

Usage:
    pip install -r requirements_redeep.txt
    python redeep_clinical.py                         # gemma-2-2b, sample 0
    python redeep_clinical.py --model llama            # Llama 3 8B
    python redeep_clinical.py --exp 2a                 # only Exp 2a
    python redeep_clinical.py --sample 3               # use row index 3 from test1
    python redeep_clinical.py --max-new-tokens 300     # control generation length
"""

import argparse
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

@dataclass
class Config:
    """Central configuration.  Override at the bottom of this file or via CLI."""

    # ── Model ──────────────────────────────────────────────────────────────────
    # "google/gemma-2-2b"  is faster and fits on most GPUs / large-RAM laptops.
    # "meta-llama/Meta-Llama-3-8B"  gives better clinical coverage.
    model_name: str = "google/gemma-2-2b"

    # ── Device ─────────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # dtype: use bfloat16 on GPU to halve memory; float32 on CPU for stability
    dtype: torch.dtype = field(
        default_factory=lambda: torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    # ── ECS computation ────────────────────────────────────────────────────────
    # "all"       — all layers (thorough, slower)
    # "last_half" — top half only (faster; captures richer contextual signal)
    ecs_layers: str = "all"

    # ── PKS computation ────────────────────────────────────────────────────────
    # "norm_ratio" — ||MLP_out|| / (||MLP_out|| + ||Attn_out||) per layer, averaged
    # "resid_frac" — ||MLP_out|| / ||resid_post|| per layer, averaged
    pks_method: str = "norm_ratio"

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset_repo:   str = "mkieffer/ACI-Bench-MedARC"
    dataset_config: str = "aci"
    dataset_split:  str = "test1"
    sample_idx:     int = 0   # which row from the split to use

    # ── Generation (Experiment 2b) ─────────────────────────────────────────────
    max_new_tokens: int = 256   # max tokens for model-generated note
    gen_temperature: float = 0.0  # 0.0 = greedy; increase for sampling

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: str = "."


# ─────────────────────────────────────────────
# 2. ACI-Bench data loading
# ─────────────────────────────────────────────

# ACI-Bench column names vary slightly across HF versions.
# This mapping lists candidates in priority order; the loader picks the
# first name that actually exists in the dataset.
_TRANSCRIPT_CANDIDATES = ["src", "dialogue", "conversation", "transcript", "input"]
_NOTE_CANDIDATES        = ["tgt", "note", "reference", "summary", "output"]


def _pick_column(columns: List[str], candidates: List[str], role: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
        f"Cannot find {role} column in dataset.  "
        f"Available columns: {columns}.  "
        f"Expected one of: {candidates}"
    )


def load_aci_sample(cfg: Config) -> Tuple[str, str]:
    """
    Load one (transcript, gold_note) pair from ACI-Bench.

    Returns
    -------
    transcript : the patient-clinician dialogue
    gold_note  : the reference SOAP note
    """
    print(f"  Loading {cfg.dataset_repo}  "
          f"config='{cfg.dataset_config}'  split='{cfg.dataset_split}'  "
          f"idx={cfg.sample_idx} …")

    ds = load_dataset(cfg.dataset_repo, cfg.dataset_config, split=cfg.dataset_split)

    cols = ds.column_names
    print(f"  Dataset columns : {cols}")
    print(f"  Rows in split   : {len(ds)}")

    t_col = _pick_column(cols, _TRANSCRIPT_CANDIDATES, "transcript")
    n_col = _pick_column(cols, _NOTE_CANDIDATES,       "note")
    print(f"  Using columns   : transcript='{t_col}'  note='{n_col}'")

    row = ds[cfg.sample_idx]
    transcript = row[t_col]
    gold_note  = row[n_col]

    # Pretty-print a preview
    print(f"\n  ── Transcript preview (first 300 chars) ──")
    print(textwrap.indent(transcript[:300].strip(), "    "))
    print(f"\n  ── Gold note preview (first 300 chars) ──")
    print(textwrap.indent(gold_note[:300].strip(), "    "))
    print()

    return transcript, gold_note


# ─────────────────────────────────────────────
# 3. Tokenisation helper
# ─────────────────────────────────────────────

def tokenize_pair(
    model: HookedTransformer,
    transcript: str,
    note: str,
) -> Tuple[torch.Tensor, int, List[str]]:
    """
    Concatenate transcript and note into one token sequence.

    Returns
    -------
    tokens          : LongTensor of shape (1, transcript_len + note_len)
    transcript_len  : number of tokens belonging to the transcript prefix
    note_str_tokens : decoded string for each note token (for labelling plots)
    """
    transcript_tok = model.to_tokens(transcript, prepend_bos=True)   # (1, T)
    note_tok       = model.to_tokens(note,       prepend_bos=False)  # (1, N)

    combined       = torch.cat([transcript_tok, note_tok], dim=1)     # (1, T+N)
    transcript_len = transcript_tok.shape[1]

    note_str = [model.tokenizer.decode([t.item()]) for t in note_tok[0]]

    return combined, transcript_len, note_str


# ─────────────────────────────────────────────
# 4. Model-based note generation (for Exp 2b)
# ─────────────────────────────────────────────

# Prompt template for SOAP note generation.
# Kept simple so any instruction-following LLM can follow it.
_GENERATION_PROMPT = (
    "You are a clinical documentation assistant.  "
    "Given the following patient-clinician conversation, write a concise SOAP note.\n\n"
    "### Conversation\n{transcript}\n\n"
    "### SOAP Note\n"
)


def generate_note(
    model: HookedTransformer,
    transcript: str,
    cfg: Config,
) -> str:
    """
    Autoregressively generate a SOAP note from the transcript using the loaded
    TransformerLens model.

    The generated text is what Experiment 2b treats as the "model note" —
    it is compared against the gold reference note from ACI-Bench.
    Any hallucinations present are natural model outputs, not synthetic.

    Notes on greedy vs sampling
    ---------------------------
    cfg.gen_temperature == 0.0 → greedy decoding (reproducible, but may be
        repetitive for weaker base models).
    cfg.gen_temperature  > 0.0 → sampling (set to ~0.7 for more varied output).
    """
    prompt  = _GENERATION_PROMPT.format(transcript=transcript.strip())
    input_ids = model.to_tokens(prompt, prepend_bos=True).to(cfg.device)  # (1, P)

    print(f"  Generating note  (max_new_tokens={cfg.max_new_tokens}, "
          f"temperature={cfg.gen_temperature}) …")

    with torch.no_grad():
        if cfg.gen_temperature == 0.0:
            # Greedy via TransformerLens built-in generate
            output_ids = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=1.0,        # ignored when do_sample=False
                do_sample=False,
                verbose=False,
            )
        else:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.gen_temperature,
                do_sample=True,
                verbose=False,
            )

    # Decode only the newly generated tokens (strip the prompt)
    prompt_len    = input_ids.shape[1]
    new_token_ids = output_ids[0, prompt_len:]
    generated     = model.tokenizer.decode(new_token_ids.tolist(), skip_special_tokens=True)

    print(f"\n  ── Generated note preview (first 300 chars) ──")
    print(textwrap.indent(generated[:300].strip(), "    "))
    print()

    return generated


# ─────────────────────────────────────────────
# 5. ECS / PKS computation
# ─────────────────────────────────────────────

def compute_ecs_pks(
    model: HookedTransformer,
    tokens: torch.Tensor,
    transcript_len: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute External Context Score and Parametric Knowledge Score for every note token.

    ECS[i]
        Mean (over layers × heads) of the total attention weight that note token i
        places on transcript tokens.  Ranges [0, 1].
        High → the token is strongly grounded in the transcript (extractive signal).

    PKS[i]
        Mean (over layers) of MLP_out's share of the combined MLP + Attn output norm.
        Ranges [0, 1].
        High → parametric knowledge (MLP weights) is driving the prediction (abstractive
        / medical-reasoning signal).

    Quadrant interpretation
    ───────────────────────
        High ECS, Low  PKS  → Extractive        (copying from transcript)
        Low  ECS, High PKS  → Parametric        (drawing on medical knowledge)
        High ECS, High PKS  → Synthesized       (grounded reasoning)
        Low  ECS, Low  PKS  → Hallucination risk (neither source is driving output)
    """
    n_layers   = model.cfg.n_layers
    n_heads    = model.cfg.n_heads
    seq_len    = tokens.shape[1]
    note_len   = seq_len - transcript_len

    assert note_len > 0, "note_len must be > 0: check tokenisation"

    # ── Forward pass with full activation cache ──────────────────────────────
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: (
                "pattern" in name or
                "mlp_out" in name or
                "attn_out" in name
            ),
        )

    # ── Layer range for ECS ───────────────────────────────────────────────────
    if cfg.ecs_layers == "last_half":
        ecs_layers = range(n_layers // 2, n_layers)
    else:
        ecs_layers = range(n_layers)

    # ── ECS ───────────────────────────────────────────────────────────────────
    # cache["pattern", L] : (batch=1, n_heads, seq, seq)
    ecs = np.zeros(note_len, dtype=np.float64)

    for layer in ecs_layers:
        # attn_pat[head, query, key]
        attn_pat = cache["pattern", layer][0].float().cpu().numpy()  # (H, S, S)

        for k in range(note_len):
            q = transcript_len + k   # absolute position of this note token
            # sum of weights this query places on all transcript key positions
            context_weight = attn_pat[:, q, :transcript_len].sum(axis=-1)  # (H,)
            ecs[k] += context_weight.mean()

    ecs /= len(ecs_layers)

    # ── PKS ───────────────────────────────────────────────────────────────────
    # cache["mlp_out",  L] : (batch=1, seq, d_model)
    # cache["attn_out", L] : (batch=1, seq, d_model)
    pks = np.zeros(note_len, dtype=np.float64)

    for layer in range(n_layers):
        mlp_out  = cache["mlp_out",  layer][0].float().cpu().numpy()  # (S, D)
        attn_out = cache["attn_out", layer][0].float().cpu().numpy()  # (S, D)

        for k in range(note_len):
            q = transcript_len + k
            mlp_norm  = float(np.linalg.norm(mlp_out[q]))
            attn_norm = float(np.linalg.norm(attn_out[q]))
            denom     = mlp_norm + attn_norm
            if denom > 1e-8:
                if cfg.pks_method == "norm_ratio":
                    pks[k] += mlp_norm / denom
                else:  # resid_frac — normalise by residual stream norm instead
                    resid = cache["resid_post", layer][0, q].float().cpu().numpy()
                    resid_norm = float(np.linalg.norm(resid))
                    pks[k] += mlp_norm / (resid_norm + 1e-8)

    pks /= n_layers

    # Clip to [0, 1] to handle any floating-point edge cases
    ecs = np.clip(ecs, 0.0, 1.0)
    pks = np.clip(pks, 0.0, 1.0)

    return ecs, pks


# ─────────────────────────────────────────────
# 6. Derived metric: hallucination risk
# ─────────────────────────────────────────────

def hallucination_risk(ecs: np.ndarray, pks: np.ndarray) -> np.ndarray:
    """
    Scalar risk score per token.  Highest when both ECS and PKS are low.
    risk = 1 - (ECS + PKS) / 2
    """
    return np.clip(1.0 - (ecs + pks) / 2.0, 0.0, 1.0)


def quadrant_stats(ecs: np.ndarray, pks: np.ndarray, label: str) -> Dict:
    """
    Summarise the fraction of tokens in each ECS/PKS quadrant.
    Uses per-array medians as the quadrant thresholds so the split is
    always 50/50 on each axis — making cross-model comparison fair.
    """
    em = np.median(ecs)
    pm = np.median(pks)
    n  = len(ecs)

    hi_ecs = ecs >= em
    hi_pks = pks >= pm

    stats = {
        "label":               label,
        "n_tokens":            n,
        "extractive_frac":     float(np.mean( hi_ecs & ~hi_pks)),
        "parametric_frac":     float(np.mean(~hi_ecs &  hi_pks)),
        "synthesized_frac":    float(np.mean( hi_ecs &  hi_pks)),
        "hallucinatory_frac":  float(np.mean(~hi_ecs & ~hi_pks)),
        "mean_ecs":            float(ecs.mean()),
        "mean_pks":            float(pks.mean()),
        "mean_risk":           float(hallucination_risk(ecs, pks).mean()),
    }

    print(f"\n{'─'*54}")
    print(f"  {label}")
    print(f"{'─'*54}")
    width = 30
    for k, v in stats.items():
        if k == "label":
            continue
        print(f"  {k:<{width}} {v:.4f}" if isinstance(v, float) else f"  {k:<{width}} {v}")

    return stats


# ─────────────────────────────────────────────
# 7. Visualisation helpers
# ─────────────────────────────────────────────

# Quadrant colour scheme (consistent across all plots)
Q_COLORS = {
    "extractive":    "#2196F3",   # blue
    "parametric":    "#FF9800",   # orange
    "synthesized":   "#4CAF50",   # green
    "hallucinatory": "#F44336",   # red
}

def _token_color(e: float, p: float, em: float, pm: float) -> str:
    if   e >= em and p <  pm: return Q_COLORS["extractive"]
    elif e <  em and p >= pm: return Q_COLORS["parametric"]
    elif e >= em and p >= pm: return Q_COLORS["synthesized"]
    else:                      return Q_COLORS["hallucinatory"]


def plot_scatter(
    ecs: np.ndarray,
    pks: np.ndarray,
    tokens: List[str],
    title: str,
    ax: Optional[plt.Axes] = None,
    highlight: Optional[List[int]] = None,
    annotate_stride: int = 5,
) -> plt.Axes:
    """
    ECS vs PKS scatter.  Each point is one note token, coloured by quadrant.
    Optional `highlight` list marks specific positions with a star (for 2b).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    em = np.median(ecs)
    pm = np.median(pks)

    colors = [_token_color(e, p, em, pm) for e, p in zip(ecs, pks)]
    ax.scatter(ecs, pks, c=colors, s=55, alpha=0.75, edgecolors="white", linewidth=0.4)

    # Annotate every N-th token
    for i, (e, p, tok) in enumerate(zip(ecs, pks, tokens)):
        if i % annotate_stride == 0:
            label = tok.replace("▁", "").replace("Ġ", "").strip()[:10]
            ax.annotate(label, (e, p), fontsize=5.5, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")

    # Highlight specific positions (contrastive experiment)
    if highlight:
        hx = [ecs[i] for i in highlight if i < len(ecs)]
        hy = [pks[i] for i in highlight if i < len(pks)]
        ax.scatter(hx, hy, s=200, c="black", marker="*", zorder=6,
                   label="Hallucinated span")
        ax.legend(fontsize=8)

    # Quadrant dividers
    ax.axvline(em, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(pm, color="gray", ls="--", lw=0.8, alpha=0.5)

    # Quadrant labels (placed near each corner)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    pad = 0.02
    ax.text(xhi - pad*(xhi-xlo), ylo + pad*(yhi-ylo), "Extractive",
            ha="right", va="bottom", fontsize=7, color=Q_COLORS["extractive"], style="italic")
    ax.text(xlo + pad*(xhi-xlo), yhi - pad*(yhi-ylo), "Parametric",
            ha="left",  va="top",    fontsize=7, color=Q_COLORS["parametric"],  style="italic")
    ax.text(xhi - pad*(xhi-xlo), yhi - pad*(yhi-ylo), "Synthesized",
            ha="right", va="top",    fontsize=7, color=Q_COLORS["synthesized"], style="italic")
    ax.text(xlo + pad*(xhi-xlo), ylo + pad*(yhi-ylo), "Hallucination\nRisk",
            ha="left",  va="bottom", fontsize=7, color=Q_COLORS["hallucinatory"], style="italic")

    legend_patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in Q_COLORS.items()]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right")

    ax.set_xlabel("External Context Score (ECS)", fontsize=10)
    ax.set_ylabel("Parametric Knowledge Score (PKS)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    return ax


def plot_heatmap(
    ecs: np.ndarray,
    pks: np.ndarray,
    tokens: List[str],
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Two-row heatmap: top row = ECS, bottom row = PKS, columns = note tokens."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(14, len(tokens) // 3), 3))

    disp = [t.replace("▁", "").replace("Ġ", "").strip()[:7] for t in tokens]
    data = np.stack([ecs, pks])   # (2, N)

    sns.heatmap(data, ax=ax, xticklabels=disp, yticklabels=["ECS", "PKS"],
                cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.25, linecolor="white", cbar_kws={"shrink": 0.6})
    ax.tick_params(axis="x", labelsize=5.5, rotation=60)
    ax.tick_params(axis="y", labelsize=9,   rotation=0)
    ax.set_title(title, fontsize=10, fontweight="bold")

    return ax


def plot_risk_bar(
    ecs: np.ndarray,
    pks: np.ndarray,
    tokens: List[str],
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of hallucination risk per token, coloured by risk magnitude."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(14, len(tokens) // 3), 3))

    risk   = hallucination_risk(ecs, pks)
    disp   = [t.replace("▁", "").replace("Ġ", "").strip()[:7] for t in tokens]
    colors = plt.cm.RdYlGn_r(risk)   # green = low risk, red = high risk

    ax.bar(range(len(risk)), risk, color=colors, width=0.85, edgecolor="none")
    ax.set_xticks(range(len(disp)))
    ax.set_xticklabels(disp, rotation=60, ha="right", fontsize=5.5)
    ax.axhline(0.5, color="#B71C1C", ls="--", lw=1.2, alpha=0.7, label="Risk threshold 0.5")
    ax.set_ylabel("Hallucination Risk", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    return ax


def plot_line_trace(
    ecs_a: np.ndarray, ecs_b: np.ndarray,
    pks_a: np.ndarray, pks_b: np.ndarray,
    diff_pos: List[int],
    label_a: str, label_b: str,
    axes,
) -> None:
    """
    Line traces of ECS and PKS across token positions for two notes.
    Vertical grey dashed lines mark positions where the notes differ.
    """
    L = min(len(ecs_a), len(ecs_b))

    # ECS trace
    axes[0].plot(ecs_a[:L], label=label_a, color="#2196F3", lw=1.4, alpha=0.9)
    axes[0].plot(ecs_b[:L], label=label_b, color="#F44336", lw=1.4, alpha=0.9, ls="--")
    for p in diff_pos:
        if p < L:
            axes[0].axvline(p, color="gray", ls=":", lw=0.8, alpha=0.5)
    axes[0].set_ylabel("ECS", fontsize=10)
    axes[0].set_xlabel("Note token position", fontsize=9)
    axes[0].set_title("ECS across token positions", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8)

    # PKS trace
    axes[1].plot(pks_a[:L], label=label_a, color="#FF9800", lw=1.4, alpha=0.9)
    axes[1].plot(pks_b[:L], label=label_b, color="#9C27B0", lw=1.4, alpha=0.9, ls="--")
    for p in diff_pos:
        if p < L:
            axes[1].axvline(p, color="gray", ls=":", lw=0.8, alpha=0.5)
    axes[1].set_ylabel("PKS", fontsize=10)
    axes[1].set_xlabel("Note token position", fontsize=9)
    axes[1].set_title("PKS across token positions", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8)


# ─────────────────────────────────────────────
# 8. Differing-position detection (for 2b)
# ─────────────────────────────────────────────

def find_differing_positions(
    tokens_a: List[str],
    tokens_b: List[str],
    top_k: int = 30,
) -> List[int]:
    """
    Simple character-level comparison to find token positions that differ
    between two sequences of the same approximate length.
    Returns the first `top_k` differing positions (shared index space).
    """
    min_len = min(len(tokens_a), len(tokens_b))
    diffs = [
        i for i in range(min_len)
        if tokens_a[i].strip().lower() != tokens_b[i].strip().lower()
        and tokens_a[i].strip() and tokens_b[i].strip()
    ]
    return diffs[:top_k]


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

    Parameters
    ----------
    transcript : patient-clinician dialogue (from ACI-Bench)
    gold_note  : reference SOAP note (from ACI-Bench)

    Outputs
    -------
    exp2a_scatter.png   — quadrant scatter of all note tokens
    exp2a_heatmap.png   — ECS/PKS heatmap across token positions
    exp2a_risk.png      — per-token hallucination risk bar chart
    """
    print("\n" + "═"*54)
    print("  EXPERIMENT 2a — ECS/PKS Token-Level Mapping")
    print("═"*54)

    tokens, t_len, note_toks = tokenize_pair(model, transcript, gold_note)
    tokens = tokens.to(cfg.device)
    print(f"  Transcript : {t_len} tokens")
    print(f"  Note       : {len(note_toks)} tokens")

    print("  Running forward pass + computing ECS / PKS …")
    ecs, pks = compute_ecs_pks(model, tokens, t_len, cfg)

    stats = quadrant_stats(ecs, pks, f"2a — Gold Note (ACI-Bench sample {cfg.sample_idx})")

    # ── Scatter ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    plot_scatter(ecs, pks, note_toks,
                 f"Exp 2a — ECS vs PKS: Clinical Note Tokens\n({cfg.model_name})",
                 ax=ax, annotate_stride=4)
    plt.tight_layout()
    fig.savefig(out / "exp2a_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2a_scatter.png")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(16, len(note_toks) // 2), 3))
    plot_heatmap(ecs, pks, note_toks,
                 "Exp 2a — ECS/PKS Heatmap across Note Tokens",
                 ax=ax)
    plt.tight_layout()
    fig.savefig(out / "exp2a_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2a_heatmap.png")

    # ── Risk bar ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(16, len(note_toks) // 2), 3))
    plot_risk_bar(ecs, pks, note_toks,
                  "Exp 2a — Per-Token Hallucination Risk (Low ECS + Low PKS)",
                  ax=ax)
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
    Experiment 2b: Contrastive ECS/PKS — gold reference vs model-generated note.

    The model is used to generate a SOAP note autoregressively from the
    transcript.  Because the model has no guarantee of faithfulness, this
    generated note is the natural source of hallucinations.

    Hypothesis:
        Tokens in the generated note that do NOT appear in the gold reference
        will cluster in the Low-ECS + Low-PKS quadrant — grounded in neither
        the transcript nor correct parametric knowledge.

    Parameters
    ----------
    transcript : patient-clinician dialogue (from ACI-Bench)
    gold_note  : reference SOAP note (from ACI-Bench)

    Outputs
    -------
    exp2b_scatter_gold.png    — quadrant scatter for gold note (stars on diff positions)
    exp2b_scatter_gen.png     — quadrant scatter for generated note (stars on diff positions)
    exp2b_line_traces.png     — ECS and PKS line traces for both notes overlaid
    exp2b_delta_heatmap.png   — heatmap of ΔECS and ΔPKS at shared token positions
    exp2b_summary.csv         — numeric delta table for downstream use
    exp2b_generated_note.txt  — the raw generated note text (for inspection)
    """
    print("\n" + "═"*54)
    print("  EXPERIMENT 2b — Contrastive Gold vs Model-Generated")
    print("═"*54)

    # ── Generate note from transcript ─────────────────────────────────────────
    generated_note = generate_note(model, transcript, cfg)

    # Save generated note for inspection
    (out / "exp2b_generated_note.txt").write_text(generated_note, encoding="utf-8")
    print("  Saved → exp2b_generated_note.txt")

    # ── Tokenise ──────────────────────────────────────────────────────────────
    tok_gold, tl_gold, nt_gold = tokenize_pair(model, transcript, gold_note)
    tok_gen,  tl_gen,  nt_gen  = tokenize_pair(model, transcript, generated_note)
    tok_gold = tok_gold.to(cfg.device)
    tok_gen  = tok_gen.to(cfg.device)

    print(f"  Gold note      : {len(nt_gold)} tokens")
    print(f"  Generated note : {len(nt_gen)} tokens")

    # ── Compute scores ────────────────────────────────────────────────────────
    print("  Computing ECS/PKS for gold note …")
    ecs_acc, pks_acc = compute_ecs_pks(model, tok_gold, tl_gold, cfg)
    print("  Computing ECS/PKS for generated note …")
    ecs_hal, pks_hal = compute_ecs_pks(model, tok_gen, tl_gen, cfg)

    # Alias for the rest of the function (keeping internal var names stable)
    nt_acc, nt_hal = nt_gold, nt_gen

    stats_acc = quadrant_stats(ecs_acc, pks_acc, f"2b — Gold Note (sample {cfg.sample_idx})")
    stats_hal = quadrant_stats(ecs_hal, pks_hal, f"2b — Generated Note (sample {cfg.sample_idx})")

    # ── Delta summary ─────────────────────────────────────────────────────────
    print(f"\n  Δ (Hallucinated − Accurate)")
    print(f"  {'metric':<30} {'Δ':>8}")
    print(f"  {'─'*38}")
    delta_keys = ["mean_ecs", "mean_pks", "mean_risk",
                  "extractive_frac", "parametric_frac",
                  "synthesized_frac", "hallucinatory_frac"]
    for k in delta_keys:
        delta = stats_hal[k] - stats_acc[k]
        print(f"  {k:<30} {delta:>+8.4f}")

    # ── Find differing positions ──────────────────────────────────────────────
    diff_pos = find_differing_positions(nt_acc, nt_hal)
    print(f"\n  Detected {len(diff_pos)} differing token positions")

    # Report ECS/PKS shift at each differing position
    print(f"\n  {'pos':>4}  {'acc_tok':>12}  {'hal_tok':>12}  "
          f"{'ECS_acc':>8}  {'ECS_hal':>8}  {'ΔECS':>8}  "
          f"{'PKS_acc':>8}  {'PKS_hal':>8}  {'ΔPKS':>8}")
    print(f"  {'─'*90}")
    rows = []
    for p in diff_pos:
        ta  = nt_acc[p].strip()[:12] if p < len(nt_acc) else "–"
        th  = nt_hal[p].strip()[:12] if p < len(nt_hal) else "–"
        ea  = ecs_acc[p] if p < len(ecs_acc) else float("nan")
        eh  = ecs_hal[p] if p < len(ecs_hal) else float("nan")
        pa_ = pks_acc[p] if p < len(pks_acc) else float("nan")
        ph  = pks_hal[p] if p < len(pks_hal) else float("nan")
        de  = eh - ea
        dp  = ph - pa_
        print(f"  {p:>4}  {ta:>12}  {th:>12}  {ea:>8.3f}  {eh:>8.3f}  {de:>+8.3f}  "
              f"{pa_:>8.3f}  {ph:>8.3f}  {dp:>+8.3f}")
        rows.append({"pos": p, "acc_tok": ta, "hal_tok": th,
                     "ecs_acc": ea, "ecs_hal": eh, "delta_ecs": de,
                     "pks_acc": pa_, "pks_hal": ph, "delta_pks": dp})

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp2b_summary.csv", index=False)
    print("  Saved → exp2b_summary.csv")

    # ── Scatter plots (gold) ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    plot_scatter(ecs_acc, pks_acc, nt_acc,
                 f"Exp 2b — Gold Reference Note\n({cfg.model_name})",
                 ax=ax, highlight=diff_pos, annotate_stride=5)
    plt.tight_layout()
    fig.savefig(out / "exp2b_scatter_gold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2b_scatter_gold.png")

    # ── Scatter plots (generated) ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    plot_scatter(ecs_hal, pks_hal, nt_hal,
                 f"Exp 2b — Model-Generated Note\n({cfg.model_name})",
                 ax=ax, highlight=diff_pos, annotate_stride=5)
    plt.tight_layout()
    fig.savefig(out / "exp2b_scatter_gen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2b_scatter_gen.png")

    # ── Line traces ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plot_line_trace(
        ecs_acc, ecs_hal, pks_acc, pks_hal, diff_pos,
        "Gold Reference", "Model Generated", axes,
    )
    fig.suptitle(f"Exp 2b — ECS/PKS Line Traces  |  ★ = differing positions  ({cfg.model_name})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "exp2b_line_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2b_line_traces.png")

    # ── Δ ECS / Δ PKS heatmap at shared positions ─────────────────────────────
    L = min(len(ecs_acc), len(ecs_hal))
    delta_ecs = ecs_hal[:L] - ecs_acc[:L]
    delta_pks = pks_hal[:L] - pks_acc[:L]
    shared_toks = nt_acc[:L]
    disp = [t.replace("▁", "").replace("Ġ", "").strip()[:7] for t in shared_toks]

    fig, ax = plt.subplots(figsize=(max(16, L // 2), 3))
    data = np.stack([delta_ecs, delta_pks])
    sns.heatmap(data, ax=ax, xticklabels=disp,
                yticklabels=["ΔECS", "ΔPKS"],
                cmap="RdBu_r", center=0,
                linewidths=0.2, linecolor="white",
                cbar_kws={"shrink": 0.6, "label": "Hallucinated − Accurate"})
    ax.tick_params(axis="x", labelsize=5.5, rotation=60)
    ax.tick_params(axis="y", labelsize=9,   rotation=0)
    ax.set_title("Exp 2b — ΔECS and ΔPKS (Hallucinated − Accurate)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "exp2b_delta_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp2b_delta_heatmap.png")

    return {
        "ecs_acc": ecs_acc, "pks_acc": pks_acc,
        "ecs_hal": ecs_hal, "pks_hal": pks_hal,
        "stats_acc": stats_acc, "stats_hal": stats_hal,
        "diff_pos": diff_pos,
    }


# ─────────────────────────────────────────────
# 10. Entry point
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REDEEP ECS/PKS — clinical note experiments")
    p.add_argument("--model", choices=["gemma", "llama"], default="gemma",
                   help="gemma → google/gemma-2-2b  |  llama → meta-llama/Meta-Llama-3-8B")
    p.add_argument("--exp", choices=["2a", "2b", "both"], default="both",
                   help="Which experiment(s) to run (default: both)")
    p.add_argument("--ecs-layers", choices=["all", "last_half"], default="all",
                   help="Layers used for ECS averaging (default: all)")
    p.add_argument("--pks-method", choices=["norm_ratio", "resid_frac"], default="norm_ratio",
                   help="PKS attribution method (default: norm_ratio)")
    p.add_argument("--sample", type=int, default=0,
                   help="Row index from ACI-Bench test1 split to use (default: 0)")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max tokens for model-generated note in Exp 2b (default: 256)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for generation (0.0 = greedy, default: 0.0)")
    p.add_argument("--out", default=".", help="Output directory for plots and CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        model_name=(
            "google/gemma-2-2b"
            if args.model == "gemma"
            else "meta-llama/Meta-Llama-3-8B"
        ),
        ecs_layers=args.ecs_layers,
        pks_method=args.pks_method,
        sample_idx=args.sample,
        max_new_tokens=args.max_new_tokens,
        gen_temperature=args.temperature,
        output_dir=args.out,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}  |  dtype: {cfg.dtype}")
    print(f"  ECS layers     : {cfg.ecs_layers}")
    print(f"  PKS method     : {cfg.pks_method}")
    print(f"  Dataset        : {cfg.dataset_repo}  [{cfg.dataset_config} / {cfg.dataset_split}]")
    print(f"  Sample index   : {cfg.sample_idx}")
    print(f"  Max new tokens : {cfg.max_new_tokens}  |  temperature: {cfg.gen_temperature}")
    print(f"  Output dir     : {out.resolve()}")

    # ── Load ACI-Bench sample ─────────────────────────────────────────────────
    transcript, gold_note = load_aci_sample(cfg)

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── Run experiments ───────────────────────────────────────────────────────
    if args.exp in ("2a", "both"):
        run_experiment_2a(model, cfg, out, transcript, gold_note)

    if args.exp in ("2b", "both"):
        run_experiment_2b(model, cfg, out, transcript, gold_note)

    print("\n" + "═"*54)
    print("  Done.  All outputs written to:", out.resolve())
    print("═"*54 + "\n")


if __name__ == "__main__":
    main()