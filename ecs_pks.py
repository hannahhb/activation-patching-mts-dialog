"""
ecs_pks.py
==========
REDEEP metrics library — External Context Score (ECS) and Parametric
Knowledge Score (PKS) for token-level factuality analysis in LLM outputs.

Provides
--------
  Config                  central configuration dataclass
  load_aci_sample         load one (transcript, gold_note) pair from ACI-Bench
  tokenize_pair           concatenate transcript + note into one token sequence
  generate_note           autoregressively generate a SOAP note from a transcript
  compute_ecs             ECS per note token — paper eq. (3/4)
  compute_pks             PKS per note token — paper eq. (5)
  compute_ecs_pks         single forward pass → (ecs, pks)
  hallucination_risk      scalar risk score: 1 − (ECS + PKS) / 2
  quadrant_stats          fraction of tokens in each quadrant
  Q_COLORS                quadrant colour palette (shared across all plots)
  plot_scatter            ECS vs PKS quadrant scatter
  plot_heatmap            two-row ECS / PKS heatmap across token positions
  plot_risk_bar           per-token hallucination risk bar chart

Imported by run_experiments.py; can also be used interactively.

Supported models (white-box access via TransformerLens required):
  "google/gemma-2-2b-it"          ~5 GB VRAM
  "meta-llama/Meta-Llama-3-8B"   ~16 GB VRAM
"""

import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
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
    dataset_split:  str = "test1"
    sample_idx:     int = 0

    # ── Generation (Experiment 2b) ─────────────────────────────────────────────
    max_new_tokens:  int   = 256
    gen_temperature: float = 0.0   # 0.0 = greedy

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: str = "."


# ─────────────────────────────────────────────
# 2. ACI-Bench data loading
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
    tokens          : LongTensor (1, transcript_len + note_len)
    transcript_len  : number of tokens belonging to the transcript prefix
    note_str_tokens : decoded string for each note token (for plot labels)
    """
    transcript_tok = model.to_tokens(transcript, prepend_bos=True)    # (1, T)
    note_tok       = model.to_tokens(note,       prepend_bos=False)   # (1, N)
    combined       = torch.cat([transcript_tok, note_tok], dim=1)      # (1, T+N)
    transcript_len = transcript_tok.shape[1]
    note_str       = [model.tokenizer.decode([t.item()]) for t in note_tok[0]]
    return combined, transcript_len, note_str


# ─────────────────────────────────────────────
# 4. Model-based note generation (Experiment 2b)
# ─────────────────────────────────────────────

_GENERATION_PROMPT = (
    "You are a clinical documentation assistant."
    "Given the following patient-clinician conversation, write a summary of the "
    "conversation with six sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
    "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN."
    "### Conversation\n{transcript}\n\n"
    "### Note: \n"
)


def generate_note(
    model: HookedTransformer,
    transcript: str,
    cfg: Config,
) -> str:
    """
    Autoregressively generate a SOAP note from the transcript.

    cfg.gen_temperature == 0.0  → greedy (reproducible)
    cfg.gen_temperature  > 0.0  → sampling (~0.7 for more variety)
    """
    prompt    = _GENERATION_PROMPT.format(transcript=transcript.strip())
    input_ids = model.to_tokens(prompt, prepend_bos=True).to(cfg.device)

    print(f"  Generating note  (max_new_tokens={cfg.max_new_tokens}, "
          f"temperature={cfg.gen_temperature}) …")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=1.0 if cfg.gen_temperature == 0.0 else cfg.gen_temperature,
            do_sample=(cfg.gen_temperature != 0.0),
            verbose=False,
        )

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

def compute_ecs(
    model: HookedTransformer,
    cache,
    transcript_len: int,
    note_len: int,
) -> np.ndarray:
    """
    External Context Score — paper eq. (3/4).

    For each note token n at every (layer l, head h):

        e^{l,h}_n = Σ_j  attn[l,h,n,j] · x^L_j        (j ∈ transcript positions)
        ECS^{l,h}_n = cosine_similarity( e^{l,h}_n ,  x^L_n )

    x^L_j is the LAST-LAYER residual stream hidden state of token j, used as
    its semantic representation (Luo et al. 2024; Chen et al. 2024a).
    The attention pattern at (l, h) provides soft weights over transcript tokens;
    this is the continuous relaxation of the set-mean in paper eq. (3).

    Token-level ECS is the mean over all (layer, head) pairs.
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads

    # Last-layer residual stream: (S, D)
    x_last       = cache["resid_post", n_layers - 1][0].float().cpu().numpy()
    x_transcript = x_last[:transcript_len]   # (T, D)
    x_note       = x_last[transcript_len:]   # (N, D)

    norm_x = np.linalg.norm(x_note, axis=-1)   # (N,) — pre-computed once

    ecs = np.zeros(note_len, dtype=np.float64)

    for layer in range(n_layers):
        attn_pat = cache["pattern", layer][0].float().cpu().numpy()   # (H, S, S)

        # Note-query → transcript-key attention weights: (H, N, T)
        w = attn_pat[:, transcript_len:, :transcript_len]

        # Attention-weighted context vectors: (H, N, D)
        e = np.einsum("hnt,td->hnd", w, x_transcript)

        # Cosine similarity: dot(e[h,n], x_note[n]) / (||e|| · ||x_note||)
        dot    = np.einsum("hnd,nd->hn", e, x_note)   # (H, N)
        norm_e = np.linalg.norm(e, axis=-1)            # (H, N)
        denom  = norm_e * norm_x[None, :]              # (H, N)

        valid = denom > 1e-8
        cos   = np.where(valid, dot / np.where(valid, denom, 1.0), 0.0)

        ecs += cos.sum(axis=0)   # sum over heads

    ecs /= (n_layers * n_heads)
    return np.clip(ecs, 0.0, 1.0)


def _logit_lens(model: HookedTransformer, x: torch.Tensor) -> torch.Tensor:
    """
    Apply the LogitLens projection to an intermediate residual-stream tensor.

    LogitLens(x) = Unembed( LayerNorm_final(x) )

    Parameters
    ----------
    x : (N, d_model) tensor on the model's device.

    Returns
    -------
    logits : (N, d_vocab) float32 vocabulary scores.
    """
    x = x.unsqueeze(0)           # (1, N, d_model)
    x = model.ln_final(x)        # (1, N, d_model)  — dtype follows model weights
    x = model.unembed(x)         # (1, N, d_vocab)
    return x.squeeze(0).float()  # (N, d_vocab) — upcast; numpy has no bfloat16


def compute_pks(
    model: HookedTransformer,
    cache,
    transcript_len: int,
    note_len: int,
    device: str,
) -> np.ndarray:
    """
    Parametric Knowledge Score — paper eq. (5).

    For each note token n at every layer l:

        q(x) = softmax( LogitLens(x) )
        P^l_n = JSD( q(x^{mid,l}_n)  ‖  q(x^l_n) )

    x^{mid,l}_n  = residual stream BEFORE the FFN (after attention sub-layer)
    x^l_n        = residual stream AFTER  the FFN

    A large JSD means the FFN shifted the next-token distribution substantially,
    indicating parametric knowledge stored in the MLP weights drove the choice.

    Token-level PKS is the mean over all layers.
    """
    n_layers = model.cfg.n_layers
    pks = np.zeros(note_len, dtype=np.float64)

    for layer in range(n_layers):
        x_mid  = cache["resid_mid",  layer][0, transcript_len:].to(
            device=device, dtype=torch.float32)
        x_post = cache["resid_post", layer][0, transcript_len:].to(
            device=device, dtype=torch.float32)

        with torch.no_grad():
            q_mid  = torch.softmax(_logit_lens(model, x_mid),  dim=-1).cpu().numpy()
            q_post = torch.softmax(_logit_lens(model, x_post), dim=-1).cpu().numpy()

        # JSD(P‖Q) = ½ KL(P‖M) + ½ KL(Q‖M),  M = ½(P+Q)
        m   = 0.5 * (q_mid + q_post)
        eps = 1e-10
        kl1 = np.sum(q_mid  * (np.log(q_mid  + eps) - np.log(m + eps)), axis=-1)
        kl2 = np.sum(q_post * (np.log(q_post + eps) - np.log(m + eps)), axis=-1)
        pks += 0.5 * kl1 + 0.5 * kl2

    pks /= n_layers
    return np.clip(pks, 0.0, 1.0)


def compute_ecs_pks(
    model: HookedTransformer,
    tokens: torch.Tensor,
    transcript_len: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single cached forward pass → (ECS, PKS) arrays for every note token.

    ECS[i]  — cosine similarity between the attention-weighted mean-pool of
               last-layer transcript hidden states and the note token's own
               last-layer hidden state, averaged over all (layer, head) pairs.
               High → token is semantically grounded in the transcript.

    PKS[i]  — mean-layer JSD between LogitLens vocab distributions before and
               after each FFN.
               High → the FFN made a large parametric update to the prediction.

    Quadrant interpretation
    ───────────────────────
        High ECS, Low  PKS  → Extractive        (copied from transcript)
        Low  ECS, High PKS  → Parametric        (driven by stored knowledge)
        High ECS, High PKS  → Synthesized       (grounded reasoning)
        Low  ECS, Low  PKS  → Hallucination risk (neither source explains token)
    """
    note_len = tokens.shape[1] - transcript_len
    assert note_len > 0, "note_len must be > 0: check tokenisation"

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: (
                "pattern"    in name or   # attention weights              → ECS
                "resid_mid"  in name or   # residual before FFN            → PKS
                "resid_post" in name      # residual after FFN; last layer → ECS
            ),
        )

    ecs = compute_ecs(model, cache, transcript_len, note_len)
    pks = compute_pks(model, cache, transcript_len, note_len, cfg.device)
    return ecs, pks


# ─────────────────────────────────────────────
# 6. Derived metrics
# ─────────────────────────────────────────────

def hallucination_risk(ecs: np.ndarray, pks: np.ndarray) -> np.ndarray:
    """
    Scalar risk score per token.  Highest when both ECS and PKS are low.
    risk = 1 − (ECS + PKS) / 2
    """
    return np.clip(1.0 - (ecs + pks) / 2.0, 0.0, 1.0)


def quadrant_stats(ecs: np.ndarray, pks: np.ndarray, label: str) -> Dict:
    """
    Fraction of tokens in each ECS/PKS quadrant.
    Quadrant boundaries are the per-array medians (always a 50/50 split on each
    axis, making cross-model / cross-note comparisons fair).
    """
    em = np.median(ecs)
    pm = np.median(pks)
    n  = len(ecs)

    hi_ecs = ecs >= em
    hi_pks = pks >= pm

    stats = {
        "label":              label,
        "n_tokens":           n,
        "extractive_frac":    float(np.mean( hi_ecs & ~hi_pks)),
        "parametric_frac":    float(np.mean(~hi_ecs &  hi_pks)),
        "synthesized_frac":   float(np.mean( hi_ecs &  hi_pks)),
        "hallucinatory_frac": float(np.mean(~hi_ecs & ~hi_pks)),
        "mean_ecs":           float(ecs.mean()),
        "mean_pks":           float(pks.mean()),
        "mean_risk":          float(hallucination_risk(ecs, pks).mean()),
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

# Shared colour palette — used by all plots in both files
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
    ECS vs PKS quadrant scatter.  Each point is one note token, coloured by
    quadrant.  Optional `highlight` list marks specific positions with a star.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    em = np.median(ecs)
    pm = np.median(pks)

    colors = [_token_color(e, p, em, pm) for e, p in zip(ecs, pks)]
    ax.scatter(ecs, pks, c=colors, s=55, alpha=0.75, edgecolors="white", linewidth=0.4)

    for i, (e, p, tok) in enumerate(zip(ecs, pks, tokens)):
        if i % annotate_stride == 0:
            label = tok.replace("▁", "").replace("Ġ", "").strip()[:10]
            ax.annotate(label, (e, p), fontsize=5.5, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")

    if highlight:
        hx = [ecs[i] for i in highlight if i < len(ecs)]
        hy = [pks[i] for i in highlight if i < len(pks)]
        ax.scatter(hx, hy, s=200, c="black", marker="*", zorder=6,
                   label="Hallucinated span")

    ax.axvline(em, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(pm, color="gray", ls="--", lw=0.8, alpha=0.5)

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
    colors = plt.cm.RdYlGn_r(risk)

    ax.bar(range(len(risk)), risk, color=colors, width=0.85, edgecolor="none")
    ax.set_xticks(range(len(disp)))
    ax.set_xticklabels(disp, rotation=60, ha="right", fontsize=5.5)
    ax.axhline(0.5, color="#B71C1C", ls="--", lw=1.2, alpha=0.7, label="Risk threshold 0.5")
    ax.set_ylabel("Hallucination Risk", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    return ax
