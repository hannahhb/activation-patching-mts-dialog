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
  compute_ecs             ECS per note token — paper eq. (3/4); returns (ecs, ecs_layers)
  compute_pks             PKS per note token — paper eq. (5);   returns (pks, pks_layers)
  compute_ecs_pks         single forward pass → (ecs, pks, ecs_layers, pks_layers)
  compute_dla             single forward pass → (attn_dla, mlp_dla) per layer
  layer_discriminability  AUROC + Cohen's d per layer for ECS/PKS
  dla_discriminability    AUROC + Cohen's d per layer for attn/mlp DLA
  hallucination_risk      scalar risk score: 1 − (ECS + PKS) / 2
  quadrant_stats          fraction of tokens in each quadrant
  Q_COLORS                quadrant colour palette (shared across all plots)
  plot_scatter            ECS vs PKS quadrant scatter
  plot_heatmap            two-row ECS / PKS heatmap across token positions
  plot_risk_bar           per-token hallucination risk bar chart
  plot_layer_discriminability  two-panel AUROC + Cohen's d figure

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

# Defined here so both tokenize_as_generated and generate_note can use it.
_GENERATION_PROMPT = (
    "You are a clinical documentation assistant."
    "Given the following patient-clinician conversation, write a summary of the "
    "conversation with six sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
    "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN."
    "### Conversation\n{transcript}\n\n"
    "### Note: \n"
)


def tokenize_as_generated(
    model: HookedTransformer,
    transcript: str,
    note: str,
) -> Tuple[torch.Tensor, int, List[str]]:
    """
    Tokenize a note in the same prompt context used for actual generation
    (Experiment 2b), so that note-token activations are conditioned identically
    to how the model would condition them during autoregressive generation.

    For a causal (decoder-only) model, teacher-forcing with the full sequence
    [prompt | note] produces exactly the same hidden states at each note position
    as step-by-step generation would — this is a fundamental property of causal
    attention.  The only thing that matters for correctness is that the *prefix*
    matches what was used during generation, which this function ensures.

    Returns
    -------
    tokens      : LongTensor (1, prompt_len + note_len)
    prompt_len  : number of tokens in the generation prompt prefix
                  (drop-in replacement for transcript_len in compute_ecs_pks)
    note_strs   : decoded string for each note token (for plot labels)
    """
    prompt     = _GENERATION_PROMPT.format(transcript=transcript.strip())
    prompt_tok = model.to_tokens(prompt, prepend_bos=True)    # (1, P)
    note_tok   = model.to_tokens(note,   prepend_bos=False)   # (1, N)
    combined   = torch.cat([prompt_tok, note_tok], dim=1)     # (1, P+N)
    prompt_len = prompt_tok.shape[1]
    note_strs  = [model.tokenizer.decode([t.item()]) for t in note_tok[0]]
    return combined, prompt_len, note_strs


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    External Context Score — paper eq. (3/4).

    For each note token n at every (layer l, head h):

        e^{l,h}_n = Σ_j  attn[l,h,n,j] · x^L_j        (j ∈ transcript positions)
        ECS^{l,h}_n = cosine_similarity( e^{l,h}_n ,  x^L_n )

    x^L_j is the LAST-LAYER residual stream hidden state of token j, used as
    its semantic representation (Luo et al. 2024; Chen et al. 2024a).
    The attention pattern at (l, h) provides soft weights over transcript tokens;
    this is the continuous relaxation of the set-mean in paper eq. (3).

    Returns
    -------
    ecs        : (note_len,)          mean ECS over all (layer, head) pairs.
    ecs_layers : (n_layers, note_len) per-layer ECS (averaged over heads at each layer).
    """
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads

    # Last-layer residual stream: (S, D)
    x_last       = cache["resid_post", n_layers - 1][0].float().cpu().numpy()
    x_transcript = x_last[:transcript_len]   # (T, D)
    x_note       = x_last[transcript_len:]   # (N, D)

    norm_x = np.linalg.norm(x_note, axis=-1)   # (N,) — pre-computed once

    ecs        = np.zeros(note_len,           dtype=np.float64)
    ecs_layers = np.zeros((n_layers, note_len), dtype=np.float64)

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
        cos   = np.where(valid, dot / np.where(valid, denom, 1.0), 0.0)  # (H, N)

        ecs_layers[layer] = cos.mean(axis=0)   # mean over heads at this layer
        ecs += cos.sum(axis=0)                 # accumulate for global mean

    ecs /= (n_layers * n_heads)
    return np.clip(ecs, 0.0, 1.0), np.clip(ecs_layers, 0.0, 1.0)


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parametric Knowledge Score — paper eq. (5).

    For each note token n at every layer l:

        q(x) = softmax( LogitLens(x) )
        P^l_n = JSD( q(x^{mid,l}_n)  ‖  q(x^l_n) )

    x^{mid,l}_n  = residual stream BEFORE the FFN (after attention sub-layer)
    x^l_n        = residual stream AFTER  the FFN

    A large JSD means the FFN shifted the next-token distribution substantially,
    indicating parametric knowledge stored in the MLP weights drove the choice.

    Returns
    -------
    pks        : (note_len,)           mean PKS over all layers.
    pks_layers : (n_layers, note_len)  per-layer JSD (not averaged).
    """
    n_layers   = model.cfg.n_layers
    pks        = np.zeros(note_len,            dtype=np.float64)
    pks_layers = np.zeros((n_layers, note_len), dtype=np.float64)

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
        layer_jsd = 0.5 * kl1 + 0.5 * kl2

        pks_layers[layer] = np.clip(layer_jsd, 0.0, 1.0)
        pks += layer_jsd

    pks /= n_layers
    return np.clip(pks, 0.0, 1.0), pks_layers


def compute_ecs_pks(
    model: HookedTransformer,
    tokens: torch.Tensor,
    transcript_len: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single cached forward pass → (ECS, PKS, ECS-per-layer, PKS-per-layer).

    ECS[i]            — cosine similarity averaged over all (layer, head) pairs.
                        High → token is semantically grounded in the transcript.
    PKS[i]            — mean-layer JSD over all FFN layers.
                        High → parametric knowledge drove the prediction.
    ecs_layers[l, i]  — ECS at layer l (averaged over heads), shape (L, N).
    pks_layers[l, i]  — JSD at layer l (not averaged),        shape (L, N).

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

    ecs, ecs_layers = compute_ecs(model, cache, transcript_len, note_len)
    pks, pks_layers = compute_pks(model, cache, transcript_len, note_len, cfg.device)
    return ecs, pks, ecs_layers, pks_layers


def compute_dla(
    model: HookedTransformer,
    tokens: torch.Tensor,
    transcript_len: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct Logit Attribution (DLA) — attention vs MLP contribution per layer.

    Decomposes each note token's position in the residual stream into additive
    contributions from every attention and MLP layer:

        attn_DLA^l_n = attn_out^l_n · W_U[:, t_n]
        mlp_DLA^l_n  = mlp_out^l_n  · W_U[:, t_n]

    where t_n is the token id at note position n and W_U is the unembedding
    matrix.  A large positive value means that component strongly pushes the
    residual stream toward t_n's direction in vocabulary space.

    Approximation note
    ------------------
    The final LayerNorm (ln_final) is non-linear and cannot be split per
    component.  DLA therefore bypasses it and projects directly through W_U.
    This is the standard approximation used in mechanistic interpretability
    (Elhage et al. 2021; Nanda & Lieberum 2022).  For *comparative* analysis
    (hallucinated vs. clean tokens at the same layer) the bias is symmetric
    and the approximation is justified.

    Parameters
    ----------
    tokens         : (1, S) full token sequence [prompt | note].
    transcript_len : number of prompt tokens (boundary between prefix and note).
    cfg            : Config (cfg.device used for cache transfer).

    Returns
    -------
    attn_dla : (n_layers, note_len) float64
               Signed attention DLA per layer per note token.
    mlp_dla  : (n_layers, note_len) float64
               Signed MLP DLA per layer per note token.
    """
    note_len = tokens.shape[1] - transcript_len
    n_layers = model.cfg.n_layers

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "attn_out" in name or "mlp_out" in name,
        )

    # Grab unembedding columns for each note token — avoids materialising the
    # full (note_len, d_vocab) logit matrix which can be hundreds of MB.
    note_token_ids = tokens[0, transcript_len:].cpu()              # (N,)
    W_U_rows = model.W_U[:, note_token_ids].T.float().cpu().detach()  # (N, d_model)

    attn_dla = np.zeros((n_layers, note_len), dtype=np.float64)
    mlp_dla  = np.zeros((n_layers, note_len), dtype=np.float64)

    for layer in range(n_layers):
        attn_out = cache["attn_out", layer][0, transcript_len:].float().cpu()  # (N, D)
        mlp_out  = cache["mlp_out",  layer][0, transcript_len:].float().cpu()  # (N, D)

        # Element-wise dot with the unembedding column → scalar DLA per token
        attn_dla[layer] = (attn_out * W_U_rows).sum(dim=-1).numpy()
        mlp_dla [layer] = (mlp_out  * W_U_rows).sum(dim=-1).numpy()

    return attn_dla, mlp_dla


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


def layer_discriminability(
    pks_layers: np.ndarray,
    ecs_layers: np.ndarray,
    halluc_mask: np.ndarray,
) -> Optional[Dict]:
    """
    Per-layer AUROC and Cohen's d for both PKS and ECS, separating
    hallucinated from non-hallucinated tokens.

    Parameters
    ----------
    pks_layers  : (n_layers, note_len) per-layer PKS values.
    ecs_layers  : (n_layers, note_len) per-layer ECS values.
    halluc_mask : (note_len,) bool — True for injected/hallucinated tokens.

    Returns
    -------
    Dict with keys pks_auroc, pks_cohens_d, ecs_auroc, ecs_cohens_d
    (each a 1-D array of length n_layers), or None if both classes are not
    present in halluc_mask.

    Interpretation
    --------------
    AUROC > 0.5  → higher scores → hallucinated (or lower if < 0.5).
    Cohen's d    → standardised mean difference (hallucinated − clean).
    For PKS: expect d < 0 (hallucinated tokens have *lower* FFN shift because
    they are not supported by parametric knowledge).
    For ECS: expect d < 0 (hallucinated tokens attend less to the transcript).
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for layer_discriminability: "
            "pip install scikit-learn"
        ) from exc

    h_mask = halluc_mask.astype(bool)
    c_mask = ~h_mask

    if h_mask.sum() < 1 or c_mask.sum() < 1:
        warnings.warn(
            "layer_discriminability: need both hallucinated and clean tokens; "
            "returning None."
        )
        return None

    n_layers = pks_layers.shape[0]
    result: Dict = {
        "pks_auroc":    np.zeros(n_layers),
        "pks_cohens_d": np.zeros(n_layers),
        "ecs_auroc":    np.zeros(n_layers),
        "ecs_cohens_d": np.zeros(n_layers),
    }

    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Cohen's d = (mean_a − mean_b) / pooled_std."""
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return float("nan")
        pooled = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
        return float((a.mean() - b.mean()) / (pooled + 1e-10))

    y = h_mask.astype(int)
    for layer in range(n_layers):
        for scores, auroc_key, d_key in [
            (pks_layers[layer], "pks_auroc", "pks_cohens_d"),
            (ecs_layers[layer], "ecs_auroc", "ecs_cohens_d"),
        ]:
            try:
                result[auroc_key][layer] = roc_auc_score(y, scores)
            except ValueError:
                result[auroc_key][layer] = 0.5

            result[d_key][layer] = _cohens_d(scores[h_mask], scores[c_mask])

    return result


def dla_discriminability(
    attn_dla: np.ndarray,
    mlp_dla: np.ndarray,
    halluc_mask: np.ndarray,
) -> Optional[Dict]:
    """
    Per-layer AUROC and Cohen's d for attention DLA and MLP DLA, separating
    hallucinated from non-hallucinated tokens.

    Parameters
    ----------
    attn_dla    : (n_layers, note_len) per-layer attention DLA values.
    mlp_dla     : (n_layers, note_len) per-layer MLP DLA values.
    halluc_mask : (note_len,) bool — True for injected/hallucinated tokens.

    Returns
    -------
    Dict with keys attn_auroc, attn_cohens_d, mlp_auroc, mlp_cohens_d
    (each a 1-D array of length n_layers), or None if both classes absent.

    Interpretation
    --------------
    For hallucinated tokens, expect:
      attn_cohens_d < 0  → attention contributes less (not copying from prompt)
      mlp_cohens_d  > 0  → MLP contributes more (parametric knowledge firing)
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for dla_discriminability: "
            "pip install scikit-learn"
        ) from exc

    h_mask = halluc_mask.astype(bool)
    c_mask = ~h_mask

    if h_mask.sum() < 1 or c_mask.sum() < 1:
        warnings.warn(
            "dla_discriminability: need both hallucinated and clean tokens; "
            "returning None."
        )
        return None

    n_layers = attn_dla.shape[0]
    result: Dict = {
        "attn_auroc":    np.zeros(n_layers),
        "attn_cohens_d": np.zeros(n_layers),
        "mlp_auroc":     np.zeros(n_layers),
        "mlp_cohens_d":  np.zeros(n_layers),
    }

    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return float("nan")
        pooled = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
        return float((a.mean() - b.mean()) / (pooled + 1e-10))

    y = h_mask.astype(int)
    for layer in range(n_layers):
        for scores, auroc_key, d_key in [
            (attn_dla[layer], "attn_auroc", "attn_cohens_d"),
            (mlp_dla [layer], "mlp_auroc",  "mlp_cohens_d"),
        ]:
            try:
                result[auroc_key][layer] = roc_auc_score(y, scores)
            except ValueError:
                result[auroc_key][layer] = 0.5
            result[d_key][layer] = _cohens_d(scores[h_mask], scores[c_mask])

    return result


def plot_layer_discriminability(
    disc: Dict,
    title: str,
    axes: Optional[Tuple] = None,
    metric_a: str = "ecs",
    metric_b: str = "pks",
    label_a: str = "ECS",
    label_b: str = "PKS",
    color_a: str = "#2196F3",
    color_b: str = "#FF9800",
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Two-panel figure showing layer-wise discriminability for two metrics.

    Top panel   : AUROC per layer (chance = 0.5).
    Bottom panel: Cohen's d per layer (hallucinated − clean; negative = lower
                  for hallucinated tokens).

    Parameters
    ----------
    disc     : dict from layer_discriminability() or dla_discriminability().
    title    : suptitle string.
    axes     : optional (ax_auroc, ax_d) tuple; created if None.
    metric_a : key prefix for first metric  (default "ecs" → "ecs_auroc" etc.)
    metric_b : key prefix for second metric (default "pks" → "pks_auroc" etc.)
    label_a  : display label for first metric  (default "ECS").
    label_b  : display label for second metric (default "PKS").
    color_a  : line/bar colour for metric_a.
    color_b  : line/bar colour for metric_b.
    """
    n_layers = len(disc[f"{metric_a}_auroc"])
    layers   = np.arange(n_layers)
    width    = 0.35

    if axes is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n_layers // 2), 8),
                                        sharex=True)
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        ax1, ax2 = axes

    auroc_a = disc[f"{metric_a}_auroc"]
    auroc_b = disc[f"{metric_b}_auroc"]
    d_a     = disc[f"{metric_a}_cohens_d"]
    d_b     = disc[f"{metric_b}_cohens_d"]

    # ── AUROC panel ──────────────────────────────────────────────────────────
    ax1.plot(layers, auroc_a, marker="s", color=color_a,
             label=f"{label_a} AUROC", lw=1.8, markersize=4)
    ax1.plot(layers, auroc_b, marker="o", color=color_b,
             label=f"{label_b} AUROC", lw=1.8, markersize=4)
    ax1.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.7, label="Chance (0.5)")
    ax1.fill_between(layers, 0.5, auroc_a,
                     where=auroc_a > 0.5, alpha=0.12, color=color_a)
    ax1.fill_between(layers, 0.5, auroc_b,
                     where=auroc_b > 0.5, alpha=0.12, color=color_b)
    ax1.set_ylabel("AUROC", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_title("Layer-wise AUROC — hallucinated vs. clean tokens", fontsize=10)

    # ── Cohen's d panel ──────────────────────────────────────────────────────
    ax2.bar(layers - width / 2, d_a, width, color=color_a, alpha=0.75,
            label=f"{label_a} Cohen's d")
    ax2.bar(layers + width / 2, d_b, width, color=color_b, alpha=0.75,
            label=f"{label_b} Cohen's d")
    ax2.axhline(0, color="gray", ls="-", lw=0.8, alpha=0.6)
    ax2.set_ylabel("Cohen's d  (halluc − clean)", fontsize=10)
    ax2.set_xlabel("Layer", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_title(
        "Layer-wise Cohen's d  (negative = hallucinated tokens score lower)",
        fontsize=10,
    )

    return ax1, ax2


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
