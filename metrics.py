"""
metrics.py
==========
Core ECS/PKS/DLA forward-pass computations and all derived token-level metrics.

Provides
--------
  compute_ecs              ECS per note token — paper eq. (3/4)
  compute_pks              PKS per note token — paper eq. (5)
  compute_ecs_pks          single forward pass → (ecs, pks, ecs_layers, pks_layers)
  compute_dla              single forward pass → (attn_dla, mlp_dla) per layer
  hallucination_risk       scalar risk score: 1 − (ECS + PKS) / 2
  quadrant_stats           fraction of tokens in each quadrant
  layer_discriminability   AUROC + Cohen's d + Pearson r per layer for ECS/PKS
  dla_discriminability     AUROC + Cohen's d per layer for attention/MLP DLA
  identify_knowledge_ffns  top-k layers (set F) by most positive PKS Pearson r
  identify_copy_head_layers top-k layers (set A) by most negative ECS Pearson r
  fit_hallucination_regressor fit logistic α/β for Ht(t) = Σα·P^l − Σβ·E^{l,h}
  calibrate_layer_thresholds Youden-J optimal thresholds from labeled data
  apply_layer_thresholds   AUROC-weighted per-layer vote → per-token halluc prob
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

from config import Config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# ECS / PKS computation
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

    ecs        = np.zeros(note_len,             dtype=np.float64)
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
    x = model.ln_final(x)        # (1, N, d_model)
    x = model.unembed(x)         # (1, N, d_vocab)
    return x.squeeze(0).float()  # (N, d_vocab)


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
    pks        = np.zeros(note_len,             dtype=np.float64)
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
    return ecs, pks, ecs_layers, pks_layers, cache


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

    Returns
    -------
    attn_dla : (n_layers, note_len) float64 — signed attention DLA per layer.
    mlp_dla  : (n_layers, note_len) float64 — signed MLP DLA per layer.
    """
    note_len = tokens.shape[1] - transcript_len
    n_layers = model.cfg.n_layers

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "attn_out" in name or "mlp_out" in name,
        )

    note_token_ids = tokens[0, transcript_len:].cpu()
    W_U_rows = model.W_U[:, note_token_ids].T.float().cpu().detach()  # (N, d_model)

    attn_dla = np.zeros((n_layers, note_len), dtype=np.float64)
    mlp_dla  = np.zeros((n_layers, note_len), dtype=np.float64)

    for layer in range(n_layers):
        attn_out = cache["attn_out", layer][0, transcript_len:].float().cpu()
        mlp_out  = cache["mlp_out",  layer][0, transcript_len:].float().cpu()
        attn_dla[layer] = (attn_out * W_U_rows).sum(dim=-1).numpy()
        mlp_dla [layer] = (mlp_out  * W_U_rows).sum(dim=-1).numpy()

    return attn_dla, mlp_dla


# ─────────────────────────────────────────────
# Derived scalar metrics
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
# Layer-wise discriminability
# ─────────────────────────────────────────────

def layer_discriminability(
    pks_layers: np.ndarray,
    ecs_layers: np.ndarray,
    halluc_mask: np.ndarray,
) -> Optional[Dict]:
    """
    Per-layer AUROC, Cohen's d, and Pearson r for both PKS and ECS, separating
    hallucinated from non-hallucinated tokens.

    Parameters
    ----------
    pks_layers  : (n_layers, note_len) per-layer PKS values.
    ecs_layers  : (n_layers, note_len) per-layer ECS values.
    halluc_mask : (note_len,) bool — True for injected/hallucinated tokens.

    Returns
    -------
    Dict with keys pks_auroc, pks_cohens_d, pks_pearson_r,
                       ecs_auroc, ecs_cohens_d, ecs_pearson_r
    (each a 1-D array of length n_layers), or None if both classes are absent.

    Interpretation
    --------------
    AUROC > 0.5   → higher score → hallucinated (or lower if < 0.5).
    Cohen's d     → standardised mean difference (hallucinated − clean).
    Pearson r     → point-biserial correlation with hallucination label (0/1).
      pks_pearson_r > 0 : higher PKS → more hallucinated → candidate Knowledge FFN (set F)
      ecs_pearson_r < 0 : higher ECS → less hallucinated → candidate Copying Head (set A)
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for layer_discriminability: pip install scikit-learn"
        ) from exc
    try:
        from scipy.stats import pearsonr as _pearsonr
    except ImportError as exc:
        raise ImportError(
            "scipy is required for layer_discriminability: pip install scipy"
        ) from exc

    h_mask = halluc_mask.astype(bool)
    c_mask = ~h_mask

    if h_mask.sum() < 1 or c_mask.sum() < 1:
        warnings.warn(
            "layer_discriminability: need both hallucinated and clean tokens; returning None."
        )
        return None

    n_layers = pks_layers.shape[0]
    result: Dict = {
        "pks_auroc":     np.zeros(n_layers),
        "pks_cohens_d":  np.zeros(n_layers),
        "pks_pearson_r": np.zeros(n_layers),
        "ecs_auroc":     np.zeros(n_layers),
        "ecs_cohens_d":  np.zeros(n_layers),
        "ecs_pearson_r": np.zeros(n_layers),
    }

    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return float("nan")
        pooled = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
        return float((a.mean() - b.mean()) / (pooled + 1e-10))

    y = h_mask.astype(int)
    for layer in range(n_layers):
        for scores, auroc_key, d_key, r_key in [
            (pks_layers[layer], "pks_auroc", "pks_cohens_d", "pks_pearson_r"),
            (ecs_layers[layer], "ecs_auroc", "ecs_cohens_d", "ecs_pearson_r"),
        ]:
            try:
                result[auroc_key][layer] = roc_auc_score(y, scores)
            except ValueError:
                result[auroc_key][layer] = 0.5

            result[d_key][layer] = _cohens_d(scores[h_mask], scores[c_mask])

            try:
                r_val, _ = _pearsonr(scores.astype(float), y.astype(float))
                result[r_key][layer] = float(r_val) if np.isfinite(r_val) else 0.0
            except Exception:
                result[r_key][layer] = 0.0

    return result


def dla_discriminability(
    attn_dla: np.ndarray,
    mlp_dla: np.ndarray,
    halluc_mask: np.ndarray,
) -> Optional[Dict]:
    """
    Per-layer AUROC and Cohen's d for attention DLA and MLP DLA.

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
            "scikit-learn is required for dla_discriminability: pip install scikit-learn"
        ) from exc

    h_mask = halluc_mask.astype(bool)
    c_mask = ~h_mask

    if h_mask.sum() < 1 or c_mask.sum() < 1:
        warnings.warn(
            "dla_discriminability: need both hallucinated and clean tokens; returning None."
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


# ─────────────────────────────────────────────
# Layer selection (sets F and A)
# ─────────────────────────────────────────────

def identify_knowledge_ffns(pks_pearson_r: np.ndarray, top_k: int = 5) -> List[int]:
    """
    Select the top-k layers (set F — Knowledge FFNs) with the most *positive*
    Pearson r between per-layer PKS and the hallucination label.

    High PKS Pearson r → model over-relies on parametric memory at this layer.
    Returns a sorted list of layer indices (ascending).
    """
    k = max(1, min(top_k, len(pks_pearson_r)))
    ranked = np.argsort(pks_pearson_r)[::-1]   # descending
    return sorted(int(l) for l in ranked[:k])


def identify_copy_head_layers(ecs_pearson_r: np.ndarray, top_k: int = 5) -> List[int]:
    """
    Select the top-k layers (set A — Copying Head layers) with the most
    *negative* Pearson r between per-layer ECS and the hallucination label.

    Most negative ECS Pearson r → copying heads at this layer fail to attend
    to the transcript when tokens are hallucinated.
    Returns a sorted list of layer indices (ascending).
    """
    k = max(1, min(top_k, len(ecs_pearson_r)))
    ranked = np.argsort(ecs_pearson_r)          # ascending (most negative first)
    return sorted(int(l) for l in ranked[:k])


# ─────────────────────────────────────────────
# REDEEP logistic regressor
# ─────────────────────────────────────────────

def fit_hallucination_regressor(
    pks_layers_all: np.ndarray,
    ecs_layers_all: np.ndarray,
    halluc_mask_all: np.ndarray,
    F: List[int],
    A: List[int],
) -> Tuple:
    """
    Fit the REDEEP hallucination regressor:

        Ht(t) = Σ_{l∈F} α · P^l_t  −  Σ_{l∈A} β · E^{l,h}_t

    α and β are learned via logistic regression on the training set, with
    features = [PKS at F layers | ECS at A layers].

    Returns
    -------
    (clf, scaler, alpha, beta)
      clf    : fitted LogisticRegression
      scaler : fitted StandardScaler (apply before clf.predict_proba)
      alpha  : mean positive coefficient for PKS features (set F)
      beta   : mean positive contribution of ECS features (set A, sign-flipped)
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for fit_hallucination_regressor: pip install scikit-learn"
        ) from exc

    pks_feats = pks_layers_all[F].T
    ecs_feats = ecs_layers_all[A].T
    X = np.concatenate([pks_feats, ecs_feats], axis=1).astype(np.float64)
    y = halluc_mask_all.astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )
    clf.fit(X_scaled, y)

    coefs     = clf.coef_[0]
    pks_coefs = coefs[:len(F)]
    ecs_coefs = coefs[len(F):]
    alpha = float(np.mean(np.maximum(pks_coefs,  0.0)))
    beta  = float(np.mean(np.maximum(-ecs_coefs, 0.0)))

    return clf, scaler, alpha, beta


# ─────────────────────────────────────────────
# Youden-J calibration helpers
# ─────────────────────────────────────────────

def calibrate_layer_thresholds(
    scores_layers: np.ndarray,
    halluc_mask: np.ndarray,
    min_j: float = 0.05,
) -> Optional[Dict[int, Dict]]:
    """
    Derive a per-layer operating threshold using Youden's J statistic (J = TPR − FPR).

    Returns
    -------
    Dict mapping layer_index → {threshold, direction, auroc, j_stat}, or None.
    direction: -1 → flag when score < threshold; +1 → flag when score > threshold.
    """
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for calibrate_layer_thresholds: pip install scikit-learn"
        ) from exc

    h = halluc_mask.astype(bool)
    if h.sum() < 1 or (~h).sum() < 1:
        warnings.warn("calibrate_layer_thresholds: both classes required; returning None.")
        return None

    y      = h.astype(int)
    result: Dict[int, Dict] = {}

    for layer in range(scores_layers.shape[0]):
        scores    = scores_layers[layer]
        auroc     = float(roc_auc_score(y, scores))
        direction = 1 if auroc >= 0.5 else -1

        fpr, tpr, thresh = roc_curve(y, direction * scores)
        j     = tpr - fpr
        best  = int(np.argmax(j))
        j_val = float(j[best])

        if j_val < min_j:
            continue

        result[layer] = {
            "threshold": float(thresh[best]) * direction,
            "direction": direction,
            "auroc":     float(max(auroc, 1.0 - auroc)),
            "j_stat":    j_val,
        }

    return result or None


def apply_layer_thresholds(
    scores_layers: np.ndarray,
    thresholds: Dict[int, Dict],
    top_k: int = 5,
) -> np.ndarray:
    """
    Apply calibrated layer thresholds → per-token hallucination probability in [0, 1].

    Selects the top-K layers ranked by Youden's J, then combines via an
    AUROC-weighted soft vote.
    """
    if not thresholds:
        return np.zeros(scores_layers.shape[1])

    selected   = sorted(thresholds, key=lambda l: thresholds[l]["j_stat"], reverse=True)[:top_k]
    note_len   = scores_layers.shape[1]
    vote_sum   = np.zeros(note_len, dtype=np.float64)
    weight_sum = 0.0

    for layer in selected:
        t    = thresholds[layer]
        w    = t["auroc"] - 0.5
        s    = scores_layers[layer]
        flag = (s < t["threshold"]) if t["direction"] == -1 else (s > t["threshold"])
        vote_sum   += w * flag.astype(np.float64)
        weight_sum += w

    return np.clip(vote_sum / (weight_sum + 1e-8), 0.0, 1.0)


# ─────────────────────────────────────────────
# Activation patching / causal tracing
# ─────────────────────────────────────────────

def compute_causal_patch_scores(
    model: HookedTransformer,
    tokens_clean:     torch.Tensor,   # (1, seq_len_clean)   — gold note
    tokens_corrupted: torch.Tensor,   # (1, seq_len_corrupt) — injected note
    halluc_seq_positions: List[int],  # sequence-space positions of hallucinated tokens
    components: List[str] = ["resid_pre", "attn_out", "mlp_out"],
    device: str = "cpu",
) -> Optional[Dict]:
    """
    Activation patching (causal tracing) for hallucination localisation.

    For every (layer, component) pair, replaces the corrupted run's activations
    at the hallucinated token positions with the corresponding activations from
    the clean (gold-note) run.  The restoration score measures how much the
    correct token's probability is recovered:

        restoration[l, c] = mean_t( (P_patch - P_corrupt) / (P_clean - P_corrupt) )

    Interpretation
    ──────────────
    • restoration ≈ 1  → patching this component fully restores correct prediction
                         → component is *causally responsible* for the hallucination
    • restoration ≈ 0  → patching has no effect → component is not on the causal path
    • restoration < 0  → patching made things worse (rare; indicates indirect effects)

    Components
    ──────────
    resid_pre  — residual stream flowing INTO layer l  (captures cumulative state)
    attn_out   — attention sublayer contribution at layer l
    mlp_out    — MLP sublayer contribution at layer l

    Cost: 1 clean pass + 1 corrupted pass + n_layers × n_components patching passes.
    For a 26-layer model with 3 components: 80 forward passes total.

    Parameters
    ----------
    tokens_clean     : tokenised gold note sequence (prompt + gold note).
    tokens_corrupted : tokenised corrupted note sequence (prompt + halluc note).
    halluc_seq_positions : absolute sequence positions (in corrupted) of hallucinated tokens.
    components       : which activation types to patch.
    device           : torch device string.

    Returns
    -------
    Dict with keys:
      restoration      : (n_layers, n_components) mean restoration scores
      raw_scores       : (n_layers, n_components, n_valid_positions)
      components       : list of component names (column labels)
      valid_positions  : filtered list of positions actually analysed
      target_tokens    : correct token id at each valid position
      baseline_clean   : mean P(correct) under clean run
      baseline_corrupt : mean P(correct) under corrupted run
    Returns None if no valid positions exist.
    """
    import torch.nn.functional as F
    from transformer_lens import utils as tl_utils

    _COMP_HOOK = {
        "resid_pre": "hook_resid_pre",
        "attn_out":  "hook_attn_out",
        "mlp_out":   "hook_mlp_out",
    }

    n_layers  = model.cfg.n_layers
    seq_clean = tokens_clean.shape[1]
    seq_corr  = tokens_corrupted.shape[1]

    # Only keep positions that exist in both sequences and have a predecessor
    # (need pred_pos = t-1 >= 0 to read the logit that predicts token t)
    valid_positions = [
        t for t in halluc_seq_positions
        if 0 < t < seq_corr and t < seq_clean
    ]
    if not valid_positions:
        return None

    # ── 1. Clean baseline forward pass (cache all patch-able activations) ────
    print(f"  [patch] Clean forward pass …")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            tokens_clean.to(device),
            names_filter=lambda name: any(
                hook in name for hook in _COMP_HOOK.values()
            ),
        )
    clean_logits = clean_logits.float().cpu()

    # ── 2. Corrupted baseline forward pass ───────────────────────────────────
    print(f"  [patch] Corrupted forward pass …")
    with torch.no_grad():
        corrupt_logits = model(tokens_corrupted.to(device)).float().cpu()

    # ── 3. Determine target token and baseline probabilities per position ─────
    target_tokens:  List[int]   = []
    p_clean_vals:   List[float] = []
    p_corrupt_vals: List[float] = []

    for t_seq in valid_positions:
        pred = t_seq - 1   # logit at pred predicts token at t_seq

        # Target = gold-note token at the same sequence position
        target_tok = int(tokens_clean[0, t_seq].item())
        target_tokens.append(target_tok)

        p_clean_vals.append(
            float(F.softmax(clean_logits[0, pred, :], dim=-1)[target_tok])
        )
        p_corrupt_vals.append(
            float(F.softmax(corrupt_logits[0, pred, :], dim=-1)[target_tok])
        )

    # ── 4. Patching passes: one per (layer, component) ───────────────────────
    restoration = np.zeros((n_layers, len(components)), dtype=np.float64)
    raw_scores  = np.zeros((n_layers, len(components), len(valid_positions)), dtype=np.float64)

    total_passes = n_layers * len(components)
    done = 0

    for ci, component in enumerate(components):
        hook_suffix = _COMP_HOOK[component]

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.{hook_suffix}"
            clean_act = clean_cache[component, layer]   # (1, seq_clean, d_model)

            def _make_hook(act, positions, max_pos):
                def _hook(value, hook):
                    for pos in positions:
                        if pos < max_pos and pos < value.shape[1]:
                            value[0, pos, :] = act[0, pos, :].to(value.device)
                    return value
                return _hook

            patch_hook = _make_hook(clean_act, valid_positions, seq_clean)

            with torch.no_grad():
                patch_logits = model.run_with_hooks(
                    tokens_corrupted.to(device),
                    fwd_hooks=[(hook_name, patch_hook)],
                    return_type="logits",
                ).float().cpu()

            for pi, (t_seq, tok, p_cl, p_co) in enumerate(
                zip(valid_positions, target_tokens, p_clean_vals, p_corrupt_vals)
            ):
                pred = t_seq - 1
                p_patch = float(F.softmax(patch_logits[0, pred, :], dim=-1)[tok])
                raw_scores[layer, ci, pi] = (p_patch - p_co) / (abs(p_cl - p_co) + 1e-8)

            restoration[layer, ci] = float(raw_scores[layer, ci, :].mean())

            done += 1
            if done % 10 == 0 or done == total_passes:
                print(f"  [patch] {done}/{total_passes} passes complete …", end="\r")

    print()  # newline after progress

    return {
        "restoration":       restoration,
        "raw_scores":        raw_scores,
        "components":        components,
        "valid_positions":   valid_positions,
        "target_tokens":     target_tokens,
        "baseline_clean":    float(np.mean(p_clean_vals)),
        "baseline_corrupt":  float(np.mean(p_corrupt_vals)),
        "n_layers":          n_layers,
    }


# ─────────────────────────────────────────────
# Note source profile (abstractiveness / extractiveness)
# ─────────────────────────────────────────────

def note_source_profile(ecs: np.ndarray, pks: np.ndarray) -> Dict:
    """
    Aggregate token-level ECS/PKS into a note-level source profile.

    Abstractiveness vs extractiveness is read directly from the quadrant
    distribution — no external reference note required.

    Returns
    -------
    dict with keys:
      extractiveness_score  — mean ECS across all note tokens (0–1)
      abstractiveness_score — mean PKS across all note tokens (0–1)
      net_abstractiveness   — mean(PKS − ECS): positive = more abstract,
                              negative = more extractive
      pct_extractive        — % tokens: high ECS, low PKS  (copied from transcript)
      pct_parametric        — % tokens: low ECS,  high PKS (parametric / abstract)
      pct_synthesized       — % tokens: high ECS, high PKS (grounded reasoning)
      pct_hallucinatory     — % tokens: low ECS,  low PKS  (neither source)
      dominant_mode         — string label for the largest quadrant
    """
    em = float(np.median(ecs))
    pm = float(np.median(pks))

    hi_ecs = ecs >= em
    hi_pks = pks >= pm

    extractive    = ( hi_ecs & ~hi_pks)
    parametric    = (~hi_ecs &  hi_pks)
    synthesized   = ( hi_ecs &  hi_pks)
    hallucinatory = (~hi_ecs & ~hi_pks)

    fracs = {
        "extractive":    float(np.mean(extractive)),
        "parametric":    float(np.mean(parametric)),
        "synthesized":   float(np.mean(synthesized)),
        "hallucinatory": float(np.mean(hallucinatory)),
    }
    dominant_mode = max(fracs, key=lambda k: fracs[k])

    return {
        "extractiveness_score":  float(np.mean(ecs)),
        "abstractiveness_score": float(np.mean(pks)),
        "net_abstractiveness":   float(np.mean(pks - ecs)),
        "pct_extractive":        fracs["extractive"]    * 100,
        "pct_parametric":        fracs["parametric"]    * 100,
        "pct_synthesized":       fracs["synthesized"]   * 100,
        "pct_hallucinatory":     fracs["hallucinatory"] * 100,
        "dominant_mode":         dominant_mode,
        "ecs_median_threshold":  em,
        "pks_median_threshold":  pm,
    }


# ─────────────────────────────────────────────
# Transcript coverage (omission detection)
# ─────────────────────────────────────────────

def compute_transcript_coverage(
    model: HookedTransformer,
    cache,
    transcript_len: int,
    note_len: int,
    copy_head_layers: Optional[List[int]] = None,
) -> np.ndarray:
    """
    For each transcript token, compute how much attention it received from
    note tokens during generation — a proxy for whether that content was
    used when writing the note.

    Low coverage at a transcript span → the model largely ignored that
    content → candidate omission.

    Implementation
    ──────────────
    For each layer l (restricted to copy_head_layers if provided):
        attn[l] : (n_heads, seq_len, seq_len)
        note_to_transcript = attn[l, :, transcript_len:, :transcript_len]
                           — shape (H, note_len, transcript_len)
        contribution[l, t] = mean over heads of sum over note positions
                           = mean_h( Σ_n attn[l, h, n, t] )

    Final coverage[t] = mean over layers, normalised to [0, 1].

    Parameters
    ----------
    copy_head_layers : list of layer indices to average over.
                       None → use all layers.

    Returns
    -------
    coverage : (transcript_len,) float in [0, 1]
               1.0 = maximally attended transcript token
               0.0 = never attended = omission candidate
    """
    n_layers = model.cfg.n_layers
    layers   = copy_head_layers if copy_head_layers is not None else list(range(n_layers))

    coverage = np.zeros(transcript_len, dtype=np.float64)

    for layer in layers:
        attn = cache["pattern", layer][0].float().cpu().numpy()  # (H, S, S)
        # note positions → transcript positions
        note_to_tr = attn[:, transcript_len:transcript_len + note_len, :transcript_len]
        # (H, note_len, transcript_len) → sum over note positions, mean over heads
        coverage += note_to_tr.sum(axis=1).mean(axis=0)          # (transcript_len,)

    coverage /= len(layers)

    # Normalise to [0, 1]
    max_val = coverage.max()
    if max_val > 0:
        coverage = coverage / max_val

    return coverage


def omission_report(
    model: HookedTransformer,
    cache,
    transcript_tokens: torch.Tensor,
    transcript_len: int,
    note_len: int,
    copy_head_layers: Optional[List[int]] = None,
    low_coverage_threshold: float = 0.2,
    min_span_tokens: int = 3,
) -> Dict:
    """
    Identify spans of the transcript that were largely ignored during note
    generation — likely omissions.

    Parameters
    ----------
    low_coverage_threshold : coverage value below which a token is considered
                             under-attended (default 0.2 = bottom 20%).
    min_span_tokens        : minimum consecutive low-coverage tokens to report
                             as a span (filters out single-token noise).

    Returns
    -------
    dict with keys:
      coverage          : (transcript_len,) float array
      omission_mask     : (transcript_len,) bool — True = low coverage
      omission_fraction : fraction of transcript tokens with low coverage
      omission_spans    : list of dicts {start, end, text, mean_coverage}
                          decoded text of each low-coverage span
    """
    coverage = compute_transcript_coverage(
        model, cache, transcript_len, note_len, copy_head_layers
    )

    omission_mask = coverage < low_coverage_threshold

    # Decode transcript tokens for span text extraction
    tr_token_ids = transcript_tokens[0, :transcript_len].tolist()
    tr_strings   = [model.tokenizer.decode([tid]) for tid in tr_token_ids]

    # Find contiguous low-coverage spans
    omission_spans = []
    in_span = False
    span_start = 0

    for i in range(transcript_len):
        if omission_mask[i] and not in_span:
            in_span = True
            span_start = i
        elif not omission_mask[i] and in_span:
            span_len = i - span_start
            if span_len >= min_span_tokens:
                span_text = "".join(tr_strings[span_start:i]).strip()
                omission_spans.append({
                    "start":         span_start,
                    "end":           i,
                    "n_tokens":      span_len,
                    "text":          span_text,
                    "mean_coverage": float(coverage[span_start:i].mean()),
                })
            in_span = False

    # Close any open span at end of sequence
    if in_span and (transcript_len - span_start) >= min_span_tokens:
        span_text = "".join(tr_strings[span_start:transcript_len]).strip()
        omission_spans.append({
            "start":         span_start,
            "end":           transcript_len,
            "n_tokens":      transcript_len - span_start,
            "text":          span_text,
            "mean_coverage": float(coverage[span_start:].mean()),
        })

    # Sort by mean_coverage ascending (worst omissions first)
    omission_spans.sort(key=lambda s: s["mean_coverage"])

    return {
        "coverage":          coverage,
        "omission_mask":     omission_mask,
        "omission_fraction": float(omission_mask.mean()),
        "omission_spans":    omission_spans,
    }


# ─────────────────────────────────────────────
# Semantic Entropy Probe (Experiment 6)
# ─────────────────────────────────────────────

def _get_nli_model(nli_model_name: str):
    """
    Lazily load and cache a sentence-transformers CrossEncoder NLI model.
    Falls back gracefully if sentence-transformers is not installed.
    """
    import functools

    @functools.lru_cache(maxsize=4)
    def _load(name):
        from sentence_transformers import CrossEncoder  # type: ignore
        return CrossEncoder(name)

    return _load(nli_model_name)


def _split_sentences(text: str) -> List[str]:
    """
    Split a clinical note into sentences.
    Handles SOAP section headers as sentence boundaries.
    Strips blank lines and bullet markers.
    """
    import re
    # Treat common SOAP headers as hard boundaries
    text = re.sub(
        r"(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN|S:|O:|A:|P:)",
        r"\n\1",
        text,
        flags=re.IGNORECASE,
    )
    # Split on sentence-ending punctuation or newlines
    parts = re.split(r"(?<=[.!?])\s+|\n{1,}", text)
    return [p.strip() for p in parts if p.strip()]


def _section_aligned_sentences(
    sentences_per_sample: List[List[str]],
) -> List[List[str]]:
    """
    Align sentences across K samples by index within SOAP section.

    Returns a list of "position groups", where each group is a list of K
    sentences (one per sample) that should be compared semantically.
    Pads with empty strings when a sample has fewer sentences at a position.
    """
    K = len(sentences_per_sample)
    max_len = max(len(s) for s in sentences_per_sample)
    groups: List[List[str]] = []
    for i in range(max_len):
        group = []
        for k in range(K):
            sents = sentences_per_sample[k]
            group.append(sents[i] if i < len(sents) else "")
        groups.append(group)
    return groups


def compute_semantic_entropy(
    model: HookedTransformer,
    transcript: str,
    cfg,
    K: int = 5,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    nli_threshold: float = 0.5,
) -> Optional[Dict]:
    """
    Generate K notes from the same transcript, align sentences across samples,
    cluster by bidirectional NLI entailment, and compute Shannon entropy per
    sentence position.

    Each token in a note inherits the SE score of the sentence it belongs to,
    enabling alignment with token-level ECS/PKS scores.

    Parameters
    ----------
    K              : number of generations (default 5).
    nli_model_name : sentence-transformers CrossEncoder model for NLI.
    nli_threshold  : entailment confidence above which two sentences are
                     considered semantically equivalent (default 0.5).

    Returns
    -------
    dict with keys:
      sentences_per_sample : List[List[str]] — K × n_sentences
      se_scores            : (n_sentences,) float — entropy per sentence
      token_se_scores      : (note_len,) float — per-token SE (sentence-level)
      note_tokens          : List[str] — tokenised note (from first sample)
      note_len             : int
      mean_se              : float — mean SE across all sentence positions
    Returns None if generation or NLI fails.
    """
    import re
    import math

    from tokenization import generate_note, tokenize_pair

    # ── 1. Generate K notes ──────────────────────────────────────────────────
    notes: List[str] = []
    for _ in range(K):
        try:
            note = generate_note(model, transcript, cfg)
            notes.append(note)
        except Exception as exc:
            print(f"  [SE] Generation failed: {exc}")
            continue

    if len(notes) < 2:
        print("  [SE] Fewer than 2 notes generated — cannot compute SE.")
        return None

    K_actual = len(notes)

    # ── 2. Sentence split ────────────────────────────────────────────────────
    sentences_per_sample: List[List[str]] = [_split_sentences(n) for n in notes]

    # ── 3. Align sentences across K samples ──────────────────────────────────
    position_groups = _section_aligned_sentences(sentences_per_sample)
    n_positions = len(position_groups)

    # ── 4. NLI entailment → SE per position ──────────────────────────────────
    try:
        nli_model = _get_nli_model(nli_model_name)
    except Exception as exc:
        print(f"  [SE] Could not load NLI model '{nli_model_name}': {exc}")
        return None

    se_scores = np.zeros(n_positions, dtype=np.float64)

    for pos_idx, group in enumerate(position_groups):
        # Filter empty strings (padding)
        valid = [s for s in group if s]
        if len(valid) < 2:
            se_scores[pos_idx] = 0.0
            continue

        # All ordered pairs for bidirectional NLI
        pairs = [(a, b) for i, a in enumerate(valid) for j, b in enumerate(valid) if i != j]

        try:
            # CrossEncoder returns (entailment_logit, neutral_logit, contradiction_logit)
            # or a single float depending on the model.  We want the entailment score.
            raw = nli_model.predict(pairs, apply_softmax=True)
            # raw shape: (n_pairs, 3) or (n_pairs,)
            raw = np.array(raw)
            if raw.ndim == 2:
                entail_scores = raw[:, 0]   # entailment column (label 0 = entailment)
            else:
                entail_scores = raw
        except Exception as exc:
            print(f"  [SE] NLI predict failed at position {pos_idx}: {exc}")
            se_scores[pos_idx] = 0.0
            continue

        # Adjacency matrix: bidirectional entailment means BOTH directions > threshold
        n_v = len(valid)
        adj = np.zeros((n_v, n_v), dtype=bool)
        for pair_idx, (i, j) in enumerate(
            (i, j) for i in range(n_v) for j in range(n_v) if i != j
        ):
            adj[i, j] = entail_scores[pair_idx] >= nli_threshold

        # Bidirectional: A ≡ B iff A→B AND B→A
        bidi = adj & adj.T

        # Connected components (DFS)
        visited = [False] * n_v
        cluster_sizes: List[int] = []

        def dfs(node: int, component: List[int]):
            visited[node] = True
            component.append(node)
            for neighbour in range(n_v):
                if bidi[node, neighbour] and not visited[neighbour]:
                    dfs(neighbour, component)

        for start in range(n_v):
            if not visited[start]:
                comp: List[int] = []
                dfs(start, comp)
                cluster_sizes.append(len(comp))

        # Shannon entropy over cluster proportions
        entropy = 0.0
        for sz in cluster_sizes:
            p = sz / n_v
            if p > 0:
                entropy -= p * math.log(p)
        se_scores[pos_idx] = entropy

    # ── 5. Map sentence SE → token positions ─────────────────────────────────
    # Tokenise the first sample's note to get note_tokens
    try:
        tokens, t_len, note_toks = tokenize_pair(model, transcript, notes[0])
        note_len = len(note_toks)
    except Exception as exc:
        print(f"  [SE] Tokenisation failed: {exc}")
        return None

    # Reconstruct sentence boundaries in token space
    first_note = notes[0]
    first_sents = sentences_per_sample[0]

    token_se_scores = np.zeros(note_len, dtype=np.float64)

    # Map token → sentence by progressive character matching
    char_pos = 0
    sent_idx = 0
    tok_idx  = 0

    for tok_i, tok in enumerate(note_toks):
        # Decode the token to a surface string
        surface = tok.replace("▁", " ").replace("Ġ", " ").lstrip()

        if sent_idx < len(first_sents):
            # Advance sent_idx while the current character position has passed the sentence end
            sent_end = len("".join(first_sents[: sent_idx + 1]))
            char_pos += max(len(surface), 1)
            while sent_idx < len(first_sents) - 1 and char_pos > sent_end:
                sent_idx += 1
                sent_end = len("".join(first_sents[: sent_idx + 1]))

        # The token's SE score = its sentence's SE (clamp to n_positions)
        pos = min(sent_idx, n_positions - 1)
        token_se_scores[tok_i] = se_scores[pos]

    return {
        "sentences_per_sample": sentences_per_sample,
        "se_scores":            se_scores,
        "token_se_scores":      token_se_scores,
        "note_tokens":          note_toks,
        "note_len":             note_len,
        "mean_se":              float(se_scores.mean()),
        "notes":                notes,
        "K_actual":             K_actual,
    }


def train_se_probe(
    resid_post_layers: np.ndarray,
    se_labels: np.ndarray,
    threshold: float = 0.5,
    cv_folds: int = 3,
) -> Optional[Dict]:
    """
    Train one linear probe per layer on `resid_post` to predict tokens with
    high semantic entropy (SE > threshold).

    Parameters
    ----------
    resid_post_layers : (n_layers, n_tokens, d_model) float — residual stream
                        for note-position tokens, stacked across training examples.
    se_labels         : (n_tokens,) float — semantic entropy score per token.
    threshold         : binarise: SE > threshold → high-uncertainty class.
    cv_folds          : number of cross-validation folds for AUROC estimation.

    Returns
    -------
    dict with keys:
      probes       : List[LogisticRegression] — one fitted probe per layer
      scalers      : List[StandardScaler]     — one scaler per layer
      auroc        : (n_layers,) float        — cross-validated AUROC per layer
      best_layer   : int                      — layer with highest AUROC
    Returns None if fewer than 2 classes present.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score

    y = (se_labels > threshold).astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        print("  [SE probe] Single class in labels — cannot train probe.")
        return None

    n_layers = resid_post_layers.shape[0]
    probes:  List = []
    scalers: List = []
    aurocs   = np.zeros(n_layers, dtype=np.float64)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for l in range(n_layers):
        X = resid_post_layers[l]   # (n_tokens, d_model)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        clf = LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=500, solver="lbfgs",
            random_state=42,
        )

        # Cross-validated AUROC
        try:
            cv_scores = cross_val_score(
                clf, X_sc, y, cv=skf,
                scoring="roc_auc", n_jobs=1,
            )
            aurocs[l] = float(cv_scores.mean())
        except Exception:
            aurocs[l] = 0.5

        # Fit on full data for inference
        clf.fit(X_sc, y)

        probes.append(clf)
        scalers.append(scaler)

    best_layer = int(np.argmax(aurocs))
    print(f"  [SE probe] Best layer: {best_layer}  AUROC: {aurocs[best_layer]:.4f}  "
          f"(mean across layers: {aurocs.mean():.4f})")

    return {
        "probes":     probes,
        "scalers":    scalers,
        "auroc":      aurocs,
        "best_layer": best_layer,
        "threshold":  threshold,
    }


def apply_se_probe(
    cache,
    probe_dict: Dict,
    transcript_len: int,
    note_len: int,
) -> Dict:
    """
    Apply trained SE probes to the cached residual stream activations.

    Extracts `resid_post` at each layer for note-position tokens directly from
    the cache returned by `compute_ecs_pks` — no additional forward pass needed.

    Parameters
    ----------
    cache        : TransformerLens ActivationCache from compute_ecs_pks.
    probe_dict   : output of train_se_probe().
    transcript_len, note_len : token counts.

    Returns
    -------
    dict with keys:
      se_prob       : (n_layers, note_len) — per-layer SE probability
      se_prob_best  : (note_len,)          — SE prob from best layer only
      best_layer    : int
    """
    probes     = probe_dict["probes"]
    scalers    = probe_dict["scalers"]
    best_layer = probe_dict["best_layer"]
    n_layers   = len(probes)

    se_prob = np.zeros((n_layers, note_len), dtype=np.float64)

    for l, (clf, scaler) in enumerate(zip(probes, scalers)):
        # resid_post: (1, seq_len, d_model)
        resid = cache["resid_post", l][0].float().cpu().numpy()
        # Slice note positions
        note_resid = resid[transcript_len: transcript_len + note_len]   # (note_len, d_model)
        X_sc = scaler.transform(note_resid)
        se_prob[l] = clf.predict_proba(X_sc)[:, 1]

    return {
        "se_prob":      se_prob,
        "se_prob_best": se_prob[best_layer],
        "best_layer":   best_layer,
    }
