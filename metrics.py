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
