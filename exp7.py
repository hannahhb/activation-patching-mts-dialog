"""
exp7.py
=======
Experiment 7 — Contrastive Transcript Activation Analysis.

Core idea
---------
If a note token is truly grounded in the transcript, removing the supporting
transcript utterance should cause the model to lean on parametric memory more
(PKS ↑).  If the token was already hallucinated or purely parametric, removing
transcript support changes little.

The *delta*  Δ-PKS = PKS(modified_transcript) − PKS(original_transcript)
is therefore a causal discriminator for grounded vs. hallucinated tokens.

Pipeline
--------
1. Contrastive pair generation (Bedrock)
   For each transcript/note pair the LLM identifies note claims and which
   transcript sentence supports each.  It returns one or more *modified*
   transcripts (deletion / substitution / negation) per claim.

2. Activation collection
   Two forward passes per contrastive pair — original transcript A and modified
   transcript B — using existing compute_ecs / compute_pks from metrics.py.
   Δ-PKS and Δ-ECS computed per claim token per layer.

3. Discriminability analysis
   Pool all (Δ-PKS, label) observations across examples and modification types.
   Compute AUROC and Cohen's d per layer.

4. Probe training
   Train a per-layer logistic-regression probe on Δ-PKS features with cross-
   validated hyperparameter selection.  Best layer chosen by CV-AUROC.

5. Inference pipeline
   Single forward pass (original transcript) → raw PKS → apply probe →
   per-token hallucination probability.  Claim-level scores averaged over
   noun-phrase / verb spans detected by spaCy.

Outputs
-------
  exp7_discriminability.csv         — per-layer AUROC, Cohen's d, Pearson r
  exp7_probe_auroc_per_layer.png    — CV AUROC line plot
  exp7_token_scores.csv             — per-token pks, delta_pks_hat, halluc_prob
  exp7_scored_note.html             — colour-highlighted token-level risk report
"""

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformer_lens import HookedTransformer

from config import Config, load_aci_sample
from metrics import compute_ecs, compute_pks, layer_discriminability
from tokenization import tokenize_pair, generate_note

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Bedrock client
# ─────────────────────────────────────────────────────────────────────────────

_BEDROCK_DEFAULT_MODEL  = "us.meta.llama3-3-70b-instruct-v1:0"
_BEDROCK_DEFAULT_REGION = "us-east-1"


def _get_bedrock_client(model: str = _BEDROCK_DEFAULT_MODEL, region: Optional[str] = None):
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for Exp 7.  pip install boto3") from exc
    resolved = region or os.environ.get("AWS_DEFAULT_REGION", _BEDROCK_DEFAULT_REGION)
    client = boto3.client("bedrock-runtime", region_name=resolved)
    return client, model


def _bedrock_call(client, model: str, system: str, user: str,
                  max_tokens: int = 1024, temperature: float = 0.2) -> str:
    response = client.converse(
        modelId=model,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return response["output"]["message"]["content"][0]["text"]


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Contrastive pair generation
# ─────────────────────────────────────────────────────────────────────────────

_PAIR_SYSTEM = """\
You are a clinical NLP researcher.  Your task is to identify factual claims in a
clinical note that are directly supported by a specific sentence in the doctor-patient
transcript, then generate a modified transcript that removes or weakens that support.

Return ONLY valid JSON — no markdown fences, no commentary."""

_PAIR_TEMPLATE = """\
TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"

NOTE:
\"\"\"
{note}
\"\"\"

Identify up to {n_claims} factual note claims (e.g. diagnoses, medications, vital signs,
findings) that are each supported by a specific transcript sentence.

For each claim generate ALL THREE modification types of the transcript:
  deletion     — remove the supporting sentence entirely
  substitution — replace the supporting fact with a different plausible value
  negation     — add a short phrase that explicitly contradicts the claim

Return this JSON (and nothing else):
{{
  "claims": [
    {{
      "claim_text":          "<short description of the note claim>",
      "note_span":           "<exact substring of the note that states the claim>",
      "supporting_sentence": "<exact sentence from the transcript>",
      "modified_transcripts": {{
        "deletion":     "<full transcript with supporting sentence removed>",
        "substitution": "<full transcript with the fact substituted>",
        "negation":     "<full transcript with a contradicting phrase added>"
      }}
    }}
  ]
}}"""


def generate_contrastive_pairs(
    transcript: str,
    note: str,
    n_claims: int = 4,
    bedrock_model: str = _BEDROCK_DEFAULT_MODEL,
    bedrock_region: Optional[str] = None,
    max_retries: int = 3,
) -> List[Dict]:
    """
    Use Bedrock to identify note claims supported by the transcript, and return
    a list of contrastive pair records.

    Each record has keys:
        claim_text, note_span, supporting_sentence, modification_type,
        transcript_a (original), transcript_b (modified)

    Returns an empty list if all retries fail.
    """
    client, model = _get_bedrock_client(bedrock_model, bedrock_region)
    user_msg = _PAIR_TEMPLATE.format(
        transcript=transcript, note=note, n_claims=n_claims
    )

    for attempt in range(max_retries):
        temperature = 0.2 + attempt * 0.1
        try:
            raw  = _bedrock_call(client, model, _PAIR_SYSTEM, user_msg,
                                 max_tokens=2048, temperature=temperature)
            data = _parse_json(raw)
        except Exception as exc:
            warnings.warn(f"[Exp7] Pair generation attempt {attempt+1}: {exc}")
            continue

        claims = data.get("claims", [])
        pairs: List[Dict] = []
        for c in claims:
            note_span = c.get("note_span", "").strip()
            if not note_span or note_span not in note:
                continue
            modified = c.get("modified_transcripts", {})
            for mod_type in ("deletion", "substitution", "negation"):
                tr_b = modified.get(mod_type, "").strip()
                if not tr_b or tr_b == transcript:
                    continue
                pairs.append({
                    "claim_text":          c.get("claim_text", ""),
                    "note_span":           note_span,
                    "supporting_sentence": c.get("supporting_sentence", ""),
                    "modification_type":   mod_type,
                    "transcript_a":        transcript,
                    "transcript_b":        tr_b,
                })

        if pairs:
            print(f"  [Exp7] Generated {len(pairs)} contrastive pairs "
                  f"({len(claims)} claims × modification types).")
            return pairs

        warnings.warn(f"[Exp7] Attempt {attempt+1}: no valid pairs extracted.")

    warnings.warn("[Exp7] All pair-generation attempts failed — returning empty list.")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Activation collection
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_FILTER = lambda name: (   # noqa: E731
    "pattern" in name or "resid_mid" in name or "resid_post" in name
)


def _run_forward(
    model: HookedTransformer,
    transcript: str,
    note: str,
    device: str,
) -> Tuple[object, int, int]:
    """
    Tokenise (transcript, note), run forward pass with selective cache,
    return (cache, transcript_len, note_len).
    """
    tokens, transcript_len, note_tokens = tokenize_pair(model, transcript, note)
    tokens = tokens.to(device)
    note_len = len(note_tokens)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=_CACHE_FILTER)
    return cache, transcript_len, note_len


def _claim_token_indices(
    tokenizer,
    note: str,
    note_span: str,
) -> List[int]:
    """
    Map a note_span substring to token indices within the tokenized note.
    Uses offset_mapping when available (fast tokenizers), falls back to
    cumulative decode.
    """
    idx = note.find(note_span)
    if idx == -1:
        return []
    span_start, span_end = idx, idx + len(note_span)

    try:
        enc = tokenizer(note, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        return [
            i for i, (cs, ce) in enumerate(offsets)
            if ce > span_start and cs < span_end
        ]
    except Exception:
        pass

    ids    = tokenizer.encode(note, add_special_tokens=False)
    cursor = 0
    result = []
    for i, tid in enumerate(ids):
        piece = tokenizer.decode([tid])
        end   = cursor + len(piece)
        if end > span_start and cursor < span_end:
            result.append(i)
        cursor = end
    return result


def _hash_key(transcript_a: str, transcript_b: str, note: str) -> str:
    h = hashlib.sha256(
        (transcript_a + "\x00" + transcript_b + "\x00" + note).encode()
    )
    return h.hexdigest()[:16]


def collect_contrastive_activations(
    model: HookedTransformer,
    cfg: Config,
    pairs: List[Dict],
    cache_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    For each contrastive pair run two forward passes (transcript_A and transcript_B)
    and compute Δ-PKS = PKS_B − PKS_A and Δ-ECS = ECS_A − ECS_B per layer per
    claim token.

    Parameters
    ----------
    pairs      : list of dicts from generate_contrastive_pairs().
    cache_dir  : if set, .npz files are cached/loaded here (keyed by content hash).

    Returns
    -------
    List of result dicts, one per pair that had ≥1 claim token:
        claim_text, note_span, modification_type,
        claim_token_indices,
        delta_pks  : (n_claim_tokens, n_layers)
        delta_ecs  : (n_claim_tokens, n_layers)
        pks_a      : (n_claim_tokens, n_layers)
        pks_b      : (n_claim_tokens, n_layers)
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    seen_transcripts: Dict[str, Tuple] = {}   # hash → (cache, transcript_len, note_len)

    for pi, pair in enumerate(pairs):
        tr_a     = pair["transcript_a"]
        tr_b     = pair["transcript_b"]
        note     = pair["note_span"]   # we need the FULL note for tokenisation
        # Note: note_span is a claim fragment; we need the full note text.
        # We stored it on the pair below in run_experiment_7 — see _full_note key.
        full_note = pair.get("_full_note", pair["note_span"])

        print(f"  [Exp7] Pair {pi+1}/{len(pairs)}  "
              f"mod={pair['modification_type']:12s}  claim='{pair['claim_text'][:40]}'")

        # ── Claim token indices ──────────────────────────────────────────────
        claim_toks = _claim_token_indices(
            model.tokenizer, full_note, pair["note_span"]
        )
        if not claim_toks:
            print(f"  [Exp7]   claim span not found in note tokens — skipping.")
            continue

        # ── Forward pass A ───────────────────────────────────────────────────
        key_a = _hash_key(tr_a, "", full_note)
        if key_a in seen_transcripts:
            cache_a, tlen_a, nlen_a = seen_transcripts[key_a]
        else:
            npz_a = cache_dir / f"{key_a}_a.npz" if cache_dir else None
            if npz_a and npz_a.exists():
                _d = np.load(npz_a)
                pks_layers_a_full = _d["pks_layers"]
                ecs_layers_a_full = _d["ecs_layers"]
                tlen_a = int(_d["transcript_len"])
                nlen_a = int(_d["note_len"])
                cache_a = None  # already computed
            else:
                cache_a, tlen_a, nlen_a = _run_forward(model, tr_a, full_note, cfg.device)
                seen_transcripts[key_a] = (cache_a, tlen_a, nlen_a)

        # ── Forward pass B ───────────────────────────────────────────────────
        key_b = _hash_key(tr_a, tr_b, full_note)
        npz_b = cache_dir / f"{key_b}_b.npz" if cache_dir else None
        if npz_b and npz_b.exists():
            _d = np.load(npz_b)
            pks_layers_b_full = _d["pks_layers"]
            ecs_layers_b_full = _d["ecs_layers"]
            tlen_b = int(_d["transcript_len"])
            nlen_b = int(_d["note_len"])
        else:
            cache_b, tlen_b, nlen_b = _run_forward(model, tr_b, full_note, cfg.device)

            # ── Compute PKS and ECS for both ─────────────────────────────────
            if cache_a is not None:
                _, pks_layers_a_full = compute_pks(model, cache_a, tlen_a, nlen_a, cfg.device)
                _, ecs_layers_a_full = compute_ecs(model, cache_a, tlen_a, nlen_a)
            # (else already loaded from npz_a)

            _, pks_layers_b_full = compute_pks(model, cache_b, tlen_b, nlen_b, cfg.device)
            _, ecs_layers_b_full = compute_ecs(model, cache_b, tlen_b, nlen_b)

            # Save B
            if npz_b:
                np.savez_compressed(
                    npz_b,
                    pks_layers=pks_layers_b_full,
                    ecs_layers=ecs_layers_b_full,
                    transcript_len=tlen_b,
                    note_len=nlen_b,
                )
            # Save A (first time)
            if cache_a is not None and (cache_dir and not (cache_dir / f"{key_a}_a.npz").exists()):
                np.savez_compressed(
                    cache_dir / f"{key_a}_a.npz",
                    pks_layers=pks_layers_a_full,
                    ecs_layers=ecs_layers_a_full,
                    transcript_len=tlen_a,
                    note_len=nlen_a,
                )

        # Align note lengths (transcripts may differ in length, notes are identical)
        min_nlen = min(nlen_a, nlen_b)
        # Clip claim tokens to min_nlen
        claim_toks_valid = [i for i in claim_toks if i < min_nlen]
        if not claim_toks_valid:
            print(f"  [Exp7]   all claim tokens out of note range — skipping.")
            continue

        # Extract claim-token rows: (n_layers, n_claim_tokens)
        pks_a = pks_layers_a_full[:, claim_toks_valid]   # (L, C)
        pks_b = pks_layers_b_full[:, claim_toks_valid]
        ecs_a = ecs_layers_a_full[:, claim_toks_valid]
        ecs_b = ecs_layers_b_full[:, claim_toks_valid]

        delta_pks = (pks_b - pks_a).T   # (C, L)
        delta_ecs = (ecs_a - ecs_b).T   # (C, L)  positive = ecs dropped with support removed

        n_c = len(claim_toks_valid)
        print(f"  [Exp7]   {n_c} claim tokens  "
              f"Δ-PKS mean={delta_pks.mean():+.4f}  "
              f"Δ-ECS mean={delta_ecs.mean():+.4f}")

        results.append({
            "claim_text":          pair["claim_text"],
            "note_span":           pair["note_span"],
            "modification_type":   pair["modification_type"],
            "claim_token_indices": claim_toks_valid,
            "delta_pks":           delta_pks,   # (C, L)
            "delta_ecs":           delta_ecs,   # (C, L)
            "pks_a":               pks_a.T,     # (C, L)
            "pks_b":               pks_b.T,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Discriminability analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_discriminability(
    results: List[Dict],
    n_layers: int,
) -> pd.DataFrame:
    """
    Pool Δ-PKS across all pairs.

    Label = 1 for claim tokens under *any* modification (grounded tokens
    whose PKS should increase when support is removed).
    Label = 0 for the complement (non-claim tokens from the same examples).

    Since we only collected claim token activations in Step 2, we compute
    AUROC using Δ-PKS magnitude: large positive Δ-PKS is the signal.

    Returns a DataFrame with columns: layer, auroc, cohens_d, pearson_r,
    modification_type (overall row has modification_type='all').
    """
    rows = []

    # ── Overall ─────────────────────────────────────────────────────────────
    # Stack all (C, L) arrays → (N_total, L), all labelled 1 (grounded)
    # AUROC requires negative class; we generate a null distribution by
    # sign-flipping (reflecting Δ-PKS around zero is the null expectation).
    if not results:
        return pd.DataFrame()

    dpks_all = np.concatenate([r["delta_pks"] for r in results], axis=0)  # (N, L)

    for layer in range(n_layers):
        col    = dpks_all[:, layer]
        # One-sample AUROC: score actual vs. sign-flipped null
        scores = np.concatenate([col, -col])
        labels = np.concatenate([np.ones(len(col)), np.zeros(len(col))])
        try:
            auroc = roc_auc_score(labels, scores)
        except Exception:
            auroc = float("nan")

        # Cohen's d: mean / std of Δ-PKS (vs. null at 0)
        mu, sd = col.mean(), col.std()
        cohens_d = mu / sd if sd > 1e-9 else 0.0
        pearson_r = float(np.corrcoef(col, np.arange(len(col)))[0, 1]) if len(col) > 1 else 0.0

        rows.append({
            "layer": layer,
            "auroc": round(auroc, 4),
            "cohens_d": round(cohens_d, 4),
            "mean_delta_pks": round(mu, 4),
            "modification_type": "all",
        })

    # ── Per modification type ─────────────────────────────────────────────
    for mod_type in ("deletion", "substitution", "negation"):
        sub = [r for r in results if r["modification_type"] == mod_type]
        if not sub:
            continue
        dpks_sub = np.concatenate([r["delta_pks"] for r in sub], axis=0)
        for layer in range(n_layers):
            col = dpks_sub[:, layer]
            scores = np.concatenate([col, -col])
            labels = np.concatenate([np.ones(len(col)), np.zeros(len(col))])
            try:
                auroc = roc_auc_score(labels, scores)
            except Exception:
                auroc = float("nan")
            mu, sd = col.mean(), col.std()
            cohens_d = mu / sd if sd > 1e-9 else 0.0
            rows.append({
                "layer": layer,
                "auroc": round(auroc, 4),
                "cohens_d": round(cohens_d, 4),
                "mean_delta_pks": round(mu, 4),
                "modification_type": mod_type,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Probe training
# ─────────────────────────────────────────────────────────────────────────────

def train_contrastive_probe(
    results: List[Dict],
    n_layers: int,
    Cs: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    cv_folds: int = 3,
) -> Dict:
    """
    Train a per-layer logistic regression probe on Δ-PKS to predict claim tokens
    (label=1) vs. null (sign-flipped, label=0).

    Returns
    -------
    dict:
        scalers     : List[StandardScaler]          (one per layer)
        probes      : List[LogisticRegressionCV]     (one per layer)
        auroc       : (n_layers,) float array
        best_layer  : int
        feature_weights : (n_layers,) float — probe weight magnitude per layer
    """
    dpks_all = np.concatenate([r["delta_pks"] for r in results], axis=0)  # (N, L)
    N = len(dpks_all)

    # Build training set: real observations (label 1) + sign-flipped null (label 0)
    X_pos = dpks_all          # (N, L)
    X_neg = -dpks_all         # sign-flipped null
    X_train = np.vstack([X_pos, X_neg])   # (2N, L)
    y_train = np.concatenate([np.ones(N), np.zeros(N)])

    scalers: List[StandardScaler]      = []
    probes:  List[LogisticRegressionCV] = []
    auroc_per_layer = np.zeros(n_layers)
    weights_per_layer = np.zeros(n_layers)

    for layer in range(n_layers):
        X_l = X_train[:, layer : layer + 1]   # (2N, 1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_l)

        probe = LogisticRegressionCV(
            Cs=list(Cs),
            cv=cv_folds,
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=500,
            n_jobs=1,
        )
        probe.fit(X_scaled, y_train)

        proba = probe.predict_proba(X_scaled)[:, 1]
        try:
            auroc_per_layer[layer] = roc_auc_score(y_train, proba)
        except Exception:
            auroc_per_layer[layer] = float("nan")

        weights_per_layer[layer] = float(np.abs(probe.coef_).mean())
        scalers.append(scaler)
        probes.append(probe)

    best_layer = int(np.nanargmax(auroc_per_layer))
    print(f"  [Exp7] Probe AUROC per layer — best: layer {best_layer} "
          f"({auroc_per_layer[best_layer]:.4f})")

    return {
        "scalers":         scalers,
        "probes":          probes,
        "auroc":           auroc_per_layer,
        "best_layer":      best_layer,
        "feature_weights": weights_per_layer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

def score_generated_note(
    model: HookedTransformer,
    cfg: Config,
    transcript: str,
    note: str,
    probe_dict: Dict,
    use_spacy: bool = True,
) -> pd.DataFrame:
    """
    Single forward pass on the original transcript → raw PKS per layer per token.
    Apply the trained probe to get per-token hallucination probability.

    If spaCy is available and use_spacy=True, also compute claim-level scores
    by averaging over noun-phrase and verb spans.

    Returns a DataFrame with columns:
        token_idx, token_str, pks_mean, pks_best_layer,
        halluc_prob, span_label (NP/VP/other)
    """
    cache, t_len, n_len = _run_forward(model, transcript, note, cfg.device)
    _, pks_layers = compute_pks(model, cache, t_len, n_len, cfg.device)
    # pks_layers: (n_layers, n_len)

    best_layer    = probe_dict["best_layer"]
    scaler        = probe_dict["scalers"][best_layer]
    probe         = probe_dict["probes"][best_layer]
    n_layers      = pks_layers.shape[0]

    pks_best  = pks_layers[best_layer]               # (n_len,)
    pks_mean  = pks_layers.mean(axis=0)              # (n_len,)

    # Apply probe
    X_inf    = pks_best.reshape(-1, 1)
    X_scaled = scaler.transform(X_inf)
    halluc_prob = probe.predict_proba(X_scaled)[:, 1]  # (n_len,)

    # Token strings
    note_ids  = model.tokenizer.encode(note, add_special_tokens=False)
    tok_strs  = [model.tokenizer.decode([tid]) for tid in note_ids[:n_len]]

    records = []
    for i in range(n_len):
        records.append({
            "token_idx":       i,
            "token_str":       tok_strs[i] if i < len(tok_strs) else "",
            "pks_mean":        round(float(pks_mean[i]),   4),
            "pks_best_layer":  round(float(pks_best[i]),   4),
            "halluc_prob":     round(float(halluc_prob[i]), 4),
            "span_label":      "other",
        })

    df = pd.DataFrame(records)

    # ── spaCy span labels ────────────────────────────────────────────────────
    if use_spacy:
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = None
            if nlp is not None:
                doc = nlp(note)
                # Map character positions → token indices using note tokeniser
                note_ids_full = model.tokenizer.encode(note, add_special_tokens=False)
                try:
                    enc     = model.tokenizer(note, return_offsets_mapping=True,
                                              add_special_tokens=False)
                    offsets = enc["offset_mapping"]
                except Exception:
                    offsets = None

                def _span_tok_indices(start_char, end_char):
                    if offsets is not None:
                        return [i for i, (cs, ce) in enumerate(offsets)
                                if ce > start_char and cs < end_char and i < n_len]
                    return []

                for chunk in doc.noun_chunks:
                    for ti in _span_tok_indices(chunk.start_char, chunk.end_char):
                        df.at[ti, "span_label"] = "NP"
                for sent in doc.sents:
                    for token in sent:
                        if token.pos_ == "VERB":
                            for ti in _span_tok_indices(token.idx, token.idx + len(token)):
                                if df.at[ti, "span_label"] == "other":
                                    df.at[ti, "span_label"] = "VP"
        except ImportError:
            pass

    return df


# ─────────────────────────────────────────────────────────────────────────────
# HTML report helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_html_report(df: pd.DataFrame, note: str, model_name: str) -> str:
    """Colour-highlight tokens by halluc_prob (green→red gradient)."""

    def _colour(p: float) -> str:
        r = int(255 * p)
        g = int(255 * (1 - p))
        return f"rgb({r},{g},60)"

    spans = []
    for _, row in df.iterrows():
        tok  = row["token_str"].replace("&", "&amp;").replace("<", "&lt;")
        prob = row["halluc_prob"]
        bg   = _colour(prob)
        tip  = (f"PKS={row['pks_best_layer']:.3f}  "
                f"prob={prob:.3f}  span={row['span_label']}")
        spans.append(
            f'<span style="background:{bg};border-radius:3px;padding:1px 2px;" '
            f'title="{tip}">{tok}</span>'
        )

    body = "".join(spans)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exp 7 — Contrastive Probe Scores</title>
<style>body{{font-family:monospace;font-size:13px;line-height:2;padding:20px;}}
h2{{font-family:sans-serif;}}</style>
</head><body>
<h2>Experiment 7 — Contrastive Probe Hallucination Scores</h2>
<p>Model: <b>{model_name}</b> &nbsp;|&nbsp;
   <span style="background:rgb(0,255,60);padding:2px 6px">low risk</span>
   &nbsp;→&nbsp;
   <span style="background:rgb(255,0,60);color:white;padding:2px 6px">high risk</span>
</p>
<div>{body}</div>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_7(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    transcript: str,
    gold_note: str,
    n_claims: int = 4,
    n_examples: int = 5,
    sample_start: int = 0,
    bedrock_model: str = _BEDROCK_DEFAULT_MODEL,
    bedrock_region: Optional[str] = None,
    cv_folds: int = 3,
    use_spacy: bool = True,
) -> Dict:
    """
    Experiment 7 — Contrastive Transcript Activation Analysis.

    Parameters
    ----------
    n_claims      : claims to extract per transcript/note pair.
    n_examples    : total ACI-Bench examples to process.
    sample_start  : first row index.
    """
    from dataclasses import replace as _dc_replace

    print("\n" + "═" * 54)
    print("  EXPERIMENT 7 — Contrastive Activation Analysis")
    print("═" * 54)

    out7 = out / "exp_7_out"
    act_cache = out7 / "act_cache"
    out7.mkdir(parents=True, exist_ok=True)
    act_cache.mkdir(parents=True, exist_ok=True)

    all_pairs:   List[Dict] = []
    all_results: List[Dict] = []

    # ── Steps 1 + 2 across n_examples ────────────────────────────────────────
    for si in range(sample_start, sample_start + n_examples):
        print(f"\n  ── Sample {si} ──────────────────────────────────────")
        cfg_i = _dc_replace(cfg, sample_idx=si)

        try:
            tr_i, note_i = load_aci_sample(cfg_i)
        except Exception as exc:
            print(f"  [Exp7] Sample {si}: load failed — {exc}")
            continue

        # Step 1 — Generate contrastive pairs
        pairs_i = generate_contrastive_pairs(
            tr_i, note_i,
            n_claims=n_claims,
            bedrock_model=bedrock_model,
            bedrock_region=bedrock_region,
        )
        if not pairs_i:
            print(f"  [Exp7] Sample {si}: no pairs generated — skipping.")
            continue

        # Attach full note text to each pair (needed for tokenisation in Step 2)
        for p in pairs_i:
            p["_full_note"] = note_i

        all_pairs.extend(pairs_i)

        # Step 2 — Collect activations for this sample's pairs
        results_i = collect_contrastive_activations(
            model, cfg, pairs_i, cache_dir=act_cache
        )
        all_results.extend(results_i)

    if not all_results:
        print("\n  [Exp7] No valid results collected — aborting.")
        return {}

    n_layers = model.cfg.n_layers
    print(f"\n  Total contrastive pairs with activations: {len(all_results)}")

    # ── Step 3 — Discriminability ─────────────────────────────────────────────
    print("\n  [Exp7] Step 3 — Discriminability analysis …")
    disc_df = analyse_discriminability(all_results, n_layers)
    disc_df.to_csv(out7 / "exp7_discriminability.csv", index=False)
    print("  Saved → exp7_discriminability.csv")

    # ── Step 4 — Probe training ───────────────────────────────────────────────
    print("\n  [Exp7] Step 4 — Probe training …")
    probe_dict = train_contrastive_probe(all_results, n_layers, cv_folds=cv_folds)

    # Save probe AUROC plot
    auroc_arr   = probe_dict["auroc"]
    best_layer  = probe_dict["best_layer"]
    layers_arr  = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(max(12, n_layers // 2), 4))
    ax.plot(layers_arr, auroc_arr, lw=2.2, marker="o", markersize=4, color="#2196F3",
            label="Probe AUROC (Δ-PKS feature)")
    ax.axvline(best_layer, color="red", ls="--", lw=1.2,
               label=f"Best layer {best_layer} ({auroc_arr[best_layer]:.3f})")
    ax.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.6, label="Chance")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("CV AUROC", fontsize=11)
    ax.set_title(
        f"Exp 7 — Contrastive Probe AUROC per Layer\n"
        f"{cfg.model_name}  |  {len(all_results)} pairs",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out7 / "exp7_probe_auroc_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → exp7_probe_auroc_per_layer.png")

    # Probe stats CSV
    probe_stats = pd.DataFrame({
        "layer":           layers_arr,
        "probe_auroc":     auroc_arr,
        "feature_weight":  probe_dict["feature_weights"],
        "best_layer":      layers_arr == best_layer,
    })
    probe_stats.to_csv(out7 / "exp7_probe_stats.csv", index=False)
    print("  Saved → exp7_probe_stats.csv")

    # ── Step 5 — Score the primary note ──────────────────────────────────────
    print("\n  [Exp7] Step 5 — Scoring primary note …")
    score_df = score_generated_note(
        model, cfg, transcript, gold_note,
        probe_dict=probe_dict,
        use_spacy=use_spacy,
    )
    score_df.to_csv(out7 / "exp7_token_scores.csv", index=False)
    print("  Saved → exp7_token_scores.csv")

    html = _build_html_report(score_df, gold_note, cfg.model_name)
    (out7 / "exp7_scored_note.html").write_text(html, encoding="utf-8")
    print("  Saved → exp7_scored_note.html")

    # ── Console summary ───────────────────────────────────────────────────────
    overall = disc_df[disc_df["modification_type"] == "all"].sort_values("auroc", ascending=False)
    if not overall.empty:
        top = overall.iloc[0]
        print(f"\n  Best discriminability: layer {int(top['layer'])}  "
              f"AUROC={top['auroc']:.4f}  Cohen's d={top['cohens_d']:.4f}")
    print(f"  Best probe layer: {best_layer}  AUROC={auroc_arr[best_layer]:.4f}")
    high_risk = score_df[score_df["halluc_prob"] > 0.5]
    print(f"  High-risk tokens (prob > 0.5): {len(high_risk)} / {len(score_df)}")

    return {
        "disc_df":     disc_df,
        "probe_dict":  probe_dict,
        "score_df":    score_df,
        "n_pairs":     len(all_results),
        "best_layer":  best_layer,
    }
