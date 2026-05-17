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
from halluc_llm import BedrockHallucinator

warnings.filterwarnings("ignore")

_BEDROCK_DEFAULT_MODEL  = "us.meta.llama3-3-70b-instruct-v1:0"
_BEDROCK_DEFAULT_REGION = "us-east-1"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Contrastive pair generation
# ─────────────────────────────────────────────────────────────────────────────

_PAIR_SYSTEM = """\
You are a clinical NLP researcher building a hallucination-detection dataset.
Given a doctor-patient transcript and the clinical note written from it, identify
factual claims in the note that are each supported by a specific line in the
transcript.  Then rewrite the transcript to remove or weaken that support."""

_PAIR_TEMPLATE = """\
TRANSCRIPT:
{transcript}

NOTE:
{note}

Task: Find up to {n_claims} factual claims in the note (medications, diagnoses, \
vital signs, findings, dosages) that are each supported by a specific transcript line.
For each claim, produce a modified transcript using modification type "{mod_type}":
  deletion     = remove the supporting line entirely
  substitution = replace the fact in the supporting line with a different plausible value
  negation     = add a short phrase near the supporting line that contradicts the claim

For each claim write a block in EXACTLY this format (no extra lines between fields):
CLAIM: <one-line description of the note claim>
NOTE_SPAN: <short exact phrase from the note stating the claim, e.g. "hypertension" or "10 mg">
SUPPORTING_LINE: <the transcript line that supports it>
MODIFIED_TRANSCRIPT:
<the full modified transcript, then a line with only three dashes --->

Repeat the block for each claim.  No other text."""


def _fuzzy_find(text: str, span: str) -> Optional[int]:
    """Exact → case-insensitive → first-20-char prefix match. Returns char idx or None."""
    idx = text.find(span)
    if idx != -1:
        return idx
    idx = text.lower().find(span.lower())
    if idx != -1:
        return idx
    prefix = span[:20].lower().strip()
    if prefix:
        idx = text.lower().find(prefix)
        if idx != -1:
            return idx
    return None


def _parse_pairs_text(raw: str, transcript: str, note: str, mod_type: str) -> List[Dict]:
    """
    Parse the plain-text block format returned by the LLM.

    Splits on CLAIM: lines so the response works with or without --- separators.
    MODIFIED_TRANSCRIPT captures everything up to the next labelled field or
    end-of-string.
    """
    pairs: List[Dict] = []

    # Split into per-claim chunks on lines that start with CLAIM:
    # (handles both --- separated and unseparated responses)
    chunks = re.split(r"(?=^CLAIM:)", raw, flags=re.MULTILINE)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or not chunk.startswith("CLAIM:"):
            continue

        claim_m = re.search(r"^CLAIM:\s*(.+)$",           chunk, re.MULTILINE)
        span_m  = re.search(r"^NOTE_SPAN:\s*(.+)$",       chunk, re.MULTILINE)
        sup_m   = re.search(r"^SUPPORTING_LINE:\s*(.+)$", chunk, re.MULTILINE)
        # Capture from MODIFIED_TRANSCRIPT: through to the next labelled keyword
        # or end of chunk.  The lookahead stops at any ALL_CAPS_WORD: pattern.
        mod_m   = re.search(
            r"^MODIFIED_TRANSCRIPT:\s*\n([\s\S]+?)(?=\n[A-Z_]+:|\Z)",
            chunk, re.MULTILINE,
        )

        claim_text  = claim_m.group(1).strip() if claim_m else ""
        note_span   = span_m.group(1).strip()  if span_m  else ""
        sup_line    = sup_m.group(1).strip()   if sup_m   else ""
        modified_tr = mod_m.group(1).strip()   if mod_m   else ""

        if not note_span:
            print(f"  [Exp7] block skipped — missing NOTE_SPAN  claim='{claim_text[:50]}'")
            continue
        if not modified_tr:
            print(f"  [Exp7] block skipped — missing MODIFIED_TRANSCRIPT  "
                  f"note_span='{note_span[:50]}'")
            continue
        if modified_tr == transcript:
            print(f"  [Exp7] block skipped — MODIFIED_TRANSCRIPT identical to original  "
                  f"note_span='{note_span[:50]}'")
            continue

        idx = _fuzzy_find(note, note_span)
        if idx is None:
            print(f"  [Exp7] block skipped — NOTE_SPAN '{note_span[:50]}' not found in note")
            continue
        resolved_span = note[idx: idx + len(note_span)]

        pairs.append({
            "claim_text":        claim_text,
            "note_span":         resolved_span,
            "supporting_line":   sup_line,
            "modification_type": mod_type,
            "transcript_a":      transcript,
            "transcript_b":      modified_tr,
        })
        print(f"  [Exp7]   accepted: mod={mod_type}  span='{resolved_span[:50]}'")

    return pairs


def generate_contrastive_pairs(
    transcript: str,
    note: str,
    n_claims: int = 4,
    bedrock_model: str = _BEDROCK_DEFAULT_MODEL,
    bedrock_region: Optional[str] = None,
    max_retries: int = 3,
) -> List[Dict]:
    """
    Use Bedrock to identify note claims and produce contrastive transcript pairs.

    One Bedrock call per modification type (deletion / substitution / negation).
    Response is plain text with a simple labeled block format — no JSON.
    """
    hallucinator = BedrockHallucinator(model=bedrock_model, region=bedrock_region)
    client   = hallucinator._client
    model_id = hallucinator.model

    pairs: List[Dict] = []

    for mod_type in ("deletion", "substitution", "negation"):
        user_msg = _PAIR_TEMPLATE.format(
            transcript=transcript,
            note=note,
            n_claims=n_claims,
            mod_type=mod_type,
        )

        for attempt in range(max_retries):
            temperature = 0.3 + attempt * 0.1
            try:
                response = client.converse(
                    modelId=model_id,
                    system=[{"text": _PAIR_SYSTEM}],
                    messages=[{"role": "user", "content": [{"text": user_msg}]}],
                    inferenceConfig={"maxTokens": 2048, "temperature": temperature},
                )
                raw = response["output"]["message"]["content"][0]["text"]
            except Exception as exc:
                warnings.warn(f"[Exp7] mod={mod_type} attempt {attempt+1}: Bedrock call failed — {exc}")
                continue

            print(f"  [Exp7] mod={mod_type} attempt {attempt+1} — "
                  f"response length {len(raw)} chars")

            new_pairs = _parse_pairs_text(raw, transcript, note, mod_type)
            if new_pairs:
                pairs.extend(new_pairs)
                print(f"  [Exp7] mod={mod_type}: {len(new_pairs)} pair(s) accepted")
                break
            else:
                print(f"  [Exp7] mod={mod_type} attempt {attempt+1}: no valid blocks found, retrying …")
                print(f"  [Exp7] raw (first 400 chars): {raw[:400]}")
        else:
            warnings.warn(f"[Exp7] mod={mod_type}: all {max_retries} attempts yielded no pairs")

    print(f"  [Exp7] Total pairs generated: {len(pairs)}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Activation collection
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_FILTER = lambda name: (   # noqa: E731
    "pattern" in name or "resid_mid" in name or "resid_post" in name
)

# Maximum transcript tokens to feed into the model.
# Attention cache is O(S²) so long transcripts dominate memory and time.
# We keep the TAIL of the transcript (most recent dialogue) so the
# supporting line — which is typically near the relevant claim — is retained.
_MAX_TRANSCRIPT_TOKENS = 512


def _truncate_transcript(model: HookedTransformer, transcript: str) -> str:
    """
    Tokenise transcript and, if it exceeds _MAX_TRANSCRIPT_TOKENS, decode only
    the last _MAX_TRANSCRIPT_TOKENS tokens back to a string.
    This keeps the most recent (most claim-relevant) dialogue turns.
    """
    ids = model.tokenizer.encode(transcript, add_special_tokens=False)
    if len(ids) <= _MAX_TRANSCRIPT_TOKENS:
        return transcript
    ids_trimmed = ids[-_MAX_TRANSCRIPT_TOKENS:]
    return model.tokenizer.decode(ids_trimmed)


def _run_forward(
    model: HookedTransformer,
    transcript: str,
    note: str,
    device: str,
) -> Tuple[object, int, int]:
    """
    Tokenise (transcript, note), run forward pass with selective cache,
    return (cache, transcript_len, note_len).

    The transcript is truncated to _MAX_TRANSCRIPT_TOKENS before tokenisation
    to keep the attention cache at a manageable size.
    """
    transcript = _truncate_transcript(model, transcript)
    tokens, transcript_len, note_tokens = tokenize_pair(model, transcript, note)
    tokens = tokens.to(device)
    note_len = len(note_tokens)
    n_total = tokens.shape[1]
    print(f"    [fwd] seq_len={n_total}  (transcript={transcript_len}  note={note_len})")
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

def _disc_stats_for(delta: np.ndarray, layer: int) -> Dict:
    """
    One-sample discriminability for a single signal column.

    We have only grounded claim tokens (positive class).  The null is generated
    by sign-flipping — reflecting Δ around zero gives the distribution expected
    under no effect.  AUROC > 0.5 means the signal is consistently positive
    (grounded tokens show the expected direction of change).

    Returns dict: auroc, cohens_d, mean_delta.
    """
    col = delta[:, layer]
    scores = np.concatenate([col, -col])
    labels = np.concatenate([np.ones(len(col)), np.zeros(len(col))])
    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = float("nan")
    mu, sd = float(col.mean()), float(col.std())
    cohens_d = mu / sd if sd > 1e-9 else 0.0
    return {"auroc": round(auroc, 4), "cohens_d": round(cohens_d, 4),
            "mean_delta": round(mu, 4)}


def analyse_discriminability(
    results: List[Dict],
    n_layers: int,
) -> pd.DataFrame:
    """
    Pool Δ-PKS **and** Δ-ECS across all pairs and report per-layer
    discriminability for each signal.

    The null distribution is generated by sign-flipping (Δ → −Δ), so AUROC > 0.5
    means claim tokens show a consistently positive shift in that signal:
      Δ-PKS > 0  ⟹  parametric memory increased when transcript support was removed
      Δ-ECS > 0  ⟹  transcript grounding decreased when support was removed

    Both are expected to be positive for genuinely grounded tokens.

    Returns a DataFrame with columns:
        signal, modification_type, layer,
        auroc, cohens_d, mean_delta
    """
    if not results:
        return pd.DataFrame()

    rows = []

    for signal_key, signal_name in [("delta_pks", "delta_pks"), ("delta_ecs", "delta_ecs")]:
        groups = {"all": results}
        for mod_type in ("deletion", "substitution", "negation"):
            sub = [r for r in results if r["modification_type"] == mod_type]
            if sub:
                groups[mod_type] = sub

        for grp_name, grp in groups.items():
            delta_all = np.concatenate([r[signal_key] for r in grp], axis=0)  # (N, L)
            for layer in range(n_layers):
                stats = _disc_stats_for(delta_all, layer)
                rows.append({
                    "signal":            signal_name,
                    "modification_type": grp_name,
                    "layer":             layer,
                    **stats,
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Probe training with held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def _build_Xy(results: List[Dict], layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for a single layer from a list of result dicts.

    Positive class (y=1): actual Δ-PKS for each claim token.
    Negative class (y=0): sign-flipped Δ-PKS (null distribution — no effect).

    X shape: (2*N_tokens, 1)
    """
    dpks = np.concatenate([r["delta_pks"] for r in results], axis=0)  # (N, L)
    col  = dpks[:, layer : layer + 1]   # (N, 1)
    X = np.vstack([col, -col])
    y = np.concatenate([np.ones(len(col)), np.zeros(len(col))])
    return X, y


def train_contrastive_probe(
    train_results: List[Dict],
    test_results:  List[Dict],
    n_layers: int,
    Cs: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    cv_folds: int = 3,
) -> Dict:
    """
    Train a per-layer logistic regression probe on Δ-PKS features and evaluate
    on a held-out test set.

    Split is done at the *example* level before this function is called (the
    orchestrator partitions results by sample index), so no example's pairs
    appear in both train and test.

    Positive class: actual Δ-PKS for claim tokens (grounded tokens should show
    PKS increase when transcript support is removed).
    Negative class: sign-flipped Δ-PKS (null — no systematic shift expected).

    Parameters
    ----------
    train_results : list of result dicts for training examples.
    test_results  : list of result dicts for held-out test examples.

    Returns
    -------
    dict:
        scalers          : List[StandardScaler]           (one per layer)
        probes           : List[LogisticRegressionCV]     (one per layer)
        train_auroc      : (n_layers,) — train-set AUROC (in-sample)
        test_auroc       : (n_layers,) — test-set  AUROC (held-out)
        best_layer       : int — layer with highest test AUROC
        feature_weights  : (n_layers,) — |probe weight| per layer
        n_train_tokens   : int
        n_test_tokens    : int
    """
    scalers:           List[StandardScaler]       = []
    probes:            List[LogisticRegressionCV] = []
    train_auroc        = np.zeros(n_layers)
    test_auroc         = np.zeros(n_layers)
    weights_per_layer  = np.zeros(n_layers)

    n_train_tokens = sum(r["delta_pks"].shape[0] for r in train_results)
    n_test_tokens  = sum(r["delta_pks"].shape[0] for r in test_results) if test_results else 0

    for layer in range(n_layers):
        X_tr, y_tr = _build_Xy(train_results, layer)

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)

        probe = LogisticRegressionCV(
            Cs=list(Cs),
            cv=cv_folds,
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=500,
            n_jobs=1,
        )
        probe.fit(X_tr_sc, y_tr)

        # Train AUROC
        try:
            train_auroc[layer] = roc_auc_score(
                y_tr, probe.predict_proba(X_tr_sc)[:, 1]
            )
        except Exception:
            train_auroc[layer] = float("nan")

        # Test AUROC (held-out)
        if test_results:
            X_te, y_te = _build_Xy(test_results, layer)
            X_te_sc    = scaler.transform(X_te)
            try:
                test_auroc[layer] = roc_auc_score(
                    y_te, probe.predict_proba(X_te_sc)[:, 1]
                )
            except Exception:
                test_auroc[layer] = float("nan")
        else:
            test_auroc[layer] = float("nan")

        weights_per_layer[layer] = float(np.abs(probe.coef_).mean())
        scalers.append(scaler)
        probes.append(probe)

    # Best layer by *test* AUROC (fall back to train if no test set)
    rank_arr  = test_auroc if test_results else train_auroc
    best_layer = int(np.nanargmax(rank_arr))

    print(f"  [Exp7] Probe  train AUROC best layer {best_layer}: "
          f"{train_auroc[best_layer]:.4f}  |  "
          f"test AUROC: {test_auroc[best_layer]:.4f}  "
          f"({n_train_tokens} train tokens / {n_test_tokens} test tokens)")

    return {
        "scalers":         scalers,
        "probes":          probes,
        "train_auroc":     train_auroc,
        "test_auroc":      test_auroc,
        "best_layer":      best_layer,
        "feature_weights": weights_per_layer,
        "n_train_tokens":  n_train_tokens,
        "n_test_tokens":   n_test_tokens,
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
    test_fraction: float = 0.3,
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
    test_fraction : fraction of examples held out for probe evaluation (default 0.3).
                    Split is at the example level — no pairs from the same example
                    appear in both train and test.
    """
    from dataclasses import replace as _dc_replace
    import math

    print("\n" + "═" * 54)
    print("  EXPERIMENT 7 — Contrastive Activation Analysis")
    print("═" * 54)

    out7      = out / "exp_7_out"
    act_cache = out7 / "act_cache"
    out7.mkdir(parents=True, exist_ok=True)
    act_cache.mkdir(parents=True, exist_ok=True)

    # results_by_sample[si] = list of result dicts for that sample
    results_by_sample: Dict[int, List[Dict]] = {}

    # ── Steps 1 + 2 across n_examples ────────────────────────────────────────
    sample_indices = list(range(sample_start, sample_start + n_examples))
    for si in sample_indices:
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

        for p in pairs_i:
            p["_full_note"] = note_i

        # Step 2 — Collect activations
        results_i = collect_contrastive_activations(
            model, cfg, pairs_i, cache_dir=act_cache
        )
        if results_i:
            results_by_sample[si] = results_i

    if not results_by_sample:
        print("\n  [Exp7] No valid results collected — aborting.")
        return {}

    # ── Train / test split at the example level ───────────────────────────────
    collected_samples = sorted(results_by_sample.keys())
    n_collected       = len(collected_samples)
    n_test            = max(1, math.ceil(n_collected * test_fraction))
    n_train           = n_collected - n_test

    # Hold out the last n_test samples (deterministic, reproducible)
    train_samples = collected_samples[:n_train]
    test_samples  = collected_samples[n_train:]

    train_results = [r for si in train_samples for r in results_by_sample[si]]
    test_results  = [r for si in test_samples  for r in results_by_sample[si]]
    all_results   = train_results + test_results

    print(f"\n  Examples collected : {n_collected}  "
          f"(train: {n_train} samples / {len(train_results)} pairs  |  "
          f"test: {n_test} samples / {len(test_results)} pairs)")

    n_layers   = model.cfg.n_layers
    layers_arr = np.arange(n_layers)

    # ── Step 3 — Discriminability (all data, both signals) ───────────────────
    print("\n  [Exp7] Step 3 — Discriminability analysis …")
    disc_df = analyse_discriminability(all_results, n_layers)
    disc_df.to_csv(out7 / "exp7_discriminability.csv", index=False)
    print("  Saved → exp7_discriminability.csv")

    # Plot discriminability — one subplot per signal
    fig_d, axes = plt.subplots(1, 2, figsize=(max(16, n_layers // 2 * 2), 4), sharey=False)
    colours = {"all": "#2196F3", "deletion": "#4CAF50",
               "substitution": "#FF9800", "negation": "#9C27B0"}
    for ax, signal in zip(axes, ["delta_pks", "delta_ecs"]):
        sub = disc_df[disc_df["signal"] == signal]
        for mod_type, grp in sub.groupby("modification_type"):
            grp_s = grp.sort_values("layer")
            ax.plot(grp_s["layer"], grp_s["auroc"],
                    lw=2.0 if mod_type == "all" else 1.2,
                    ls="-"  if mod_type == "all" else "--",
                    marker="o", markersize=3,
                    color=colours.get(mod_type, "gray"),
                    label=mod_type)
        ax.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.6)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("AUROC", fontsize=10)
        ax.set_title(f"Discriminability — {signal}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig_d.suptitle(
        f"Exp 7 — Δ-PKS and Δ-ECS Discriminability\n{cfg.model_name}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig_d.savefig(out7 / "exp7_discriminability.png", dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print("  Saved → exp7_discriminability.png")

    # ── Step 4 — Probe training with held-out test set ───────────────────────
    print("\n  [Exp7] Step 4 — Probe training …")
    probe_dict = train_contrastive_probe(
        train_results, test_results, n_layers, cv_folds=cv_folds
    )

    best_layer     = probe_dict["best_layer"]
    train_auroc    = probe_dict["train_auroc"]
    test_auroc     = probe_dict["test_auroc"]

    # Probe AUROC plot — train vs test
    fig_p, ax_p = plt.subplots(figsize=(max(12, n_layers // 2), 4))
    ax_p.plot(layers_arr, train_auroc, lw=2.2, marker="o", markersize=3,
              color="#2196F3", label=f"Train AUROC ({n_train} examples)")
    ax_p.plot(layers_arr, test_auroc, lw=2.2, marker="s", markersize=3,
              color="#FF5722", label=f"Test AUROC ({n_test} examples, held-out)")
    ax_p.axvline(best_layer, color="red", ls="--", lw=1.2,
                 label=f"Best layer {best_layer}  "
                       f"(test={test_auroc[best_layer]:.3f})")
    ax_p.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.6, label="Chance")
    ax_p.set_xlabel("Layer", fontsize=11)
    ax_p.set_ylabel("AUROC", fontsize=11)
    ax_p.set_title(
        f"Exp 7 — Contrastive Probe AUROC per Layer  (Δ-PKS feature)\n"
        f"{cfg.model_name}  |  {probe_dict['n_train_tokens']} train tokens  "
        f"/ {probe_dict['n_test_tokens']} test tokens",
        fontsize=11, fontweight="bold",
    )
    ax_p.legend(fontsize=9)
    ax_p.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    fig_p.savefig(out7 / "exp7_probe_auroc_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig_p)
    print("  Saved → exp7_probe_auroc_per_layer.png")

    probe_stats = pd.DataFrame({
        "layer":          layers_arr,
        "train_auroc":    train_auroc,
        "test_auroc":     test_auroc,
        "feature_weight": probe_dict["feature_weights"],
        "best_layer":     layers_arr == best_layer,
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
    for signal in ["delta_pks", "delta_ecs"]:
        top = (disc_df[(disc_df["signal"] == signal) & (disc_df["modification_type"] == "all")]
               .sort_values("auroc", ascending=False))
        if not top.empty:
            r = top.iloc[0]
            print(f"  {signal}: best disc layer {int(r['layer'])}  "
                  f"AUROC={r['auroc']:.4f}  Cohen's d={r['cohens_d']:.4f}")

    print(f"  Best probe layer  : {best_layer}  "
          f"train={train_auroc[best_layer]:.4f}  test={test_auroc[best_layer]:.4f}")
    high_risk = score_df[score_df["halluc_prob"] > 0.5]
    print(f"  High-risk tokens (prob > 0.5): {len(high_risk)} / {len(score_df)}")

    return {
        "disc_df":        disc_df,
        "probe_dict":     probe_dict,
        "score_df":       score_df,
        "train_results":  train_results,
        "test_results":   test_results,
        "n_pairs":        len(all_results),
        "best_layer":     best_layer,
    }
