"""
lookback_lens.py
================
Lookback Lens (Chuang et al., 2024, "Detecting and Mitigating Contextual
Hallucinations in Large Language Models Using Only Attention Maps") for the
AI-Scribe SOAP-note setting.

Idea
----
For each attention head (l, h) at a generated note-token position p, the
LOOKBACK RATIO is the average attention weight the head places on the
CONTEXT (the transcript) versus the tokens the model has already GENERATED:

    A_ctx = mean_{k in [0, T)}   attn[l,h,p,k]     # per-context-token avg
    A_new = mean_{k in [T, p)}   attn[l,h,p,k]     # per-generated-token avg
    LR[l,h,p] = A_ctx / (A_ctx + A_new)

A whole sentence is summarised by averaging LR over its tokens, giving a
feature VECTOR of dimension L*H (one ratio per head). A logistic-regression
probe on those vectors detects hallucinated sentences. Unlike ReDeEP ECS
(which pools RAW context attention over a static, weights-selected head set),
the probe learns WHICH heads' lookback behaviour is predictive -- so it does
not depend on the OV-eigenvalue copying-head mask that we verified is
causally inert (see select_copying_heads_ecs.py).

This reuses redeep_sentence.py for the model, tokenisation and sentence->token
span mapping, so the sentence-level AUROC is directly comparable to
redeep_sentence.py's fig3_auroc.

Key conventions (kept consistent with redeep_sentence.py, not the paper's
loose phrasing):
  * Context = [0, T) = transcript + prompt scaffold (everything before the
    note). Pass --context-transcript-only to exclude the ~24 scaffold tokens.
  * Query row = the note token's OWN position p (matches ReDeEP's
    attn[:, note_rows, :T]). "new" region = [T, p), i.e. previously generated
    note tokens, excluding self at p. First note token (empty new region) ->
    LR := 1.0.
  * Attention region scores are MEANS (per-token averages), per the paper --
    not summed mass. --region-sum switches to summed mass (ReDeEP-like) for
    an ablation.

Labels come from the span-judge CSVs (sentence_idx, sentence, label,
note_span). Two units (paper §2.2), selected with --unit:
  sentence : one unit per judged sentence, label 1 iff any judged row has
             label != "Faithful". Sliding-window labeling at sentence size;
             directly comparable to redeep_sentence.py's fig3_auroc.
  span     : the paper's PREDEFINED-SPAN setting — positives are the exact
             annotated hallucinated note_span substrings; negatives are clean
             spans (fully-Faithful sentences, or before-first/after-last
             segments with --clean-mode paper). Use this when the judge
             annotations are span-level.

Outputs (under --out)
---------------------
  features.npz          X (n_sent, L*H), y, groups (note id), head_index
  oof_predictions.csv   per-sentence out-of-fold probability + label + note
  cv_report.txt         per-fold and pooled AUROC, cluster-bootstrap CI,
                        comparison vs an aggregate context-reliance baseline
  head_coefficients.csv mean LR coefficient per (layer, head) across folds
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Reuse the existing, verified infrastructure.
from redeep_sentence import (
    load_model,
    tokenize_prompt_and_note,
    find_sentence_token_spans,
)
# Whitespace/case-tolerant substring locator (independent, cursor-free) — used
# to map annotated note_span strings to char ranges for the predefined-span unit.
from redeep_word_plots import find_span_char_range


def build_char_token_map(model, full_ids, transcript_len):
    """
    Build a cursor-free char->token mapping over the NOTE region.

    Returns (note_text, char_to_tok, n_search):
      note_text    reconstructed note string (special tokens stripped from tail)
      char_to_tok  fn: char index in note_text -> note-relative token index
      n_search     number of searchable (non-special) note tokens

    Same prefix-decode reconstruction as find_sentence_token_spans, but WITHOUT
    the monotonic search cursor, so arbitrary/overlapping spans (annotated
    hallucinated substrings) each locate independently.
    """
    tokenizer = model.tokenizer
    note_token_ids = full_ids[0, transcript_len:].tolist()
    special_ids = set(tokenizer.all_special_ids or [])
    n_search = len(note_token_ids)
    while n_search > 0 and note_token_ids[n_search - 1] in special_ids:
        n_search -= 1

    cumulative_len = []
    note_text = ""
    for i in range(n_search):
        note_text = tokenizer.decode(note_token_ids[: i + 1],
                                     skip_special_tokens=False)
        cumulative_len.append(len(note_text))

    def char_to_tok(char_pos: int) -> int:
        for i, clen in enumerate(cumulative_len):
            if clen > char_pos:
                return i
        return max(0, n_search - 1)

    return note_text, char_to_tok, n_search


def _charrange_to_tokspan(cs, ce, char_to_tok, n_search):
    """Half-open note-relative token span covering char range [cs, ce)."""
    a = char_to_tok(cs)
    b = char_to_tok(max(cs, ce - 1)) + 1
    a = max(0, min(a, n_search - 1))
    b = max(a + 1, min(b, n_search))
    return a, b


def note_span_units(df, note_text, char_to_tok, n_search, clean_mode: str):
    """
    Predefined-span units for ONE note (paper §2.2 setting 1).

    Positives  = the annotated hallucinated note_span substrings (label 1).
    Negatives  = clean spans:
      clean_mode="sentence": every fully-Faithful sentence (all its rows
                             Faithful) -> one clean span each. Better for long
                             SOAP notes; keeps each span entirely clean.
      clean_mode="paper":    the segment before the first and after the last
                             hallucinated span; or the whole note if the note
                             has no hallucinated span. (Text between spans is
                             discarded, exactly as in the paper.)

    Returns (tok_spans, labels, types, n_hallu_attempt, n_hallu_found).
    """
    tok_spans, labels, types = [], [], []
    n_hallu_attempt = n_hallu_found = 0
    hallu_ranges = []

    # --- positives: annotated hallucinated spans ---
    for _, row in df.iterrows():
        lab = str(row.get("label", "")).strip()
        span = str(row.get("note_span", "") or "").strip()
        if lab.lower() == "faithful" or not span:
            continue
        n_hallu_attempt += 1
        rng = find_span_char_range(note_text, span)
        if rng is None:
            continue
        n_hallu_found += 1
        cs, ce = rng
        hallu_ranges.append((cs, ce))
        a, b = _charrange_to_tokspan(cs, ce, char_to_tok, n_search)
        tok_spans.append((a, b)); labels.append(1); types.append(lab or "Hallucinated")

    # --- negatives: clean spans ---
    if clean_mode == "paper":
        if not hallu_ranges:
            clean_ranges = [(0, len(note_text))]
        else:
            first = min(cs for cs, _ in hallu_ranges)
            last = max(ce for _, ce in hallu_ranges)
            clean_ranges = []
            if first > 0:
                clean_ranges.append((0, first))
            if last < len(note_text):
                clean_ranges.append((last, len(note_text)))
        for cs, ce in clean_ranges:
            if ce <= cs:
                continue
            a, b = _charrange_to_tokspan(cs, ce, char_to_tok, n_search)
            tok_spans.append((a, b)); labels.append(0); types.append("Faithful")
    else:  # "sentence": fully-Faithful sentences as clean spans
        for _, grp in df.groupby("sentence_idx", sort=True):
            fully_faithful = (grp["label"].astype(str).str.strip().str.lower()
                              == "faithful").all()
            if not fully_faithful:
                continue
            sent = str(grp["sentence"].iloc[0])
            rng = find_span_char_range(note_text, sent)
            if rng is None:
                continue
            cs, ce = rng
            a, b = _charrange_to_tokspan(cs, ce, char_to_tok, n_search)
            tok_spans.append((a, b)); labels.append(0); types.append("Faithful")

    return tok_spans, labels, types, n_hallu_attempt, n_hallu_found


def window_units(hallu_typed, n_search: int, w: int):
    """
    Non-overlapping w-token chunks over the note (paper §2.2 sliding window).
    A chunk is positive iff it overlaps any hallucinated span token range.

    hallu_typed : list of ((a, b), creola_type) hallucinated span token ranges.
    Returns (tok_spans, labels, types) — a chunk's type is the CREOLA label of
    the first hallucinated token it covers, else "Faithful".
    """
    hallu_mask = np.zeros(n_search, dtype=bool)
    hallu_type = np.full(n_search, "Faithful", dtype=object)
    for (a, b), t in hallu_typed:
        a = max(0, a)
        b = min(n_search, b)
        if b > a:
            hallu_mask[a:b] = True
            hallu_type[a:b] = t
    tok_spans, labels, types = [], [], []
    for a in range(0, n_search, w):
        b = min(n_search, a + w)
        if b <= a:
            continue
        seg = hallu_mask[a:b]
        pos = bool(seg.any())
        tok_spans.append((a, b))
        labels.append(1 if pos else 0)
        types.append(str(hallu_type[a + int(np.argmax(seg))]) if pos else "Faithful")
    return tok_spans, labels, types


# ─────────────────────────────────────────────────────────────────────────────
# Label loading — one row-group per sentence from the span-judge CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_labeled_sentences(span_csv: Path) -> Tuple[List[str], np.ndarray]:
    """
    Read a span_judge CSV and collapse it to one entry per sentence_idx.

    Returns (sentences, labels) where labels[i] == 1 iff any judged span for
    that sentence carries a non-"Faithful" label.
    """
    df = pd.read_csv(span_csv)
    if df.empty or "sentence" not in df.columns:
        return [], np.array([], dtype=int)

    # Preserve first-seen order of sentences.
    sentences: List[str] = []
    labels: List[int] = []
    for _, grp in df.groupby("sentence_idx", sort=True):
        text = str(grp["sentence"].iloc[0])
        hallu = int((grp["label"].astype(str).str.strip().str.lower()
                     != "faithful").any())
        sentences.append(text)
        labels.append(hallu)
    return sentences, np.asarray(labels, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Lookback-ratio extraction — one memory-light forward pass per note
# ─────────────────────────────────────────────────────────────────────────────

def compute_lookback_ratios(
    model,
    full_ids: torch.Tensor,
    transcript_len: int,
    context_start: int = 0,
    region_sum: bool = False,
    include_self: bool = False,
) -> np.ndarray:
    """
    Single forward pass; return LR of shape (n_layers, n_heads, n_note).

    The attention hook reduces the (H, seq, seq) pattern to two small
    (H, n_note) tensors ON DEVICE (context score, new score) per layer, so we
    never hold the full attention tensor in Python or move it to CPU.

    context_start : first key index counted as "context" (>0 excludes the
                    prompt scaffold when --context-transcript-only is set).
    region_sum    : if True use summed attention mass per region instead of
                    the paper's per-token mean (ablation only).
    include_self  : if True the "new" (generated) region for note row r is
                    [T, T+r] INCLUSIVE, i.e. it contains the query token's own
                    self-attention. This matches the paper's off-by-one
                    convention (the paper attributes step t's lookback to the
                    attention row of y_{t-1}, whose region ends at itself). The
                    default (False) excludes self: region [T, T+r), which is
                    self-consistent with attributing the row to the note token
                    at its own position (aligns with redeep_sentence.py's
                    attn[:, note_rows, :T]). At sentence granularity the two
                    differ only in boundary tokens; see the paper-fidelity note.
    """
    device = full_ids.device
    seq = full_ids.shape[1]
    T = transcript_len
    n_note = seq - T
    if n_note <= 0:
        raise ValueError("note region is empty")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    ctx_score = torch.zeros((n_layers, n_heads, n_note), device=device)
    new_score = torch.zeros((n_layers, n_heads, n_note), device=device)

    # Lower-triangular mask + per-row token counts for the "new" region.
    # include_self=False -> cols c < r  (r previously-generated tokens)
    # include_self=True  -> cols c <= r (r+1 tokens, self included)
    tri_diag = 0 if include_self else -1
    tri = torch.tril(torch.ones((n_note, n_note), device=device),
                     diagonal=tri_diag)                        # (n_note, n_note)
    new_counts = tri.sum(-1)                                   # (n_note,) row token count

    def make_hook(l: int):
        def hook(value, hook):  # value: (batch, H, seq, seq)
            attn = value[0]  # (H, seq, seq)

            # Context region: rows = note positions, cols = [context_start, T).
            ctx_block = attn[:, T:, context_start:T].float()      # (H, n_note, T')
            if region_sum:
                ctx_score[l] = ctx_block.sum(-1)
            else:
                ctx_score[l] = ctx_block.mean(-1)

            # New/generated region: rows = note positions, cols = note positions.
            note_block = attn[:, T:, T:].float()                  # (H, n_note, n_note)
            new_sum = (note_block * tri).sum(-1)                  # (H, n_note)
            if region_sum:
                new_score[l] = new_sum
            else:
                new_score[l] = new_sum / new_counts.clamp(min=1.0)
            return value
        return hook

    fwd_hooks = [(f"blocks.{l}.attn.hook_pattern", make_hook(l))
                 for l in range(n_layers)]

    with torch.no_grad():
        model.run_with_hooks(full_ids, return_type=None, fwd_hooks=fwd_hooks)

    denom = ctx_score + new_score
    lr = torch.where(denom > 0, ctx_score / denom,
                     torch.full_like(denom, float("nan")))
    # Row 0 with include_self=False has an empty "new" region (no prior tokens)
    # -> all lookback is on context.
    if not include_self:
        lr[:, :, 0] = torch.where(ctx_score[:, :, 0] > 0,
                                  torch.ones_like(lr[:, :, 0]), lr[:, :, 0])
    return lr.cpu().numpy()  # (n_layers, n_heads, n_note)


def sentence_features(
    lr: np.ndarray,
    spans: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Average LR over each sentence's note-relative token span [a, b).

    lr    : (n_layers, n_heads, n_note)
    spans : note-relative [tok_start, tok_end) per sentence
    returns (n_sent, n_layers * n_heads)
    """
    n_layers, n_heads, n_note = lr.shape
    feats = np.full((len(spans), n_layers * n_heads), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(spans):
        a = max(0, a)
        b = min(n_note, b)
        if b <= a:
            continue
        with np.errstate(invalid="ignore"):
            v = np.nanmean(lr[:, :, a:b], axis=2)  # (n_layers, n_heads)
        feats[i] = v.reshape(-1)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Feature building over the whole labeled set
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    model,
    span_files: List[Path],
    gen_dir: Path,
    device: str,
    context_transcript_only: bool,
    region_sum: bool,
    include_self: bool,
    unit: str = "sentence",
    clean_mode: str = "sentence",
    window: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Returns (X, y, groups, types, (n_layers, n_heads)).

    unit="sentence": one row per judged sentence; label 1 if the sentence
                     overlaps any hallucinated span (paper §2.2 sliding-window
                     labeling, sentence-sized). Comparable to ReDeEP fig3.
    unit="span":     predefined-span setting (paper §2.2 setting 1) — positives
                     are the annotated hallucinated note_span substrings,
                     negatives are clean spans (see clean_mode). Use this when
                     the judge annotations are span-level.
    unit="window":   paper §2.2 setting 2 — non-overlapping fixed `window`-token
                     chunks, each labelled 1 iff it overlaps any hallucinated
                     span. The annotation-free segmentation used for guided
                     decoding; the honest sliding-window detector.

    groups[i] is an integer note id (rows from the same note share it) so a note
    never straddles a CV fold. types[i] is the CREOLA label for a positive unit
    ("Faithful" for negatives).
    """
    scaffold_len = 0
    if context_transcript_only:
        _, scaffold_len = tokenize_prompt_and_note(model, "", "x", device)
        print(f"  scaffold length (excluded from context): {scaffold_len} tokens")

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    g_parts: List[np.ndarray] = []
    t_parts: List[np.ndarray] = []
    shape: Optional[Tuple[int, int]] = None

    gen_cache: Dict[int, dict] = {}
    note_id = 0

    for span_csv in span_files:
        m = re.search(r"sample_(\d+)_note_(\d+)", span_csv.stem)
        if not m:
            print(f"  [skip] cannot parse sample/note from {span_csv.name}")
            continue
        si, k = int(m.group(1)), int(m.group(2))

        df = pd.read_csv(span_csv)
        if df.empty or "sentence" not in df.columns:
            continue

        if si not in gen_cache:
            gen_path = gen_dir / f"sample_{si:03d}_generations.json"
            if not gen_path.exists():
                print(f"  [skip] no generations for sample_{si:03d}")
                continue
            import json
            with open(gen_path) as f:
                gen_cache[si] = json.load(f)
        gen_data = gen_cache[si]

        transcript = gen_data["transcript"]
        notes = gen_data["notes"]
        if k >= len(notes):
            print(f"  [skip] sample_{si:03d} note {k} >= K={len(notes)}")
            continue
        note = notes[k]

        try:
            full_ids, T = tokenize_prompt_and_note(model, transcript, note, device)
        except Exception as exc:
            print(f"  [skip] sample_{si:03d}_note_{k:02d} tokenise: {exc}")
            continue

        # --- unit-specific span/label/type construction ---
        diag = ""
        if unit == "span":
            note_text, char_to_tok, n_search = build_char_token_map(
                model, full_ids, T)
            spans, lab_list, typ_list, n_att, n_found = note_span_units(
                df, note_text, char_to_tok, n_search, clean_mode)
            if not spans:
                continue
            lab = np.asarray(lab_list, dtype=int)
            typ = np.asarray(typ_list, dtype=object)
            diag = f"hallu_spans={n_found}/{n_att}"
        elif unit == "window":
            note_text, char_to_tok, n_search = build_char_token_map(
                model, full_ids, T)
            # Hallucinated span token ranges (positives from note_span_units).
            h_spans, h_labs, h_typs, n_att, n_found = note_span_units(
                df, note_text, char_to_tok, n_search, "sentence")
            hallu_typed = [(sp, t) for sp, l, t in zip(h_spans, h_labs, h_typs)
                           if l == 1]
            spans, lab_list, typ_list = window_units(hallu_typed, n_search, window)
            if not spans:
                continue
            lab = np.asarray(lab_list, dtype=int)
            typ = np.asarray(typ_list, dtype=object)
            diag = f"windows={len(spans)}, hallu_spans={n_found}/{n_att}"
        else:  # "sentence"
            sentences, lab = load_labeled_sentences(span_csv)
            if len(sentences) == 0:
                continue
            try:
                spans, n_fail, n_fuzzy = find_sentence_token_spans(
                    model, full_ids, T, sentences)
            except Exception as exc:
                print(f"  [skip] sample_{si:03d}_note_{k:02d} spans: {exc}")
                continue
            typ = np.where(lab == 1, "Hallucinated", "Faithful").astype(object)
            diag = f"span_fail={n_fail}, fuzzy={n_fuzzy}"

        context_start = scaffold_len if context_transcript_only else 0
        context_start = min(context_start, max(T - 1, 0))
        try:
            lr = compute_lookback_ratios(
                model, full_ids, T,
                context_start=context_start, region_sum=region_sum,
                include_self=include_self,
            )
        except Exception as exc:
            print(f"  [skip] sample_{si:03d}_note_{k:02d} forward: {exc}")
            continue

        feats = sentence_features(lr, spans)  # (n_units, L*H)

        # Drop units whose span was degenerate (all-NaN feature row).
        good = ~np.all(np.isnan(feats), axis=1)
        if not good.any():
            continue
        feats = feats[good]
        lab = lab[good]
        typ = typ[good]

        # Impute any residual per-head NaNs with 0.5 (neutral lookback).
        feats = np.where(np.isnan(feats), 0.5, feats)

        X_parts.append(feats)
        y_parts.append(lab)
        g_parts.append(np.full(len(lab), note_id, dtype=int))
        t_parts.append(typ)
        shape = (lr.shape[0], lr.shape[1])
        note_id += 1
        print(f"  sample_{si:03d}_note_{k:02d}: "
              f"{len(lab)} {unit}s, {int(lab.sum())} hallucinated  "
              f"(T={T}, {diag})")

    if not X_parts:
        raise RuntimeError("no usable notes — check --spans / --generations paths")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    types = np.concatenate(t_parts)
    return X, y, groups, types, shape


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated probe + evaluation
# ─────────────────────────────────────────────────────────────────────────────

def cluster_bootstrap_auroc(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray,
    n_boot: int = 2000, seed: int = 0,
) -> Tuple[float, float, float]:
    """AUROC point estimate + 95% CI, resampling whole notes (groups)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    point = roc_auc_score(y, p)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return point, lo, hi


def _fit_predict(X_tr, y_tr, X_te, standardize, class_weight):
    """
    One train/predict cycle inside a Pipeline so any standardisation is fit on
    train only. Returns (test_probabilities, train_probabilities, coef).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    # Paper: sklearn LogisticRegression defaults (L2, C=1.0), max_iter bumped.
    logreg = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, class_weight=class_weight,
    )
    if standardize:
        pipe = make_pipeline(StandardScaler(), logreg).fit(X_tr, y_tr)
    else:
        pipe = make_pipeline(logreg).fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]
    p_tr = pipe.predict_proba(X_tr)[:, 1]
    return p_te, p_tr, logreg.coef_.ravel()


def run_probe(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    shape: Tuple[int, int], out_dir: Path,
    n_splits: int = 5, seed: int = 0,
    standardize: bool = True, class_weight=None,
    types: Optional[np.ndarray] = None, unit: str = "sentence",
) -> None:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    n_layers, n_heads = shape
    oof = np.full(len(y), np.nan)
    coefs = []
    fold_test, fold_train = [], []   # per-fold Test / Train AUROC (paper §2.3)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X, y, groups)):
        p, p_tr, coef = _fit_predict(X[tr], y[tr], X[te], standardize, class_weight)
        oof[te] = p
        coefs.append(coef)
        a_te = roc_auc_score(y[te], p) if len(np.unique(y[te])) == 2 else float("nan")
        a_tr = roc_auc_score(y[tr], p_tr) if len(np.unique(y[tr])) == 2 else float("nan")
        fold_test.append(a_te)
        fold_train.append(a_tr)
        print(f"  fold {fold}: Train AUROC={a_tr:.3f}  Test AUROC={a_te:.3f}  "
              f"(n_train={len(tr)}, n_test={len(te)})")

    # Aggregate context-reliance baseline: single feature = mean LR over heads.
    base_feat = X.mean(axis=1, keepdims=True)
    base_oof = np.full(len(y), np.nan)
    for tr, te in skf.split(base_feat, y, groups):
        p, _, _ = _fit_predict(base_feat[tr], y[tr], base_feat[te],
                               standardize, class_weight)
        base_oof[te] = p

    valid = ~np.isnan(oof)
    point, lo, hi = cluster_bootstrap_auroc(y[valid], oof[valid], groups[valid])
    b_point, b_lo, b_hi = cluster_bootstrap_auroc(
        y[valid], base_oof[valid], groups[valid])

    # Persist.
    out_dir.mkdir(parents=True, exist_ok=True)
    head_idx = np.array([(l, h) for l in range(n_layers) for h in range(n_heads)])
    if types is None:
        types = np.where(y == 1, "Hallucinated", "Faithful").astype(object)
    np.savez(out_dir / "features.npz",
             X=X, y=y, groups=groups, head_index=head_idx, types=types)
    pd.DataFrame({
        "note_group": groups, "label": y, "type": types, "oof_prob": oof,
        "baseline_prob": base_oof,
    }).to_csv(out_dir / "oof_predictions.csv", index=False)

    # Signed mean coefficient (across folds) and its magnitude. NOTE polarity:
    # here label 1 == HALLUCINATED (opposite of the paper's 1 == factual), so a
    # POSITIVE coef means "higher lookback ratio -> more hallucinated" and a
    # NEGATIVE coef means "higher lookback -> more faithful" (the paper's
    # context-grounding heads). Magnitude ranking matches the paper's Table 7
    # "largest magnitude" head selection regardless of polarity.
    coef_mat = np.vstack(coefs)
    mean_coef = coef_mat.mean(axis=0)
    mean_abs_coef = np.abs(coef_mat).mean(axis=0)
    pd.DataFrame({
        "layer": head_idx[:, 0], "head": head_idx[:, 1],
        "mean_coef": mean_coef, "mean_abs_coef": mean_abs_coef,
    }).sort_values("mean_abs_coef", ascending=False).to_csv(
        out_dir / "head_coefficients.csv", index=False)
    mean_coef = mean_abs_coef  # keep downstream report (top heads by magnitude)

    report = [
        f"Lookback Lens — {unit}-level hallucination probe",
        "=" * 52,
        f"{unit}s={len(y)}  notes={len(np.unique(groups))}  "
        f"hallucinated={int(y.sum())} ({y.mean():.1%})",
        f"features = {n_layers}x{n_heads} = {X.shape[1]} lookback ratios",
        "",
        f"mean Train AUROC   = {np.nanmean(fold_train):.3f}   "
        f"(per fold: {', '.join(f'{a:.3f}' for a in fold_train)})",
        f"mean Test  AUROC   = {np.nanmean(fold_test):.3f}   "
        f"(per fold: {', '.join(f'{a:.3f}' for a in fold_test)})",
        f"pooled OOF AUROC   = {point:.3f}  (95% CI {lo:.3f}-{hi:.3f})",
        f"aggregate baseline = {b_point:.3f}  (95% CI {b_lo:.3f}-{b_hi:.3f})",
    ]

    # Per-CREOLA-type detectability: AUROC of each hallucination type's
    # positives vs ALL faithful negatives (holding negatives fixed so the
    # numbers are comparable across types).
    neg = valid & (y == 0)
    pos_types = sorted({t for t, yy in zip(types[valid], y[valid]) if yy == 1})
    if len(pos_types) > 1:
        report += ["", "Per-type AUROC (this type's positives vs all faithful):"]
        for t in pos_types:
            sel = neg | (valid & (y == 1) & (types == t))
            ys, ps = y[sel], oof[sel]
            n_t = int((valid & (y == 1) & (types == t)).sum())
            if len(np.unique(ys)) == 2:
                report.append(f"  {t:<14} AUROC={roc_auc_score(ys, ps):.3f}  (n={n_t})")
            else:
                report.append(f"  {t:<14} (n={n_t}, undefined)")

    report += ["", "Top-10 heads by |coef| (see head_coefficients.csv):"]
    top = (pd.DataFrame({"layer": head_idx[:, 0], "head": head_idx[:, 1],
                         "c": mean_coef})
           .sort_values("c", ascending=False).head(10))
    for _, r in top.iterrows():
        report.append(
            f"  L{int(r['layer']):02d}.H{int(r['head']):02d}  {r['c']:.3f}")
    text = "\n".join(report)
    (out_dir / "cv_report.txt").write_text(text + "\n")
    print("\n" + text)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spans", required=True,
                    help="dir of sample_*_note_*_span_judge.csv label files")
    ap.add_argument("--generations", required=True,
                    help="dir of sample_*_generations.json")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="luq_out/lookback_lens")
    ap.add_argument("--samples", type=int, default=None,
                    help="cap number of span CSVs (debug)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context-transcript-only", action="store_true",
                    help="exclude prompt-scaffold tokens from the context region")
    ap.add_argument("--region-sum", action="store_true",
                    help="use summed attention mass instead of per-token mean "
                         "(ReDeEP-like ablation)")
    ap.add_argument("--include-self-in-new", action="store_true",
                    help="include the query token's self-attention in the "
                         "'new' region (matches the paper's off-by-one "
                         "convention; default excludes self)")
    ap.add_argument("--no-standardize", action="store_true",
                    help="skip z-scoring features (exact paper setup: raw "
                         "lookback ratios into default LogisticRegression)")
    ap.add_argument("--balance", action="store_true",
                    help="class_weight='balanced' for imbalanced clinical data "
                         "(paper uses sklearn default, i.e. no weighting)")
    ap.add_argument("--unit", choices=["sentence", "span", "window"],
                    default="sentence",
                    help="'sentence': one unit per judged sentence, label from "
                         "span overlap (comparable to ReDeEP fig3). 'span': "
                         "paper's predefined-span setting — positives are the "
                         "annotated note_span substrings. 'window': paper's "
                         "sliding-window setting — non-overlapping --window-token "
                         "chunks labelled by span overlap.")
    ap.add_argument("--clean-mode", choices=["sentence", "paper"],
                    default="sentence",
                    help="[--unit span only] negatives from fully-Faithful "
                         "sentences ('sentence', better for long notes) or the "
                         "before-first/after-last segments ('paper').")
    ap.add_argument("--window", type=int, default=8,
                    help="[--unit window] chunk size in tokens (paper uses 8).")
    args = ap.parse_args()

    span_files = sorted(Path(args.spans).glob("sample_*_span_judge.csv"))
    if args.samples is not None:
        span_files = span_files[:args.samples]
    if not span_files:
        sys.exit(f"no span CSVs under {args.spans}")
    print(f"Loading {args.model} …")
    model = load_model(args.model, args.device)

    print(f"Building {args.unit}-level features from {len(span_files)} "
          f"labeled notes …")
    X, y, groups, types, shape = build_dataset(
        model, span_files, Path(args.generations), args.device,
        context_transcript_only=args.context_transcript_only,
        region_sum=args.region_sum,
        include_self=args.include_self_in_new,
        unit=args.unit, clean_mode=args.clean_mode, window=args.window,
    )
    print(f"\nDataset: X={X.shape}  hallucinated={int(y.sum())}/{len(y)}  "
          f"notes={len(np.unique(groups))}")
    run_probe(X, y, groups, shape, Path(args.out),
              n_splits=args.n_splits, seed=args.seed,
              standardize=not args.no_standardize,
              class_weight="balanced" if args.balance else None,
              types=types, unit=args.unit)


if __name__ == "__main__":
    main()
