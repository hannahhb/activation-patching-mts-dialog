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
note_span); a sentence is hallucinated iff any of its judged rows has
label != "Faithful".

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Returns (X, y, groups, (n_layers, n_heads)).
    groups[i] is an integer note id (samples in the same note share it) so a
    note never straddles a CV fold.
    """
    # Scaffold length: token count of the prompt WITHOUT any transcript, so
    # context_start skips exactly the chat/instruction wrapper. Computed once
    # via the same longest-common-prefix trick tokenize_prompt_and_note uses.
    scaffold_len = 0
    if context_transcript_only:
        # Cheap probe: transcript_len for an empty transcript == scaffold size.
        _, scaffold_len = tokenize_prompt_and_note(model, "", "x", device)
        scaffold_len = max(scaffold_len - 0, 0)
        print(f"  scaffold length (excluded from context): {scaffold_len} tokens")

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    g_parts: List[np.ndarray] = []
    shape: Optional[Tuple[int, int]] = None

    gen_cache: Dict[int, dict] = {}
    note_id = 0

    for span_csv in span_files:
        m = re.search(r"sample_(\d+)_note_(\d+)", span_csv.stem)
        if not m:
            print(f"  [skip] cannot parse sample/note from {span_csv.name}")
            continue
        si, k = int(m.group(1)), int(m.group(2))

        sentences, labels = load_labeled_sentences(span_csv)
        if not sentences:
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

        try:
            spans, n_fail, n_fuzzy = find_sentence_token_spans(
                model, full_ids, T, sentences
            )
        except Exception as exc:
            print(f"  [skip] sample_{si:03d}_note_{k:02d} spans: {exc}")
            continue

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

        feats = sentence_features(lr, spans)  # (n_sent, L*H)

        # Drop sentences whose span was degenerate (all-NaN feature row).
        good = ~np.all(np.isnan(feats), axis=1)
        if not good.any():
            continue
        feats = feats[good]
        lab = labels[good]

        # Impute any residual per-head NaNs with 0.5 (neutral lookback).
        feats = np.where(np.isnan(feats), 0.5, feats)

        X_parts.append(feats)
        y_parts.append(lab)
        g_parts.append(np.full(len(lab), note_id, dtype=int))
        shape = (lr.shape[0], lr.shape[1])
        note_id += 1
        print(f"  sample_{si:03d}_note_{k:02d}: "
              f"{len(lab)} sents, {int(lab.sum())} hallucinated  "
              f"(T={T}, span_fail={n_fail}, fuzzy={n_fuzzy})")

    if not X_parts:
        raise RuntimeError("no usable notes — check --spans / --generations paths")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    return X, y, groups, shape


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
    train only. Returns (test_probabilities, coef_in_input_space).
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
    p = pipe.predict_proba(X_te)[:, 1]
    return p, logreg.coef_.ravel()


def run_probe(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    shape: Tuple[int, int], out_dir: Path,
    n_splits: int = 5, seed: int = 0,
    standardize: bool = True, class_weight=None,
) -> None:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    n_layers, n_heads = shape
    oof = np.full(len(y), np.nan)
    coefs = []
    fold_aurocs = []

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X, y, groups)):
        p, coef = _fit_predict(X[tr], y[tr], X[te], standardize, class_weight)
        oof[te] = p
        coefs.append(coef)
        if len(np.unique(y[te])) == 2:
            a = roc_auc_score(y[te], p)
            fold_aurocs.append(a)
            print(f"  fold {fold}: AUROC={a:.3f}  (n_test={len(te)})")

    # Aggregate context-reliance baseline: single feature = mean LR over heads.
    base_feat = X.mean(axis=1, keepdims=True)
    base_oof = np.full(len(y), np.nan)
    for tr, te in skf.split(base_feat, y, groups):
        p, _ = _fit_predict(base_feat[tr], y[tr], base_feat[te],
                            standardize, class_weight)
        base_oof[te] = p

    valid = ~np.isnan(oof)
    point, lo, hi = cluster_bootstrap_auroc(y[valid], oof[valid], groups[valid])
    b_point, b_lo, b_hi = cluster_bootstrap_auroc(
        y[valid], base_oof[valid], groups[valid])

    # Persist.
    out_dir.mkdir(parents=True, exist_ok=True)
    head_idx = np.array([(l, h) for l in range(n_layers) for h in range(n_heads)])
    np.savez(out_dir / "features.npz",
             X=X, y=y, groups=groups, head_index=head_idx)
    pd.DataFrame({
        "note_group": groups, "label": y, "oof_prob": oof,
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
        "Lookback Lens — sentence-level hallucination probe",
        "=" * 52,
        f"sentences={len(y)}  notes={len(np.unique(groups))}  "
        f"hallucinated={int(y.sum())} ({y.mean():.1%})",
        f"features = {n_layers}x{n_heads} = {X.shape[1]} lookback ratios",
        "",
        f"per-fold AUROC: "
        f"{', '.join(f'{a:.3f}' for a in fold_aurocs)}",
        f"pooled OOF AUROC   = {point:.3f}  (95% CI {lo:.3f}-{hi:.3f})",
        f"aggregate baseline = {b_point:.3f}  (95% CI {b_lo:.3f}-{b_hi:.3f})",
        "",
        "Top-10 heads by |coef| (see head_coefficients.csv):",
    ]
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
    args = ap.parse_args()

    span_files = sorted(Path(args.spans).glob("sample_*_span_judge.csv"))
    if args.samples is not None:
        span_files = span_files[:args.samples]
    if not span_files:
        sys.exit(f"no span CSVs under {args.spans}")
    print(f"Loading {args.model} …")
    model = load_model(args.model, args.device)

    print(f"Building features from {len(span_files)} labeled notes …")
    X, y, groups, shape = build_dataset(
        model, span_files, Path(args.generations), args.device,
        context_transcript_only=args.context_transcript_only,
        region_sum=args.region_sum,
        include_self=args.include_self_in_new,
    )
    print(f"\nDataset: X={X.shape}  hallucinated={int(y.sum())}/{len(y)}  "
          f"notes={len(np.unique(groups))}")
    run_probe(X, y, groups, shape, Path(args.out),
              n_splits=args.n_splits, seed=args.seed,
              standardize=not args.no_standardize,
              class_weight="balanced" if args.balance else None)


if __name__ == "__main__":
    main()
