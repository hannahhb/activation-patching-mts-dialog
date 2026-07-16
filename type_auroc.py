"""
type_auroc.py
=============
Per-hallucination-type detection AUROC from the mechanistic features, comparing
three feature sets on the SAME spans:

  attn      Lookback-ratio vector          (L*H  attention/context features)
  mlp       signed direct MLP contribution (L    per-layer Δ features)
  combined  concatenation                  (L*H + L)

For each target we train a grouped-CV logistic-regression probe and report the
pooled out-of-fold AUROC with a note-clustered bootstrap CI:

  any          any hallucination vs faithful (overall detectability)
  fabrication  \
  contextual    }  each CREOLA type vs faithful  (default --vs faithful)
  negation      |  or vs all other spans         (--vs rest)
  causality    /

The point is the CROSS-FEATURE comparison per type: e.g. fabrication detectable
from mlp but not attn (parametric injection), negation detectable from attn but
not mlp (misintegration of attended context). That table is the direct answer to
"where does each hallucination type originate in the attention-MLP
decomposition."

Both feature sets are extracted from the SAME tokenization and the SAME span
range [a, b) (reusing the two validated forward functions), so a span's attn and
mlp rows are aligned by construction.

Output (under --out)
--------------------
  type_auroc_long.csv   feature, target, n_pos, n_neg, auroc, ci_lo, ci_hi
  type_auroc_table.csv  target x feature -> AUROC (wide, the headline table)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from redeep_sentence import (
    load_model,
    tokenize_prompt_and_note,
    find_sentence_token_spans,
)
from lookback_lens import (
    build_char_token_map,
    note_span_units,
    load_labeled_sentences,
    compute_lookback_ratios,
    cluster_bootstrap_auroc,
    _fit_predict,
)
from mlp_contribution import compute_mlp_contributions

CREOLA = ["fabrication", "contextual", "negation", "causality"]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (attn + mlp, same spans, one loop)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    model, span_files, gen_dir: Path, device: str,
    unit: str, clean_mode: str, context_start_only: bool, include_self: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Returns (X_attn [n,L*H], X_mlp [n,L], labels [n], groups [n], (L,H))."""
    scaffold = 0
    if context_start_only:
        _, scaffold = tokenize_prompt_and_note(model, "", "x", device)

    gen_cache: Dict[int, dict] = {}
    Xa: List[np.ndarray] = []
    Xm: List[np.ndarray] = []
    labs: List[str] = []
    grp: List[int] = []
    shape: Optional[Tuple[int, int]] = None
    note_id = 0

    for span_csv in span_files:
        m = re.search(r"sample_(\d+)_note_(\d+)", span_csv.stem)
        if not m:
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
            with open(gen_path) as f:
                gen_cache[si] = json.load(f)
        gen_data = gen_cache[si]
        notes = gen_data["notes"]
        if k >= len(notes):
            continue
        transcript, note = gen_data["transcript"], notes[k]

        try:
            full_ids, T = tokenize_prompt_and_note(model, transcript, note, device)
        except Exception as exc:
            print(f"  [skip] sample_{si:03d}_note_{k:02d} tokenise: {exc}")
            continue

        if unit == "span":
            note_text, char_to_tok, n_search = build_char_token_map(model, full_ids, T)
            spans, _labs, typs, n_att, n_found = note_span_units(
                df, note_text, char_to_tok, n_search, clean_mode)
            diag = f"hallu_spans={n_found}/{n_att}"
        else:
            sentences, lab_arr = load_labeled_sentences(span_csv)
            if len(sentences) == 0:
                continue
            spans, n_fail, n_fuzzy = find_sentence_token_spans(
                model, full_ids, T, sentences)
            typs = ["Hallucinated" if v == 1 else "Faithful" for v in lab_arr]
            diag = f"span_fail={n_fail}"
        if not spans:
            continue

        cs = min(scaffold, max(T - 1, 0)) if context_start_only else 0
        try:
            lr = compute_lookback_ratios(
                model, full_ids, T, context_start=cs, include_self=include_self)
            delta = compute_mlp_contributions(model, full_ids, T)
        except Exception as exc:
            print(f"  [skip] sample_{si:03d}_note_{k:02d} forward: {exc}")
            continue

        n_note = lr.shape[2]
        for j, (a, b) in enumerate(spans):
            a = max(0, a)
            b = min(n_note, b)
            if b <= a:
                continue
            with np.errstate(invalid="ignore"):
                av = np.nanmean(lr[:, :, a:b], axis=2).reshape(-1)   # [L*H]
            if np.all(np.isnan(av)):
                continue
            av = np.where(np.isnan(av), 0.5, av)
            mv = delta[:, a:b].mean(axis=1)                          # [L]
            Xa.append(av.astype(np.float32))
            Xm.append(mv.astype(np.float32))
            labs.append(str(typs[j]).strip().lower())
            grp.append(note_id)
        shape = (lr.shape[0], lr.shape[1])
        note_id += 1
        print(f"  sample_{si:03d}_note_{k:02d}: {len(spans)} {unit}s  (T={T}, {diag})")

    if not Xa:
        raise RuntimeError("no usable spans — check --spans / --generations paths")
    return (np.vstack(Xa), np.vstack(Xm),
            np.asarray(labs, dtype=object), np.asarray(grp), shape)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped CV probe + per-type AUROC
# ─────────────────────────────────────────────────────────────────────────────

def grouped_oof(X, y, groups, n_splits, standardize, class_weight, seed):
    """Out-of-fold probabilities via note-grouped stratified CV, or None if the
    class/group counts are too small to split."""
    from sklearn.model_selection import StratifiedGroupKFold
    ng_pos = len(np.unique(groups[y == 1]))
    ng_neg = len(np.unique(groups[y == 0]))
    k = min(n_splits, ng_pos, ng_neg)
    if k < 2:
        return None
    oof = np.full(len(y), np.nan)
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    try:
        for tr, te in skf.split(X, y, groups):
            p, _ = _fit_predict(X[tr], y[tr], X[te], standardize, class_weight)
            oof[te] = p
    except ValueError:
        return None
    return oof


def per_type_auroc(feat_name, X, labels, groups, n_splits, seed,
                   standardize, class_weight, vs="faithful"):
    from sklearn.metrics import roc_auc_score
    faith = labels == "faithful"
    targets = ["any"] + [c for c in CREOLA if c in set(labels)]
    rows = []
    for tgt in targets:
        if tgt == "any":
            pos = labels != "faithful"
            neg = faith
        else:
            pos = labels == tgt
            neg = faith if vs == "faithful" else ~pos
        sel = pos | neg
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        if n_pos < 3 or n_neg < 3:
            rows.append((feat_name, tgt, n_pos, n_neg, np.nan, np.nan, np.nan))
            continue
        Xs, ys, gs = X[sel], pos[sel].astype(int), groups[sel]
        oof = grouped_oof(Xs, ys, gs, n_splits, standardize, class_weight, seed)
        if oof is None:
            rows.append((feat_name, tgt, n_pos, n_neg, np.nan, np.nan, np.nan))
            continue
        valid = ~np.isnan(oof)
        if len(np.unique(ys[valid])) < 2:
            rows.append((feat_name, tgt, n_pos, n_neg, np.nan, np.nan, np.nan))
            continue
        auroc, lo, hi = cluster_bootstrap_auroc(ys[valid], oof[valid], gs[valid])
        rows.append((feat_name, tgt, n_pos, n_neg, auroc, lo, hi))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="luq_out/type_auroc")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--unit", choices=["sentence", "span"], default="span")
    ap.add_argument("--clean-mode", choices=["sentence", "paper"], default="sentence")
    ap.add_argument("--context-transcript-only", action="store_true")
    ap.add_argument("--include-self-in-new", action="store_true")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vs", choices=["faithful", "rest"], default="faithful",
                    help="per-type negatives: faithful spans (default) or all "
                         "other spans")
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--no-balance", action="store_true",
                    help="disable class_weight='balanced' (default is balanced, "
                         "since CREOLA types are rare)")
    ap.add_argument("--features", default="mlp,attn,combined",
                    help="comma list from {mlp,attn,combined}")
    args = ap.parse_args()

    span_files = sorted(Path(args.spans).glob("sample_*_span_judge.csv"))
    if args.samples is not None:
        span_files = span_files[:args.samples]
    if not span_files:
        sys.exit(f"no span CSVs under {args.spans}")

    print(f"Loading {args.model} …")
    model = load_model(args.model, args.device)
    print(f"Extracting attn+mlp features from {len(span_files)} notes …")
    Xa, Xm, labels, groups, shape = build_features(
        model, span_files, Path(args.generations), args.device,
        unit=args.unit, clean_mode=args.clean_mode,
        context_start_only=args.context_transcript_only,
        include_self=args.include_self_in_new)

    feat_sets = {
        "attn": Xa,
        "mlp": Xm,
        "combined": np.hstack([Xa, Xm]),
    }
    standardize = not args.no_standardize
    class_weight = None if args.no_balance else "balanced"

    all_rows = []
    for name in [f.strip() for f in args.features.split(",") if f.strip()]:
        if name not in feat_sets:
            print(f"  [skip] unknown feature set '{name}'")
            continue
        all_rows += per_type_auroc(
            name, feat_sets[name], labels, groups,
            args.n_splits, args.seed, standardize, class_weight, vs=args.vs)

    long = pd.DataFrame(all_rows, columns=[
        "feature", "target", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_dir / "type_auroc_long.csv", index=False)
    table = long.pivot(index="target", columns="feature", values="auroc")
    order = [t for t in ["any"] + CREOLA if t in table.index]
    table = table.reindex(order)
    table.to_csv(out_dir / "type_auroc_table.csv")

    print(f"\nPer-type detection AUROC (vs {args.vs}, grouped CV, "
          f"note-clustered) — n={len(labels)} spans, "
          f"{len(np.unique(groups))} notes\n")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(table.to_string())
    print("\nPositives per target:")
    for t in order:
        n = int((labels == t).sum()) if t != "any" else int((labels != "faithful").sum())
        print(f"  {t:<12} n_pos={n}")
    print(f"\nFull detail with CIs: {out_dir/'type_auroc_long.csv'}")


if __name__ == "__main__":
    main()
