"""
select_copying_heads_ecs.py
===========================
Re-select ReDeEP "Copying Heads" by their RUNTIME label-discriminability
instead of the static OV-eigenvalue proxy in redeep_sentence.py.

Why this exists
---------------
redeep_sentence.compute_copying_head_scores() scores each head by the
fraction of its OV-matrix eigenvalues with positive real part -- a
weights-only property that identifies heads *capable* of copying whatever
they attend to, but says nothing about whether a head actually attends to
and routes the TRANSCRIPT (the external context) at runtime. Verified
empirically: restricting ECS to those OV-copying heads reproduces the
all-heads ECS AUROC almost exactly (mean layer AUROC diff -0.003), and
ablating them is indistinguishable from ablating random heads -- i.e. the
OV mask does not isolate the causally-relevant heads.

This script instead ranks each (layer, head) by how well ITS OWN per-word
ECS separates hallucinated from faithful words (|Cohen's d|), pooled over
all notes, and keeps the top fraction. That directly selects the heads whose
external-context alignment predicts faithfulness -- the mechanism ECS
measures -- and is self-consistent with the ECS diagnostic and with how the
Knowledge-FFN / PKS side is already selected (by label correlation).

Inputs
------
  Per-note activation NPZs (from redeep_sentence.py, AFTER adding the
  ecs_word_head key -- rerun redeep_sentence.py once to populate it):
      <act-dir>/sample_NNN_gen_KK_tokens.npz
        ecs_word_head  (n_layers, n_heads, n_words)  float16 per-head word ECS
        word_strs, word_spans                        for label alignment
  Per-note span-judge CSVs (hallucination labels), same layout
  causal_intervention.py / redeep_word_plots.py use:
      <span-dir>/[config/split/]spans/sample_NNN_note_KK_span_judge.csv

Output
------
  <out>/copying_head_mask_ecs.npy   (n_layers, n_heads) bool -- the new mask
  <out>/copying_head_ecs_scores.csv  layer, head, cohens_d, |d|, n_hallu, n_faith, selected

Usage
-----
  python select_copying_heads_ecs.py \\
      --act-dir  redeep_out/virtscribe_test1/activations ... \\
      --span-dir luq_out/llama_judge/virtscribe/test1 ... \\
      --out      redeep_out/virtscribe_test1 \\
      --top-frac 0.15

  (pass multiple --act-dir / --span-dir pairs to pool across splits -- head
   identity is a model property, so more data gives a steadier ranking.)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from redeep_word_plots import build_word_labels  # per-word hallucination labels (no torch dep)


def find_span_files(span_dir: Path) -> Dict[Tuple[int, int], Path]:
    """Same single-split / multi-split resolution as causal_intervention.py:
    span_dir/spans/... (single) or span_dir/*/*/spans/... (multi-split root)."""
    lookup: Dict[Tuple[int, int], Path] = {}
    single = span_dir / "spans"
    patterns = (["spans/sample_*_span_judge.csv"] if single.is_dir()
                else ["*/*/spans/sample_*_span_judge.csv"])
    for pattern in patterns:
        for f in span_dir.glob(pattern):
            m = re.search(r"sample_(\d+)_note_(\d+)_span_judge", f.stem)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                lookup.setdefault(key, f)
    return lookup


def accumulate(act_dir: Path, span_dir: Path,
               acc: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """Fold one split's notes into the running per-(layer,head) accumulators.
    Returns (n_notes_used, n_words_labelled)."""
    span_lookup = find_span_files(span_dir)
    tok_files = sorted(act_dir.glob("sample_*_gen_*_tokens.npz"))
    n_notes = 0
    n_words = 0

    for tp in tok_files:
        m = re.search(r"sample_(\d+)_gen_(\d+)_tokens", tp.stem)
        if not m:
            continue
        si, k = int(m.group(1)), int(m.group(2))
        span_csv = span_lookup.get((si, k))
        if span_csv is None:
            continue

        d = np.load(str(tp), allow_pickle=True)
        if "ecs_word_head" not in d:
            print(f"  [skip] {tp.name}: no ecs_word_head key "
                  f"(rerun redeep_sentence.py to populate it)")
            continue
        ecs = d["ecs_word_head"].astype(np.float32)   # (L, H, W)
        word_strs = d["word_strs"]

        sdf = pd.read_csv(span_csv)
        note_spans = [s for s in sdf.get("note_span", pd.Series([], dtype=str)).fillna("").tolist()
                      if str(s).strip()]
        labels, valid, _, _, _ = build_word_labels(word_strs, note_spans)

        W = ecs.shape[2]
        labels = labels[:W]
        valid = valid[:W]
        if ecs.shape[2] == 0 or valid.sum() == 0:
            continue

        # Lazily size the accumulators to (L, H) on first note.
        if acc["hallu_sum"] is None:
            L, H = ecs.shape[0], ecs.shape[1]
            for key in ("hallu_sum", "hallu_sumsq", "hallu_n",
                        "faith_sum", "faith_sumsq", "faith_n"):
                acc[key] = np.zeros((L, H), dtype=np.float64)

        hallu_w = valid & (labels == 1)
        faith_w = valid & (labels == 0)

        for mask_w, s_key, sq_key, n_key in [
            (hallu_w, "hallu_sum", "hallu_sumsq", "hallu_n"),
            (faith_w, "faith_sum", "faith_sumsq", "faith_n"),
        ]:
            if not mask_w.any():
                continue
            sub = ecs[:, :, mask_w]                    # (L, H, w)
            finite = np.isfinite(sub)
            acc[s_key]  += np.nansum(sub, axis=2)
            acc[sq_key] += np.nansum(np.where(finite, sub * sub, 0.0), axis=2)
            acc[n_key]  += finite.sum(axis=2)

        n_notes += 1
        n_words += int(valid.sum())

    return n_notes, n_words


def cohens_d_from_acc(acc: Dict[str, np.ndarray], min_n: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-(layer, head) Cohen's d (hallu − faith), plus the two class counts.
    NaN where either class has < min_n finite samples for that head."""
    hn, fn = acc["hallu_n"], acc["faith_n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_h = acc["hallu_sum"] / hn
        mean_f = acc["faith_sum"] / fn
        var_h  = acc["hallu_sumsq"] / hn - mean_h ** 2
        var_f  = acc["faith_sumsq"] / fn - mean_f ** 2
        # ddof=1 correction: var above is population; scale to sample variance.
        var_h *= np.where(hn > 1, hn / (hn - 1), np.nan)
        var_f *= np.where(fn > 1, fn / (fn - 1), np.nan)
        pooled = np.sqrt(((hn - 1) * var_h + (fn - 1) * var_f) / (hn + fn - 2))
        d = (mean_h - mean_f) / (pooled + 1e-10)
    ok = (hn >= min_n) & (fn >= min_n) & np.isfinite(d)
    d = np.where(ok, d, np.nan)
    return d, hn, fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--act-dir", nargs="+", required=True,
                   help="One or more activation dirs of *_tokens.npz (paired positionally "
                        "with --span-dir).")
    p.add_argument("--span-dir", nargs="+", required=True,
                   help="One or more span-judge dirs (same count/order as --act-dir).")
    p.add_argument("--out", required=True,
                   help="Output dir for copying_head_mask_ecs.npy + scores CSV.")
    p.add_argument("--top-frac", type=float, default=0.15,
                   help="Fraction of (layer, head) cells to keep as Copying Heads, "
                        "ranked by |Cohen's d| (default 0.15, matching the OV selection).")
    p.add_argument("--min-n", type=int, default=20,
                   help="Min finite ECS samples per class for a head to be scorable "
                        "(default 20; unscorable heads never selected).")
    args = p.parse_args()

    if len(args.act_dir) != len(args.span_dir):
        raise SystemExit(f"--act-dir ({len(args.act_dir)}) and --span-dir "
                         f"({len(args.span_dir)}) must have the same count.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc: Dict[str, np.ndarray] = {k: None for k in
                                  ("hallu_sum", "hallu_sumsq", "hallu_n",
                                   "faith_sum", "faith_sumsq", "faith_n")}
    total_notes = total_words = 0
    for a, s in zip(args.act_dir, args.span_dir):
        n_notes, n_words = accumulate(Path(a), Path(s), acc)
        print(f"[{a}] pooled {n_notes} notes, {n_words} labelled words")
        total_notes += n_notes
        total_words += n_words

    if acc["hallu_sum"] is None or total_notes == 0:
        raise SystemExit("No notes with both ecs_word_head and span labels found. "
                         "Did you rerun redeep_sentence.py to save ecs_word_head?")

    d, hn, fn = cohens_d_from_acc(acc, args.min_n)
    L, H = d.shape
    abs_d = np.abs(d)

    # Rank scorable heads by |Cohen's d|, keep the top fraction of ALL cells
    # (so top-frac is directly comparable to the OV mask's top-frac).
    n_total = L * H
    n_keep = max(1, int(round(args.top_frac * n_total)))
    flat = abs_d.flatten()
    order = np.argsort(np.where(np.isfinite(flat), flat, -np.inf))[::-1]
    keep_flat = order[:n_keep]
    # Drop any NaN-scored cells that slipped in only because n_keep exceeded the
    # number of scorable heads.
    keep_flat = [i for i in keep_flat if np.isfinite(flat[i])]

    mask = np.zeros((L, H), dtype=bool)
    for i in keep_flat:
        mask[i // H, i % H] = True

    mask_path = out_dir / "copying_head_mask_ecs.npy"
    np.save(str(mask_path), mask)

    rows = []
    for l in range(L):
        for h in range(H):
            rows.append({
                "layer": l, "head": h,
                "cohens_d": round(float(d[l, h]), 4) if np.isfinite(d[l, h]) else np.nan,
                "abs_d":    round(float(abs_d[l, h]), 4) if np.isfinite(abs_d[l, h]) else np.nan,
                "n_hallu":  int(hn[l, h]), "n_faith": int(fn[l, h]),
                "selected": bool(mask[l, h]),
            })
    scores_csv = out_dir / "copying_head_ecs_scores.csv"
    pd.DataFrame(rows).to_csv(scores_csv, index=False)

    per_layer = mask.sum(axis=1)
    print(f"\nPooled {total_notes} notes, {total_words} labelled words.")
    print(f"Selected {int(mask.sum())}/{n_total} heads "
          f"({100*mask.sum()/n_total:.1f}%) by |Cohen's d| of per-head ECS.")
    print(f"  |d| range among selected: "
          f"{np.nanmin(abs_d[mask]):.3f} – {np.nanmax(abs_d[mask]):.3f}")
    print(f"  layers with >=1 selected head: {int((per_layer > 0).sum())}/{L}")
    print(f"  -> {mask_path}")
    print(f"  -> {scores_csv}")
    print(f"\nRerun the ablation with:  --copy-mask {mask_path}")


if __name__ == "__main__":
    main()
