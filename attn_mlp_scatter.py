"""
attn_mlp_scatter.py
==================
Headline figure: every annotated span as one point in the attention-vs-MLP
decomposition, colored by CREOLA hallucination type.

  x  A_S = context-attention magnitude  (mean Lookback ratio over all L*H heads)
  y  M_S = signed direct MLP contribution toward the generated span
           (sum over layers of Δ_MLP; see mlp_contribution.py)

For each span both scalars are computed from the SAME tokenization and the SAME
note-relative token range [a, b), reusing the two numerically-validated forward
passes (lookback_lens.compute_lookback_ratios, mlp_contribution.
compute_mlp_contributions). Faithful spans are plotted as a grey reference cloud
so hallucination-type clusters are read as DEVIATIONS from ordinary clinical
content, and quadrant lines are drawn at the faithful medians.

IMPORTANT caveat baked into the axis label: A_S is context-attention MAGNITUDE,
not grounding CORRECTNESS. Lookback measures context-vs-generated attention per
head; a contextual/negation error can attend strongly to the WRONG transcript
evidence and still land at high A_S. Distinguishing "attends a lot" from
"attends to the right thing" needs evidence localization (counterfactual
patching / ContextCite), not this axis. The optional --late-mlp switch reduces
M_S over the later half of layers only, where parametric-recall MLP signatures
tend to concentrate.

Outputs (under --out)
---------------------
  span_scalars.csv          note_id, span_id, label, span_length, A_S, M_S, M_S_late
  fig_attn_mlp_scatter.png  combined scatter (types + faithful cloud + centroids)
  fig_attn_mlp_panels.png   one panel per CREOLA type vs faithful background
  centroids.csv             per-type mean A_S / M_S / M_S_late and counts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
)
from mlp_contribution import compute_mlp_contributions

CREOLA = ["fabrication", "contextual", "negation", "causality"]
COLORS = {"fabrication": "#d62728", "contextual": "#1f77b4",
          "negation": "#2ca02c", "causality": "#9467bd",
          "hallucinated": "#d62728"}


def span_scalars(
    lr: np.ndarray, delta: np.ndarray,
    spans: List[Tuple[int, int]], late_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    lr    (L, H, n_note) lookback ratios ; delta (L, n_note) signed MLP Δ.
    Returns A_S, M_S, M_S_late — one scalar each per span.
      A_S      = mean lookback over heads/layers, averaged over span tokens
      M_S      = sum over layers of the span-mean Δ
      M_S_late = same, summed over the later `late_frac` of layers only
    """
    n_layers, n_note = delta.shape
    l0 = int(n_layers * (1.0 - late_frac))
    A = np.full(len(spans), np.nan, dtype=np.float32)
    M = np.full(len(spans), np.nan, dtype=np.float32)
    Ml = np.full(len(spans), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(spans):
        a = max(0, a)
        b = min(n_note, b)
        if b <= a:
            continue
        with np.errstate(invalid="ignore"):
            A[i] = np.nanmean(lr[:, :, a:b])
        d = delta[:, a:b].mean(axis=1)          # (L,)
        M[i] = float(d.sum())
        Ml[i] = float(d[l0:].sum())
    return A, M, Ml


def build_scatter_data(
    model, span_files, gen_dir: Path, device: str,
    unit: str, clean_mode: str, context_start_only: bool, include_self: bool,
) -> pd.DataFrame:
    scaffold = 0
    if context_start_only:
        _, scaffold = tokenize_prompt_and_note(model, "", "x", device)

    gen_cache: Dict[int, dict] = {}
    rows: List[dict] = []

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
            spans, labs, typs, n_att, n_found = note_span_units(
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

        A, M, Ml = span_scalars(lr, delta, spans)
        note_id = f"{si:03d}_{k:02d}"
        for j, (a, b) in enumerate(spans):
            if np.isnan(A[j]) or np.isnan(M[j]):
                continue
            rows.append({
                "note_id": note_id, "span_id": j,
                "label": str(typs[j]).strip().lower(),
                "span_length": int(b - a),
                "A_S": float(A[j]), "M_S": float(M[j]), "M_S_late": float(Ml[j]),
            })
        print(f"  sample_{si:03d}_note_{k:02d}: {len(spans)} {unit}s  (T={T}, {diag})")

    if not rows:
        raise RuntimeError("no usable spans — check --spans / --generations paths")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_dir: Path, y_col: str = "M_S") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "span_scalars.csv", index=False)

    faith = df[df["label"] == "faithful"]
    types = [c for c in CREOLA if c in set(df["label"])]
    if not types and "hallucinated" in set(df["label"]):
        types = ["hallucinated"]

    x_med = faith["A_S"].median() if not faith.empty else df["A_S"].median()
    y_med = faith[y_col].median() if not faith.empty else df[y_col].median()
    xlabel = "A_S  =  context-attention magnitude (mean lookback)\n" \
             "[magnitude, NOT grounding correctness]"
    ylabel = ("M_S  =  signed MLP contribution toward generated span"
              + ("  (late layers)" if y_col == "M_S_late" else "  (all layers)"))

    # Combined scatter.
    fig, ax = plt.subplots(figsize=(8, 6.5))
    if not faith.empty:
        ax.scatter(faith["A_S"], faith[y_col], s=10, c="#bbbbbb", alpha=0.35,
                   label=f"faithful (n={len(faith)})", zorder=1)
    cents = []
    for c in types:
        sub = df[df["label"] == c]
        ax.scatter(sub["A_S"], sub[y_col], s=26, c=COLORS.get(c, "#333"),
                   alpha=0.75, edgecolors="white", linewidths=0.3,
                   label=f"{c} (n={len(sub)})", zorder=3)
        cx, cy = sub["A_S"].mean(), sub[y_col].mean()
        cents.append({"label": c, "n": len(sub),
                      "A_S_mean": cx, "M_S_mean": sub["M_S"].mean(),
                      "M_S_late_mean": sub["M_S_late"].mean()})
        ax.scatter([cx], [cy], s=340, marker="X", c=COLORS.get(c, "#333"),
                   edgecolors="black", linewidths=1.3, zorder=5)
    ax.axvline(x_med, color="grey", ls="--", lw=0.9, zorder=0)
    ax.axhline(y_med, color="grey", ls="--", lw=0.9, zorder=0)
    if not faith.empty:
        cents.append({"label": "faithful", "n": len(faith),
                      "A_S_mean": faith["A_S"].mean(),
                      "M_S_mean": faith["M_S"].mean(),
                      "M_S_late_mean": faith["M_S_late"].mean()})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Where do hallucination types originate?\n"
                 "attention (context reliance) vs direct MLP contribution")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_attn_mlp_scatter.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(cents).to_csv(out_dir / "centroids.csv", index=False)

    # Per-type panels vs faithful background.
    if types:
        n = len(types)
        ncol = 2
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.6 * nrow),
                                 squeeze=False, sharex=True, sharey=True)
        for idx, c in enumerate(types):
            ax = axes[idx // ncol][idx % ncol]
            if not faith.empty:
                ax.scatter(faith["A_S"], faith[y_col], s=8, c="#cccccc",
                           alpha=0.4, zorder=1)
            sub = df[df["label"] == c]
            ax.scatter(sub["A_S"], sub[y_col], s=24, c=COLORS.get(c, "#333"),
                       alpha=0.8, edgecolors="white", linewidths=0.3, zorder=3)
            ax.axvline(x_med, color="grey", ls="--", lw=0.8)
            ax.axhline(y_med, color="grey", ls="--", lw=0.8)
            ax.set_title(f"{c}  (n={len(sub)})")
        for idx in range(n, nrow * ncol):
            axes[idx // ncol][idx % ncol].axis("off")
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_attn_mlp_panels.png", dpi=150)
        plt.close(fig)

    # Console summary.
    print(f"\nSpans: {len(df)}")
    print(f"{'type':<14}{'n':>5}{'A_S':>9}{'M_S':>9}{'M_S_late':>10}")
    for c in cents:
        print(f"{c['label']:<14}{c['n']:>5}{c['A_S_mean']:>9.3f}"
              f"{c['M_S_mean']:>9.3f}{c['M_S_late_mean']:>10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="luq_out/attn_mlp_scatter")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--unit", choices=["sentence", "span"], default="span")
    ap.add_argument("--clean-mode", choices=["sentence", "paper"], default="sentence")
    ap.add_argument("--context-transcript-only", action="store_true")
    ap.add_argument("--include-self-in-new", action="store_true")
    ap.add_argument("--late-mlp", action="store_true",
                    help="plot M_S over later-half layers only (parametric-recall band)")
    args = ap.parse_args()

    span_files = sorted(Path(args.spans).glob("sample_*_span_judge.csv"))
    if args.samples is not None:
        span_files = span_files[:args.samples]
    if not span_files:
        sys.exit(f"no span CSVs under {args.spans}")

    print(f"Loading {args.model} …")
    model = load_model(args.model, args.device)
    print(f"Computing A_S, M_S for {len(span_files)} notes …")
    df = build_scatter_data(
        model, span_files, Path(args.generations), args.device,
        unit=args.unit, clean_mode=args.clean_mode,
        context_start_only=args.context_transcript_only,
        include_self=args.include_self_in_new)
    plot(df, Path(args.out), y_col="M_S_late" if args.late_mlp else "M_S")


if __name__ == "__main__":
    main()
