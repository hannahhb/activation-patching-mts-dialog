"""
hallucination_emergence.py
============================
Figure 1 -- hallucination emergence trajectories, broken out by CREOLA
clinical error type (Fabrication / Negation / Contextual / Causality), vs.
a Faithful baseline.

For each sampled word:
  score_l(y_actual) - score_l(y_runnerup)   plotted against layer l

score_l(.) is LogitLens(resid_post at layer l) -- redeep_sentence._logit_lens,
already used in counterfactual_patch_demo.py for one hand-picked example.
This script generalises that to many words across the whole dataset, which
needs a different (y_H, y_F) choice: a hand-picked semantic antonym (like
" denies"/" reports" in the single-example script) does not scale -- there is
no general recipe for "the faithful alternative" to an arbitrary fabricated
vital sign or a wrong causal relation. The scalable proxy (same one
causal_intervention.py already uses for its ablation necessity test):

  y_H := the token the model ACTUALLY generated at that position
  y_F := its logit runner-up at that position (2nd-highest logit)

This is a *confidence-in-the-actual-choice* trajectory, not a strict
hallucinated-vs-faithful semantic contrast -- read it as "how much does the
model prefer what it ended up saying over its next-best alternative, and at
which layer does that preference lock in", separately for each error type.
For Faithful words the "actual choice" is (by construction, via the LLM
judge) the correct one, so that line is the "confidence in a correct
continuation" baseline the hallucination-type lines are compared against.

Category assignment reuses build_word_labels()'s note_category_sets
parameter (redeep_word_plots.py) which already supports per-word category
propagation from the span-judge CSV's "label" column (Fabrication/
Negation/Contextual/Causality/Faithful) -- existing callers
(causal_intervention.py, redeep_word_classifier.py) just never passed it,
they only used the collapsed binary hallucinated/faithful signal.

Answers: "when does the hallucination emerge, and does that differ by
clinical error type?"

Data scale (from luq_out/llama_judge/*/*/spans/*.csv, checked once):
  Faithful 6215, Fabrication 669, Causality 174, Contextual 129, Negation 33
Negation is thin (33 spans total across the whole dataset) -- expect a
noticeably wider CI band on that line than the others.

Runs across all 6 dataset splits by default (aci/test1..3,
virtscribe/test1..3), reading the per-split activation caches produced by
run_redeep_all_splits.sh and the per-split span-judge CSVs -- REQUIRED to
use the single-split path form for --span-dir per split (not the shared
multi-split root), for the same reason causal_intervention.py's
find_span_files() docstring warns about: every split numbers samples from 0.

Outputs (--out, default hallucination_emergence_out/)
--------------------------------------------------------
  raw_scores.csv              -- one row per (word, layer): sample_idx,
                                  gen_idx, word_idx, category, layer,
                                  score_diff (reusable later for Figure 3's
                                  per-hallucination mechanistic features)
  aggregated_trajectories.csv -- mean/std/count/ci95 per (category, layer)
  fig1_hallucination_emergence.png

Usage
-----
    python hallucination_emergence.py
    python hallucination_emergence.py --splits aci/test1,aci/test2 --max-words-per-category 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redeep_sentence import load_model, tokenize_prompt_and_note, _logit_lens
from causal_intervention import find_span_files
from redeep_word_plots import build_word_labels

DEFAULT_SPLITS = ["aci/test1", "aci/test2", "aci/test3",
                  "virtscribe/test1", "virtscribe/test2", "virtscribe/test3"]
DEFAULT_CATEGORIES = ["Fabrication", "Negation", "Contextual", "Causality"]

_COLORS = {"Fabrication": "tomato", "Negation": "steelblue",
          "Contextual": "seagreen", "Causality": "darkorange", "Faithful": "grey"}


# ─────────────────────────────────────────────────────────────────────────────
# Target selection (cache-only, no model needed) -- category-aware sibling
# of causal_intervention.select_word_targets(), which only kept the
# collapsed binary hallucinated/faithful label.
# ─────────────────────────────────────────────────────────────────────────────

def select_category_word_targets(
    act_dir: Path,
    span_dir: Path,
    categories: List[str],
    max_per_category: int = 3,
    max_notes: Optional[int] = None,
    max_gen_idx: Optional[int] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For every cached (sample, gen) with a matching span-judge CSV, sample up
    to `max_per_category` words per category (each of `categories`, plus a
    "Faithful" baseline) using the span CSV's own "label" column as the
    category -- not the binary hallucinated/faithful collapse.

    Returns: sample_idx, gen_idx, word_idx, word_text, word_tok_end
    (NOTE-relative, matching word_spans in the cached npz), category.
    """
    rng = np.random.default_rng(seed)
    span_lookup = find_span_files(span_dir)
    tok_files = sorted(act_dir.glob("sample_*_gen_*_tokens.npz"))
    if max_gen_idx is not None:
        tok_files = [tp for tp in tok_files
                    if int(re.search(r"_gen_(\d+)_tokens", tp.stem).group(1)) < max_gen_idx]
    if max_notes is not None:
        tok_files = tok_files[:max_notes]

    all_cats = list(categories) + ["Faithful"]
    rows = []
    for tp in tok_files:
        m = re.search(r"sample_(\d+)_gen_(\d+)_tokens", tp.stem)
        if not m:
            continue
        si, k = int(m.group(1)), int(m.group(2))
        span_csv = span_lookup.get((si, k))
        if span_csv is None:
            continue

        d = np.load(str(tp), allow_pickle=True)
        if not all(key in d for key in ("word_strs", "word_spans")):
            continue
        word_strs  = d["word_strs"]
        word_spans = d["word_spans"]

        sdf = pd.read_csv(span_csv)
        if "note_span" not in sdf.columns or "label" not in sdf.columns:
            continue
        # Keep note_span and its category ('label') aligned 1:1 -- the
        # previous (binary) version filtered note_spans independently,
        # which would desync from a category list filtered separately.
        mask = sdf["note_span"].fillna("").astype(str).str.strip().astype(bool)
        note_spans = sdf.loc[mask, "note_span"].astype(str).tolist()
        span_categories = sdf.loc[mask, "label"].astype(str).tolist()
        if not note_spans:
            continue

        labels, valid, n_spans, n_matched, cats = build_word_labels(
            word_strs, note_spans, note_category_sets={"category": span_categories})
        word_categories = cats["category"]

        n_words = len(word_spans)
        valid = valid[:n_words]
        word_categories = word_categories[:n_words]
        labels = labels[:n_words]

        for cat in all_cats:
            if cat == "Faithful":
                cand = np.where(valid & (labels == 0))[0]
            else:
                cand = np.where(valid & (word_categories == cat))[0]
            if len(cand) == 0:
                continue
            n_take = min(max_per_category, len(cand))
            pick = rng.choice(cand, size=n_take, replace=False)
            for wi in pick:
                _, we = word_spans[wi]
                rows.append({
                    "sample_idx": si, "gen_idx": k, "word_idx": int(wi),
                    "word_text": str(word_strs[wi]),
                    "word_tok_end": int(we),
                    "category": cat,
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-note forward pass + scoring -- ONE forward pass per note regardless of
# how many target words it contains.
# ─────────────────────────────────────────────────────────────────────────────

def score_note_targets(model, transcript: str, note: str, device: str,
                       n_layers: int, note_targets: pd.DataFrame) -> List[dict]:
    """
    note_targets: rows of the targets df for THIS (sample, gen) note only.
    For each target word, y_H = the actual generated token, y_F = its logit
    runner-up (same convention as causal_intervention.get_baseline_pair),
    read out via LogitLens at every layer's resid_post.
    """
    full_ids, transcript_len = tokenize_prompt_and_note(model, transcript, note, device)
    seq_len = full_ids.shape[1]

    positions, valid_rows = [], []
    for r in note_targets.itertuples():
        pos = transcript_len + int(r.word_tok_end) - 1
        if 0 < pos < seq_len:
            positions.append(pos)
            valid_rows.append(r)
    if not positions:
        return []

    pos_tensor = torch.as_tensor(positions, dtype=torch.long, device=device)
    src_positions = pos_tensor - 1  # resid/logits read from pos-1 to predict token AT pos

    resid_cache = {}
    def mk_hook(l):
        def fn(value, hook):
            resid_cache[l] = value[0, src_positions, :].detach().clone()  # (n_targets, d_model)
            return value
        return fn
    hooks = [(f"blocks.{l}.hook_resid_post", mk_hook(l)) for l in range(n_layers)]

    with torch.no_grad():
        logits = model.run_with_hooks(full_ids, fwd_hooks=hooks, return_type="logits")
    logits_at_src = logits[0, src_positions, :].float()  # (n_targets, d_vocab)

    rows = []
    for i, r in enumerate(valid_rows):
        pos_logits = logits_at_src[i]
        actual_tok = int(full_ids[0, positions[i]])
        top2 = torch.topk(pos_logits, k=2).indices.tolist()
        runnerup_tok = top2[1] if top2[0] == actual_tok else top2[0]
        for l in range(n_layers):
            logits_l = _logit_lens(model, resid_cache[l][i])
            rows.append({
                "sample_idx": r.sample_idx, "gen_idx": r.gen_idx, "word_idx": r.word_idx,
                "word_text": r.word_text, "category": r.category, "layer": l,
                "score_diff": float(logits_l[actual_tok] - logits_l[runnerup_tok]),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    p.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    p.add_argument("--max-words-per-category", type=int, default=3,
                   help="Per note, per category (default 3). Faithful gets the same cap.")
    p.add_argument("--max-gen-idx", type=int, default=5,
                   help="Restrict to gen_idx < N -- must be <= the N used when the "
                        "activation caches were built (run_redeep_all_splits.sh).")
    p.add_argument("--max-notes", type=int, default=None,
                   help="Cap on (sample,gen) files considered PER SPLIT after the "
                        "--max-gen-idx filter -- quick-smoke-test knob.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="hallucination_emergence_out")
    args = p.parse_args()

    categories = args.categories.split(",")
    splits = args.splits.split(",")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, args.device)
    n_layers = model.cfg.n_layers

    all_rows = []
    for config_split in splits:
        tag = config_split.replace("/", "_")
        act_dir = Path(f"redeep_out/{tag}/activations")
        span_dir = Path(f"luq_out/llama_judge/{config_split}")
        gen_dir = Path(f"luq_out/llama/generations/{config_split}")
        if not act_dir.is_dir():
            print(f"[skip] {config_split}: no {act_dir} "
                 f"(run run_redeep_all_splits.sh first)")
            continue

        targets = select_category_word_targets(
            act_dir, span_dir, categories,
            max_per_category=args.max_words_per_category,
            max_notes=args.max_notes, max_gen_idx=args.max_gen_idx, seed=args.seed)
        if targets.empty:
            print(f"[skip] {config_split}: no targets found")
            continue
        print(f"{config_split}: {len(targets)} targets "
             f"{targets['category'].value_counts().to_dict()}")

        for (si, k), group in targets.groupby(["sample_idx", "gen_idx"]):
            gen_path = gen_dir / f"sample_{si:03d}_generations.json"
            if not gen_path.exists():
                continue
            with open(gen_path) as f:
                gen_data = json.load(f)
            transcript = gen_data["transcript"]
            note = gen_data["notes"][k]
            try:
                rows = score_note_targets(model, transcript, note, args.device,
                                          n_layers, group)
            except Exception as exc:
                print(f"  [skip] {config_split} sample {si} gen {k}: {exc}")
                continue
            for row in rows:
                row["split"] = config_split
            all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    if raw_df.empty:
        print("No data collected -- check that run_redeep_all_splits.sh has been "
             "run for these splits, and that span-judge CSVs exist.")
        return
    raw_df.to_csv(out_dir / "raw_scores.csv", index=False)
    print(f"\nSaved {len(raw_df)} (word, layer) rows -> raw_scores.csv")

    n_words_per_cat = (raw_df.drop_duplicates(["split", "sample_idx", "gen_idx", "word_idx"])
                       .groupby("category").size())
    print("Distinct words per category:")
    print(n_words_per_cat.to_string())

    agg = (raw_df.groupby(["category", "layer"])["score_diff"]
          .agg(["mean", "std", "count"]).reset_index())
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg["ci95"] = 1.96 * agg["sem"]
    agg.to_csv(out_dir / "aggregated_trajectories.csv", index=False)
    print(f"Saved aggregated_trajectories.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in categories + ["Faithful"]:
        sub = agg[agg["category"] == cat].sort_values("layer")
        if sub.empty:
            continue
        n = int(n_words_per_cat.get(cat, 0))
        color = _COLORS.get(cat)
        ax.plot(sub["layer"], sub["mean"], marker="o", ms=3,
               label=f"{cat} (n={n})", color=color)
        ax.fill_between(sub["layer"], sub["mean"] - sub["ci95"], sub["mean"] + sub["ci95"],
                        alpha=0.15, color=color)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"LogitLens: score($y_{actual}$) $-$ score($y_{runner-up}$)")
    ax.set_title("Hallucination-preference emergence by clinical error type")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_hallucination_emergence.png", dpi=150)
    plt.close(fig)
    print(f"Saved fig1_hallucination_emergence.png -> {out_dir}/")


if __name__ == "__main__":
    main()
