"""
causal_intervention.py
=======================
Causal (interventional) test of which pathway -- external-context attention
(Copying Heads, the ECS mechanism) or parametric-knowledge injection
(Knowledge FFN, the PKS mechanism) -- is actually responsible for a
hallucinated word, as opposed to merely correlated with it.

ECS and PKS (redeep_sentence.py) are READ-ONLY diagnostics: they observe how
much a head/layer contributes, then correlate that with a hallucination
label. This script intervenes on the SAME hook points (hook_z for the
flagged copying heads, hook_mlp_out for the flagged knowledge-FFN layer) and
measures the effect on the model's own output, following the exploratory ->
confirmatory workflow recommended by Heimersheim & Nanda (2024, "How to use
and interpret activation patching"):

--mode ablation (Design 1, DEFAULT, runs first): Necessity test. Zero the
  flagged pathway at a single target word's own position and measure the
  shift in the model's confidence in the token it actually produced there.
  Cheap (no second run to source "clean" activations from -- every word is
  tested independently, hallucinated and faithful alike, no pairing needed)
  and a legitimate (if synthetic/off-distribution) exploratory screen per
  the paper's Section 2.1 -- run this first to see if there's any signal at
  all before paying for the more expensive Design 2 sweep.

--mode patch (Design 2, opt-in follow-up): genuine ACTIVATION PATCHING --
  real cached activations from one position are substituted into another
  position within the SAME forward pass, avoiding the off-distribution
  problem naive ablation/scaling has. Uses within-note (hallucinated word,
  faithful word) pairs (build_word_labels() already gives us both per note)
  so no cross-note token alignment is needed. Two directions (NOT symmetric
  -- paper Section 3): denoising (faithful -> hallucinated position,
  sufficiency) and noising (hallucinated -> faithful position, necessity),
  swept over an interpolation fraction (--interps) for a dose-response
  curve with REAL activations at both endpoints.

Metric (both modes): logit difference (Heimersheim & Nanda Section 4.1), not
raw log-probability. For each target position we fix, ONCE, from the
baseline (unintervened) run: the actual observed token and its runner-up
(2nd-highest logit) token. Logit diff = logit(actual) - logit(runner-up),
tracked across all intervened runs using those SAME two fixed token ids.
This avoids the saturation and unspecificity problems of raw logprob the
paper warns about.

Two pathways, both reusing existing infrastructure:
  * Attention / external-context pathway: hook_z (per-head output, pre-W_O)
    for the heads in copying_head_mask.npy, at every layer that has a
    flagged head. Causal analogue of ECS.
  * FFN / internal-knowledge pathway: hook_mlp_out (whole-layer FFN output,
    pre-residual-add) at the PKS-peak layer(s) passed via --pks-layers.
    Causal analogue of PKS.

Placebo control (mandatory, not optional, both modes): for each real
intervention we also run a SIZE-MATCHED intervention on randomly chosen
non-flagged heads/layer, so "any perturbation at this position looks like it
helps" can be ruled out rather than assumed away.

Target words: hallucinated/faithful words come from the LLM-judge span
labels via the same build_word_labels() logic as redeep_word_plots.py.

Requires the model (Llama 3.1 8B Instruct via TransformerLens) for the
actual forward passes -- target-word selection is cache-only and needs no
GPU.

Outputs
-------
  causal_out/word_targets.csv           -- selected hallucinated/faithful words (both modes)
  causal_out/ablation_results.csv       -- Design 1 output, one row per (word, pathway, real|placebo)
  causal_out/fig_ablation.png           -- Design 1 bar chart
  causal_out/intervention_results.csv   -- Design 2 output, one row per (pair, pathway, direction, interp, real|placebo)
  causal_out/fig_dose_response.png      -- Design 2 dose-response curves

Usage
-----
    python causal_intervention.py --samples 20                       # Design 1, ablation (default)
    python causal_intervention.py --mode patch --interps 0,0.5,1     # Design 2, patching follow-up
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redeep_sentence import load_model, tokenize_prompt_and_note
from redeep_word_plots import build_word_labels

DEFAULT_ACT_DIR   = Path("activations")
DEFAULT_SPAN_DIR  = Path("luq_out/llama_judge")
DEFAULT_GEN_DIR   = Path("luq_out/llama/generations")
DEFAULT_COPY_MASK = Path("copying_head_mask.npy")
DEFAULT_OUT_DIR   = Path("causal_out")


# ─────────────────────────────────────────────────────────────────────────────
# Target-word selection (cache-only, no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def find_span_files(span_dir: Path) -> Dict[Tuple[int, int], Path]:
    """
    Supports two layouts:
      * single-split: span_dir/spans/sample_*_span_judge.csv
        (pass e.g. --span-dir luq_out/llama_judge/aci/test2)
      * multi-split root: span_dir/{config}/{split}/spans/sample_*_span_judge.csv
        (pass --span-dir luq_out/llama_judge)

    IMPORTANT: every split numbers its samples from 0, so mixing the
    multi-split root with a single-split --act-dir silently matches the
    WRONG split's judge labels (first-glob-wins on the colliding
    (sample_idx, gen_idx) key). Always pass the single-split form when
    --act-dir is itself split-specific.
    """
    lookup = {}
    single_split = span_dir / "spans"
    patterns = (["spans/sample_*_span_judge.csv"] if single_split.is_dir()
               else ["*/*/spans/sample_*_span_judge.csv"])
    for pattern in patterns:
        for f in span_dir.glob(pattern):
            m = re.search(r"sample_(\d+)_note_(\d+)_span_judge", f.stem)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key in lookup and lookup[key] != f:
                    print(f"  [warn] duplicate span file for sample/gen {key}: "
                          f"keeping {lookup[key]}, ignoring {f}")
                    continue
                lookup[key] = f
    return lookup


def select_word_targets(
    act_dir: Path,
    span_dir: Path,
    max_words_per_note: int = 4,
    max_notes: Optional[int] = None,
    max_gen_idx: Optional[int] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For every cached (sample, gen) with a matching span-judge CSV, pick up to
    `max_words_per_note` hallucinated words and an equal number of faithful
    words (both sampled, not just the first N, to avoid a positional bias).

    `max_gen_idx` restricts to gen_idx < max_gen_idx (e.g. 5 -> first 5
    generations per sample, matching redeep_sentence.py's --notes N). This is
    a hard filter on which generations are eligible at all; `max_notes` caps
    the total number of (sample, gen) files considered afterward and is
    mainly a quick-smoke-test knob, not a per-sample cap.

    Returns a DataFrame: sample_idx, gen_idx, word_idx, word_text,
    word_tok_start, word_tok_end (NOTE-relative token span, matching
    word_spans in the cached npz -- the caller adds transcript_len), label
    (1=hallucinated, 0=faithful).
    """
    rng = np.random.default_rng(seed)
    span_lookup = find_span_files(span_dir)
    tok_files = sorted(act_dir.glob("sample_*_gen_*_tokens.npz"))
    if max_gen_idx is not None:
        tok_files = [tp for tp in tok_files
                    if int(re.search(r"_gen_(\d+)_tokens", tp.stem).group(1)) < max_gen_idx]
    if max_notes is not None:
        tok_files = tok_files[:max_notes]

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
        note_spans = [s for s in sdf.get("note_span", pd.Series([], dtype=str)).fillna("").tolist()
                      if str(s).strip()]
        labels, valid, _, _ = build_word_labels(word_strs, note_spans)

        n_words = len(word_spans)
        valid = valid[:n_words]
        content_idx = np.where(valid)[0]
        hallu_idx   = content_idx[labels[content_idx] == 1]
        faith_idx   = content_idx[labels[content_idx] == 0]
        if len(hallu_idx) == 0 or len(faith_idx) == 0:
            continue

        n_take = min(max_words_per_note, len(hallu_idx), len(faith_idx))
        hallu_pick = rng.choice(hallu_idx, size=n_take, replace=False)
        faith_pick = rng.choice(faith_idx, size=n_take, replace=False)

        for idx, label in [(hallu_pick, 1), (faith_pick, 0)]:
            for wi in idx:
                ws, we = word_spans[wi]
                rows.append({
                    "sample_idx": si, "gen_idx": k, "word_idx": int(wi),
                    "word_text": str(word_strs[wi]),
                    "word_tok_start": int(ws), "word_tok_end": int(we),
                    "label": label,
                })

    return pd.DataFrame(rows)


def pair_targets_within_note(targets: pd.DataFrame, max_pairs_per_note: int = 4,
                             seed: int = 0) -> pd.DataFrame:
    """
    Zips each note's hallucinated words with its faithful words into
    (hallu_row, faith_row) pairs -- the clean/corrupt pair for patching,
    entirely within one note so no cross-note token alignment is needed.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (si, k), group in targets.groupby(["sample_idx", "gen_idx"]):
        hallu = group[group["label"] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
        faith = group[group["label"] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
        n = min(len(hallu), len(faith), max_pairs_per_note)
        for i in range(n):
            rows.append({
                "sample_idx": si, "gen_idx": k, "pair_idx": i,
                "hallu_word_idx": hallu.loc[i, "word_idx"],
                "hallu_word_text": hallu.loc[i, "word_text"],
                "hallu_tok_end": hallu.loc[i, "word_tok_end"],
                "faith_word_idx": faith.loc[i, "word_idx"],
                "faith_word_text": faith.loc[i, "word_text"],
                "faith_tok_end": faith.loc[i, "word_tok_end"],
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Placebo (size-matched random) targets
# ─────────────────────────────────────────────────────────────────────────────

def random_placebo_heads(copying_mask: np.ndarray, n_layers: int, n_heads: int,
                         rng: np.random.Generator) -> Dict[int, List[int]]:
    """Size-matched random selection of NON-copying heads, same count and
    same per-layer distribution as the real copying-head mask, so the
    placebo intervention perturbs 'the same amount' of the network."""
    real_layers = {l: list(np.where(copying_mask[l])[0]) for l in range(n_layers)
                   if copying_mask[l].any()}
    non_copy = [(l, h) for l in range(n_layers) for h in range(n_heads) if not copying_mask[l, h]]
    placebo: Dict[int, List[int]] = {}
    for layer, heads in real_layers.items():
        n = len(heads)
        candidates = [h for (l, h) in non_copy if l == layer]
        if len(candidates) < n:
            continue
        chosen = rng.choice(candidates, size=n, replace=False)
        placebo[layer] = list(chosen)
    return placebo


def random_placebo_layer(real_layers: List[int], n_layers: int,
                         rng: np.random.Generator) -> List[int]:
    """Size-matched random non-flagged layer(s) for the FFN placebo."""
    candidates = [l for l in range(n_layers) if l not in real_layers]
    return list(rng.choice(candidates, size=len(real_layers), replace=False))


# ─────────────────────────────────────────────────────────────────────────────
# Design 1 -- ablation (necessity test, cheap, no caching needed)
# ─────────────────────────────────────────────────────────────────────────────

def make_ablation_hooks(layer_spec, pathway: str, position: int):
    """
    Zero out the flagged pathway's contribution at a single position. This
    is a synthetic, off-distribution intervention (the paper recommends
    real activation patching over ablation where possible -- see Design 2,
    make_patch_hooks below), but it's cheap: one forward pass, no second
    run needed to source a "clean" activation from. Used as the fast
    exploratory necessity screen that runs first.
    """
    hooks = []
    if pathway == "attn":
        for layer, heads in layer_spec.items():
            head_idx = torch.as_tensor(heads, dtype=torch.long)

            def fn(value, hook, head_idx=head_idx):
                value[0, position, head_idx, :] = 0.0
                return value

            hooks.append((f"blocks.{layer}.attn.hook_z", fn))
    else:  # ffn
        for layer in layer_spec:
            def fn(value, hook):
                value[0, position, :] = 0.0
                return value

            hooks.append((f"blocks.{layer}.hook_mlp_out", fn))
    return hooks


# ─────────────────────────────────────────────────────────────────────────────
# Design 2 -- real activation patching (sufficiency + necessity, opt-in follow-up)
# ─────────────────────────────────────────────────────────────────────────────

def cache_positionwise_activations(model, full_ids: torch.Tensor,
                                   attn_layers: List[int], mlp_layers: List[int]) -> dict:
    """
    One forward pass. Caches, for every position, the full hook_z (per-head
    output, pre-W_O) at each layer in `attn_layers`, and the full
    hook_mlp_out at each layer in `mlp_layers`. These caches are the source
    of BOTH the real-head/real-layer patch values and, since attn_layers /
    mlp_layers can be the real OR the placebo set, the placebo patch values
    too -- same cache, different layer/head selection at patch time.
    """
    cache = {"z": {}, "mlp": {}}

    def mk_z_hook(l):
        def fn(value, hook):
            cache["z"][l] = value[0].detach().clone()  # (pos, n_heads, d_head)
            return value
        return fn

    def mk_mlp_hook(l):
        def fn(value, hook):
            cache["mlp"][l] = value[0].detach().clone()  # (pos, d_model)
            return value
        return fn

    hooks = [(f"blocks.{l}.attn.hook_z", mk_z_hook(l)) for l in attn_layers]
    hooks += [(f"blocks.{l}.hook_mlp_out", mk_mlp_hook(l)) for l in mlp_layers]
    with torch.no_grad():
        model.run_with_hooks(full_ids, fwd_hooks=hooks, return_type=None)
    return cache


def make_patch_hooks(cache: dict, layer_spec, pathway: str,
                     src_pos: int, dst_pos: int, interp: float):
    """
    Soft-patch hook(s): at `dst_pos`, overwrite the activation with
    (1-interp)*original + interp*cached[src_pos]. interp=0 is a no-op
    (baseline); interp=1 is a full patch (Heimersheim & Nanda's standard
    denoising/noising); intermediate values interpolate between the two
    REAL activations rather than scaling a single run synthetically.
    """
    hooks = []
    if pathway == "attn":
        for layer, heads in layer_spec.items():
            head_idx = torch.as_tensor(heads, dtype=torch.long)
            src_val = cache["z"][layer][src_pos, head_idx, :]  # (n_heads_flagged, d_head)

            def fn(value, hook, head_idx=head_idx, src_val=src_val):
                orig = value[0, dst_pos, head_idx, :]
                value[0, dst_pos, head_idx, :] = (1.0 - interp) * orig + interp * src_val
                return value

            hooks.append((f"blocks.{layer}.attn.hook_z", fn))
    else:  # ffn
        for layer in layer_spec:
            src_val = cache["mlp"][layer][src_pos, :]  # (d_model,)

            def fn(value, hook, src_val=src_val):
                orig = value[0, dst_pos, :]
                value[0, dst_pos, :] = (1.0 - interp) * orig + interp * src_val
                return value

            hooks.append((f"blocks.{layer}.hook_mlp_out", fn))
    return hooks


# ─────────────────────────────────────────────────────────────────────────────
# Logit-difference metric (Heimersheim & Nanda Section 4.1)
# ─────────────────────────────────────────────────────────────────────────────

def get_baseline_pair(model, full_ids: torch.Tensor, position: int) -> Tuple[int, int, float]:
    """
    From the UNPATCHED forward pass: the actual observed token at `position`
    and its runner-up (2nd-highest logit) token, fixed once and reused
    across every patched run so the metric always compares the SAME two
    logits (not a freshly-recomputed runner-up each time, which would
    confound the comparison -- see paper Section 4).
    Returns (actual_token_id, runnerup_token_id, baseline_logit_diff).
    """
    with torch.no_grad():
        logits = model.run_with_hooks(full_ids, fwd_hooks=[], return_type="logits")
    pos_logits = logits[0, position - 1].float()
    actual_tok = int(full_ids[0, position])
    top2 = torch.topk(pos_logits, k=2).indices.tolist()
    runnerup_tok = top2[1] if top2[0] == actual_tok else top2[0]
    ld = float(pos_logits[actual_tok] - pos_logits[runnerup_tok])
    return actual_tok, runnerup_tok, ld


def logit_diff(model, full_ids: torch.Tensor, position: int,
               token_a: int, token_b: int, fwd_hooks: Optional[list] = None) -> float:
    """logit(token_a) - logit(token_b) at `position`, under the given hooks."""
    with torch.no_grad():
        logits = model.run_with_hooks(full_ids, fwd_hooks=fwd_hooks or [], return_type="logits")
    pos_logits = logits[0, position - 1].float()
    return float(pos_logits[token_a] - pos_logits[token_b])


# ─────────────────────────────────────────────────────────────────────────────
# Design 1 entry point -- ablation (runs first, default)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_test(args):
    """
    Necessity test. Every selected word (hallucinated AND faithful, as a
    within-note comparison group) is tested independently -- no pairing
    needed, since ablation only ever touches the word's own position. Far
    cheaper than Design 2: one baseline call + 4 condition calls per word
    (vs. up to 26 per pair in the patch sweep), and no positionwise
    activation cache to build first.
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = select_word_targets(
        Path(args.act_dir), Path(args.span_dir),
        max_words_per_note=args.max_words_per_note,
        max_notes=args.samples,
        max_gen_idx=args.max_gen_idx,
        seed=args.seed,
    )
    if targets.empty:
        print("No word targets found -- check --act-dir / --span-dir.")
        return
    targets.to_csv(out_dir / "word_targets.csv", index=False)
    print(f"Selected {len(targets)} word targets "
          f"({(targets['label'] == 1).sum()} hallucinated, "
          f"{(targets['label'] == 0).sum()} faithful) "
          f"across {targets[['sample_idx', 'gen_idx']].drop_duplicates().shape[0]} notes")

    model = load_model(args.model, args.device)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    copying_mask = np.load(args.copy_mask)
    real_attn_layers = {l: list(np.where(copying_mask[l])[0]) for l in range(n_layers)
                        if copying_mask[l].any()}
    pks_layers = [int(x) for x in args.pks_layers.split(",")]

    rng = np.random.default_rng(args.seed)
    placebo_attn_layers = random_placebo_heads(copying_mask, n_layers, n_heads, rng)
    placebo_ffn_layers  = random_placebo_layer(pks_layers, n_layers, rng)
    print(f"Real copying heads: {sum(len(v) for v in real_attn_layers.values())} "
          f"across {len(real_attn_layers)} layers; placebo matched.")
    print(f"Real knowledge-FFN layer(s): {pks_layers}; placebo layer(s): {placebo_ffn_layers}")

    conditions = [
        ("attn", "real",    real_attn_layers),
        ("attn", "placebo", placebo_attn_layers),
        ("ffn",  "real",    pks_layers),
        ("ffn",  "placebo", placebo_ffn_layers),
    ]

    gen_dir = Path(args.gen_dir)
    rows = []
    grouped = targets.groupby(["sample_idx", "gen_idx"])
    for (si, k), group in grouped:
        gen_path = gen_dir / f"sample_{si:03d}_generations.json"
        if not gen_path.exists():
            continue
        with open(gen_path) as f:
            gen_data = json.load(f)
        transcript = gen_data["transcript"]
        note = gen_data["notes"][k]

        try:
            full_ids, transcript_len = tokenize_prompt_and_note(model, transcript, note, args.device)
        except Exception as exc:
            print(f"  [skip] sample {si} gen {k}: tokenisation error: {exc}")
            continue
        seq_len = full_ids.shape[1]

        for r in group.itertuples():
            position = transcript_len + int(r.word_tok_end) - 1
            if not (0 <= position < seq_len):
                continue
            actual_tok, runnerup_tok, ld_base = get_baseline_pair(model, full_ids, position)

            for pathway, kind, layer_spec in conditions:
                hooks = make_ablation_hooks(layer_spec, pathway, position)
                ld = logit_diff(model, full_ids, position, actual_tok, runnerup_tok, fwd_hooks=hooks)
                rows.append({
                    "sample_idx": si, "gen_idx": k, "word_idx": r.word_idx,
                    "word_text": r.word_text, "label": r.label,
                    "pathway": pathway, "kind": kind,
                    "baseline_logit_diff": ld_base,
                    "ablated_logit_diff": ld,
                    "delta_logit_diff": ld - ld_base,
                })
        print(f"  sample {si} gen {k}: {len(group)} words x 4 conditions done")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "ablation_results.csv", index=False)
    print(f"\nSaved {len(results)} rows -> {out_dir / 'ablation_results.csv'}")

    plot_ablation(results, out_dir / "fig_ablation.png")


def plot_ablation(df: pd.DataFrame, out_path: Path):
    if df.empty:
        print("No results to plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    labels_map = {1: "Hallucinated", 0: "Faithful"}
    colors = {("real", 1): "tomato", ("real", 0): "seagreen",
             ("placebo", 1): "lightsalmon", ("placebo", 0): "lightgreen"}

    for ax, pathway, title in [(axes[0], "attn", "Attention pathway\n(copying heads)"),
                               (axes[1], "ffn", "FFN pathway\n(knowledge injection)")]:
        sub = df[df["pathway"] == pathway]
        bar_data, bar_labels, bar_colors, bar_err = [], [], [], []
        for kind in ["real", "placebo"]:
            for label in [1, 0]:
                s = sub[(sub["kind"] == kind) & (sub["label"] == label)]
                if s.empty:
                    continue
                bar_data.append(s["delta_logit_diff"].mean())
                bar_err.append(s["delta_logit_diff"].std() / np.sqrt(len(s)))
                bar_labels.append(f"{labels_map[label]}\n({kind}, n={len(s)})")
                bar_colors.append(colors[(kind, label)])
        x = np.arange(len(bar_data))
        ax.bar(x, bar_data, yerr=bar_err, color=bar_colors, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\Delta$ logit diff (ablated $-$ baseline)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Design 2 entry point -- real activation patching (opt-in follow-up)
# ─────────────────────────────────────────────────────────────────────────────

def run_patch_sweep(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = select_word_targets(
        Path(args.act_dir), Path(args.span_dir),
        max_words_per_note=args.max_words_per_note,
        max_notes=args.samples,
        max_gen_idx=args.max_gen_idx,
        seed=args.seed,
    )
    if targets.empty:
        print("No word targets found -- check --act-dir / --span-dir.")
        return
    targets.to_csv(out_dir / "word_targets.csv", index=False)

    pairs = pair_targets_within_note(targets, max_pairs_per_note=args.max_words_per_note,
                                     seed=args.seed)
    if pairs.empty:
        print("No (hallucinated, faithful) pairs found within any note -- "
              "every note needs at least one word of each label.")
        return
    print(f"Paired {len(pairs)} (hallucinated, faithful) word pairs across "
          f"{pairs[['sample_idx', 'gen_idx']].drop_duplicates().shape[0]} notes")

    model = load_model(args.model, args.device)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    copying_mask = np.load(args.copy_mask)
    real_attn_layers = {l: list(np.where(copying_mask[l])[0]) for l in range(n_layers)
                        if copying_mask[l].any()}
    pks_layers = [int(x) for x in args.pks_layers.split(",")]

    rng = np.random.default_rng(args.seed)
    placebo_attn_layers = random_placebo_heads(copying_mask, n_layers, n_heads, rng)
    placebo_ffn_layers  = random_placebo_layer(pks_layers, n_layers, rng)
    print(f"Real copying heads: {sum(len(v) for v in real_attn_layers.values())} "
          f"across {len(real_attn_layers)} layers; placebo matched.")
    print(f"Real knowledge-FFN layer(s): {pks_layers}; placebo layer(s): {placebo_ffn_layers}")

    # union of all layers we ever need to cache, per pathway
    all_attn_layers = sorted(set(real_attn_layers) | set(placebo_attn_layers))
    all_mlp_layers  = sorted(set(pks_layers) | set(placebo_ffn_layers))

    interps = [float(x) for x in args.interps.split(",")]
    gen_dir = Path(args.gen_dir)

    rows = []
    grouped = pairs.groupby(["sample_idx", "gen_idx"])
    for (si, k), group in grouped:
        gen_path = gen_dir / f"sample_{si:03d}_generations.json"
        if not gen_path.exists():
            continue
        with open(gen_path) as f:
            gen_data = json.load(f)
        transcript = gen_data["transcript"]
        note = gen_data["notes"][k]

        try:
            full_ids, transcript_len = tokenize_prompt_and_note(model, transcript, note, args.device)
        except Exception as exc:
            print(f"  [skip] sample {si} gen {k}: tokenisation error: {exc}")
            continue
        seq_len = full_ids.shape[1]

        cache = cache_positionwise_activations(model, full_ids, all_attn_layers, all_mlp_layers)

        for r in group.itertuples():
            # last token of each word, matching the earlier convention: for
            # multi-token words this is where the model's choice is most
            # "committed" (e.g. the final BPE piece of a number or name).
            hallu_pos = transcript_len + int(r.hallu_tok_end) - 1
            faith_pos = transcript_len + int(r.faith_tok_end) - 1
            if not (0 <= hallu_pos < seq_len and 0 <= faith_pos < seq_len):
                continue

            hallu_tok, hallu_runnerup, hallu_ld_base = get_baseline_pair(model, full_ids, hallu_pos)
            faith_tok, faith_runnerup, faith_ld_base = get_baseline_pair(model, full_ids, faith_pos)

            conditions = [
                ("attn", "real",    real_attn_layers),
                ("attn", "placebo", placebo_attn_layers),
                ("ffn",  "real",    pks_layers),
                ("ffn",  "placebo", placebo_ffn_layers),
            ]
            directions = [
                # (name, src_pos, dst_pos, token_a, token_b, baseline_ld)
                ("denoise", faith_pos, hallu_pos, hallu_tok, hallu_runnerup, hallu_ld_base),
                ("noise",   hallu_pos, faith_pos, faith_tok, faith_runnerup, faith_ld_base),
            ]

            for pathway, kind, layer_spec in conditions:
                for direction, src_pos, dst_pos, tok_a, tok_b, ld_base in directions:
                    for interp in interps:
                        hooks = make_patch_hooks(cache, layer_spec, pathway, src_pos, dst_pos, interp)
                        ld = logit_diff(model, full_ids, dst_pos, tok_a, tok_b, fwd_hooks=hooks)
                        rows.append({
                            "sample_idx": si, "gen_idx": k, "pair_idx": r.pair_idx,
                            "pathway": pathway, "kind": kind, "direction": direction,
                            "interp": interp,
                            "baseline_logit_diff": ld_base,
                            "patched_logit_diff": ld,
                            "delta_logit_diff": ld - ld_base,
                        })
        print(f"  sample {si} gen {k}: {len(group)} pairs x 4 conditions x 2 directions x "
              f"{len(interps)} interps done")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "intervention_results.csv", index=False)
    print(f"\nSaved {len(results)} rows -> {out_dir / 'intervention_results.csv'}")

    plot_dose_response(results, out_dir / "fig_dose_response.png")


def plot_dose_response(df: pd.DataFrame, out_path: Path):
    if df.empty:
        print("No results to plot.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    colors = {"real": "tomato", "placebo": "lightsalmon"}

    for row, direction in enumerate(["denoise", "noise"]):
        for col, pathway in enumerate(["attn", "ffn"]):
            ax = axes[row, col]
            sub = df[(df["direction"] == direction) & (df["pathway"] == pathway)]
            for kind in ["real", "placebo"]:
                s = sub[sub["kind"] == kind]
                if s.empty:
                    continue
                agg = s.groupby("interp")["delta_logit_diff"].agg(["mean", "std", "count"])
                agg["se"] = agg["std"] / np.sqrt(agg["count"])
                ax.errorbar(agg.index, agg["mean"], yerr=agg["se"],
                            color=colors[kind], marker="o", ms=4, capsize=3,
                            linestyle="-" if kind == "real" else "--",
                            label=kind)
            ax.axhline(0.0, color="grey", lw=0.8)
            pathway_name = "Attention (copying heads)" if pathway == "attn" else "FFN (knowledge injection)"
            direction_name = ("Denoise: faithful → hallucinated (sufficiency)" if direction == "denoise"
                              else "Noise: hallucinated → faithful (necessity)")
            ax.set_title(f"{direction_name}\n{pathway_name}", fontsize=10)
            ax.set_xlabel("Interpolation (0=baseline, 1=full patch)")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(r"$\Delta$ logit diff")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["ablation", "patch"], default="ablation",
                   help="'ablation' (default): Design 1, necessity test -- zero the "
                        "pathway at each word's own position. Cheap, no pairing or "
                        "activation caching needed; run this first. "
                        "'patch': Design 2, real clean/corrupt activation patching "
                        "(denoise + noise + interpolation) between within-note "
                        "hallucinated/faithful word pairs. More expensive; run as a "
                        "confirmatory follow-up once ablation shows a signal.")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--act-dir",  default=str(DEFAULT_ACT_DIR))
    p.add_argument("--span-dir", default=str(DEFAULT_SPAN_DIR))
    p.add_argument("--gen-dir",  default=str(DEFAULT_GEN_DIR))
    p.add_argument("--copy-mask", default=str(DEFAULT_COPY_MASK))
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--pks-layers", default="10",
                   help="Comma-separated knowledge-FFN layer(s) to patch (default: 10, "
                        "the peak PKS-discrimination layer from layer_word_metrics.csv).")
    p.add_argument("--interps", default="0,0.5,1",
                   help="Comma-separated interpolation fractions for the soft patch "
                        "(1-t)*original + t*cached. 0 = baseline, 1 = full denoise/noise patch.")
    p.add_argument("--max-words-per-note", type=int, default=4,
                   help="Hallucinated words sampled per note (faithful control matched 1:1, "
                        "also the cap on paired (hallu, faith) targets per note).")
    p.add_argument("--max-gen-idx", type=int, default=None,
                   help="Restrict to gen_idx < N (e.g. 5 = first 5 generations per sample, "
                        "matching redeep_sentence.py's --notes N). Default: no restriction.")
    p.add_argument("--samples", type=int, default=None,
                   help="Max number of (sample, gen) files to use after the --max-gen-idx "
                        "filter (default: all). A quick-smoke-test cap, not a per-sample limit.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "ablation":
        run_ablation_test(args)
    else:
        run_patch_sweep(args)


if __name__ == "__main__":
    main()
