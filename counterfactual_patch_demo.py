"""
counterfactual_patch_demo.py
=============================
Single-example proof of concept for MATCHED COUNTERFACTUAL activation
patching (as opposed to causal_intervention.py's WITHIN-NOTE word-pair
patching). This is the design where two separate TRANSCRIPTS, differing only
in the critical fact, are each forward-passed under the identical generation
prefix, and activations are transplanted from one transcript's run into the
other's to find which layer/component causally carries the source evidence
that determines the model's word choice at t*.

Worked example (real data, not synthetic)
------------------------------------------
sample_003 (aci/test1), note_00_gen_03. The transcript has:
    [doctor] and did you fall down or did you just kind of cut catch yourself
    [patient] no i fell
Ground truth: the patient DID fall. The model's note (llama, gen idx 3)
instead says:
    "...twisting her left ankle, which landed on the outside of her hip.
    She denies falling down but states she caught herself..."
i.e. a Negation-type hallucination (CREOLA taxonomy) confirmed by the
llama-judge span label at luq_out/llama_judge/aci/test1/spans/
sample_003_note_03_span_judge.csv (note_span="denies falling down",
label="Negation").

t* is fixed as the token immediately after "...outside of her hip. She",
where the model must choose between y_H=" denies" (what it actually said,
wrong) and y_F=" reports" (the faithful alternative). The SAME literal
prefix text is used in every condition below, so t* is always the LAST
token position of full_ids (no absolute-length alignment needed -- see
"Why position -1" below).

IMPORTANT correction to the naive recipe ("counterfactual transcript =
transcript edited to MATCH the hallucination")
------------------------------------------------------------------------
That framing is directionally wrong for a *restoration* experiment. If the
counterfactual transcript is edited so "denies" becomes the TRUE fact (i.e.
matches what the model said), then patching FROM that counterfactual INTO
the real run pushes the model FURTHER toward "denies" (more confident,
now-correct evidence for the same wrong-in-reality word) -- it does not
"restore" faithfulness in the real transcript's world. That is a real,
useful experiment (an evidence-sensitivity / manipulation check: does the
model's internal computation actually respond to this evidence at all,
regardless of which way it points?), but it is not the one described by
"Δ_restore > 0 shifts the model away from the hallucinated continuation and
towards the faithful one."

To get that experiment you need the standard causal-tracing direction
(Meng et al. 2022 ROME; Heimersheim & Nanda 2024 "denoising"): the CLEAN
source run must be one where the model actually lands on the FAITHFUL
continuation -- i.e. the TRUE fact stated more saliently/unambiguously than
in the real transcript, not a flipped fact. Patching clean -> corrupted then
tests whether stronger legibility of the (unchanged) true evidence is
sufficient to fix the hallucination.

This script implements BOTH, clearly separated, so the contrast between them
is itself a check that the patching machinery is wired correctly:

  --direction restore     (default) CLEAN source: real transcript + one
                           inserted doctor/patient confirmation exchange
                           re-affirming the SAME true fact ("yes, I fell
                           down") in unmistakable terms. Expect
                           M_clean < M_corrupted (baseline sanity check,
                           checked automatically before any patching), and
                           Δ_restore = M_corrupted - M_patched > 0 at the
                           layer(s) that causally carry the evidence.

  --direction sensitivity FLIP source: real transcript with the patient's
                           line negated to literally match the hallucination
                           ("no i did not fall, i caught myself before
                           hitting the ground" -- also lifts the fabricated
                           "caught herself" detail the model invented, so the
                           flipped world is now fully consistent with the
                           note). Expect M_flip > M_corrupted, and patching
                           flip -> corrupted should INCREASE M (push further
                           towards "denies"), i.e. Δ (defined the same way)
                           should come out NEGATIVE if that layer/component
                           really is an evidence-reading channel. A layer
                           that shows a large positive Δ under --restore and
                           a large negative Δ under --sensitivity is the
                           strongest evidence of a genuine external-evidence
                           channel (symmetric, bidirectional causal effect);
                           a layer that only moves under one direction is
                           more likely be a direction-specific artifact.

Why position -1 (no length alignment needed)
----------------------------------------------
The three transcripts (corrupted / clean / flip) differ ONLY inside the
`transcript` field, which is fully consumed before the fixed SOAP
template + style guidelines + assistant-header scaffold + generation
prefix. That entire suffix is byte-identical across conditions and begins
right after a "\n\n" boundary, so its tokenisation is context-independent
of what precedes it (verified at runtime by `assert_suffix_alignment`,
which hard-fails if the tokenised suffixes ever diverge). Consequently t*
is always the LAST token of `full_ids` in every condition -- source and
destination positions for patching are both simply -1.

IMPORTANT (fixed after the first run): matching the SUFFIX TEXT is not
enough on its own. If the transcripts differ in TOKEN COUNT, position -1
still lands at a DIFFERENT ABSOLUTE sequence index in each condition (the
transcript edit shifts everything downstream), and Llama's RoPE encodes
relative position -- so a patched-in residual carries a positional-context
shift on top of the intended fact content, confounding the two. The first
run of this script showed exactly this signature: both the "restore" and
"sensitivity" directions pushed the corrupted run the SAME way at deep
layers, which is what you'd expect from a generic positional artifact, not
a content-specific evidence channel (those should push opposite ways).
`equalize_lengths()` fixes this by padding every condition with a verified
single-token neutral filler (a doctor backchannel, " okay") until
`build_full_ids(...)` returns the EXACT same length for corrupted / clean /
flip -- enforced by a hard assertion in `main()` before anything else runs.
Only with equal total length does position -1 sit at the same absolute
index everywhere, isolating fact content as the only thing that differs.

Hierarchy (per Anthropic's circuit-tracing point that whole heads/layers
are often too coarse)
------------------------------------------------------------------------
  1. Full residual stream (hook_resid_pre) at every layer -- coarsest,
     cheapest, run first. Produces the Figure-1-style trajectory.
  2. At the layer with peak |Δ|, split into attention (hook_z, all heads)
     vs MLP (hook_mlp_out) -- which pathway carries the effect.
  3. Whichever of attn/mlp dominates at that layer, if attention: per-head
     hook_z patch, one head at a time -- which specific head(s).

Metric (Heimersheim & Nanda 2024, Section 4.1): logit difference, not raw
log-prob, using the SAME fixed token pair (y_H, y_F) throughout -- never a
freshly-recomputed runner-up.

Outputs (--out, default counterfactual_patch_demo_out/)
---------------------------------------------------------
  baseline_summary.txt        -- the sanity-gate numbers (M_corrupted,
                                  M_clean, M_flip) with a plain-English read
  resid_layer_sweep.csv        -- Δ per layer, both directions
  fig_resid_layer_sweep.png
  peak_layer_breakdown.csv     -- attn-all vs mlp vs per-head at peak layer
  fig_peak_layer_heads.png
  logit_lens_trajectory.csv    -- score_l(y_H) - score_l(y_F) on the
                                  corrupted run (when does the hallucination
                                  first emerge, independent of patching)

Usage
-----
    python counterfactual_patch_demo.py
    python counterfactual_patch_demo.py --direction sensitivity
    python counterfactual_patch_demo.py --direction both   # run both, one script
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redeep_sentence import load_model, _build_user_content, _logit_lens

_PROMPT_PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
_PROMPT_MIDDLE = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# ─────────────────────────────────────────────────────────────────────────────
# The one example (sample_003, aci/test1, note_00_gen_03 — real transcript
# and real hallucination, see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

_TRANSCRIPT_CORRUPTED = """\
[doctor] so stephanie morales is a 36 -year-old female today complaining of her ankle pain and she also has a history of diabetes and high blood pressure so stephanie tell me what's going on with your ankle
[patient] well we had a long spring and the other day we got some snow and ice i was walking to my car and i slipped and my left ankle kinda turned underneath me
[doctor] okay and so this happened couple days ago or how many days ago
[patient] two days ago
[doctor] two days ago okay and so it's your left ankle and it kinda just twisted underneath you on the kind of on the inside
[patient] yeah more on the it's more on the outside of my hips
[doctor] and did you fall down or did you just kind of cut catch yourself
[patient] no i fell
[doctor] okay and were you able to get up afterwards or somebody helped you up
[patient] i was but it was very sore and then started swelling and quite a bit of pain so it's hard to walk
[doctor] sure have you tried anything for pain or the swelling or anything like that
[patient] ibuprofen some ice and elevation
[doctor] okay has that helped much or not really
[patient] a little bit
"""

# CLEAN (restoration) source: same fact, restated more emphatically in the
# SAME line (not a new inserted exchange -- keeps the edit close in length to
# the original, minimising how much artificial padding equalize_lengths()
# below has to add).
_TRANSCRIPT_CLEAN = _TRANSCRIPT_CORRUPTED.replace(
    "[patient] no i fell\n",
    "[patient] no i fell, i definitely fell down\n",
)

# FLIP (sensitivity) source: the fact itself is negated to literally match
# what the model hallucinated, including the fabricated "caught herself"
# detail, so this world is fully self-consistent with the note's error.
_TRANSCRIPT_FLIP = _TRANSCRIPT_CORRUPTED.replace(
    "[patient] no i fell\n",
    "[patient] no i did not fall, i caught myself before hitting the ground\n",
)

# Fixed generation prefix — identical text appended after the prompt
# scaffold in every condition. Ends exactly where the model must choose
# between y_H (" denies") and y_F (" reports").
_GENERATION_PREFIX = (
    "Subjective:\n"
    "HPI: The patient, Stephanie Morales, is a 36-year-old female presenting "
    "with complaints of ankle pain. She reports that the pain started two "
    "days ago after slipping on ice and twisting her left ankle, which "
    "landed on the outside of her hip. She"
)

_Y_H_TEXT = " denies"   # what the model actually generated (hallucination)
_Y_F_TEXT = " reports"  # faithful alternative


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def build_full_ids(model, transcript: str, device: str) -> torch.Tensor:
    """Prompt scaffold + fixed generation prefix, NO trailing note/eot — the
    last token of the returned ids is where t* logits are read (position -1
    everywhere, never an existing token to predict "at", unlike
    redeep_sentence.tokenize_prompt_and_note which handles a COMPLETE note)."""
    user_content = _build_user_content(transcript)
    full_text = f"{_PROMPT_PREFIX}{user_content}{_PROMPT_MIDDLE}{_GENERATION_PREFIX}"
    return model.tokenizer.encode(
        full_text, return_tensors="pt", add_special_tokens=False
    ).to(device)


def assert_suffix_alignment(model, ids_a: torch.Tensor, ids_b: torch.Tensor,
                            n_check: int = 15, label: str = "") -> None:
    """Hard-fail if the last n_check tokens differ between two conditions —
    the whole position=-1 patching scheme is invalid if they do. See
    module docstring 'Why position -1'."""
    a = ids_a[0, -n_check:]
    b = ids_b[0, -n_check:]
    if a.shape[0] != b.shape[0] or not torch.equal(a, b):
        da = model.tokenizer.decode(a)
        db = model.tokenizer.decode(b)
        raise AssertionError(
            f"Suffix tokenisation diverged for {label}! This breaks the "
            f"position=-1 alignment the whole script depends on.\n"
            f"  a: {da!r}\n  b: {db!r}"
        )


# Neutral filler used ONLY to equalise total token length across conditions
# (see equalize_lengths). Deliberately a generic doctor backchannel that is
# extremely common and near content-free in real clinical dialogue, so
# padding with it doesn't introduce new evidence.
_FILLER_UNIT = " okay"


def equalize_lengths(model, device: str, transcripts: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """
    LENGTH-CONFOUND FIX. Position -1 being the SAME relative offset in every
    condition (guaranteed by assert_suffix_alignment) is not enough: if the
    transcripts differ in token count, position -1 sits at a DIFFERENT
    absolute sequence index in each condition, so RoPE gives the model a
    different relative-distance profile to every earlier token regardless of
    what those tokens say. A patched-in residual then carries that positional
    shift as well as the intended fact content, confounding the two.

    Fix: splice raw pad TOKEN IDS (a single verified-one-token filler,
    _FILLER_UNIT) directly into the already-tokenised id sequence, right
    before the fixed suffix (_PROMPT_MIDDLE + _GENERATION_PREFIX), so every
    condition reaches EXACTLY the same total length.

    NOT implemented as repeated filler TEXT re-tokenised from scratch: a
    first attempt did that and failed at runtime (e.g. requested gap=10,
    got 1012 tokens instead of 1008) because BPE can merge a run of the same
    repeated word differently than an isolated occurrence -- "N repeats" and
    "N tokens" are not the same guarantee once retokenised in context.
    Splicing already-tokenised IDs sidesteps this entirely: nothing gets
    retokenised at the padded region, so `gap` requested == `gap` added,
    always, by construction.
    """
    suffix_ids = model.tokenizer.encode(_PROMPT_MIDDLE + _GENERATION_PREFIX,
                                        add_special_tokens=False)
    suffix_len = len(suffix_ids)
    suffix_tensor = torch.tensor(suffix_ids)

    pad_ids = model.tokenizer.encode(_FILLER_UNIT, add_special_tokens=False)
    assert len(pad_ids) == 1, (
        f"_FILLER_UNIT {_FILLER_UNIT!r} is {len(pad_ids)} tokens, not 1 -- "
        f"pick a different filler word before relying on equalize_lengths."
    )
    pad_id = pad_ids[0]

    ids = {k: build_full_ids(model, t, device) for k, t in transcripts.items()}
    for k, x in ids.items():
        tail = x[0, -suffix_len:].cpu()
        if not torch.equal(tail, suffix_tensor):
            raise AssertionError(
                f"[{k}] the fixed suffix did not tokenise identically in "
                f"context (got {model.tokenizer.decode(tail)!r}, expected "
                f"{model.tokenizer.decode(suffix_tensor)!r}) -- cannot "
                f"locate a safe splice point before it."
            )

    lengths = {k: x.shape[1] for k, x in ids.items()}
    target = max(lengths.values())
    print(f"equalize_lengths: raw token counts {lengths}, padding all to {target}")

    out = {}
    for k, x in ids.items():
        gap = target - lengths[k]
        if gap == 0:
            out[k] = x
            continue
        split = x.shape[1] - suffix_len
        pad_block = torch.full((1, gap), pad_id, dtype=x.dtype, device=x.device)
        padded = torch.cat([x[:, :split], pad_block, x[:, split:]], dim=1)
        assert padded.shape[1] == target  # exact by construction; cheap defense-in-depth
        out[k] = padded
    return out


def single_token_id(model, text: str) -> int:
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        print(f"  [warn] {text!r} is {len(ids)} tokens {ids}, using the first "
              f"({model.tokenizer.decode([ids[0]])!r}) only.")
    return ids[0]


# ─────────────────────────────────────────────────────────────────────────────
# Baseline + caching
# ─────────────────────────────────────────────────────────────────────────────

def logit_diff_at_last_pos(model, full_ids: torch.Tensor, tok_h: int, tok_f: int,
                           fwd_hooks=None) -> float:
    with torch.no_grad():
        logits = model.run_with_hooks(full_ids, fwd_hooks=fwd_hooks or [],
                                      return_type="logits")
    pos_logits = logits[0, -1].float()
    return float(pos_logits[tok_h] - pos_logits[tok_f])


def cache_last_position(model, full_ids: torch.Tensor, n_layers: int) -> dict:
    """One forward pass; cache hook_resid_pre / attn.hook_z / hook_mlp_out at
    the LAST position only, for every layer."""
    names_filter = lambda name: (
        name.endswith("hook_resid_pre")
        or name.endswith("attn.hook_z")
        or name.endswith("hook_mlp_out")
    )
    with torch.no_grad():
        _, cache = model.run_with_cache(full_ids, names_filter=names_filter,
                                        return_type=None)
    out = {"resid_pre": {}, "z": {}, "mlp": {}}
    for l in range(n_layers):
        out["resid_pre"][l] = cache[f"blocks.{l}.hook_resid_pre"][0, -1, :].detach().clone()
        out["z"][l] = cache[f"blocks.{l}.attn.hook_z"][0, -1, :, :].detach().clone()
        out["mlp"][l] = cache[f"blocks.{l}.hook_mlp_out"][0, -1, :].detach().clone()
    return out


def logit_lens_trajectory(model, full_ids: torch.Tensor, n_layers: int,
                          tok_h: int, tok_f: int) -> pd.DataFrame:
    """score_l(y_H) - score_l(y_F) via LogitLens on resid_post at every layer,
    on the CORRUPTED (unpatched) run — Figure-1-style 'when does the
    hallucination-preference first emerge', reusing redeep_sentence._logit_lens."""
    names_filter = lambda name: name.endswith("hook_resid_post")
    with torch.no_grad():
        _, cache = model.run_with_cache(full_ids, names_filter=names_filter,
                                        return_type=None)
    rows = []
    for l in range(n_layers):
        resid = cache[f"blocks.{l}.hook_resid_post"][0, -1, :]
        logits_l = _logit_lens(model, resid)
        rows.append({
            "layer": l,
            "score_diff": float(logits_l[tok_h] - logits_l[tok_f]),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_resid_pre(src_cache: dict, layer: int):
    def fn(value, hook):
        value[0, -1, :] = src_cache["resid_pre"][layer]
        return value
    return [(f"blocks.{layer}.hook_resid_pre", fn)]


def patch_mlp(src_cache: dict, layer: int):
    def fn(value, hook):
        value[0, -1, :] = src_cache["mlp"][layer]
        return value
    return [(f"blocks.{layer}.hook_mlp_out", fn)]


def patch_attn_all(src_cache: dict, layer: int):
    def fn(value, hook):
        value[0, -1, :, :] = src_cache["z"][layer]
        return value
    return [(f"blocks.{layer}.attn.hook_z", fn)]


def patch_attn_head(src_cache: dict, layer: int, head: int):
    def fn(value, hook):
        value[0, -1, head, :] = src_cache["z"][layer][head, :]
        return value
    return [(f"blocks.{layer}.attn.hook_z", fn)]


def run_direction(model, direction: str, ids_corrupted: torch.Tensor,
                  ids_source: torch.Tensor, tok_h: int, tok_f: int,
                  n_layers: int, n_heads: int, out_dir: Path,
                  m_corrupted: float) -> list:
    tag = "restore" if direction == "clean" else "sensitivity"
    print(f"\n=== Direction: {tag} (source={'clean' if direction == 'clean' else 'flip'}) ===")

    m_source = logit_diff_at_last_pos(model, ids_source, tok_h, tok_f)
    expect = "< M_corrupted" if direction == "clean" else "> M_corrupted"
    print(f"  M_corrupted (baseline, no patch) = {m_corrupted:+.3f}")
    print(f"  M_source ({tag})                 = {m_source:+.3f}  (expected {expect})")
    ok = (m_source < m_corrupted) if direction == "clean" else (m_source > m_corrupted)
    print(f"  Sanity check {'PASSED' if ok else '[!!] FAILED — reconsider the transcript edit'}")
    summary_lines = [
        f"Direction {tag}: M_source = {m_source:+.4f} (expected {expect}, "
        f"M_corrupted = {m_corrupted:+.4f}) -- sanity check "
        f"{'PASSED' if ok else '[!!] FAILED — reconsider the transcript edit'}",
    ]

    src_cache = cache_last_position(model, ids_source, n_layers)

    # 1) Residual-stream layer sweep
    rows = []
    for l in range(n_layers):
        m_patched = logit_diff_at_last_pos(model, ids_corrupted, tok_h, tok_f,
                                           fwd_hooks=patch_resid_pre(src_cache, l))
        rows.append({
            "layer": l,
            "granularity": "resid_pre",
            "m_corrupted": m_corrupted,
            "m_patched": m_patched,
            "delta_restore": m_corrupted - m_patched,
        })
    resid_df = pd.DataFrame(rows)
    resid_df.to_csv(out_dir / f"resid_layer_sweep_{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(resid_df["layer"], resid_df["delta_restore"], marker="o", ms=3)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("Layer (resid_pre patched)")
    ax.set_ylabel(r"$\Delta$ = M_corrupted $-$ M_patched")
    ax.set_title(f"Full-residual-stream patch, {tag} direction\n"
                 f"(patch source: {'clean/salient' if direction == 'clean' else 'flip/matches-hallucination'})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_resid_layer_sweep_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved resid_layer_sweep_{tag}.csv / fig_resid_layer_sweep_{tag}.png")

    # 2) Drill down at the peak layer: attn-all vs mlp, then per-head
    peak_row = resid_df.loc[resid_df["delta_restore"].abs().idxmax()]
    peak_layer = int(peak_row["layer"])
    print(f"  Peak layer (|Δ| max) = {peak_layer} (Δ={peak_row['delta_restore']:+.3f})")

    breakdown_rows = [
        {"granularity": "resid_pre (whole layer)", "layer": peak_layer, "head": None,
         "delta_restore": float(peak_row["delta_restore"])},
    ]
    m_attn = logit_diff_at_last_pos(model, ids_corrupted, tok_h, tok_f,
                                    fwd_hooks=patch_attn_all(src_cache, peak_layer))
    breakdown_rows.append({"granularity": "attn (all heads)", "layer": peak_layer,
                           "head": None, "delta_restore": m_corrupted - m_attn})
    m_mlp = logit_diff_at_last_pos(model, ids_corrupted, tok_h, tok_f,
                                   fwd_hooks=patch_mlp(src_cache, peak_layer))
    breakdown_rows.append({"granularity": "mlp", "layer": peak_layer,
                           "head": None, "delta_restore": m_corrupted - m_mlp})

    head_deltas = []
    for h in range(n_heads):
        m_h = logit_diff_at_last_pos(model, ids_corrupted, tok_h, tok_f,
                                     fwd_hooks=patch_attn_head(src_cache, peak_layer, h))
        d = m_corrupted - m_h
        head_deltas.append(d)
        breakdown_rows.append({"granularity": "attn_head", "layer": peak_layer,
                               "head": h, "delta_restore": d})

    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_df.to_csv(out_dir / f"peak_layer_breakdown_{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(n_heads), head_deltas, color="tomato")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel(f"Head index (layer {peak_layer})")
    ax.set_ylabel(r"$\Delta$ = M_corrupted $-$ M_patched")
    ax.set_title(f"Per-head attn.hook_z patch at peak layer {peak_layer}, {tag} direction")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_peak_layer_heads_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved peak_layer_breakdown_{tag}.csv / fig_peak_layer_heads_{tag}.png")

    summary_lines.append(
        f"  Peak layer {peak_layer}: resid_pre Δ={peak_row['delta_restore']:+.3f}, "
        f"attn-all Δ={m_corrupted - m_attn:+.3f}, mlp Δ={m_corrupted - m_mlp:+.3f}, "
        f"max |per-head Δ|={max(abs(d) for d in head_deltas):+.3f}"
    )
    return summary_lines


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
    p.add_argument("--direction", choices=["restore", "sensitivity", "both"],
                   default="both")
    p.add_argument("--out", default="counterfactual_patch_demo_out")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, args.device)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    equalized = equalize_lengths(model, args.device, {
        "corrupted": _TRANSCRIPT_CORRUPTED,
        "clean": _TRANSCRIPT_CLEAN,
        "flip": _TRANSCRIPT_FLIP,
    })
    ids_corrupted = equalized["corrupted"]
    ids_clean = equalized["clean"]
    ids_flip = equalized["flip"]

    assert ids_corrupted.shape[1] == ids_clean.shape[1] == ids_flip.shape[1], (
        f"Length-confound fix failed: shapes are "
        f"{ids_corrupted.shape[1]}, {ids_clean.shape[1]}, {ids_flip.shape[1]} "
        f"-- position -1 would sit at different absolute indices across "
        f"conditions, invalidating the patching comparison."
    )
    print(f"All three conditions equalised to {ids_corrupted.shape[1]} tokens "
         f"-- position -1 is the same absolute index in every condition.")

    assert_suffix_alignment(model, ids_corrupted, ids_clean, label="corrupted vs clean")
    assert_suffix_alignment(model, ids_corrupted, ids_flip, label="corrupted vs flip")

    tok_h = single_token_id(model, _Y_H_TEXT)
    tok_f = single_token_id(model, _Y_F_TEXT)
    print(f"y_H = {_Y_H_TEXT!r} -> token {tok_h} ({model.tokenizer.decode([tok_h])!r})")
    print(f"y_F = {_Y_F_TEXT!r} -> token {tok_f} ({model.tokenizer.decode([tok_f])!r})")

    m_corrupted = logit_diff_at_last_pos(model, ids_corrupted, tok_h, tok_f)

    if m_corrupted > 0:
        interpretation = "model prefers y_H ('denies') — reproduces the hallucination, as expected."
    else:
        interpretation = ("[!!] model does NOT prefer y_H here — baseline does not reproduce "
                          "the hallucination; re-check the generation prefix / tokenisation.")
    summary_lines = [
        f"All conditions equalised to {ids_corrupted.shape[1]} tokens (length-confound fix).",
        f"M_corrupted (real transcript, real prefix, no patch) = {m_corrupted:+.4f}",
        f"  Interpretation: {interpretation}",
    ]

    ll_df = logit_lens_trajectory(model, ids_corrupted, n_layers, tok_h, tok_f)
    ll_df.to_csv(out_dir / "logit_lens_trajectory.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ll_df["layer"], ll_df["score_diff"], marker="o", ms=3, color="purple")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"LogitLens: score($y_H$) $-$ score($y_F$)")
    ax.set_title("When does the hallucination-preference emerge? (corrupted run, no patching)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_logit_lens_trajectory.png", dpi=150)
    plt.close(fig)
    print("Saved logit_lens_trajectory.csv / fig_logit_lens_trajectory.png")

    if args.direction in ("restore", "both"):
        summary_lines += run_direction(model, "clean", ids_corrupted, ids_clean, tok_h, tok_f,
                                       n_layers, n_heads, out_dir, m_corrupted)
    if args.direction in ("sensitivity", "both"):
        summary_lines += run_direction(model, "flip", ids_corrupted, ids_flip, tok_h, tok_f,
                                       n_layers, n_heads, out_dir, m_corrupted)

    (out_dir / "baseline_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n" + "\n".join(summary_lines))
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
