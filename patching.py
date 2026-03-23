"""
Experiment 3 — PDSQI-9 Blind Spot Detection via Activation Patching.

Implements denoising activation patching following Heimersheim & Nanda (2024).

Clean run  : transcript with original clinical fact → cache activations
Corrupted  : minimally modified transcript (fact swap) → run with hooks

Patching sweeps:
  - Residual stream at each layer (resid_pre)
  - Attention output  (hook_attn_out)
  - MLP output        (hook_mlp_out)

Logit difference between correct and incorrect token at the target position
is used as the restoration metric.

Three candidate categories (selected from Exp 1+2 results):
  A. Coincidental correctness  — high MLP fraction, low lookback, high accuracy score
  B. Penalised inference       — high MLP mid, moderate lookback, low accuracy score
  C. Undetected fragility      — high overall PDSQI-9, but specific tokens MLP-dominated
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from config import N_LAYERS


# ── Hook name templates ────────────────────────────────────────────────────────

HOOK_RESID_PRE = "blocks.{}.hook_resid_pre"
HOOK_ATTN_OUT  = "blocks.{}.hook_attn_out"
HOOK_MLP_OUT   = "blocks.{}.hook_mlp_out"


# ── Hook factories ─────────────────────────────────────────────────────────────

def _patch_all(clean_act: torch.Tensor) -> Callable:
    def hook_fn(value: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return clean_act.to(value.device, value.dtype)
    return hook_fn


def _patch_position(clean_act: torch.Tensor, pos: int) -> Callable:
    """Patch a single token position."""
    def hook_fn(value: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        if pos < value.shape[1]:
            value = value.clone()
            value[:, pos, :] = clean_act[:, pos, :].to(value.device, value.dtype)
        return value
    return hook_fn


# ── Metric helpers ─────────────────────────────────────────────────────────────

def logit_diff(logits: torch.Tensor, correct_id: int, wrong_id: int) -> float:
    """Last-position logit difference: logit[correct] - logit[wrong]."""
    last = logits[0, -1]
    return (last[correct_id] - last[wrong_id]).item()


def normalised_recovery(
    patched_logits: torch.Tensor,
    clean_ld:       float,
    corrupted_ld:   float,
    correct_id:     int,
    wrong_id:       int,
) -> float:
    """
    Normalised logit diff recovery.
    0 = no recovery (still at corrupted level).
    1 = full recovery (at clean level).
    Values outside [0,1] are possible (super-recovery or over-correction).
    """
    patched_ld = logit_diff(patched_logits, correct_id, wrong_id)
    gap = clean_ld - corrupted_ld
    if abs(gap) < 1e-6:
        return 0.0
    return (patched_ld - corrupted_ld) / gap


# ── Corruption strategies ──────────────────────────────────────────────────────

def corrupt_dosage_swap(
    dialogue: str,
    original: str,
    replacement: str,
) -> str:
    """Replace a specific string in the dialogue (e.g. '500mg' → '1000mg')."""
    return dialogue.replace(original, replacement, 1)


def corrupt_negation_flip(
    dialogue: str,
    phrase:   str,
    negated:  str,
) -> str:
    """
    Flip a negation (e.g. 'denies chest pain' → 'reports chest pain').
    Case-insensitive.
    """
    import re
    return re.sub(re.escape(phrase), negated, dialogue, count=1, flags=re.I)


# ── Candidate data structure ───────────────────────────────────────────────────

@dataclass
class PatchingCandidate:
    """One (clean, corrupted) encounter pair ready for patching analysis."""
    encounter_idx:   int
    category:        str       # "A", "B", or "C"
    description:     str       # human-readable description of the fact being tested
    dialogue_clean:  str
    dialogue_corrupt: str
    target_fact:     str       # the clinical fact under test (for the report)
    correct_token:   str       # the correct token at the target position
    wrong_token:     str       # the incorrect token (fact-swapped version)
    # Filled after patching:
    correct_id:      int  = -1
    wrong_id:        int  = -1
    target_pos:      int  = -1   # position in the generated sequence to measure
    clean_ld:        float = 0.0
    corrupted_ld:    float = 0.0
    resid_scores:    List[float] = field(default_factory=list)
    attn_scores:     List[float] = field(default_factory=list)
    mlp_scores:      List[float] = field(default_factory=list)


# ── Candidate selection ────────────────────────────────────────────────────────

def select_candidates(
    examples,            # List[ACIExample] with generated_note filled
    token_dfs,           # List[pd.DataFrame] from Experiment 1
    pdsqi9_scores,       # List[Dict] full-note scores from Experiment 2
    n_per_category: int = 7,
) -> List[PatchingCandidate]:
    """
    Scan Experiment 1+2 outputs and select candidates for each category.
    The caller must still set correct_token, wrong_token, and build
    corrupted dialogue strings for each candidate — this requires manual
    inspection or a separate entity extraction step.

    This function outputs a scaffold list with category labels and encounter
    indices. The run_exp3.py script completes the corruption strings.
    """
    from config import (
        COINCIDENTAL_CORRECTNESS_MLP_FRAC,
        COINCIDENTAL_CORRECTNESS_LOOKBACK,
        PENALISED_INFERENCE_ACCURATE_MAX,
    )
    import pandas as pd

    candidates: List[PatchingCandidate] = []

    for i, (ex, token_df, scores) in enumerate(
        zip(examples, token_dfs, pdsqi9_scores)
    ):
        if not isinstance(token_df, pd.DataFrame) or token_df.empty:
            continue

        full_scores = scores.get("full", {})
        accurate    = full_scores.get("accurate")

        mean_mlp_frac = token_df["mlp_fraction"].mean()
        mean_lookback = token_df["lookback_ratio"].mean()

        # ── Category A: coincidental correctness ──────────────────────────
        if (
            len(candidates) < n_per_category
            and mean_mlp_frac > COINCIDENTAL_CORRECTNESS_MLP_FRAC
            and mean_lookback < COINCIDENTAL_CORRECTNESS_LOOKBACK
            and isinstance(accurate, (int, float))
            and accurate >= 4
        ):
            candidates.append(PatchingCandidate(
                encounter_idx=i,
                category="A",
                description=(
                    f"High MLP fraction ({mean_mlp_frac:.2f}), "
                    f"low lookback ({mean_lookback:.2f}), accurate={accurate}"
                ),
                dialogue_clean=ex.dialogue,
                dialogue_corrupt="",   # to be filled by run_exp3.py
                target_fact="",
                correct_token="",
                wrong_token="",
            ))

        # ── Category B: penalised inference ───────────────────────────────
        assessment_rows = token_df[token_df["section"] == "assessment"]
        if not assessment_rows.empty:
            assess_mlp_mid = assessment_rows["mlp_contribution_mid"].mean()
            assess_lookback = assessment_rows["lookback_ratio"].mean()
        else:
            assess_mlp_mid = assess_lookback = 0.0

        if (
            len([c for c in candidates if c.category == "B"]) < n_per_category
            and assess_mlp_mid > 0.3
            and 0.2 < assess_lookback < 0.6
            and isinstance(accurate, (int, float))
            and accurate <= PENALISED_INFERENCE_ACCURATE_MAX
        ):
            candidates.append(PatchingCandidate(
                encounter_idx=i,
                category="B",
                description=(
                    f"Assessment: mid-MLP={assess_mlp_mid:.2f}, "
                    f"lookback={assess_lookback:.2f}, accurate={accurate}"
                ),
                dialogue_clean=ex.dialogue,
                dialogue_corrupt="",
                target_fact="",
                correct_token="",
                wrong_token="",
            ))

        # ── Category C: undetected fragility ──────────────────────────────
        # Good overall PDSQI-9 but specific high-MLP tokens with low extractive score
        high_mlp_low_ext = token_df[
            (token_df["mlp_fraction"] > 0.65) &
            (token_df["extractive_score"] < 0.1)
        ]
        overall_mean_score = np.mean([
            v for v in full_scores.values()
            if isinstance(v, (int, float)) and v > 0
        ] or [0])

        if (
            len([c for c in candidates if c.category == "C"]) < n_per_category
            and overall_mean_score >= 3.5
            and len(high_mlp_low_ext) >= 5
        ):
            candidates.append(PatchingCandidate(
                encounter_idx=i,
                category="C",
                description=(
                    f"Overall PDSQI-9 mean={overall_mean_score:.2f}, "
                    f"{len(high_mlp_low_ext)} high-MLP/low-extractive tokens"
                ),
                dialogue_clean=ex.dialogue,
                dialogue_corrupt="",
                target_fact="",
                correct_token="",
                wrong_token="",
            ))

    return candidates


# ── Patching sweeps ────────────────────────────────────────────────────────────

def _find_target_position(
    model:         HookedTransformer,
    clean_tokens:  torch.Tensor,
    correct_token: str,
    prompt_len:    int,
) -> Tuple[int, int]:
    """
    Find the first position in the generated portion where `correct_token`
    appears and return (position_index, token_id).
    Raises ValueError if not found.
    """
    token_ids = clean_tokens[0, prompt_len:].tolist()
    target_id = model.to_single_token(correct_token)

    for i, tid in enumerate(token_ids):
        if tid == target_id:
            return prompt_len + i, target_id

    raise ValueError(
        f"Token '{correct_token}' (id={target_id}) not found in generated sequence."
    )


def run_patching_candidate(
    model:     HookedTransformer,
    candidate: PatchingCandidate,
    prompt_fn: Callable[[str], str],
    verbose:   bool = True,
) -> PatchingCandidate:
    """
    Run the full patching suite for one candidate and populate its score fields.

    Args:
        model      : loaded HookedTransformer
        candidate  : PatchingCandidate with dialogue_clean + dialogue_corrupt set
        prompt_fn  : function mapping dialogue → prompt string (same as generation)
        verbose    : print layer-by-layer scores

    Returns:
        The same candidate object with patching results filled in.
    """
    clean_prompt = prompt_fn(candidate.dialogue_clean)
    corr_prompt  = prompt_fn(candidate.dialogue_corrupt)

    clean_tokens = model.to_tokens(clean_prompt,  prepend_bos=True)
    corr_tokens  = model.to_tokens(corr_prompt,   prepend_bos=True)

    prompt_len   = clean_tokens.shape[1]

    correct_id = model.to_single_token(candidate.correct_token)
    wrong_id   = model.to_single_token(candidate.wrong_token)

    candidate.correct_id = correct_id
    candidate.wrong_id   = wrong_id

    # ── Baseline forward passes ────────────────────────────────────────────
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corr_logits, _            = model.run_with_cache(corr_tokens)

    candidate.clean_ld     = logit_diff(clean_logits, correct_id, wrong_id)
    candidate.corrupted_ld = logit_diff(corr_logits,  correct_id, wrong_id)

    if verbose:
        print(
            f"  [Cat {candidate.category}] enc={candidate.encounter_idx} "
            f"clean_LD={candidate.clean_ld:+.3f}  "
            f"corrupted_LD={candidate.corrupted_ld:+.3f}"
        )

    # ── Layer sweeps ───────────────────────────────────────────────────────
    def sweep(hook_template: str) -> List[float]:
        scores = []
        for L in range(N_LAYERS):
            hook_name = hook_template.format(L)
            if hook_name not in clean_cache:
                scores.append(0.0)
                continue
            clean_act = clean_cache[hook_name]
            with torch.no_grad():
                patched = model.run_with_hooks(
                    corr_tokens,
                    fwd_hooks=[(hook_name, _patch_all(clean_act))],
                )
            rec = normalised_recovery(
                patched, candidate.clean_ld, candidate.corrupted_ld,
                correct_id, wrong_id,
            )
            scores.append(round(rec, 4))
            if verbose:
                print(f"    L{L:02d} {hook_template.split('.')[2][:4]}  {rec:+.4f}")
        return scores

    if verbose: print("  [resid_pre sweep]")
    candidate.resid_scores = sweep(HOOK_RESID_PRE)

    if verbose: print("  [attn_out sweep]")
    candidate.attn_scores  = sweep(HOOK_ATTN_OUT)

    if verbose: print("  [mlp_out sweep]")
    candidate.mlp_scores   = sweep(HOOK_MLP_OUT)

    return candidate


# ── Blind spot analysis ────────────────────────────────────────────────────────

def classify_grounding(candidate: PatchingCandidate, threshold: float = 0.3) -> str:
    """
    Classify whether the clinical fact is retrieval-grounded or parametric.

    A fact is 'retrieval-grounded' if patching clean attention restores the
    logit diff significantly. Specifically: does the max restoration across
    layers when patching attn_out exceed `threshold`?

    Returns: "retrieval", "parametric", or "mixed"
    """
    if not candidate.attn_scores:
        return "unknown"

    max_attn_restore = max(candidate.attn_scores)
    max_mlp_restore  = max(candidate.mlp_scores)

    if max_attn_restore > threshold and max_mlp_restore <= threshold:
        return "retrieval"
    elif max_mlp_restore > threshold and max_attn_restore <= threshold:
        return "parametric"
    elif max_attn_restore > threshold and max_mlp_restore > threshold:
        return "mixed"
    else:
        return "neither"


def build_blind_spot_report(candidates: List[PatchingCandidate]) -> List[Dict]:
    """
    Build a structured report entry for each patched candidate.
    """
    rows = []
    for c in candidates:
        grounding = classify_grounding(c)
        pdsqi9_correct = {
            "A": grounding == "parametric",    # PDSQI-9 said high accurate but it's parametric
            "B": grounding == "retrieval",     # PDSQI-9 penalised but it's actually grounded
            "C": grounding == "parametric",    # looks good but fragile
        }.get(c.category, False)

        rows.append({
            "encounter_idx":   c.encounter_idx,
            "category":        c.category,
            "description":     c.description,
            "target_fact":     c.target_fact,
            "correct_token":   c.correct_token,
            "wrong_token":     c.wrong_token,
            "clean_ld":        round(c.clean_ld, 3),
            "corrupted_ld":    round(c.corrupted_ld, 3),
            "grounding":       grounding,
            "max_attn_restore": max(c.attn_scores) if c.attn_scores else None,
            "max_mlp_restore":  max(c.mlp_scores)  if c.mlp_scores  else None,
            "pdsqi9_blind_spot": pdsqi9_correct,
            "peak_attn_layer": (
                int(np.argmax(c.attn_scores)) if c.attn_scores else None
            ),
            "peak_mlp_layer":  (
                int(np.argmax(c.mlp_scores))  if c.mlp_scores  else None
            ),
        })
    return rows
