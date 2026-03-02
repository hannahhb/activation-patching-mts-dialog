"""
Metric functions for activation patching experiments.

Core metric: Normalised Logit-Difference Recovery
  - 0 = still looks like the corrupted run
  - 1 = fully restored to clean behaviour
  - <0 = patching made things worse
"""

from dataclasses import dataclass
from typing import Tuple, List

import torch


@dataclass
class AnswerTokens:
    correct_str: str
    wrong_str:   str
    correct_id:  int
    wrong_id:    int


def find_answer_tokens(
    model,
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    top_k: int = 20,
) -> AnswerTokens:
    """
    Auto-detect the most-diverging token pair between clean and corrupted runs.

    Strategy: for each token in the top-k of clean, find the corresponding
    rank in the corrupted run, and pick the pair with the largest rank gap.

    Args:
        model            : HookedTransformer (for to_string / to_single_token).
        clean_logits     : [1, seq, vocab] from the clean forward pass.
        corrupted_logits : [1, seq, vocab] from the corrupted forward pass.
        top_k            : How many top tokens to consider.

    Returns:
        AnswerTokens with the best (correct, wrong) token pair.
    """
    clean_final     = clean_logits[0, -1, :]       # [vocab]
    corrupted_final = corrupted_logits[0, -1, :]   # [vocab]

    # Top-k tokens from each run
    clean_top_ids     = clean_final.topk(top_k).indices.tolist()
    corrupted_top_ids = corrupted_final.topk(top_k).indices.tolist()

    # Find the token that ranks highest in clean but lowest in corrupted
    best_correct_id, best_wrong_id, best_gap = None, None, -1.0

    for correct_id in clean_top_ids:
        for wrong_id in corrupted_top_ids:
            if correct_id == wrong_id:
                continue
            # Logit difference at clean position vs corrupted position
            clean_ld     = (clean_final[correct_id]     - clean_final[wrong_id]).item()
            corrupted_ld = (corrupted_final[correct_id] - corrupted_final[wrong_id]).item()
            gap = clean_ld - corrupted_ld
            if gap > best_gap:
                best_gap        = gap
                best_correct_id = correct_id
                best_wrong_id   = wrong_id

    correct_str = model.to_string([best_correct_id])
    wrong_str   = model.to_string([best_wrong_id])

    print(f"  Correct answer : {correct_str!r}  (id={best_correct_id})")
    print(f"  Wrong answer   : {wrong_str!r}     (id={best_wrong_id})")
    print(f"  Logit-diff gap : {best_gap:.3f}")

    return AnswerTokens(
        correct_str = correct_str,
        wrong_str   = wrong_str,
        correct_id  = best_correct_id,
        wrong_id    = best_wrong_id,
    )


def logit_diff(logits: torch.Tensor, correct_id: int, wrong_id: int) -> float:
    """Logit(correct) - Logit(wrong) at the final token position."""
    final = logits[0, -1, :]
    return (final[correct_id] - final[wrong_id]).item()


def normalised_recovery(
    patched_logits: torch.Tensor,
    clean_ld: float,
    corrupted_ld: float,
    correct_id: int,
    wrong_id: int,
) -> float:
    """
    (LD_patched - LD_corrupted) / (LD_clean - LD_corrupted)
    0 = no recovery, 1 = full recovery.
    """
    ld = logit_diff(patched_logits, correct_id, wrong_id)
    denom = clean_ld - corrupted_ld
    if abs(denom) < 1e-6:
        return 0.0
    return (ld - corrupted_ld) / denom


def top_k_predictions(model, logits: torch.Tensor, k: int = 10) -> List[Tuple[str, float]]:
    """Return top-k (token_string, logit) pairs from the final position."""
    final   = logits[0, -1, :]
    top_ids = final.topk(k).indices.tolist()
    return [(model.to_string([t]), final[t].item()) for t in top_ids]
