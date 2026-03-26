"""
Per-entity Direct Logit Attribution (DLA) for clinical entity recall.

Methodology (analogous to ROME / Ferrando et al.)
--------------------------------------------------
For each clinical entity that the model reproduces in the generated SOAP note,
and for each output token position where that entity is generated:

  head_contribution[l, h] =
      (z[l, pos, h] @ W_O[l, h]) · W_U[:, token_id_at_pos]

  mlp_contribution[l] =
      mlp_out[l, pos] · W_U[:, token_id_at_pos]

These are averaged over all output positions of the entity (for multi-token
entities) to give a single [n_layers, n_heads] and [n_layers] summary.

Attention-to-source check
-------------------------
For each of the top-k contributing heads, the attention pattern
(hook_pattern) is inspected at the output positions: what fraction of
attention weight is directed to the entity's source positions in the
transcript?  A high score (> threshold) means the head is both (a) pushing
logits toward the entity token and (b) attending to where that entity
appears in the dialogue — strong evidence of a copy / retrieval circuit.

Memory note
-----------
hook_pattern is [batch, n_heads, seq, seq] per layer — for a 2048-token
sequence with 16 heads × 42 layers this is ~10 GB.  The entity_cache_filter
is selective (hook_z + hook_mlp_out + hook_pattern) but can still be large.
For long sequences consider limiting to a subset of layers or entities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

from entities import ClinicalEntity


# ── Result Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class EntityDLAResult:
    """Mechanistic attribution result for one clinical entity."""

    entity_text:  str
    entity_type:  str

    # Absolute positions in the full [prompt + generated] token sequence
    output_positions:  List[int]
    entity_token_ids:  List[int]   # token ID at each output position

    # Contributions averaged over output positions
    head_contributions: np.ndarray   # [n_layers, n_heads] — float32
    mlp_contributions:  np.ndarray   # [n_layers]          — float32

    # Top-k summaries (sorted by |contribution|, descending)
    top_heads:      List[Tuple[int, int, float]]   # (layer, head, mean_contrib)
    top_mlp_layers: List[Tuple[int, float]]        # (layer, mean_contrib)

    # Aggregate fractions
    attn_fraction: float
    mlp_fraction:  float

    # Attention-to-source: for each top head, fraction of attention at output
    # positions directed to the entity's source positions in the transcript
    source_attention_scores:    Dict[Tuple[int, int], float] = field(default_factory=dict)
    top_heads_attend_to_source: bool = False

    def to_dict(self) -> dict:
        return {
            "entity_text":     self.entity_text,
            "entity_type":     self.entity_type,
            "output_positions": self.output_positions,
            "attn_fraction":   self.attn_fraction,
            "mlp_fraction":    self.mlp_fraction,
            "top_heads": [
                {"layer": l, "head": h, "contribution": round(v, 5)}
                for l, h, v in self.top_heads
            ],
            "top_mlp_layers": [
                {"layer": l, "contribution": round(v, 5)}
                for l, v in self.top_mlp_layers
            ],
            "source_attention_scores": {
                f"L{l}H{h}": round(v, 5)
                for (l, h), v in self.source_attention_scores.items()
            },
            "top_heads_attend_to_source": self.top_heads_attend_to_source,
        }


@dataclass
class EncounterEntitySummary:
    """All entity DLA results for one encounter."""

    encounter_idx: int
    n_entities_extracted: int      # before token-level filtering
    n_entities_analysed:  int      # after successful DLA

    results: List[EntityDLAResult]

    def to_dict(self) -> dict:
        return {
            "encounter_idx":        self.encounter_idx,
            "n_entities_extracted": self.n_entities_extracted,
            "n_entities_analysed":  self.n_entities_analysed,
            "entities":             [r.to_dict() for r in self.results],
        }


# ── Cache Filter ───────────────────────────────────────────────────────────────

def entity_cache_filter(name: str) -> bool:
    """
    Capture the tensors needed for entity-level DLA:
      hook_z        — per-head pre-projection activations  [batch, seq, n_heads, d_head]
      hook_mlp_out  — MLP output                           [batch, seq, d_model]
      hook_pattern  — attention weights                    [batch, n_heads, seq, seq]
    """
    return (
        name.endswith("attn.hook_z")
        or name.endswith("hook_mlp_out")
        or name.endswith("attn.hook_pattern")
    )


# ── Core DLA per Entity ────────────────────────────────────────────────────────

@torch.inference_mode()
def compute_entity_dla(
    model:                 HookedTransformer,
    cache,
    entity:                ClinicalEntity,
    full_tokens:           torch.Tensor,    # [1, seq_len]
    source_attn_threshold: float = 0.05,
    top_k_heads:           int   = 10,
    top_k_mlp:             int   = 5,
) -> Optional[EntityDLAResult]:
    """
    Compute per-entity DLA at every output position where the entity is generated.

    For each output position `pos` and each layer `L`:

        head_out[h] = z[L, pos, h, :] @ W_O[L, h, :, :]     # [d_model]
        head_logit[h] = head_out[h] · W_U[:, token_id_at_pos]

        mlp_logit = mlp_out[L, pos, :] · W_U[:, token_id_at_pos]

    Contributions are averaged across the entity's output positions.

    Returns None if the entity has no output positions in the cache.
    """
    output_positions = entity.note_token_positions
    if not output_positions:
        return None

    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    W_U      = model.W_U           # [d_model, d_vocab]

    head_contribs = np.zeros((n_layers, n_heads), dtype=np.float32)
    mlp_contribs  = np.zeros(n_layers,            dtype=np.float32)
    entity_token_ids: List[int] = []

    for pos in output_positions:
        token_id = int(full_tokens[0, pos].item())
        entity_token_ids.append(token_id)
        unembed = W_U[:, token_id]   # [d_model]

        for L in range(n_layers):
            # ── Attention head contributions ───────────────────────────────────
            z_key = f"blocks.{L}.attn.hook_z"
            W_O   = model.blocks[L].attn.W_O   # [n_heads, d_head, d_model]
            z_pos = cache[z_key][0, pos]        # [n_heads, d_head]

            # head_out[h] = z_pos[h, :] @ W_O[h, :, :]  →  [d_model]
            # contribution = head_out @ unembed           →  scalar per head
            head_out    = torch.einsum("hd, hde -> he", z_pos, W_O)  # [n_heads, d_model]
            contributions = head_out @ unembed                         # [n_heads]
            head_contribs[L] += contributions.cpu().float().numpy()

            # ── MLP contributions ──────────────────────────────────────────────
            mlp_key = f"blocks.{L}.hook_mlp_out"
            mlp_pos = cache[mlp_key][0, pos]   # [d_model]
            mlp_contribs[L] += float((mlp_pos @ unembed).item())

    # Average over output positions
    n_pos = len(output_positions)
    head_contribs /= n_pos
    mlp_contribs  /= n_pos

    # ── Top-k summaries ────────────────────────────────────────────────────────
    flat_heads = [
        (L, h, float(head_contribs[L, h]))
        for L in range(n_layers)
        for h in range(n_heads)
    ]
    top_heads = sorted(flat_heads, key=lambda x: abs(x[2]), reverse=True)[:top_k_heads]

    top_mlp = sorted(
        [(L, float(mlp_contribs[L])) for L in range(n_layers)],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_k_mlp]

    # ── Fractions ──────────────────────────────────────────────────────────────
    total_attn = float(np.abs(head_contribs).sum()) + 1e-8
    total_mlp  = float(np.abs(mlp_contribs).sum())  + 1e-8
    total      = total_attn + total_mlp
    attn_fraction = total_attn / total
    mlp_fraction  = total_mlp  / total

    # ── Attention-to-source check ──────────────────────────────────────────────
    source_positions     = entity.source_token_positions
    source_attn_scores: Dict[Tuple[int, int], float] = {}

    if source_positions:
        for L, h, _ in top_heads:
            pattern_key = f"blocks.{L}.attn.hook_pattern"
            if pattern_key not in cache:
                continue
            # hook_pattern: [batch, n_heads, seq_q, seq_k]
            attn_to_src = 0.0
            for pos in output_positions:
                row = cache[pattern_key][0, h, pos, :]          # [seq_k]
                attn_to_src += float(row[source_positions].sum().item())
            source_attn_scores[(L, h)] = round(attn_to_src / n_pos, 5)

    top_heads_attend = (
        float(np.mean(list(source_attn_scores.values()))) > source_attn_threshold
        if source_attn_scores
        else False
    )

    return EntityDLAResult(
        entity_text              = entity.text,
        entity_type              = entity.entity_type,
        output_positions         = output_positions,
        entity_token_ids         = entity_token_ids,
        head_contributions       = head_contribs,
        mlp_contributions        = mlp_contribs,
        top_heads                = top_heads,
        top_mlp_layers           = top_mlp,
        attn_fraction            = round(attn_fraction, 4),
        mlp_fraction             = round(mlp_fraction,  4),
        source_attention_scores  = source_attn_scores,
        top_heads_attend_to_source = bool(top_heads_attend),
    )


# ── Encounter-level Runner ─────────────────────────────────────────────────────

@torch.inference_mode()
def run_entity_analysis(
    model:        HookedTransformer,
    full_tokens:  torch.Tensor,
    prompt_len:   int,
    entities:     List[ClinicalEntity],
    encounter_idx: int = 0,
    top_k_heads:  int  = 10,
    top_k_mlp:    int  = 5,
    source_attn_threshold: float = 0.05,
) -> EncounterEntitySummary:
    """
    Run a single forward pass with entity-level cache hooks and compute DLA
    for all provided entities.

    All entities share the same cache, so only one forward pass is needed
    regardless of how many entities are present.

    Parameters
    ----------
    model:
        Loaded TransformerLens HookedTransformer.
    full_tokens:
        [1, seq_len] tensor of prompt + generated token IDs.
    prompt_len:
        Number of tokens in the prompt (boundary between source and generated).
    entities:
        List of ClinicalEntity objects from ``entities.extract_entities``.
    encounter_idx:
        Identifier for the encounter (used in output metadata).
    top_k_heads:
        Number of top attention heads to record per entity.
    top_k_mlp:
        Number of top MLP layers to record per entity.
    source_attn_threshold:
        Minimum mean attention-to-source fraction for a head to be considered
        "attending to the source."
    """
    if not entities:
        return EncounterEntitySummary(
            encounter_idx=encounter_idx,
            n_entities_extracted=0,
            n_entities_analysed=0,
            results=[],
        )

    _, cache = model.run_with_cache(
        full_tokens,
        names_filter=entity_cache_filter,
        return_type=None,
    )

    results: List[EntityDLAResult] = []
    for entity in entities:
        result = compute_entity_dla(
            model=model,
            cache=cache,
            entity=entity,
            full_tokens=full_tokens,
            source_attn_threshold=source_attn_threshold,
            top_k_heads=top_k_heads,
            top_k_mlp=top_k_mlp,
        )
        if result is not None:
            results.append(result)

    del cache
    torch.cuda.empty_cache()

    return EncounterEntitySummary(
        encounter_idx         = encounter_idx,
        n_entities_extracted  = len(entities),
        n_entities_analysed   = len(results),
        results               = results,
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def save_entity_summaries(
    summaries: List[EncounterEntitySummary],
    out_path:  Path,
) -> None:
    """Serialise a list of EncounterEntitySummary objects to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([s.to_dict() for s in summaries], f, indent=2)
    print(f"Entity summaries → {out_path}")


# ── Convenience: cross-encounter head ranking ──────────────────────────────────

def aggregate_head_rankings(
    summaries: List[EncounterEntitySummary],
    entity_types: Optional[List[str]] = None,
) -> List[Tuple[int, int, float, int]]:
    """
    Aggregate top-head contributions across all encounters and entity types.

    Returns a list of (layer, head, mean_abs_contribution, n_appearances)
    sorted by mean_abs_contribution descending.

    Useful for identifying globally important heads for clinical entity recall.

    Parameters
    ----------
    entity_types:
        If given, restrict to entities of these types.
    """
    from collections import defaultdict

    scores:     Dict[Tuple[int, int], List[float]] = defaultdict(list)
    appearances: Dict[Tuple[int, int], int]        = defaultdict(int)

    for summary in summaries:
        for result in summary.results:
            if entity_types and result.entity_type not in entity_types:
                continue
            for L, h, v in result.top_heads:
                scores[(L, h)].append(abs(v))
                appearances[(L, h)] += 1

    ranked = [
        (L, h, float(np.mean(vs)), appearances[(L, h)])
        for (L, h), vs in scores.items()
    ]
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked
