"""
Data loading for ACI-Bench.

Also provides:
  - extract_soap_sections(note_text)  → dict[section_name, section_text]
  - assign_sections(token_df, tokenizer, generated_note) → token_df with 'section' col
  - compute_complexity_features(dialogue, tokenizer)      → dict of scalar features
"""

from __future__ import annotations
import re
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from datasets import load_dataset

from config import (
    DATASET_HF_ID, DATASET_SPLIT, DIALOGUE_COL, NOTE_COL,
    N_FULL, RANDOM_SEED, SOAP_SECTIONS,
)


@dataclass
class ACIExample:
    idx:            int
    dialogue:       str
    reference_note: str
    generated_note: str = ""


def load_aci_examples(
    n:    int = N_FULL,
    seed: int = RANDOM_SEED,
) -> List[ACIExample]:
    """Load n ACI-Bench examples, reproducibly shuffled."""
    print(f"Loading ACI-Bench ({DATASET_HF_ID}, split={DATASET_SPLIT}) ...")
    ds = load_dataset(DATASET_HF_ID, split=DATASET_SPLIT, trust_remote_code=True)

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    subset  = ds.select(sorted(indices))

    examples = []
    for i, row in enumerate(subset):
        dialogue = row.get(DIALOGUE_COL, "").strip()
        ref_note = row.get(NOTE_COL, "").strip()
        if not dialogue:
            continue
        examples.append(ACIExample(
            idx=sorted(indices)[i],
            dialogue=dialogue,
            reference_note=ref_note,
        ))

    print(f"  Loaded {len(examples)} examples.")
    return examples


# ── SOAP Section Segmentation ──────────────────────────────────────────────────

# Regex matching any SOAP section header (case-insensitive, colon optional)
_HEADER_RE = re.compile(
    r"(?:^|\n)\s*("
    + "|".join(re.escape(s) for s in SOAP_SECTIONS)
    + r")\s*:?",
    re.IGNORECASE,
)


def extract_soap_sections(note_text: str) -> Dict[str, str]:
    """
    Split a generated SOAP note into its constituent sections.

    Returns:
        dict mapping canonical section name (lower-case) → section text.
        If no headers are found, returns {"unknown": note_text}.
    """
    boundaries: List[tuple] = []
    for m in _HEADER_RE.finditer(note_text):
        name = m.group(1).lower()
        boundaries.append((m.start(), m.end(), name))
    boundaries.sort(key=lambda x: x[0])

    if not boundaries:
        return {"unknown": note_text.strip()}

    sections: Dict[str, str] = {}
    for i, (_, end, name) in enumerate(boundaries):
        next_start = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(note_text)
        sections[name] = note_text[end:next_start].strip()

    return sections


def assign_sections(token_df, tokenizer, generated_note: str):
    """
    Add a 'section' column to token_df by mapping each generated token to the
    SOAP section it falls within (by reconstructed character offset).

    Tokens before the first detected header are labelled 'preamble'.
    """
    # Build sorted list of (char_start, section_name) from headers in the note
    boundaries: List[tuple] = []
    for m in _HEADER_RE.finditer(generated_note):
        boundaries.append((m.start(), m.group(1).lower()))
    boundaries.sort(key=lambda x: x[0])

    token_ids = token_df["token_id"].tolist()
    sections: List[str] = []
    cum_chars = 0

    for tid in token_ids:
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        # Use the midpoint of this token's span to decide section membership
        mid = cum_chars + max(len(tok_str) // 2, 0)
        sec = "preamble"
        for start, name in boundaries:
            if start <= mid:
                sec = name
            else:
                break
        sections.append(sec)
        cum_chars += len(tok_str)

    return token_df.assign(section=sections)


# ── Input Complexity Features ──────────────────────────────────────────────────

_HEDGE_WORDS = frozenset({
    "maybe", "perhaps", "might", "could", "possibly", "probably",
    "think", "feel", "seem", "appears", "appears", "suggest",
    "sometimes", "often", "usually", "generally",
})
_NEG_WORDS = frozenset({
    "not", "no", "never", "none", "neither", "nor", "without",
    "n't", "nobody", "nothing", "nowhere",
})


def compute_complexity_features(dialogue: str, tokenizer) -> Dict:
    """
    Compute scalar input-complexity features from the source dialogue.

    Features
    --------
    entity_density      Capitalised-word fraction (rough NER proxy).
    hedge_density       Fraction of tokens that are hedging words.
    negation_density    Fraction of tokens that are negation words.
    speaker_turns       Total Doctor + Patient turn count.
    source_len_tokens   Subword token count of the dialogue.
    type_token_ratio    Vocabulary richness (unique words / total words).
    """
    words = dialogue.split()
    n_words = len(words) + 1  # +1 avoids div-by-zero

    entity_density   = sum(bool(re.match(r"^[A-Z][a-z]+$", w)) for w in words) / n_words
    hedge_density    = sum(w.lower().rstrip(",.;") in _HEDGE_WORDS for w in words) / n_words
    negation_density = sum(w.lower().rstrip(",.;") in _NEG_WORDS  for w in words) / n_words

    doctor_turns  = len(re.findall(r"(?i)(?:doctor|physician|dr\.?)\s*:", dialogue))
    patient_turns = len(re.findall(r"(?i)patient\s*:",              dialogue))

    unique_words = len({w.lower() for w in words})
    ttr = unique_words / n_words

    src_tokens = tokenizer.encode(dialogue, add_special_tokens=False)

    return {
        "entity_density":    round(float(entity_density),   4),
        "hedge_density":     round(float(hedge_density),    4),
        "negation_density":  round(float(negation_density), 4),
        "speaker_turns":     doctor_turns + patient_turns,
        "source_len_tokens": len(src_tokens),
        "type_token_ratio":  round(float(ttr), 4),
    }
