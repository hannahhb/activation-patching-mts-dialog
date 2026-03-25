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
