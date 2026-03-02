"""
Data loading and preprocessing for MTS Dialog Summarisation experiments.

Dataset: har1/MTS_Dialogue-Clinical_Note
  - dialogue     : doctor-patient conversation (labelled "Doctor:" / "Patient:")
  - section_text : structured clinical note  (Symptoms / Diagnosis / History / Plan)
"""

import re
import random
from dataclasses import dataclass
from typing import List

from datasets import load_dataset

from config import (
    DATASET_NAME, DATASET_SPLIT,
    DIALOG_FIELD, SUMMARY_FIELD,
    N_EXAMPLES, RANDOM_SEED,
)


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class MTSExample:
    idx: int
    raw_dialogue: str
    raw_summary: str
    clean_prompt: str
    corrupted_prompt: str
    summary_stem: str          # the part of summary included in the prompt


# ── Prompt template ────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "Medical conversation transcript:\n"
    "{dialogue}\n\n"
    "Clinical note:\n"
    "{stem}"
)

# We feed the section header as a stem so the model predicts the first value.
# E.g. stem = "Symptoms:" → model predicts "  chest pain" or similar.
SUMMARY_STEM = "Symptoms:"


# ── Corruption strategies ──────────────────────────────────────────────────

def corrupt_speaker_swap(dialogue: str) -> str:
    """
    Swap Doctor and Patient speaker labels.
    Preserves exact token count — ideal for position-level patching.
    """
    # Use a neutral placeholder to avoid double-replacement
    swapped = re.sub(r"\bDoctor\b",  "__DOC__",  dialogue)
    swapped = re.sub(r"\bPatient\b", "Doctor",   swapped)
    swapped = re.sub(r"__DOC__",     "Patient",  swapped)
    return swapped


CORRUPTION_FNS = {
    "speaker_swap": corrupt_speaker_swap,
}


# ── Loading ────────────────────────────────────────────────────────────────

def load_mts_examples(
    n: int = N_EXAMPLES,
    seed: int = RANDOM_SEED,
    corruption: str = "speaker_swap",
    stem: str = SUMMARY_STEM,
) -> List[MTSExample]:
    """
    Load `n` examples from the MTS-Dialog dataset and prepare
    (clean_prompt, corrupted_prompt) pairs.

    Args:
        n          : Number of examples to return.
        seed       : Random seed for example selection.
        corruption : Which corruption function to apply ('speaker_swap').
        stem       : Summary prefix included in the prompt (model predicts next token).

    Returns:
        List of MTSExample dataclasses.
    """
    print(f"Loading dataset '{DATASET_NAME}' ...")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Reproducible shuffle then take n
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    subset  = ds.select(indices)

    corrupt_fn = CORRUPTION_FNS[corruption]
    examples   = []

    for i, row in enumerate(subset):
        dialogue = row[DIALOG_FIELD].strip()
        summary  = row[SUMMARY_FIELD].strip()

        corrupted_dialogue = corrupt_fn(dialogue)

        clean_prompt     = PROMPT_TEMPLATE.format(dialogue=dialogue,          stem=stem)
        corrupted_prompt = PROMPT_TEMPLATE.format(dialogue=corrupted_dialogue, stem=stem)

        examples.append(MTSExample(
            idx              = indices[i],
            raw_dialogue     = dialogue,
            raw_summary      = summary,
            clean_prompt     = clean_prompt,
            corrupted_prompt = corrupted_prompt,
            summary_stem     = stem,
        ))

    print(f"Loaded {len(examples)} examples (corruption='{corruption}').")
    return examples


# ── Inspection helper ──────────────────────────────────────────────────────

def preview_example(ex: MTSExample, char_limit: int = 600) -> None:
    """Print a readable preview of one MTSExample."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"Example #{ex.idx}")
    print(sep)
    print("CLEAN PROMPT (truncated):")
    print(ex.clean_prompt[:char_limit])
    print("\nCORRUPTED PROMPT (first 300 chars of dialogue only):")
    # Show just enough to confirm the swap
    first_300 = ex.corrupted_prompt[:300]
    print(first_300)
    print("\nSUMMARY (ground truth):")
    print(ex.raw_summary[:400])
    print(sep)


if __name__ == "__main__":
    examples = load_mts_examples(n=3)
    for ex in examples:
        preview_example(ex)
