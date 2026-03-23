"""
Data loading, SOAP segmentation, and input complexity features for ACI-Bench.
"""

from __future__ import annotations
import re
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from datasets import load_dataset

from config import (
    DATASET_HF_ID, DATASET_SPLIT, DIALOGUE_COL, NOTE_COL,
    SOAP_SECTIONS, N_FULL, RANDOM_SEED,
)


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class ACIExample:
    idx:            int
    dialogue:       str
    reference_note: str
    generated_note: str  = ""    # filled by run_exp1.py
    soap_sections:  Dict[str, str] = field(default_factory=dict)  # section → text
    token_df_path:  str  = ""    # path to per-token parquet saved by Experiment 1


# ── SOAP segmentation ──────────────────────────────────────────────────────────

# Maps varied header spellings to canonical section names
_HEADER_ALIASES: Dict[str, str] = {
    "cc":                            "chief complaint",
    "chief complaint":               "chief complaint",
    "hpi":                           "history of present illness",
    "history of present illness":    "history of present illness",
    "history of the present illness":"history of present illness",
    "ros":                           "review of systems",
    "review of systems":             "review of systems",
    "pe":                            "physical examination",
    "physical exam":                 "physical examination",
    "physical examination":          "physical examination",
    "vitals":                        "physical examination",
    "assessment":                    "assessment",
    "impression":                    "assessment",
    "assessment and plan":           "assessment",   # split below
    "plan":                          "plan",
    "medications":                   "medications",
    "current medications":           "medications",
    "allergies":                     "allergies",
    "pmh":                           "past medical history",
    "past medical history":          "past medical history",
    "past history":                  "past medical history",
    "sh":                            "social history",
    "social history":                "social history",
    "fh":                            "family history",
    "family history":                "family history",
}

_HEADER_RE = re.compile(
    r"(?m)^[ \t]*("
    + "|".join(re.escape(k) for k in sorted(_HEADER_ALIASES, key=len, reverse=True))
    + r")[ \t]*:[ \t]*\n?",
    re.IGNORECASE,
)


def segment_soap(note_text: str) -> Dict[str, str]:
    """
    Split a clinical note into canonical SOAP sections.
    Returns a dict mapping section name → section text.
    Tokens not under any recognised header go to "preamble".
    """
    sections: Dict[str, str] = {}
    spans: List[tuple] = []

    for m in _HEADER_RE.finditer(note_text):
        canonical = _HEADER_ALIASES[m.group(1).strip().lower()]
        spans.append((m.start(), m.end(), canonical))

    if not spans:
        return {"preamble": note_text.strip()}

    # Text before the first header
    if spans[0][0] > 0:
        preamble = note_text[: spans[0][0]].strip()
        if preamble:
            sections["preamble"] = preamble

    for i, (start, end, name) in enumerate(spans):
        content_end = spans[i + 1][0] if i + 1 < len(spans) else len(note_text)
        content = note_text[end:content_end].strip()

        # "assessment and plan" → split on "Plan:" if present inside
        if name == "assessment":
            plan_match = re.search(r"(?m)^[ \t]*plan[ \t]*:[ \t]*\n?", content, re.I)
            if plan_match:
                sections["assessment"] = content[: plan_match.start()].strip()
                sections["plan"] = content[plan_match.end():].strip()
                continue

        # Merge if section already exists (e.g. two "plan" blocks)
        if name in sections:
            sections[name] = sections[name] + "\n" + content
        else:
            sections[name] = content

    return {k: v for k, v in sections.items() if v}


def assign_token_sections(
    token_strs: List[str],
    generated_note: str,
    sections: Dict[str, str],
) -> List[str]:
    """
    Map each generated token (by approximate character offset) to its SOAP section.
    Returns a list of section labels, one per token.
    Falls back to 'unknown' if a token cannot be placed.
    """
    # Build a list of (start_char, section_name) boundaries in note order
    boundaries: List[tuple] = []
    pos = 0
    for section in SOAP_SECTIONS + ["preamble", "unknown"]:
        text = sections.get(section, "")
        if not text:
            continue
        idx = generated_note.find(text, pos)
        if idx >= 0:
            boundaries.append((idx, section))
            pos = idx

    boundaries.sort()

    # Reconstruct character offsets for each token via cumulative concatenation
    labels = []
    char_pos = 0
    current_section = boundaries[0][1] if boundaries else "unknown"

    for tok in token_strs:
        # Advance current section based on char_pos
        for start, sec in boundaries:
            if char_pos >= start:
                current_section = sec
        labels.append(current_section)
        char_pos += len(tok)

    return labels


# ── Complexity features ────────────────────────────────────────────────────────

_HEDGE_RE = re.compile(
    r"\b(may|might|possibly|possible|likely|appears|suggests?|consider|"
    r"perhaps|probably|uncertain|unclear|suspected?|possibly|could|would)\b",
    re.I,
)

_NEGATION_RE = re.compile(
    r"\b(no|not|denies?|without|negative|absent|none|never|nor|"
    r"unremarkable|non-?contributory)\b",
    re.I,
)

_SPEAKER_TURN_RE = re.compile(r"^(Doctor|Patient)\s*:", re.M)


def compute_complexity_features(dialogue: str, tokenizer) -> Dict[str, float]:
    """
    Compute input complexity features from the raw transcript.
    SciSpacy NER is optional — falls back to word-count proxy if unavailable.
    """
    words = dialogue.split()
    n_words = max(len(words), 1)
    turns = _SPEAKER_TURN_RE.findall(dialogue)

    # Entity density via SciSpacy (optional)
    entity_density = _entity_density_scispacy(dialogue, n_words)

    return {
        "entity_density":    entity_density,
        "hedge_density":     len(_HEDGE_RE.findall(dialogue)) / n_words,
        "negation_density":  len(_NEGATION_RE.findall(dialogue)) / n_words,
        "speaker_turns":     len(turns),
        "source_len_tokens": len(tokenizer.encode(dialogue)),
        "type_token_ratio":  len(set(w.lower() for w in words)) / n_words,
    }


def _entity_density_scispacy(text: str, n_words: int) -> float:
    """Return medical entity density. Falls back to 0.0 if SciSpacy not installed."""
    try:
        import spacy
        nlp = _get_scispacy_model()
        doc = nlp(text)
        return len(doc.ents) / max(n_words, 1)
    except Exception:
        return 0.0


_SPACY_NLP = None


def _get_scispacy_model():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load("en_core_sci_sm")
        except OSError:
            _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


# ── Extractive score ───────────────────────────────────────────────────────────

def extractive_scores(
    gen_ids:  List[int],
    src_ids:  List[int],
    window:   int = 5,
) -> List[float]:
    """
    Per-token extractive score: how long is the longest verbatim token n-gram
    starting at this position that also exists in the source?
    Score = min(lcs_length / window, 1.0).
    """
    n_gen = len(gen_ids)
    n_src = len(src_ids)
    scores: List[float] = []

    for i in range(n_gen):
        best = 0
        for j in range(n_src):
            run = 0
            while (
                i + run < n_gen and
                j + run < n_src and
                gen_ids[i + run] == src_ids[j + run]
            ):
                run += 1
            if run > best:
                best = run
        scores.append(min(best / window, 1.0))

    return scores


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_aci_examples(
    n: int = N_FULL,
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
