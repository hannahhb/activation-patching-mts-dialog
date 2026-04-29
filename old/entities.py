"""
Clinical entity extraction for ACI-Bench encounters.

Primary NER: scispaCy ``en_ner_bc5cdr_md``
  CHEMICAL → "medication"   (drug names, dosage forms)
  DISEASE  → "diagnosis"    (diagnoses, symptoms, conditions)

Supplementary regex (types scispaCy does not label):
  Medication + dosage  e.g. "lisinopril 10 mg"     → "medication"
  Lab / imaging        e.g. "WBC 12.5", "no fracture" → "lab_result"
  Exam findings        e.g. "2/6 systolic murmur"   → "exam_finding"

The scispaCy model is lazy-loaded on first call. Install once with:
  pip install scispacy
  pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

Only entities reproduced (verbatim) in the generated note are returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ── Entity Type Registry ───────────────────────────────────────────────────────

ENTITY_TYPES = frozenset(
    {"medication", "diagnosis", "symptom", "exam_finding", "lab_result"}
)

# BC5CDR label → our entity type
_BC5CDR_LABEL_MAP = {
    "CHEMICAL": "medication",
    "DISEASE":  "diagnosis",
}

# ── scispaCy lazy loader ───────────────────────────────────────────────────────

_SCISPACY_MODEL = "en_ner_bc5cdr_md"
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load(_SCISPACY_MODEL)
        except OSError:
            raise RuntimeError(
                f"scispaCy model '{_SCISPACY_MODEL}' not found.\n"
                f"Install with:\n"
                f"  pip install scispacy\n"
                f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/"
                f"releases/v0.5.4/{_SCISPACY_MODEL}-0.5.4.tar.gz"
            )
    return _NLP


# ── Supplementary Regex ────────────────────────────────────────────────────────

_DOSE_UNITS = r"(?:mg|mcg|μg|g|ml|mL|mEq|mmol|units?|IU|tabs?|capsules?|puffs?)"

# Medication name + explicit dosage — supplements scispaCy with dosage info
_MED_DOSE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,2})"
    r"\s+(\d+(?:\.\d+)?)\s*" + _DOSE_UNITS,
    re.IGNORECASE,
)

# Lab / imaging result phrases
_LAB_RE = re.compile(
    r"\b("
    r"no\s+(?:fracture|dislocation|acute\s+\w+|significant\s+\w+|elevated?\s+\w+)|"
    r"(?:WBC|RBC|Hgb|Hct|BMP|CMP|CBC|BNP|TSH|HbA1c|eGFR|creatinine|glucose|"
    r"sodium|potassium|chloride|bicarbonate|BUN|ALT|AST|ALP|bilirubin|"
    r"troponin|INR|PT|PTT|proBNP)"
    r"(?:\s*[:\-]?\s*\d+\.?\d*\s*(?:[a-zA-Z/%]+)?)?"
    r")",
    re.IGNORECASE,
)

# Exam finding patterns (physical exam observations not covered by BC5CDR)
_EXAM_FINDING_RE = re.compile(
    r"\b("
    r"(?:\d\s*/\s*\d\s+)?(?:systolic|diastolic)(?:\s+ejection)?\s+murmur|"
    r"(?:heart\s+)?murmur|"
    r"regular\s+rate\s+and\s+rhythm|RRR\b|"
    r"clear\s+to\s+auscultation(?:\s+bilaterally)?|"
    r"(?:decreased|diminished|absent)\s+breath\s+sounds|"
    r"wheezing|rales|rhonchi|crackles|stridor|"
    r"(?:negative|positive)\s+straight\s+leg\s+raise|straight\s+leg\s+raise|"
    r"(?:CVA|costovertebral\s+angle)\s+tenderness|"
    r"(?:pitting|pedal|bilateral|lower\s+extremity)\s+edema|"
    r"jugular\s+venous\s+distension|JVD\b|"
    r"point\s+tenderness|rebound\s+tenderness|"
    r"hepatomegaly|splenomegaly|lymphadenopathy"
    r")",
    re.IGNORECASE,
)


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class ClinicalEntity:
    """
    A clinical entity detected in the dialogue that is also reproduced in
    the generated note.

    Token positions are absolute indices into the full ``[prompt + generated]``
    token sequence passed to TransformerLens.
    """
    text:        str
    entity_type: str   # one of ENTITY_TYPES

    # Character spans in the original strings
    source_char_start: int
    source_char_end:   int
    note_char_start:   int   # -1 if not found
    note_char_end:     int   # -1 if not found

    # Absolute token indices in full_token_ids
    source_token_positions: List[int] = field(default_factory=list)
    note_token_positions:   List[int] = field(default_factory=list)

    # Token ID of the first subword of the entity (DLA target)
    first_token_id: int = -1


# ── Internal helpers ───────────────────────────────────────────────────────────

def _find_subseq(needle: List[int], haystack: List[int]) -> List[int]:
    """
    Return all absolute positions in ``haystack`` covered by any occurrence
    of the token subsequence ``needle``.
    """
    n, m = len(haystack), len(needle)
    positions: List[int] = []
    for i in range(n - m + 1):
        if haystack[i: i + m] == needle:
            positions.extend(range(i, i + m))
    return positions


def _dedup_spans(
    spans: List[Tuple[int, int, str, str]],
) -> List[Tuple[int, int, str, str]]:
    """Remove overlapping spans, keeping the longest at each character offset."""
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out: List[Tuple[int, int, str, str]] = []
    prev_end = -1
    for s, e, text, etype in spans:
        if s >= prev_end:
            out.append((s, e, text, etype))
            prev_end = e
    return out


def _spans_from_regex(
    text: str,
    pattern: re.Pattern,
    entity_type: str,
) -> List[Tuple[int, int, str, str]]:
    return [
        (m.start(), m.end(), m.group().strip(), entity_type)
        for m in pattern.finditer(text)
    ]


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_entities(
    dialogue:       str,
    generated_note: str,
    tokenizer,
    full_token_ids: List[int],
    prompt_len:     int,
) -> List[ClinicalEntity]:
    """
    Extract clinical entities from the source dialogue and return only those
    also reproduced in the generated note.

    Uses scispaCy ``en_ner_bc5cdr_md`` for medication (CHEMICAL) and
    diagnosis/symptom (DISEASE) detection, with supplementary regex for
    lab results, exam findings, and dosage-annotated medications.

    Each returned ``ClinicalEntity`` carries absolute token positions for
    both the source (prompt) and generated (note) sides of the full sequence,
    enabling entity-level Direct Logit Attribution in ``entity_dla.py``.

    Parameters
    ----------
    dialogue:
        Raw doctor-patient dialogue string.
    generated_note:
        The model's generated SOAP note string.
    tokenizer:
        HuggingFace / TransformerLens tokenizer.
    full_token_ids:
        Flat list of token IDs for the ``[prompt + generated]`` sequence.
    prompt_len:
        Number of prompt tokens (boundary between source and generated).
    """
    nlp = _get_nlp()

    # ── 1. Collect candidate spans from the dialogue ───────────────────────────
    spans: List[Tuple[int, int, str, str]] = []

    # scispaCy NER: medications (CHEMICAL) and diagnoses/symptoms (DISEASE)
    doc = nlp(dialogue)
    for ent in doc.ents:
        etype = _BC5CDR_LABEL_MAP.get(ent.label_)
        if etype is None:
            continue
        spans.append((ent.start_char, ent.end_char, ent.text.strip(), etype))

    # Supplementary regex — dosage-annotated medications, labs, exam findings
    spans.extend(_spans_from_regex(dialogue, _MED_DOSE_RE,      "medication"))
    spans.extend(_spans_from_regex(dialogue, _LAB_RE,            "lab_result"))
    spans.extend(_spans_from_regex(dialogue, _EXAM_FINDING_RE,   "exam_finding"))

    spans = _dedup_spans(spans)

    # ── 2. Filter to entities reproduced in note; map to token positions ───────
    note_lower    = generated_note.lower()
    prompt_ids    = full_token_ids[:prompt_len]
    generated_ids = full_token_ids[prompt_len:]
    entities: List[ClinicalEntity] = []

    for src_start, src_end, entity_text, etype in spans:
        entity_lower = entity_text.lower().strip()
        note_idx = note_lower.find(entity_lower)
        if note_idx == -1:
            continue  # not reproduced in the generated note

        # Tokenize with and without leading space (handles BPE word boundaries)
        ids_space   = tokenizer.encode(" " + entity_text.strip(),
                                       add_special_tokens=False)
        ids_nospace = tokenizer.encode(entity_text.strip(),
                                       add_special_tokens=False)

        src_positions = _find_subseq(ids_space, prompt_ids)
        if not src_positions:
            src_positions = _find_subseq(ids_nospace, prompt_ids)

        gen_local = _find_subseq(ids_space, generated_ids)
        if gen_local:
            entity_ids = ids_space
        else:
            gen_local  = _find_subseq(ids_nospace, generated_ids)
            entity_ids = ids_nospace

        note_positions = [p + prompt_len for p in gen_local]
        if not note_positions:
            continue  # could not locate entity in token stream

        first_token_id = entity_ids[0] if entity_ids else -1

        entities.append(ClinicalEntity(
            text                   = entity_text.strip(),
            entity_type            = etype,
            source_char_start      = src_start,
            source_char_end        = src_end,
            note_char_start        = note_idx,
            note_char_end          = note_idx + len(entity_text),
            source_token_positions = src_positions,
            note_token_positions   = note_positions,
            first_token_id         = first_token_id,
        ))

    return entities
