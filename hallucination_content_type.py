"""
hallucination_content_type.py
==============================
Rule-based (no LLM) content-type classifier for erroneous note spans.

Buckets a `note_span` string (the verbatim unsupported/contradicted text
identified by the LLM judge, see llm_hallucination_label.py --mode span) into
one of six content types:
    condition | procedure | medication | numerical | name | word

This is a coarser cut of the same idea as annotator.py's 11-class human
taxonomy (Hegselmann et al. 2402.15422), restricted to the 6 categories that
can be reasonably distinguished with lexicons + regex rather than a trained
NER model or an LLM -- deliberately avoiding the LLM-judge circularity this
whole analysis is trying to get away from.

Method (fully offline, deterministic):
  - `condition` / `procedure` / `medication`: substring lexicon match (curated
    for ambulatory-care encounters, the ACI-bench / virtscribe domain) plus
    common clinical suffix patterns (e.g. "-itis" -> condition).
  - `numerical`: any digit run, once the above three have been ruled out.
  - `name`: a capitalized token not at the span's start, not a stopword, and
    not a known SOAP section header -- a heuristic proxy for a proper noun in
    the absence of a real NER model.
  - `word`: fallback for anything else, including template/placeholder
    artifacts (e.g. "[NOT MENTIONED]", "Vital signs", "Current medications" --
    section headers the model echoes verbatim, not real clinical content).

Priority when a span could match more than one bucket (mirrors annotator.py's
stated rule: "if unsure between types, prefer the one listed earlier"):
    condition > procedure > medication > numerical > name > word

This is a precision-oriented heuristic, not a substitute for human annotation
or a trained biomedical NER model -- treat its output as a useful large-n
approximation, not ground truth.
"""

from __future__ import annotations

import re
from typing import List

CONTENT_TYPES = ["condition", "procedure", "medication", "numerical", "name", "word"]

# ── SOAP / physical-exam section headers the model echoes verbatim ──────────
# (harvested from luq_out/llama/generations/**/*.json "notes" fields). A span
# that IS one of these (post strip/lower, minus trailing colon) is template
# artifact, not real clinical content -- always "word", checked first.
SECTION_HEADERS = frozenset({
    "subjective", "subjective continued", "objective", "assessment",
    "assessment / problem list", "plan", "hpi", "history of present illness",
    "past medical history", "review of systems", "physical exam",
    "physical examination", "vital signs", "current medications",
    "current medication", "problem list", "test results", "test result",
    "allergies", "social history", "family history", "chief complaint",
    "here is the soap note", "heart exam", "lung exam", "lower extremity exam",
    "head and neck", "abdomen", "extremities", "heart", "lungs",
    "temperature", "pulse", "pulse rate", "respirations", "respiratory rate",
    "blood pressure", "oxygen saturation", "weight", "jugular venous distention",
    "carotid bruits", "echo", "ekg", "ecg",
})

# ── Medication lexicon: common drug names/classes for ambulatory-care notes ──
MEDICATION_TERMS = frozenset({
    "tylenol", "acetaminophen", "advil", "motrin", "ibuprofen", "aspirin",
    "naproxen", "aleve", "amoxicillin", "augmentin", "azithromycin", "zithromax",
    "penicillin", "cephalexin", "ciprofloxacin", "doxycycline", "clindamycin",
    "metronidazole", "bactrim", "lisinopril", "losartan", "amlodipine",
    "metoprolol", "atenolol", "carvedilol", "hydrochlorothiazide", "furosemide",
    "lasix", "atorvastatin", "lipitor", "simvastatin", "rosuvastatin", "crestor",
    "metformin", "glipizide", "insulin", "lantus", "humalog", "januvia",
    "albuterol", "ventolin", "proair", "symbicort", "advair", "singulair",
    "montelukast", "flonase", "fluticasone", "prednisone", "prednisolone",
    "omeprazole", "prilosec", "pantoprazole", "ranitidine", "famotidine",
    "pepcid", "gabapentin", "neurontin", "duloxetine", "cymbalta",
    "sertraline", "zoloft", "fluoxetine", "prozac", "escitalopram", "lexapro",
    "citalopram", "bupropion", "wellbutrin", "trazodone", "alprazolam", "xanax",
    "lorazepam", "ativan", "diazepam", "valium", "clonazepam", "klonopin",
    "adderall", "ritalin", "methylphenidate", "amphetamine", "warfarin",
    "coumadin", "eliquis", "apixaban", "xarelto", "rivaroxaban", "plavix",
    "clopidogrel", "levothyroxine", "synthroid", "benadryl", "diphenhydramine",
    "claritin", "loratadine", "zyrtec", "cetirizine", "oxycodone", "percocet",
    "hydrocodone", "vicodin", "norco", "morphine", "methadone", "naltrexone",
    "buprenorphine", "suboxone", "tramadol", "methotrexate", "hydroxychloroquine",
    "plaquenil", "nsaid", "nsaids", "statin",
    "opioid", "opioids", "antibiotic", "antibiotics", "steroid", "steroids",
    "beta blocker", "ace inhibitor",
})
# Common drug-name suffixes (checked on individual tokens, case-insensitive).
MEDICATION_SUFFIXES = (
    "cillin", "mycin", "oxacin", "azole", "statin", "olol", "pril", "sartan",
    "zepam", "zolam", "profen", "codone", "dipine", "tidine", "prazole",
    "triptan", "barbital",
)

# ── Condition lexicon: common ambulatory-care symptoms / diagnoses ──────────
CONDITION_TERMS = frozenset({
    "fever", "cough", "shortness of breath", "dyspnea", "chest pain",
    "headache", "migraine", "nausea", "vomiting", "diarrhea", "constipation",
    "fatigue", "dizziness", "vertigo", "rash", "swelling", "edema", "numbness",
    "tingling", "weakness", "fracture", "sprain", "strain", "infection",
    "pneumonia", "bronchitis", "asthma", "copd", "diabetes", "hypertension",
    "hypotension", "arrhythmia", "anxiety", "depression", "insomnia",
    "obesity", "arthritis", "osteoarthritis", "rheumatoid arthritis",
    "concussion", "sinusitis", "uti", "urinary tract infection", "otitis",
    "conjunctivitis", "dermatitis", "eczema", "psoriasis", "cancer", "tumor",
    "stroke", "seizure", "epilepsy", "withdrawal", "addiction", "macromastia",
    "bruising", "nosebleed", "sore throat", "ear infection", "pink eye",
    "reflux", "gerd", "ulcer", "anemia", "palpitations", "murmur",
    "hyperlipidemia", "hypothyroidism", "hyperthyroidism", "gout", "sciatica",
})
CONDITION_SUFFIXES = ("itis", "osis", "emia", "algia", "opathy", "oma", "plegia")

# ── Procedure lexicon: common diagnostics / interventions ───────────────────
PROCEDURE_TERMS = frozenset({
    "x-ray", "xray", "ct scan", "mri", "ultrasound", "biopsy", "ekg", "ecg",
    "echocardiogram", "colonoscopy", "endoscopy", "surgery", "intubation",
    "catheter", "injection", "vaccination", "vaccine", "immunization", "physical therapy",
    "skin testing", "blood test", "blood work", "lab work", "labs",
    "screening", "mammogram", "pap smear", "dialysis", "chemotherapy",
    "radiation", "transplant", "amputation", "incision", "suture", "splint",
    "cast", "sling", "stitches", "vasectomy", "circumcision", "stent",
    "angioplasty", "pacemaker", "bypass",
})
PROCEDURE_SUFFIXES = ("ectomy", "otomy", "oscopy", "ostomy", "plasty", "ography", "centesis")

# Words that must not be misread as a proper noun by the `name` heuristic.
_NAME_STOPWORDS = frozenset({
    "the", "he", "she", "his", "her", "him", "they", "their", "i", "a", "an",
    "this", "that", "these", "those", "there", "here", "current", "review",
    "past", "test", "vital", "physical", "problem", "no", "yes", "not",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
})

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d[\d./%-]*")


def _tokens(span: str) -> List[str]:
    return _TOKEN_RE.findall(span)


def _lexicon_hit(span_lower: str, terms: frozenset, suffixes: tuple, words: List[str]) -> bool:
    if any(term in span_lower for term in terms):
        return True
    return any(len(w) > len(sfx) and w.lower().endswith(sfx) for w in words for sfx in suffixes)


def classify_content_type(span: str) -> str:
    """Classify a note_span string into one of CONTENT_TYPES. See module
    docstring for method and priority order."""
    raw = str(span).strip()
    if not raw:
        return "word"

    span_lower = raw.lower()
    if not any(c.isalnum() for c in raw):
        return "word"
    # Template/placeholder artifacts (the model echoing an unfilled section
    # header, a "[NOT MENTIONED]"/"[Unknown]" placeholder, or a bracketed
    # slot like "[Condition] [Status: ...]") are not real clinical content --
    # always "word", checked before any lexicon match.
    if "not mentioned" in span_lower or "[unknown]" in span_lower:
        return "word"
    header_part = raw.split(":")[0].strip().strip("[]").strip().lower()
    if header_part in SECTION_HEADERS:
        return "word"
    if re.sub(r"\[[^\]]*\]", "", raw).strip(" :;,.") == "":
        return "word"
    words = _tokens(raw)

    if _lexicon_hit(span_lower, CONDITION_TERMS, CONDITION_SUFFIXES, words):
        return "condition"
    if _lexicon_hit(span_lower, PROCEDURE_TERMS, PROCEDURE_SUFFIXES, words):
        return "procedure"
    if _lexicon_hit(span_lower, MEDICATION_TERMS, MEDICATION_SUFFIXES, words):
        return "medication"
    if any(ch.isdigit() for ch in raw):
        return "numerical"

    for idx, w in enumerate(words):
        if idx == 0:
            continue  # skip span-initial word: capitalization there is just sentence case
        # Title Case only (e.g. "Charles", "James") -- excludes ALL-CAPS clinical
        # abbreviations (EMG, PCP, NCV) and shouted emphasis (NO, MENTION), which
        # are not proper nouns.
        if w[0].isupper() and w[1:].islower() and \
           w.lower() not in _NAME_STOPWORDS and w.lower() not in SECTION_HEADERS:
            return "name"

    return "word"


if __name__ == "__main__":
    _tests = [
        "active rheumatoid arthritis", "Pulse 70", "124/76",
        "a 30-year-old male", "second naltrexone injection",
        "I prescribed the patient to continue taking Tylenol",
        "Current medications: [NOT MENTIONED]", "[Unknown]",
        "to schedule the skin testing", "Charles reported a fever",
        "suspected concussion", "2 puffs every 4-6 hours as needed",
    ]
    for t in _tests:
        print(f"{classify_content_type(t):12s}  {t!r}")
