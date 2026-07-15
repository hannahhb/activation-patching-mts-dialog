"""
atomic_luq.py
=============
LUQ-ATOMIC: atomic-fact-level uncertainty scoring across K generated notes.

For each sample:
  1. Load all K=10 notes; parse raw sections for every note.
  2. Decompose only the first DECOMP_K=3 notes into atomic facts via LLM
     (cached to disk).
  3. For each fact in a decomposed note, score it against the raw section
     text of every other note (K-1 = 9 reference notes) using NLI:
       - premise  = sliding windows over the reference section text
       - hypothesis = atomic fact
     Soft entailment probability: softmax(logits)[entail_idx], max over
     windows, then averaged across the 9 reference notes.
  4. uncertainty(fact) = 1 - mean_entail_prob
  5. Save per-note CSVs: fact_idx, section, fact, uncertainty
     → <out>/facts/sample_NNN_note_KK_facts.csv

Usage:
    python atomic_luq.py
    python atomic_luq.py --start 0 --end 44
    python atomic_luq.py --gen-dir luq_out/llama/generations --out luq_out/llama_atomic
    python atomic_luq.py --no-cache
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import factmatch_sentence as fm
import luq_sentence as luq
from llm_client import gptoss_yesno, GPTOSS_MAX_WORKERS, tracker as llm_tracker, get_llm
from redeep_word_plots import find_span_char_range

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_K         = 10   # total notes in the NLI reference pool
DEFAULT_DECOMP_K  = 3    # how many notes to decompose into atomic facts
DEFAULT_GEN_DIR   = "luq_out/llama/generations"
DEFAULT_OUT_DIR   = "luq_out/llama_atomic"

# If the same-section entailment probability falls below this, also check
# the reference note's OTHER sections before concluding "unsupported" --
# guards against LLM generations inconsistently filing the same fact under
# different SOAP sections across resamples (see score_sample Step 3).
SECTION_FALLBACK_THRESHOLD = 0.5

# Section groupings — first match wins (checked against lowercased header lines)
SECTION_PATTERNS: List[Tuple[str, str]] = [
    ("subjective",  r"^subjective|^hpi|^history of present|^review of systems|^ros"
                    r"|^past medical|^pmh|^social history|^family history"
                    r"|^current medications|^medications|^allergies"),
    ("objective",   r"^objective|^vital|^physical exam|^pe\b|^test results"
                    r"|^labs|^imaging|^results"),
    ("assessment",  r"^assessment|^problem list|^impression|^diagnosis"),
    ("plan",        r"^plan|^follow.?up|^orders|^disposition"),
]
SECTION_NAMES = [s for s, _ in SECTION_PATTERNS]
OTHER_SECTION = "other"   # preamble / unmatched lines


# ── Section parser ─────────────────────────────────────────────────────────────

def parse_sections(note: str) -> Dict[str, str]:
    """Split a note into section text blocks keyed by section name."""
    blocks: Dict[str, List[str]] = {s: [] for s in SECTION_NAMES}
    blocks[OTHER_SECTION] = []
    current = OTHER_SECTION

    for line in note.split("\n"):
        header = line.lower().strip().rstrip(":").rstrip()
        matched = False
        if len(header) < 50:   # only short lines can be headers
            for sec, pat in SECTION_PATTERNS:
                if re.match(pat, header):
                    current = sec
                    matched = True
                    break
        if not matched:
            blocks[current].append(line)

    return {sec: "\n".join(lines).strip() for sec, lines in blocks.items()}


# ── Decomposition ──────────────────────────────────────────────────────────────

def decompose_section(text: str) -> List[str]:
    if not text.strip():
        return []
    raw = fm._llm_extract(fm._NOTE_PROMPT.format(note=text.strip()))
    return fm.filter_facts(raw)


# ── Decomposition with span-level origin (--decomp-mode span) ──────────────────
# Second decomposition option, opt-in via --decomp-mode span: same task as
# decompose_section() above, but the LLM also reports each fact's verbatim
# source span within the section text, in one call, instead of the two-stage
# decompose-then-locate-heuristically approach fact_sentence_match.py uses.
# Every returned span is verified as a real substring before being trusted
# (see demo_fact_span_decompose.py for the standalone demo + rationale) --
# an LLM-reported span that doesn't literally appear in the section text is
# kept with span_verified=False rather than silently discarded, so callers
# can see and filter hallucinated spans themselves.
#
# Deliberately does NOT touch decompose_section() or its prompt/cache format
# above -- this is a fully separate code path (own prompt, own cache
# filename via decomp_cache stem below, own CSV columns) so existing
# plain-mode caches/outputs are completely unaffected.

_SPAN_SYSTEM_PROMPT = (
    "You are a meticulous physician extracting atomic clinical facts from a "
    "SOAP note section, with each fact's exact textual origin. Return ONLY "
    "valid JSON — no explanation, no preamble, no markdown code fences."
)

_SPAN_USER_TEMPLATE = """\
Extract every clinical fact from the note section below as a JSON array. For \
each fact, also give its "spans": a JSON list of one or more exact minimal \
substrings of the text (copy-pasted verbatim, not paraphrased) that \
together support the fact.

VERBATIM RULE (the rule most often broken — follow it exactly):
- Every string in "spans" MUST be an EXACT, character-for-character, \
case-preserving substring of the text below. Copy it, do not retype or \
improve it.
- Do NOT upgrade informal or colloquial wording into clinical phrasing for \
the span. If the source text says "seems to make it worse", the span must \
contain that exact wording — NOT a rewritten version like "exacerbated by".
    WRONG:  fact: "The patient's pain is exacerbated by walking."
            spans: ["exacerbated by walking"]        <- rewritten, not in the text
    RIGHT:  fact: "The patient's pain is exacerbated by walking."
            spans: ["walking seems to make it worse"] <- copied verbatim
- Before writing a span, check it word-for-word against the text below. If \
you cannot find an exact match, search again for the closest literal \
wording in the text and use that — never invent clinical-sounding text.

MULTIPLE-SPANS RULE:
- Almost every fact has exactly ONE span: "spans": ["<one string>"].
- A fact split off a shared local clause (e.g. "X and Y were normal" split \
into a fact about X and a fact about Y) still has ONE span — give BOTH \
facts the SAME full shared span. This is NOT a multi-span case: the \
evidence is local and contiguous, just shared between two split facts.
- Only use TWO OR MORE entries in "spans" when the fact genuinely requires \
evidence from separate, non-adjacent parts of the text (e.g. a diagnosis \
combining a symptom mentioned in one sentence with a lab value mentioned \
several sentences later). List each supporting substring separately.

Other rules:
- Split conjunctions: "denied X and Y" -> two separate facts, one for X and \
one for Y.
- Split medications: drug name, dose, frequency, indication as separate \
facts (repeat the drug name in each). Each fact's span is the specific \
phrase that supports IT alone, not the whole medication sentence (e.g. the \
dose fact's span is "10 mg", not the full sentence).
- Include all clinical findings, diagnoses, plans, and demographics.

Return this JSON structure and nothing else:
[
  {{"fact": "<atomic fact as a full sentence>", "spans": ["<exact verbatim substring>", ...]}},
  ...
]

Example 1 — single local span shared across a conjunction split:
Text: "Chest X-ray and pulmonary function test were normal."
Output:
[
  {{"fact": "The patient's chest X-ray is normal.", "spans": ["Chest X-ray and pulmonary function test were normal"]}},
  {{"fact": "The patient's pulmonary function test is normal.", "spans": ["Chest X-ray and pulmonary function test were normal"]}}
]

Example 2 — verbatim, not paraphrased (keep the source's own words):
Text: "The patient reports standing and walking seems to make the pain worse; coughing and sneezing make it worse too."
Output:
[
  {{"fact": "The patient's pain is exacerbated by standing.", "spans": ["standing and walking seems to make the pain worse"]}},
  {{"fact": "The patient's pain is exacerbated by walking.", "spans": ["standing and walking seems to make the pain worse"]}},
  {{"fact": "The patient's pain is exacerbated by coughing.", "spans": ["coughing and sneezing make it worse"]}},
  {{"fact": "The patient's pain is exacerbated by sneezing.", "spans": ["coughing and sneezing make it worse"]}}
]

Example 3 — genuinely disjoint spans (two separate, non-adjacent parts):
Text: "HPI: The patient reports chest pain radiating to the left arm. ... Assessment: Troponin returned at 0.8, consistent with myocardial infarction."
Output:
[
  {{"fact": "The patient has chest pain radiating to the left arm.", "spans": ["chest pain radiating to the left arm"]}},
  {{"fact": "The patient's troponin is elevated at 0.8, consistent with myocardial infarction.", "spans": ["Troponin returned at 0.8", "consistent with myocardial infarction"]}}
]

Text:
\"\"\"
{note}
\"\"\"

Output:\
"""


def _parse_span_json(raw: str) -> Optional[List[dict]]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _extract_spans(item: dict) -> List[str]:
    """Read the "spans" list, with a defensive fallback to a legacy/singular
    "span" string key in case the model doesn't follow the schema exactly."""
    spans = item.get("spans")
    if isinstance(spans, list):
        return [str(s).strip() for s in spans if str(s).strip()]
    single = item.get("span")
    return [str(single).strip()] if single and str(single).strip() else []


# ─────────────────────────────────────────────────────────────────────────────
# Multi-tier span location: exact -> word-level -> semantic
#
# The stricter prompt (VERBATIM RULE, WRONG/RIGHT example) cuts how often the
# LLM's span isn't a real substring, but doesn't eliminate it -- the model
# still occasionally normalises tense/casing beyond what find_span_char_range
# tolerates, or paraphrases a word (e.g. "worse" -> "exacerbated"). Rather
# than just discarding those as span_verified=False, retry with two
# progressively looser (and progressively coarser-grained) matchers before
# giving up, and record WHICH tier found it so callers can weight confidence.
# ─────────────────────────────────────────────────────────────────────────────

_WORD_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "if", "then",
    "than", "as", "that", "this", "these", "those", "there", "here",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "who", "whom",
    "whose", "which", "what", "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "to", "of", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "out", "off", "over",
    "under", "again", "further", "not", "no", "own", "same", "just", "also",
    # Near-universal clinical-note words that appear in almost every fact
    # regardless of content, so a match on these alone is not meaningful
    # signal -- same list fact_sentence_match.py's _LEMMA_STOP excludes for
    # the identical reason (there, it caused a fact about "denied X" to
    # spuriously match on "denied" instead of X; here it would let an
    # UNRELATED query match on "patient" alone).
    "patient", "patients", "doctor", "symptom", "symptoms", "report",
    "reports", "reported", "present", "presents", "presented", "mention",
    "mentioned", "state", "stated", "current", "medication", "medications",
    "history", "review", "system", "assessment", "problem", "list",
    "objective", "subjective", "plan", "follow", "followup", "test",
    "result", "results", "physical", "exam", "vital", "signs", "sign",
    "issue", "deny", "denied", "denies", "note", "noted", "admit",
    "admitted", "endorse", "endorsed", "complain", "complained",
})

# Minimum matched content-word run required to accept a tier-2 match. A
# single incidental overlapping word (e.g. "patient") is not trustworthy
# evidence on its own -- require 2, unless the query itself only HAS 1
# content word to begin with (then that's the best achievable).
_MIN_WORD_MATCH = 2


def _content_words(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [w for w in words if w not in _WORD_STOPWORDS and len(w) > 1]


def _find_word_level_span(text: str, query: str) -> Optional[Tuple[int, int]]:
    """Fallback tier 2: longest contiguous run of `text` tokens whose
    lowercase forms are content words of `query` (the LLM's span guess),
    bridging up to 2 connector words but never crossing a clause/list-item
    boundary (;/./:). Same algorithm validated in fact_sentence_match.py's
    _locate_span, reimplemented self-contained here (no scispaCy) since
    `query` is a free-text LLM guess, not NER-tagged fact/sentence info.
    Requires at least _MIN_WORD_MATCH matched words (see above) to avoid
    accepting a spurious single-generic-word coincidence."""
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[a-zA-Z0-9']+", text)]
    query_words = set(_content_words(query))
    if not query_words or not tokens:
        return None
    min_required = min(_MIN_WORD_MATCH, len(query_words))

    hits = [tok.lower() in query_words for tok, _, _ in tokens]
    n = len(tokens)
    GAP_TOLERANCE = 2
    HARD_BOUNDARY = re.compile(r"[;.:]")

    def _crosses_boundary(a: int, b: int) -> bool:
        return bool(HARD_BOUNDARY.search(text[tokens[a][2]:tokens[b][1]]))

    best: Optional[Tuple[int, int, int]] = None  # (run_len, start_idx, end_idx)
    i = 0
    while i < n:
        if not hits[i]:
            i += 1
            continue
        j = i
        while True:
            next_hit = None
            for g in range(1, GAP_TOLERANCE + 2):
                if j + g < n and hits[j + g] and not _crosses_boundary(j, j + g):
                    next_hit = j + g
                    break
            if next_hit is None:
                break
            j = next_hit
        run_len = sum(1 for k in range(i, j + 1) if hits[k])
        if best is None or run_len > best[0]:
            best = (run_len, i, j)
        i = j + 1

    if best is None:
        return None
    run_len, start_idx, end_idx = best
    if run_len < min_required:
        return None
    while end_idx > start_idx and not hits[end_idx]:
        end_idx -= 1
    while start_idx < end_idx and not hits[start_idx]:
        start_idx += 1
    return tokens[start_idx][1], tokens[end_idx][2]


# Min cosine sim to accept a tier-3 clause match. Calibrated empirically,
# not theoretically: for SHORT clinical clauses, pritamdeka/S-PubMedBert-MS-MARCO
# cosine similarities don't spread out as cleanly as one might hope -- a
# genuinely unrelated query ("denied fever or chills" vs. a walking/pain
# clause) still scored 0.87, while true paraphrases scored 0.90-0.99. 0.90
# is the empirical cut that separated them in that test; there is no
# guarantee it generalises perfectly, which is exactly why every span
# carries its span_match_methods entry -- a "semantic" match is inherently
# the least reliable of the three tiers and downstream consumers that need
# high precision should filter to "exact"/"word" only.
_SEMANTIC_MATCH_FLOOR = 0.90


def _split_clauses(text: str) -> List[str]:
    """luq.split_sentences() only breaks on '.', so a semicolon-joined
    compound sentence ("X worse; Y worse too.") comes back as ONE candidate
    -- which trivially "wins" any similarity comparison since there's
    nothing to compare it against, defeating the whole purpose of tier 3.
    Pre-split on ';' (a clause boundary the rest of this file already treats
    as a hard boundary, e.g. _find_word_level_span's HARD_BOUNDARY) before
    handing each piece to the real sentence splitter."""
    clauses: List[str] = []
    for chunk in text.split(";"):
        clauses.extend(luq.split_sentences(chunk))
    return clauses


def _find_semantic_span(text: str, query: str, floor: float = _SEMANTIC_MATCH_FLOOR
                        ) -> Optional[Tuple[int, int]]:
    """Fallback tier 3 (last resort): split `text` into sentences/clauses,
    embed each plus `query` with the project's biomedical sentence encoder
    (same model factmatch_sentence.py uses for fact dedup), and return the
    char range of the highest-cosine candidate if it clears `floor`. Coarser
    than tier 2 (clause/sentence granularity, not a minimal phrase) but
    catches genuine paraphrases with near-zero literal word overlap."""
    sentences = _split_clauses(text)
    if not sentences:
        return None
    if len(sentences) == 1:
        # Only one candidate to compare against -- any query would trivially
        # "win", which isn't a meaningful match. Refuse rather than return
        # the whole text as a fake precise span.
        return None
    embs = fm.embed_facts(sentences + [query])  # L2-normalised, so dot = cosine
    sent_embs, query_emb = embs[:-1], embs[-1]
    sims = sent_embs @ query_emb
    best_idx = int(np.argmax(sims))
    if sims[best_idx] < floor:
        return None
    best_sentence = sentences[best_idx]
    pos = text.find(best_sentence)
    if pos != -1:
        return pos, pos + len(best_sentence)
    return find_span_char_range(text, best_sentence)  # whitespace/case fallback


def _locate_span_multi_tier(text: str, query: str) -> Tuple[Optional[Tuple[int, int]], str]:
    """Try exact -> word-level -> semantic, in that order (cheapest and most
    precise first). Returns ((start, end) or None, method), method one of
    "exact" / "word" / "semantic" / "unmatched"."""
    located = find_span_char_range(text, query)
    if located is not None:
        return located, "exact"
    located = _find_word_level_span(text, query)
    if located is not None:
        return located, "word"
    located = _find_semantic_span(text, query)
    if located is not None:
        return located, "semantic"
    return None, "unmatched"


def decompose_section_with_spans(text: str) -> List[dict]:
    """Same decomposition task as decompose_section(), returning dicts with
    span-level provenance instead of plain fact strings. Each dict:
      fact, spans (list of verbatim strings as the LLM returned them),
      span_offsets (list of (start, end) or None per entry in `spans`, None
        only if ALL THREE location tiers failed for that span),
      span_match_methods (list of "exact"/"word"/"semantic"/"unmatched",
        one per entry in `spans` -- see _locate_span_multi_tier),
      span_verified (bool -- True only if EVERY span in `spans` was located
        by some tier; False means at least one is "unmatched" and the
        fact's provenance should not be fully trusted for highlighting),
      synthesized (bool, computed as len(spans) > 1 -- NOT self-reported by
        the LLM: an earlier version trusted a model-authored "synthesized"
        flag and it was unreliable, e.g. flagging a fact as disjoint when it
        just shared one local span with a sibling fact from a conjunction
        split. Deriving it from the verified span count is deterministic.)"""
    if not text.strip():
        return []
    user_msg = _SPAN_USER_TEMPLATE.format(note=text.strip())
    try:
        resp = get_llm().converse(
            stage="decomp_span", model_id=fm.BEDROCK_GEN_MODEL,
            system=[{"text": _SPAN_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            inference_config={"maxTokens": 2048, "temperature": 0.0},
        )
    except Exception:
        return []
    raw = resp["output"]["message"]["content"][0]["text"]
    parsed = _parse_span_json(raw)
    if parsed is None:
        return []

    results = []
    for item in parsed:
        fact = str(item.get("fact", "")).strip()
        if not fact or len(fact) <= 4:
            continue
        spans = _extract_spans(item)
        if not spans:
            continue
        located = [_locate_span_multi_tier(text, s) for s in spans]
        offsets = [loc for loc, _ in located]
        methods = [method for _, method in located]
        results.append({
            "fact": fact,
            "spans": spans,
            "span_offsets": offsets,
            "span_match_methods": methods,
            "span_verified": all(o is not None for o in offsets),
            "synthesized": len(spans) > 1,
        })
    return results


# Column names shared by score_sample() and score_sample_gptoss() when
# decomp_mode == "span". spans/span_offsets/span_match_methods are
# JSON-encoded (a CSV cell can't hold a Python list directly); span_offsets
# entries are [start, end] or null per corresponding entry in spans;
# span_match_methods entries are "exact"/"word"/"semantic"/"unmatched".
SPAN_CSV_COLUMNS = ["spans", "span_offsets", "span_match_methods", "span_verified", "synthesized"]


def _span_row_fields(d: dict) -> dict:
    """Build the SPAN_CSV_COLUMNS fields for one facts-CSV row from a
    decompose_section_with_spans() item."""
    return {
        "spans":              json.dumps(d.get("spans", [])),
        "span_offsets":       json.dumps(d.get("span_offsets", [])),
        "span_match_methods": json.dumps(d.get("span_match_methods", [])),
        "span_verified":      d.get("span_verified"),
        "synthesized":        d.get("synthesized"),
    }


def dedupe_across_sections_spans(note_decomp: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Span-aware analogue of dedupe_across_sections() -- exact-match dedup
    on the fact text, keeping the first (section, span-info) occurrence."""
    flat, flat_secs = [], []
    for sec in SECTION_NAMES:
        for d in note_decomp.get(sec, []):
            flat.append(d)
            flat_secs.append(sec)
    if len(flat) < 2:
        return note_decomp

    seen: set = set()
    out: Dict[str, List[dict]] = {sec: [] for sec in SECTION_NAMES}
    for d, sec in zip(flat, flat_secs):
        key = d["fact"].strip()
        if key in seen:
            continue
        seen.add(key)
        out[sec].append(d)
    return out


def dedupe_across_sections(note_decomp: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Facts are decomposed per SOAP section independently (decompose_section
    is called once per section with no visibility into the others), so the
    same real-world fact occasionally gets extracted VERBATIM in two
    sections -- e.g. "The patient has diabetes." appearing both in the
    assessment problem list and again in the plan's opening line.

    Dedup by EXACT string match only, not embedding cosine similarity.
    Cosine dedup (via factmatch_sentence.deduplicate_facts, threshold 0.92)
    was tried and reverted: pritamdeka/S-PubMedBert-MS-MARCO scores
    same-topic-but-different-content clinical sentences in the same
    0.89-0.94 range as genuine paraphrase duplicates -- confirmed on a real
    note, "The patient presents with a 4-day history of worsening right
    elbow pain." scored 0.9358 against "The patient's right elbow pain is
    exacerbated by activities such as pottery and ceramics.", above the
    0.92 cutoff, despite being two entirely distinct facts (duration vs.
    aggravating trigger) -- the trigger fact was silently dropped with no
    surviving equivalent. Exact-match can only remove true verbatim
    repeats, so it can't produce that kind of false merge."""
    flat_facts, flat_secs = [], []
    for sec in SECTION_NAMES:
        for f in note_decomp.get(sec, []):
            flat_facts.append(f)
            flat_secs.append(sec)
    if len(flat_facts) < 2:
        return note_decomp

    seen: set = set()
    out = {sec: [] for sec in SECTION_NAMES}
    for f, sec in zip(flat_facts, flat_secs):
        key = f.strip()
        if key in seen:
            continue
        seen.add(key)
        out[sec].append(f)
    return out


# ── Sliding window NLI ────────────────────────────────────────────────────────

def _max_entail_prob(nli, premise_windows: List[str], hypothesis: str, entail_idx: int) -> float:
    """Softmax entailment probability for a fact against a section, max over windows."""
    from scipy.special import softmax as sp_softmax
    pairs = [(w, hypothesis) for w in premise_windows]
    raw = np.asarray(
        nli.predict(pairs,
                    batch_size=len(pairs),
                    apply_softmax=False,
                    show_progress_bar=False)
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    probs = sp_softmax(raw, axis=1)          # (n_windows, n_labels)
    return float(probs[:, entail_idx].max()) # max entailment prob across windows


def _max_entail_probs_batch(
    nli,
    premise_windows: List[str],
    hypotheses: List[str],
    entail_idx: int,
) -> np.ndarray:
    """Max entailment probability for many facts against one section's windows.

    This keeps the scoring logic identical to repeated _max_entail_prob calls,
    but collapses thousands of tiny CrossEncoder.predict() invocations into
    larger batched calls that actually use the GPU.
    """
    from scipy.special import softmax as sp_softmax

    if not premise_windows or not hypotheses:
        return np.zeros(len(hypotheses), dtype=np.float64)

    n_w = len(premise_windows)
    n_h = len(hypotheses)
    pairs = [(w, h) for h in hypotheses for w in premise_windows]
    raw = np.asarray(
        nli.predict(
            pairs,
            batch_size=min(len(pairs), max(luq.MICRO_BATCH_PAIRS, luq.NLI_BATCH_SIZE)),
            apply_softmax=False,
            show_progress_bar=False,
        )
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    probs = sp_softmax(raw, axis=1)[:, entail_idx].reshape(n_h, n_w)
    return probs.max(axis=1).astype(np.float64)


# ── Paths ──────────────────────────────────────────────────────────────────────

def facts_csv_path(out_dir: Path, sample_idx: int, note_idx: int) -> Path:
    return out_dir / "facts" / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts.csv"


# ── Scoring ────────────────────────────────────────────────────────────────────

def _prepare_decomp(sample_idx: int, notes: List[str], decomp_k: int,
                    out_dir: Path, use_cache: bool, decomp_mode: str = "plain",
                    ) -> Tuple[List[Dict[str, str]], List[Dict[str, List]]]:
    """Steps 1-2 shared by every scoring backend: parse raw sections for all
    K notes, then decompose the first decomp_k of them into atomic facts
    (LLM, cached to disk).

    decomp_mode:
      "plain" (default) -- decompose_section(), note_decomp[sec] is a
        List[str] of fact text, exactly as before this option existed.
      "span" -- decompose_section_with_spans(), note_decomp[sec] is a
        List[dict] with fact/span/span_start/span_end/span_verified/
        synthesized. Uses a separate decomp_cache filename (_decomp_span vs
        _decomp) so it can never collide with or corrupt an existing
        plain-mode cache."""
    facts_dir = out_dir / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: parse raw sections for all K notes ───────────────────────────
    # all_sections[note_idx][sec] = raw section text (used as NLI premise)
    all_sections: List[Dict[str, str]] = [parse_sections(note) for note in notes]

    # ── Step 2: decompose first decomp_k notes into atomic facts (LLM, cached)
    # Cache is a per-sample list covering however many notes were decomposed
    # the LAST time this ran. If decomp_k grows between runs (e.g. someone
    # reruns with a larger --atomic-decomp-k), only decompose the newly
    # requested notes rather than trusting a shorter cached list as-is --
    # that used to IndexError on the notes past the old cache's length.
    cache_stem = "decomp_span" if decomp_mode == "span" else "decomp"
    decomp_cache = out_dir / "facts" / f"sample_{sample_idx:03d}_{cache_stem}.json"

    all_decomp = []
    if use_cache and decomp_cache.exists():
        all_decomp = json.loads(decomp_cache.read_text())
        tqdm.write(f"  [cache] decomp sample {sample_idx}: {len(all_decomp)}/{decomp_k} notes cached")

    if len(all_decomp) < decomp_k:
        for note_idx in range(len(all_decomp), decomp_k):
            note_decomp = {}
            for sec in SECTION_NAMES:
                text = all_sections[note_idx].get(sec, "")
                if not text:
                    facts = []
                elif decomp_mode == "span":
                    facts = decompose_section_with_spans(text)
                else:
                    facts = decompose_section(text)
                note_decomp[sec] = facts
                if facts:
                    tqdm.write(f"    note {note_idx} [{sec}]: {len(facts)} facts")
            n_before = sum(len(v) for v in note_decomp.values())
            if decomp_mode == "span":
                note_decomp = dedupe_across_sections_spans(note_decomp)
            else:
                note_decomp = dedupe_across_sections(note_decomp)
            n_after = sum(len(v) for v in note_decomp.values())
            if n_after < n_before:
                tqdm.write(f"    note {note_idx}: cross-section dedup {n_before} -> {n_after} facts")
            all_decomp.append(note_decomp)
        decomp_cache.write_text(json.dumps(all_decomp, indent=2))

    return all_sections, all_decomp


def score_sample(sample_idx: int, notes: List[str], decomp_k: int,
                 out_dir: Path, use_cache: bool, decomp_mode: str = "plain") -> List[dict]:
    """
    notes     : all K notes (default 10) — used as NLI reference pool.
    decomp_k  : first decomp_k notes are decomposed; their facts are scored
                against the raw section text of every other note (K-1 refs).
    uncertainty = 1 - mean(softmax entailment prob over K-1 reference notes)
    decomp_mode: "plain" (default) or "span" -- see _prepare_decomp. In
                 "span" mode the output CSV gains span/span_start/span_end/
                 span_verified/synthesized columns alongside the existing
                 fact_idx/section/fact/uncertainty ones.
    """
    K = len(notes)
    all_sections, all_decomp = _prepare_decomp(sample_idx, notes, decomp_k, out_dir, use_cache,
                                               decomp_mode=decomp_mode)

    # ── Step 3: NLI scoring ──────────────────────────────────────────────────
    # For each fact in a decomposed note:
    #   premise    = sliding windows over the matching section of each reference note
    #   hypothesis = atomic fact
    #   score      = max softmax entailment prob across windows (per ref note)
    #   uncertainty = 1 - mean(score over K-1 reference notes)
    nli = luq.get_nli()
    entail_idx = luq._label_indices[0]
    tokenizer = nli.tokenizer

    # Precompute windowed premises once per note/section. This used to be
    # recomputed inside the innermost fact loop, which dominated runtime even
    # before NLI inference.
    windows_cache: List[Dict[str, List[str]]] = []
    for sec_map in all_sections:
        sec_windows = {}
        for sec in SECTION_NAMES:
            text = sec_map.get(sec, "").strip()
            sec_windows[sec] = luq.sentence_windows(text, tokenizer=tokenizer) if text else []
        windows_cache.append(sec_windows)

    rows = []
    for note_idx in range(decomp_k):
        csv_p = facts_csv_path(out_dir, sample_idx, note_idx)
        if use_cache and csv_p.exists():
            df = pd.read_csv(csv_p)
            tqdm.write(f"  [cache] scores sample {sample_idx} note {note_idx}")
            rows.extend(df.to_dict("records"))
            continue

        ref_indices = [j for j in range(K) if j != note_idx]
        note_rows = []

        for sec in SECTION_NAMES:
            my_items = all_decomp[note_idx].get(sec, [])
            if not my_items:
                continue
            # NLI always needs plain fact text; span mode stores dicts, so
            # unpack the text for scoring but keep my_items around to attach
            # span columns to note_rows below.
            my_facts = [d["fact"] for d in my_items] if decomp_mode == "span" else my_items

            support_sum = np.zeros(len(my_facts), dtype=np.float64)
            support_cnt = np.zeros(len(my_facts), dtype=np.int32)

            for ref_idx in ref_indices:
                windows = windows_cache[ref_idx].get(sec, [])
                if windows:
                    probs = _max_entail_probs_batch(nli, windows, my_facts, entail_idx)
                    checked = np.ones(len(my_facts), dtype=bool)
                else:
                    probs = np.zeros(len(my_facts), dtype=np.float64)
                    checked = np.zeros(len(my_facts), dtype=bool)

                low_mask = probs < SECTION_FALLBACK_THRESHOLD
                if low_mask.any():
                    # Same-section support is weak. Only for those weak facts,
                    # search the other sections of the same reference note.
                    weak_facts = [my_facts[i] for i in np.where(low_mask)[0]]
                    for other_sec in SECTION_NAMES:
                        if other_sec == sec:
                            continue
                        other_windows = windows_cache[ref_idx].get(other_sec, [])
                        if not other_windows:
                            continue
                        checked[low_mask] = True
                        other_probs = _max_entail_probs_batch(
                            nli, other_windows, weak_facts, entail_idx
                        )
                        probs[low_mask] = np.maximum(probs[low_mask], other_probs)

                support_sum += probs
                support_cnt += checked.astype(np.int32)

            for local_idx, fact in enumerate(my_facts):
                if support_cnt[local_idx] == 0:
                    uncertainty = 1.0
                else:
                    uncertainty = 1.0 - float(support_sum[local_idx] / support_cnt[local_idx])

                row = {
                    "fact_idx":    len(note_rows),
                    "section":     sec,
                    "fact":        fact,
                    "uncertainty": round(uncertainty, 4),
                }
                if decomp_mode == "span":
                    row.update(_span_row_fields(my_items[local_idx]))
                note_rows.append(row)

        base_cols = ["fact_idx", "section", "fact", "uncertainty"]
        cols = base_cols + SPAN_CSV_COLUMNS if decomp_mode == "span" else base_cols
        df = pd.DataFrame(note_rows) if note_rows else pd.DataFrame(columns=cols)
        df.to_csv(csv_p, index=False)
        mean_u = df["uncertainty"].mean() if len(df) else float("nan")
        tqdm.write(f"  [saved] sample {sample_idx} note {note_idx}: "
                   f"{len(df)} facts, mean_u={mean_u:.3f}")
        rows.extend(df.to_dict("records"))

    return rows


# ── gpt-oss-20b yes/no scoring ───────────────────────────────────────────────
# Alternative to the cross-encoder NLI backend above. An LLM with a 128K
# context window doesn't need sliding windows or the same-section fallback --
# it just reads the whole matching section directly. Demoed in
# demo_gptoss_nli.py first to check the raw Converse response shape (gpt-oss
# is a Harmony/reasoning model: content[0] is a `reasoningContent` block,
# the actual "Yes"/"No" is the LAST block, under the `text` key).
#
# build_yesno_prompt/gptoss_yesno/GPTOSS_MAX_WORKERS live in llm_client.py,
# not here, so luq_sentence.py's sentence-level gpt-oss scoring can reuse the
# exact same primitive without importing atomic_luq (which would be
# circular -- atomic_luq already imports luq_sentence at module level).


def score_sample_gptoss(sample_idx: int, notes: List[str], decomp_k: int,
                        out_dir: Path, use_cache: bool,
                        reasoning_effort: str = "low",
                        max_workers: int = GPTOSS_MAX_WORKERS,
                        decomp_mode: str = "plain") -> List[dict]:
    """Same fact decomposition as score_sample; NLI is replaced by a
    gpt-oss-20b yes/no call per (fact, reference note) pair, premise = the
    full matching section of that reference note.
    uncertainty = 1 - mean(yes/no over K-1 reference notes)

    Every (fact, reference note) pair for a given note is independent, so
    they're all submitted to one thread pool up front instead of looping
    ref_idx serially per fact -- that serial version capped concurrency at
    K-1 (=4) in-flight calls at a time; flattening across every fact in the
    note keeps `max_workers` calls in flight continuously.

    decomp_mode: "plain" (default) or "span" -- see _prepare_decomp / score_sample.
    """
    K = len(notes)
    all_sections, all_decomp = _prepare_decomp(sample_idx, notes, decomp_k, out_dir, use_cache,
                                               decomp_mode=decomp_mode)

    rows = []
    for note_idx in range(decomp_k):
        csv_p = facts_csv_path(out_dir, sample_idx, note_idx)
        if use_cache and csv_p.exists():
            df = pd.read_csv(csv_p)
            tqdm.write(f"  [cache] scores sample {sample_idx} note {note_idx}")
            rows.extend(df.to_dict("records"))
            continue

        ref_indices = [j for j in range(K) if j != note_idx]

        # Flatten every (section, fact) this note decomposed to, in the same
        # order score_sample uses, so fact_idx/output ordering is unchanged.
        # sec_item_list keeps the original item (dict in span mode, else the
        # bare fact string again) so span columns can be attached below.
        sec_fact_list: List[Tuple[str, str]] = []
        sec_item_list: List = []
        for sec in SECTION_NAMES:
            for item in all_decomp[note_idx].get(sec, []):
                fact = item["fact"] if decomp_mode == "span" else item
                sec_fact_list.append((sec, fact))
                sec_item_list.append(item)
        votes_by_item: Dict[int, List[float]] = {i: [] for i in range(len(sec_fact_list))}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_item = {}
            for item_idx, (sec, fact) in enumerate(sec_fact_list):
                for ref_idx in ref_indices:
                    ref_text = all_sections[ref_idx].get(sec, "").strip()
                    if not ref_text:
                        continue
                    fut = pool.submit(gptoss_yesno, ref_text, fact, reasoning_effort)
                    future_to_item[fut] = item_idx

            for fut in tqdm(as_completed(future_to_item), total=len(future_to_item),
                            desc=f"  sample {sample_idx} note {note_idx} [gptoss]",
                            unit="call", leave=False):
                votes_by_item[future_to_item[fut]].append(fut.result())

        note_rows = []
        for item_idx, (sec, fact) in enumerate(sec_fact_list):
            votes = votes_by_item[item_idx]
            uncertainty = 1.0 if not votes else 1.0 - float(np.mean(votes))
            row = {
                "fact_idx":    item_idx,
                "section":     sec,
                "fact":        fact,
                "uncertainty": round(uncertainty, 4),
            }
            if decomp_mode == "span":
                row.update(_span_row_fields(sec_item_list[item_idx]))
            note_rows.append(row)

        base_cols = ["fact_idx", "section", "fact", "uncertainty"]
        cols = base_cols + SPAN_CSV_COLUMNS if decomp_mode == "span" else base_cols
        df = pd.DataFrame(note_rows) if note_rows else pd.DataFrame(columns=cols)
        df.to_csv(csv_p, index=False)
        mean_u = df["uncertainty"].mean() if len(df) else float("nan")
        tqdm.write(f"  [saved] sample {sample_idx} note {note_idx}: "
                   f"{len(df)} facts, mean_u={mean_u:.3f}")
        rows.extend(df.to_dict("records"))

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LUQ-ATOMIC: section-matched atomic-fact uncertainty scoring")
    p.add_argument("--start",    type=int, default=0)
    p.add_argument("--end",      type=int, default=132)
    p.add_argument("--K",        type=int, default=DEFAULT_K,
                   help="Total note pool per sample used as NLI reference (default 10)")
    p.add_argument("--decomp-k", type=int, default=DEFAULT_DECOMP_K,
                   help="How many notes to decompose into atomic facts (default 3)")
    p.add_argument("--gen-dir",  default=DEFAULT_GEN_DIR)
    p.add_argument("--out",      default=DEFAULT_OUT_DIR)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--backend", choices=["cross-encoder", "gptoss"], default="cross-encoder",
                   help="NLI backend for Step 3 scoring: the local "
                        "cross-encoder (default) or gpt-oss-20b yes/no via "
                        "Bedrock Converse.")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low",
                   help="gpt-oss reasoning_effort (only used with --backend gptoss)")
    p.add_argument("--gptoss-workers", type=int, default=GPTOSS_MAX_WORKERS,
                   help="Concurrent Bedrock calls for --backend gptoss "
                        f"(default {GPTOSS_MAX_WORKERS})")
    p.add_argument("--decomp-mode", choices=["plain", "span"], default="plain",
                   help="'plain' (default): existing decompose_section(), "
                        "flat fact list, unchanged output schema. "
                        "'span': decompose_section_with_spans() -- one LLM "
                        "call per section also returns each fact's verbatim "
                        "source span (verified against the section text), "
                        "adding span/span_start/span_end/span_verified/"
                        "synthesized columns to the facts CSV. Uses a "
                        "separate decomp cache filename, so it never touches "
                        "an existing --decomp-mode plain cache in the same "
                        "--out dir -- but point --out at a fresh directory "
                        "anyway to keep the two runs' facts CSVs separate.")
    return p.parse_args()


def main():
    args     = parse_args()
    gen_dir  = Path(args.gen_dir).resolve()
    out_dir  = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    print(f"Gen dir  : {gen_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"Samples  : {args.start} – {args.end - 1}")
    print(f"K (pool) : {args.K}")
    print(f"Decomp K : {args.decomp_k}")
    print(f"Sections : {SECTION_NAMES}")
    print(f"Cache    : {'on' if use_cache else 'off'}")
    print(f"Backend  : {args.backend}"
          + (f" (reasoning_effort={args.reasoning_effort})" if args.backend == "gptoss" else ""))
    print(f"Decomp mode : {args.decomp_mode}")

    summary_rows = []
    n_processed = 0
    for sample_idx in tqdm(range(args.start, args.end), desc="samples", unit="sample"):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            continue
        gen = json.loads(gen_path.read_text())
        notes = gen["notes"][: args.K]
        if len(notes) < 2:
            tqdm.write(f"  [skip] sample {sample_idx}: fewer than 2 notes")
            continue

        decomp_k = min(args.decomp_k, len(notes))
        print(f"\n[sample {sample_idx}] {len(notes)} notes, decomposing first {decomp_k}")
        if args.backend == "gptoss":
            rows = score_sample_gptoss(sample_idx, notes, decomp_k, out_dir, use_cache,
                                       reasoning_effort=args.reasoning_effort,
                                       max_workers=args.gptoss_workers,
                                       decomp_mode=args.decomp_mode)
        else:
            rows = score_sample(sample_idx, notes, decomp_k, out_dir, use_cache,
                                decomp_mode=args.decomp_mode)

        # Decomposition (stage="decomp") runs for every sample regardless of
        # backend; NLI scoring only shows up here as stage="nli" when
        # --backend gptoss (the cross-encoder is local/free, not tracked).
        n_processed += 1
        if n_processed % 5 == 0:
            print(f"  [cost] after {n_processed} samples:\n{llm_tracker.summary()}")

        if rows:
            df = pd.DataFrame(rows)
            summary_rows.append({
                "sample_idx":  sample_idx,
                "n_facts":     len(df),
                "mean_u":      round(float(df["uncertainty"].mean()), 4),
                "high_u_frac": round(float((df["uncertainty"] > 0.5).mean()), 4),
            })

    print(f"\n[cost] final:\n{llm_tracker.summary()}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out_csv = out_dir / "atomic_luq_results.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\n[done] summary → {out_csv}")
        print(summary.describe())


if __name__ == "__main__":
    main()
