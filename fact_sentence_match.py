"""
fact_sentence_match.py
=======================
Post-decomposition fact -> sentence attribution for atomic_luq.py's output,
using medical-entity matching (via scispacy's en_ner_bc5cdr_md, a DISEASE/
CHEMICAL NER model) with an LCS-precision fallback for facts that share no
recognized medical entity with any candidate sentence.

Why this design (validated empirically, see notes below):
  - Plain word-level LCS precision alone: 74.6% clean-match rate. Misses
    facts where decomposition rewrites terse content into different wording
    (e.g. "Diabetes is a problem." from a bare problem-list enumeration
    sharing almost no literal tokens with "Problem list: ...Diabetes...").
  - TF-IDF-weighted LCS: 70.6% -- WORSE. Down-weighting common words fixes
    the problem above, but over-penalizes facts where the specific content
    word itself is paraphrased ("ACL injury" vs "ACL tear"), because that
    single high-weight mismatch now dominates the score.
  - Sequential fact-by-fact processing with a forward-only, no-backtrack
    sentence pointer: 27.3% -- MUCH worse, even with entity matching. One
    spurious early match (or even a correct-looking but premature one)
    permanently strands every subsequent fact whose true source sentence
    is behind the pointer. This is a structural failure of the search
    strategy, not a matching-quality problem, and entity-matching alone
    does not fix it (it can even produce confidently WRONG matches, e.g.
    matching "denied a rash" to an unrelated later "no rash" exam finding
    once the pointer has skipped past the true source sentence).
  - Entity-first (any shared DISEASE/CHEMICAL entity = automatic match),
    LCS-precision fallback, EXHAUSTIVE search per fact (no pointer/order
    dependency): 87.7% -- best of all approaches tested, because entity
    matching handles the paraphrase-divergence cases word-overlap can't,
    while exhaustive independent search avoids the cascade-failure mode
    of sequential processing.

Requires the en_ner_bc5cdr_md scispacy model, which is NOT installed in
this project's default environment -- only in a sibling conda env
(curebench). Run with that interpreter, e.g.:
    /Users/hannah_mac/miniconda3/envs/curebench/bin/python fact_sentence_match.py

Outputs:
  <out>/sample_NNN_note_KK_facts_matched.csv
      fact_idx, section, fact, uncertainty, match_sent_idx, sentence_text,
      precision, matched_by ("entity" | "lcs" | None), clean_match
  <out>/fact_sentence_match_summary.csv
      one row per note: n_facts, n_clean, n_entity, n_lcs, clean_rate

Usage:
    python fact_sentence_match.py
    python fact_sentence_match.py --start 0 --end 44
    python fact_sentence_match.py --facts-dir luq_out/llama_atomic/facts \\
                                  --gen-dir luq_out/llama/generations \\
                                  --out luq_out/llama_atomic/facts_matched
"""

from __future__ import annotations

import argparse
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import luq_sentence as luq

DEFAULT_FACTS_DIR = str(BASE_DIR / "luq_out/llama_atomic/facts")
DEFAULT_GEN_DIR   = str(BASE_DIR / "luq_out/llama/generations")
DEFAULT_OUT_DIR   = str(BASE_DIR / "luq_out/llama_atomic/facts_matched")
DEFAULT_THRESH    = 0.6
SPACY_MODEL       = "en_ner_bc5cdr_md"

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NEG_RE = re.compile(r"\b(?:no|not|denies?|denied|without|negative for|absence of)\b", re.IGNORECASE)
_ALNUM_MED_RE = re.compile(r"^(?:[A-Za-z]*\d+[A-Za-z]*|[A-Za-z]+-[A-Za-z0-9]+)$")
_DOSE_RE = re.compile(r"^\d+(?:\.\d+)?(?:mg|mcg|g|ml|mm|cm|kg|lb|l|bpm|%)$", re.IGNORECASE)
_GENERIC_ENTITY_TERMS = {
    "pain", "rash", "fever", "nausea", "vomiting", "diarrhea", "swelling",
    "infection", "ulcer",
}
_MED_CUE_WORDS = {
    "abi", "a1c", "mri", "ct", "xray", "x-ray", "ultrasound", "pft", "pt",
    "l5", "mtp", "dorsiflexion", "angioedema", "stridor", "wheezing",
    "osteomyelitis", "meloxicam", "singulair", "albuterol", "clindamycin",
    "asthma", "diabetes", "ulcer", "debridement", "immunotherapy",
}
_LEMMA_STOP = {
    "patient", "doctor", "symptom", "report", "present", "mention", "state",
    "current", "medication", "history", "review", "system", "assessment",
    "problem", "list", "objective", "subjective", "plan", "follow", "followup",
    "test", "result", "physical", "exam", "vital", "sign", "issue",
}
_HEADER_RE = re.compile(
    r"^(?:here is the soap note based on the transcript:|subjective:|objective:|assessment(?: / problem list)?:|problem list:|plan:|follow-?up:|instructions:)\s*$",
    re.IGNORECASE,
)


def _tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _is_negated(text: str) -> bool:
    return bool(_NEG_RE.search(text or ""))


def _is_header_sentence(text: str) -> bool:
    return bool(_HEADER_RE.match((text or "").strip()))


def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], row[j - 1])
    return int(dp[n, m])


def _norm_sec(sec: str) -> str:
    """atomic_luq.py sections -> luq_sentence.py sections normalization."""
    sec = (sec or "").lower()
    if sec == "followup":
        return "plan"
    if sec in ("all", "other"):
        return "_any"
    return sec


# ── Name matching (BC5CDR only covers DISEASE/CHEMICAL, not PERSON) ────────────
# Capitalized-token heuristic instead of a second NER pass: cheap, and avoids
# a dependency on a generic English model just for names. Excludes common
# capitalized clinical-note words (section headers, pronouns, abbreviations)
# that would otherwise look like proper nouns.
_CAP_WORD_RE = re.compile(r"[A-Z][a-zA-Z]+")
_CAP_STOPWORDS = {
    "the", "he", "she", "his", "her", "patient", "objective", "subjective",
    "assessment", "plan", "follow", "followup", "review", "current", "past",
    "test", "vital", "physical", "problem", "list", "hpi", "pmh", "ros", "i",
    "medications", "results", "history", "exam", "signs", "systems",
}


def extract_names(text: str) -> set:
    out = set()
    for w in text.split():
        clean = re.sub(r"[^\w'-]", "", w)
        if len(clean) < 2:
            continue
        if _CAP_WORD_RE.fullmatch(clean) and clean.lower() not in _CAP_STOPWORDS:
            out.add(clean.lower())
    return out


# ── Numeric / measurement matching (vitals, doses, labs -- as distinctive as
# a drug name, but not covered by entity matching) ─────────────────────────────
_NUM_RE = re.compile(r"\d+(?:[./]\d+)?(?:mg|mcg|ml|kg|lb|bpm|%)?", re.IGNORECASE)


def extract_numbers(text: str) -> set:
    return set(m.group(0).lower() for m in _NUM_RE.finditer(text))


def analyze_batch(nlp, texts: List[str]) -> List[dict]:
    rows = []
    for doc in nlp.pipe(texts, batch_size=64):
        ents = {ent.text.lower().strip() for ent in doc.ents if ent.text.strip()}
        lemmas: List[str] = []
        content_seq: List[str] = []
        salient: Set[str] = set()
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            base = (tok.lemma_ or tok.text).strip().lower()
            if not base:
                continue
            if not tok.is_stop and base not in _LEMMA_STOP:
                lemmas.append(base)
                content_seq.append(base)
            if (
                base in _MED_CUE_WORDS
                or tok.text.lower() in _MED_CUE_WORDS
                or tok.like_num
                or _DOSE_RE.match(tok.text.lower())
                or _ALNUM_MED_RE.match(tok.text)
            ):
                salient.add(base)

        rows.append({
            "ents": ents,
            "names": extract_names(doc.text),
            "nums": extract_numbers(doc.text),
            "lemmas": lemmas,
            "content_seq": content_seq,
            "salient": salient,
            "neg": _is_negated(doc.text),
        })
    return rows


def _entity_overlap_score(fact_ents: Set[str], sent_ents: Set[str]) -> float:
    shared = fact_ents & sent_ents
    if not shared:
        return 0.0
    score = 0.0
    for ent in shared:
        toks = _tokenize(ent)
        if len(toks) > 1:
            score = max(score, 1.0)
        elif toks and toks[0] not in _GENERIC_ENTITY_TERMS:
            score = max(score, 0.85)
        else:
            score = max(score, 0.45)
    return score


def _overlap_count(a: Sequence[str] | Set[str], b: Sequence[str] | Set[str]) -> int:
    return len(set(a) & set(b))


def load_nlp():
    import spacy
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise SystemExit(
            f"Could not load spaCy model '{SPACY_MODEL}'. This model is not "
            f"installed in the current environment. Run this script with the "
            f"conda env that has it, e.g.:\n"
            f"  /Users/hannah_mac/miniconda3/envs/curebench/bin/python {__file__}"
        ) from exc


def best_match(
    fact_text: str,
    fact_info: dict,
    sentences: List[str],
    sent_info_list: List[dict],
    sent_secs: List[str],
    fact_sec: str,
    thresh: float = DEFAULT_THRESH,
) -> Tuple[Optional[int], float, Optional[str], dict]:
    f_tok = _tokenize(fact_text)
    f_lem = fact_info["lemmas"]
    f_content = fact_info["content_seq"]
    f_sal = fact_info["salient"]
    f_ents = fact_info["ents"]
    f_names = fact_info["names"]
    f_nums = fact_info["nums"]
    f_neg = fact_info["neg"]
    if not f_tok and not f_sal and not f_ents and not f_names and not f_nums:
        return None, 0.0, None, {}

    fsec = _norm_sec(fact_sec)
    candidates = [
        i for i in range(len(sentences))
        if not _is_header_sentence(sentences[i])
        and (_norm_sec(sent_secs[i]) == fsec or fsec == "_any" or _norm_sec(sent_secs[i]) == "_any")
    ]
    if not candidates:
        candidates = list(range(len(sentences)))

    best_i, best_score, best_meta = None, 0.0, {}
    for i in candidates:
        sinfo = sent_info_list[i]
        s_tok = _tokenize(sentences[i])
        s_content = sinfo["content_seq"]

        ent_score = _entity_overlap_score(f_ents, sinfo["ents"])
        name_overlap = 1.0 if (f_names and (f_names & sinfo["names"])) else 0.0
        num_overlap = 1.0 if (f_nums and (f_nums & sinfo["nums"])) else 0.0
        lcs_prec = _safe_div(
            _lcs_len(f_content or f_tok, s_content or s_tok),
            max(len(f_content or f_tok), 1),
        )
        lemma_prec = _safe_div(_overlap_count(f_lem, sinfo["lemmas"]), max(len(set(f_lem)), 1))
        sal_prec = _safe_div(_overlap_count(f_sal, sinfo["salient"]), max(len(f_sal), 1))
        ent_effective = ent_score if len(f_content) <= 3 else ent_score * max(0.3, sal_prec, lemma_prec, lcs_prec)
        overlap_signal = max(ent_score, name_overlap, num_overlap, sal_prec, lemma_prec, lcs_prec)
        neg_bonus = 0.0
        if overlap_signal > 0:
            neg_bonus = 0.05 if f_neg == sinfo["neg"] else -0.05

        shared_sal = set(f_sal) & set(sinfo["salient"])
        score = (
            0.32 * ent_effective
            + 0.18 * name_overlap
            + 0.16 * num_overlap
            + 0.16 * sal_prec
            + 0.09 * lemma_prec
            + 0.09 * lcs_prec
            + neg_bonus
        )

        if ent_score >= 0.85 and (sal_prec > 0 or lemma_prec > 0):
            score = max(score, 0.93)
        elif num_overlap and (sal_prec > 0 or lcs_prec > 0):
            score = max(score, 0.88)
        elif name_overlap and lcs_prec > 0:
            score = max(score, 0.9)
        elif (
            sal_prec > 0
            and len(f_sal) <= 2
            and len(shared_sal) >= 1
            and (lemma_prec >= 0.4 or lcs_prec >= 0.4 or f_neg == sinfo["neg"])
        ):
            score = max(score, 0.68)

        meta = {
            "entity_score": round(ent_score, 3),
            "entity_effective": round(ent_effective, 3),
            "name_overlap": int(bool(name_overlap)),
            "number_overlap": int(bool(num_overlap)),
            "salient_precision": round(sal_prec, 3),
            "lemma_precision": round(lemma_prec, 3),
            "lcs_precision": round(lcs_prec, 3),
            "neg_match": int(f_neg == sinfo["neg"]),
            "score": round(score, 3),
        }
        if score > best_score or (score == best_score and best_i is not None and i < best_i):
            best_i, best_score, best_meta = i, score, meta

    if best_i is None:
        return None, 0.0, None, {}

    matched_by: Optional[str] = None
    if best_meta.get("entity_score", 0.0) >= 0.85 and (
        best_meta.get("salient_precision", 0.0) > 0
        or best_meta.get("lemma_precision", 0.0) >= 0.5
        or best_meta.get("lcs_precision", 0.0) >= 0.5
    ):
        matched_by = "entity"
    elif best_meta.get("name_overlap", 0) and best_meta.get("lcs_precision", 0.0) >= 0.25:
        matched_by = "name"
    elif best_meta.get("number_overlap", 0) and (
        best_meta.get("lcs_precision", 0.0) >= 0.25 or best_meta.get("salient_precision", 0.0) > 0
    ):
        matched_by = "number"
    elif best_meta.get("salient_precision", 0.0) >= 0.5 and (
        best_meta.get("lemma_precision", 0.0) >= 0.3
        or best_meta.get("lcs_precision", 0.0) >= 0.3
        or best_meta.get("neg_match", 0)
    ):
        matched_by = "medical_token"
    elif (
        best_meta.get("lcs_precision", 0.0) >= thresh
        or best_meta.get("lemma_precision", 0.0) >= 0.75
        or (
            best_meta.get("lemma_precision", 0.0) >= 0.5
            and best_meta.get("lcs_precision", 0.0) >= 0.5
        )
    ):
        matched_by = "lcs"
    return best_i, best_score, matched_by, best_meta


def match_note(nlp, sample_idx: int, note_idx: int, note_text: str,
              facts_dir: Path, out_dir: Path, thresh: float) -> Optional[dict]:
    facts_csv = facts_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts.csv"
    if not facts_csv.exists():
        return None
    fdf = pd.read_csv(facts_csv)
    if fdf.empty:
        return None

    sentences = luq.split_sentences(note_text)
    if not sentences:
        return None
    sent_secs = luq.assign_sentence_sections(note_text, sentences)
    sent_info_list = analyze_batch(nlp, sentences)

    fact_texts = fdf["fact"].astype(str).tolist()
    fact_info_list = analyze_batch(nlp, fact_texts)

    rows = []
    for (_, r), finfo in zip(fdf.iterrows(), fact_info_list):
        idx, prec, by, meta = best_match(
            str(r["fact"]), finfo, sentences, sent_info_list, sent_secs, r.get("section", ""), thresh
        )
        rows.append({
            **r.to_dict(),
            "match_sent_idx": idx,
            "sentence_text":  sentences[idx] if idx is not None else "",
            "precision":      round(prec, 3),
            "matched_by":     by,
            "clean_match":    by is not None,
            **meta,
        })

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts_matched.csv"
    out_df.to_csv(out_csv, index=False)

    return {
        "sample_idx": sample_idx, "note_idx": note_idx,
        "n_facts":    len(out_df),
        "n_clean":    int(out_df["clean_match"].sum()),
        "n_entity":   int((out_df["matched_by"] == "entity").sum()),
        "n_name":     int((out_df["matched_by"] == "name").sum()),
        "n_number":   int((out_df["matched_by"] == "number").sum()),
        "n_medtok":   int((out_df["matched_by"] == "medical_token").sum()),
        "n_lcs":      int((out_df["matched_by"] == "lcs").sum()),
        "clean_rate": round(float(out_df["clean_match"].mean()), 4) if len(out_df) else float("nan"),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Entity-aware fact -> sentence attribution")
    p.add_argument("--start",     type=int, default=0)
    p.add_argument("--end",       type=int, default=44)
    p.add_argument("--decomp-k",  type=int, default=3)
    p.add_argument("--K",         type=int, default=10)
    p.add_argument("--facts-dir", default=DEFAULT_FACTS_DIR)
    p.add_argument("--gen-dir",   default=DEFAULT_GEN_DIR)
    p.add_argument("--out",       default=DEFAULT_OUT_DIR)
    p.add_argument("--thresh",    type=float, default=DEFAULT_THRESH)
    return p.parse_args()


def main():
    args = parse_args()
    facts_dir = Path(args.facts_dir).resolve()
    gen_dir   = Path(args.gen_dir).resolve()
    out_dir   = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {SPACY_MODEL} ...")
    nlp = load_nlp()

    summary_rows = []
    for sample_idx in range(args.start, args.end):
        gen_path = gen_dir / f"sample_{sample_idx:03d}_generations.json"
        if not gen_path.exists():
            continue
        gen = json.loads(gen_path.read_text())
        notes = gen["notes"][: args.K]

        for note_idx in range(min(args.decomp_k, len(notes))):
            result = match_note(nlp, sample_idx, note_idx, notes[note_idx],
                                facts_dir, out_dir, args.thresh)
            if result is not None:
                summary_rows.append(result)
                print(f"  sample {sample_idx} note {note_idx}: "
                      f"{result['n_clean']}/{result['n_facts']} clean "
                      f"({result['clean_rate']:.1%})  "
                      f"[entity={result['n_entity']} name={result['n_name']} "
                      f"number={result['n_number']} medtok={result['n_medtok']} "
                      f"lcs={result['n_lcs']}]")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "fact_sentence_match_summary.csv"
        summary.to_csv(summary_csv, index=False)

        n_facts  = int(summary["n_facts"].sum())
        n_clean  = int(summary["n_clean"].sum())
        n_entity = int(summary["n_entity"].sum())
        n_name   = int(summary["n_name"].sum())
        n_number = int(summary["n_number"].sum())
        n_medtok = int(summary["n_medtok"].sum())
        n_lcs    = int(summary["n_lcs"].sum())
        print(f"\n[done] {len(summary)} notes, {n_facts} facts")
        print(f"  clean match: {n_clean} ({n_clean/n_facts:.1%})  "
              f"[entity={n_entity} ({n_entity/n_facts:.1%}), "
              f"name={n_name} ({n_name/n_facts:.1%}), "
              f"number={n_number} ({n_number/n_facts:.1%}), "
              f"medtok={n_medtok} ({n_medtok/n_facts:.1%}), "
              f"lcs={n_lcs} ({n_lcs/n_facts:.1%})]")
        print(f"  no match:    {n_facts - n_clean} ({(n_facts-n_clean)/n_facts:.1%})")
        print(f"  -> {summary_csv}")


if __name__ == "__main__":
    main()
