"""
se_highlight.py
===============
Loads saved generation JSONs from generations/, computes typed-claim semantic
entropy, and generates an HTML report highlighting the top 3 most uncertain
samples.

Usage
-----
    python se_highlight.py --gen-dir generations/ --out se_highlight.html
    python se_highlight.py --gen-dir . --bedrock-region us-east-1 --nli-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from math import log
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Constants
# ─────────────────────────────────────────────────────────────────────────────

BEDROCK_MODEL = "us.meta.llama3-3-70b-instruct-v1:0"
THRESH_HALLUCIN = 0.5  # NLI entailment threshold for clustering

CLAIM_TYPES = [
    # What brought the patient in and what they feel
    "chief_complaint",
    "symptom",          # location, character, timing, mechanism, laterality all together
    "denied_symptom",   # all denials together
    
    # Background
    "past_history",     # surgical, medical, prior treatments tried
    "current_medication",
    
    # Objective findings
    "vital_sign",       # all vitals — BP, HR, RR, SpO2, temp as one cluster
    "exam_finding",     # all physical exam findings
    "test_result",      # imaging type + finding together
    
    # Clinical reasoning output  
    "diagnosis",        # primary + secondary
    
    # What happens next
    "plan_medication",  # name + dose + frequency together
    "plan_procedure",   # imaging ordered, injections, referrals
    "plan_restriction", # activity restrictions, lifestyle advice
    "follow_up",        # timing + action together
]

_SYSTEM_PROMPT = (
    "You are a clinical information extractor. Extract the following claim types "
    "from the clinical note.\n"
    "For each claim type return the exact text span as it appears in the note, "
    "or null if not mentioned.\n"
    "Return ONLY valid JSON with these exact keys. No explanation."
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Lazy-loaded globals
# ─────────────────────────────────────────────────────────────────────────────

_sapbert_model = None
_sapbert_tokenizer = None
_bedrock_client = None


def _get_bedrock_client(region: str):
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        _bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return _bedrock_client


def _get_nli_model():
    """Load cross-encoder NLI model once and reuse."""
    global _sapbert_model, _sapbert_tokenizer
    if _sapbert_model is None:
        print("[init] Loading NLI model (cross-encoder/nli-deberta-v3-small) ...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name = "cross-encoder/nli-deberta-v3-small"
        _sapbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _sapbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _sapbert_model.eval()
        print("[init] NLI model loaded.")
    return _sapbert_tokenizer, _sapbert_model


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Claim extraction via Bedrock
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_message(note: str) -> str:
    keys_block = "\n".join(f"  - {ct}" for ct in CLAIM_TYPES)
    example = "{" + ", ".join(f'"{ct}": null' for ct in CLAIM_TYPES[:3]) + ", ...}"
    return (
        f"Claim types to extract:\n{keys_block}\n\n"
        f"Clinical note:\n{note}\n\n"
        f"Return JSON like:\n{example}"
    )


def _parse_claims_json(raw: str) -> Optional[Dict[str, Optional[str]]]:
    """Try to parse a JSON dict from the raw LLM response."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    # Find first { ... }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        # Normalise: keep only known keys, fill missing with None
        result: Dict[str, Optional[str]] = {}
        for ct in CLAIM_TYPES:
            val = data.get(ct)
            result[ct] = str(val).strip() if val and str(val).strip().lower() not in ("null", "none", "") else None
        return result
    except json.JSONDecodeError:
        return None


def extract_claims_from_note(
    note: str,
    bedrock_region: str,
) -> Dict[str, Optional[str]]:
    """
    Call Bedrock Converse to extract typed claims from a single clinical note.
    Retries once on parse failure.
    """
    client = _get_bedrock_client(bedrock_region)
    user_msg = _build_user_message(note)

    for attempt in range(2):
        response = client.converse(
            modelId=BEDROCK_MODEL,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
        )
        raw = response["output"]["message"]["content"][0]["text"]
        claims = _parse_claims_json(raw)
        if claims is not None:
            return claims
        if attempt == 0:
            print("    [warn] JSON parse failed on attempt 1, retrying ...")

    # If both attempts fail, return all nulls
    print("    [warn] Both parse attempts failed — returning all nulls.")
    return {ct: None for ct in CLAIM_TYPES}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  NLI entailment scoring
# ─────────────────────────────────────────────────────────────────────────────

def _entailment_label_index(model) -> int:
    id2label = model.config.id2label
    for idx, label in id2label.items():
        if "entail" in label.lower():
            return idx
    raise ValueError(f"No entailment label found in {id2label}")


def _nli_score_pairs(pairs: List[Tuple[str, str]], threshold: float) -> List[bool]:
    """Return True for each (premise, hypothesis) pair where entailment > threshold."""
    if not pairs:
        return []
    import torch
    tokenizer, model = _get_nli_model()
    entail_idx = _entailment_label_index(model)
    enc = tokenizer(
        [p for p, _ in pairs], [h for _, h in pairs],
        padding=True, truncation=True, max_length=256, return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    return [p > threshold for p in probs[:, entail_idx].tolist()]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Bedrock atomic decomposition
# ─────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = (
    "You are a clinical fact extractor. "
    "Decompose the given clinical value into atomic facts — one independently "
    "verifiable fact per line. "
    "Output only the facts with no numbering, bullets, or explanation. "
    "If the value is already a single atomic fact, output it unchanged on one line."
)


def decompose_claim_value(
    value: str,
    claim_type: str,
    bedrock_region: str,
) -> List[str]:
    """
    Call Bedrock to split a claim value into atomic facts (one per line).
    Returns a list of non-empty stripped strings.
    """
    client  = _get_bedrock_client(bedrock_region)
    user_msg = (
        f"Claim type: {claim_type}\n"
        f"Value: {value}\n\n"
        f"Atomic facts (one per line):"
    )
    response = client.converse(
        modelId=BEDROCK_MODEL,
        system=[{"text": _DECOMPOSE_SYSTEM}],
        messages=[{"role": "user", "content": [{"text": user_msg}]}],
        inferenceConfig={"maxTokens": 256, "temperature": 0.0},
    )
    raw   = response["output"]["message"]["content"][0]["text"]
    facts = [line.strip().lstrip("-•* ").strip() for line in raw.splitlines()]
    return [f for f in facts if len(f) > 3]


def decompose_all_claims(
    claims: Dict[str, Optional[str]],
    bedrock_region: str,
) -> Dict[str, List[str]]:
    """
    For each claim type, decompose the value into atomic facts.
    None values map to an empty list.
    """
    result: Dict[str, List[str]] = {}
    for ct, val in claims.items():
        if val is None:
            result[ct] = []
        else:
            result[ct] = decompose_claim_value(val, ct, bedrock_region)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6a.  Clustering via BFS on bidirectional NLI entailment
# ─────────────────────────────────────────────────────────────────────────────

def cluster_strings(
    strings: List[Optional[str]],
    threshold: float,
) -> List[List[int]]:
    """
    Cluster strings (with Nones) by bidirectional NLI entailment.
    Nones are singleton clusters. Returns list of index-lists.
    """
    none_clusters  = [[i] for i, s in enumerate(strings) if s is None]
    non_none_idxs  = [i for i, s in enumerate(strings) if s is not None]

    if len(non_none_idxs) <= 1:
        return none_clusters + [[i] for i in non_none_idxs]

    # All directed pairs
    pairs = [
        (strings[non_none_idxs[a]], strings[non_none_idxs[b]])  # type: ignore[index]
        for a in range(len(non_none_idxs))
        for b in range(len(non_none_idxs))
        if a != b
    ]
    entails = _nli_score_pairs(pairs, threshold)

    pair_entail: Dict[Tuple[int, int], bool] = {}
    idx = 0
    for a in range(len(non_none_idxs)):
        for b in range(len(non_none_idxs)):
            if a != b:
                pair_entail[(non_none_idxs[a], non_none_idxs[b])] = entails[idx]
                idx += 1

    adj: Dict[int, List[int]] = defaultdict(list)
    for a in range(len(non_none_idxs)):
        for b in range(a + 1, len(non_none_idxs)):
            i, j = non_none_idxs[a], non_none_idxs[b]
            if pair_entail.get((i, j)) and pair_entail.get((j, i)):
                adj[i].append(j)
                adj[j].append(i)

    visited, components = set(), []
    for start in non_none_idxs:
        if start in visited:
            continue
        queue, comp = [start], []
        visited.add(start)
        while queue:
            node = queue.pop()
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)

    return none_clusters + components


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Entropy calculation
# ─────────────────────────────────────────────────────────────────────────────

def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * log(p) + (1 - p) * log(1 - p))


def compute_claim_type_se(
    decompositions: List[List[str]],
    K: int,
    threshold: float,
) -> float:
    """
    Compute semantic entropy for one claim type using atomic-fact decompositions.

    decompositions : list of K lists — decompositions[k] is the atomic facts
                     from generation k (empty list if the claim was absent).
    K              : total number of generations (for absence penalty).

    Pipeline:
      1. Pool all facts, tagging each with its generation index.
      2. Cluster all facts by bidirectional NLI entailment.
      3. For each cluster: k_present = number of distinct generations
         that contributed at least one fact to this cluster.
      4. Cluster SE = binary_entropy(k_present / K).
      5. Return MEAN SE across all clusters (overall uncertainty).
    """
    # Pool facts with generation tags
    all_facts: List[str]  = []
    gen_tags:  List[int]  = []
    for gen_idx, facts in enumerate(decompositions):
        for fact in facts:
            all_facts.append(fact)
            gen_tags.append(gen_idx)

    # Generations with no facts at all get a synthetic "ABSENT" entry so that
    # absence is treated as a distinct cluster competing against the present facts.
    absent_gens = [k for k in range(K) if not decompositions[k]]
    for k in absent_gens:
        all_facts.append(f"__ABSENT_{k}__")
        gen_tags.append(k)

    if not all_facts:
        return 0.0

    # Mark absent sentinels as None so cluster_strings treats them as singletons
    nullable: List[Optional[str]] = [
        None if f.startswith("__ABSENT_") else f for f in all_facts
    ]

    clusters = cluster_strings(nullable, threshold)

    se_vals: List[float] = []
    for cluster_idxs in clusters:
        gens_in_cluster = {gen_tags[i] for i in cluster_idxs}
        k_present = len(gens_in_cluster)
        se_vals.append(binary_entropy(k_present / K))

    return sum(se_vals) / len(se_vals) if se_vals else 0.0


def compute_sample_se(
    all_decompositions: List[Dict[str, List[str]]],
    K: int,
    threshold: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Given K claim-dicts (one per generation), compute:
    - claim_type_ses: dict[claim_type -> SE]
    - mean_se: mean over all claim types

    Returns (mean_se, claim_type_ses).
    """
    claim_type_ses: Dict[str, float] = {}
    for ct in CLAIM_TYPES:
        decomps = [d.get(ct, []) for d in all_decompositions]
        claim_type_ses[ct] = compute_claim_type_se(decomps, K, threshold)

    mean_se = sum(claim_type_ses.values()) / len(CLAIM_TYPES)
    return mean_se, claim_type_ses


def compute_generation_se(
    note_claims: Dict[str, Optional[str]],
    claim_type_ses: Dict[str, float],
) -> float:
    """
    Mean SE for a single generation (using global claim-type SEs).
    We use the precomputed per-claim-type SEs (consistent with sample-level SE).
    """
    return sum(claim_type_ses.values()) / len(CLAIM_TYPES)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Load generation files
# ─────────────────────────────────────────────────────────────────────────────

def load_generations(gen_dir: Path) -> List[Dict]:
    """Load all sample_NNN_generations.json files, sorted by sample_idx."""
    files = sorted(gen_dir.glob("sample_*_generations.json"))
    if not files:
        raise FileNotFoundError(f"No generation files found in {gen_dir}")
    samples = []
    for f in files:
        with open(f) as fh:
            samples.append(json.load(fh))
    print(f"[load] Loaded {len(samples)} generation files from {gen_dir}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> Dict:
    if cache_path.exists():
        with open(cache_path) as fh:
            data = json.load(fh)
        print(f"[cache] Loaded cache from {cache_path}")
        return data
    return {}


def save_cache(cache_path: Path, cache: Dict) -> None:
    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  HTML generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def se_to_colour(se: float) -> str:
    """Smooth gradient: dark green (SE=0) → white (SE=0.7) → dark red (SE=1)."""
    if se <= THRESH_HALLUCIN:
        t = se / THRESH_HALLUCIN
        r = int(20  + t * (255 - 20))
        g = int(110 + t * (255 - 110))
        b = int(20  + t * (255 - 20))
    else:
        t = (se - THRESH_HALLUCIN) / (1 - THRESH_HALLUCIN)
        r = int(255 + t * (160 - 255))
        g = int(255 + t * (20  - 255))
        b = int(255 + t * (20  - 255))
    return f"rgb({r},{g},{b})"


def _build_span_map(
    note: str,
    claims: Dict[str, Optional[str]],
    claim_type_ses: Dict[str, float],
) -> List[Tuple[int, int, str, float]]:
    """
    Find each non-null claim's text in the note and return sorted, de-overlapped
    list of (start, end, claim_type, se_value).
    On overlap, the span with higher SE wins.
    """
    raw_spans: List[Tuple[int, int, str, float]] = []
    for ct, text in claims.items():
        if text is None:
            continue
        pos = note.find(text)
        if pos == -1:
            # Try case-insensitive
            lower_pos = note.lower().find(text.lower())
            if lower_pos == -1:
                continue
            pos = lower_pos
        se = claim_type_ses.get(ct, 0.0)
        raw_spans.append((pos, pos + len(text), ct, se))

    if not raw_spans:
        return []

    # Sort by start, then by SE descending (so higher SE wins on overlap)
    raw_spans.sort(key=lambda x: (x[0], -x[3]))

    # Greedy de-overlap: keep span if it doesn't overlap with last kept span;
    # on partial overlap prefer higher SE
    kept: List[Tuple[int, int, str, float]] = []
    for span in raw_spans:
        start, end, ct, se = span
        if not kept:
            kept.append(span)
            continue
        last_start, last_end, last_ct, last_se = kept[-1]
        if start >= last_end:
            # No overlap
            kept.append(span)
        else:
            # Overlap: keep whichever has higher SE
            if se > last_se:
                kept[-1] = span

    return kept


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_highlighted_note(
    note: str,
    claims: Dict[str, Optional[str]],
    claim_type_ses: Dict[str, float],
    absent_claims: List[Tuple[str, List[Tuple[str, int]], float, int, int]],
    all_claims: List[Dict[str, Optional[str]]],
) -> str:
    """
    Render the note as HTML with:
      - <mark> tags coloured by SE for present claims
      - Yellow inline annotations appended for absent claims (missing from this generation)
    """
    spans = _build_span_map(note, claims, claim_type_ses)
    events: Dict[int, List] = defaultdict(list)
    for start, end, ct, se in spans:
        events[start].append(("open", ct, se))
        events[end].append(("close", ct, se))

    # Build CSS hover popup content per claim type
    def _popup_html(ct: str, se: float) -> str:
        counts = _build_other_gen_values(ct, claims.get(ct), all_claims)
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;gap:12px;'
            f'{"color:#c0392b;" if count == 1 else ""}">'
            f'<span>{_escape(val)}</span>'
            f'<span style="color:#888;flex-shrink:0;">×{count}</span></div>'
            for val, count in counts
        )
        return (
            f'<span class="tip-box">'
            f'<span style="font-weight:600;font-size:11px;color:#aaa;">'
            f'{ct} &nbsp; SE={se:.3f}</span><br>'
            f'{rows}'
            f'</span>'
        )

    html_parts = []
    i = 0
    note_len = len(note)
    while i <= note_len:
        for kind, ct, se in sorted(events.get(i, []), key=lambda x: (x[0] == "open",)):
            if kind == "close":
                html_parts.append("</span></mark>")
            else:
                colour = se_to_colour(se)
                html_parts.append(
                    f'<mark class="tip-wrap" style="background:{colour};padding:1px 2px;'
                    f'border-radius:2px;cursor:help;position:relative;">'
                    + _popup_html(ct, se)
                )
        if i < note_len:
            c = note[i]
            if c == "\n":
                html_parts.append("<br>")
            else:
                html_parts.append(_escape(c))
        i += 1

    # ── Absent claims: append inline in yellow after the note body ────────────
    if absent_claims:
        html_parts.append(
            "<br><br>"
            "<span style='font-size:11px;color:#888;font-style:italic;'>"
            "── missing from this generation ──</span><br>"
        )
        for ct, val_counts, se, n_present, K in absent_claims:
            se_badge = (
                f'<span style="font-size:10px;color:#888;margin-left:6px;">'
                f'SE={se:.2f} &nbsp;·&nbsp; present {n_present}/{K}</span>'
            )
            # Show each distinct value with its count so outliers are visible
            variants_html = []
            for val, count in val_counts:
                # Minority values (count==1) get a red tint to flag potential hallucinations
                bg = "#ffcccc" if count == 1 else "#fff59d"
                variants_html.append(
                    f'<span style="background:{bg};padding:1px 5px;border-radius:3px;'
                    f'margin-right:4px;font-size:13px;">'
                    f'{_escape(val)}'
                    f'<span style="font-size:10px;color:#888;margin-left:2px;">×{count}</span>'
                    f'</span>'
                )
            html_parts.append(
                f'<div style="margin:3px 0;display:flex;align-items:baseline;flex-wrap:wrap;gap:2px;">'
                f'<span style="font-size:10px;color:#7a6000;font-weight:600;'
                f'font-family:monospace;margin-right:4px;">{ct}:</span>'
                + "".join(variants_html)
                + se_badge
                + f'</div>'
            )

    return "".join(html_parts)


def _colour_legend_html() -> str:
    """Build a horizontal gradient legend bar."""
    stops = []
    for i in range(101):
        se = i / 100
        colour = se_to_colour(se)
        pct = i
        stops.append(f"{colour} {pct}%")
    gradient = ", ".join(stops)
    return f"""
<div style="margin:12px 0;">
  <div style="height:20px;background:linear-gradient(to right,{gradient});
       border-radius:4px;border:1px solid #ccc;"></div>
  <div style="display:flex;justify-content:space-between;font-size:11px;
       color:#555;margin-top:2px;">
    <span>SE = 0.0 (certain)</span>
    <span>SE = 0.7</span>
    <span>SE = 1.0 (uncertain)</span>
  </div>
</div>
"""


def _value_counts(vals: List[str]) -> List[Tuple[str, int]]:
    """Return (value, count) pairs sorted by count descending."""
    from collections import Counter
    return Counter(vals).most_common()


def _build_absent_claims(
    displayed_claims: Dict[str, Optional[str]],
    all_claims: List[Dict[str, Optional[str]]],
    claim_type_ses: Dict[str, float],
) -> List[Tuple[str, List[Tuple[str, int]], float, int, int]]:
    """
    For each claim type null in the displayed generation but present in ≥1 other,
    return (claim_type, value_counts, se, n_present, K).

    value_counts is a list of (value, count) sorted by count descending — ALL
    distinct values are returned so minority hallucinations are visible.
    Sorted by SE descending so the most uncertain missing claims appear first.
    """
    K = len(all_claims)
    absent = []
    for ct in CLAIM_TYPES:
        if displayed_claims.get(ct) is not None:
            continue
        other_vals = [c.get(ct) for c in all_claims if c.get(ct) is not None]
        if not other_vals:
            continue
        se = claim_type_ses.get(ct, 0.0)
        absent.append((ct, _value_counts(other_vals), se, len(other_vals), K))
    absent.sort(key=lambda x: -x[2])  # highest SE first
    return absent


def _build_other_gen_values(
    ct: str,
    displayed_val: Optional[str],
    all_claims: List[Dict[str, Optional[str]]],
) -> List[Tuple[str, int]]:
    """
    For a claim type present in the displayed generation, return value_counts
    for ALL other generations (including ones that differ), so outliers surface.
    """
    other_vals = [c.get(ct) for c in all_claims if c.get(ct) is not None]
    counts = _value_counts(other_vals)
    # Highlight values that differ from what's displayed
    return counts


def _claim_summary_table(
    claims: Dict[str, Optional[str]],
    claim_type_ses: Dict[str, float],
) -> str:
    """Right panel: all claim types with SE, presence, and value."""
    rows = []
    for ct in CLAIM_TYPES:
        val = claims.get(ct)
        se = claim_type_ses.get(ct, 0.0)
        colour = se_to_colour(se)
        present_str = "✓" if val is not None else "—"
        val_str = val if val is not None else "<em style='color:#aaa;'>null</em>"
        rows.append(
            f"<tr>"
            f'<td style="font-family:monospace;font-size:11px;padding:3px 6px;">{ct}</td>'
            f'<td style="background:{colour};text-align:center;padding:3px 6px;">{se:.3f}</td>'
            f'<td style="text-align:center;padding:3px 6px;">{present_str}</td>'
            f'<td style="font-size:12px;padding:3px 6px;">{val_str}</td>'
            f"</tr>"
        )
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:13px;'>"
        "<tr style='background:#f0f0f0;'>"
        "<th style='text-align:left;padding:4px 6px;'>Claim Type</th>"
        "<th style='padding:4px 6px;'>SE</th>"
        "<th style='padding:4px 6px;'>Present</th>"
        "<th style='text-align:left;padding:4px 6px;'>Value</th>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )


def _sample_section_html(
    rank: int,
    sample: Dict,
    all_claims: List[Dict[str, Optional[str]]],
    claim_type_ses: Dict[str, float],
    mean_se: float,
    best_gen_idx: int,
) -> str:
    """Generate one HTML section for a single top-3 sample."""
    sample_idx = sample["sample_idx"]
    notes = sample["notes"]
    K = len(notes)
    displayed_note = notes[best_gen_idx]
    displayed_claims = all_claims[best_gen_idx]

    absent_claims = _build_absent_claims(displayed_claims, all_claims, claim_type_ses)
    highlighted   = _render_highlighted_note(displayed_note, displayed_claims, claim_type_ses, absent_claims, all_claims)
    summary_table = _claim_summary_table(displayed_claims, claim_type_ses)

    return f"""
<div style="margin-bottom:40px;border:1px solid #ddd;border-radius:8px;
     padding:20px;background:#fafafa;">
  <h2 style="margin-top:0;">
    #{rank} — Sample {sample_idx}
    <span style="font-size:14px;font-weight:normal;color:#666;">
      Mean SE = {mean_se:.4f} &nbsp;|&nbsp; Displaying generation {best_gen_idx}
    </span>
  </h2>
  {_colour_legend_html()}
  <div style="display:flex;gap:24px;align-items:flex-start;">
    <!-- Left panel: highlighted note + missing claims inline -->
    <div style="flex:1;min-width:0;">
      <h3 style="margin-top:0;">Generated Note (generation {best_gen_idx})</h3>
      <div style="font-family:Georgia,serif;font-size:14px;line-height:1.9;
           background:#fff;padding:16px;border:1px solid #e0e0e0;border-radius:6px;">
        {highlighted}
      </div>
    </div>
    <!-- Right panel: claim type summary -->
    <div style="width:420px;flex-shrink:0;">
      <h3 style="margin-top:0;">Claim Type Summary</h3>
      {summary_table}
    </div>
  </div>
</div>
"""


def build_html(
    top_samples: List[Tuple[Dict, List[Dict[str, Optional[str]]], Dict[str, float], float, int]],
) -> str:
    """Build the full HTML document."""
    sections = []
    for rank, (sample, all_claims, claim_type_ses, mean_se, best_gen_idx) in enumerate(
        top_samples, start=1
    ):
        sections.append(
            _sample_section_html(rank, sample, all_claims, claim_type_ses, mean_se, best_gen_idx)
        )

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Typed-Claim Semantic Entropy — Top-3 Uncertain Samples</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
      color: #222;
      background: #f7f7f7;
    }}
    h1 {{ margin-bottom: 8px; }}
    mark {{ color: #000; }}
    table {{ border-collapse: collapse; }}

    /* CSS hover tooltip */
    .tip-wrap {{ position: relative; }}
    .tip-box {{
      display: none;
      position: absolute;
      bottom: calc(100% + 6px);
      left: 50%;
      transform: translateX(-50%);
      background: #1e1e1e;
      color: #f0f0f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 12px;
      line-height: 1.5;
      padding: 8px 10px;
      border-radius: 6px;
      white-space: nowrap;
      min-width: 180px;
      max-width: 360px;
      white-space: normal;
      z-index: 999;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
      pointer-events: none;
    }}
    .tip-wrap:hover .tip-box {{ display: block; }}
    th, td {{ border: 1px solid #e0e0e0; }}
  </style>
</head>
<body>
  <h1>Typed-Claim Semantic Entropy — Top-3 Uncertain Samples</h1>
  <p style="color:#555;margin-bottom:28px;">
    Samples ranked by mean typed-claim semantic entropy across {len(CLAIM_TYPES)} claim types.
    Highlighted generation is the one with the highest within-sample mean SE.
    Hover over highlighted spans for claim type and SE value.
  </p>
  {body}
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute typed-claim semantic entropy and generate HTML highlight report."
    )
    parser.add_argument(
        "--gen-dir",
        default="generations",
        help="Directory containing sample_NNN_generations.json files (default: generations/)",
    )
    parser.add_argument(
        "--out",
        default="se_highlight.html",
        help="Output HTML file path (default: se_highlight.html)",
    )
    parser.add_argument(
        "--bedrock-region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="AWS region for Bedrock (default: AWS_DEFAULT_REGION env or us-east-1)",
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="NLI entailment threshold for atomic fact clustering (default: 0.5)",
    )
    parser.add_argument(
        "--cache",
        default="se_highlight_cache.json",
        help="Path to intermediate cache JSON (default: se_highlight_cache.json)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top uncertain samples to highlight (default: 3)",
    )
    args = parser.parse_args()

    gen_dir    = Path(args.gen_dir)
    cache_path = Path(args.cache)
    out_path   = Path(args.out)

    # ── Load generation files ──────────────────────────────────────────────
    samples = load_generations(gen_dir)

    # ── Load or build claim extraction cache ──────────────────────────────
    cache = load_cache(cache_path)
    # cache structure: { str(sample_idx): {"claims": [...], "decompositions": [...]} }
    # Old flat-list format is handled gracefully below.

    for sample in samples:
        sample_idx = sample["sample_idx"]
        key = str(sample_idx)
        notes = sample["notes"]
        K = len(notes)

        entry = cache.get(key)
        # Migrate old flat-list cache entries
        if isinstance(entry, list):
            entry = {"claims": entry, "decompositions": None}
            cache[key] = entry

        has_claims = entry is not None and entry.get("claims")
        has_decomps = entry is not None and entry.get("decompositions")

        if not has_claims:
            print(f"[sample {sample_idx}] Extracting claims from {K} notes via Bedrock ...")
            all_claims = []
            for gen_i, note in enumerate(notes):
                print(f"  note {gen_i+1}/{K} ...", end="\r", flush=True)
                all_claims.append(extract_claims_from_note(note, args.bedrock_region))
            print()
            if key not in cache or not isinstance(cache.get(key), dict):
                cache[key] = {}
            cache[key]["claims"] = all_claims
            save_cache(cache_path, cache)
            print(f"[sample {sample_idx}] Claims cached.")
        else:
            print(f"[sample {sample_idx}] Claims cached, skipping extraction.")

        if not has_decomps:
            all_claims = cache[key]["claims"]
            print(f"[sample {sample_idx}] Decomposing claims into atomic facts ...")
            all_decomps = []
            for gen_i, claims in enumerate(all_claims):
                print(f"  decomposing note {gen_i+1}/{K} ...", end="\r", flush=True)
                all_decomps.append(decompose_all_claims(claims, args.bedrock_region))
            print()
            cache[key]["decompositions"] = all_decomps
            save_cache(cache_path, cache)
            print(f"[sample {sample_idx}] Decompositions cached.")
        else:
            print(f"[sample {sample_idx}] Decompositions cached, skipping.")

    # ── Compute SE for each sample ─────────────────────────────────────────
    print("\n[SE] Computing typed-claim semantic entropy ...")
    sample_results: List[Tuple[Dict, List, Dict, float, int]] = []

    for sample in samples:
        sample_idx = sample["sample_idx"]
        key = str(sample_idx)
        all_claims: List[Dict[str, Optional[str]]] = cache[key]["claims"]
        all_decomps: List[Dict[str, List[str]]]    = cache[key]["decompositions"]
        K = len(all_claims)

        print(f"[sample {sample_idx}] Clustering atomic facts and computing SE ...")
        mean_se, claim_type_ses = compute_sample_se(all_decomps, K, args.sim_threshold)

        # Pick best generation: highest mean SE (here all share same claim_type_ses)
        # To differentiate, use the fraction of non-null claims * their SEs as proxy,
        # or more precisely: sum of SE for claim types that ARE present in this gen.
        def gen_se(claims_dict):
            return sum(
                claim_type_ses[ct]
                for ct in CLAIM_TYPES
                if claims_dict.get(ct) is not None
            ) / max(1, sum(1 for ct in CLAIM_TYPES if claims_dict.get(ct) is not None)
            ) if any(claims_dict.get(ct) is not None for ct in CLAIM_TYPES) else 0.0

        best_gen_idx = max(range(K), key=lambda i: gen_se(all_claims[i]))

        print(f"  [sample {sample_idx}] mean_se={mean_se:.4f}, best_gen={best_gen_idx}")
        sample_results.append((sample, all_claims, claim_type_ses, mean_se, best_gen_idx))

    # ── Pick top-N by mean SE ─────────────────────────────────────────────
    sample_results.sort(key=lambda x: x[3], reverse=True)
    top_samples = sample_results[: args.top_n]

    print(f"\n[top-{args.top_n}] Selected samples:")
    for rank, (sample, _, _, mean_se, best_gen_idx) in enumerate(top_samples, 1):
        print(f"  #{rank}: sample {sample['sample_idx']}, mean_se={mean_se:.4f}")

    # ── Generate HTML ──────────────────────────────────────────────────────
    print("\n[html] Generating HTML report ...")
    html = build_html(top_samples)
    out_path.write_text(html, encoding="utf-8")
    print(f"[html] Written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
