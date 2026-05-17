"""
run_se_batch.py
===============
Batch semantic-entropy experiment across all 44 rows of the ACI-Bench test1 split.

Methodology (Farquhar et al., NeurIPS 2024)
--------------------------------------------
For each transcript we generate K=10 notes using:
  • Temperature sampling  (temperature > 0 to produce diverse outputs)
  • Slight prompt variations (6 phrasings rotated across generations) to
    encourage semantic diversity without changing the underlying task —
    following the paper's recommendation that prompt perturbation improves
    cluster separation compared to temperature alone.

Semantic entropy is then computed via atomic claim clustering:
  1. Extract atomic claims from each of the K generated notes (split on
     sentence boundaries, semicolons, and conjunctions; assign SOAP section).
  2. Pool all K×N claims; run pairwise bidirectional NLI entailment.
  3. Connected-component clustering on the entailment adjacency matrix.
  4. For each cluster c spanning k_c of K notes:
       SE[c] = H_binary(k_c / K)  — binary presence entropy.
     A cluster present in all K notes → SE=0; present in ~half → SE≈1.
  5. Each token in the reference note inherits the SE of the cluster whose
     centroid claim overlaps that token's character span.

Additionally we compute token-level predictive entropy:
  H[t] = -Σ_v  p(v | context_t) * log p(v | context_t)
from the logit distribution at each note token position. This is a genuine
token-level signal (independent of NLI clustering) and is fast — one forward
pass per sample.

Outputs (written to --out / se_batch_out/)
------------------------------------------
  se_batch_results.csv       — per-sample summary statistics
  token_scores/sample_NNN_tokens.csv  — per-token SE + predictive entropy
  top3_report.html           — highlighted HTML for the 3 most uncertain transcripts

Usage
-----
  python run_se_batch.py
  python run_se_batch.py --model llama --K 10 --temperature 0.8
  python run_se_batch.py --start 0 --end 44 --out results/
"""

import argparse
import os
import re
import warnings
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from config import Config, load_aci_sample
from tokenization import _GENERATION_PROMPT

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt variations  (Farquhar et al. §3 — prompt perturbation)
#
# Core structure adapted from Asgari et al. (2025) npj Digital Medicine,
# Experiment 8 — the best-performing configuration in their clinical SOAP-note
# benchmark.  All six variants share:
#   • "DO NOT ADD any content that isn't specifically mentioned IN THE TRANSCRIPT"
#   • SOAP template (Subjective/Objective/Assessment/Plan)
#   • Style preferences: first person, ultra-concise, bullet points,
#     [NOT MENTIONED] for absent information, drug allergies always documented,
#     examination findings = physical exam signs only, preserve quantities,
#     include ALL negations.
#   • Only the framing / persona sentence and minor phrasing differ across
#     variants, following the prompt-perturbation strategy of Farquhar et al.
# ─────────────────────────────────────────────────────────────────────────────

# Shared SOAP template block (Asgari et al. 2025, Table 1, Exp 8)
_SOAP_TEMPLATE = """\
SUBJECTIVE:
- Chief Complaint:
- History of Present Illness:
- Past Medical History:
- Review of Systems:
- Current Medications: [DRUG NAME] [DOSE] [FREQUENCY] [INDICATION]
- Drug Allergies:

OBJECTIVE:
- Vital Signs:
- Physical Exam:
- Test Results:

ASSESSMENT/PROBLEM LIST:

PLAN:

FOLLOW-UP:\
"""

# Shared style block (Asgari et al. 2025, Table 1, Exp 8)
_STYLE_PREFERENCES = """\
Style preferences:
- Write in first person from the physician's perspective.
- Be ultra-concise and ultra-precise; use bullet points throughout.
- Use [NOT MENTIONED] for any template field not discussed in the transcript.
- Always document drug allergies even if the patient reports none.
- Examination findings = physical exam signs only (do not conflate with symptoms).
- Preserve exact quantities, doses, frequencies, and durations as stated.
- Include ALL negations (e.g. "denies chest pain", "no fever").
- Do not add any content not explicitly mentioned in the transcript.
- Do not repeat information across sections.\
"""

_PROMPT_VARIANTS = [
    # Variant 0 — medical office assistant (closest to Asgari et al. Exp 8 wording)
    (
        "You are a medical office assistant drafting documentation for a physician.\n"
        "DO NOT ADD any content that isn't specifically mentioned IN THE TRANSCRIPT.\n"
        "From the transcript below, generate a SOAP note for the physician to review. "
        "Include all relevant information and omit anything not explicitly mentioned. "
        "If nothing is mentioned for a section, return [NOT MENTIONED]. "
        "It is VITAL that all information in the note is as accurate as possible. "
        "Avoid repeating the same information in different sections. "
        "Write the note from the perspective of the physician.\n\n"
        f"{_STYLE_PREFERENCES}\n\n"
        f"Use this template:\n{_SOAP_TEMPLATE}\n\n"
        "Transcript:\n{transcript}\n\n"
        "SUBJECTIVE:\n- Chief Complaint:"
    ),
    # Variant 1 — clinical scribe framing
    (
        "You are a clinical scribe producing an accurate SOAP note for the attending physician.\n"
        "DO NOT ADD any content that isn't specifically mentioned IN THE TRANSCRIPT.\n"
        "Document the encounter below faithfully. "
        "If a template field was not discussed, write [NOT MENTIONED]. "
        "Accuracy is paramount — do not invent or infer details beyond what is stated.\n\n"
        f"{_STYLE_PREFERENCES}\n\n"
        f"Template:\n{_SOAP_TEMPLATE}\n\n"
        "Encounter transcript:\n{transcript}\n\n"
        "SUBJECTIVE:\n- Chief Complaint:"
    ),
    # Variant 2 — physician dictation framing
    (
        "You are an attending physician dictating a SOAP note immediately after the visit.\n"
        "Only document what was explicitly stated in the transcript — "
        "DO NOT ADD inferences or assumptions.\n"
        "Use [NOT MENTIONED] for any section with no relevant transcript content.\n\n"
        f"{_STYLE_PREFERENCES}\n\n"
        f"Follow this structure exactly:\n{_SOAP_TEMPLATE}\n\n"
        "Visit transcript:\n{transcript}\n\n"
        "SUBJECTIVE:\n- Chief Complaint:"
    ),
    # Variant 3 — EHR documentation specialist framing
    (
        "You are an EHR documentation specialist generating a structured SOAP note.\n"
        "Rule: DO NOT ADD any content not explicitly mentioned in the transcript below.\n"
        "All absent fields must be marked [NOT MENTIONED]. "
        "The physician will review this note for clinical decision-making — "
        "accuracy and completeness are critical.\n\n"
        f"{_STYLE_PREFERENCES}\n\n"
        f"Required format:\n{_SOAP_TEMPLATE}\n\n"
        "Transcript:\n{transcript}\n\n"
        "SUBJECTIVE:\n- Chief Complaint:"
    ),
    # Variant 4 — medicolegal / quality assurance framing
    (
        "You are preparing a medicolegal clinical note from a recorded patient encounter.\n"
        "CRITICAL: include only information explicitly stated in the transcript. "
        "DO NOT ADD diagnoses, findings, or medications not mentioned by the clinician or patient.\n"
        "Mark any unanswered template field as [NOT MENTIONED].\n\n"
        f"{_STYLE_PREFERENCES}\n\n"
        f"Note template:\n{_SOAP_TEMPLATE}\n\n"
        "Patient encounter transcript:\n{transcript}\n\n"
        "SUBJECTIVE:\n- Chief Complaint:"
    ),
    # Variant 5 — handover / continuity of care framing
    (
        "You are creating a SOAP note to support continuity of care between clinicians.\n"
        "The receiving provider depends on this note being complete and accurate. "
        "DO NOT ADD any content beyond what is discussed in the transcript. "
        "If information for a field is absent, write [NOT MENTIONED].\n\n"
        f"{_STYLE_PREFERENCES}\n\n"
        f"Use the following template:\n{_SOAP_TEMPLATE}\n\n"
        "Transcript:\n{transcript}\n\n"
        "SUBJECTIVE:\n- Chief Complaint:"
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Generation with prompt variation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_with_variant(
    model: HookedTransformer,
    transcript: str,
    cfg: Config,
    variant_idx: int,
    temperature: float,
) -> str:
    """
    Generate one note using prompt variant `variant_idx` at the given temperature.
    """
    prompt_tmpl = _PROMPT_VARIANTS[variant_idx % len(_PROMPT_VARIANTS)]
    prompt      = prompt_tmpl.format(transcript=transcript.strip())
    input_ids   = model.to_tokens(prompt, prepend_bos=True).to(cfg.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            stop_at_eos=True,
            verbose=False,
        )

    generated_ids = out[0, input_ids.shape[1]:]
    return model.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Token-level predictive entropy
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_predictive_entropy(
    model: HookedTransformer,
    transcript: str,
    note: str,
    cfg: Config,
) -> np.ndarray:
    """
    Single forward pass → per-note-token predictive entropy.

    H[t] = -Σ_v  p(v | x_{<t}) * log p(v | x_{<t})

    Uses the standard generation prompt prefix so the conditioning matches
    what was used during generation.

    Returns
    -------
    entropy : (note_len,) float64  — token-level predictive entropy in nats.
    """
    prompt     = _GENERATION_PROMPT.format(transcript=transcript.strip())
    prompt_tok = model.to_tokens(prompt,  prepend_bos=True).to(cfg.device)   # (1, P)
    note_tok   = model.to_tokens(note,    prepend_bos=False).to(cfg.device)  # (1, N)
    full_seq   = torch.cat([prompt_tok, note_tok], dim=1)                    # (1, P+N)
    P          = prompt_tok.shape[1]
    N          = note_tok.shape[1]

    with torch.no_grad():
        logits = model(full_seq)   # (1, P+N, V)

    # Note logits: at position P-1 … P+N-2 we predict note tokens 0 … N-1
    # i.e. logits[:, P-1 : P+N-1, :] predicts note_tok positions 0..N-1
    note_logits = logits[0, P - 1 : P + N - 1, :].float()   # (N, V)
    log_probs   = torch.log_softmax(note_logits, dim=-1)      # (N, V)
    probs       = log_probs.exp()                             # (N, V)
    entropy     = -(probs * log_probs).sum(dim=-1).cpu().numpy()  # (N,)
    return entropy.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# NLI helper — entailment index resolved from model config
# ─────────────────────────────────────────────────────────────────────────────

def _get_entailment_idx(nli_model) -> int:
    """
    Return the column index for the ENTAILMENT class in this NLI model's output.

    Many NLI models use the label order (contradiction=0, neutral=1, entailment=2),
    but some use (entailment=0, neutral=1, contradiction=2).  We resolve this
    by inspecting the model's id2label config rather than hardcoding an index.
    Falls back to index 0 if the config is unavailable.
    """
    try:
        id2label = nli_model.model.config.id2label  # {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        for idx, label in id2label.items():
            if "entail" in label.lower():
                return int(idx)
    except AttributeError:
        pass
    # Safe fallback: nli-deberta-v3-small uses contradiction=0, neutral=1, entailment=2
    return 2


def _bfs_components(adj: np.ndarray) -> List[List[int]]:
    """Connected components of a boolean adjacency matrix via BFS."""
    n       = adj.shape[0]
    visited = [False] * n
    components: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        cluster, queue = [], [start]
        while queue:
            node = queue.pop()
            if visited[node]:
                continue
            visited[node] = True
            cluster.append(node)
            queue.extend(j for j in range(n) if adj[node, j] and not visited[j])
        components.append(cluster)
    return components


# ─────────────────────────────────────────────────────────────────────────────
# Atomic claim extraction  (Bedrock LLM)
# ─────────────────────────────────────────────────────────────────────────────

_CLAIM_EXTRACTOR_DEFAULT_MODEL  = "us.meta.llama3-3-70b-instruct-v1:0"
_CLAIM_EXTRACTOR_DEFAULT_REGION = "us-east-1"

_CLAIM_SYSTEM_PROMPT = (
    "You are a clinical NLP specialist extracting atomic facts from clinical notes.\n"
    "Rules:\n"
    "1. Extract each distinct clinical fact as a short, self-contained statement.\n"
    "2. Stay close to the wording in the note — minimal rephrasing only where needed "
    "   for grammatical completeness.\n"
    "3. If a bullet or sentence contains multiple independent facts, split them.\n"
    "4. Skip lines that say [NOT MENTIONED] or are purely field labels "
    "   (e.g. 'Chief Complaint:', 'Drug Allergies:').\n"
    "5. Output ONLY the numbered list — no preamble, no commentary.\n"
    "Format exactly:\n"
    "1. claim text\n"
    "2. claim text\n"
    "..."
)

_CLAIM_USER_TEMPLATE = (
    "Extract every atomic clinical fact from the note below as a numbered list.\n\n"
    "NOTE:\n{note}\n\n"
    "Numbered list:"
)


# Lazy-initialised Bedrock client (created on first call)
_bedrock_claim_client = None


def _get_bedrock_claim_client(
    model: str  = _CLAIM_EXTRACTOR_DEFAULT_MODEL,
    region: str = _CLAIM_EXTRACTOR_DEFAULT_REGION,
):
    """Return (client, model_id), initialising boto3 on first call."""
    global _bedrock_claim_client
    if _bedrock_claim_client is None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for LLM claim extraction.  "
                "Install with:  pip install boto3"
            ) from exc
        resolved_region = region or os.environ.get("AWS_DEFAULT_REGION",
                                                    _CLAIM_EXTRACTOR_DEFAULT_REGION)
        _bedrock_claim_client = (
            boto3.client("bedrock-runtime", region_name=resolved_region),
            model,
        )
    return _bedrock_claim_client


def _extract_claims(
    note: str,
    bedrock_model: str  = _CLAIM_EXTRACTOR_DEFAULT_MODEL,
    bedrock_region: str = _CLAIM_EXTRACTOR_DEFAULT_REGION,
    max_tokens: int     = 1024,
    temperature: float  = 0.0,
) -> List[Dict]:
    """
    Use a Bedrock LLM to decompose a clinical note into atomic claims.

    Returns a list of dicts:
        text       : str  — the claim text
        section    : str  — SOAP section label inferred from position in note
        char_start : int  — start offset in note  (-1 if not locatable)
        char_end   : int  — end offset in note
    """
    client, model_id = _get_bedrock_claim_client(bedrock_model, bedrock_region)

    user_msg = _CLAIM_USER_TEMPLATE.format(note=note.strip())

    response = client.converse(
        modelId=model_id,
        system=[{"text": _CLAIM_SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_msg}]}],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )

    raw = response["output"]["message"]["content"][0]["text"]

    # Parse numbered list: "1. claim text"
    claims: List[Dict] = []
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s+(.+)$", line)
        if not m:
            continue
        text = m.group(1).strip()
        if len(text) < 6:
            continue
        claims.append({"text": text})

    return claims


# ─────────────────────────────────────────────────────────────────────────────
# Atomic-claim semantic entropy
# ─────────────────────────────────────────────────────────────────────────────

def _compute_claim_se(
    all_claims_per_note: List[List[Dict]],
    nli_model,
    nli_threshold: float,
    K: int,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Pool all atomic claims across K notes, cluster semantically equivalent ones,
    then compute per-cluster binary entropy over claim presence.

    Missing claims (a cluster absent from some notes) are treated as their own
    evidence: if only k out of K notes include a claim cluster, the entropy is
    H_binary(k/K) = -(k/K)log(k/K) - ((K-k)/K)log((K-k)/K).
    High entropy → uncertain whether this fact belongs in the note.

    Returns
    -------
    clusters  : list of cluster dicts:
                    members      : list of claim dicts
                    k_present    : int — how many of the K notes contain it
                    se           : float — binary entropy
                    section      : str — majority section
    claim_se  : (total_ref_claims,) — SE score for each claim in note 0 (reference)
    """
    entail_idx = _get_entailment_idx(nli_model)

    # Reference note is notes[0]; we will score its claims
    ref_claims = all_claims_per_note[0]

    # Pool all claims with a note-of-origin tag
    all_claims: List[Dict] = []
    for note_idx, claims in enumerate(all_claims_per_note):
        for c in claims:
            all_claims.append({**c, "_note_idx": note_idx})

    if not all_claims:
        return [], np.zeros(len(ref_claims))

    # Pairwise NLI among all claims (batched)
    texts = [c["text"] for c in all_claims]
    n     = len(texts)
    pairs = [(texts[i], texts[j]) for i in range(n) for j in range(n) if i != j]

    try:
        raw = np.array(nli_model.predict(pairs, apply_softmax=True))
        if raw.ndim == 2:
            entail_scores = raw[:, entail_idx]
        else:
            entail_scores = raw
    except Exception as exc:
        print(f"    [SE] NLI predict failed: {exc}")
        return [], np.zeros(len(ref_claims))

    # Build symmetric entailment adjacency: both A→B and B→A must exceed threshold
    adj = np.zeros((n, n), dtype=bool)
    pair_idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i, j] = entail_scores[pair_idx] > nli_threshold
                pair_idx += 1
    sym = adj & adj.T

    # Cluster via connected components
    raw_components = _bfs_components(sym)

    # Build cluster records
    eps = 1e-10
    clusters: List[Dict] = []
    for comp in raw_components:
        members     = [all_claims[i] for i in comp]
        note_ids    = {m["_note_idx"] for m in members}
        k_present   = len(note_ids)
        # Binary presence entropy: H(k/K) + H((K-k)/K)
        p = k_present / K
        se = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps)) if 0 < p < 1 else 0.0
        clusters.append({
            "members":   members,
            "k_present": k_present,
            "se":        se,
        })

    # Assign SE to each reference claim via cluster membership
    # Build a map: claim text → cluster SE
    claim_to_se: Dict[str, float] = {}
    for cluster in clusters:
        for m in cluster["members"]:
            if m["_note_idx"] == 0:
                claim_to_se[m["text"]] = cluster["se"]

    claim_se = np.array([
        claim_to_se.get(c["text"], 0.0) for c in ref_claims
    ], dtype=np.float64)

    print(f"    [SE] {n} claims pooled → {len(clusters)} clusters  "
          f"({len(ref_claims)} ref claims)")

    return clusters, claim_se


# ─────────────────────────────────────────────────────────────────────────────
# Token-level SE mapping
# ─────────────────────────────────────────────────────────────────────────────

def _map_claims_to_tokens(
    note: str,
    ref_claims: List[Dict],
    claim_se: np.ndarray,
    model,
    note_len: int,
) -> np.ndarray:
    """
    Assign each note token the SE of the claim whose char span covers it.
    Tokens not covered by any claim get SE = 0.
    """
    token_se = np.zeros(note_len, dtype=np.float64)
    try:
        enc     = model.tokenizer(note, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
    except Exception:
        return token_se

    # Build char→SE lookup by searching each claim text in the note
    char_se = np.zeros(len(note), dtype=np.float64)
    for ci, claim in enumerate(ref_claims):
        text = claim["text"]
        idx  = note.find(text)
        if idx != -1:
            char_se[idx : idx + len(text)] = claim_se[ci]

    for ti, (cs, ce) in enumerate(offsets):
        if ti >= note_len:
            break
        mid = int((cs + ce) / 2.0)
        if mid < len(char_se):
            token_se[ti] = char_se[mid]
    return token_se


# ─────────────────────────────────────────────────────────────────────────────
# Main per-sample computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_se_for_sample(
    model: HookedTransformer,
    transcript: str,
    cfg: Config,
    K: int = 10,
    temperature: float = 0.8,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    nli_threshold: float = 0.5,
    claim_bedrock_model: str  = _CLAIM_EXTRACTOR_DEFAULT_MODEL,
    claim_bedrock_region: str = _CLAIM_EXTRACTOR_DEFAULT_REGION,
) -> Optional[Dict]:
    """
    Generate K notes → extract atomic claims → cluster with NLI → binary-entropy SE.

    Key methodological improvements over sentence-position alignment:
    • Atomic claim extraction splits on sentence boundaries, semicolons, and
      conjunctions → smaller, more comparable units.
    • Claims are pooled across ALL K notes and clustered globally, so two notes
      that express the same fact in different sentences are correctly merged.
    • Absent claims (cluster present in k < K notes) contribute
      H_binary(k/K) entropy — absence is treated as meaningful evidence of
      uncertainty, not ignored.
    • NLI entailment index is resolved from model.config.id2label — not hardcoded.

    Returns dict with keys:
        notes            : List[str]
        token_se         : (note_len,) — claim-cluster SE per token
        token_pred_ent   : (note_len,) — predictive entropy per token
        note_tokens      : List[str]
        note_len         : int
        mean_se, mean_pred_ent : float
        clusters         : cluster dicts (k_present, se, section, members)
        ref_claims       : atomic claims from reference note (note 0)
        claim_se         : (n_ref_claims,) — SE per reference claim
    """
    from metrics import _get_nli_model

    # ── Step 1: K generations ─────────────────────────────────────────────────
    notes: List[str] = []
    for k in range(K):
        try:
            note = _generate_with_variant(model, transcript, cfg,
                                          variant_idx=k, temperature=temperature)
            notes.append(note)
        except Exception as exc:
            print(f"    [SE] generation {k+1} failed: {exc}")

    if len(notes) < 2:
        print("    [SE] fewer than 2 notes generated — skipping.")
        return None

    K_actual = len(notes)
    print(f"    [SE] {K_actual} notes generated")

    # ── Step 2: Extract atomic claims from each note (via Bedrock LLM) ────────
    all_claims_per_note = [
        _extract_claims(n,
                        bedrock_model=claim_bedrock_model,
                        bedrock_region=claim_bedrock_region)
        for n in notes
    ]
    total_claims        = sum(len(c) for c in all_claims_per_note)
    print(f"    [SE] {total_claims} atomic claims extracted across {K_actual} notes")

    # ── Step 3: Load NLI model ────────────────────────────────────────────────
    try:
        nli_model = _get_nli_model(nli_model_name)
    except Exception as exc:
        print(f"    [SE] NLI model load failed: {exc}")
        return None

    # ── Step 4: Cluster claims + compute binary-entropy SE ───────────────────
    clusters, claim_se = _compute_claim_se(
        all_claims_per_note, nli_model, nli_threshold, K_actual
    )
    ref_claims = all_claims_per_note[0]

    # ── Step 5: Map claim SE → token level ───────────────────────────────────
    ref_note    = notes[0]
    note_ids    = model.tokenizer.encode(ref_note, add_special_tokens=False)
    note_len    = len(note_ids)
    note_tokens = [model.tokenizer.decode([tid]) for tid in note_ids]

    token_se = _map_claims_to_tokens(ref_note, ref_claims, claim_se, model, note_len)

    # ── Step 6: Token predictive entropy (one forward pass) ──────────────────
    token_pred_ent = compute_token_predictive_entropy(model, transcript, ref_note, cfg)

    min_len        = min(note_len, len(token_pred_ent))
    token_se       = token_se[:min_len]
    token_pred_ent = token_pred_ent[:min_len]
    note_tokens    = note_tokens[:min_len]
    note_len       = min_len

    print(f"    [SE] mean_SE={token_se.mean():.4f}  "
          f"mean_PredEnt={token_pred_ent.mean():.4f}  "
          f"note_len={note_len}")

    return {
        "notes":          notes,
        "token_se":       token_se,
        "token_pred_ent": token_pred_ent,
        "note_tokens":    note_tokens,
        "note_len":       note_len,
        "mean_se":        float(token_se.mean()),
        "mean_pred_ent":  float(token_pred_ent.mean()),
        "clusters":       clusters,
        "ref_claims":     ref_claims,
        "claim_se":       claim_se,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _colour_from_entropy(val: float, max_val: float) -> str:
    """Map a normalised entropy value to a red-tinted background colour."""
    p = min(val / (max_val + 1e-9), 1.0)
    r = int(200 + 55 * p)
    g = int(240 * (1 - p))
    b = int(200 * (1 - p))
    return f"rgb({r},{g},{b})"


def build_top3_html(
    top3: List[Dict],
    out_path: Path,
) -> None:
    """
    Build a single HTML file showing token-level uncertainty highlights for the
    3 most uncertain transcripts.  Each section shows:
      • The transcript (plain)
      • The generated note with tokens coloured by BOTH semantic entropy (border)
        and predictive entropy (background fill) so the two signals are readable
        simultaneously.
    """
    sections = []

    for rank, item in enumerate(top3, start=1):
        si          = item["sample_idx"]
        transcript  = item["transcript"]
        ref_note    = item["notes"][0]
        tokens      = item["note_tokens"]
        se          = item["token_se"]
        pred_ent    = item["token_pred_ent"]
        mean_se     = item["mean_se"]
        mean_pe     = item["mean_pred_ent"]

        max_se  = float(se.max())  if se.max()  > 0 else 1.0
        max_pe  = float(pred_ent.max()) if pred_ent.max() > 0 else 1.0

        token_spans = []
        for tok, s_val, pe_val in zip(tokens, se, pred_ent):
            tok_html = (tok
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
            bg      = _colour_from_entropy(pe_val, max_pe)
            border  = _colour_from_entropy(s_val,  max_se)
            tip     = f"SE={s_val:.3f}  PredEnt={pe_val:.3f}"
            token_spans.append(
                f'<span style="background:{bg};'
                f'border-bottom:3px solid {border};'
                f'border-radius:2px;padding:1px 1px;margin:0 1px;" '
                f'title="{tip}">{tok_html}</span>'
            )

        note_html = "".join(token_spans)

        tr_preview = (transcript[:600] + "…") if len(transcript) > 600 else transcript
        tr_html = tr_preview.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")

        sections.append(f"""
<div class="sample">
  <h2>#{rank} — Sample {si}
    <span class="badge">mean SE={mean_se:.3f}</span>
    <span class="badge2">mean PredEnt={mean_pe:.3f}</span>
  </h2>

  <h3>Transcript (excerpt)</h3>
  <div class="transcript">{tr_html}</div>

  <h3>Generated Note — token uncertainty
    <span class="legend">
      <span style="background:rgb(255,100,100);padding:2px 6px;border-radius:3px">fill = predictive entropy</span>
      &nbsp;
      <span style="border-bottom:3px solid rgb(255,100,100);padding:2px 6px">underline = semantic entropy</span>
    </span>
  </h3>
  <div class="note">{note_html}</div>
</div>
<hr>
""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Semantic Entropy — Top 3 Uncertain Transcripts</title>
<style>
  body      {{ font-family: Georgia, serif; font-size: 14px; line-height: 1.9;
               max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1        {{ font-family: sans-serif; border-bottom: 2px solid #888; padding-bottom: 8px; }}
  h2        {{ font-family: sans-serif; margin-top: 36px; color: #333; }}
  h3        {{ font-family: sans-serif; font-size: 13px; color: #555;
               margin-top: 18px; margin-bottom: 6px; }}
  .badge    {{ font-size: 12px; background: #e74c3c; color: white;
               border-radius: 12px; padding: 2px 10px; margin-left: 10px;
               vertical-align: middle; }}
  .badge2   {{ font-size: 12px; background: #8e44ad; color: white;
               border-radius: 12px; padding: 2px 10px; margin-left: 6px;
               vertical-align: middle; }}
  .transcript {{ background: #f8f8f8; border-left: 4px solid #aaa;
                 padding: 10px 16px; font-family: monospace; font-size: 12px;
                 white-space: pre-wrap; margin-bottom: 12px; }}
  .note     {{ line-height: 2.4; font-family: monospace; font-size: 13px;
               background: #fff; border: 1px solid #ddd;
               padding: 14px; border-radius: 4px; }}
  .legend   {{ font-size: 11px; font-weight: normal; color: #555; margin-left: 12px; }}
  hr        {{ border: none; border-top: 1px solid #ddd; margin: 30px 0; }}
</style>
</head>
<body>
<h1>Semantic Entropy — Top 3 Most Uncertain Transcripts</h1>
<p style="font-size:12px;color:#666;">
  <b>Background fill</b> = token predictive entropy H[t] (logit distribution — one forward pass).<br>
  <b>Underline colour</b> = sentence-level semantic entropy (NLI clustering over K=10 generations).<br>
  Deeper red → higher uncertainty → higher hallucination risk.
  Hover over a token to see exact values.
</p>
{"".join(sections)}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main batch loop
# ─────────────────────────────────────────────────────────────────────────────

def run_se_batch(
    model: HookedTransformer,
    cfg: Config,
    out: Path,
    start: int = 0,
    end: int   = 44,
    K: int     = 10,
    temperature: float = 0.8,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    claim_bedrock_model: str  = _CLAIM_EXTRACTOR_DEFAULT_MODEL,
    claim_bedrock_region: str = _CLAIM_EXTRACTOR_DEFAULT_REGION,
) -> pd.DataFrame:
    """
    Run semantic entropy over ACI-Bench test1 rows [start, end).
    """
    se_out      = out / "se_batch_out"
    token_dir   = se_out / "token_scores"
    claim_dir   = se_out / "claim_scores"
    se_out.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    claim_dir.mkdir(parents=True, exist_ok=True)

    summary_rows   = []
    all_results    = {}   # sample_idx → result dict

    for si in range(start, end):
        print(f"\n{'='*54}")
        print(f"  Sample {si}  ({si - start + 1}/{end - start})")
        print(f"{'='*54}")

        cfg_i = _dc_replace(cfg, sample_idx=si, gen_temperature=temperature)
        try:
            transcript, gold_note = load_aci_sample(cfg_i)
        except Exception as exc:
            print(f"  load failed: {exc}")
            continue

        print(f"  Transcript length: {len(transcript)} chars")

        result = compute_se_for_sample(
            model, transcript, cfg_i,
            K=K, temperature=temperature,
            nli_model_name=nli_model_name,
            claim_bedrock_model=claim_bedrock_model,
            claim_bedrock_region=claim_bedrock_region,
        )
        if result is None:
            summary_rows.append({
                "sample_idx": si,
                "status": "failed",
                "mean_se": float("nan"),
                "mean_pred_ent": float("nan"),
                "note_len": 0,
                "K_generated": 0,
            })
            continue

        all_results[si] = {**result, "sample_idx": si,
                           "transcript": transcript, "gold_note": gold_note}

        # ── Save raw generations ──────────────────────────────────────────────
        gen_dir = se_out / "generations"
        gen_dir.mkdir(exist_ok=True)
        gen_path = gen_dir / f"sample_{si:03d}_generations.json"
        import json
        with open(gen_path, "w") as f:
            json.dump({
                "sample_idx": si,
                "transcript": transcript,
                "gold_note":  gold_note,
                "notes":      result["notes"],
            }, f, indent=2)

        # ── Save per-token scores ─────────────────────────────────────────────
        tok_df = pd.DataFrame({
            "token_idx":          np.arange(result["note_len"]),
            "token_str":          result["note_tokens"],
            "semantic_entropy":   result["token_se"].round(4),
            "predictive_entropy": result["token_pred_ent"].round(4),
        })
        tok_df.to_csv(token_dir / f"sample_{si:03d}_tokens.csv", index=False)

        # ── Save per-claim scores ─────────────────────────────────────────────
        ref_claims = result.get("ref_claims", [])
        claim_se   = result.get("claim_se",   np.array([]))
        K_actual   = len(result["notes"])
        claim_rows = []
        for ci, claim in enumerate(ref_claims):
            se_val   = float(claim_se[ci]) if ci < len(claim_se) else float("nan")
            # Find which cluster this claim belongs to, to get k_present
            k_present = None
            for cluster in result.get("clusters", []):
                if any(m["text"] == claim["text"] and m["_note_idx"] == 0
                       for m in cluster["members"]):
                    k_present = cluster["k_present"]
                    break
            claim_rows.append({
                "claim_idx":        ci,
                "claim_text":       claim["text"],
                "k_present":        k_present if k_present is not None else float("nan"),
                "p_present":        round(k_present / K_actual, 4)
                                    if k_present is not None else float("nan"),
                "semantic_entropy": round(se_val, 4),
            })
        if claim_rows:
            pd.DataFrame(claim_rows).to_csv(
                claim_dir / f"sample_{si:03d}_claims.csv", index=False
            )

        # ── Summary row ───────────────────────────────────────────────────────
        summary_rows.append({
            "sample_idx":   si,
            "status":       "ok",
            "mean_se":      round(result["mean_se"],       4),
            "max_se":       round(float(result["token_se"].max()),       4),
            "mean_pred_ent": round(result["mean_pred_ent"],              4),
            "max_pred_ent": round(float(result["token_pred_ent"].max()), 4),
            "note_len":     result["note_len"],
            "K_generated":  len(result["notes"]),
        })

        print(f"  mean_SE={result['mean_se']:.4f}  "
              f"mean_PredEnt={result['mean_pred_ent']:.4f}  "
              f"note_len={result['note_len']}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(se_out / "se_batch_results.csv", index=False)
    print(f"\n  Saved → se_batch_results.csv  ({len(summary_rows)} rows)")

    # ── Top 3 by mean semantic entropy → HTML ────────────────────────────────
    if not all_results:
        print("  No successful results — skipping HTML report.")
        return summary_df

    ok_rows = summary_df[summary_df["status"] == "ok"].sort_values(
        "mean_se", ascending=False
    )
    top3_indices = ok_rows["sample_idx"].head(3).tolist()
    top3_data    = [all_results[si] for si in top3_indices if si in all_results]

    if top3_data:
        build_top3_html(top3_data, se_out / "top3_report.html")
        print(f"\n  Top 3 most uncertain samples (by mean SE): {top3_indices}")
        for si in top3_indices:
            r = all_results.get(si)
            if r:
                print(f"    Sample {si}: mean_SE={r['mean_se']:.4f}  "
                      f"mean_PredEnt={r['mean_pred_ent']:.4f}")

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch semantic entropy over ACI-Bench test1"
    )
    p.add_argument("--model", choices=["gemma", "llama"], default="gemma")
    p.add_argument("--start",       type=int,   default=0,    help="First sample index (inclusive)")
    p.add_argument("--end",         type=int,   default=44,   help="Last sample index (exclusive)")
    p.add_argument("--K",           type=int,   default=10,   help="Generations per sample")
    p.add_argument("--temperature", type=float, default=0.8,  help="Sampling temperature")
    p.add_argument("--nli-model",   default="cross-encoder/nli-deberta-v3-small")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out",         default=".", help="Output root directory")
    # Bedrock claim extractor
    p.add_argument("--claim-bedrock-model",
                   default=_CLAIM_EXTRACTOR_DEFAULT_MODEL,
                   help="Bedrock model ID for atomic claim extraction")
    p.add_argument("--claim-bedrock-region",
                   default=_CLAIM_EXTRACTOR_DEFAULT_REGION,
                   help="AWS region for Bedrock claim extraction calls")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        model_name=(
            "google/gemma-2-2b-it"
            if args.model == "gemma"
            else "meta-llama/Meta-Llama-3-8B-instruct"
        ),
        sample_idx=args.start,
        max_new_tokens=args.max_new_tokens,
        gen_temperature=args.temperature,
        output_dir=args.out,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Model          : {cfg.model_name}")
    print(f"  Samples        : {args.start} – {args.end - 1}  ({args.end - args.start} total)")
    print(f"  K generations  : {args.K}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  NLI model      : {args.nli_model}")
    print(f"  Claim extractor: {args.claim_bedrock_model}  ({args.claim_bedrock_region})")
    print(f"  Output dir     : {out.resolve()}")

    print(f"\nLoading {cfg.model_name} …")
    model = HookedTransformer.from_pretrained(
        cfg.model_name,
        dtype=cfg.dtype,
        default_padding_side="right",
    )
    model.eval()
    model.to(cfg.device)
    print(f"  Layers={model.cfg.n_layers}  Heads={model.cfg.n_heads}  d_model={model.cfg.d_model}")

    run_se_batch(
        model, cfg, out,
        start=args.start,
        end=args.end,
        K=args.K,
        temperature=args.temperature,
        nli_model_name=args.nli_model,
        claim_bedrock_model=args.claim_bedrock_model,
        claim_bedrock_region=args.claim_bedrock_region,
    )

    print("\n" + "═" * 54)
    print(f"  Done.  Results in: {out.resolve() / 'se_batch_out'}")
    print("═" * 54 + "\n")


if __name__ == "__main__":
    main()
