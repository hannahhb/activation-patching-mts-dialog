"""
halluc_llm.py
=============
LLM-backed hallucination injection for Experiment 2c.

The LLM reads the original clinical note and suggests specific (original,
replacement, category) substitution pairs.  We apply those substitutions
ourselves so that character-level span tracking stays exact — the LLM never
rewrites the note directly.

Backends
--------
  hf       HuggingFace Inference API  (huggingface_hub.InferenceClient)
  bedrock  AWS Bedrock Converse API   (boto3, model-agnostic)

Usage (standalone)
------------------
    from halluc_llm import inject_hallucinations_llm

    note = "Patient is on albuterol 2 puffs QID and Singulair 10 mg nightly."
    corrupted, injections = inject_hallucinations_llm(note, max_injections=2, backend="hf")

Usage via run_experiments.py
-----------------------------
    python run_experiments.py --exp 2c --halluc-backend hf
    python run_experiments.py --exp 2c --halluc-backend bedrock

Environment variables
---------------------
  HF_TOKEN              HuggingFace access token   (required for backend="hf")
  AWS_DEFAULT_REGION    AWS region                  (required for backend="bedrock")
  AWS_ACCESS_KEY_ID     }  Standard AWS credential
  AWS_SECRET_ACCESS_KEY }  chain (IAM role also works)

Model overrides (kwargs to inject_hallucinations_llm)
-----------------------------------------------------
  hf_model      str   HF model ID   (default: "Qwen/Qwen2.5-72B-Instruct")
  hf_token      str   override HF_TOKEN env var
  bedrock_model str   Bedrock model ID  (default: "anthropic.claude-3-haiku-20240307-v1:0")
  bedrock_region str  override AWS_DEFAULT_REGION env var
"""

import json
import os
import random
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a clinical documentation auditor specialising in hallucination detection.
Your task is to identify specific phrases in a clinical SOAP note that could be
plausibly replaced with an incorrect alternative, simulating the kind of factual
error an LLM might hallucinate.

Return ONLY valid JSON — no markdown fences, no commentary, no preamble."""

_USER_TEMPLATE = """\
Clinical note:
\"\"\"
{note}
\"\"\"

Suggest exactly {n} hallucination substitutions for the note above.
Each substitution must:
  1. Replace a real clinical fact (medication, dosage, diagnosis, vital sign, or
     lab value) with a plausible-but-wrong alternative.
  2. Use an "original" that is an exact substring of the note (copy-paste it).
  3. Be subtle — a clinician might not notice immediately.
  4. Not alter section headers or punctuation structure.

Return this JSON structure and nothing else:
{{
  "injections": [
    {{
      "original":    "<exact text from the note to replace>",
      "replacement": "<plausible but incorrect alternative>",
      "category":    "<wrong_medication | wrong_dosage | wrong_diagnosis | wrong_vital | wrong_finding>"
    }}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# 2. JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from an LLM response.

    Strips markdown code fences (```json … ```) if present, then attempts
    json.loads.  Raises ValueError on failure.
    """
    # Strip leading/trailing whitespace and optional markdown fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text).strip()

    # Some models wrap the object in extra prose — extract the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Apply LLM suggestions → corrupted note + char-level injection records
# ─────────────────────────────────────────────────────────────────────────────

def _apply_candidates(
    note: str,
    candidates: List[Dict[str, str]],
    max_injections: int,
) -> Tuple[str, List[Dict]]:
    """
    Apply LLM-suggested substitutions to the note one by one, tracking exact
    character positions.  We do NOT trust the LLM to rewrite the note — we
    apply the substitutions ourselves for reproducibility.

    A candidate is skipped if:
      - its "original" field is absent, empty, or equal to "replacement"
      - the "original" text cannot be found (case-insensitive search) in the
        current working text

    Returns
    -------
    corrupted_note : modified note string
    injections     : list of dicts matching the schema expected by
                     halluc_token_indices() in run_experiments.py:
                     {char_start, char_end, original, replacement, category}
    """
    injections: List[Dict] = []
    text = note

    for cand in candidates:
        if len(injections) >= max_injections:
            break

        original    = cand.get("original",    "").strip()
        replacement = cand.get("replacement", "").strip()
        category    = cand.get("category",    "unknown")

        if not original or not replacement or original == replacement:
            continue

        # Exact match first, then case-insensitive fallback
        idx = text.find(original)
        if idx == -1:
            lo = text.lower()
            ol = original.lower()
            idx = lo.find(ol)
            if idx == -1:
                warnings.warn(
                    f"[halluc_llm] LLM suggested original '{original[:40]}' "
                    f"not found in note — skipping"
                )
                continue
            # Preserve original casing from note
            original = text[idx: idx + len(original)]

        cs = idx
        ce_original = cs + len(original)
        injections.append({
            "char_start":  cs,
            "char_end":    cs + len(replacement),
            "original":    original,
            "replacement": replacement,
            "category":    category,
        })
        text = text[:cs] + replacement + text[ce_original:]

    return text, injections


# ─────────────────────────────────────────────────────────────────────────────
# 4. HuggingFace backend
# ─────────────────────────────────────────────────────────────────────────────

_HF_DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class HFHallucinator:
    """
    Hallucination suggester backed by the HuggingFace Inference API.

    Requires:  pip install huggingface_hub
    Token:     set HF_TOKEN env var or pass token= kwarg.
    """

    def __init__(
        self,
        model: str = _HF_DEFAULT_MODEL,
        token: Optional[str] = None,
        max_tokens: int = 512,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for backend='hf'.  "
                "Install with:  pip install huggingface_hub"
            ) from exc

        from huggingface_hub import InferenceClient

        self.model      = model
        self.max_tokens = max_tokens
        self._client    = InferenceClient(token=token or os.environ.get("HF_TOKEN"))

    def suggest(
        self,
        note: str,
        n: int,
        temperature: float = 0.3,
    ) -> List[Dict[str, str]]:
        """
        Call the HF Inference API and return the raw list of injection candidates.
        Raises ValueError if the response cannot be parsed as valid JSON.
        """
        user_msg = _USER_TEMPLATE.format(note=note, n=n)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=temperature,
        )
        raw = response.choices[0].message.content
        data = _parse_json(raw)
        return data.get("injections", [])


# ─────────────────────────────────────────────────────────────────────────────
# 5. AWS Bedrock backend
# ─────────────────────────────────────────────────────────────────────────────

_BEDROCK_DEFAULT_MODEL  = "anthropic.claude-3-haiku-20240307-v1:0"
_BEDROCK_DEFAULT_REGION = "us-east-1"


class BedrockHallucinator:
    """
    Hallucination suggester backed by the AWS Bedrock Converse API.

    The Converse API is model-agnostic — it works with Claude, Llama, Mistral,
    Amazon Nova, and any other Bedrock-hosted chat model.

    Requires:  pip install boto3
    Credentials: standard AWS credential chain (env vars, IAM role, ~/.aws).
    """

    def __init__(
        self,
        model: str = _BEDROCK_DEFAULT_MODEL,
        region: Optional[str] = None,
        max_tokens: int = 512,
    ) -> None:
        try:
            import boto3  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for backend='bedrock'.  "
                "Install with:  pip install boto3"
            ) from exc

        import boto3

        resolved_region = region or os.environ.get("AWS_DEFAULT_REGION", _BEDROCK_DEFAULT_REGION)
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = boto3.client("bedrock-runtime", region_name=resolved_region)

    def suggest(
        self,
        note: str,
        n: int,
        temperature: float = 0.3,
    ) -> List[Dict[str, str]]:
        """
        Call the Bedrock Converse API and return the raw list of injection
        candidates.  Raises ValueError if the response cannot be parsed.
        """
        user_msg = _USER_TEMPLATE.format(note=note, n=n)

        response = self._client.converse(
            modelId=self.model,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[
                {"role": "user", "content": [{"text": user_msg}]}
            ],
            inferenceConfig={
                "maxTokens":   self.max_tokens,
                "temperature": temperature,
            },
        )
        raw  = response["output"]["message"]["content"][0]["text"]
        data = _parse_json(raw)
        return data.get("injections", [])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Public API
# ─────────────────────────────────────────────────────────────────────────────

class HallucinationGenerationError(RuntimeError):
    """Raised when all retry attempts fail to produce valid injection candidates."""


def inject_hallucinations_llm(
    note: str,
    max_injections: int = 3,
    backend: str = "hf",
    max_retries: int = 3,
    base_temperature: float = 0.3,
    **kwargs,
) -> Tuple[str, List[Dict]]:
    """
    Generate and inject hallucinations into a clinical note using an LLM.

    The LLM suggests (original, replacement, category) pairs; we apply them
    ourselves to guarantee exact character-span tracking.  Retries with
    gradually increasing temperature when JSON parsing fails.

    Parameters
    ----------
    note             : original clinical note text.
    max_injections   : maximum number of substitutions to apply (default 3).
    backend          : "hf" (HuggingFace) or "bedrock" (AWS Bedrock).
    max_retries      : number of retry attempts on JSON parse failure (default 3).
    base_temperature : starting temperature; increases by 0.1 per retry.
    **kwargs         : forwarded to the backend constructor:
        hf_model        str  (default "Qwen/Qwen2.5-72B-Instruct")
        hf_token        str  (overrides HF_TOKEN env var)
        bedrock_model   str  (default "anthropic.claude-3-haiku-20240307-v1:0")
        bedrock_region  str  (overrides AWS_DEFAULT_REGION env var)

    Returns
    -------
    corrupted_note : note text with injected hallucinations.
    injections     : list of {char_start, char_end, original, replacement, category}
                     records — same schema as the regex-based inject_hallucinations().

    Raises
    ------
    HallucinationGenerationError
        If all retry attempts fail to return parseable JSON with at least one
        valid candidate.
    """
    # ── Build backend ────────────────────────────────────────────────────────
    if backend == "hf":
        hallucinator = HFHallucinator(
            model=kwargs.get("hf_model", _HF_DEFAULT_MODEL),
            token=kwargs.get("hf_token"),
        )
    elif backend == "bedrock":
        hallucinator = BedrockHallucinator(
            model=kwargs.get("bedrock_model", _BEDROCK_DEFAULT_MODEL),
            region=kwargs.get("bedrock_region"),
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'.  Choose 'hf' or 'bedrock'.")

    # ── Retry loop ───────────────────────────────────────────────────────────
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        temperature = base_temperature + attempt * 0.1

        try:
            candidates = hallucinator.suggest(note, n=max_injections, temperature=temperature)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            warnings.warn(
                f"[halluc_llm] attempt {attempt + 1}/{max_retries} — "
                f"JSON parse failed: {exc}"
            )
            continue
        except Exception as exc:
            # Network / auth / quota errors — re-raise immediately
            raise HallucinationGenerationError(
                f"Backend '{backend}' call failed: {exc}"
            ) from exc

        if not candidates:
            warnings.warn(
                f"[halluc_llm] attempt {attempt + 1}/{max_retries} — "
                f"LLM returned empty injections list"
            )
            last_error = ValueError("Empty injections list")
            continue

        corrupted_note, injections = _apply_candidates(note, candidates, max_injections)

        if injections:
            print(f"  [halluc_llm] backend={backend}  model={hallucinator.model}"
                  f"  attempt={attempt + 1}  applied={len(injections)}/{len(candidates)} candidates")
            return corrupted_note, injections

        # All candidates were invalid (not found in note)
        last_error = ValueError("No valid candidates found in note text")
        warnings.warn(f"[halluc_llm] attempt {attempt + 1}/{max_retries} — {last_error}")

    raise HallucinationGenerationError(
        f"All {max_retries} attempts failed for backend='{backend}'.  "
        f"Last error: {last_error}"
    )

_HALLUC_RULES: List[Tuple[str, str, str]] = [
    # ── Medications ───────────────────────────────────────────────────────────
    (r"\balbuterol\b",           "salmeterol",            "wrong_medication"),
    (r"\bSingulair\b",           "Symbicort",             "wrong_medication"),
    (r"\bmontelukast\b",         "fluticasone",           "wrong_medication"),
    (r"\bmetformin\b",           "lisinopril",            "wrong_medication"),
    (r"\blisinopril\b",          "metformin",             "wrong_medication"),
    (r"\batorvastatin\b",        "rosuvastatin",          "wrong_medication"),
    (r"\baspirin\b",             "warfarin",              "wrong_medication"),
    (r"\bamoxicillin\b",         "azithromycin",          "wrong_medication"),
    (r"\bibuprofen\b",           "naproxen",              "wrong_medication"),
    (r"\bomeprazole\b",          "pantoprazole",          "wrong_medication"),
    (r"\bsertraline\b",          "fluoxetine",            "wrong_medication"),
    # ── Dosages ───────────────────────────────────────────────────────────────
    (r"\b10\s*mg\b",             "20 mg",                 "wrong_dosage"),
    (r"\b5\s*mg\b",              "10 mg",                 "wrong_dosage"),
    (r"\b500\s*mg\b",            "250 mg",                "wrong_dosage"),
    (r"\b20\s*mg\b",             "40 mg",                 "wrong_dosage"),
    (r"\b25\s*mg\b",             "50 mg",                 "wrong_dosage"),
    (r"\b50\s*mg\b",             "100 mg",                "wrong_dosage"),
    # ── Diagnoses ────────────────────────────────────────────────────────────
    (r"\basthma\b",              "COPD",                  "wrong_diagnosis"),
    (r"\bhypertension\b",        "hypotension",           "wrong_diagnosis"),
    (r"\bdiabetes\b",            "hypothyroidism",        "wrong_diagnosis"),
    (r"\ballergic\s+rhinitis\b", "chronic sinusitis",     "wrong_diagnosis"),
    (r"\bangina\b",              "myocardial infarction", "wrong_diagnosis"),
    (r"\beczema\b",              "psoriasis",             "wrong_diagnosis"),
    # ── Vitals / findings ────────────────────────────────────────────────────
    (r"\b120\s*/\s*80\b",        "180/110",               "wrong_vital"),
    (r"\b98\.6\b",               "101.4",                 "wrong_vital"),
    (r"\bnormal\b",              "elevated",              "wrong_finding"),
    (r"\bnegative\b",            "positive",              "wrong_finding"),
]

_FALLBACK_HALLUC = (
    " Additionally, patient reports recent chest tightness at rest; "
    "troponin I elevated at 0.8 ng/mL on last draw."
)


def inject_hallucinations(
    note: str,
    max_injections: int = 3,
    seed: int = 42,
) -> Tuple[str, List[Dict]]:
    """
    Apply a shuffled subset of _HALLUC_RULES to the note (up to max_injections).

    Rules are shuffled with `seed` so different dataset samples exercise
    different hallucination categories.  Each rule fires at most once.
    Falls back to appending a fabricated finding if no rule matches.

    Returns
    -------
    corrupted_note : note text with injected hallucinations.
    injections     : list of {char_start, char_end, original, replacement, category}.
    """
    rng   = random.Random(seed)
    rules = list(_HALLUC_RULES)
    rng.shuffle(rules)

    injections: List[Dict] = []
    text = note

    for pattern, replacement, category in rules:
        if len(injections) >= max_injections:
            break
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            cs = m.start()
            ce = m.end()
            injections.append({
                "char_start":  cs,
                "char_end":    cs + len(replacement),
                "original":    m.group(0),
                "replacement": replacement,
                "category":    category,
            })
            text = text[:cs] + replacement + text[ce:]

    if not injections:
        cs   = len(text)
        text += _FALLBACK_HALLUC
        injections.append({
            "char_start":  cs,
            "char_end":    len(text),
            "original":    "",
            "replacement": _FALLBACK_HALLUC.strip(),
            "category":    "inserted_hallucination",
        })

    return text, injections


def halluc_token_indices(
    tokenizer,
    hallucinated_note: str,
    injections: List[Dict],
) -> List[int]:
    """
    Map char-level injection spans → token indices in the tokenized note.

    Parameters
    ----------
    tokenizer : any HuggingFace tokenizer (PreTrainedTokenizer / Fast variant).
                Pass ``model.tokenizer`` when calling from run_experiments.py.

    Tries offset_mapping (fast tokenizers, exact), falls back to cumulative
    decode.  A token is flagged if its char span overlaps any injection span.
    """
    halluc_idx: List[int] = []

    # ── Primary path: offset_mapping (fast tokenizers) ───────────────────────
    try:
        enc     = tokenizer(hallucinated_note, return_offsets_mapping=True,
                            add_special_tokens=False)
        offsets = enc["offset_mapping"]
        for i, (cs, ce) in enumerate(offsets):
            if ce == cs:
                continue
            for inj in injections:
                if cs < inj["char_end"] and ce > inj["char_start"]:
                    halluc_idx.append(i)
                    break
        return sorted(set(halluc_idx))
    except Exception:
        pass

    # ── Fallback: cumulative decode ───────────────────────────────────────────
    ids    = tokenizer.encode(hallucinated_note, add_special_tokens=False)
    cursor = 0
    for i, tid in enumerate(ids):
        piece = tokenizer.decode([tid])
        end   = cursor + len(piece)
        for inj in injections:
            if cursor < inj["char_end"] and end > inj["char_start"]:
                halluc_idx.append(i)
                break
        cursor = end

    return sorted(set(halluc_idx))
