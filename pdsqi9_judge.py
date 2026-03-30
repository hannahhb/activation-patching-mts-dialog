"""
Experiment 2 — PDSQI-9 LLM-as-a-Judge with 5-shot prompting.

Supports three inference backends:
  - "openai"  : OpenAI-compatible API (default; requires OPENAI_API_KEY)
  - "hf"      : HuggingFace Inference API (requires HF_API_KEY)
  - "bedrock" : AWS Bedrock (requires AWS credentials in env/profile)

Scores each generated note (and optionally each SOAP section) on all 9
PDSQI-9 attributes.  Results are cached to disk so API calls are not
repeated on re-runs.

Attributes (1–5 unless noted):
  cited, accurate, thorough, useful, organized, comprehensible, succinct,
  synthesized, stigmatizing (0/1)
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    PDSQI9_ATTRIBUTES, JUDGE_MODEL, JUDGE_TEMPERATURE,
    CACHE_DIR,
)


# ── Rubric definitions (condensed for prompt efficiency) ─────────────────────

_RUBRICS: Dict[str, str] = {
    "cited": (
        "Are citations (<Note ID:#>) present and correctly paired with each assertion?\n"
        "1=No citations or multiple wrong; 2=One wrong or all grouped; "
        "3=All correct but some missing; 4=All correct, some relevance prioritisation; "
        "5=Every assertion correctly cited and relevance-prioritised."
    ),
    "accurate": (
        "Is the summary free of fabrications and falsifications?\n"
        "1=Multiple major fabrications/falsifications; 2=One major error; "
        "3=At least one assertion misaligned (wrong context/specificity); "
        "4=At least one assertion misaligned but still factual; "
        "5=All assertions traceable to the source."
    ),
    "thorough": (
        "Does the summary cover all clinically important issues?\n"
        "1=>1 pertinent omission; 2=1 pertinent + multiple potentially pertinent; "
        "3=1 pertinent omission; 4=Some potentially pertinent omissions; "
        "5=No omissions."
    ),
    "useful": (
        "Is all information relevant and appropriately detailed for the target provider?\n"
        "1=No relevant assertions; 2=Some relevant; 3=Relevant but wrong detail level; "
        "4=No non-pertinent content but some possibly irrelevant; "
        "5=All content relevant and appropriately detailed."
    ),
    "organized": (
        "Is the summary logically structured (temporal/systems-based ordering and grouping)?\n"
        "1=Completely disorganised; 2=Some out-of-order or incoherent grouping; "
        "3=No improvement from source order; 4=Good order OR good grouping (not both); "
        "5=Logical order AND grouping throughout."
    ),
    "comprehensible": (
        "Is the language clear and appropriate for the target reader?\n"
        "1=Overly complex/inconsistent throughout; 2=Any overly complex/unfamiliar term; "
        "3=Unchanged complex terms when simplification was possible; "
        "4=Some improvement in structure/terminology; "
        "5=Plain, well-structured language throughout."
    ),
    "succinct": (
        "Is the summary concise and without redundancy?\n"
        "1=Wordy throughout with redundancy; 2=>1 contextually redundant assertion; "
        "3=At least one redundant assertion; "
        "4=No syntactic redundancy but some could be shorter; "
        "5=All assertions expressed with minimum necessary words."
    ),
    "synthesized": (
        "Does the summary reflect medical reasoning and synthesis beyond simple extraction? "
        "(Score NA→1 if no synthesis was needed.)\n"
        "1=Incorrect reasoning or connections; 2=Inappropriate abstraction or groupings; "
        "3=Assertions stated independently with missed synthesis opportunities; "
        "4=Themes grouped but limited integrated reasoning; "
        "5=Fully integrated clinical synopsis with prioritised reasoning."
    ),
    "stigmatizing": (
        "Does the summary contain stigmatising language as defined by OCR/NIDA guidelines?\n"
        "0=No stigmatising language; 1=Definite stigmatising language present."
    ),
}

# ── 5-shot examples (one set per attribute) ───────────────────────────────────
# Each example: (dialogue_snippet, note_snippet, score, justification)

_SHOTS: Dict[str, List[tuple]] = {
    "accurate": [
        (
            "Doctor: You had a hemoglobin A1c of 7.2 last month.",
            "The patient had an HbA1c of 8.5.",
            2,
            "The HbA1c value was changed from 7.2 to 8.5 — a clear falsification.",
        ),
        (
            "Doctor: You had a hemoglobin A1c of 7.2 last month.",
            "The patient's recent HbA1c was 7.2, indicating fair glycaemic control.",
            5,
            "Value correctly extracted and contextualised from the source.",
        ),
        (
            "Doctor: You denied any chest pain.",
            "The patient reports chest pain.",
            1,
            "Negation reversed — multiple facts falsified throughout the note.",
        ),
        (
            "Doctor: BP today 130/80.",
            "Vital signs: BP 130/82.",
            4,
            "Minor inaccuracy in diastolic value; overall factually grounded.",
        ),
        (
            "Doctor: Start metformin 500mg twice daily.",
            "Plan: metformin 500mg daily.",
            3,
            "Dosing frequency misaligned — stated once daily instead of twice daily.",
        ),
    ],
    "synthesized": [
        (
            "Doctor: Your A1c is high and your weight has increased.",
            "A1c elevated. Weight increased.",
            3,
            "Facts stated independently; missed opportunity to synthesise as "
            "worsening metabolic syndrome pattern.",
        ),
        (
            "Doctor: A1c 9.1, weight up 8kg, BP 145/92, patient reports fatigue.",
            "Patient presents with poorly controlled diabetes (A1c 9.1) "
            "with features of metabolic syndrome including hypertension and obesity, "
            "contributing to fatigue. Recommend intensification of diabetes management.",
            5,
            "All facts integrated into a coherent clinical assessment with plan rationale.",
        ),
        (
            "Doctor: Your A1c is 7.0 and your BP is controlled.",
            "A1c 7.0. BP controlled.",
            3,
            "No synthesis — could have noted continued good control and reinforced adherence.",
        ),
        (
            "Doctor: You have knee pain and I see your BMI is 35.",
            "The patient has knee pain. The patient has a BMI of 35. "
            "The patient is obese. The patient should exercise.",
            2,
            "Redundant restatement of BMI; missed causal link between obesity and joint load.",
        ),
        (
            "Doctor: I want to add lisinopril given your diabetes and early proteinuria.",
            "Plan: add lisinopril for blood pressure management.",
            3,
            "Partially synthesised — missed the nephroprotective rationale specific to diabetic nephropathy.",
        ),
    ],
    "thorough": [
        (
            "Doctor: Any chest pain? Patient: No. Doctor: Any shortness of breath? Patient: A little.",
            "ROS: denies chest pain.",
            3,
            "Shortness of breath — a pertinent positive — was omitted.",
        ),
        (
            "Doctor: Any chest pain? Patient: No. Doctor: Any shortness of breath? Patient: A little.",
            "ROS: denies chest pain; reports mild shortness of breath.",
            5,
            "Both relevant ROS findings documented.",
        ),
        (
            "Doctor discussed allergies, social history, and three active problems.",
            "Active problem 1 documented.",
            1,
            "Two active problems, allergies, and social history all omitted.",
        ),
        (
            "Doctor reviewed 5 medications in detail.",
            "Medications listed: 4 of 5 (omitting aspirin 81mg).",
            4,
            "One potentially pertinent medication omitted.",
        ),
        (
            "Short encounter — only chief complaint discussed.",
            "Chief complaint documented accurately.",
            5,
            "All clinically relevant content for this encounter captured.",
        ),
    ],
    "useful": [
        (
            "Patient is seeing a cardiologist post-MI.",
            "Note includes 2 paragraphs on unrelated dermatology history.",
            2,
            "Non-pertinent content dominates for a cardiology context.",
        ),
        (
            "Patient is seeing a cardiologist post-MI.",
            "Focused cardiac summary with EF, medications, and follow-up plan.",
            5,
            "Content is entirely relevant and appropriately detailed for cardiology.",
        ),
        (
            "Primary care visit for diabetes management.",
            "Includes full surgical history from 20 years ago with irrelevant detail.",
            3,
            "Relevant content present but diluted by excessive historical detail.",
        ),
        (
            "Brief nurse triage note context.",
            "Note contains extensive lab interpretation appropriate for physician.",
            3,
            "Level of detail inappropriate for triage audience.",
        ),
        (
            "Primary care visit.",
            "Concise summary of active issues, vitals, and plan.",
            5,
            "Appropriate level of detail for primary care context.",
        ),
    ],
    "organized": [
        (
            "Standard SOAP encounter.",
            "Assessment listed before History of Present Illness.",
            2,
            "Assessment placed before HPI — violates clinical note convention.",
        ),
        (
            "Standard SOAP encounter.",
            "CC → HPI → ROS → PE → Assessment → Plan in logical order.",
            5,
            "Correct SOAP structure with logical temporal flow.",
        ),
        (
            "Multi-problem encounter.",
            "Problems scattered throughout without grouping.",
            1,
            "Completely disorganised — no temporal or problem-based structure.",
        ),
        (
            "Multi-problem encounter.",
            "Problems grouped by system but not temporally ordered.",
            4,
            "Good grouping but lacks temporal ordering within sections.",
        ),
        (
            "Single-problem encounter.",
            "Note structure unchanged from dialogue order.",
            3,
            "No reorganisation applied to improve clinical readability.",
        ),
    ],
    "comprehensible": [
        (
            "Patient is a lay person asking about their diagnosis.",
            "Patient has idiopathic thrombocytopenic purpura with thrombocytopenia.",
            2,
            "Technical jargon not appropriate for the target reader without explanation.",
        ),
        (
            "Note for primary care physician.",
            "Patient has ITP with platelet count 45K, on watch-and-wait.",
            5,
            "Appropriate clinical shorthand for a physician audience.",
        ),
        (
            "Note for primary care physician.",
            "The patient possibly maybe could potentially have issues.",
            1,
            "Excessive hedging creates ambiguity throughout.",
        ),
        (
            "Note for primary care physician.",
            "Mixed use of metric and imperial units inconsistently.",
            2,
            "Inconsistent unit usage reduces clarity.",
        ),
        (
            "Note for primary care physician.",
            "Clear, structured note with standard clinical terminology.",
            5,
            "Language is precise and appropriate for the physician reader.",
        ),
    ],
    "succinct": [
        (
            "Brief medication reconciliation.",
            "Patient is taking aspirin. Aspirin 81mg is listed. The aspirin dose is 81mg daily.",
            1,
            "Aspirin mentioned three times with no added information.",
        ),
        (
            "Brief medication reconciliation.",
            "Aspirin 81mg daily.",
            5,
            "Single, precise statement with no redundancy.",
        ),
        (
            "Short encounter.",
            "Three-paragraph note restating the chief complaint five times.",
            2,
            "Significant semantic redundancy across paragraphs.",
        ),
        (
            "Moderate length encounter.",
            "Note is appropriately concise with minor restatement of plan in summary.",
            4,
            "Minor redundancy in plan restatement, otherwise succinct.",
        ),
        (
            "Complex multi-problem encounter.",
            "Comprehensive note covering all problems without unnecessary repetition.",
            5,
            "Detail is necessary; no redundancy present.",
        ),
    ],
    "cited": [
        (
            "Multi-note summary task.",
            "Patient has diabetes. <Note ID:1> Patient has hypertension.",
            3,
            "One assertion correctly cited; diabetes assertion missing citation.",
        ),
        (
            "Multi-note summary task.",
            "Patient has diabetes <Note ID:1> and hypertension <Note ID:2>.",
            4,
            "All assertions cited; some relevance prioritisation present.",
        ),
        (
            "Multi-note summary task.",
            "Patient has diabetes, hypertension, and CKD. <Note ID:1><Note ID:2><Note ID:3>",
            2,
            "Citations grouped at end rather than paired with each assertion.",
        ),
        (
            "Multi-note summary task.",
            "Summary with no citations anywhere.",
            1,
            "No citations provided.",
        ),
        (
            "Multi-note summary task.",
            "Patient has diabetes <Note ID:1>, hypertension <Note ID:2>, "
            "and CKD <Note ID:3>, prioritised by clinical urgency.",
            5,
            "Every assertion correctly cited and ordered by relevance.",
        ),
    ],
    "stigmatizing": [
        (
            "Patient has opioid use disorder.",
            "Patient is a drug addict seeking narcotics.",
            1,
            "'Drug addict' and 'seeking' are stigmatising per NIDA guidelines.",
        ),
        (
            "Patient has opioid use disorder.",
            "Patient has opioid use disorder, currently on MAT with buprenorphine.",
            0,
            "Person-first, non-stigmatising language throughout.",
        ),
        (
            "Patient uses alcohol heavily.",
            "Patient is an alcoholic.",
            1,
            "'Alcoholic' is stigmatising; preferred term is 'alcohol use disorder'.",
        ),
        (
            "Patient uses alcohol heavily.",
            "Patient reports heavy alcohol use; meets criteria for alcohol use disorder.",
            0,
            "Descriptive, non-judgemental framing.",
        ),
        (
            "Patient with psychiatric history.",
            "Psychiatric history documented with neutral clinical language.",
            0,
            "No stigmatising language present.",
        ),
    ],
}


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    attribute:    str,
    dialogue:     str,
    note:         str,
    section_name: Optional[str] = None,
    zero_shot:    bool = False,
) -> str:
    rubric = _RUBRICS[attribute]
    shots  = [] if zero_shot else _SHOTS.get(attribute, [])

    scope = f"the '{section_name}' section of " if section_name else ""
    scale = "an integer 0 or 1" if attribute == "stigmatizing" else "an integer from 1 to 5"

    lines = [
        f"You are a clinical documentation quality expert. "
        f"Score {scope}the CLINICAL NOTE below on the '{attribute.upper()}' dimension.",
        "",
        f"RUBRIC:\n{rubric}",
        "",
    ]

    if shots:
        lines.append("--- EXAMPLES ---")
        for i, (ex_dial, ex_note, ex_score, ex_just) in enumerate(shots, 1):
            lines += [
                f"Example {i}:",
                f"Dialogue excerpt: {ex_dial}",
                f"Note excerpt: {ex_note}",
                f"Score: {ex_score}",
                f"Justification: {ex_just}",
                "",
            ]

    lines += [
        "--- YOUR TASK ---",
        f"SOURCE DIALOGUE (truncated to 3000 chars):\n{dialogue[:3000]}",
        "",
        f"CLINICAL NOTE (truncated to 2000 chars):\n{note[:2000]}",
        "",
        f"Respond with ONLY {scale} on the first line, then a one-sentence justification.",
        "Score:",
    ]
    return "\n".join(lines)


# ── Disk cache ─────────────────────────────────────────────────────────────────

class _ScoreCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path) as f:
                self._store: Dict[str, dict] = json.load(f)
        else:
            self._store = {}

    def _key(self, attribute: str, dialogue: str, note: str, section: Optional[str]) -> str:
        raw = f"{attribute}|{section}|{dialogue[:500]}|{note[:500]}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, attribute, dialogue, note, section):
        return self._store.get(self._key(attribute, dialogue, note, section))

    def set(self, attribute, dialogue, note, section, value):
        self._store[self._key(attribute, dialogue, note, section)] = value
        with open(self.path, "w") as f:
            json.dump(self._store, f)


_cache = _ScoreCache(Path(CACHE_DIR) / "pdsqi9_scores.json")


# ── Backend implementations ────────────────────────────────────────────────────

class _OpenAIBackend:
    """OpenAI-compatible API (also works with Azure OpenAI via OPENAI_BASE_URL)."""

    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI()   # reads OPENAI_API_KEY from env
        self.model  = model

    def call(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=JUDGE_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


class _HFBackend:
    """
    HuggingFace Inference API (serverless or dedicated endpoint).

    Set HF_API_KEY (or HUGGINGFACE_API_KEY) in your environment.
    `model` should be a repo ID, e.g. "mistralai/Mixtral-8x7B-Instruct-v0.1".
    """

    def __init__(self, model: str):
        from huggingface_hub import InferenceClient
        api_key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
        self.client = InferenceClient(model=model, token=api_key)
        self.model  = model

    def call(self, prompt: str) -> str:
        resp = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=max(JUDGE_TEMPERATURE, 1e-3),  # HF API rejects 0
            max_tokens=64,
        )
        return resp.choices[0].message.content.strip()


class _BedrockBackend:
    """
    AWS Bedrock — Converse API (works with Claude, Llama, Mistral, Qwen, etc.).

    Credentials are read from env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
    or the default AWS profile.  Set AWS_DEFAULT_REGION or AWS_REGION if needed.

    `model` should be a Bedrock model ID, e.g.
      "anthropic.claude-3-5-sonnet-20241022-v2:0"
      "us.meta.llama3-3-70b-instruct-v1:0"
      "qwen.qwen3-235b-a22b-2507-v1:0"
    """

    def __init__(self, model: str, pool_size: int = 3):
        import boto3
        import itertools
        import threading
        from botocore.config import Config as BotoConfig

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        cfg = BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=180,
            max_pool_connections=25,
        )
        self._clients = [
            boto3.client("bedrock-runtime", region_name=region, config=cfg)
            for _ in range(pool_size)
        ]
        self._cycle = itertools.cycle(self._clients)
        self._lock  = threading.Lock()
        self.model  = model

    def _client(self):
        with self._lock:
            return next(self._cycle)

    @staticmethod
    def _extract_text(resp: dict) -> str:
        out = resp.get("output")
        if isinstance(out, dict) and "message" in out:
            for c in out["message"].get("content", []):
                if "text" in c:
                    return c["text"]
        if isinstance(out, list):
            for c in out:
                if isinstance(c, dict) and "text" in c:
                    return c["text"]
        return json.dumps(resp)[:512]

    def call(self, prompt: str) -> str:
        params = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": 64,
                "temperature": float(max(JUDGE_TEMPERATURE, 1e-3)),
                "topP": 0.9,
            },
        }
        import time
        for attempt in range(3):
            try:
                resp = self._client().converse(**params)
                return self._extract_text(resp).strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def _make_backend(backend: str, model: str):
    backend = backend.lower()
    if backend == "openai":
        return _OpenAIBackend(model)
    if backend in ("hf", "huggingface"):
        return _HFBackend(model)
    if backend in ("bedrock", "aws", "aws_bedrock"):
        return _BedrockBackend(model)
    raise ValueError(f"Unknown judge backend {backend!r}. Choose: openai, hf, bedrock")


# ── Judge class ────────────────────────────────────────────────────────────────

class PDSQI9Judge:
    def __init__(self, model: str = JUDGE_MODEL, backend: str = "openai", zero_shot: bool = False):
        """
        Parameters
        ----------
        model:
            Model identifier — meaning depends on backend:
            - openai  : OpenAI model name, e.g. "gpt-4o"
            - hf      : HuggingFace repo ID, e.g. "mistralai/Mixtral-8x7B-Instruct-v0.1"
            - bedrock : Bedrock model ID, e.g. "anthropic.claude-3-5-sonnet-20241022-v2:0"
        backend:
            One of "openai", "hf", "bedrock".
        zero_shot:
            If True, omit the few-shot examples from the prompt.
        """
        self._backend  = _make_backend(backend, model)
        self.model     = model
        self.backend   = backend
        self.zero_shot = zero_shot

    def score_attribute(
        self,
        attribute:    str,
        dialogue:     str,
        note:         str,
        section_name: Optional[str] = None,
        retries:      int = 2,
    ) -> Optional[int]:
        """
        Score a single PDSQI-9 attribute. Returns an integer or None on failure.
        Results are cached — repeated calls with the same inputs are free.
        """
        cached = _cache.get(attribute, dialogue, note, section_name)
        if cached is not None:
            return cached["score"]

        prompt = _build_prompt(attribute, dialogue, note, section_name, zero_shot=self.zero_shot)

        for attempt in range(retries + 1):
            try:
                text  = self._backend.call(prompt)
                score = int(text.splitlines()[0].strip())
                _cache.set(attribute, dialogue, note, section_name, {"score": score})
                return score
            except Exception as e:
                print(f"    [Judge/{self.backend}] {attribute} attempt {attempt+1} failed: {e}")

        return None

    def score_note(
        self,
        dialogue:  str,
        note:      str,
        sections:  Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict]:
        """
        Score all 9 PDSQI-9 attributes for the full note and (optionally)
        each SOAP section independently.

        Returns:
            {
              "full": {"cited": 3, "accurate": 4, ...},
              "sections": {
                  "assessment": {"cited": ..., ...},
                  "plan": {...},
                  ...
              }
            }
        """
        result: Dict[str, Dict] = {"full": {}, "sections": {}}

        for attr in PDSQI9_ATTRIBUTES:
            result["full"][attr] = self.score_attribute(attr, dialogue, note)

        if sections:
            for sec_name, sec_text in sections.items():
                if not sec_text.strip():
                    continue
                result["sections"][sec_name] = {}
                for attr in PDSQI9_ATTRIBUTES:
                    result["sections"][sec_name][attr] = self.score_attribute(
                        attr, dialogue, sec_text, section_name=sec_name
                    )

        return result
