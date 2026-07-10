"""
llm_client.py
=============
Single tracked entry point for every Bedrock Converse call in the pipeline.

Before this, each of the places that call an LLM -- note generation
(luq_sentence.py), atomic-fact decomposition and fact-coverage checking
(factmatch_sentence.py), LLM-judge span labeling (llm_hallucination_label.py),
and gpt-oss NLI entailment scoring (atomic_luq.py) -- built its own boto3
client, its own retry loop, and (only for gpt-oss so far) its own ad hoc
usage counter. Centralizing means cost is tracked everywhere for free and
broken down by pipeline stage, instead of being reimplemented -- or silently
missing -- at each call site.

Usage:
    from llm_client import get_llm

    resp = get_llm().converse(
        stage="generation", model_id=BEDROCK_GEN_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inference_config={"maxTokens": 1024, "temperature": 0.7},
    )
    text = resp["output"]["message"]["content"][0]["text"]

    print(get_llm().tracker.summary())
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Per-1M-token on-demand Bedrock pricing (input, output), confirmed against
# the Bedrock pricing page. A model missing here still works -- usage is
# still recorded, cost just reports as $0 for it (with a one-time warning)
# rather than crashing a run over a pricing-table gap.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "us.meta.llama3-1-8b-instruct-v1:0":  (0.22, 0.22),
    "us.meta.llama3-3-70b-instruct-v1:0": (0.72, 0.72),
    "openai.gpt-oss-20b-1:0":             (0.07, 0.30),
}

BEDROCK_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")


@dataclass
class _Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def cost(self, model_id: str) -> float:
        price_in, price_out = MODEL_PRICING.get(model_id, (0.0, 0.0))
        return self.input_tokens / 1e6 * price_in + self.output_tokens / 1e6 * price_out


class CostTracker:
    """Thread-safe usage/cost accounting, broken down by (stage, model_id).
    One instance is shared across every call the pipeline makes, so a report
    at the end of a run can show cost per stage (decomp/generation/judge/nli)
    without each stage having to track this on its own."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage: Dict[Tuple[str, str], _Usage] = {}
        self._warned_models: set = set()

    def record(self, stage: str, model_id: str, input_tokens: int, output_tokens: int) -> None:
        if model_id not in MODEL_PRICING and model_id not in self._warned_models:
            self._warned_models.add(model_id)
            tqdm.write(f"[cost] WARNING: no pricing entry for model {model_id!r} -- "
                      f"its cost will report as $0. Add it to MODEL_PRICING in llm_client.py.")
        with self._lock:
            key = (stage, model_id)
            if key not in self._usage:
                self._usage[key] = _Usage()
            self._usage[key].add(input_tokens, output_tokens)

    def total_cost(self) -> float:
        with self._lock:
            return sum(u.cost(model_id) for (_, model_id), u in self._usage.items())

    def stage_cost(self, stage: str) -> float:
        with self._lock:
            return sum(u.cost(model_id) for (s, model_id), u in self._usage.items() if s == stage)

    def summary(self) -> str:
        """One line per (stage, model) plus a grand total."""
        with self._lock:
            items = sorted(self._usage.items())
        if not items:
            return "no LLM calls recorded"
        lines = []
        for (stage, model_id), u in items:
            lines.append(f"  {stage:<10} {model_id:<32} {u.calls:>5} calls  "
                         f"{u.input_tokens:>10,} in / {u.output_tokens:>9,} out  "
                         f"${u.cost(model_id):.4f}")
        total_calls = sum(u.calls for _, u in items)
        total_cost = sum(u.cost(model_id) for (_, model_id), u in items)
        lines.append(f"  {'TOTAL':<10} {'':<32} {total_calls:>5} calls  "
                     f"{'':>10}    {'':>9}     ${total_cost:.4f}")
        return "\n".join(lines)


# AWS error codes that mean "I'm overloaded, not that something is wrong
# with your request" -- these need real breathing room (exponential, several
# seconds+) before retrying, unlike a generic transient error where the
# short linear backoff is fine. Retrying a throttle signal quickly just
# re-triggers it.
THROTTLE_ERROR_CODES = {
    "ServiceUnavailableException", "ThrottlingException",
    "TooManyRequestsException", "ModelNotReadyException",
}


def _error_code(exc: Exception) -> str:
    return getattr(exc, "response", {}).get("Error", {}).get("Code", "")


class BedrockLLM:
    """Thin wrapper around bedrock-runtime Converse: lazy client, retry with
    backoff, and cost tracking. Callers own prompt construction and response
    parsing -- this only executes the call and records usage."""

    def __init__(self, tracker: CostTracker, max_retries: int = 5, retry_sleep: float = 2.0,
                connect_timeout: float = 20.0, read_timeout: float = 120.0,
                botocore_max_attempts: int = 5, botocore_retry_mode: str = "adaptive",
                throttle_retry_base: float = 5.0):
        self._client = None
        self.tracker = tracker
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.botocore_max_attempts = botocore_max_attempts
        self.botocore_retry_mode = botocore_retry_mode
        self.throttle_retry_base = throttle_retry_base

    def _get_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=BEDROCK_REGION,
                config=Config(
                    retries={"max_attempts": self.botocore_max_attempts,
                            "mode": self.botocore_retry_mode},
                    connect_timeout=self.connect_timeout,
                    read_timeout=self.read_timeout,
                ),
            )
            tqdm.write(f"[bedrock] Client initialised in {BEDROCK_REGION} "
                      f"(read_timeout={self.read_timeout}s, "
                      f"botocore_max_attempts={self.botocore_max_attempts})")
        return self._client

    def converse(self, *, stage: str, model_id: str, messages: List[dict],
                system: Optional[List[dict]] = None,
                inference_config: Optional[dict] = None,
                additional_model_request_fields: Optional[dict] = None) -> dict:
        """Returns the raw Converse response dict, or raises after
        max_retries -- callers decide their own fallback value (None, [],
        0, ...) with a try/except around this call.

        Records usage on every response that comes back -- including one
        whose answer text later turns out unparseable downstream -- since it
        still cost tokens.
        """
        client = self._get_client()
        kwargs = {"modelId": model_id, "messages": messages}
        if system:
            kwargs["system"] = system
        if inference_config:
            kwargs["inferenceConfig"] = inference_config
        if additional_model_request_fields:
            kwargs["additionalModelRequestFields"] = additional_model_request_fields

        for attempt in range(self.max_retries):
            try:
                resp = client.converse(**kwargs)
                usage = resp.get("usage", {})
                self.tracker.record(stage, model_id, usage.get("inputTokens", 0),
                                    usage.get("outputTokens", 0))
                return resp
            except Exception as exc:
                code = _error_code(exc)
                is_throttle = code in THROTTLE_ERROR_CODES
                if attempt < self.max_retries - 1:
                    if is_throttle:
                        # Real exponential backoff: 5s, 10s, 20s... -- a
                        # throttle/capacity signal needs several seconds to
                        # clear, not the ~2s linear step used for generic
                        # transient errors. Retrying it fast just adds to
                        # the load that triggered it.
                        sleep_s = self.throttle_retry_base * (2 ** attempt)
                    else:
                        sleep_s = self.retry_sleep * (attempt + 1)
                    tqdm.write(f"  [{stage}] error (attempt {attempt + 1}, "
                              f"{code or 'unknown'}): {exc} -- sleeping {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                else:
                    tqdm.write(f"  [{stage}] failed after {self.max_retries} attempts: {exc}")
                    raise


# Shared default instance -- mirrors the get_bedrock_client() singleton
# pattern already used across the codebase, so every module still uses one
# boto3 client under the hood, but now with tracking built in.
tracker = CostTracker()
_llm = BedrockLLM(tracker)

# gpt-oss (nli stage) gets its own client, tuned much tighter than the
# default: it's occasionally observed to hang (connection stays open, no
# data) rather than fail fast, and at reasoning_effort="low" a healthy call
# finishes in a few seconds -- so there's no reason to wait long before
# giving up.
#
# The default client's Config(retries={"max_attempts": 5, "mode": "adaptive"})
# is botocore's OWN internal retry layer, separate from (and multiplicative
# with) our outer `for attempt in range(max_retries)` loop in converse() --
# a single "attempt" there could silently retry up to 5x inside botocore
# before ever raising back to us. That's why shortening just read_timeout
# earlier didn't fully fix the visible stalls: worst case was still
# `our_max_retries x botocore_max_attempts x read_timeout`. Setting
# Two DIFFERENT failure modes were getting the same fix, which backfired:
#   - A genuine hang (connection open, no data) -- a short read_timeout is
#     the right tool, and no backoff is needed since it's rare.
#   - ServiceUnavailableException / throttling -- Bedrock's fast, explicit
#     "I'm overloaded" signal. This is NOT a hang; a short timeout does
#     nothing for it. What it needs is fewer concurrent requests and a real
#     backoff before retrying, so the endpoint gets a chance to recover.
# Disabling botocore's own retry entirely (max_attempts=0) removed the one
# mechanism ("adaptive" mode's client-side rate limiter) built specifically
# to detect this signal and throttle our own request rate in response --
# with it gone, 8 workers all retried the moment they saw a 503, near-
# instantly, which keeps the endpoint saturated instead of letting it drain.
# That's the "endless" storm: not a timeout problem, a thundering-herd
# problem, made worse (not better) by an aggressive timeout+retry policy.
#
# Fix: fewer concurrent workers (GPTOSS_MAX_WORKERS below), a modest read
# timeout that won't misfire on normal latency variance, real linear
# backoff between our own retries, AND botocore's adaptive retry mode given
# back a little room (max_attempts=1, i.e. one internal retry) so its rate
# limiter can actually do its job on throttling errors specifically.
GPTOSS_READ_TIMEOUT = 8.0
GPTOSS_CONNECT_TIMEOUT = 5.0
GPTOSS_MAX_RETRIES = 3
GPTOSS_RETRY_SLEEP = 2.0

_gptoss_llm = BedrockLLM(tracker, max_retries=GPTOSS_MAX_RETRIES, retry_sleep=GPTOSS_RETRY_SLEEP,
                         connect_timeout=GPTOSS_CONNECT_TIMEOUT, read_timeout=GPTOSS_READ_TIMEOUT,
                         botocore_max_attempts=1, botocore_retry_mode="adaptive")


def get_llm() -> BedrockLLM:
    return _llm


def get_gptoss_llm() -> BedrockLLM:
    return _gptoss_llm


# ── Shared gpt-oss yes/no entailment primitive ───────────────────────────────
# Used by both atomic_luq.py (fact-level uncertainty) and luq_sentence.py
# (sentence/claim-level uncertainty) as an alternative NLI backend to the
# local cross-encoder. Lives here -- not in either of those two files, which
# have a documented one-way circular-import constraint between them (atomic_luq
# imports luq_sentence at module level, so luq_sentence can never import
# atomic_luq back) -- so neither has to duplicate it or reach into the other.

GPTOSS_MODEL_ID          = "openai.gpt-oss-20b-1:0"
GPTOSS_ANSWER_MAX_TOKENS = 300  # headroom for the reasoning trace ahead of
                                # the final Yes/No -- observed 26-55 output
                                # tokens at reasoning_effort="low"; too tight
                                # a cap truncates mid-reasoning and leaves no
                                # final text block at all.
GPTOSS_PARSE_RETRIES     = 3    # retries when a response comes back with NO
                                # final text block (truncated mid-reasoning)
                           
GPTOSS_MAX_WORKERS       = 8    # concurrent Bedrock calls; each gptoss_yesno
                                # call is a blocking network round-trip
                         

def build_yesno_prompt(premise: str, hypothesis: str) -> str:
    return (
        "You are checking whether a clinical note supports a specific claim.\n\n"
        f'Note text:\n"{premise}"\n\n'
        f'Claim: "{hypothesis}"\n\n'
        'Does the note text support (entail) this claim? Answer with only '
        '"Yes" or "No".'
    )


def gptoss_yesno(premise: str, hypothesis: str, reasoning_effort: str = "low") -> float:
    """1.0 if gpt-oss-20b answers Yes (entailed), 0.0 if No/unparseable/empty
    premise. Reasoning tokens are billed as output but discarded -- only the
    final `text` content block (not `reasoningContent`) is the answer."""
    if not premise.strip():
        return 0.0

    prompt = build_yesno_prompt(premise, hypothesis)
    for attempt in range(GPTOSS_PARSE_RETRIES):
        try:
            resp = get_gptoss_llm().converse(
                stage="nli", model_id=GPTOSS_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inference_config={"maxTokens": GPTOSS_ANSWER_MAX_TOKENS, "temperature": 0.0},
                additional_model_request_fields={"reasoning_effort": reasoning_effort},
            )
        except Exception:
            return 0.0  # get_gptoss_llm().converse() already retried (short timeout) and logged

        text_blocks = [b["text"] for b in resp["output"]["message"]["content"] if "text" in b]
        if text_blocks:
            answer = text_blocks[-1].strip().lower()
            if answer.startswith("yes"):
                return 1.0
            if answer.startswith("no"):
                return 0.0
            tqdm.write(f"  [gptoss] unparseable answer: {answer!r} -- treating as No")
            return 0.0

        tqdm.write(f"  [gptoss] no text block (stopReason={resp.get('stopReason')}), "
                  f"retrying ({attempt + 1}/{GPTOSS_PARSE_RETRIES})")

    return 0.0
