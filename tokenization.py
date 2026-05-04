"""
tokenization.py
===============
Tokenisation helpers and model-based note generation.

All functions wrap HookedTransformer I/O and use the same generation prompt
so that teacher-forced activations are conditioned identically to autoregressive
generation (a fundamental property of causal attention).
"""

import textwrap
from typing import List, Tuple

import torch
from transformer_lens import HookedTransformer

from config import Config


# ─────────────────────────────────────────────
# Generation prompt
# Shared by tokenize_as_generated and generate_note so the prompt prefix
# always matches between calibration and scoring.
# ─────────────────────────────────────────────

_GENERATION_PROMPT = (
    "You are a clinical documentation assistant."
    "Given the following patient-clinician conversation, write a summary of the "
    "conversation with six sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
    "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN."
    "### Conversation\n{transcript}\n\n"
    "### Note: \n"
)


# ─────────────────────────────────────────────
# Tokenisation helpers
# ─────────────────────────────────────────────

def tokenize_pair(
    model: HookedTransformer,
    transcript: str,
    note: str,
) -> Tuple[torch.Tensor, int, List[str]]:
    """
    Concatenate transcript and note into one token sequence.

    Returns
    -------
    tokens          : LongTensor (1, transcript_len + note_len)
    transcript_len  : number of tokens belonging to the transcript prefix
    note_str_tokens : decoded string for each note token (for plot labels)
    """
    transcript_tok = model.to_tokens(transcript, prepend_bos=True)    # (1, T)
    note_tok       = model.to_tokens(note,       prepend_bos=False)   # (1, N)
    combined       = torch.cat([transcript_tok, note_tok], dim=1)      # (1, T+N)
    transcript_len = transcript_tok.shape[1]
    note_str       = [model.tokenizer.decode([t.item()]) for t in note_tok[0]]
    return combined, transcript_len, note_str


def tokenize_as_generated(
    model: HookedTransformer,
    transcript: str,
    note: str,
) -> Tuple[torch.Tensor, int, List[str]]:
    """
    Tokenize a note in the same prompt context used for actual generation
    (Experiment 2b), so that note-token activations are conditioned identically
    to how the model would condition them during autoregressive generation.

    For a causal (decoder-only) model, teacher-forcing with the full sequence
    [prompt | note] produces exactly the same hidden states at each note position
    as step-by-step generation would — this is a fundamental property of causal
    attention.  The only thing that matters for correctness is that the *prefix*
    matches what was used during generation, which this function ensures.

    Returns
    -------
    tokens      : LongTensor (1, prompt_len + note_len)
    prompt_len  : number of tokens in the generation prompt prefix
                  (drop-in replacement for transcript_len in compute_ecs_pks)
    note_strs   : decoded string for each note token (for plot labels)
    """
    prompt     = _GENERATION_PROMPT.format(transcript=transcript.strip())
    prompt_tok = model.to_tokens(prompt, prepend_bos=True)    # (1, P)
    note_tok   = model.to_tokens(note,   prepend_bos=False)   # (1, N)
    combined   = torch.cat([prompt_tok, note_tok], dim=1)     # (1, P+N)
    prompt_len = prompt_tok.shape[1]
    note_strs  = [model.tokenizer.decode([t.item()]) for t in note_tok[0]]
    return combined, prompt_len, note_strs


# ─────────────────────────────────────────────
# Note generation (Experiment 2b)
# ─────────────────────────────────────────────

def generate_note(
    model: HookedTransformer,
    transcript: str,
    cfg: Config,
) -> str:
    """
    Autoregressively generate a SOAP note from the transcript.

    cfg.gen_temperature == 0.0  → greedy (reproducible)
    cfg.gen_temperature  > 0.0  → sampling (~0.7 for more variety)
    """
    prompt    = _GENERATION_PROMPT.format(transcript=transcript.strip())
    input_ids = model.to_tokens(prompt, prepend_bos=True).to(cfg.device)

    print(f"  Generating note  (max_new_tokens={cfg.max_new_tokens}, "
          f"temperature={cfg.gen_temperature}) …")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=1.0 if cfg.gen_temperature == 0.0 else cfg.gen_temperature,
            do_sample=(cfg.gen_temperature != 0.0),
            verbose=False,
        )

    prompt_len    = input_ids.shape[1]
    new_token_ids = output_ids[0, prompt_len:]
    generated     = model.tokenizer.decode(new_token_ids.tolist(), skip_special_tokens=True)

    print(f"\n  ── Generated note preview (first 300 chars) ──")
    print(textwrap.indent(generated[:300].strip(), "    "))
    print()

    return generated
