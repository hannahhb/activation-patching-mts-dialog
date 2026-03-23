"""
Experiment 1 — Per-token DLA and activation patching on generated clinical notes.

Pipeline:
  1. Load ACI-Bench examples
  2. Generate clinical notes via Gemma 2 (greedy decoding)
  3. run_with_cache → per-token DLA (attn_out / mlp_out per layer)
  4. Zero-ablation patching sweep → per-layer importance per generated token
  5. Save token DataFrame (parquet) + patching arrays (npz) per encounter

Usage:
    python run_exp1.py [--n 10] [--results-dir results] [--device cuda]
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from config import (
    MODEL_NAME, DTYPE, SOAP_PROMPT_TEMPLATE,
    MAX_NEW_TOKENS, GENERATION_TEMP,
    N_PILOT, RESULTS_DIR, CACHE_DIR,
)
from data import load_aci_examples
from mechanistic import profile_example, run_activation_patching, cache_filter


# ── Note generation ────────────────────────────────────────────────────────────

def generate_note(
    model:     HookedTransformer,
    tokenizer,
    dialogue:  str,
) -> tuple[str, torch.Tensor, int]:
    """
    Generate a SOAP note for one encounter using greedy decoding.

    Returns:
        generated_text : the note text (without the prompt)
        full_tokens    : [1, total_seq] prompt + generated token IDs
        prompt_len     : number of tokens in the prompt
    """
    prompt_str = SOAP_PROMPT_TEMPLATE.format(dialogue=dialogue)

    # Use HF tokenizer for chat template if model is instruction-tuned
    try:
        messages = [{"role": "user", "content": prompt_str}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        pass  # fall back to raw prompt

    prompt_tokens = model.to_tokens(prompt_str, prepend_bos=False)
    prompt_len    = prompt_tokens.shape[1]

    with torch.inference_mode():
        output_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=GENERATION_TEMP if GENERATION_TEMP > 0 else 1.0,
            do_sample=GENERATION_TEMP > 0,
            freq_penalty=2.0,
            prepend_bos=False,
            stop_at_eos=True,
            verbose=False,
        )

    # Gemma 2 IT ends turns with <end_of_turn>; TL's stop_at_eos only checks
    # <eos> (id=1), so truncate manually at whichever stop token comes first.
    stop_ids = {tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<end_of_turn>")}
    gen_list = output_tokens[0, prompt_len:].tolist()
    cut = next((j for j, t in enumerate(gen_list) if t in stop_ids), len(gen_list))
    output_tokens = output_tokens[:, :prompt_len + cut]

    generated_ids  = output_tokens[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, output_tokens, prompt_len


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1: Mechanistic profiling")
    p.add_argument("--n",          type=int, default=N_PILOT,   help="Number of examples")
    p.add_argument("--results-dir",type=str, default=RESULTS_DIR)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip examples whose parquet already exists")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.results_dir) / "exp1"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME,
        dtype=getattr(torch, DTYPE),
        default_padding_side="left",
        device=args.device,
    )
    torch.set_default_dtype(getattr(torch, DTYPE))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    examples = load_aci_examples(n=args.n)

    all_example_meta = []

    for ex in examples:
        parquet_path = out_dir / f"tokens_enc{ex.idx}.parquet"
        patch_path   = out_dir / f"patch_enc{ex.idx}.npz"

        if args.skip_existing and parquet_path.exists() and patch_path.exists():
            print(f"  enc{ex.idx} — skipping (outputs exist)")
            continue

        print(f"\n── Encounter {ex.idx} ──────────────────────────────────────")
        t0 = time.time()

        # ── Step 1: generate note ────────────────────────────────────────
        gen_note, full_tokens, prompt_len = generate_note(model, tokenizer, ex.dialogue)
        ex.generated_note = gen_note
        n_gen = full_tokens.shape[1] - prompt_len
        print(f"  Generated {n_gen} tokens in {time.time()-t0:.1f}s")
        print(f"  Note preview: {gen_note[:120].replace(chr(10),' ')}")

        # ── Step 2: DLA via run_with_cache ────────────────────────────────
        print("  Computing DLA ...")
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                full_tokens,
                names_filter=cache_filter,
                return_type=None,
            )
        token_df = profile_example(model, cache, full_tokens, prompt_len)
        del cache
        torch.cuda.empty_cache()

        # ── Step 3: zero-ablation activation patching ─────────────────────
        print(f"  Running activation patching ({2 * model.cfg.n_layers} passes) ...")
        attn_patch, mlp_patch = run_activation_patching(model, full_tokens, prompt_len)

        # ── Save ──────────────────────────────────────────────────────────
        token_df.to_parquet(parquet_path)
        np.savez(patch_path, attn=attn_patch, mlp=mlp_patch)

        all_example_meta.append({
            "idx":            ex.idx,
            "n_gen_tokens":   n_gen,
            "prompt_len":     prompt_len,
            "generated_note": ex.generated_note,
            "parquet_path":   str(parquet_path),
            "patch_path":     str(patch_path),
        })

        print(f"  Done in {time.time()-t0:.1f}s  |  "
              f"DLA → {parquet_path.name}  patch → {patch_path.name}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta_path = out_dir / "example_meta.json"
    with open(meta_path, "w") as f:
        json.dump(all_example_meta, f, indent=2)
    print(f"\nMetadata → {meta_path}")


if __name__ == "__main__":
    main()
