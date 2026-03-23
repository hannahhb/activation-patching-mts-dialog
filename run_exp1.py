"""
Experiment 1 — Per-token mechanistic profiling of generated clinical notes.

Pipeline:
  1. Load ACI-Bench examples
  2. Generate SOAP notes via Gemma 2 9B-IT (greedy decoding)
  3. Run model.run_with_cache() on (prompt + generated note)
  4. Compute per-token DLA, lookback ratio, extractive score, SOAP section labels
  5. Compute per-encounter complexity features
  6. Save per-token DataFrames (parquet) and per-encounter feature JSON

Usage:
    python run_exp1.py [--n 10] [--results-dir results] [--device cuda]
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import pandas as pd
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from config import (
    MODEL_NAME, DTYPE, SOAP_PROMPT_TEMPLATE,
    MAX_NEW_TOKENS, GENERATION_TEMP,
    N_PILOT, RESULTS_DIR, CACHE_DIR,
    EARLY_LAYERS, MID_LAYERS, LATE_LAYERS,
)
from data import load_aci_examples, segment_soap, assign_token_sections, compute_complexity_features
from mechanistic import profile_example, aggregate_encounter_features, cache_filter
from visualise import plot_token_profile


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
        if GENERATION_TEMP == 0:
            output_tokens = model.generate(
                prompt_tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=None,
                do_sample=False,
                stop_at_eos=True,
                verbose=False,
            )
        else:
            output_tokens = model.generate(
                prompt_tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=GENERATION_TEMP,
                do_sample=True,
                stop_at_eos=True,
                verbose=False,
            )

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
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        dtype=getattr(torch, DTYPE),
        default_padding_side="left",
        device=args.device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    examples = load_aci_examples(n=args.n)

    all_encounter_features = []
    all_example_meta       = []

    for ex in examples:
        parquet_path = out_dir / f"tokens_enc{ex.idx}.parquet"
        if args.skip_existing and parquet_path.exists():
            print(f"  enc{ex.idx} — skipping (parquet exists)")
            token_df = pd.read_parquet(parquet_path)
            feats = aggregate_encounter_features(token_df)
            all_encounter_features.append(feats)
            continue

        print(f"\n── Encounter {ex.idx} ──────────────────────────────────────")
        t0 = time.time()

        # ── Step 1: generate note ────────────────────────────────────────
        gen_note, full_tokens, prompt_len = generate_note(model, tokenizer, ex.dialogue)
        ex.generated_note = gen_note
        print(f"  Generated {full_tokens.shape[1] - prompt_len} tokens in "
              f"{time.time()-t0:.1f}s")
        print(f"  Note preview: {gen_note[:120].replace(chr(10),' ')}")

        # ── Step 2: SOAP segmentation ─────────────────────────────────────
        sections = segment_soap(gen_note)
        ex.soap_sections = sections
        print(f"  Sections detected: {list(sections.keys())}")

        # ── Step 3: run_with_cache ────────────────────────────────────────
        print("  Running forward pass with cache ...")
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                full_tokens,
                names_filter=cache_filter,
                return_type=None,
            )

        # ── Step 4: per-token mechanistic profile ─────────────────────────
        # Build section labels for each generated token
        gen_token_strs = [
            tokenizer.decode([tid])
            for tid in full_tokens[0, prompt_len:].tolist()
        ]
        section_labels = assign_token_sections(gen_token_strs, gen_note, sections)

        token_df = profile_example(model, cache, full_tokens, prompt_len, section_labels)
        del cache  # free VRAM before next example
        torch.cuda.empty_cache()

        # ── Step 5: complexity features ───────────────────────────────────
        complexity = compute_complexity_features(ex.dialogue, tokenizer)
        compression = (
            (full_tokens.shape[1] - prompt_len) / max(prompt_len, 1)
        )
        complexity["compression_ratio"] = round(compression, 3)

        # ── Aggregate and save ────────────────────────────────────────────
        feats = aggregate_encounter_features(token_df)
        feats.update(complexity)
        feats["encounter_idx"]  = ex.idx
        all_encounter_features.append(feats)

        token_df.to_parquet(parquet_path)
        ex.token_df_path = str(parquet_path)

        plot_token_profile(token_df, ex.idx, out_dir=out_dir)

        # Save note text alongside
        all_example_meta.append({
            "idx":             ex.idx,
            "generated_note":  ex.generated_note,
            "soap_sections":   ex.soap_sections,
            "token_df_path":   ex.token_df_path,
            "complexity":      complexity,
        })

        print(f"  Done in {time.time()-t0:.1f}s")

    # ── Save aggregate outputs ────────────────────────────────────────────────
    feat_path = out_dir / "encounter_features.json"
    with open(feat_path, "w") as f:
        json.dump(all_encounter_features, f, indent=2)
    print(f"\nEncounter features → {feat_path}")

    meta_path = out_dir / "example_meta.json"
    with open(meta_path, "w") as f:
        json.dump(all_example_meta, f, indent=2)
    print(f"Example metadata  → {meta_path}")

    # Summary stats
    print("\n── Summary ──────────────────────────────────────────────────")
    feat_df = pd.DataFrame(all_encounter_features)
    for col in ["attn_fraction", "mlp_fraction", "lookback_ratio",
                "extractive_score", "entity_density"]:
        if col in feat_df.columns:
            print(f"  {col:<30} mean={feat_df[col].mean():.3f}  "
                  f"std={feat_df[col].std():.3f}")


if __name__ == "__main__":
    main()
