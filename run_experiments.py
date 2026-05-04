"""
run_experiments.py
==================
CLI entry point for ECS/PKS-based factuality analysis of clinical notes.
All experiment logic lives in experiments.py; this file is parse_args + main only.

Usage
-----
    python run_experiments.py                              # 2a + 2b, gemma, sample 0
    python run_experiments.py --exp 2c                    # only 2c
    python run_experiments.py --exp 3                     # Exp 3 (REDEEP scoring)
    python run_experiments.py --exp 3 --n-train-samples 5 --n-ffn-layers 3
    python run_experiments.py --exp 4 --n-examples 20    # Exp 4 (multi-example stats)
    python run_experiments.py --exp all                   # 2a + 2b + 2c + 2d + 3 + 4
    python run_experiments.py --model llama               # Llama 3 8B
    python run_experiments.py --sample 3                  # ACI-Bench row 3
"""

import argparse
import warnings
from pathlib import Path

from transformer_lens import HookedTransformer

from config import Config, load_aci_sample
from experiments import (
    run_experiment_2a,
    run_experiment_2b,
    run_experiment_2c,
    run_experiment_2d,
    run_experiment_3,
    run_experiment_4,
)

warnings.filterwarnings("ignore")

# Optional LLM-based hallucination injector
try:
    from halluc_llm import (
        HallucinationGenerationError,
        inject_hallucinations,
        inject_hallucinations_llm,
    )
    _LLM_HALLUC_AVAILABLE = True
except ImportError:
    _LLM_HALLUC_AVAILABLE = False


# ─────────────────────────────────────────────
# Injection backend resolver
# ─────────────────────────────────────────────

def _resolve_inject_fn(args):
    """
    Return the hallucination injection callable to use for Exp 2c/2d/3/4.

    Priority:
      1. --halluc-backend regex  → always use the built-in regex injector
      2. --halluc-backend hf/bedrock  → use inject_hallucinations_llm if
         halluc_llm loaded, else warn and fall back to regex
    """
    backend = args.halluc_backend

    if backend == "regex":
        print("  [inject] Hallucination backend: regex (rule-based)")
        return inject_hallucinations

    if not _LLM_HALLUC_AVAILABLE:
        warnings.warn(
            f"[inject] --halluc-backend={backend} requested but halluc_llm could not "
            f"be imported (missing huggingface_hub or boto3).  "
            f"Falling back to regex-based injection."
        )
        return inject_hallucinations

    llm_kwargs = {"backend": backend}
    if backend == "hf" and args.hf_model:
        llm_kwargs["hf_model"] = args.hf_model
    if backend == "bedrock":
        if args.bedrock_model:
            llm_kwargs["bedrock_model"] = args.bedrock_model
        if args.bedrock_region:
            llm_kwargs["bedrock_region"] = args.bedrock_region

    def _llm_inject(note: str, max_injections: int = 10, seed: int = 42):
        try:
            return inject_hallucinations_llm(note, max_injections=max_injections, **llm_kwargs)
        except HallucinationGenerationError as exc:
            warnings.warn(
                f"[inject] LLM injection failed ({exc}).  "
                f"Falling back to regex-based injection."
            )
            return inject_hallucinations(note, max_injections=max_injections, seed=seed)

    model_id = llm_kwargs.get("hf_model") or llm_kwargs.get("bedrock_model") or "(default)"
    print(f"  [inject] Hallucination backend: {backend}  model: {model_id}")
    return _llm_inject


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REDEEP ECS/PKS — clinical note experiments")

    # ── Model / data ────────────────────────────────────────────────────────────
    p.add_argument("--model", choices=["gemma", "llama"], default="gemma",
                   help="gemma → google/gemma-2-2b-it  |  llama → meta-llama/Meta-Llama-3-8B-instruct")
    p.add_argument("--exp",
                   choices=["2a", "2b", "2c", "2d", "3", "4", "both", "all"],
                   default="both",
                   help="Experiment(s) to run  (both=2a+2b [default]  |  all=2a…4)")
    p.add_argument("--sample", type=int, default=0,
                   help="ACI-Bench test1 row index for the primary sample (default: 0)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens for note generation in Exp 2b (default: 512)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0.0 = greedy)")
    p.add_argument("--out", default=".", help="Output directory for plots and CSV")

    # ── Hallucination injection ─────────────────────────────────────────────────
    p.add_argument("--halluc-backend", choices=["regex", "hf", "bedrock"], default="bedrock",
                   help="Injection method for Exp 2c/2d/3/4  (regex | hf | bedrock)")
    p.add_argument("--hf-model", default=None,
                   help="HuggingFace model ID for --halluc-backend hf")
    p.add_argument("--bedrock-model", default=None,
                   help="Bedrock model ID for --halluc-backend bedrock")
    p.add_argument("--bedrock-region", default=None,
                   help="AWS region for --halluc-backend bedrock")

    # ── Exp 3 ───────────────────────────────────────────────────────────────────
    p.add_argument("--n-train-samples", type=int, default=3,
                   help="Training samples for Exp 3 (default: 3)")
    p.add_argument("--n-injections", type=int, default=5,
                   help="Max injections per training sample in Exp 3/4 (default: 5)")
    p.add_argument("--n-ffn-layers", type=int, default=5,
                   help="Size of set F (Knowledge FFN layers) in Exp 3 (default: 5)")
    p.add_argument("--n-copy-layers", type=int, default=5,
                   help="Size of set A (Copying Head layers) in Exp 3 (default: 5)")
    p.add_argument("--halluc-threshold", type=float, default=0.5,
                   help="Probability cutoff for binary hallucination flag in Exp 3 (default: 0.5)")
    p.add_argument("--exp4-out", default=None,
                   help="Path to a previous Exp 4 output directory.  If set, Exp 3 loads "
                        "pre-computed discriminability stats and saved activations from that "
                        "run instead of re-running forward passes for training data.")

    # ── Exp 4 ───────────────────────────────────────────────────────────────────
    p.add_argument("--n-examples", type=int, default=10,
                   help="Number of ACI-Bench examples to process in Exp 4 (default: 10)")
    p.add_argument("--sample-start", type=int, default=0,
                   help="First ACI-Bench row index for Exp 4 range (default: 0)")

    return p.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = Config(
        model_name=(
            "google/gemma-2-2b-it"
            if args.model == "gemma"
            else "meta-llama/Meta-Llama-3-8B-instruct"
        ),
        sample_idx=args.sample,
        max_new_tokens=args.max_new_tokens,
        gen_temperature=args.temperature,
        output_dir=args.out,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}  |  dtype: {cfg.dtype}")
    print(f"  Dataset        : {cfg.dataset_repo}  [{cfg.dataset_config} / {cfg.dataset_split}]")
    print(f"  Sample index   : {cfg.sample_idx}")
    print(f"  Max new tokens : {cfg.max_new_tokens}  |  temperature: {cfg.gen_temperature}")
    print(f"  Output dir     : {out.resolve()}")

    transcript, gold_note = load_aci_sample(cfg)

    print(f"\nLoading {cfg.model_name} via TransformerLens …")
    model = HookedTransformer.from_pretrained(
        cfg.model_name,
        dtype=cfg.dtype,
        default_padding_side="right",
    )
    model.eval()
    model.to(cfg.device)
    print(f"  Layers: {model.cfg.n_layers}  |  Heads: {model.cfg.n_heads}"
          f"  |  d_model: {model.cfg.d_model}")

    if args.exp in ("2a", "all"):
        run_experiment_2a(model, cfg, out, transcript, gold_note)

    if args.exp in ("2b", "all"):
        run_experiment_2b(model, cfg, out, transcript, gold_note)

    if args.exp in ("2c", "both", "all"):
        inject_fn = _resolve_inject_fn(args)
        run_experiment_2c(model, cfg, out, transcript, gold_note, inject_fn=inject_fn)

    if args.exp in ("2d", "both", "all"):
        inject_fn = _resolve_inject_fn(args)
        run_experiment_2d(model, cfg, out, transcript, gold_note, inject_fn=inject_fn)

    if args.exp in ("3", "all"):
        inject_fn = _resolve_inject_fn(args)
        run_experiment_3(
            model, cfg, out, transcript, gold_note,
            inject_fn=inject_fn,
            n_train_samples=args.n_train_samples,
            n_injections=args.n_injections,
            n_ffn_layers=args.n_ffn_layers,
            n_copy_layers=args.n_copy_layers,
            halluc_threshold=args.halluc_threshold,
            exp4_out=Path(args.exp4_out) if args.exp4_out else None,
        )

    if args.exp in ("4", "all"):
        inject_fn = _resolve_inject_fn(args)
        run_experiment_4(
            model, cfg, out,
            inject_fn=inject_fn,
            n_examples=args.n_examples,
            n_injections=args.n_injections,
            sample_start=args.sample_start,
        )

    print("\n" + "═"*54)
    print("  Done.  All outputs written to:", out.resolve())
    print("═"*54 + "\n")


if __name__ == "__main__":
    main()
