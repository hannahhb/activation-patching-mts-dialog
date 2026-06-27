"""
circuit_tracing.py
==================
Attribution graph analysis for SOAP note generation using Llama 3.1 8B Instruct
and Facebook's CRV transcoders.

For each sample the full chat-formatted prompt (transcript + SOAP template) is
passed to attribute(), which computes the attribution graph for what the model
predicts next — i.e. what internal features drive the generation of this note.

Install:
  pip install git+https://github.com/zsquaredz/circuit-tracer.git

Usage:
  python circuit_tracing.py --start 0 --end 10
  python circuit_tracing.py --sample 5
  python circuit_tracing.py --start 0 --end 10 --server
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
from tqdm import tqdm

from prompts import build_prompt

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_LLAMA_MODEL     = "meta-llama/Llama-3.1-8B-Instruct"
_TRANSCODER_REPO = "facebook/crv-8b-instruct-transcoders"

_BOS        = "<|begin_of_text|>"
_USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
_USER_END   = "<|eot_id|>"
_ASST_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _format_prompt(transcript: str) -> str:
    """Full Llama 3.1 Instruct chat-formatted prompt for SOAP note generation."""
    return (
        f"{_BOS}"
        f"{_USER_START}{build_prompt(transcript)}{_USER_END}"
        f"{_ASST_START}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_model: Optional[ReplacementModel] = None


def _get_model(dtype: torch.dtype) -> ReplacementModel:
    global _model
    if _model is None:
        print(f"[model] Loading {_LLAMA_MODEL} + {_TRANSCODER_REPO} …")
        _model = ReplacementModel.from_pretrained(
            _LLAMA_MODEL,
            _TRANSCODER_REPO,
            dtype=dtype,
        )
        print("[model] Ready.")
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    sample_indices: list[int],
    gen_idx: int,
    generations_dir: Path,
    out_dir: Path,
    batch_size: int,
    offload: str,
    node_threshold: float,
    edge_threshold: float,
    serve: bool,
    dtype: torch.dtype,
) -> None:
    model     = _get_model(dtype)
    out_dir.mkdir(parents=True, exist_ok=True)

    for si in tqdm(sample_indices, desc="circuit trace", unit="sample"):
        gen_path = generations_dir / f"sample_{si:03d}_generations.json"
        if not gen_path.exists():
            tqdm.write(f"  [skip] sample {si}: no generation file.")
            continue

        saved = json.loads(gen_path.read_text())
        notes = saved["notes"]
        if gen_idx >= len(notes):
            tqdm.write(f"  [skip] sample {si}: gen_idx {gen_idx} out of range (K={len(notes)}).")
            continue

        transcript = saved["transcript"]
        prompt     = _format_prompt(transcript)
        slug       = f"sample_{si:03d}_gen_{gen_idx:02d}"
        graph_path = out_dir / f"{slug}.pt"

        tqdm.write(f"\n[trace] Sample {si}  prompt={len(prompt)} chars")

        graph = attribute(
            prompt=prompt,
            model=model,
            max_n_logits=10,
            desired_logit_prob=0.95,
            batch_size=batch_size,
            max_feature_nodes=8192,
            offload=offload,
            verbose=False,
        )

        graph.to_pt(graph_path)

        create_graph_files(
            graph_or_path=graph_path,
            slug=slug,
            output_path=str(out_dir / "graph_files"),
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
        tqdm.write(f"  [saved] {graph_path.name}")

    if serve:
        from circuit_tracer.server import serve as ct_serve
        print(f"\n[server] Launching viewer → http://localhost:8041")
        ct_serve(graph_file_dir=str(out_dir / "graph_files"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Circuit tracing for SOAP note generation via CRV transcoders"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=int,
                   help="Single sample index")
    g.add_argument("--start",  type=int,
                   help="Start of sample range (inclusive)")

    p.add_argument("--end",             type=int, default=None,
                   help="End of sample range (exclusive) — required with --start")
    p.add_argument("--gen-idx",         type=int, default=0,
                   help="Which generation to use (default 0)")
    p.add_argument("--generations-dir", default="luq_out/llama/generations")
    p.add_argument("--out",             default="circuit_graphs")
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--offload",         default="cpu", choices=["cpu", "disk", "none"])
    p.add_argument("--node-threshold",  type=float, default=0.8)
    p.add_argument("--edge-threshold",  type=float, default=0.98)
    p.add_argument("--server",          action="store_true",
                   help="Launch local viewer after attribution")
    p.add_argument("--dtype",           default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.sample is not None:
        indices = [args.sample]
    else:
        if args.end is None:
            raise ValueError("--end is required when using --start")
        indices = list(range(args.start, args.end))

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    run(
        sample_indices  = indices,
        gen_idx         = args.gen_idx,
        generations_dir = Path(args.generations_dir),
        out_dir         = Path(args.out),
        batch_size      = args.batch_size,
        offload         = args.offload,
        node_threshold  = args.node_threshold,
        edge_threshold  = args.edge_threshold,
        serve           = args.server,
        dtype           = dtype,
    )
