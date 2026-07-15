"""
mlp_contribution.py
===================
Signed, layer-wise DIRECT MLP contribution toward the actually-generated span,
as an interpretable replacement for ReDeEP's unsigned JSD-based PKS.

Motivation
----------
ReDeEP's PKS (redeep_sentence.py) measures the JSD between logit-lens(resid_mid)
and logit-lens(resid_post): how much an FFN CHANGES the output distribution. It
is unsigned — it cannot say whether the FFN pushed toward the hallucinated token
or away from it. This module computes, per layer l and generated token t:

    Δ_MLP^{l,t} = logit_lens_{y_t}(resid_mid + mlp_out) - logit_lens_{y_t}(resid_mid)

i.e. the SIGNED change in the logit of the token the model actually produced,
caused by layer l's MLP.  Δ>0: the MLP increased support for the generated
token; Δ<0: it opposed it. This is the MLP-side analogue of the Lookback Lens
attention feature: each span becomes a vector M_S = [Δ̄_{1,S}, …, Δ̄_{L,S}],
one signed contribution per layer, averaged over the span's tokens.

Two precision decisions (this model runs bf16, fold_ln=False):
  * We capture hook_mlp_out DIRECTLY and reconstruct resid_post = resid_mid +
    mlp_out in fp32, rather than reading hook_resid_post and differencing two
    large bf16 residuals — the latter loses the small MLP signal in bf16
    rounding (verified: ~8x larger error, which swamps Δ≈0.01 values).
  * The logit lens (final RMSNorm + unembed) is done in fp32. Because Δ is a
    difference of two logits that share the SAME resid_mid, the residual's bf16
    rounding largely cancels; only the (directly captured) mlp_out drives Δ.

Off-by-one (handled internally): the logit for generated token at note-relative
position r is produced at sequence position (T + r − 1). Δ is therefore read at
prediction position T+r−1 with target id = note token r, and STORED at index r.
So span token ranges [a, b) are identical to the Lookback Lens ones — no
separate alignment. (Same span construction is imported from lookback_lens.)

Caveat inherited from the logit lens: readout of intermediate residuals in the
final unembedding basis is unreliable at early/middle layers (tuned-lens
literature). Signedness fixes PKS's direction problem, not the lens's
readability problem — interpret early-layer Δ with care. ReDeEP PKS remains
available in redeep_sentence.py as the unsigned baseline.

Outputs (under --out)
---------------------
  per_span_mlp.csv     note_id, span_id, label, start_token, end_token,
                       span_length, abs_start, mlp_0 … mlp_{L-1}
  class_delta_star.csv Δ*_{c,l} = E[Δ̄_l | c] − E[Δ̄_l | faithful] per class/layer
  class_raw_means.csv  E[Δ̄_l | class] per class/layer (faithful included)
  class_delta_star_lenmatched.csv  length-bucketed Δ*_{c,l} (confound control)
  fig_mlp_delta_star.png  Δ*_{c,l} across layers, one line per hallucination type
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redeep_sentence import (
    load_model,
    tokenize_prompt_and_note,
    find_sentence_token_spans,
)
from lookback_lens import (
    build_char_token_map,
    note_span_units,
    load_labeled_sentences,
)

CREOLA = ["fabrication", "contextual", "negation", "causality"]


# ─────────────────────────────────────────────────────────────────────────────
# fp32 logit lens (final norm + target-column unembed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_lens(model, target_ids: torch.Tensor):
    """
    Return a fn: resid[N,d_model] -> logit[N] of each row's own target token,
    computed in fp32 through the model's FINAL normalisation and unembedding.
    The unembed bias cancels in Δ, so it is omitted.
    """
    norm_type = getattr(model.cfg, "normalization_type", "RMS") or "RMS"
    eps = float(getattr(model.cfg, "eps", 1e-5))
    w = model.ln_final.w.detach().float()                        # [d_model]
    b = getattr(model.ln_final, "b", None)
    b = b.detach().float() if b is not None else None
    WU_cols = model.W_U[:, target_ids].detach().float()          # [d_model, N]
    WU_rows = WU_cols.T                                          # [N, d_model]

    def lens(resid: torch.Tensor) -> torch.Tensor:
        x = resid.float()                                        # [N, d_model]
        if norm_type in ("LN", "LNPre"):
            x = x - x.mean(-1, keepdim=True)
        scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        x = x / scale * w
        if b is not None:
            x = x + b
        return (x * WU_rows).sum(-1)                             # [N]
    return lens


def compute_mlp_contributions(model, full_ids, transcript_len) -> np.ndarray:
    """
    One teacher-forced forward pass; return delta of shape (n_layers, n_note),
    where delta[l, r] is the signed MLP contribution of layer l toward the
    generated note token at note-relative position r.
    """
    device = full_ids.device
    seq = full_ids.shape[1]
    T = transcript_len
    n_note = seq - T
    if n_note <= 0:
        raise ValueError("note region is empty")
    if T < 1:
        raise ValueError("no prediction position before the first note token")

    n_layers = model.cfg.n_layers
    target_ids = full_ids[0, T:T + n_note]                       # [n_note]
    lens = _make_lens(model, target_ids)

    # Prediction positions: token r (abs T+r) is predicted at position T+r-1.
    pred_pos = torch.arange(T - 1, T - 1 + n_note, device=device)  # [n_note]

    z_pre = torch.zeros((n_layers, n_note), device=device)
    z_post = torch.zeros((n_layers, n_note), device=device)
    rm_buf: Dict[int, torch.Tensor] = {}

    def make_mid(l):
        def hook(value, hook):
            rm = value[0][pred_pos].float()                     # [n_note, d_model]
            z_pre[l] = lens(rm)
            rm_buf[l] = rm                                       # reused by mlp_out hook
            return value
        return hook

    def make_mlp(l):
        def hook(value, hook):
            mo = value[0][pred_pos].float()                     # [n_note, d_model]
            rm = rm_buf.pop(l)
            z_post[l] = lens(rm + mo)                           # resid_post reconstructed
            return value
        return hook

    fwd_hooks = []
    for l in range(n_layers):
        fwd_hooks.append((f"blocks.{l}.hook_resid_mid", make_mid(l)))
        fwd_hooks.append((f"blocks.{l}.hook_mlp_out", make_mlp(l)))

    with torch.no_grad():
        model.run_with_hooks(full_ids, return_type=None, fwd_hooks=fwd_hooks)

    return (z_post - z_pre).cpu().numpy()                        # [n_layers, n_note]


def span_mlp_features(delta: np.ndarray,
                      spans: List[Tuple[int, int]]) -> np.ndarray:
    """Average Δ over each span's note-relative token range -> (n_spans, n_layers)."""
    n_layers, n_note = delta.shape
    feats = np.full((len(spans), n_layers), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(spans):
        a = max(0, a)
        b = min(n_note, b)
        if b <= a:
            continue
        feats[i] = delta[:, a:b].mean(axis=1)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset build (mirrors lookback_lens.build_dataset; same span indices)
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp_dataset(
    model,
    span_files: List[Path],
    gen_dir: Path,
    device: str,
    unit: str,
    clean_mode: str,
) -> Tuple[pd.DataFrame, int]:
    gen_cache: Dict[int, dict] = {}
    rows: List[dict] = []
    n_layers = model.cfg.n_layers

    for span_csv in span_files:
        m = re.search(r"sample_(\d+)_note_(\d+)", span_csv.stem)
        if not m:
            continue
        si, k = int(m.group(1)), int(m.group(2))

        df = pd.read_csv(span_csv)
        if df.empty or "sentence" not in df.columns:
            continue

        if si not in gen_cache:
            gen_path = gen_dir / f"sample_{si:03d}_generations.json"
            if not gen_path.exists():
                print(f"  [skip] no generations for sample_{si:03d}")
                continue
            with open(gen_path) as f:
                gen_cache[si] = json.load(f)
        gen_data = gen_cache[si]
        notes = gen_data["notes"]
        if k >= len(notes):
            continue
        transcript, note = gen_data["transcript"], notes[k]

        try:
            full_ids, T = tokenize_prompt_and_note(model, transcript, note, device)
        except Exception as exc:
            print(f"  [skip] sample_{si:03d}_note_{k:02d} tokenise: {exc}")
            continue

        if unit == "span":
            note_text, char_to_tok, n_search = build_char_token_map(model, full_ids, T)
            spans, labs, typs, n_att, n_found = note_span_units(
                df, note_text, char_to_tok, n_search, clean_mode)
            diag = f"hallu_spans={n_found}/{n_att}"
        else:
            sentences, lab_arr = load_labeled_sentences(span_csv)
            if len(sentences) == 0:
                continue
            spans, n_fail, n_fuzzy = find_sentence_token_spans(
                model, full_ids, T, sentences)
            labs = list(lab_arr)
            typs = ["Hallucinated" if v == 1 else "Faithful" for v in lab_arr]
            diag = f"span_fail={n_fail}, fuzzy={n_fuzzy}"

        if not spans:
            continue

        try:
            delta = compute_mlp_contributions(model, full_ids, T)   # (L, n_note)
        except Exception as exc:
            print(f"  [skip] sample_{si:03d}_note_{k:02d} forward: {exc}")
            continue

        feats = span_mlp_features(delta, spans)                      # (n_spans, L)
        note_id = f"{si:03d}_{k:02d}"
        n_pos = 0
        for j, (a, b) in enumerate(spans):
            v = feats[j]
            if np.all(np.isnan(v)):
                continue
            lab = str(typs[j]).strip().lower()
            row = {
                "note_id": note_id, "span_id": j, "label": lab,
                "start_token": int(a), "end_token": int(b),
                "span_length": int(b - a), "abs_start": int(T + a),
            }
            row.update({f"mlp_{l}": float(v[l]) for l in range(n_layers)})
            rows.append(row)
            n_pos += int(labs[j] == 1)
        print(f"  sample_{si:03d}_note_{k:02d}: {len(spans)} {unit}s, "
              f"{n_pos} hallucinated  (T={T}, {diag})")

    if not rows:
        raise RuntimeError("no usable spans — check --spans / --generations paths")
    return pd.DataFrame(rows), n_layers


# ─────────────────────────────────────────────────────────────────────────────
# Class comparison + plot
# ─────────────────────────────────────────────────────────────────────────────

def _mlp_matrix(df: pd.DataFrame, n_layers: int) -> np.ndarray:
    return df[[f"mlp_{l}" for l in range(n_layers)]].to_numpy(dtype=float)


def summarize(df: pd.DataFrame, n_layers: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_span_mlp.csv", index=False)

    cols = [f"mlp_{l}" for l in range(n_layers)]
    present = [c for c in df["label"].unique()]
    faithful = df[df["label"] == "faithful"]
    if faithful.empty:
        print("  [warn] no faithful spans — Δ* undefined; writing raw means only")

    # Raw per-class means (faithful included).
    raw = df.groupby("label")[cols].mean()
    raw.insert(0, "n", df.groupby("label").size())
    raw.to_csv(out_dir / "class_raw_means.csv")

    if not faithful.empty:
        faith_mean = faithful[cols].mean().to_numpy()
        # Marginal Δ*_{c,l}.
        rows = []
        for c in [x for x in CREOLA if x in present]:
            sub = df[df["label"] == c]
            dstar = sub[cols].mean().to_numpy() - faith_mean
            rows.append([c, len(sub)] + list(dstar))
        star = pd.DataFrame(rows, columns=["label", "n"] + cols)
        star.to_csv(out_dir / "class_delta_star.csv", index=False)

        # Length-bucketed Δ* (control span-length confound).
        qs = df["span_length"].quantile([0, .25, .5, .75, 1.0]).to_numpy()
        edges = np.unique(qs)
        df = df.copy()
        df["_lb"] = np.clip(np.searchsorted(edges, df["span_length"], "right") - 1,
                            0, len(edges) - 2)
        lm_rows = []
        for c in [x for x in CREOLA if x in present]:
            acc = np.zeros(n_layers)
            wsum = 0
            for lb in sorted(df["_lb"].unique()):
                cc = df[(df["label"] == c) & (df["_lb"] == lb)]
                ff = df[(df["label"] == "faithful") & (df["_lb"] == lb)]
                if cc.empty or ff.empty:
                    continue
                acc += (cc[cols].mean().to_numpy()
                        - ff[cols].mean().to_numpy()) * len(cc)
                wsum += len(cc)
            if wsum:
                lm_rows.append([c, wsum] + list(acc / wsum))
        if lm_rows:
            pd.DataFrame(lm_rows, columns=["label", "n_matched"] + cols).to_csv(
                out_dir / "class_delta_star_lenmatched.csv", index=False)

        # Plot Δ*_{c,l} across layers.
        fig, ax = plt.subplots(figsize=(9, 5))
        layers = np.arange(n_layers)
        for _, r in star.iterrows():
            ax.plot(layers, r[cols].to_numpy(dtype=float), marker="o", ms=3,
                    lw=1.8, label=f"{r['label']} (n={int(r['n'])})")
        ax.axhline(0, color="grey", lw=1, ls="--", label="faithful baseline")
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\Delta^*_{c,l}$ = mean MLP push toward span "
                      r"$-$ faithful")
        ax.set_title("Signed direct MLP contribution by hallucination type")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_mlp_delta_star.png", dpi=150)
        plt.close(fig)

    # Console summary: peak discriminative layer per type.
    print(f"\nSpans: {len(df)}  ({', '.join(f'{k}={v}' for k, v in df['label'].value_counts().items())})")
    if not faithful.empty:
        print("Peak |Δ*| layer per type:")
        for _, r in star.iterrows():
            v = r[cols].to_numpy(dtype=float)
            l = int(np.nanargmax(np.abs(v)))
            print(f"  {r['label']:<12} layer {l:2d}  Δ*={v[l]:+.3f}  (n={int(r['n'])})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spans", required=True,
                    help="dir of sample_*_note_*_span_judge.csv label files")
    ap.add_argument("--generations", required=True,
                    help="dir of sample_*_generations.json")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="luq_out/mlp_contribution")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--unit", choices=["sentence", "span"], default="span",
                    help="'span' (default): predefined annotated spans, CREOLA "
                         "labels. 'sentence': one unit per judged sentence.")
    ap.add_argument("--clean-mode", choices=["sentence", "paper"],
                    default="sentence",
                    help="[--unit span] negative/faithful span construction.")
    args = ap.parse_args()

    span_files = sorted(Path(args.spans).glob("sample_*_span_judge.csv"))
    if args.samples is not None:
        span_files = span_files[:args.samples]
    if not span_files:
        sys.exit(f"no span CSVs under {args.spans}")

    print(f"Loading {args.model} …")
    model = load_model(args.model, args.device)
    print(f"Building {args.unit}-level MLP contributions from "
          f"{len(span_files)} notes …")
    df, n_layers = build_mlp_dataset(
        model, span_files, Path(args.generations), args.device,
        unit=args.unit, clean_mode=args.clean_mode)
    summarize(df, n_layers, Path(args.out))


if __name__ == "__main__":
    main()
