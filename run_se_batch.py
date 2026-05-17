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

Semantic entropy is then computed as:
  1. Sentence-split each of the K notes.
  2. Align sentences across K samples by SOAP section then position index.
  3. For each sentence position, run bidirectional NLI on all K(K-1) pairs.
  4. Connected-component clustering on the entailment adjacency matrix.
  5. SE[pos] = -Σ_c (|c|/K) * log(|c|/K)   (Shannon entropy over clusters)
  6. Each token inherits the SE of its sentence → token_se_scores (note_len,)

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
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_VARIANTS = [
    # Variant 0 — baseline (identical to _GENERATION_PROMPT)
    (
        "You are a clinical documentation assistant.\n"
        "Given the following patient-clinician conversation, write a concise clinical note "
        "with exactly six sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
        "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN.\n"
        "Use only information present in the conversation. Do not add disclaimers or preamble.\n\n"
        "### Conversation\n{transcript}\n\n"
        "### Note:\n"
    ),
    # Variant 1 — re-ordered instruction emphasis
    (
        
        "You are a clinical documentation assistant.\n"
        "Read the conversation below and produce a structured clinical note "
        "with exactly six sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
        "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN.\n"
        "Base the note strictly on what is said in the conversation. "
        "Do not add disclaimers or preamble.\n\n"
        "### Conversation\n{transcript}\n\n"
        "### Note:\n"

    ),
    # Variant 2 — patient-centred framing
    (
        "As a clinical documentation specialist, summarise the following "
        "doctor-patient encounter into a formal clinical note.\n"
        "Include these sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
        "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN.\n"
        "Only include facts explicitly mentioned in the conversation.\n\n"
        "Conversation:\n{transcript}\n\n"
        "Note:\n"
    ),
    # Variant 3 — brief instruction
    (
        "Write a clinical note from the conversation.\n"
        "Sections required: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
        "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN.\n"
        "Stick to information in the conversation only.\n\n"
        "{transcript}\n\n"
        "### Note:\n"
    ),
    # Variant 4 — formal EHR framing
    (
        "You are generating an electronic health record entry.\n"
        "Transcribe the clinical encounter below into a structured note with sections: "
        "CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, REVIEW OF SYSTEMS, "
        "PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN.\n"
        "Do not fabricate information not present in the conversation.\n\n"
        "Encounter transcript:\n{transcript}\n\n"
        "EHR Note:\n"
    ),
    # Variant 5 — imperative minimal
    (
        
        "Convert the following medical conversation into a structured clinical note.\n"
        "Use exactly six sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, "
        "REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN.\n"
        "Use only information present in the conversation. "
        "Do not add disclaimers or preamble.\n\n"
        "Conversation:\n{transcript}\n\n"
        "Note:\n"

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
# NLI-based semantic entropy  (reuses metrics.py helpers)
# ─────────────────────────────────────────────────────────────────────────────

def compute_se_for_sample(
    model: HookedTransformer,
    transcript: str,
    cfg: Config,
    K: int = 10,
    temperature: float = 0.8,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    nli_threshold: float = 0.5,
) -> Optional[Dict]:
    """
    Generate K notes with rotating prompt variants + temperature sampling,
    then compute sentence-level semantic entropy and token-level predictive entropy.

    Returns dict with keys:
        notes            : List[str] — K generated notes
        token_se         : (note_len,) — NLI-based SE per token (sentence granularity)
        token_pred_ent   : (note_len,) — predictive entropy per token (logit-based)
        note_tokens      : List[str]   — decoded note tokens
        note_len         : int
        mean_se          : float
        mean_pred_ent    : float
        se_scores        : (n_sentences,) — sentence-level SE
        sentences        : List[str]   — sentences from first generated note
    Returns None on failure.
    """
    # ── Step 1: K generations with rotating prompt variants ──────────────────
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
    print(f"    [SE] generated {K_actual} notes")

    # ── Step 2: NLI-based semantic entropy via metrics.py ────────────────────
    # Import the internal helpers directly to reuse the existing implementation
    from metrics import _split_sentences, _section_aligned_sentences, _get_nli_model

    sentences_per_sample = [_split_sentences(n) for n in notes]
    position_groups      = _section_aligned_sentences(sentences_per_sample)
    n_positions          = len(position_groups)

    try:
        nli_model = _get_nli_model(nli_model_name)
    except Exception as exc:
        print(f"    [SE] NLI model load failed: {exc}")
        return None

    se_scores = np.zeros(n_positions, dtype=np.float64)

    for pos_idx, group in enumerate(position_groups):
        valid = [s for s in group if s.strip()]
        if len(valid) < 2:
            se_scores[pos_idx] = 0.0
            continue

        pairs = [(a, b) for i, a in enumerate(valid)
                 for j, b in enumerate(valid) if i != j]
        try:
            raw = np.array(nli_model.predict(pairs, apply_softmax=True))
            entail_scores = raw[:, 0] if raw.ndim == 2 else raw
        except Exception:
            se_scores[pos_idx] = 0.0
            continue

        n_v = len(valid)
        adj = np.zeros((n_v, n_v), dtype=bool)
        pair_idx = 0
        for i in range(n_v):
            for j in range(n_v):
                if i != j:
                    adj[i, j] = entail_scores[pair_idx] > nli_threshold
                    pair_idx += 1
        # Bidirectional: both directions must hold
        sym = adj & adj.T

        # Connected components via BFS
        visited = [False] * n_v
        clusters = []
        for start in range(n_v):
            if visited[start]:
                continue
            cluster = []
            queue   = [start]
            while queue:
                node = queue.pop()
                if visited[node]:
                    continue
                visited[node] = True
                cluster.append(node)
                queue.extend(j for j in range(n_v) if sym[node, j] and not visited[j])
            clusters.append(cluster)

        eps = 1e-10
        se_scores[pos_idx] = -sum(
            (len(c) / n_v) * np.log(len(c) / n_v + eps)
            for c in clusters
        )

    # ── Step 3: Map sentence SE → token level using first note ───────────────
    ref_note    = notes[0]
    note_ids    = model.tokenizer.encode(ref_note, add_special_tokens=False)
    note_len    = len(note_ids)
    note_tokens = [model.tokenizer.decode([tid]) for tid in note_ids]

    ref_sentences = _split_sentences(ref_note)
    token_se      = np.zeros(note_len, dtype=np.float64)

    # Map each token to its sentence via character offset
    try:
        enc     = model.tokenizer(ref_note, return_offsets_mapping=True,
                                  add_special_tokens=False)
        offsets = enc["offset_mapping"]   # list of (char_start, char_end)
    except Exception:
        offsets = None

    if offsets:
        # Build sentence char spans
        cursor      = 0
        sent_spans  = []
        for sent in ref_sentences:
            idx = ref_note.find(sent, cursor)
            if idx == -1:
                idx = cursor
            sent_spans.append((idx, idx + len(sent)))
            cursor = idx + len(sent)

        # Assign each token the SE of the sentence whose span contains it
        # Use the sentence's position index in position_groups
        for ti, (cs, ce) in enumerate(offsets):
            if ti >= note_len:
                break
            mid = (cs + ce) / 2
            best_sent = 0
            for si, (ss, se_end) in enumerate(sent_spans):
                if ss <= mid < se_end:
                    best_sent = si
                    break
            # Map sent index → position_groups index (capped)
            pos_idx = min(best_sent, n_positions - 1)
            token_se[ti] = se_scores[pos_idx]
    else:
        # Fallback: divide note tokens evenly across sentence positions
        for ti in range(note_len):
            pos_idx = min(int(ti * n_positions / note_len), n_positions - 1)
            token_se[ti] = se_scores[pos_idx]

    # ── Step 4: Token predictive entropy (single forward pass) ───────────────
    token_pred_ent = compute_token_predictive_entropy(model, transcript, ref_note, cfg)
    # Align length (generation may be slightly longer/shorter than encoding)
    min_len = min(note_len, len(token_pred_ent))
    token_se       = token_se[:min_len]
    token_pred_ent = token_pred_ent[:min_len]
    note_tokens    = note_tokens[:min_len]
    note_len       = min_len

    return {
        "notes":          notes,
        "token_se":       token_se,
        "token_pred_ent": token_pred_ent,
        "note_tokens":    note_tokens,
        "note_len":       note_len,
        "mean_se":        float(token_se.mean()),
        "mean_pred_ent":  float(token_pred_ent.mean()),
        "se_scores":      se_scores,
        "sentences":      ref_sentences,
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
) -> pd.DataFrame:
    """
    Run semantic entropy over ACI-Bench test1 rows [start, end).
    """
    se_out      = out / "se_batch_out"
    token_dir   = se_out / "token_scores"
    se_out.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)

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

        # ── Save per-token scores ─────────────────────────────────────────────
        tok_df = pd.DataFrame({
            "token_idx":       np.arange(result["note_len"]),
            "token_str":       result["note_tokens"],
            "semantic_entropy":   result["token_se"].round(4),
            "predictive_entropy": result["token_pred_ent"].round(4),
        })
        tok_df.to_csv(token_dir / f"sample_{si:03d}_tokens.csv", index=False)

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
    )

    print("\n" + "═" * 54)
    print(f"  Done.  Results in: {out.resolve() / 'se_batch_out'}")
    print("═" * 54 + "\n")


if __name__ == "__main__":
    main()
