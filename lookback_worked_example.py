"""
lookback_worked_example.py
==========================
Single-note qualitative walk-through of the Lookback Lens: for ONE generated
SOAP note, compute the per-token lookback ratio (context vs. generated attention,
averaged over all layers/heads) and show it three ways, with the LLM-judge
hallucinated spans highlighted:

  1. TEXT HEATMAP  — the note rendered token-by-token, each token shaded by its
     lookback ratio (blue = attends to transcript/context, orange = attends to
     its own generation); judge-flagged hallucinated spans boxed in red.
  2. WINDOW LEVEL  — sliding fixed-size window (default 8 tokens) lookback across
     the note; the annotation-free view a guided-decoder would see.
  3. SPAN LEVEL    — mean lookback per annotated span, hallucinated (colored by
     CREOLA type) vs faithful (grey).

Lookback ratio per token t (paper §2.1), averaged over heads:
    LR_t = mean_{l,h} [ A_ctx / (A_ctx + A_new) ]
    A_ctx = mean attention to context tokens [0,T) ; A_new = to generated [T,t)

Output: a single self-contained <out>/worked_example.html (text heatmap + an
embedded PNG of the window/span panels). Reuses the validated
compute_lookback_ratios and the same tokenization/span machinery as the rest of
the pipeline, so this is a faithful zoom-in on one row of the dataset.

Example (the fabricated-prescription note):
  python3 lookback_worked_example.py \
    --span-csv    luq_out/llama_judge/aci/test3/spans/sample_018_note_00_span_judge.csv \
    --generations luq_out/llama/generations/aci/test3 \
    --out luq_out/worked_example/sample_018
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redeep_sentence import load_model, tokenize_prompt_and_note
from lookback_lens import (
    compute_lookback_ratios,
    build_char_token_map,
    find_span_char_range,
)

TYPE_COLOR = {"fabrication": "#d62728", "contextual": "#1f77b4",
              "negation": "#2ca02c", "causality": "#9467bd",
              "hallucinated": "#d62728"}


def per_token_strings(model, full_ids, transcript_len) -> Tuple[List[str], int]:
    """Surface string of each note token via prefix decoding (handles spaces/BPE)."""
    tok = model.tokenizer
    ids = full_ids[0, transcript_len:].tolist()
    special = set(tok.all_special_ids or [])
    n = len(ids)
    while n > 0 and ids[n - 1] in special:
        n -= 1
    strs, prev = [], ""
    for i in range(n):
        cur = tok.decode(ids[: i + 1], skip_special_tokens=False)
        strs.append(cur[len(prev):])
        prev = cur
    return strs, n


def map_spans_to_tokens(model, full_ids, T, span_strings):
    """Each span string -> note-relative [a,b) via cursor-free char matching."""
    note_text, char_to_tok, n_search = build_char_token_map(model, full_ids, T)
    out = []
    for s in span_strings:
        rng = find_span_char_range(note_text, s)
        if rng is None:
            out.append(None)
            continue
        cs, ce = rng
        a = char_to_tok(cs)
        b = char_to_tok(max(cs, ce - 1)) + 1
        out.append((max(0, a), min(n_search, max(a + 1, b))))
    return out


def sliding_window(tok_lb: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean of per-token lookback; win[i] centered on token i."""
    n = len(tok_lb)
    out = np.full(n, np.nan)
    half = w // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, a + w)
        a = max(0, b - w)
        out[i] = np.nanmean(tok_lb[a:b])
    return out


def _color_for(lb: float) -> str:
    """Lookback in [0,1] -> hex. High=blue (context), mid=white, low=orange."""
    lb = 0.5 if np.isnan(lb) else float(np.clip(lb, 0, 1))
    if lb >= 0.5:
        t = (lb - 0.5) / 0.5           # white -> blue
        r, g, b = int(255 - t * 150), int(255 - t * 90), 255
    else:
        t = (0.5 - lb) / 0.5           # white -> orange
        r, g, b = 255, int(255 - t * 120), int(255 - t * 200)
    return f"#{r:02x}{g:02x}{b:02x}"


def render_html(tok_strs, tok_lb, hallu_tok, hallu_label, panels_png_b64,
                title, w, out_path):
    cells = []
    n = len(tok_strs)
    i = 0
    while i < n:
        s = tok_strs[i].replace("<", "&lt;").replace(">", "&gt;")
        disp = s if s.strip() != "" else s.replace(" ", "&nbsp;")
        disp = disp.replace("\n", "⏎<br>")
        bg = _color_for(tok_lb[i])
        style = (f"background:{bg};padding:1px 0;border-radius:2px;"
                 f"line-height:1.9;")
        if hallu_tok[i]:
            col = TYPE_COLOR.get(hallu_label[i], "#d62728")
            style += (f"border-bottom:3px solid {col};"
                      f"box-shadow:inset 0 0 0 1px {col}55;")
            title_attr = f' title="hallucinated: {hallu_label[i]}"'
        else:
            title_attr = ""
        cells.append(f'<span style="{style}"{title_attr}>{disp}</span>')
        i += 1
    text_html = "".join(cells)

    legend_types = "".join(
        f'<span style="border-bottom:3px solid {c};margin-right:12px;'
        f'padding-bottom:1px">{t}</span>'
        for t, c in TYPE_COLOR.items() if t != "hallucinated")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
margin:24px auto;color:#111;padding:0 16px}}
h1{{font-size:19px}} h2{{font-size:15px;margin-top:26px}}
.note{{font-size:15px;font-family:ui-monospace,Menlo,monospace}}
.bar{{display:inline-block;height:12px;vertical-align:middle;border-radius:2px}}
.cap{{color:#555;font-size:12.5px;line-height:1.5}}
</style></head><body>
<h1>{title}</h1>
<p class="cap">Lookback ratio LR<sub>t</sub> = A<sub>ctx</sub> / (A<sub>ctx</sub> + A<sub>new</sub>),
averaged over all layers &amp; heads. <b>Blue</b> = attends to the transcript
(context); <b>orange</b> = attends to its own generated text.
Coloured underline = span the LLM judge flagged as hallucinated.</p>
<p class="cap">
<span class="bar" style="width:40px;background:linear-gradient(90deg,#ff9933,#fff,#4a9bff)"></span>
&nbsp;low&nbsp;→&nbsp;high context attention &nbsp;&nbsp;|&nbsp;&nbsp; types: {legend_types}</p>
<h2>1 &amp; 3. Note with per-token lookback (span-level highlights)</h2>
<div class="note">{text_html}</div>
<h2>2. Window-level (sliding window = {w}) and span-level lookback</h2>
<img src="data:image/png;base64,{panels_png_b64}" style="max-width:100%">
<p class="cap">Left: sliding-window lookback across the note (the annotation-free
view); shaded bands are judge-flagged hallucinated spans. Right: mean lookback
per annotated span — hallucinated (by CREOLA type) vs faithful.</p>
</body></html>"""
    out_path.write_text(html)


def make_panels_png(tok_lb, win_lb, hallu_ranges_typed, span_rows, w) -> str:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw={"width_ratios": [2, 1]})
    n = len(tok_lb)
    x = np.arange(n)
    axL.plot(x, tok_lb, color="#bbb", lw=0.8, label="per-token")
    axL.plot(x, win_lb, color="#111", lw=1.8, label=f"sliding window={w}")
    axL.axhline(0.5, color="grey", ls=":", lw=0.8)
    for (a, b, typ) in hallu_ranges_typed:
        axL.axvspan(a, b, color=TYPE_COLOR.get(typ, "#d62728"), alpha=0.18)
    axL.set_xlabel("note token position")
    axL.set_ylabel("lookback ratio")
    axL.set_title("Window-level lookback (hallucinated spans shaded)")
    axL.legend(fontsize=8, loc="lower left")

    labels = [r["label"] for r in span_rows]
    vals = [r["lb"] for r in span_rows]
    cols = ["#cccccc" if l == "faithful" else TYPE_COLOR.get(l, "#d62728")
            for l in labels]
    order = np.argsort(vals)
    axR.barh(range(len(vals)), np.array(vals)[order], color=np.array(cols)[order])
    axR.set_yticks(range(len(vals)))
    axR.set_yticklabels([span_rows[i]["short"] for i in order], fontsize=7)
    axR.axvline(0.5, color="grey", ls=":", lw=0.8)
    axR.set_xlabel("mean lookback")
    axR.set_title("Span-level lookback")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--span-csv", required=True,
                    help="one sample_NNN_note_KK_span_judge.csv")
    ap.add_argument("--generations", required=True,
                    help="dir containing sample_NNN_generations.json")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", default="luq_out/worked_example")
    args = ap.parse_args()

    csv = Path(args.span_csv)
    m = re.search(r"sample_(\d+)_note_(\d+)", csv.stem)
    si, k = int(m.group(1)), int(m.group(2))
    gen = json.load(open(Path(args.generations) / f"sample_{si:03d}_generations.json"))
    transcript, note = gen["transcript"], gen["notes"][k]
    df = pd.read_csv(csv)

    print(f"Loading {args.model} …")
    model = load_model(args.model, args.device)
    full_ids, T = tokenize_prompt_and_note(model, transcript, note, args.device)

    lr = compute_lookback_ratios(model, full_ids, T)         # (L,H,n_note)
    tok_lb = np.nanmean(lr, axis=(0, 1))                     # (n_note,)
    tok_strs, n_search = per_token_strings(model, full_ids, T)
    tok_lb = tok_lb[:n_search]
    win_lb = sliding_window(tok_lb, args.window)

    # Hallucinated spans (typed) + faithful sentences, mapped to token ranges.
    hallu_rows, faith_rows = [], []
    for _, r in df.iterrows():
        lab = str(r["label"]).strip().lower()
        span = str(r.get("note_span", "") or "").strip()
        sent = str(r["sentence"])
        if lab != "faithful" and span:
            hallu_rows.append((span, lab))
        elif lab == "faithful":
            faith_rows.append((sent, "faithful"))

    hallu_spans = map_spans_to_tokens(model, full_ids, T, [s for s, _ in hallu_rows])
    faith_spans = map_spans_to_tokens(model, full_ids, T, [s for s, _ in faith_rows])

    hallu_tok = np.zeros(n_search, dtype=bool)
    hallu_label = np.array(["hallucinated"] * n_search, dtype=object)
    hallu_ranges_typed, span_rows = [], []
    for (span, lab), rng in zip(hallu_rows, hallu_spans):
        if rng is None:
            continue
        a, b = rng
        b = min(b, n_search)
        if b <= a:
            continue
        hallu_tok[a:b] = True
        hallu_label[a:b] = lab
        hallu_ranges_typed.append((a, b, lab))
        span_rows.append({"label": lab, "lb": float(np.nanmean(tok_lb[a:b])),
                          "short": f"[{lab[:4]}] {span[:22]}"})
    for (sent, lab), rng in zip(faith_rows, faith_spans):
        if rng is None:
            continue
        a, b = rng
        b = min(b, n_search)
        if b <= a:
            continue
        span_rows.append({"label": "faithful", "lb": float(np.nanmean(tok_lb[a:b])),
                          "short": f"[faith] {sent[:22]}"})

    png_b64 = make_panels_png(tok_lb, win_lb, hallu_ranges_typed, span_rows, args.window)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Lookback Lens worked example — sample_{si:03d}_note_{k:02d}"
    render_html(tok_strs[:n_search], tok_lb, hallu_tok, hallu_label,
                png_b64, title, args.window, out_dir / "worked_example.html")

    # Also dump the raw per-token series for reproducibility.
    pd.DataFrame({"pos": np.arange(n_search), "token": tok_strs[:n_search],
                  "lookback": tok_lb, "window_lookback": win_lb,
                  "hallucinated": hallu_tok,
                  "type": hallu_label}).to_csv(
        out_dir / "per_token_lookback.csv", index=False)

    print(f"\nWrote {out_dir/'worked_example.html'}")
    print(f"  hallucinated spans mapped: "
          f"{sum(r is not None for r in hallu_spans)}/{len(hallu_rows)}")
    print("  span-level lookback:")
    for r in sorted(span_rows, key=lambda z: z["lb"]):
        print(f"    {r['lb']:.3f}  {r['short']}")


if __name__ == "__main__":
    main()
