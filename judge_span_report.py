"""
judge_span_report.py
=====================
Renders an HTML report showing, for the same example notes as
fact_uncertainty_report.py, the LLM-as-judge (CREOLA taxonomy) span-level
faithfulness annotations. Each sentence is coloured by its judge label
(Faithful / Fabrication / Negation / Causality / Contextual); the specific
flagged span within a non-faithful sentence is underlined and bolded.

Data source (per note, self-contained):
    luq_out/llama_judge/<config>/<split>/spans/sample_NNN_note_KK_span_judge.csv
    columns: sentence_idx, sentence, label, note_span

Usage
-----
    python judge_span_report.py
    python judge_span_report.py --config aci --split test1 \\
        --examples 1:2 2:2 4:2 10:0 19:0 --out judge_span_report.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

BASE = Path(__file__).parent
LUQ  = BASE / "luq_out"

DEFAULT_EXAMPLES: List[Tuple[int, int]] = [(1, 2), (2, 2), (4, 2), (10, 0), (19, 0)]

# CREOLA taxonomy colours
_LABEL_COLOURS = {
    "Faithful":    "#ffffff",
    "Fabrication": "#f8b4b4",
    "Negation":    "#a8d5ff",
    "Causality":   "#ffd28a",
    "Contextual":  "#d7b8f5",
}
_LABEL_BORDER = {
    "Faithful":    "#ccc",
    "Fabrication": "#c0392b",
    "Negation":    "#2980b9",
    "Causality":   "#d68910",
    "Contextual":  "#8e44ad",
}


def _escape(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_span_judge(config: str, split: str, sample_idx: int, note_idx: int) -> pd.DataFrame:
    p = LUQ / "llama_judge" / config / split / "spans" / f"sample_{sample_idx:03d}_note_{note_idx:02d}_span_judge.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _highlight_span(sentence: str, note_span: str) -> str:
    """Bold+underline the flagged sub-span within the sentence, if present."""
    note_span = str(note_span).strip()
    if not note_span or note_span.lower() == "nan":
        return _escape(sentence)
    pos = sentence.lower().find(note_span.lower())
    if pos == -1:
        return _escape(sentence)
    before = sentence[:pos]
    match  = sentence[pos:pos + len(note_span)]
    after  = sentence[pos + len(note_span):]
    return (
        f"{_escape(before)}"
        f'<span style="text-decoration:underline;font-weight:700;">{_escape(match)}</span>'
        f"{_escape(after)}"
    )


def render_example(config: str, split: str, sample_idx: int, note_idx: int) -> str:
    df = load_span_judge(config, split, sample_idx, note_idx)

    n_total     = len(df)
    n_nonfaith  = int((df["label"] != "Faithful").sum())
    halluc_rate = n_nonfaith / n_total if n_total else 0.0

    rows_html = []
    for _, row in df.iterrows():
        label = row["label"]
        bg     = _LABEL_COLOURS.get(label, "#eee")
        border = _LABEL_BORDER.get(label, "#999")
        text   = _highlight_span(row["sentence"], row.get("note_span", ""))
        badge  = (
            f'<span style="font-size:10px;font-weight:700;color:{border};'
            f'text-transform:uppercase;letter-spacing:0.05em;">{label}</span>'
        )
        rows_html.append(f"""
<div style="margin-bottom:8px;padding:8px 12px;border-left:4px solid {border};
     background:{bg};border-radius:0 6px 6px 0;">
  <div style="margin-bottom:3px;">{badge}</div>
  <div style="font-family:Georgia,serif;font-size:14px;line-height:1.6;color:#222;">
    {text}
  </div>
</div>
""")

    body = "\n".join(rows_html)
    return f"""
<div style="margin-bottom:44px;border:1px solid #ddd;border-radius:8px;padding:20px;background:#f5f5f5;">
  <h2 style="margin-top:0;">Sample {sample_idx} · Note {note_idx}
    <span style="font-size:14px;font-weight:normal;color:#666;">
      &nbsp;·&nbsp; {config}/{split} &nbsp;·&nbsp; halluc_rate = {halluc_rate:.2%}
      &nbsp;·&nbsp; {n_nonfaith}/{n_total} sentences flagged
    </span>
  </h2>
  {body}
</div>
"""


def build_html(config: str, split: str, examples: List[Tuple[int, int]]) -> str:
    sections = []
    for sample_idx, note_idx in examples:
        try:
            sections.append(render_example(config, split, sample_idx, note_idx))
        except FileNotFoundError as exc:
            sections.append(f'<p style="color:#c0392b;">Missing data: {exc}</p>')

    legend = "".join(
        f'<span style="display:inline-block;margin-right:14px;">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{_LABEL_COLOURS[l]};'
        f'border:1px solid {_LABEL_BORDER[l]};border-radius:3px;margin-right:4px;vertical-align:middle;"></span>'
        f'{l}</span>'
        for l in ["Faithful", "Fabrication", "Negation", "Causality", "Contextual"]
    )

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LLM-as-Judge Span-Level Faithfulness Annotations</title>
  <style>
    body {{ font-family: -apple-system, "Segoe UI", sans-serif;
           max-width: 1100px; margin: 0 auto; padding: 24px;
           background: #eee; color: #222; }}
    h1 {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
  <h1>LLM-as-Judge Span-Level Faithfulness Annotations</h1>
  <p style="color:#555;margin-bottom:10px;">
    CREOLA taxonomy (Faithful / Fabrication / Negation / Causality / Contextual).
    Each sentence is coloured by its judge label; the specific flagged span within
    a non-faithful sentence is <span style="text-decoration:underline;font-weight:700;">underlined and bold</span>.
  </p>
  <div style="margin-bottom:24px;">{legend}</div>
  {body}
</body>
</html>
"""


def _parse_examples(raw: List[str]) -> List[Tuple[int, int]]:
    out = []
    for item in raw:
        s, n = item.split(":")
        out.append((int(s), int(n)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LLM-as-judge span-level annotation HTML report")
    parser.add_argument("--config", default="aci")
    parser.add_argument("--split",  default="test1")
    parser.add_argument("--examples", nargs="+", default=None,
                        help='sample:note pairs, e.g. --examples 1:2 2:2 10:0 (default: 5 curated examples)')
    parser.add_argument("--out", default="judge_span_report.html")
    args = parser.parse_args()

    examples = _parse_examples(args.examples) if args.examples else DEFAULT_EXAMPLES

    html = build_html(args.config, args.split, examples)
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"[report] {len(examples)} examples -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
