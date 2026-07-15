"""
fact_uncertainty_report.py
===========================
Renders an HTML report showing, for a handful of example notes, a two-column
layout: the source transcript on the left, and the note on the right rendered
as span-level sentences — each sentence-span carries its LUQ uncertainty score
on top (green = certain, red = uncertain), with the atomic facts that were
decomposed from it (span-level matching, via match_sent_idx) nested inside.

Span-in-sentence highlighting uses the span_start/span_end columns computed
by fact_sentence_match.py's _locate_span() (entity/name/number/lemma-driven
phrase location) directly from facts_matched.csv — no re-derivation here.

Data sources (per note):
    luq_out/llama_atomic/facts_matched/<config>/<split>/sample_NNN_note_KK_facts_matched.csv
    columns: fact_idx, section, fact, uncertainty, match_sent_idx, sentence_text,
             span_start, span_end, span_text, ...

    luq_out/llama/generations/<config>/<split>/sentences/sample_NNN_note_KK_sentences.csv
    columns: sentence_idx, sentence, uncertainty   (defines sentence order + span-level score)

    luq_out/llama/generations/<config>/<split>/sample_NNN_generations.json
    columns: transcript, gold_note, notes, ...  (transcript used for left panel)

Usage
-----
    python fact_uncertainty_report.py
    python fact_uncertainty_report.py --config aci --split test1 \\
        --examples 1:2 2:2 4:2 10:0 19:0 --out fact_uncertainty_report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

BASE = Path(__file__).parent
LUQ  = BASE / "luq_out"

DEFAULT_EXAMPLES: List[Tuple[int, int]] = [(1, 2), (2, 2), (4, 2), (10, 0), (19, 0)]


# ─────────────────────────────────────────────────────────────────────────────
# Colour scale (green = certain, red = uncertain) — consistent with se_highlight.py
# ─────────────────────────────────────────────────────────────────────────────

def _uncertainty_colour(u: float) -> str:
    u = max(0.0, min(1.0, u))
    if u <= 0.5:
        t = u / 0.5
        r = int(20 + t * (255 - 20))
        g = int(120 + t * (200 - 120))
        b = int(20 + t * (60 - 20))
    else:
        t = (u - 0.5) / 0.5
        r = int(255)
        g = int(200 - t * (200 - 20))
        b = int(60 - t * (60 - 20))
    return f"rgb({r},{g},{b})"


def _escape(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ─────────────────────────────────────────────────────────────────────────────
# Fact -> substring-in-sentence matching
#
# facts_matched.csv now carries span_start/span_end (char offsets into
# sentence_text) computed authoritatively by fact_sentence_match.py's
# _locate_span(), using entity/name/number/lemma signals from the matching
# pipeline itself. We only resolve overlaps here when multiple facts' spans
# collide (e.g. a broad fact and a narrower fact covering the same phrase) —
# no re-derivation of the span itself.
# ─────────────────────────────────────────────────────────────────────────────

FactWithSpan = Tuple[str, float, Optional[float], Optional[float]]  # fact, uncertainty, span_start, span_end


def _resolve_spans(
    facts: List[FactWithSpan],
) -> Tuple[List[Tuple[int, int, str, float]], List[Tuple[str, float]]]:
    """
    Greedily keep non-overlapping spans from the precomputed span_start/span_end
    columns, preferring longer spans and then first-seen (fact_idx order) on ties.
    Returns (placed_spans, unplaced_facts) where placed_spans is
    [(char_start, char_end, fact_text, uncertainty), ...] sorted by position,
    and unplaced_facts lists facts with no span (or that lost an overlap).
    """
    candidates = []
    unplaced: List[Tuple[str, float]] = []
    for order, (fact, uncertainty, span_start, span_end) in enumerate(facts):
        if span_start is None or span_end is None or pd.isna(span_start) or pd.isna(span_end):
            unplaced.append((fact, uncertainty))
            continue
        candidates.append((int(span_start), int(span_end), fact, uncertainty, order))

    # Greedy: longest span first, then first-seen
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[4]))
    placed: List[Tuple[int, int, str, float]] = []
    for char_start, char_end, fact, uncertainty, _ in candidates:
        if any(char_start < pe and char_end > ps for ps, pe, _, _ in placed):
            unplaced.append((fact, uncertainty))
            continue
        placed.append((char_start, char_end, fact, uncertainty))

    placed.sort(key=lambda p: p[0])
    return placed, unplaced


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_facts_matched(config: str, split: str, sample_idx: int, note_idx: int) -> pd.DataFrame:
    p = LUQ / "llama_atomic" / "facts_matched" / config / split / f"sample_{sample_idx:03d}_note_{note_idx:02d}_facts_matched.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def load_transcript(config: str, split: str, sample_idx: int) -> str:
    p = LUQ / "llama" / "generations" / config / split / f"sample_{sample_idx:03d}_generations.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())["transcript"]


def load_sentences(config: str, split: str, sample_idx: int, note_idx: int) -> pd.DataFrame:
    """Master sentence order + per-sentence (span-level) uncertainty score."""
    p = LUQ / "llama" / "generations" / config / split / "sentences" / f"sample_{sample_idx:03d}_note_{note_idx:02d}_sentences.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _fact_row(fact: str, uncertainty: float) -> str:
    """One unplaced atomic fact (no distinct substring found) — small fallback list."""
    colour = _uncertainty_colour(uncertainty)
    return (
        f'<div style="display:flex;align-items:baseline;gap:6px;margin:3px 0;">'
        f'<span style="flex-shrink:0;font-size:10px;font-weight:700;padding:1px 6px;'
        f'border-radius:8px;background:{colour};color:#111;">{uncertainty:.2f}</span>'
        f'<span style="font-size:13px;color:#333;">{_escape(fact)}</span>'
        f'</div>'
    )


def _render_sentence_inline(sentence: str, placed: List[Tuple[int, int, str, float]]) -> str:
    """
    Render sentence text with each placed fact's substring wrapped in a coloured
    <mark>, with its uncertainty score sitting directly on top of the highlighted
    phrase via a <ruby>/<rt> interlinear annotation (not a hover tooltip).
    """
    parts: List[str] = []
    cursor = 0
    for char_start, char_end, fact, uncertainty in placed:
        parts.append(_escape(sentence[cursor:char_start]))
        colour = _uncertainty_colour(uncertainty)
        span_text = _escape(sentence[char_start:char_end])
        parts.append(
            f'<ruby title="{_escape(fact)}">'
            f'<mark style="background:{colour};padding:1px 3px;border-radius:3px;">'
            f'{span_text}</mark>'
            f'<rt style="font-size:11px;font-weight:700;color:#111;background:{colour};'
            f'padding:0 5px;border-radius:6px;">{uncertainty:.2f}</rt>'
            f'</ruby>'
        )
        cursor = char_end
    parts.append(_escape(sentence[cursor:]))
    return "".join(parts)


def _sentence_span(sentence_idx: int, sentence: str, sent_uncertainty: float,
                   facts: List[FactWithSpan]) -> str:
    """
    One note sentence rendered as a span: uncertainty score on top (header bar),
    then the sentence text with each matched atomic fact's specific phrase
    inline-highlighted and coloured by that fact's own uncertainty score.
    Facts with no locatable substring fall back to a small list underneath.
    """
    colour = _uncertainty_colour(sent_uncertainty)
    placed, unplaced = _resolve_spans(facts)
    inline_html = _render_sentence_inline(sentence, placed)
    fallback_html = (
        "".join(_fact_row(f, u) for f, u in sorted(unplaced, key=lambda x: -x[1]))
        if unplaced else ""
    )
    return f"""
<div style="margin-bottom:10px;border-radius:6px;overflow:hidden;
     border:1px solid rgba(0,0,0,0.12);">
  <div style="background:{colour};padding:3px 10px;font-size:11px;font-weight:700;color:#111;">
    span uncertainty = {sent_uncertainty:.3f}
  </div>
  <div style="background:#fff;padding:8px 10px;">
    <div style="font-family:Georgia,serif;font-size:14px;line-height:2.6;color:#222;padding-top:6px;">
      {inline_html}
    </div>
    {f'<div style="padding-left:8px;margin-top:6px;border-left:2px solid #eee;">{fallback_html}</div>' if unplaced else ''}
  </div>
</div>
"""


def render_example(config: str, split: str, sample_idx: int, note_idx: int) -> str:
    facts_df = load_facts_matched(config, split, sample_idx, note_idx)
    sent_df  = load_sentences(config, split, sample_idx, note_idx)
    transcript = load_transcript(config, split, sample_idx)

    mean_u = facts_df["uncertainty"].mean()

    # Group facts by their matched sentence index (span-level matching),
    # carrying each fact's precomputed span_start/span_end along with it.
    facts_by_sent: Dict[int, List[FactWithSpan]] = {}
    for _, row in facts_df.iterrows():
        sidx = int(row["match_sent_idx"])
        facts_by_sent.setdefault(sidx, []).append(
            (row["fact"], float(row["uncertainty"]), row.get("span_start"), row.get("span_end"))
        )

    # Right panel: every note sentence in order, with its matched facts nested inside
    spans_html = "".join(
        _sentence_span(int(row["sentence_idx"]), row["sentence"], float(row["uncertainty"]),
                       facts_by_sent.get(int(row["sentence_idx"]), []))
        for _, row in sent_df.iterrows()
    )

    return f"""
<div style="margin-bottom:44px;border:1px solid #ddd;border-radius:8px;padding:20px;background:#f5f5f5;">
  <h2 style="margin-top:0;">Sample {sample_idx} · Note {note_idx}
    <span style="font-size:14px;font-weight:normal;color:#666;">
      &nbsp;·&nbsp; {config}/{split} &nbsp;·&nbsp; mean fact uncertainty = {mean_u:.4f}
      &nbsp;·&nbsp; {len(facts_df)} facts &nbsp;·&nbsp; {len(sent_df)} sentence-spans
    </span>
  </h2>
  <div style="display:flex;gap:20px;align-items:flex-start;">
    <div style="flex:1;min-width:0;position:sticky;top:12px;">
      <h3 style="margin-top:0;font-size:13px;text-transform:uppercase;letter-spacing:0.04em;color:#666;">
        Transcript
      </h3>
      <div style="background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:14px;
           font-family:monospace;font-size:12.5px;line-height:1.6;white-space:pre-wrap;
           max-height:640px;overflow-y:auto;color:#333;">{_escape(transcript)}</div>
    </div>
    <div style="flex:1;min-width:0;">
      <h3 style="margin-top:0;font-size:13px;text-transform:uppercase;letter-spacing:0.04em;color:#666;">
        Note (span-level, facts mapped per sentence)
      </h3>
      <div style="max-height:640px;overflow-y:auto;padding-right:4px;">
        {spans_html}
      </div>
    </div>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Full HTML
# ─────────────────────────────────────────────────────────────────────────────

def build_html(config: str, split: str, examples: List[Tuple[int, int]]) -> str:
    sections = []
    for sample_idx, note_idx in examples:
        try:
            sections.append(render_example(config, split, sample_idx, note_idx))
        except FileNotFoundError as exc:
            sections.append(f'<p style="color:#c0392b;">Missing data: {exc}</p>')

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Atomic Fact Uncertainty — Transcript vs. Decomposed Facts</title>
  <style>
    body {{ font-family: -apple-system, "Segoe UI", sans-serif;
           max-width: 1400px; margin: 0 auto; padding: 24px;
           background: #eee; color: #222; }}
    h1 {{ margin-bottom: 6px; }}
    ruby {{ ruby-position: over; ruby-align: center; }}
    rt {{ line-height: 1; }}
  </style>
</head>
<body>
  <h1>Atomic Fact Uncertainty — Transcript vs. Span-Level Note</h1>
  <p style="color:#555;margin-bottom:24px;">
    Left: source transcript. Right: the note rendered sentence-by-sentence.
    Each sentence-span shows its mean uncertainty on top (green = certain,
    red = uncertain); within the sentence, the specific phrase each atomic
    fact decomposed from is highlighted and coloured by that fact's own
    uncertainty score, with the score displayed directly above the highlight
    (hover a highlight for the full fact text). Facts with no distinct
    matching phrase are listed underneath instead.
  </p>
  {body}
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_examples(raw: List[str]) -> List[Tuple[int, int]]:
    out = []
    for item in raw:
        s, n = item.split(":")
        out.append((int(s), int(n)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render fact-to-sentence uncertainty HTML report")
    parser.add_argument("--config", default="aci")
    parser.add_argument("--split",  default="test1")
    parser.add_argument("--examples", nargs="+", default=None,
                        help='sample:note pairs, e.g. --examples 1:2 2:2 10:0 (default: 5 curated examples)')
    parser.add_argument("--out", default="fact_uncertainty_report.html")
    args = parser.parse_args()

    examples = _parse_examples(args.examples) if args.examples else DEFAULT_EXAMPLES

    html = build_html(args.config, args.split, examples)
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"[report] {len(examples)} examples -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
