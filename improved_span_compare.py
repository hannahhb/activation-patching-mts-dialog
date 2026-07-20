"""
improved_span_compare.py
========================
Span-level ground truth by DIRECT span-on-span overlap, using the character
spans that now exist on BOTH sides (fact decomposition span + judge note_span).
This resolves the ambiguity that forced the old design to DISCARD facts sitting
in a flagged sentence but outside the flagged span.

Old design (task_span_strict): a fact in a Not-Faithful sentence that does not
overlap the note_span is EXCLUDED (ambiguous) -> ~41% of flagged-sentence facts
thrown away. New design: such a fact lies OUTSIDE the flagged error, so it is
the faithful part of a mixed sentence -> a NEGATIVE. Nothing is discarded.

Label for each verified fact (assigned to its sentence via its span):
  positive : fact's span char-overlaps the judge's flagged note_span
             (char-range overlap within the sentence; content-word overlap
             fallback if either span can't be located literally)
  negative : otherwise (Faithful sentence, OR flagged sentence but the fact's
             span lies outside the flagged span)
No exclusions. Score = fact uncertainty U(f); metric = AUROC per split+combined.
Prints old-vs-new counts so the recovered facts are visible.
"""
import json, re
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

NOT_FAITHFUL = {"Fabrication", "Negation", "Causality", "Contextual"}
DATASETS = [("aci", "test1"), ("aci", "test2"), ("aci", "test3"),
            ("virtscribe", "test1"), ("virtscribe", "test2"), ("virtscribe", "test3")]


def norm(s): return re.sub(r"\s+", " ", str(s).strip().lower())
def cwords(s): return {w.lower().strip(".,;:") for w in str(s).split() if len(w) > 3}
def shares_word(a, b): return bool(cwords(a) & cwords(b))


def _compact(s: str):
    out, idx, prev = [], [], False
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev:
                continue
            out.append(" "); idx.append(i); prev = True
        else:
            out.append(ch); idx.append(i); prev = False
    return "".join(out), idx


def find_range(recon: str, span: str) -> Optional[Tuple[int, int]]:
    span = str(span).strip()
    if not span:
        return None
    cr, im = _compact(recon); cs, _ = _compact(span); cs = cs.strip()
    if not cs:
        return None
    pos = cr.find(cs)
    if pos < 0:
        pos = cr.lower().find(cs.lower())
    if pos < 0:
        return None
    return im[pos], im[pos + len(cs) - 1] + 1


def first_span(cell) -> str:
    try:
        a = json.loads(cell)
        return str(a[0]) if isinstance(a, list) and a else ""
    except Exception:
        return str(cell or "")


def assign_sentence(span_text, fact_text, j_df):
    ns = norm(span_text)
    if ns:
        for _, jr in j_df.iterrows():
            if ns in norm(jr["sentence"]):
                return jr
    q = cwords(span_text) or cwords(fact_text)
    best, bo = None, 0
    for _, jr in j_df.iterrows():
        o = len(q & cwords(jr["sentence"]))
        if o > bo:
            bo, best = o, jr
    return best


def label_fact(fr, jr):
    """Return ('pos'|'neg', old_verdict) where old_verdict in
    {'pos','neg','excl'} is what task_span_strict would have done."""
    label = str(jr["label"]).strip()
    note_span = str(jr.get("note_span", "") or "")
    fact_span = first_span(fr.get("spans", ""))
    sent = str(jr["sentence"])

    # ---- OLD verdict (for comparison) ----
    if label == "Faithful":
        old = "neg"
    elif label in NOT_FAITHFUL and note_span and shares_word(fr["fact"], note_span):
        old = "pos"
    else:
        old = "excl"

    # ---- NEW verdict: span-on-span, no exclusions ----
    if label == "Faithful" or not note_span:
        return "neg", old
    # both spans located in the sentence -> precise char overlap
    fr_rng = find_range(sent, fact_span) if fact_span else None
    js_rng = find_range(sent, note_span)
    if fr_rng and js_rng:
        a1, b1 = fr_rng; a2, b2 = js_rng
        return ("pos" if (a1 < b2 and a2 < b1) else "neg"), old
    # fallback: content-word overlap between fact and flagged span
    return ("pos" if shares_word(fr["fact"], note_span) else "neg"), old


def main():
    pooled = []
    old_counts = {"pos": 0, "neg": 0, "excl": 0}
    recovered = 0  # old-excluded now labeled
    for cfg, sp in DATASETS:
        fdir = Path(f"luq_out/llama_atomic_span/{cfg}/{sp}/facts")
        jdir = Path(f"luq_out/llama_judge/{cfg}/{sp}/spans")
        for fp in sorted(fdir.glob("sample_*_note_*_facts.csv")):
            m = re.search(r"sample_(\d+)_note_(\d+)", fp.stem)
            si, k = int(m.group(1)), int(m.group(2))
            jp = jdir / f"sample_{si:03d}_note_{k:02d}_span_judge.csv"
            if not jp.exists():
                continue
            fdf = pd.read_csv(fp); jdf = pd.read_csv(jp)
            if fdf.empty or jdf.empty or "sentence" not in jdf.columns:
                continue
            for _, fr in fdf.iterrows():
                if not bool(fr.get("span_verified", False)):
                    continue
                jr = assign_sentence(first_span(fr.get("spans", "")), fr.get("fact", ""), jdf)
                if jr is None:
                    continue
                new, old = label_fact(fr, jr)
                old_counts[old] += 1
                if old == "excl":
                    recovered += 1
                pooled.append((float(fr["uncertainty"]),
                               1 if new == "pos" else 0, f"{cfg}/{sp}"))
    pdf = pd.DataFrame(pooled, columns=["uncertainty", "y", "split"])

    print("Old design counts:  pos=%d  neg=%d  EXCLUDED=%d  (eval_n=%d)"
          % (old_counts["pos"], old_counts["neg"], old_counts["excl"],
             old_counts["pos"] + old_counts["neg"]))
    print("New design: nothing excluded; %d previously-discarded facts now "
          "labelled (as span-precise negatives/positives)." % recovered)
    print(f"New eval_n = {len(pdf)}   positives = {int(pdf.y.sum())}   "
          f"positive rate = {pdf.y.mean():.1%}\n")

    print("=== SPAN-LEVEL AUROC (span-on-span overlap, no exclusions) ===")
    print(f"{'Split':<18}{'n':>7}{'Positive':>10}{'AUROC':>9}")
    for cfg, sp in DATASETS:
        s = f"{cfg}/{sp}"
        d = pdf[pdf.split == s]
        au = roc_auc_score(d.y, d.uncertainty) if d.y.nunique() == 2 else float("nan")
        print(f"{s:<18}{len(d):>7}{int(d.y.sum()):>10}{au:>9.3f}")
    au = roc_auc_score(pdf.y, pdf.uncertainty) if pdf.y.nunique() == 2 else float("nan")
    print(f"{'Combined':<18}{len(pdf):>7}{int(pdf.y.sum()):>10}{au:>9.3f}")


if __name__ == "__main__":
    main()
