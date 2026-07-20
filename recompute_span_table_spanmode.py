"""
recompute_span_table_spanmode.py
================================
Recompute the span-level AUROC table (paper Table 3 / sec:span-level) from the
NEW span-mode atomic decomposition (luq_out/llama_atomic_span/...), where each
fact carries its own source span. Fact->sentence attribution is done directly
from the span (the sentence whose text contains the span, with a content-word
fallback), replacing the old fact_sentence_match matcher.

Ground-truth logic is IDENTICAL to analyze_uncertainty_vs_judge.task_span_strict:
  gate     : span_verified == True   (analog of the old clean_match)
  positive : fact's sentence is Not-Faithful AND fact text shares >=1 content
             word with the judge's flagged note_span
  negative : fact's sentence is Faithful
  excluded : Not-Faithful sentence without a note_span, or no content-word
             overlap with it (ambiguous)
Score = fact uncertainty U(f); metric = AUROC per split + combined.
Prints coverage so the partial-run status is visible.
"""
import json, re
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

NOT_FAITHFUL = {"Fabrication", "Negation", "Causality", "Contextual"}
DATASETS = [("aci", "test1"), ("aci", "test2"), ("aci", "test3"),
            ("virtscribe", "test1"), ("virtscribe", "test2"), ("virtscribe", "test3")]
FACTS_BASE = Path("luq_out/llama_atomic_span")
JUDGE_BASE = Path("luq_out/llama_judge")


def norm(s): return re.sub(r"\s+", " ", str(s).strip().lower())
def cwords(s): return {w.lower().strip(".,;:") for w in str(s).split() if len(w) > 3}
def shares_word(a, b): return bool(cwords(a) & cwords(b))


def first_span(cell):
    try:
        arr = json.loads(cell)
        return str(arr[0]) if isinstance(arr, list) and arr else ""
    except Exception:
        return str(cell or "")


def assign_sentence(span_text, fact_text, j_df):
    """Sentence whose text contains the span (tolerant); else best content-word
    overlap with span (fallback to fact). Returns (row, method) or (None, 'none')."""
    ns = norm(span_text)
    if ns:
        for _, jr in j_df.iterrows():
            if ns in norm(jr["sentence"]):
                return jr, "span-substring"
    q = cwords(span_text) or cwords(fact_text)
    best, best_ov = None, 0
    for _, jr in j_df.iterrows():
        ov = len(q & cwords(jr["sentence"]))
        if ov > best_ov:
            best, best_ov = jr, ov
    return (best, "word-overlap") if best is not None else (None, "none")


def run():
    all_rows, cov = [], []
    for cfg, split in DATASETS:
        fdir = FACTS_BASE / cfg / split / "facts"
        jdir = JUDGE_BASE / cfg / split / "spans"
        fact_files = sorted(fdir.glob("sample_*_note_*_facts.csv"))
        n_notes = n_facts = n_verified = n_assigned = 0
        rows = []
        for fp in fact_files:
            m = re.search(r"sample_(\d+)_note_(\d+)", fp.stem)
            si, k = int(m.group(1)), int(m.group(2))
            jp = jdir / f"sample_{si:03d}_note_{k:02d}_span_judge.csv"
            if not jp.exists():
                continue
            fdf = pd.read_csv(fp); jdf = pd.read_csv(jp)
            if fdf.empty or jdf.empty or "sentence" not in jdf.columns:
                continue
            n_notes += 1
            for _, fr in fdf.iterrows():
                n_facts += 1
                if not bool(fr.get("span_verified", False)):
                    continue
                n_verified += 1
                span = first_span(fr.get("spans", ""))
                jr, _meth = assign_sentence(span, fr.get("fact", ""), jdf)
                if jr is None:
                    continue
                n_assigned += 1
                label = str(jr["label"]).strip()
                note_span = str(jr.get("note_span", "") or "")
                if label == "Faithful":
                    y = 0
                elif label in NOT_FAITHFUL and note_span and shares_word(fr["fact"], note_span):
                    y = 1
                else:
                    continue
                rows.append({"uncertainty": float(fr["uncertainty"]),
                             "y": y, "split": f"{cfg}/{split}"})
        df = pd.DataFrame(rows)
        cov.append({"split": f"{cfg}/{split}", "note_files": len(fact_files),
                    "facts": n_facts, "verified": n_verified,
                    "assigned": n_assigned, "eval_n": len(df),
                    "positive": int(df["y"].sum()) if len(df) else 0})
        if len(df) and df["y"].nunique() == 2:
            auroc = roc_auc_score(df["y"], df["uncertainty"])
        else:
            auroc = float("nan")
        all_rows.append((f"{cfg}/{split}", len(df), int(df["y"].sum()) if len(df) else 0, auroc))
        all_rows[-1] = all_rows[-1]
    comb = pd.concat([pd.DataFrame() for _ in [0]])  # placeholder
    # Combined
    big = []
    for cfg, split in DATASETS:
        pass
    return all_rows, cov


def main():
    all_rows, cov = run()
    print("\n=== COVERAGE (span-mode run on disk) ===")
    covdf = pd.DataFrame(cov)
    print(covdf.to_string(index=False))
    # Rebuild combined from per-split eval rows for a correct pooled AUROC.
    # (Re-collect to pool.)
    pooled = []
    for cfg, split in DATASETS:
        fdir = FACTS_BASE / cfg / split / "facts"
        jdir = JUDGE_BASE / cfg / split / "spans"
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
                span = first_span(fr.get("spans", ""))
                jr, _ = assign_sentence(span, fr.get("fact", ""), jdf)
                if jr is None:
                    continue
                label = str(jr["label"]).strip()
                note_span = str(jr.get("note_span", "") or "")
                if label == "Faithful":
                    y = 0
                elif label in NOT_FAITHFUL and note_span and shares_word(fr["fact"], note_span):
                    y = 1
                else:
                    continue
                pooled.append((float(fr["uncertainty"]), y, f"{cfg}/{split}"))
    pdf = pd.DataFrame(pooled, columns=["uncertainty", "y", "split"])
    print("\n=== SPAN-LEVEL AUROC (new span-mode provenance) ===")
    print(f"{'Split':<18}{'n':>7}{'Positive':>10}{'AUROC':>9}")
    for cfg, split in DATASETS:
        s = f"{cfg}/{split}"
        d = pdf[pdf.split == s]
        au = roc_auc_score(d.y, d.uncertainty) if d.y.nunique() == 2 else float("nan")
        print(f"{s:<18}{len(d):>7}{int(d.y.sum()):>10}{au:>9.3f}")
    au = roc_auc_score(pdf.y, pdf.uncertainty) if pdf.y.nunique() == 2 else float("nan")
    print(f"{'Combined':<18}{len(pdf):>7}{int(pdf.y.sum()):>10}{au:>9.3f}")
    print(f"\npositive rate: {pdf.y.mean():.1%}")


if __name__ == "__main__":
    main()
