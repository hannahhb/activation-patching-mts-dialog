
from ast import List, Tuple
from typing import Dict, Optional
import warnings
from zipfile import Path
import numpy as np


def build_html_report(
    scatter_png: Path,
    tokens: List[str],
    ecs: np.ndarray,
    pks: np.ndarray,
    generated_note: str,
    model_name: str,
    sample_idx: int,
    out_path: Path,
) -> None:
    """
    Self-contained HTML report containing:
      1. ECS/PKS scatter (embedded as base64 PNG).
      2. Full generated note with hallucination-risk tokens highlighted.

    Risk tokens = those below the median on BOTH ECS and PKS (Low-Low quadrant).
    Token spacing is reconstructed from SentencePiece / BPE leading-space markers.
    """
    import base64

    with open(scatter_png, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    em        = float(np.median(ecs))
    pm        = float(np.median(pks))
    risk_mask = (ecs < em) & (pks < pm)

    html_parts: List[str] = []
    for tok, is_risky in zip(tokens, risk_mask):
        space = " " if (tok.startswith("▁") or tok.startswith("Ġ")) else ""
        word  = tok[1:] if space else tok
        word_esc = (word
                    .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace("\n", "<br>\n"))
        if is_risky:
            html_parts.append(f'{space}<mark class="risk">{word_esc}</mark>')
        else:
            html_parts.append(f"{space}{word_esc}")
    note_html = "".join(html_parts)

    n         = len(ecs)
    hi_ecs    = ecs >= em
    hi_pks    = pks >= pm
    pct_extr  = 100 * float(np.mean( hi_ecs & ~hi_pks))
    pct_param = 100 * float(np.mean(~hi_ecs &  hi_pks))
    pct_synth = 100 * float(np.mean( hi_ecs &  hi_pks))
    pct_risk  = 100 * float(np.mean(~hi_ecs & ~hi_pks))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exp 2b — Hallucination Risk Report</title>
  <style>
    body      {{ font-family: Georgia, serif; max-width: 960px;
                margin: 40px auto; padding: 0 24px; color: #222; }}
    h2        {{ border-bottom: 2px solid #eee; padding-bottom: 8px; color: #333; }}
    h3        {{ color: #555; margin-top: 32px; }}
    img       {{ max-width: 100%; border: 1px solid #ddd;
                border-radius: 6px; margin: 12px 0; }}
    .meta     {{ font-size: 13px; color: #777; margin-bottom: 24px; }}
    .legend   {{ display: flex; gap: 20px; flex-wrap: wrap;
                margin: 12px 0 20px; font-size: 13px; }}
    .leg-item {{ display: flex; align-items: center; gap: 8px; }}
    .swatch   {{ width: 18px; height: 18px; border-radius: 3px;
                border: 1px solid rgba(0,0,0,.15); flex-shrink: 0; }}
    .note-box {{ background: #fafafa; border: 1px solid #ddd;
                border-radius: 6px; padding: 22px 26px;
                line-height: 2.0; font-size: 14px; white-space: pre-wrap; }}
    mark.risk {{ background: #FFCDD2; color: #B71C1C;
                border-radius: 3px; padding: 1px 3px; font-style: normal; }}
    table     {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td    {{ border: 1px solid #ddd; padding: 7px 12px; text-align: left; }}
    th        {{ background: #f5f5f5; }}
    .risk-row {{ background: #FFEBEE; font-weight: bold; }}
  </style>
</head>
<body>
<h2>Experiment 2b — ECS/PKS Hallucination Risk Report</h2>
<p class="meta">
  Model: <strong>{model_name}</strong> &nbsp;|&nbsp;
  ACI-Bench sample: <strong>{sample_idx}</strong> &nbsp;|&nbsp;
  Note tokens: <strong>{n}</strong><br>
  ECS threshold (median): <strong>{em:.4f}</strong> &nbsp;|&nbsp;
  PKS threshold (median): <strong>{pm:.4f}</strong>
</p>

<h3>ECS vs PKS Scatter — Model-Generated Note</h3>
<img src="data:image/png;base64,{img_b64}" alt="ECS/PKS scatter">

<h3>Quadrant Distribution</h3>
<table>
  <tr><th>Quadrant</th><th>Condition</th><th>% of tokens</th></tr>
  <tr><td>Extractive</td>
      <td>High ECS, Low PKS — copied from transcript</td>
      <td>{pct_extr:.1f}%</td></tr>
  <tr><td>Parametric</td>
      <td>Low ECS, High PKS — drawn from medical knowledge</td>
      <td>{pct_param:.1f}%</td></tr>
  <tr><td>Synthesized</td>
      <td>High ECS, High PKS — grounded reasoning</td>
      <td>{pct_synth:.1f}%</td></tr>
  <tr class="risk-row"><td>Hallucination Risk</td>
      <td>Low ECS, Low PKS — grounded in neither source</td>
      <td>{pct_risk:.1f}%</td></tr>
</table>

<h3>Generated Note — Highlighted Tokens</h3>
<div class="legend">
  <div class="leg-item">
    <div class="swatch" style="background:#FFCDD2;"></div>
    <span>Hallucination risk (Low ECS + Low PKS)</span>
  </div>
  <div class="leg-item">
    <div class="swatch" style="background:#fff;"></div>
    <span>Extractive, parametric, or synthesized</span>
  </div>
</div>
<div class="note-box">{note_html}</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


def plot_distribution_comparison(
    ecs_gold: np.ndarray,
    pks_gold: np.ndarray,
    ecs_gen: np.ndarray,
    pks_gen: np.ndarray,
    out_path: Path,
    model_name: str,
) -> None:
    """
    Compare the *distributions* of ECS and PKS between the gold and generated
    notes via KDE.

    Per-token positional subtraction is invalid when the two notes have
    different structures (both may be clinically correct but worded differently).
    Distributional comparison requires no alignment and is always valid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, vals_g, vals_m, label in [
        (axes[0], ecs_gold, ecs_gen, "ECS"),
        (axes[1], pks_gold, pks_gen, "PKS"),
    ]:
        sns.kdeplot(vals_g, ax=ax, label="Gold reference",
                    color="#4CAF50", fill=True, alpha=0.25, linewidth=1.8)
        sns.kdeplot(vals_m, ax=ax, label="Model generated",
                    color="#F44336", fill=True, alpha=0.25, linewidth=1.8)
        ax.axvline(np.median(vals_g), color="#4CAF50", ls="--", lw=1.2, alpha=0.8)
        ax.axvline(np.median(vals_m), color="#F44336", ls="--", lw=1.2, alpha=0.8)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{label} distribution: Gold vs Generated",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle(f"Exp 2b — Distributional Comparison  ({model_name})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gold_coverage_labels(
    generated_note: str,
    gold_note: str,
    note_toks: List[str],
    ngram: int = 3,
) -> np.ndarray:
    """
    Token-level pseudo-labels derived from n-gram coverage of the gold note.

    For each token position, reconstruct overlapping n-grams of decoded text
    centred on that token.  If none of the n-grams appear in the (normalised)
    gold note, the token is flagged as potentially hallucinated (label = 1).

    Returns
    -------
    labels : (note_len,) float in {0.0, 1.0}
    """
    import re as _re

    def _normalise(text: str) -> str:
        return _re.sub(r"\s+", " ", text.lower().strip())

    gold_norm = _normalise(gold_note)
    # Pre-tokenise gold into word n-grams for fast lookup
    gold_words = gold_norm.split()
    gold_ngrams: set = set()
    for i in range(len(gold_words) - ngram + 1):
        gold_ngrams.add(" ".join(gold_words[i: i + ngram]))

    # Decode note tokens to plain words
    cleaned = [t.replace("▁", " ").replace("Ġ", " ").strip() for t in note_toks]
    n = len(cleaned)
    labels = np.ones(n, dtype=np.float64)   # default: not covered

    for i in range(n):
        # Build n-gram window centred roughly on position i
        start = max(0, i - ngram + 1)
        end   = min(n, i + ngram)
        window = " ".join(cleaned[start:end])
        window_norm = _normalise(window)
        window_words = window_norm.split()
        for j in range(len(window_words) - ngram + 1):
            cand = " ".join(window_words[j: j + ngram])
            if cand in gold_ngrams:
                labels[i] = 0.0   # covered by gold → not flagged
                break

    return labels


def _nli_sentence_labels(
    transcript: str,
    generated_note: str,
    note_toks: List[str],
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> Optional[np.ndarray]:
    """
    Sentence-level NLI pseudo-labels mapped to token positions.

    For each sentence in the generated note, queries an NLI cross-encoder to
    compute P(entailment | transcript, sentence).  Low entailment probability
    → the sentence makes a claim unsupported by the transcript → tokens in that
    sentence get label ≈ 1.

    Requires:  pip install sentence-transformers

    Returns
    -------
    labels : (note_len,) float in [0, 1] — 1 = likely hallucinated.
             Returns None if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        warnings.warn(
            "[Exp 3] sentence-transformers not installed — skipping NLI validation. "
            "Install with:  pip install sentence-transformers"
        )
        return None

    import re as _re

    # Split generated note into sentences (simple split on . ? !)
    sentence_spans: List[Tuple[int, int]] = []   # (char_start, char_end)
    sentences: List[str] = []
    for m in _re.finditer(r"[^.!?\n]+[.!?\n]?", generated_note):
        s = m.group(0).strip()
        if len(s) > 10:
            sentence_spans.append((m.start(), m.end()))
            sentences.append(s)

    if not sentences:
        return None

    print(f"  [NLI] Loading {nli_model_name} …")
    nli = CrossEncoder(nli_model_name)

    # NLI: premise = transcript, hypothesis = each generated sentence
    pairs   = [(transcript[:2000], sent) for sent in sentences]   # truncate transcript
    logits  = nli.predict(pairs, apply_softmax=True)               # (S, 3): contra/neut/entail
    entail_prob = logits[:, 2]                                      # entailment column

    # Map sentence-level score to token-level via character offsets
    n = len(note_toks)
    token_labels = np.zeros(n, dtype=np.float64)

    # Build cumulative char position for each token
    cursor = 0
    tok_char_starts = []
    for tok in note_toks:
        piece = tok.replace("▁", " ").replace("Ġ", " ")
        tok_char_starts.append(cursor)
        cursor += len(piece)

    for (cs, ce), ep in zip(sentence_spans, entail_prob):
        halluc_score = float(1.0 - ep)
        for i, tc in enumerate(tok_char_starts):
            if cs <= tc < ce:
                token_labels[i] = halluc_score

    return token_labels


def _build_exp3_html(
    scatter_png: Path,
    note_toks: List[str],
    halluc_prob: np.ndarray,
    ecs: np.ndarray,
    pks: np.ndarray,
    gold_labels: Optional[np.ndarray],
    nli_labels: Optional[np.ndarray],
    calib_summary: Dict,
    generated_note: str,
    model_name: str,
    sample_idx: int,
    threshold: float,
    out_path: Path,
) -> None:
    """
    Self-contained HTML report for Experiment 3.

    Sections
    --------
    1. Calibration summary table (selected layers, J-stat, AUROC, threshold).
    2. ECS/PKS scatter of the generated note, points coloured by halluc_prob.
    3. Validation metrics (gold coverage, NLI) if available.
    4. Generated note with gradient highlighting (white → red ∝ halluc_prob).
    """
    import base64

    with open(scatter_png, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    # ── Token HTML ────────────────────────────────────────────────────────────
    def _prob_to_style(p: float) -> str:
        if p < 0.15:
            return ""
        # Interpolate white → #FFCDD2 (light red) → #F44336 (red)
        g = int(255 - (255 - 67)  * p)
        b = int(255 - (255 - 54)  * p)
        r = 255
        return f"background:rgb({r},{g},{b});border-radius:3px;padding:1px 2px;"

    html_toks: List[str] = []
    for tok, p in zip(note_toks, halluc_prob):
        space = " " if (tok.startswith("▁") or tok.startswith("Ġ")) else ""
        word  = tok[1:] if space else tok
        word_esc = (word
                    .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace("\n", "<br>"))
        style = _prob_to_style(float(p))
        title = f'title="p={p:.3f}"'
        if style:
            html_toks.append(f'{space}<span style="{style}" {title}>{word_esc}</span>')
        else:
            html_toks.append(f"{space}{word_esc}")
    note_html = "".join(html_toks)

    # ── Calibration table rows ────────────────────────────────────────────────
    cal_rows = ""
    for metric, layers_info in calib_summary.items():
        if not layers_info:
            continue
        for layer, info in sorted(layers_info.items(), key=lambda x: -x[1]["j_stat"]):
            dir_sym = "↓ low" if info["direction"] == -1 else "↑ high"
            cal_rows += (
                f"<tr><td>{metric}</td><td>{layer}</td>"
                f"<td>{info['auroc']:.3f}</td><td>{info['j_stat']:.3f}</td>"
                f"<td>{info['threshold']:.4f}</td><td>{dir_sym}</td></tr>\n"
            )

    # ── Validation rows ────────────────────────────────────────────────────────
    val_rows = ""
    flagged  = halluc_prob >= threshold
    if gold_labels is not None:
        from sklearn.metrics import roc_auc_score as _auc
        try:
            auc_gold = _auc(gold_labels, halluc_prob)
            prec = float(np.mean(gold_labels[flagged])) if flagged.any() else float("nan")
            recall = float(np.sum(flagged & (gold_labels > 0.5)) / (gold_labels > 0.5).sum()) if (gold_labels > 0.5).any() else float("nan")
            val_rows += (
                f"<tr><td>Gold n-gram coverage</td>"
                f"<td>{auc_gold:.3f}</td><td>{prec:.3f}</td><td>{recall:.3f}</td></tr>"
            )
        except Exception:
            pass
    if nli_labels is not None:
        from sklearn.metrics import roc_auc_score as _auc
        try:
            auc_nli = _auc(nli_labels > 0.5, halluc_prob)
            prec = float(np.mean((nli_labels > 0.5)[flagged])) if flagged.any() else float("nan")
            val_rows += (
                f"<tr><td>NLI (low entailment)</td>"
                f"<td>{auc_nli:.3f}</td><td>{prec:.3f}</td><td>—</td></tr>"
            )
        except Exception:
            pass

    n_flagged = int(flagged.sum())
    n_total   = len(halluc_prob)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exp 3 — Calibrated Hallucination Flagging</title>
  <style>
    body      {{ font-family: Georgia, serif; max-width: 1000px;
                margin: 40px auto; padding: 0 24px; color: #222; }}
    h2        {{ border-bottom: 2px solid #eee; padding-bottom: 8px; }}
    h3        {{ color: #555; margin-top: 32px; }}
    img       {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 12px 0; }}
    .meta     {{ font-size: 13px; color: #777; margin-bottom: 24px; }}
    .note-box {{ background: #fafafa; border: 1px solid #ddd; border-radius: 6px;
                padding: 22px 26px; line-height: 2.2; font-size: 14px;
                white-space: pre-wrap; word-break: break-word; }}
    table     {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }}
    th, td    {{ border: 1px solid #ddd; padding: 7px 12px; text-align: left; }}
    th        {{ background: #f5f5f5; }}
    .legend   {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0;
                font-size: 13px; align-items: center; }}
    .grad     {{ width: 180px; height: 16px; border-radius: 4px;
                background: linear-gradient(to right, #ffffff, #f44336);
                border: 1px solid #ddd; }}
  </style>
</head>
<body>
<h2>Experiment 3 — Calibrated Token-Level Hallucination Flagging</h2>
<p class="meta">
  Model: <strong>{model_name}</strong> &nbsp;|&nbsp;
  Sample: <strong>{sample_idx}</strong> &nbsp;|&nbsp;
  Tokens: <strong>{n_total}</strong> &nbsp;|&nbsp;
  Flagged (prob ≥ {threshold:.2f}): <strong>{n_flagged}</strong>
  ({100*n_flagged/n_total:.1f}%)
</p>

<h3>ECS vs PKS — Generated Note (coloured by hallucination probability)</h3>
<img src="data:image/png;base64,{img_b64}" alt="Exp 3 scatter">

<h3>Calibration — Selected Layers from Exp 2c</h3>
<table>
  <tr><th>Metric</th><th>Layer</th><th>AUROC</th><th>Youden J</th>
      <th>Threshold</th><th>Flag when</th></tr>
  {cal_rows}
</table>

{"<h3>Validation</h3><table><tr><th>Signal</th><th>AUROC vs halluc_prob</th><th>Precision@flagged</th><th>Recall@flagged</th></tr>" + val_rows + "</table>" if val_rows else ""}

<h3>Generated Note — Hallucination Probability Highlights</h3>
<div class="legend">
  <span>Low risk</span>
  <div class="grad"></div>
  <span>High risk</span>
  &nbsp; (hover token for exact probability)
</div>
<div class="note-box">{note_html}</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


