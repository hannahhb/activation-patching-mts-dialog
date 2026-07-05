"""
annotator.py
============
Hallucination annotation tool for clinical SOAP notes.

Token-level  (select text in the note → pick type + severity):
  Taxonomy from Hegselmann et al. 2402.15422 (11 classes):
    Unsupported: condition | procedure | medication | time | location |
                 number | name | word | other
    contradicted | incorrect
  Severity: 1–5 numeric scale

Sentence-level  (one label + severity per sentence, Asagri et al.):
  Faithful | Fabrication | Negation | Causality | Contextual
  Severity: 1–5 (N/A for Faithful)

Saves character offsets (char_start, char_end) for every token span.

Sources:
  - generations:  luq_out/llama/generations/aci/test1/sample_NNN_generations.json
  - sentences:    luq_out/llama/sentences/sample_NNN_note_KK_sentences.csv
  - output:       annotations/sample_NNN_note_KK.json

Usage:
    python annotator.py            # starts at sample 0, note 0
    python annotator.py --sample 2 --note 1
"""

import argparse
import json
import re
import csv
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

# ── Paths ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
GEN_DIR    = BASE / "luq_out" / "llama" / "generations" / "aci" / "test1"
SENT_DIR   = BASE / "luq_out" / "llama" / "sentences"
ANNOT_DIR  = BASE / "annotations"
ANNOT_DIR.mkdir(exist_ok=True)

# The three human annotators whose sentence_labels get cross-checked for
# disagreements. "consensus" is a fourth pseudo-annotator identity (reuses
# the existing annot_path/save machinery) used to record the adjudicated
# final label once a disagreement is resolved.
ANNOTATORS = ["hannah", "rashika_bahl", "daniel"]
CONSENSUS_NAME = "consensus"

app = Flask(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────
def gen_path(sid):
    return GEN_DIR / f"sample_{sid:03d}_generations.json"

def sent_path(sid, nid):
    return SENT_DIR / f"sample_{sid:03d}_note_{nid:02d}_sentences.csv"

def sanitize_name(name):
    import re
    return re.sub(r'[^a-z0-9_]', '_', name.lower().strip())[:32] or "annotator"

def annot_path(sid, nid, annotator="default"):
    name = sanitize_name(annotator)
    return ANNOT_DIR / f"sample_{sid:03d}_note_{nid:02d}_{name}.json"

def list_annotators(sid, nid):
    pattern = f"sample_{sid:03d}_note_{nid:02d}_*.json"
    return [p.stem.split("_", 4)[-1] for p in sorted(ANNOT_DIR.glob(pattern))]

def available_samples():
    return sorted(int(p.stem.split("_")[1]) for p in GEN_DIR.glob("sample_*_generations.json"))

def split_sentences(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sentences = []
    for line in lines:
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences

def load_sentences(sid, nid, note_text):
    p = sent_path(sid, nid)
    if p.exists():
        with open(p) as f:
            rows = list(csv.DictReader(f))
        return [r["sentence"] for r in rows if r["sentence"].strip()]
    return split_sentences(note_text)

def load_generation(sid, nid):
    p = gen_path(sid)
    if not p.exists():
        return None, None, None
    data = json.loads(p.read_text())
    notes = data.get("notes", [])
    if nid >= len(notes):
        return None, None, None
    return data.get("transcript", ""), notes[nid], len(notes)

def load_annotations(sid, nid, annotator="default"):
    p = annot_path(sid, nid, annotator)
    if p.exists():
        data = json.loads(p.read_text())
        sl = data.get("sentence_labels", {})
        _sev_map = {"Major": "4", "Minor": "2", "N/A": "N/A"}
        _err_types = {"Fabrication", "Negation", "Causality", "Contextual"}
        def _migrate(v):
            if not isinstance(v, dict):
                old_type = v
                faithful = "Faithful" if old_type == "Faithful" else ("Not Faithful" if old_type in _err_types else None)
                err_type = old_type if old_type in _err_types else None
                return {"faithful": faithful, "type": err_type, "severity": "N/A"}
            sev = v.get("severity", "N/A")
            if sev in _sev_map:
                sev = _sev_map[sev]
            # Migrate old single-field format where type included "Faithful"
            old_type = v.get("type")
            if "faithful" not in v:
                faithful = "Faithful" if old_type == "Faithful" else ("Not Faithful" if old_type in _err_types else None)
                err_type = old_type if old_type in _err_types else None
                return {"faithful": faithful, "type": err_type, "severity": sev}
            return {"faithful": v.get("faithful"), "type": v.get("type"), "severity": sev}
        data["sentence_labels"] = {k: _migrate(v) for k, v in sl.items()}
        return data
    return {"token_spans": [], "sentence_labels": {}, "note_html": ""}

def annotated_sample_notes():
    """(sid, nid) pairs that have a saved annotation file from >=1 of ANNOTATORS."""
    combos = set()
    for name in ANNOTATORS:
        for p in ANNOT_DIR.glob(f"sample_*_note_*_{sanitize_name(name)}.json"):
            m = re.match(r"sample_(\d+)_note_(\d+)_", p.stem)
            if m:
                combos.add((int(m.group(1)), int(m.group(2))))
    return sorted(combos)

def find_disagreements():
    """For every (sample, note) with >=2 annotators, compare sentence_labels
    per sentence index. A disagreement is: annotators don't all give the same
    'faithful' verdict, OR (given they all say Not Faithful) they don't all
    give the same error 'type'. Returns a list of dicts, one per disagreed
    sentence, each with every annotator's response plus any existing
    consensus decision."""
    out = []
    for sid, nid in annotated_sample_notes():
        per_annotator = {}
        for name in ANNOTATORS:
            if annot_path(sid, nid, name).exists():
                per_annotator[name] = load_annotations(sid, nid, name).get("sentence_labels", {})
        if len(per_annotator) < 2:
            continue

        transcript, note, n_notes = load_generation(sid, nid)
        if transcript is None:
            continue
        sentences = load_sentences(sid, nid, note)
        consensus_labels = load_annotations(sid, nid, CONSENSUS_NAME).get("sentence_labels", {})

        all_idxs = set()
        for labels in per_annotator.values():
            all_idxs.update(int(k) for k in labels.keys())

        for idx in sorted(all_idxs):
            key = str(idx)
            responses = {name: per_annotator[name][key]
                        for name in ANNOTATORS if name in per_annotator and key in per_annotator[name]}
            if len(responses) < 2:
                continue
            faiths = {r.get("faithful") for r in responses.values()}
            not_faithful_types = {r.get("type") for r in responses.values() if r.get("faithful") == "Not Faithful"}
            disagree = len(faiths) > 1 or len(not_faithful_types) > 1
            if not disagree:
                continue
            out.append({
                "sid": sid, "nid": nid, "idx": idx,
                "sentence": sentences[idx] if idx < len(sentences) else "<<sentence index out of range>>",
                "responses": responses,
                "consensus": consensus_labels.get(key),
            })
    return out

# ── HTML Template ──────────────────────────────────────────────────────────
TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Annotator — Sample {{ sid }} Note {{ nid }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; font-size: 14px; background: #f1f5f9; }

  header {
    display: flex; align-items: center; gap: 14px;
    padding: 9px 18px; background: #0f172a; color: #fff; position: sticky; top:0; z-index:100;
  }
  header h1 { font-size: 14px; font-weight: 600; flex: 1; }
  header select, header button {
    padding: 3px 9px; border-radius: 4px; border: none; cursor: pointer; font-size: 12px;
  }
  header button { background: #3b82f6; color: #fff; font-weight: 600; }
  header button:hover { background: #2563eb; }
  .btn-guide { background: #475569 !important; }
  .btn-guide:hover { background: #334155 !important; }
  .saved-msg { color: #86efac; font-size: 11px; opacity: 0; transition: opacity .6s; }
  .saved-msg.show { opacity: 1; }

  /* Annotator badge in header */
  .annotator-badge {
    display: flex; align-items: center; gap: 6px;
    background: #1e293b; border-radius: 5px; padding: 3px 10px;
    font-size: 11px; color: #94a3b8;
  }
  .annotator-badge strong { color: #e2e8f0; font-size: 12px; }
  .annotator-badge button { background: #334155 !important; font-size: 10px; padding: 2px 6px !important; }

  /* Name modal */
  #name-modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,.65); z-index: 300;
    align-items: center; justify-content: center;
  }
  #name-modal-overlay.show { display: flex; }
  #name-modal {
    background: #fff; border-radius: 12px; padding: 32px 36px;
    width: 360px; box-shadow: 0 20px 60px rgba(0,0,0,.35); text-align: center;
  }
  #name-modal h2 { font-size: 18px; color: #0f172a; margin-bottom: 8px; }
  #name-modal p  { font-size: 13px; color: #64748b; margin-bottom: 20px; }
  #name-modal input {
    width: 100%; padding: 9px 12px; border: 1.5px solid #cbd5e1; border-radius: 6px;
    font-size: 14px; outline: none; margin-bottom: 14px;
  }
  #name-modal input:focus { border-color: #3b82f6; }
  #name-modal button {
    width: 100%; padding: 9px; background: #3b82f6; color: #fff;
    border: none; border-radius: 6px; font-size: 14px; font-weight: 700; cursor: pointer;
  }
  #name-modal button:hover { background: #2563eb; }
  #name-modal .others { margin-top: 16px; font-size: 11px; color: #94a3b8; }
  #name-modal .others span { display: inline-block; background: #f1f5f9; border-radius: 3px; padding: 1px 7px; margin: 2px; cursor: pointer; color: #475569; }
  #name-modal .others span:hover { background: #e2e8f0; }

  /* Guidelines drawer */
  #guide-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,.4); z-index: 200;
  }
  #guide-drawer {
    position: fixed; top: 0; right: -600px; width: 580px; height: 100vh;
    background: #fff; box-shadow: -4px 0 24px rgba(0,0,0,.2);
    z-index: 201; overflow-y: auto; transition: right .25s ease;
    padding: 0;
  }
  #guide-drawer.open { right: 0; }
  .guide-hdr {
    position: sticky; top: 0; background: #0f172a; color: #fff;
    padding: 11px 16px; display: flex; align-items: center; gap: 10px; z-index: 1;
  }
  .guide-hdr h2 { font-size: 13px; font-weight: 700; flex: 1; }
  .guide-hdr button { background: #334155; color: #fff; border: none; border-radius: 4px; padding: 3px 9px; cursor: pointer; font-size: 12px; }
  .guide-body { padding: 16px; }
  .guide-section { margin-bottom: 22px; }
  .guide-section h3 { font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .07em; color: #64748b; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e2e8f0; }
  .guide-section h4 { font-size: 12px; font-weight: 700; margin: 10px 0 4px; }
  .guide-tag { display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 11px; font-weight: 700; color: #fff; margin-right: 4px; margin-bottom: 2px; vertical-align: middle; }
  .guide-row { margin-bottom: 8px; font-size: 12px; line-height: 1.6; color: #334155; }
  .guide-row em { color: #64748b; font-style: normal; }
  .guide-example { margin: 3px 0 3px 12px; color: #475569; font-size: 11px; font-style: italic; }
  .guide-note { background: #f0f9ff; border-left: 3px solid #0891b2; padding: 6px 10px; font-size: 11px; color: #0c4a6e; margin: 8px 0; border-radius: 0 4px 4px 0; }
  .sev-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 6px; }
  .sev-table th { background: #f1f5f9; padding: 4px 8px; text-align: left; font-weight: 700; border: 1px solid #e2e8f0; }
  .sev-table td { padding: 4px 8px; border: 1px solid #e2e8f0; vertical-align: top; }

  .workspace { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px 16px; height: calc(100vh - 46px); }
  .panel { background:#fff; border:1px solid #e2e8f0; border-radius:8px; display:flex; flex-direction:column; overflow:hidden; }
  .panel-hdr {
    padding: 7px 12px; background:#f8fafc; border-bottom:1px solid #e2e8f0;
    font-weight:700; font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.06em;
    display:flex; align-items:center; gap:6px;
  }
  .panel-body { overflow-y:auto; padding:12px; flex:1; }

  /* transcript */
  .transcript { white-space:pre-wrap; line-height:1.7; font-size:13px; color:#334155; }
  .turn-d { color:#1d4ed8; }
  .turn-p { color:#047857; }

  /* floating toolbar — 11-class taxonomy (Hegselmann et al. 2402.15422) */
  #toolbar {
    display:none; position:fixed; z-index:999;
    background:#0f172a; border-radius:8px; padding:8px 10px; gap:5px;
    box-shadow:0 6px 20px rgba(0,0,0,.35); flex-direction:column; min-width:420px;
  }
  #toolbar .tb-row { display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
  #toolbar .tb-lbl { color:#94a3b8; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.05em; width:54px; flex-shrink:0; }
  #toolbar .tb-section-lbl { color:#64748b; font-size:9px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; padding: 2px 0 1px 54px; }
  .tb-btn {
    padding:3px 7px; border:none; border-radius:4px;
    font-size:11px; font-weight:600; cursor:pointer; color:#fff; transition:opacity .15s;
  }
  .tb-btn:hover { opacity:.82; }
  .tb-btn.active { outline:2px solid #fff; }

  /* unsupported subtypes */
  .btn-cond { background:#dc2626; }
  .btn-proc { background:#ea580c; }
  .btn-med  { background:#d97706; }
  .btn-time { background:#16a34a; }
  .btn-loc  { background:#0891b2; }
  .btn-num  { background:#2563eb; }
  .btn-name { background:#7c3aed; }
  .btn-word { background:#db2777; }
  .btn-oth  { background:#6b7280; }
  /* other classes */
  .btn-cont { background:#7f1d1d; }
  .btn-incr { background:#78350f; }
  /* severity 1–5 */
  .btn-sev  { background:#334155; min-width:28px; text-align:center; }
  .btn-sev.active { background:#1e40af; }

  .btn-clear{ background:#475569; }
  #toolbar .apply-btn {
    background:#3b82f6; color:#fff; border:none; border-radius:4px;
    padding:4px 14px; font-size:12px; font-weight:700; cursor:pointer; align-self:flex-end;
  }
  #toolbar .apply-btn:hover { background:#2563eb; }
  #toolbar .tb-divider { border:none; border-top:1px solid #334155; margin:2px 0; }

  /* highlights — 11 classes */
  .hl-cond { background:#fca5a5; border-bottom:2px solid #dc2626; border-radius:2px; cursor:pointer; }
  .hl-proc { background:#fed7aa; border-bottom:2px solid #ea580c; border-radius:2px; cursor:pointer; }
  .hl-med  { background:#fde68a; border-bottom:2px solid #d97706; border-radius:2px; cursor:pointer; }
  .hl-time { background:#bbf7d0; border-bottom:2px solid #16a34a; border-radius:2px; cursor:pointer; }
  .hl-loc  { background:#bae6fd; border-bottom:2px solid #0891b2; border-radius:2px; cursor:pointer; }
  .hl-num  { background:#bfdbfe; border-bottom:2px solid #2563eb; border-radius:2px; cursor:pointer; }
  .hl-name { background:#e9d5ff; border-bottom:2px solid #7c3aed; border-radius:2px; cursor:pointer; }
  .hl-word { background:#fbcfe8; border-bottom:2px solid #db2777; border-radius:2px; cursor:pointer; }
  .hl-oth  { background:#e2e8f0; border-bottom:2px solid #6b7280; border-radius:2px; cursor:pointer; }
  .hl-cont { background:#fecaca; border-bottom:2px solid #7f1d1d; border-radius:2px; cursor:pointer; }
  .hl-incr { background:#ffedd5; border-bottom:2px solid #78350f; border-radius:2px; cursor:pointer; }

  /* severity superscript on highlighted spans */
  [data-hl-sev]::after {
    content: attr(data-hl-sev);
    font-size: 8px; vertical-align: super; margin-left: 1px;
    font-weight: 700; color: #1e293b;
  }

  #note-text { white-space:pre-wrap; line-height:1.75; font-size:13px; color:#1e293b; user-select:text; }

  /* sentence blocks */
  .sent-block { margin:5px 0; padding:8px 10px 8px 12px; border-left:3px solid #e2e8f0; border-radius:0 4px 4px 0; background:#f8fafc; }
  .sent-text { font-size:13px; color:#334155; margin-bottom:7px; line-height:1.6; }
  .sent-row { display:flex; align-items:center; gap:4px; margin-bottom:4px; }
  .sent-row-lbl { font-size:10px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em; width:56px; flex-shrink:0; }
  .radio-group { display:flex; gap:4px; flex-wrap:wrap; }
  .radio-group label {
    display:flex; align-items:center; gap:3px; font-size:11px; font-weight:500; cursor:pointer;
    padding:2px 8px; border-radius:10px; border:1px solid #cbd5e1; background:#fff; transition:background .12s;
  }
  .radio-group input[type=radio] { display:none; }

  /* sentence type colours */
  .lbl-faithful    { --c:#16a34a; }
  .lbl-fabrication { --c:#dc2626; }
  .lbl-negation    { --c:#ea580c; }
  .lbl-causality   { --c:#7c3aed; }
  .lbl-contextual  { --c:#0891b2; }
  /* severity 1–5 colours (cool-warm gradient) */
  .lbl-sev1 { --c:#6b7280; }
  .lbl-sev2 { --c:#d97706; }
  .lbl-sev3 { --c:#ea580c; }
  .lbl-sev4 { --c:#dc2626; }
  .lbl-sev5 { --c:#7f1d1d; }
  .lbl-na   { --c:#94a3b8; }
  .radio-group label:has(input:checked) {
    background: color-mix(in srgb, var(--c) 18%, #fff);
    border-color: var(--c); color: var(--c); font-weight:700;
  }

  /* legend */
  .legend { display:flex; gap:8px; padding:5px 12px 7px; border-top:1px solid #e2e8f0; flex-wrap:wrap; }
  .ld { display:flex; align-items:center; gap:4px; font-size:10px; color:#64748b; }
  .ld-dot { width:10px; height:10px; border-radius:2px; flex-shrink:0; }
</style>
</head>
<body>

<header>
  <h1>Clinical Note Annotator</h1>
  <label style="color:#94a3b8;font-size:11px">Sample
    <select id="sel-sample" onchange="navigate()">
      {% for s in samples %}<option value="{{ s }}" {% if s==sid %}selected{% endif %}>{{ "%03d"|format(s) }}</option>{% endfor %}
    </select>
  </label>
  <label style="color:#94a3b8;font-size:11px">Note
    <select id="sel-note" onchange="navigate()">
      {% for n in range(n_notes) %}<option value="{{ n }}" {% if n==nid %}selected{% endif %}>{{ "%02d"|format(n) }}</option>{% endfor %}
    </select>
  </label>
  <div class="annotator-badge">
    👤 <strong id="annotator-display">{{ annotator or "—" }}</strong>
    <button onclick="showNameModal()">Change</button>
  </div>
  <button class="btn-guide" onclick="toggleGuide()">📖 Guidelines</button>
  <button class="btn-guide" onclick="window.location.href='/disagreements'">⚖️ Disagreements</button>
  <button onclick="saveAnnotations()">Save</button>
  <span class="saved-msg" id="saved-msg">✓ Saved</span>
</header>

<!-- Name modal -->
<div id="name-modal-overlay" class="{{ 'show' if not annotator else '' }}">
  <div id="name-modal">
    <h2>Who are you?</h2>
    <p>Enter your name so your annotations are saved separately.</p>
    <input type="text" id="name-input" placeholder="e.g. Alice" maxlength="32"
      value="{{ annotator }}"
      onkeydown="if(event.key==='Enter') confirmName()">
    <button onclick="confirmName()">Start Annotating</button>
    {% if annotators %}
    <div class="others">
      Recent annotators:
      {% for a in annotators %}
      <span onclick="loadAs('{{ a }}')">{{ a }}</span>
      {% endfor %}
    </div>
    {% endif %}
  </div>
</div>

<!-- Guidelines drawer -->
<div id="guide-overlay" onclick="toggleGuide()"></div>
<div id="guide-drawer">
  <div class="guide-hdr">
    <h2>Annotation Guidelines</h2>
    <button onclick="toggleGuide()">✕ Close</button>
  </div>
  <div class="guide-body">

    <!-- TOKEN LEVEL -->
    <div class="guide-section">
      <h3>Token-level — Hegselmann et al. (2402.15422)</h3>
      <div class="guide-note">Highlight the <strong>minimal span</strong> needed — removing any part would lose the error. One span = one error type. Ground truth is the <strong>transcript only</strong>; general medical advice is allowed (e.g. "take your medications as prescribed").</div>

      <h4><span class="guide-tag" style="background:#b91c1c">UNSUPPORTED</span> — fact not supported by the transcript</h4>

      <div class="guide-row">
        <span class="guide-tag" style="background:#dc2626">condition</span>
        Unsupported medical condition, diagnosis, or finding.<br>
        <span class="guide-example">e.g. "You were found to have a left clavicle fracture" — no such condition in transcript.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#ea580c">procedure</span>
        Unsupported medical procedure or intervention.<br>
        <span class="guide-example">e.g. "You had a filter placed in your vein" — procedure not mentioned.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#d97706">medication</span>
        Unsupported medication name, class, route, frequency, or dosage.<br>
        <span class="guide-example">e.g. "You were placed on antibiotics" when only blood thinners were prescribed.</span><br>
        <span class="guide-example">e.g. "We gave you blood thinners by mouth" when they were given by IV.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#16a34a">time</span>
        Unsupported time or interval statement.<br>
        <span class="guide-example">e.g. "Keep your arm in a sling for the next 6 weeks" — duration not stated.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#0891b2">location</span>
        Unsupported physical place <em>or</em> body region/side.<br>
        <span class="guide-example">e.g. "There was concern for a thrombus in the right leg" — side not specified.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#2563eb">number</span>
        Unsupported number (digits or written), including "a" / "an".<br>
        <span class="guide-example">e.g. "Your pacemaker rate was increased to 50" — rate not in transcript.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#7c3aed">name</span>
        Unsupported named entity (person, service, facility). <em>Use medication for drug names.</em><br>
        <span class="guide-example">e.g. "You were seen by the interventional pulmonary service" — not mentioned.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#db2777">word</span>
        Incorrect or inappropriate word/phrase that doesn't fit any above type.<br>
        <span class="guide-example">e.g. "Limit your use of stairs" when movement was encouraged.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#6b7280">other</span>
        Clearly a mistake but doesn't fit any above category. Use as last resort.
      </div>

      <h4>Other error classes</h4>
      <div class="guide-row">
        <span class="guide-tag" style="background:#7f1d1d">contradicted</span>
        Directly contradicts a fact <em>stated</em> in the transcript.<br>
        <span class="guide-example">e.g. "Your pacemaker rate was increased to 50" when transcript says 40.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#78350f">incorrect</span>
        Contradicts general medical knowledge (independent of transcript).<br>
        <span class="guide-example">e.g. "You can continue driving your car" after diagnosing a seizure.</span>
      </div>

      <div class="guide-note" style="margin-top:10px">
        <strong>Priority rule:</strong> if unsure between types, prefer the one listed earlier above. E.g. unsupported medication name → use <em>medication</em>, not <em>name</em>.
      </div>
    </div>

    <!-- SENTENCE LEVEL -->
    <div class="guide-section">
      <h3>Sentence-level — Asagri et al. (CREOLA)</h3>
      <div class="guide-note">Label each sentence for whether it contains a hallucination and how severe. <strong>Major</strong> = impacts diagnosis or management. <strong>Minor</strong> = does not affect clinical decisions.</div>

      <div class="guide-row">
        <span class="guide-tag" style="background:#16a34a">Faithful</span>
        The sentence is fully supported by the transcript. No errors.
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#dc2626">Fabrication</span>
        The model produced information not present in the transcript.<br>
        <span class="guide-example">e.g. Note mentions a condition or medication the patient never mentioned.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#ea580c">Negation</span>
        The model negates a clinically relevant fact from the transcript.<br>
        <span class="guide-example">e.g. Transcript: "patient has hypertension" → Note: "no hypertension noted".</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#7c3aed">Causality</span>
        The model speculates a causal relationship not explicitly supported by the transcript.<br>
        <span class="guide-example">e.g. "Dyspnoea was caused by anxiety" when the transcript gave no causal explanation.</span>
      </div>
      <div class="guide-row">
        <span class="guide-tag" style="background:#0891b2">Contextual</span>
        The model mixes topics otherwise unrelated in the given transcript context.<br>
        <span class="guide-example">e.g. Linking a patient's knee pain to their respiratory complaint without transcript basis.</span>
      </div>
    </div>

    <!-- SEVERITY -->
    <div class="guide-section">
      <h3>Severity scale (both levels)</h3>
      <table class="sev-table">
        <tr><th>Score</th><th>Meaning</th></tr>
        <tr><td><strong>1</strong></td><td>Trivial — no clinical impact, purely cosmetic error</td></tr>
        <tr><td><strong>2</strong></td><td>Minor — unlikely to affect clinical decisions; short-term at worst</td></tr>
        <tr><td><strong>3</strong></td><td>Moderate — could cause confusion but unlikely to cause direct harm</td></tr>
        <tr><td><strong>4</strong></td><td>Major — impacts diagnosis or management; potential for patient harm</td></tr>
        <tr><td><strong>5</strong></td><td>Critical — could directly lead to serious adverse outcome</td></tr>
      </table>
      <div class="guide-note" style="margin-top:8px">Use <strong>N/A</strong> for Faithful sentences. Negation errors tend to score higher as they directly contradict clinical facts.</div>
    </div>

  </div>
</div>

<!-- Floating token toolbar (2402.15422 taxonomy) -->
<div id="toolbar">
  <div class="tb-section-lbl">Unsupported</div>
  <div class="tb-row">
    <span class="tb-lbl">Type</span>
    <button class="tb-btn btn-cond" onclick="selectType('cond',this)">condition</button>
    <button class="tb-btn btn-proc" onclick="selectType('proc',this)">procedure</button>
    <button class="tb-btn btn-med"  onclick="selectType('med', this)">medication</button>
    <button class="tb-btn btn-time" onclick="selectType('time',this)">time</button>
    <button class="tb-btn btn-loc"  onclick="selectType('loc', this)">location</button>
    <button class="tb-btn btn-num"  onclick="selectType('num', this)">number</button>
    <button class="tb-btn btn-name" onclick="selectType('name',this)">name</button>
    <button class="tb-btn btn-word" onclick="selectType('word',this)">word</button>
    <button class="tb-btn btn-oth"  onclick="selectType('oth', this)">other</button>
  </div>
  <hr class="tb-divider">
  <div class="tb-row">
    <span class="tb-lbl"></span>
    <button class="tb-btn btn-cont" onclick="selectType('cont',this)">contradicted</button>
    <button class="tb-btn btn-incr" onclick="selectType('incr',this)">incorrect</button>
  </div>
  <hr class="tb-divider">
  <div class="tb-row">
    <span class="tb-lbl">Severity</span>
    <button class="tb-btn btn-sev" onclick="selectSev('1',this)">1</button>
    <button class="tb-btn btn-sev" onclick="selectSev('2',this)">2</button>
    <button class="tb-btn btn-sev" onclick="selectSev('3',this)">3</button>
    <button class="tb-btn btn-sev" onclick="selectSev('4',this)">4</button>
    <button class="tb-btn btn-sev" onclick="selectSev('5',this)">5</button>
    <span style="color:#64748b;font-size:10px;margin-left:4px">1=trivial · 5=critical</span>
  </div>
  <div class="tb-row" style="justify-content:space-between;margin-top:2px">
    <button class="tb-btn btn-clear" onclick="clearHighlight()">Clear selection</button>
    <button class="apply-btn" onclick="applyHighlight()">Apply ▸</button>
  </div>
</div>

<div class="workspace">

  <!-- Transcript -->
  <div class="panel">
    <div class="panel-hdr">📋 Transcript</div>
    <div class="panel-body">
      <div class="transcript">{{ transcript_html|safe }}</div>
    </div>
  </div>

  <!-- Note -->
  <div class="panel">
    <div class="panel-hdr">
      📝 Generated SOAP Note &nbsp;
      <span style="font-size:10px;color:#94a3b8;font-weight:400;margin-left:auto">Select text → tag | Labels per sentence below</span>
    </div>

    <!-- Token annotation zone -->
    <div class="panel-body" style="flex:0 0 38%;border-bottom:1px solid #e2e8f0;">
      <div id="note-text" onmouseup="onTextSelect(event)">{{ note_html|safe }}</div>
    </div>

    <!-- Sentence labels -->
    <div class="panel-body" style="flex:1">
      <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">
        Sentence-level labels (Asagri et al.)
      </div>
      {% for i, sent in sentences %}
      {% set sl = annotations.sentence_labels.get(i|string, {}) %}
      {% set is_faithful = sl.get('faithful') == 'Faithful' %}
      <div class="sent-block" id="sb-{{ i }}">
        <div class="sent-text">{{ sent }}</div>

        <!-- Faithful / Not Faithful -->
        <div class="sent-row">
          <span class="sent-row-lbl">Faithful?</span>
          <div class="radio-group">
            <label class="lbl-faithful">
              <input type="radio" name="faith-{{ i }}" value="Faithful"
                {% if sl.get('faithful') == 'Faithful' %}checked{% endif %}
                onchange="onFaithChange({{ i }},'Faithful')">
              <span>Faithful</span>
            </label>
            <label class="lbl-fabrication">
              <input type="radio" name="faith-{{ i }}" value="Not Faithful"
                {% if sl.get('faithful') == 'Not Faithful' %}checked{% endif %}
                onchange="onFaithChange({{ i }},'Not Faithful')">
              <span>Not Faithful</span>
            </label>
          </div>
        </div>

        <!-- Error type — hidden when Faithful -->
        <div class="sent-row" id="type-row-{{ i }}" style="{{ 'display:none' if is_faithful else '' }}">
          <span class="sent-row-lbl">Type</span>
          <div class="radio-group">
            {% for lbl,cls in [("Fabrication","lbl-fabrication"),("Negation","lbl-negation"),("Causality","lbl-causality"),("Contextual","lbl-contextual")] %}
            <label class="{{ cls }}">
              <input type="radio" name="sl-{{ i }}" value="{{ lbl }}"
                {% if sl.get('type') == lbl %}checked{% endif %}
                onchange="onSentChange({{ i }},'type',this.value)">
              <span>{{ lbl }}</span>
            </label>
            {% endfor %}
          </div>
        </div>

        <!-- Severity 1–5 -->
        <div class="sent-row" id="sev-row-{{ i }}" style="{{ 'display:none' if is_faithful else '' }}">
          <span class="sent-row-lbl">Severity</span>
          <div class="radio-group">
            {% for lbl,cls in [("1","lbl-sev1"),("2","lbl-sev2"),("3","lbl-sev3"),("4","lbl-sev4"),("5","lbl-sev5")] %}
            <label class="{{ cls }}">
              <input type="radio" name="sev-{{ i }}" value="{{ lbl }}"
                {% if sl.get('severity') == lbl %}checked{% endif %}
                onchange="onSentChange({{ i }},'severity',this.value)">
              <span>{{ lbl }}</span>
            </label>
            {% endfor %}
          </div>
        </div>
      </div>
      {% endfor %}
    </div>

    <!-- Legend -->
    <div class="legend">
      <div class="ld"><div class="ld-dot" style="background:#fca5a5;border-bottom:2px solid #dc2626"></div>condition</div>
      <div class="ld"><div class="ld-dot" style="background:#fed7aa;border-bottom:2px solid #ea580c"></div>procedure</div>
      <div class="ld"><div class="ld-dot" style="background:#fde68a;border-bottom:2px solid #d97706"></div>medication</div>
      <div class="ld"><div class="ld-dot" style="background:#bbf7d0;border-bottom:2px solid #16a34a"></div>time</div>
      <div class="ld"><div class="ld-dot" style="background:#bae6fd;border-bottom:2px solid #0891b2"></div>location</div>
      <div class="ld"><div class="ld-dot" style="background:#bfdbfe;border-bottom:2px solid #2563eb"></div>number</div>
      <div class="ld"><div class="ld-dot" style="background:#e9d5ff;border-bottom:2px solid #7c3aed"></div>name</div>
      <div class="ld"><div class="ld-dot" style="background:#fbcfe8;border-bottom:2px solid #db2777"></div>word</div>
      <div class="ld"><div class="ld-dot" style="background:#e2e8f0;border-bottom:2px solid #6b7280"></div>other</div>
      <div class="ld"><div class="ld-dot" style="background:#fecaca;border-bottom:2px solid #7f1d1d"></div>contradicted</div>
      <div class="ld"><div class="ld-dot" style="background:#ffedd5;border-bottom:2px solid #78350f"></div>incorrect</div>
      <div class="ld" style="margin-left:6px;color:#94a3b8">severity superscript ¹²³⁴⁵</div>
    </div>
  </div>
</div>

<script>
const sid = {{ sid }};
const nid = {{ nid }};
let annotator = {{ annotator|tojson }};
let annotations = {{ annotations_json|safe }};
if (!annotations.sentence_labels) annotations.sentence_labels = {};
if (!annotations.token_spans)    annotations.token_spans = [];

let currentRange = null;
let pendingType = null;
let pendingSev  = null;

// ── Annotator name modal ────────────────────────────────────────────────────
function showNameModal() {
  document.getElementById('name-input').value = annotator || '';
  document.getElementById('name-modal-overlay').classList.add('show');
  setTimeout(() => document.getElementById('name-input').focus(), 50);
}
function confirmName() {
  const raw = document.getElementById('name-input').value.trim();
  if (!raw) return;
  annotator = raw;
  document.getElementById('annotator-display').textContent = raw;
  document.getElementById('name-modal-overlay').classList.remove('show');
  // Reload page with annotator so server loads their annotations
  const s = document.getElementById('sel-sample').value;
  const n = document.getElementById('sel-note').value;
  window.location.href = `/?sample=${s}&note=${n}&annotator=${encodeURIComponent(raw)}`;
}
function loadAs(name) {
  document.getElementById('name-input').value = name;
  confirmName();
}

// ── Navigation ──────────────────────────────────────────────────────────────
function navigate() {
  const s = document.getElementById('sel-sample').value;
  const n = document.getElementById('sel-note').value;
  const a = annotator ? `&annotator=${encodeURIComponent(annotator)}` : '';
  window.location.href = `/?sample=${s}&note=${n}${a}`;
}

// ── Char offset helpers ─────────────────────────────────────────────────────
function getCharOffset(root, targetNode, targetOffset) {
  let total = 0;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    if (walker.currentNode === targetNode) return total + targetOffset;
    total += walker.currentNode.textContent.length;
  }
  return total + targetOffset;
}

// ── Token selection ─────────────────────────────────────────────────────────
function onTextSelect(e) {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.toString().trim()) { hideToolbar(); return; }
  const noteEl = document.getElementById('note-text');
  if (!noteEl.contains(sel.anchorNode)) { hideToolbar(); return; }
  currentRange = sel.getRangeAt(0).cloneRange();
  const tb = document.getElementById('toolbar');
  tb.style.display = 'flex';
  const x = Math.min(e.pageX - 10, window.innerWidth - 440);
  const y = e.pageY + 14;
  tb.style.left = x + 'px';
  tb.style.top  = y + 'px';
}

document.addEventListener('mousedown', e => {
  if (!document.getElementById('toolbar').contains(e.target)) hideToolbar();
});

function hideToolbar() {
  document.getElementById('toolbar').style.display = 'none';
  pendingType = null; pendingSev = null;
  document.querySelectorAll('.tb-btn.active').forEach(b => b.classList.remove('active'));
}

function selectType(t, btn) {
  pendingType = t;
  document.querySelectorAll('.tb-btn[onclick^="selectType"]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}
function selectSev(s, btn) {
  pendingSev = s;
  document.querySelectorAll('.tb-btn[onclick^="selectSev"]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function applyHighlight() {
  if (!currentRange || !pendingType) {
    alert('Select a type first.'); return;
  }
  const noteEl = document.getElementById('note-text');
  const selText = currentRange.toString();
  if (!selText.trim()) return;

  const charStart = getCharOffset(noteEl, currentRange.startContainer, currentRange.startOffset);
  const charEnd   = getCharOffset(noteEl, currentRange.endContainer,   currentRange.endOffset);

  const sev = pendingSev || '3';
  const span = document.createElement('span');
  span.className = `hl-${pendingType}`;
  span.dataset.hlType = pendingType;
  span.dataset.hlSev  = sev;
  span.dataset.charStart = charStart;
  span.dataset.charEnd   = charEnd;
  span.title = `${pendingType} · severity ${sev} [${charStart}:${charEnd}]`;

  try {
    currentRange.surroundContents(span);
  } catch {
    const frag = currentRange.extractContents();
    span.appendChild(frag);
    currentRange.insertNode(span);
  }

  hideToolbar();
  persistTokenSpans();
}

function clearHighlight() {
  if (!currentRange) return;
  const noteEl = document.getElementById('note-text');
  noteEl.querySelectorAll('[data-hl-type]').forEach(s => {
    if (currentRange.intersectsNode(s)) {
      const p = s.parentNode;
      while (s.firstChild) p.insertBefore(s.firstChild, s);
      p.removeChild(s);
    }
  });
  hideToolbar();
  persistTokenSpans();
}

function persistTokenSpans() {
  const noteEl = document.getElementById('note-text');
  annotations.token_spans = Array.from(noteEl.querySelectorAll('[data-hl-type]')).map(s => ({
    text:       s.textContent,
    type:       s.dataset.hlType,
    severity:   s.dataset.hlSev,
    char_start: parseInt(s.dataset.charStart),
    char_end:   parseInt(s.dataset.charEnd),
  }));
}

// ── Sentence labels ─────────────────────────────────────────────────────────
function onFaithChange(idx, value) {
  const key = String(idx);
  if (!annotations.sentence_labels[key]) annotations.sentence_labels[key] = {};
  annotations.sentence_labels[key]['faithful'] = value;
  const typeRow = document.getElementById(`type-row-${idx}`);
  const sevRow  = document.getElementById(`sev-row-${idx}`);
  if (value === 'Faithful') {
    typeRow.style.display = 'none';
    sevRow.style.display  = 'none';
    // clear error type and severity
    annotations.sentence_labels[key]['type']     = null;
    annotations.sentence_labels[key]['severity']  = 'N/A';
    document.querySelectorAll(`input[name="sl-${idx}"]`).forEach(r => r.checked = false);
    document.querySelectorAll(`input[name="sev-${idx}"]`).forEach(r => r.checked = false);
  } else {
    typeRow.style.display = 'flex';
    sevRow.style.display  = 'flex';
  }
}

function onSentChange(idx, field, value) {
  const key = String(idx);
  if (!annotations.sentence_labels[key]) annotations.sentence_labels[key] = {};
  annotations.sentence_labels[key][field] = value;
}

// ── Save ────────────────────────────────────────────────────────────────────
function saveAnnotations() {
  if (!annotator) { showNameModal(); return; }
  persistTokenSpans();
  annotations.note_html = document.getElementById('note-text').innerHTML;
  fetch('/save', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({sid, nid, annotator, annotations}),
  }).then(r => r.json()).then(() => {
    const m = document.getElementById('saved-msg');
    m.classList.add('show');
    setTimeout(() => m.classList.remove('show'), 2000);
  });
}

document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); saveAnnotations(); }
  if (e.key === 'Escape') closeGuide();
});

// ── Guidelines drawer ───────────────────────────────────────────────────────
function toggleGuide() {
  const drawer = document.getElementById('guide-drawer');
  const overlay = document.getElementById('guide-overlay');
  const isOpen = drawer.classList.contains('open');
  if (isOpen) { closeGuide(); } else {
    drawer.classList.add('open');
    overlay.style.display = 'block';
  }
}
function closeGuide() {
  document.getElementById('guide-drawer').classList.remove('open');
  document.getElementById('guide-overlay').style.display = 'none';
}
</script>
</body>
</html>
"""

# ── Disagreements page template ─────────────────────────────────────────────
DISAGREEMENTS_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Disagreements — Annotator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; font-size: 14px; background: #f1f5f9; }
  header {
    display: flex; align-items: center; gap: 14px;
    padding: 9px 18px; background: #0f172a; color: #fff; position: sticky; top:0; z-index:100;
  }
  header h1 { font-size: 14px; font-weight: 600; flex: 1; }
  header button {
    padding: 3px 9px; border-radius: 4px; border: none; cursor: pointer; font-size: 12px;
    background: #3b82f6; color: #fff; font-weight: 600;
  }
  header button:hover { background: #2563eb; }
  header .count { color:#94a3b8; font-size: 12px; }

  .content { padding: 16px; max-width: 1100px; margin: 0 auto; }
  .empty { text-align:center; color:#64748b; padding: 60px 20px; font-size: 14px; }

  .card {
    background:#fff; border:1px solid #e2e8f0; border-radius:8px; margin-bottom: 14px; overflow:hidden;
  }
  .card-hdr {
    display:flex; align-items:center; gap:10px;
    padding: 8px 14px; background:#f8fafc; border-bottom:1px solid #e2e8f0;
    font-size: 11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
  }
  .card-hdr .loc { flex:1; }
  .card-hdr a {
    background:#3b82f6; color:#fff; text-decoration:none; padding: 4px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 700; text-transform: none; letter-spacing: normal;
  }
  .card-hdr a:hover { background:#2563eb; }
  .card-body { padding: 12px 14px; }
  .sent-text { font-size: 13px; color:#334155; margin-bottom: 10px; line-height:1.6; padding: 8px 10px; background:#f8fafc; border-left:3px solid #cbd5e1; border-radius:0 4px 4px 0; }

  table.resp-table { width:100%; border-collapse: collapse; font-size: 12px; }
  table.resp-table th { text-align:left; padding: 5px 8px; color:#94a3b8; font-weight:700; text-transform:uppercase; font-size:10px; letter-spacing:.04em; border-bottom:1px solid #e2e8f0; }
  table.resp-table td { padding: 6px 8px; border-bottom:1px solid #f1f5f9; }
  .tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:700; color:#fff; }
  .tag-faithful    { background:#16a34a; }
  .tag-fabrication { background:#dc2626; }
  .tag-negation    { background:#ea580c; }
  .tag-causality   { background:#7c3aed; }
  .tag-contextual  { background:#0891b2; }
  .tag-none        { background:#94a3b8; }
  .missing { color:#cbd5e1; font-style:italic; }
  .consensus-row td { background:#eff6ff; font-weight:600; }
  .consensus-empty { color:#94a3b8; font-style:italic; }
</style>
</head>
<body>
<header>
  <h1>⚖️ Disagreements</h1>
  <span class="count">{{ items|length }} disagreed sentence(s) across {{ n_samples }} sample/note pairs</span>
  <button onclick="window.location.href='/'">← Back to Annotator</button>
</header>
<div class="content">
{% if not items %}
  <div class="empty">No disagreements found yet — needs at least 2 of {{ annotators|join(', ') }} to have annotated the same sample/note.</div>
{% endif %}
{% for it in items %}
  <div class="card">
    <div class="card-hdr">
      <span class="loc">Sample {{ "%03d"|format(it.sid) }} · Note {{ "%02d"|format(it.nid) }} · Sentence {{ it.idx }}</span>
      <a href="/?sample={{ it.sid }}&note={{ it.nid }}&annotator=consensus#sb-{{ it.idx }}">Re-annotate as consensus ▸</a>
    </div>
    <div class="card-body">
      <div class="sent-text">{{ it.sentence }}</div>
      <table class="resp-table">
        <tr><th>Annotator</th><th>Faithful?</th><th>Type</th><th>Severity</th></tr>
        {% for name in annotators %}
        {% set r = it.responses.get(name) %}
        <tr>
          <td>{{ name }}</td>
          {% if r %}
          <td><span class="tag tag-{{ 'faithful' if r.faithful=='Faithful' else 'none' }}">{{ r.faithful or '—' }}</span></td>
          <td>{% if r.type %}<span class="tag tag-{{ r.type|lower }}">{{ r.type }}</span>{% else %}—{% endif %}</td>
          <td>{{ r.severity or '—' }}</td>
          {% else %}
          <td colspan="3" class="missing">not annotated</td>
          {% endif %}
        </tr>
        {% endfor %}
        <tr class="consensus-row">
          <td>Consensus</td>
          {% if it.consensus %}
          <td><span class="tag tag-{{ 'faithful' if it.consensus.faithful=='Faithful' else 'none' }}">{{ it.consensus.faithful or '—' }}</span></td>
          <td>{% if it.consensus.type %}<span class="tag tag-{{ it.consensus.type|lower }}">{{ it.consensus.type }}</span>{% else %}—{% endif %}</td>
          <td>{{ it.consensus.severity or '—' }}</td>
          {% else %}
          <td colspan="3" class="consensus-empty">not decided yet — click "Re-annotate as consensus" above</td>
          {% endif %}
        </tr>
      </table>
    </div>
  </div>
{% endfor %}
</div>
</body>
</html>
"""

# ── Server-side helpers ────────────────────────────────────────────────────
def format_transcript(text):
    def replace_turn(m):
        role = m.group(1).lower()
        content = m.group(2).strip()
        cls = "turn-d" if "doctor" in role else "turn-p"
        return f'<span class="{cls}">[{m.group(1)}]</span> {content}\n'
    return re.compile(r'\[([^\]]+)\]\s*(.*?)(?=\[[^\]]+\]|$)', re.DOTALL).sub(replace_turn, text)

def restore_note_html(note_text, saved_html):
    if saved_html:
        return saved_html
    import html as h
    return h.escape(note_text)

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    samples    = available_samples()
    sid        = int(request.args.get("sample", samples[0] if samples else 0))
    nid        = int(request.args.get("note", 0))
    annotator  = request.args.get("annotator", "").strip()

    transcript, note, n_notes = load_generation(sid, nid)
    if transcript is None:
        return f"<h3>No generation found for sample {sid}</h3>", 404

    sentences   = list(enumerate(load_sentences(sid, nid, note)))
    annotations = load_annotations(sid, nid, annotator) if annotator else {"token_spans": [], "sentence_labels": {}, "note_html": ""}
    annotators  = list_annotators(sid, nid)

    return render_template_string(
        TEMPLATE,
        sid=sid, nid=nid,
        samples=samples, n_notes=n_notes,
        annotator=annotator,
        annotators=annotators,
        transcript_html=format_transcript(transcript),
        note_html=restore_note_html(note, annotations.get("note_html", "") if annotator else ""),
        sentences=sentences,
        annotations=annotations,
        annotations_json=json.dumps(annotations),
    )

@app.route("/save", methods=["POST"])
def save():
    data      = request.json
    sid       = int(data["sid"])
    nid       = int(data["nid"])
    annotator = data.get("annotator", "default")
    annot     = data["annotations"]
    p = annot_path(sid, nid, annotator)
    p.write_text(json.dumps(annot, indent=2))
    return jsonify({"status": "ok", "path": str(p)})

@app.route("/disagreements")
def disagreements_page():
    items = find_disagreements()
    n_samples = len({(it["sid"], it["nid"]) for it in items})
    return render_template_string(
        DISAGREEMENTS_TEMPLATE,
        items=items,
        annotators=ANNOTATORS,
        n_samples=n_samples,
    )

# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical note hallucination annotator")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--note",   type=int, default=0)
    parser.add_argument("--port",   type=int, default=5050)
    args = parser.parse_args()

    samples = available_samples()
    print(f"\n  Annotator →  http://localhost:{args.port}/?sample={args.sample}&note={args.note}")
    print(f"  {len(samples)} samples | annotations → {ANNOT_DIR}\n")
    app.run(port=args.port, debug=False)
