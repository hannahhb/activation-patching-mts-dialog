# SAE Experiments — MTS Dialog Summarisation

Mechanistic interpretability experiments on real **MTS (Multi-Turn Speaker) doctor-patient
dialogues** using **Gemma 2 2B** + TransformerLens activation patching.

Dataset: [`har1/MTS_Dialogue-Clinical_Note`](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note)
(1.7 k de-identified clinical conversations — MEDIQA-Sum / MEDIQA-Chat 2023)

---

## Hypothesis

| Role | Sublayer | Layer | Mechanism |
|---|---|---|---|
| **Copying / Extraction** | Attention-Out | ~10 | Induction heads copy symptom tokens from the dialogue to the summary position |
| **Construction / Abstraction** | MLP-Out | ~20 | Key-value memories synthesise clinical concepts (symptom cluster → diagnosis) |

---

## File Structure

```
sae_experiments/
├── config.py           # All hyperparameters (model, layers, dataset, N_EXAMPLES)
├── data.py             # HuggingFace dataset loading + speaker-swap corruption
├── metrics.py          # Logit-diff metric + auto answer-token detection
├── patching.py         # Activation patching functions (layer, head, position)
├── visualise.py        # Matplotlib/seaborn plotting helpers
├── run_experiment.py   # Main entry point (CLI)
└── results/            # Auto-created — PNGs + patching_results.json
```

---

## Experiment Design

**Corruption strategy: speaker swap**

```
[Doctor]: I have chest pain ...  →  [Patient]: I have chest pain ...
[Patient]: When did it start?   →  [Doctor]:  When did it start?
```

- Same token length as the original → clean position-level patching
- Breaks the attribution of who has symptoms → changes summary prediction

**Metric: Normalised Logit-Difference Recovery**

```
recovery = (LD_patched − LD_corrupted) / (LD_clean − LD_corrupted)
```

| Value | Meaning |
|---|---|
| 0 | Still looks corrupted |
| 1 | Fully restored to clean |
| < 0 | Patching made things worse |

---

## Patching Experiments

| Experiment | What it measures |
|---|---|
| **Residual stream (all layers)** | Where in the network does information flow in? |
| **Attention-Out (all layers)** | Which layers carry the "copying" signal? |
| **MLP-Out (all layers)** | Which layers carry the "construction" signal? |
| **Per-head heatmap (L6–L14)** | Which specific attention heads are the induction heads? |
| **Token-position sweep (L10, L20)** | Which dialogue tokens (which turns) drive each sublayer? |

---

## Quick Start

```bash
# Run on 10 examples, save plots only
python run_experiment.py

# Run on 5 examples, show plots interactively
python run_experiment.py --n_examples 5 --show_plots

# Fast run (skip per-token-position sweep)
python run_experiment.py --n_examples 3 --skip_position_sweep --verbose

# Preview the dataset only
python data.py
```

---

## Output (results/)

| File | Description |
|---|---|
| `patching_results.json` | All numeric results + per-example metadata |
| `layer_patching_ex{N}.png` | Resid / Attn-Out / MLP-Out curves for example N |
| `head_heatmap_ex{N}.png` | Per-head recovery heatmap for example N |
| `position_patching_ex{N}.png` | Token-position recovery for example N |
| `aggregate_layer_patching.png` | Mean ± std across all examples |
| `aggregate_head_heatmap.png` | Mean head heatmap across all examples |

---

## Phase 2 — Gemma Scope SAE Analysis (next)

`02_sae_analysis.py` will load Gemma Scope SAEs at the identified layers:

| Layer | SAE type | Expected features |
|---|---|---|
| `ATTN_TARGET_LAYER` (10) | Attention-Out | Symptom-copying features |
| `MLP_TARGET_LAYER` (20) | MLP-Out | Clinical concept construction features |

---

## Environment

Assumes GPU environment with:
```
transformer_lens  torch (CUDA)  datasets  matplotlib  seaborn  numpy
```

Sources:
- [MTS-Dialog dataset (har1/MTS_Dialogue-Clinical_Note)](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note)
- [MTS-Dialog GitHub (abachaa/MTS-Dialog)](https://github.com/abachaa/MTS-Dialog)
- [EACL 2023 paper](https://aclanthology.org/2023.eacl-main.168/)
