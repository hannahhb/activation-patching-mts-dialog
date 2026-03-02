"""
Experiment configuration for MTS Dialog Summarisation — Activation Patching
"""

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b"
DEVICE     = "cuda"
DTYPE      = "bfloat16"   # Gemma 2 native precision

# ── Gemma 2 2B architecture reference ─────────────────────────────────────
N_LAYERS = 26
N_HEADS  = 8
D_MODEL  = 2304
D_HEAD   = 256

# ── Dataset ────────────────────────────────────────────────────────────────
DATASET_NAME   = "har1/MTS_Dialogue-Clinical_Note"
DATASET_SPLIT  = "train"      # dataset only has a train split
DIALOG_FIELD   = "dialogue"
SUMMARY_FIELD  = "section_text"

N_EXAMPLES     = 10    # number of MTS examples to run patching over
RANDOM_SEED    = 42

# ── Corruption strategy ────────────────────────────────────────────────────
# "speaker_swap" → swap Doctor/Patient labels (same token length, clean symmetry)
CORRUPTION     = "speaker_swap"

# ── Target layers (hypothesis-driven) ─────────────────────────────────────
ATTN_TARGET_LAYER = 10   # Copying / Extraction via Induction Heads
MLP_TARGET_LAYER  = 20   # Construction / Abstraction

# Layers to include in the per-head patching sweep (window around attn target)
HEAD_SWEEP_LAYERS = list(range(
    max(0,            ATTN_TARGET_LAYER - 4),
    min(N_LAYERS,     ATTN_TARGET_LAYER + 5),
))

# ── Output ─────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
