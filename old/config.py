"""
Configuration for ACI-Bench × Gemma 2 9B-IT mechanistic interpretability experiments.
"""

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-9b-it"
DEVICE     = "cuda"
DTYPE      = "bfloat16"

# Gemma 2 9B-IT architecture
N_LAYERS = 42
N_HEADS  = 16
D_MODEL  = 3584
D_HEAD   = 256   # 16 heads × 256 = 4096; output-projected back to 3584

# Layer bands (equal thirds of 42 layers)
EARLY_LAYERS = list(range(0,  14))   # 0–13
MID_LAYERS   = list(range(14, 28))   # 14–27
LATE_LAYERS  = list(range(28, 42))   # 28–41

# Upper half of the network — used for source attention entropy
UPPER_LAYERS = list(range(21, 42))

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_HF_ID  = "mkieffer/ACI-Bench-MedARC"
DATASET_SPLIT  = "aci"
DIALOGUE_COL   = "dialogue"
NOTE_COL       = "note"

N_PILOT = 10    # Experiment 1 pilot
N_FULL  = 50    # Full-scale runs

RANDOM_SEED = 42

# ── Note generation ────────────────────────────────────────────────────────────
MAX_NEW_TOKENS   = 600
GENERATION_TEMP  = 0.0   # greedy decoding (temperature=0 → argmax)

SOAP_PROMPT_TEMPLATE = (
    "You are an expert clinical documentation specialist.\n"
    "Generate a structured SOAP clinical note from the following doctor-patient "
    "dialogue. Include all relevant clinical information.\n\n"
    "Format the note with exactly these section headers (include only sections "
    "that have content):\n"
    "CHIEF COMPLAINT:\n"
    "HISTORY OF PRESENT ILLNESS:\n"
    "REVIEW OF SYSTEMS:\n"
    "PHYSICAL EXAMINATION:\n"
    "ASSESSMENT:\n"
    "PLAN:\n\n"
    "DIALOGUE:\n{dialogue}\n\n"
    "SOAP NOTE:\n"
)

# ── PDSQI-9 judge ──────────────────────────────────────────────────────────────
# 9 attributes used for scoring and correlation analysis
PDSQI9_ATTRIBUTES = [
    "cited",
    "accurate",
    "thorough",
    "useful",
    "organized",
    "comprehensible",
    "succinct",
    "synthesized",
    "stigmatizing",
]

# Attributes that correlate with retrieval (attention) vs synthesis (MLP)
# These drive the primary mechanistic hypothesis
RETRIEVAL_ATTRS  = ["cited", "accurate", "thorough"]   # expect late-attn correlation
SYNTHESIS_ATTRS  = ["synthesized", "useful"]            # expect mid-MLP correlation

JUDGE_MODEL       = "gpt-4o"
JUDGE_TEMPERATURE = 0
JUDGE_N_SHOTS     = 5

# ── SOAP sections ──────────────────────────────────────────────────────────────
# Ordered list of sections — used for SOAP segmentation
SOAP_SECTIONS = [
    "chief complaint",
    "history of present illness",
    "review of systems",
    "physical examination",
    "assessment",
    "plan",
    "medications",
    "allergies",
    "past medical history",
    "social history",
    "family history",
]

# Sections expected to be extractive vs abstractive (for Experiment 2C)
EXTRACTIVE_SECTIONS  = {"chief complaint", "history of present illness",
                         "review of systems", "medications", "allergies",
                         "past medical history"}
ABSTRACTIVE_SECTIONS = {"assessment", "plan"}

# ── Mechanistic thresholds (Experiment 3 candidate selection) ──────────────────
COINCIDENTAL_CORRECTNESS_MLP_FRAC = 0.60   # Category A: MLP fraction > this
COINCIDENTAL_CORRECTNESS_LOOKBACK  = 0.30   # Category A: lookback ratio < this
PENALISED_INFERENCE_ACCURATE_MAX   = 3      # Category B: accurate score ≤ this

# ── Output ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
CACHE_DIR   = "cache"
