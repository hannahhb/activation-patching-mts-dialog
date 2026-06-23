"""
ecs_pks.py — compatibility shim
================================
This file is retained for backward compatibility only.
All symbols are now defined in the new modules:

    config.py       — Config, Q_COLORS, load_aci_sample
    tokenization.py — tokenize_pair, tokenize_as_generated, generate_note
    metrics.py      — ECS/PKS computation and statistical metrics
    plots.py        — matplotlib/seaborn visualisation helpers

Any code that does ``from ecs_pks import X`` will continue to work.
"""

# ruff: noqa: F401, F403

from sae_experiments.old.config import Config, Q_COLORS, load_aci_sample

from sae_experiments.old.tokenization import tokenize_pair, tokenize_as_generated, generate_note

from metrics import (
    compute_ecs,
    compute_pks,
    compute_ecs_pks,
    compute_dla,
    hallucination_risk,
    quadrant_stats,
    layer_discriminability,
    dla_discriminability,
    identify_knowledge_ffns,
    identify_copy_head_layers,
    fit_hallucination_regressor,
    calibrate_layer_thresholds,
    apply_layer_thresholds,
)

from sae_experiments.old.plots import (
    plot_scatter,
    plot_heatmap,
    plot_risk_bar,
    plot_layer_discriminability,
)

__all__ = [
    # config
    "Config",
    "Q_COLORS",
    "load_aci_sample",
    # tokenization
    "tokenize_pair",
    "tokenize_as_generated",
    "generate_note",
    # metrics
    "compute_ecs",
    "compute_pks",
    "compute_ecs_pks",
    "compute_dla",
    "hallucination_risk",
    "quadrant_stats",
    "layer_discriminability",
    "dla_discriminability",
    "identify_knowledge_ffns",
    "identify_copy_head_layers",
    "fit_hallucination_regressor",
    "calibrate_layer_thresholds",
    "apply_layer_thresholds",
    # plots
    "plot_scatter",
    "plot_heatmap",
    "plot_risk_bar",
    "plot_layer_discriminability",
]
