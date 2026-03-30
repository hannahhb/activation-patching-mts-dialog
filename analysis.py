"""
Experiment 2 — Statistical analyses linking mechanistic features to PDSQI-9 scores.

Two analyses:
  A. Mechanistic Signature Matrix   — Pearson r between DLA band features and
                                       PDSQI-9 attributes across encounters.
  B. Section-level ANOVA            — Does SOAP section type predict each
                                       mechanistic feature?  Correlate deviations
                                       with per-section Accuracy scores.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from config import PDSQI9_ATTRIBUTES


# ── A. Mechanistic Signature Matrix ───────────────────────────────────────────

MECH_BAND_FEATURES = [
    "attn_contribution_early",
    "attn_contribution_mid",
    "attn_contribution_late",
    "mlp_contribution_early",
    "mlp_contribution_mid",
    "mlp_contribution_late",
]

MECH_RATIO_FEATURES = [
    "attn_fraction",
    "mlp_fraction",
    "lookback_ratio",
    "source_attention_entropy",
    "extractive_score",
]


def build_signature_matrix(
    encounter_features: List[Dict],
    pdsqi9_scores:      List[Dict],
    min_n: int = 3,
) -> pd.DataFrame:
    """
    Build a (n_attributes × n_mech_features) Pearson correlation matrix.

    Args:
        encounter_features : list of per-encounter aggregate feature dicts
                             (output of mechanistic.aggregate_encounter_features)
        pdsqi9_scores      : list of per-encounter full-note PDSQI-9 score dicts

    Returns:
        DataFrame indexed by PDSQI-9 attribute, columns = mechanistic features.
        Values are Pearson r coefficients.
    """
    feat_df  = pd.DataFrame(encounter_features)
    score_df = pd.DataFrame(pdsqi9_scores)

    all_features = MECH_BAND_FEATURES + MECH_RATIO_FEATURES
    all_features = [f for f in all_features if f in feat_df.columns]

    corr_rows = {}
    for attr in PDSQI9_ATTRIBUTES:
        if attr not in score_df.columns:
            continue
        y = pd.to_numeric(score_df[attr], errors="coerce")
        row = {}
        for feat in all_features:
            x = pd.to_numeric(feat_df[feat], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < min_n:
                row[feat] = np.nan
            else:
                r, _ = stats.pearsonr(x[mask], y[mask])
                row[feat] = round(r, 3)
        corr_rows[attr] = row

    return pd.DataFrame(corr_rows).T


def signature_matrix_pvalues(
    encounter_features: List[Dict],
    pdsqi9_scores:      List[Dict],
    min_n: int = 3,
) -> pd.DataFrame:
    """Same shape as build_signature_matrix but values are p-values."""
    feat_df  = pd.DataFrame(encounter_features)
    score_df = pd.DataFrame(pdsqi9_scores)

    all_features = MECH_BAND_FEATURES + MECH_RATIO_FEATURES
    all_features = [f for f in all_features if f in feat_df.columns]

    pval_rows = {}
    for attr in PDSQI9_ATTRIBUTES:
        if attr not in score_df.columns:
            continue
        y = pd.to_numeric(score_df[attr], errors="coerce")
        row = {}
        for feat in all_features:
            x = pd.to_numeric(feat_df[feat], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < min_n:
                row[feat] = np.nan
            else:
                _, p = stats.pearsonr(x[mask], y[mask])
                row[feat] = round(p, 4)
        pval_rows[attr] = row

    return pd.DataFrame(pval_rows).T


# ── B. Section-level ANOVA + mechanistic deviations ──────────────────────────

def section_anova(token_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    For each mechanistic feature, run a one-way ANOVA across SOAP section types
    pooled over all encounters.

    Args:
        token_dfs : list of per-token DataFrames (one per encounter)

    Returns:
        DataFrame with columns [feature, F_stat, p_value, eta_sq] indexed by feature.
    """
    combined = pd.concat(token_dfs, ignore_index=True)
    features = MECH_BAND_FEATURES + MECH_RATIO_FEATURES
    features = [f for f in features if f in combined.columns]

    rows = []
    for feat in features:
        groups = [
            grp[feat].dropna().values
            for _, grp in combined.groupby("section")
            if len(grp[feat].dropna()) >= 3
        ]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        # Eta-squared: SS_between / SS_total
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total = ((all_vals - grand_mean) ** 2).sum()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2 for g in groups
        )
        eta_sq = ss_between / (ss_total + 1e-10)
        rows.append({
            "feature": feat,
            "F_stat":  round(F, 3),
            "p_value": round(p, 4),
            "eta_sq":  round(eta_sq, 3),
        })

    if not rows:
        return pd.DataFrame(columns=["F_stat", "p_value", "eta_sq"])
    return pd.DataFrame(rows).set_index("feature")


def section_mechanistic_deviations(
    token_dfs:    List[pd.DataFrame],
    pdsqi9_scores: List[Dict],
    feature:      str = "attn_fraction",
    accuracy_key: str = "accurate",
) -> pd.DataFrame:
    """
    Compute per-section mechanistic deviation (observed − section-type mean)
    and correlate with per-section Accuracy score.

    Returns a long DataFrame: columns [encounter_idx, section, deviation, accuracy].
    """
    combined = pd.concat(
        [df.assign(encounter=i) for i, df in enumerate(token_dfs)],
        ignore_index=True,
    )

    # Section-type expected means (pooled over all encounters)
    section_means = (
        combined.groupby("section")[feature]
        .mean()
        .to_dict()
    )

    rows = []
    for enc_i, df in enumerate(token_dfs):
        if enc_i >= len(pdsqi9_scores):
            continue
        scores = pdsqi9_scores[enc_i]
        if "sections" not in scores:
            continue
        for sec_name, sec_scores in scores["sections"].items():
            if accuracy_key not in sec_scores:
                continue
            sub     = df[df["section"] == sec_name]
            if sub.empty:
                continue
            obs     = sub[feature].mean()
            exp     = section_means.get(sec_name, obs)
            rows.append({
                "encounter_idx": enc_i,
                "section":       sec_name,
                "deviation":     obs - exp,
                accuracy_key:    sec_scores[accuracy_key],
            })

    return pd.DataFrame(rows)
