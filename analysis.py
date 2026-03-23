"""
Experiment 2 — Statistical analyses linking mechanistic features to PDSQI-9 scores.

Three analyses:
  A. Mechanistic Signature Matrix   — Pearson r between DLA band features and
                                       PDSQI-9 attributes across encounters.
  B. Section-level ANOVA            — Does SOAP section type predict each
                                       mechanistic feature?  Correlate deviations
                                       with per-section Accuracy scores.
  C. Input complexity interaction   — Two-way ANOVA: complexity × section type
                                       → MLP contribution.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    PDSQI9_ATTRIBUTES, RETRIEVAL_ATTRS, SYNTHESIS_ATTRS,
    EXTRACTIVE_SECTIONS, ABSTRACTIVE_SECTIONS,
)


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
            if mask.sum() < 5:
                row[feat] = np.nan
            else:
                r, _ = stats.pearsonr(x[mask], y[mask])
                row[feat] = round(r, 3)
        corr_rows[attr] = row

    return pd.DataFrame(corr_rows).T


def signature_matrix_pvalues(
    encounter_features: List[Dict],
    pdsqi9_scores:      List[Dict],
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
            if mask.sum() < 5:
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


# ── C. Input complexity interaction ───────────────────────────────────────────

def complexity_interaction_anova(
    encounter_features: List[Dict],
    pdsqi9_scores:      List[Dict],
    mech_feature:       str = "mlp_contribution_mid",
    complexity_key:     str = "entity_density",
) -> Dict:
    """
    Two-way ANOVA: complexity (high/low, median-split) × section_type
    (extractive vs abstractive) → mean MLP contribution.

    Returns dict with F-stats and p-values for main effects and interaction.
    """
    feat_df = pd.DataFrame(encounter_features)
    if complexity_key not in feat_df.columns:
        return {"error": f"{complexity_key} not in features"}

    # Median-split on complexity
    median_complexity = feat_df[complexity_key].median()
    feat_df["complexity_group"] = (
        feat_df[complexity_key] > median_complexity
    ).map({True: "high", False: "low"})

    # Extractive vs abstractive section means per encounter
    # Pull from per-section mlp contributions if available
    rows = []
    for i, ef in enumerate(encounter_features):
        cgroup = feat_df.loc[i, "complexity_group"]
        for sec in EXTRACTIVE_SECTIONS:
            key = f"mlp_frac_{sec}"
            if key in ef:
                rows.append({"complexity": cgroup, "section_type": "extractive", "value": ef[key]})
        for sec in ABSTRACTIVE_SECTIONS:
            key = f"mlp_frac_{sec}"
            if key in ef:
                rows.append({"complexity": cgroup, "section_type": "abstractive", "value": ef[key]})

    if not rows:
        return {"error": "no per-section mlp features found in encounter_features"}

    df = pd.DataFrame(rows).dropna()

    # Simple 2×2 ANOVA via OLS (statsmodels) or fallback to F-tests
    try:
        import statsmodels.formula.api as smf
        model  = smf.ols("value ~ C(complexity) * C(section_type)", data=df).fit()
        anova  = sm_anova(model)
        return {
            "main_complexity":   {
                "F": round(anova.loc["C(complexity)",    "F"], 3),
                "p": round(anova.loc["C(complexity)",    "PR(>F)"], 4),
            },
            "main_section_type": {
                "F": round(anova.loc["C(section_type)",  "F"], 3),
                "p": round(anova.loc["C(section_type)",  "PR(>F)"], 4),
            },
            "interaction":       {
                "F": round(anova.loc["C(complexity):C(section_type)", "F"], 3),
                "p": round(anova.loc["C(complexity):C(section_type)", "PR(>F)"], 4),
            },
        }
    except ImportError:
        # Fallback: manual F-tests for each effect
        return _manual_two_way_anova(df)


def sm_anova(model):
    from statsmodels.stats.anova import anova_lm
    return anova_lm(model, typ=2)


def _manual_two_way_anova(df: pd.DataFrame) -> Dict:
    """Fallback two-way ANOVA without statsmodels."""
    high_ext  = df[(df.complexity == "high") & (df.section_type == "extractive")]["value"]
    high_abs  = df[(df.complexity == "high") & (df.section_type == "abstractive")]["value"]
    low_ext   = df[(df.complexity == "low")  & (df.section_type == "extractive")]["value"]
    low_abs   = df[(df.complexity == "low")  & (df.section_type == "abstractive")]["value"]

    F_main_complexity, p_main_complexity = stats.f_oneway(
        pd.concat([high_ext, high_abs]), pd.concat([low_ext, low_abs])
    )
    F_main_section, p_main_section = stats.f_oneway(
        pd.concat([high_ext, low_ext]), pd.concat([high_abs, low_abs])
    )
    F_inter, p_inter = stats.f_oneway(high_ext - high_abs, low_ext - low_abs)

    return {
        "main_complexity":   {"F": round(F_main_complexity, 3), "p": round(p_main_complexity, 4)},
        "main_section_type": {"F": round(F_main_section,    3), "p": round(p_main_section,    4)},
        "interaction":       {"F": round(F_inter,            3), "p": round(p_inter,            4)},
        "note": "fallback manual F-tests; install statsmodels for proper two-way ANOVA",
    }


def complexity_correlation_table(
    encounter_features: List[Dict],
    pdsqi9_scores:      List[Dict],
    complexity_keys:    Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Pearson correlations between input complexity features and both mechanistic
    features and PDSQI-9 scores.
    """
    if complexity_keys is None:
        complexity_keys = [
            "entity_density", "hedge_density", "negation_density",
            "speaker_turns", "source_len_tokens", "type_token_ratio",
        ]

    feat_df  = pd.DataFrame(encounter_features)
    score_df = pd.DataFrame(pdsqi9_scores)

    target_cols = (
        [f for f in MECH_BAND_FEATURES if f in feat_df.columns]
        + PDSQI9_ATTRIBUTES
    )

    rows = {}
    for ck in complexity_keys:
        if ck not in feat_df.columns:
            continue
        x = pd.to_numeric(feat_df[ck], errors="coerce")
        row = {}
        for tc in target_cols:
            y = (
                pd.to_numeric(feat_df[tc],  errors="coerce")
                if tc in feat_df.columns
                else pd.to_numeric(score_df[tc], errors="coerce")
                if tc in score_df.columns
                else pd.Series(dtype=float)
            )
            mask = x.notna() & y.notna()
            if mask.sum() < 5:
                row[tc] = np.nan
            else:
                r, _ = stats.pearsonr(x[mask], y[mask])
                row[tc] = round(r, 3)
        rows[ck] = row

    return pd.DataFrame(rows).T
