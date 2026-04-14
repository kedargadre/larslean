"""
TOPSIS recommender module for the Léarslán ML scoring pipeline.

Ranks Electoral Divisions by multi-criteria closeness to user preferences.
"""

import logging

import numpy as np
import pandas as pd

from ml.feature_engineering import minmax_norm

logger = logging.getLogger(__name__)

_SLIDER_KEYS = [
    "slider_affordability",
    "slider_quality",
    "slider_transport",
    "slider_energy",
    "slider_jobs",
    "slider_stability",
]


def topsis_rank(df: pd.DataFrame, user_profile: dict) -> pd.DataFrame:
    """
    TOPSIS multi-criteria ranking for ED-level data.

    Args:
        df: Scored DataFrame at ED granularity (must contain 'ed_type' column).
        user_profile: Dict with keys:
            max_rent_budget, selected_area_types,
            slider_affordability, slider_quality, slider_transport,
            slider_energy, slider_jobs, slider_stability.

    Returns:
        Filtered DataFrame sorted by match_score descending.

    Raises:
        ValueError: if 'ed_type' column is absent (county-level data).
    """
    if "ed_type" not in df.columns:
        raise ValueError(
            "TOPSIS recommender requires ED-level data with ed_type column"
        )

    max_budget = user_profile["max_rent_budget"]
    selected_types = user_profile["selected_area_types"]

    # Hard filters
    mask = (df["avg_monthly_rent"] <= max_budget) & (df["ed_type"].isin(selected_types))
    filtered = df[mask].copy()

    if len(filtered) < 3:
        logger.warning(
            "Only %d areas match your filters. "
            "Consider relaxing Area Type or Budget.",
            len(filtered),
        )

    if filtered.empty:
        filtered["match_score"] = pd.Series(dtype=float)
        return filtered

    # Build 6-column decision matrix
    D = pd.DataFrame(index=filtered.index)
    D["affordability_score"] = filtered["affordability_score"].values
    D["livability_score"] = filtered["livability_score"].values
    D["transport_score"] = filtered["transport_score"].values
    D["ber_inv"] = (1 - minmax_norm(filtered["ber_avg_score"])).values
    D["employment_rate"] = filtered["employment_rate"].values
    D["stability"] = (100 - filtered["risk_score"]).values

    # Weight vector: w_i = max(slider_i, 1) / sum(max(slider_j, 1))
    raw_weights = np.array(
        [max(user_profile[k], 1) for k in _SLIDER_KEYS], dtype=float
    )
    W = raw_weights / raw_weights.sum()

    X = D.values.astype(float)

    # Step 1: Vector normalise each column
    col_norms = np.sqrt((X ** 2).sum(axis=0))
    # Avoid division by zero for zero-variance columns
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    R = X / col_norms

    # Step 2: Weighted normalised matrix
    V = R * W

    # Step 3 & 4: Ideal and anti-ideal solutions
    A_plus = V.max(axis=0)
    A_minus = V.min(axis=0)

    # Step 5 & 6: Euclidean distances
    D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

    # Step 7: Closeness coefficient × 100
    denom = D_plus + D_minus
    # Guard against zero denominator (all rows identical)
    denom = np.where(denom == 0, 1.0, denom)
    C = D_minus / denom * 100

    filtered["match_score"] = C
    return filtered.sort_values("match_score", ascending=False)
