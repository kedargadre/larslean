"""
Feature engineering module for the Léarslán ML scoring pipeline.

Transforms raw socioeconomic DataFrames into enriched DataFrames with 6 derived
features used downstream by the GBM scoring models. All operations are pure
pandas/numpy with safe division-by-zero handling.
"""

import numpy as np
import pandas as pd


def minmax_norm(series: pd.Series) -> pd.Series:
    """
    Min-max normalise a Series to [0, 1].

    Returns 0.5 for every element when max == min (zero-variance guard),
    preventing division-by-zero on constant columns.
    """
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (series - s_min) / (s_max - s_min)


def engineer_features(df: pd.DataFrame, daft_summaries=None) -> pd.DataFrame:
    """
    Compute 6 derived features and append them to the DataFrame.

    New columns added:
      - affordability_index      : (avg_income / 12) / avg_monthly_rent
      - rent_to_income_pct       : (avg_monthly_rent * 12) / avg_income * 100
      - commute_cost_monthly     : congestion_delay_minutes * 2 * 22
      - true_cost_index          : weighted composite of rent, energy, commute (0–100)
      - energy_tax               : avg_monthly_rent + (est_annual_energy_cost / 12)
      - commute_to_rent_ratio    : commute_cost_monthly / avg_monthly_rent

    All original columns are preserved.
    """
    out = df.copy()

    # Safe denominators
    rent_denom = out["avg_monthly_rent"].replace(0, 1)
    income_denom = out["avg_income"].replace(0, 1)

    out["affordability_index"] = (out["avg_income"] / 12) / rent_denom

    out["rent_to_income_pct"] = (out["avg_monthly_rent"] * 12) / income_denom * 100

    out["commute_cost_monthly"] = out["congestion_delay_minutes"] * 2 * 22

    out["true_cost_index"] = (
        0.50 * minmax_norm(out["avg_monthly_rent"])
        + 0.25 * minmax_norm(out["est_annual_energy_cost"])
        + 0.25 * minmax_norm(out["commute_cost_monthly"])
    ) * 100

    out["energy_tax"] = out["avg_monthly_rent"] + (out["est_annual_energy_cost"] / 12)

    out["commute_to_rent_ratio"] = out["commute_cost_monthly"] / rent_denom

    return out
