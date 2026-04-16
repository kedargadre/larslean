"""Feature Engineering V2 - Enhanced with affordability and housing pressure metrics."""
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame, daft_summaries: dict = None) -> pd.DataFrame:
    """
    Calculate derived features:
    - commute_to_rent_ratio: estimated commute cost / monthly rent
    - energy_tax: BER-adjusted housing cost (rent + monthly energy)
    - true_cost_index: composite of rent, energy, and commute costs (normalized 0-100)
    
    V2 additions (when daft_summaries provided):
    - affordability_index: (avg_income / 12) / median_daft_rent  (>1 = affordable)
    - rent_to_income_pct: (median_rent * 12) / avg_income * 100
    - housing_pressure_index: demand_proxy / listing_count  (high = scarce)
    - supply_score: normalized listing count (higher = more choice)
    - price_per_bedroom: median rent / avg bedrooms
    - live_median_rent: actual Daft median rent (overrides synthetic where available)
    """
    df = df.copy()

    # ── V1 features ────────────────────────────────────────────
    # Estimate monthly commute cost from congestion delay
    df["est_monthly_commute_cost"] = df["congestion_delay_minutes"] * 2 * 22

    # Commute-to-Rent Ratio
    df["commute_to_rent_ratio"] = np.where(
        df["avg_monthly_rent"] > 0,
        df["est_monthly_commute_cost"] / df["avg_monthly_rent"],
        0
    )

    # Energy Tax: rent + monthly energy cost
    df["monthly_energy_cost"] = df["est_annual_energy_cost"] / 12
    df["energy_tax"] = df["avg_monthly_rent"] + df["monthly_energy_cost"]

    # True Cost Index: normalized composite (0-100)
    rent_norm = _normalize(df["avg_monthly_rent"])
    energy_norm = _normalize(df["est_annual_energy_cost"])
    commute_norm = _normalize(df["est_monthly_commute_cost"])

    df["true_cost_index"] = (
        0.50 * rent_norm +
        0.25 * energy_norm +
        0.25 * commute_norm
    ) * 100

    # ── V2 features (Daft-enriched) ───────────────────────────
    if daft_summaries:
        live_medians = []
        listing_counts = []
        ppb_values = []
        sale_medians = []

        for _, row in df.iterrows():
            county = row["county"]
            summary = daft_summaries.get(county, {})
            
            live_median = summary.get("rental_median", 0)
            live_medians.append(live_median if live_median > 0 else row["avg_monthly_rent"])
            listing_counts.append(summary.get("rental_listing_count", 0))
            ppb_values.append(summary.get("rental_price_per_bedroom", 0))
            sale_medians.append(summary.get("sale_median", 0))

        df["live_median_rent"] = live_medians
        df["rental_listing_count"] = listing_counts
        df["price_per_bedroom"] = ppb_values
        df["sale_median_price"] = sale_medians

        # Affordability Index: >1 means median income covers rent
        rent_for_affordability = df["live_median_rent"]
        df["affordability_index"] = np.where(
            rent_for_affordability > 0,
            (df["avg_income"] / 12) / rent_for_affordability,
            0
        )

        # Rent-to-Income Percentage (30% threshold = stressed)
        df["rent_to_income_pct"] = np.where(
            df["avg_income"] > 0,
            (df["live_median_rent"] * 12) / df["avg_income"] * 100,
            0
        )

        # Housing Pressure Index: population proxy / available listings
        # Use traffic volume as demand proxy (higher traffic = more people)
        df["housing_pressure_index"] = np.where(
            df["rental_listing_count"] > 0,
            _normalize(df["traffic_volume"]) / _normalize(df["rental_listing_count"].clip(lower=1)),
            50
        )
        df["housing_pressure_index"] = df["housing_pressure_index"].clip(0, 100)

        # Supply Score: normalized listing count (higher = better)
        df["supply_score"] = _normalize(df["rental_listing_count"]) * 100

    else:
        # Fallback values when no Daft data
        df["live_median_rent"] = df["avg_monthly_rent"]
        df["rental_listing_count"] = 0
        df["price_per_bedroom"] = 0
        df["sale_median_price"] = 0
        df["affordability_index"] = np.where(
            df["avg_monthly_rent"] > 0,
            (df["avg_income"] / 12) / df["avg_monthly_rent"],
            0
        )
        df["rent_to_income_pct"] = np.where(
            df["avg_income"] > 0,
            (df["avg_monthly_rent"] * 12) / df["avg_income"] * 100,
            0
        )
        df["housing_pressure_index"] = 50
        df["supply_score"] = 50

    return df


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)
