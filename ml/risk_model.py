"""Risk Prediction Model V2 - GBM for Risk, Livability, Transport, and Affordability scores."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler


def _create_target_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target scores from raw features using a weighted formula.
    V2: adds Affordability Score.
    """
    df = df.copy()

    # Normalize individual features to [0, 1]
    scaler = MinMaxScaler()
    feature_cols = [
        "avg_monthly_rent", "rent_growth_pct", "congestion_delay_minutes",
        "ber_avg_score", "est_annual_energy_cost", "employment_rate",
        "avg_income", "traffic_volume",
    ]

    # V2 features
    v2_cols = ["affordability_index", "housing_pressure_index", "rent_to_income_pct", "supply_score"]
    for col in v2_cols:
        if col in df.columns:
            feature_cols.append(col)

    available_cols = [c for c in feature_cols if c in df.columns]
    scaled = pd.DataFrame(
        scaler.fit_transform(df[available_cols].fillna(0)),
        columns=available_cols,
        index=df.index,
    )

    # ── Risk Score (0-100, higher = more risk / worse) ─────────
    risk_components = (
        scaled.get("rent_growth_pct", 0) * 20 +
        scaled.get("congestion_delay_minutes", 0) * 15 +
        scaled.get("ber_avg_score", 0) * 10 +
        (1 - scaled.get("employment_rate", 0.5)) * 15 +
        scaled.get("est_annual_energy_cost", 0) * 10
    )
    # V2 components
    if "housing_pressure_index" in scaled.columns:
        risk_components = risk_components + scaled["housing_pressure_index"] * 15
    if "rent_to_income_pct" in scaled.columns:
        risk_components = risk_components + scaled["rent_to_income_pct"] * 15

    df["risk_score"] = np.clip(risk_components, 0, 100).round(1)

    # ── Livability Score (0-100, higher = more livable) ────────
    livability_components = (
        scaled.get("employment_rate", 0.5) * 20 +
        scaled.get("avg_income", 0.5) * 15 +
        (1 - scaled.get("ber_avg_score", 0.5)) * 15 +
        (1 - scaled.get("avg_monthly_rent", 0.5)) * 15 +
        (1 - scaled.get("congestion_delay_minutes", 0.5)) * 10
    )
    if "supply_score" in scaled.columns:
        livability_components = livability_components + scaled["supply_score"] * 15
    if "affordability_index" in scaled.columns:
        livability_components = livability_components + scaled["affordability_index"] * 10

    df["livability_score"] = np.clip(livability_components, 0, 100).round(1)

    # ── Transport Score (0-100, higher = better transport) ─────
    df["transport_score"] = np.clip(
        (
            (1 - scaled.get("congestion_delay_minutes", 0.5)) * 40 +
            scaled.get("traffic_volume", 0.5) * 30 +
            scaled.get("employment_rate", 0.5) * 15 +
            (1 - scaled.get("ber_avg_score", 0.5)) * 15
        ), 0, 100
    ).round(1)

    # ── Affordability Score (0-100, higher = more affordable) ──
    afford_components = 50.0  # default
    if "affordability_index" in scaled.columns:
        afford_components = (
            scaled["affordability_index"] * 35 +
            (1 - scaled.get("rent_to_income_pct", 0.5)) * 30 +
            (1 - scaled.get("avg_monthly_rent", 0.5)) * 20 +
            (1 - scaled.get("est_annual_energy_cost", 0.5)) * 15
        )
    else:
        afford_components = (
            (1 - scaled.get("avg_monthly_rent", 0.5)) * 40 +
            scaled.get("avg_income", 0.5) * 35 +
            (1 - scaled.get("est_annual_energy_cost", 0.5)) * 25
        )
    df["affordability_score"] = np.clip(afford_components, 0, 100).round(1)

    return df


def train_risk_model(df: pd.DataFrame) -> tuple:
    """
    Train GBM models for Risk, Livability, Transport, and Affordability scores.
    Returns: (models_dict, scored_df, feature_names)
    """
    # Create target scores
    df = _create_target_scores(df)

    # Define feature columns
    feature_cols = [
        "avg_monthly_rent", "rent_growth_pct", "avg_income",
        "employment_rate", "traffic_volume", "congestion_delay_minutes",
        "ber_avg_score", "est_annual_energy_cost",
    ]

    # Add engineered features if available
    for col in ["commute_to_rent_ratio", "energy_tax", "true_cost_index",
                 "affordability_index", "housing_pressure_index",
                 "rent_to_income_pct", "supply_score"]:
        if col in df.columns:
            feature_cols.append(col)

    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].fillna(0)

    models = {}
    for target in ["risk_score", "livability_score", "transport_score", "affordability_score"]:
        if target not in df.columns:
            continue
        y = df[target]
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        df[f"{target}_pred"] = np.clip(model.predict(X), 0, 100).round(1)
        models[target] = model

    return models, df, available_features


def get_risk_label(score: float) -> str:
    """Convert numeric risk score to label."""
    if score >= 67:
        return "High"
    elif score >= 34:
        return "Medium"
    else:
        return "Low"


def get_affordability_label(score: float) -> str:
    """Convert affordability score to label."""
    if score >= 67:
        return "Affordable"
    elif score >= 34:
        return "Moderate"
    else:
        return "Expensive"


def get_risk_trend(county: str, df: pd.DataFrame) -> str:
    """Determine if risk is increasing, stable, or decreasing."""
    row = df[df["county"] == county].iloc[0] if len(df[df["county"] == county]) > 0 else None
    if row is None:
        return "Stable"
    rent_growth = row.get("rent_growth_pct", 0)
    if rent_growth > 0.10:
        return "Increasing"
    elif rent_growth < 0.03:
        return "Decreasing"
    return "Stable"
