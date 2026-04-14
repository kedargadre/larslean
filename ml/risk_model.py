"""
GBM scoring module for the Léarslán ML scoring pipeline.

Trains 4 self-supervised GradientBoostingRegressor models (risk, livability,
transport, affordability), validates them, and assigns human-readable labels.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Expected top-3 features per model (for validation logging)
_EXPECTED_TOP3 = {
    "risk_score": ["rental_yield", "congestion_delay_minutes", "rent_to_income_pct"],
    "livability_score": ["employment_rate", "avg_income", "avg_monthly_rent"],
    "transport_score": ["congestion_delay_minutes", "traffic_volume", "employment_rate"],
    "affordability_score": ["affordability_index", "rent_to_income_pct", "avg_monthly_rent"],
}


def _create_targets(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, pd.Series]:
    """
    Generate self-supervised target scores using weighted formulas on MinMax-scaled features.

    Scales all feature_cols to [0, 1] then applies 4 weighted formulas:
      - risk_score
      - livability_score
      - transport_score
      - affordability_score

    All targets are clipped to [0, 100].

    Returns:
        dict mapping score name to target Series.
    """
    scaler = MinMaxScaler()
    scaled_arr = scaler.fit_transform(df[feature_cols].fillna(0))
    scaled = pd.DataFrame(scaled_arr, columns=feature_cols, index=df.index)

    def col(name: str) -> pd.Series:
        return scaled[name] if name in scaled.columns else pd.Series(0.0, index=df.index)

    risk = (
        col("rental_yield") * 20
        + col("congestion_delay_minutes") * 15
        + col("rent_to_income_pct") * 15
        + (1 - col("employment_rate")) * 15
        + col("ber_avg_score") * 10
        + col("est_annual_energy_cost") * 10
        + col("true_cost_index") * 15
    )

    livability = (
        col("employment_rate") * 20
        + col("avg_income") * 15
        + (1 - col("ber_avg_score")) * 15
        + (1 - col("avg_monthly_rent")) * 15
        + (1 - col("congestion_delay_minutes")) * 10
        + col("affordability_index") * 25
    )

    transport = (
        (1 - col("congestion_delay_minutes")) * 40
        + col("traffic_volume") * 30
        + col("employment_rate") * 15
        + (1 - col("ber_avg_score")) * 15
    )

    affordability = (
        col("affordability_index") * 35
        + (1 - col("rent_to_income_pct")) * 30
        + (1 - col("avg_monthly_rent")) * 20
        + (1 - col("est_annual_energy_cost")) * 15
    )

    return {
        "risk_score": risk.clip(0, 100),
        "livability_score": livability.clip(0, 100),
        "transport_score": transport.clip(0, 100),
        "affordability_score": affordability.clip(0, 100),
    }


def train_gbm_models(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict, list[str], dict[str, pd.Series]]:
    """
    Generate self-supervised targets, train 4 GBMs, predict and clip.

    Returns:
        (scored_df, models_dict, feature_names, targets_dict)
    """
    targets = _create_targets(df, feature_cols)
    X = df[feature_cols].fillna(0)

    scored_df = df.copy()
    models_dict: dict[str, GradientBoostingRegressor] = {}

    for score_name, y in targets.items():
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        preds = np.clip(model.predict(X), 0, 100).round(1)
        scored_df[score_name] = preds
        models_dict[score_name] = model

    return scored_df, models_dict, feature_cols, targets


def validate_models(
    models: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    targets: dict[str, pd.Series],
) -> None:
    """
    Run diagnostic checks on trained GBM models and log warnings for any issues.

    Checks per model:
      - Spearman ρ between target and prediction (warn if < 0.95)
      - Boundary saturation: % of rows at exactly 0.0 or 100.0 (warn if > 10%)
      - R²: warn if > 0.99 (overfitting) or < 0.90 (divergence)
      - Logs top-3 actual vs expected feature importances

    All checks are warnings only — never raises exceptions.
    """
    X = df[feature_cols].fillna(0)

    for score_name, model in models.items():
        y_true = targets[score_name]
        y_pred = df[score_name]

        # Spearman correlation
        rho, _ = spearmanr(y_true, y_pred)
        if rho < 0.95:
            logger.warning(
                "%s: Spearman ρ=%.3f is below 0.95 threshold — predictions may diverge from targets.",
                score_name, rho,
            )

        # Boundary saturation
        boundary_pct = ((y_pred == 0.0) | (y_pred == 100.0)).mean() * 100
        if boundary_pct > 10:
            logger.warning(
                "%s: %.1f%% of predictions are at boundary (0 or 100) — exceeds 10%% threshold.",
                score_name, boundary_pct,
            )

        # R² score
        r2 = model.score(X, y_true)
        if r2 > 0.99:
            logger.warning(
                "%s: R²=%.4f > 0.99 — possible overfitting.",
                score_name, r2,
            )
        elif r2 < 0.90:
            logger.warning(
                "%s: R²=%.4f < 0.90 — model may be diverging from targets.",
                score_name, r2,
            )

        # Feature importances
        importances = model.feature_importances_
        top3_idx = np.argsort(importances)[::-1][:3]
        top3_actual = [feature_cols[i] for i in top3_idx]
        top3_expected = _EXPECTED_TOP3.get(score_name, [])
        logger.info(
            "%s: top-3 features actual=%s | expected=%s",
            score_name, top3_actual, top3_expected,
        )


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable labels and trend to each scored row.

    Adds columns:
      - risk_label        : Low / Medium / High
      - affordability_label: Expensive / Moderate / Affordable
      - livability_label  : Poor / Fair / Good
      - transport_label   : Isolated / Moderate / Well-Connected
      - risk_trend        : Increasing / Stable / Decreasing

    Returns a copy of df with the 5 new columns appended.
    """
    out = df.copy()

    def _label(score: pd.Series, low: str, mid: str, high: str) -> pd.Series:
        return pd.cut(
            score,
            bins=[-np.inf, 33, 66, np.inf],
            labels=[low, mid, high],
        ).astype(str)

    out["risk_label"] = _label(out["risk_score"], "Low", "Medium", "High")
    out["affordability_label"] = _label(out["affordability_score"], "Expensive", "Moderate", "Affordable")
    out["livability_label"] = _label(out["livability_score"], "Poor", "Fair", "Good")
    out["transport_label"] = _label(out["transport_score"], "Isolated", "Moderate", "Well-Connected")

    out["risk_trend"] = np.select(
        [out["risk_score"] > 60, out["risk_score"] < 30],
        ["Increasing", "Decreasing"],
        default="Stable",
    )

    return out


# ── Compatibility functions used by app.py ─────────────────────────────────────

def train_risk_model(df: pd.DataFrame):
    """Wrapper for app.py compatibility. Trains GBMs and returns (models, scored_df, feature_names)."""
    from config import FEATURE_COLS
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include="number").columns
                        if c not in ("cluster", "umap_x", "umap_y", "anomaly_flag")]
    scored_df, models_dict, feature_names, _ = train_gbm_models(df, feature_cols)
    scored_df = assign_labels(scored_df)
    return models_dict, scored_df, feature_names


def get_risk_label(score: float) -> str:
    if score >= 67:
        return "High"
    if score >= 34:
        return "Medium"
    return "Low"


def get_affordability_label(score: float) -> str:
    if score >= 67:
        return "Affordable"
    if score >= 34:
        return "Moderate"
    return "Expensive"


def get_risk_trend(county: str, scores_df: pd.DataFrame) -> str:
    row = scores_df[scores_df["county"] == county]
    if row.empty:
        return "Stable"
    risk = row.iloc[0].get("risk_score", 50)
    if risk > 60:
        return "Increasing"
    if risk < 30:
        return "Decreasing"
    return "Stable"
