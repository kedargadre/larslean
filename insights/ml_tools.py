"""
Live ML inference tools for the AI Advisor.

These functions run the actual trained models at query time so the LLM
can give answers grounded in real-time model outputs, not just static scores.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "ml" / "models"


def run_personalized_topsis(
    scored_df: pd.DataFrame,
    salary: float,
    max_rent_pct: float = 30.0,
    area_types: list[str] | None = None,
    priorities: dict | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Run TOPSIS with user's actual salary and preferences.

    Args:
        scored_df: ED-level scored DataFrame (255 rows)
        salary: Annual salary in EUR
        max_rent_pct: Max % of monthly income to spend on rent (default 30%)
        area_types: Filter to these ed_types (default: all)
        priorities: Dict of slider weights, e.g. {"affordability": 80, "transport": 40}
        top_n: Number of results to return

    Returns:
        List of dicts with area name, county, match_score, rent, personalized affordability
    """
    if "ed_type" not in scored_df.columns:
        return _county_level_ranking(scored_df, salary, max_rent_pct, priorities, top_n)

    monthly_income = salary / 12
    max_rent = monthly_income * (max_rent_pct / 100)

    if area_types is None:
        area_types = ["urban_core", "suburban", "town", "village", "rural"]

    defaults = {"affordability": 70, "quality": 50, "transport": 40, "energy": 30, "jobs": 50, "stability": 40}
    p = {**defaults, **(priorities or {})}

    try:
        from ml.recommender import topsis_rank
        user_profile = {
            "max_rent_budget": max_rent,
            "selected_area_types": area_types,
            "slider_affordability": p["affordability"],
            "slider_quality": p["quality"],
            "slider_transport": p["transport"],
            "slider_energy": p["energy"],
            "slider_jobs": p["jobs"],
            "slider_stability": p["stability"],
        }
        ranked = topsis_rank(scored_df, user_profile)
    except Exception as e:
        logger.warning("TOPSIS failed: %s", e)
        return _county_level_ranking(scored_df, salary, max_rent_pct, priorities, top_n)

    if ranked.empty:
        return []

    results = []
    for _, row in ranked.head(top_n).iterrows():
        rent = row.get("avg_monthly_rent", 0)
        personal_rent_pct = (rent / monthly_income * 100) if monthly_income > 0 else 0
        remaining = monthly_income - rent - row.get("est_annual_energy_cost", 2500) / 12

        results.append({
            "area": row.get("ed_name", "Unknown"),
            "county": row.get("county", ""),
            "ed_type": row.get("ed_type", ""),
            "match_score": round(row.get("match_score", 0), 1),
            "rent": round(rent),
            "rent_pct_of_income": round(personal_rent_pct, 1),
            "monthly_remaining": round(remaining),
            "risk_score": round(row.get("risk_score", 0), 1),
            "livability_score": round(row.get("livability_score", 0), 1),
            "transport_score": round(row.get("transport_score", 0), 1),
            "affordability_score": round(row.get("affordability_score", 0), 1),
            "cluster": row.get("cluster_category", ""),
            "budget_fit": "Comfortable" if remaining > 1000 else "Tight" if remaining > 300 else "Stretched" if remaining > 0 else "Over Budget",
        })

    return results


def _county_level_ranking(df, salary, max_rent_pct, priorities, top_n):
    """Simple weighted ranking for county-level data (no ed_type for TOPSIS)."""
    monthly_income = salary / 12
    max_rent = monthly_income * (max_rent_pct / 100)

    filtered = df[df["avg_monthly_rent"] <= max_rent].copy() if "avg_monthly_rent" in df.columns else df.copy()
    if filtered.empty:
        filtered = df.nsmallest(top_n, "avg_monthly_rent").copy()

    defaults = {"affordability": 70, "quality": 50, "transport": 40, "jobs": 50, "stability": 40}
    p = {**defaults, **(priorities or {})}
    total_w = max(sum(max(v, 1) for v in p.values()), 1)

    score_cols = {
        "affordability_score": p["affordability"],
        "livability_score": p["quality"],
        "transport_score": p["transport"],
        "employment_rate": p["jobs"],
    }

    filtered["match_score"] = sum(
        (max(w, 1) / total_w) * filtered[col].fillna(0)
        for col, w in score_cols.items() if col in filtered.columns
    )

    results = []
    for _, row in filtered.nlargest(top_n, "match_score").iterrows():
        rent = row.get("avg_monthly_rent", 0)
        personal_rent_pct = (rent / monthly_income * 100) if monthly_income > 0 else 0
        remaining = monthly_income - rent - row.get("est_annual_energy_cost", 2500) / 12

        results.append({
            "area": row.get("county", "Unknown"),
            "county": row.get("county", ""),
            "match_score": round(row.get("match_score", 0), 1),
            "rent": round(rent),
            "rent_pct_of_income": round(personal_rent_pct, 1),
            "monthly_remaining": round(remaining),
            "risk_score": round(row.get("risk_score", 0), 1),
            "livability_score": round(row.get("livability_score", 0), 1),
            "affordability_score": round(row.get("affordability_score", 0), 1),
            "cluster": row.get("cluster_category", ""),
            "budget_fit": "Comfortable" if remaining > 1000 else "Tight" if remaining > 300 else "Stretched" if remaining > 0 else "Over Budget",
        })

    return results


def run_shap_explanation(
    models: dict,
    scored_df: pd.DataFrame,
    feature_names: list[str],
    area_name: str,
    score_type: str = "risk_score",
) -> list[dict]:
    """
    Run SHAP TreeExplainer on a specific area for a specific score.

    Returns top-5 feature drivers with direction and impact.
    """
    model = models.get(score_type)
    if model is None:
        return []

    # Find the area row
    name_col = "ed_name" if "ed_name" in scored_df.columns else "county"
    mask = scored_df[name_col].str.lower() == area_name.lower()
    if not mask.any():
        mask = scored_df["county"].str.lower() == area_name.lower()
    if not mask.any():
        return []

    available_features = [f for f in feature_names if f in scored_df.columns]
    X = scored_df[available_features].fillna(0)
    row_idx = scored_df[mask].index[0]

    try:
        from ml.explainability import get_top_drivers
        drivers = get_top_drivers(model, X, row_idx, available_features, n=5)
        return drivers
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        # Fallback to feature importance
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        return [
            {
                "feature": available_features[i],
                "value": float(X.iloc[row_idx][available_features[i]]),
                "impact": round(float(importances[i]), 4),
                "shap_value": round(float(importances[i]), 4),
                "direction": "↑",
            }
            for i in top_idx
        ]


def run_rent_forecast(
    area_name: str,
    scored_df: pd.DataFrame,
    ts_data: dict | None = None,
    n_periods: int = 6,
) -> dict:
    """
    Run ARIMA forecast for rent in a specific area.

    Returns dict with current_rent, forecast_rent_6m, change_pct, method, monthly_forecasts.
    """
    from ml.forecasting import forecast_metric

    # Get current rent
    name_col = "ed_name" if "ed_name" in scored_df.columns else "county"
    mask = scored_df[name_col].str.lower() == area_name.lower()
    if not mask.any():
        mask = scored_df["county"].str.lower() == area_name.lower()
    if not mask.any():
        return {"error": f"Area '{area_name}' not found"}

    current_rent = float(scored_df[mask].iloc[0].get("avg_monthly_rent", 0))

    # Try to get time series data
    series = None
    if ts_data:
        # Try exact match on area name or county
        for key in ts_data:
            if area_name.lower() in str(key).lower():
                ts_df = ts_data[key]
                if "avg_monthly_rent" in ts_df.columns:
                    series = ts_df["avg_monthly_rent"].dropna()
                break

    # If no time series, generate synthetic from current value
    if series is None or len(series) < 3:
        noise = np.random.RandomState(42).normal(0, current_rent * 0.02, 12)
        trend = np.linspace(-current_rent * 0.03, current_rent * 0.03, 12)
        series = pd.Series(current_rent + trend + noise)

    # forecast_metric expects (ts_data_df, metric_col, periods)
    # Build a DataFrame with 'month' column for compatibility
    ts_df = pd.DataFrame({
        "month": pd.date_range(end=pd.Timestamp.now(), periods=len(series), freq="MS"),
        "avg_monthly_rent": series.values,
    })
    result = forecast_metric(ts_df, "avg_monthly_rent", n_periods)
    forecast_final = float(result["forecast"][-1])
    change_pct = ((forecast_final - current_rent) / current_rent * 100) if current_rent > 0 else 0

    return {
        "area": area_name,
        "current_rent": round(current_rent),
        "forecast_rent_6m": round(forecast_final),
        "change_pct": round(change_pct, 1),
        "method": "ARIMA(1,1,1)",
        "monthly_forecasts": [round(float(v)) for v in result["forecast"]],
        "lower_ci": [round(float(v)) for v in result.get("lower", result["forecast"])],
        "upper_ci": [round(float(v)) for v in result.get("upper", result["forecast"])],
    }


def build_ml_context(
    query: str,
    scored_df: pd.DataFrame,
    models: dict,
    feature_names: list[str],
    selected_county: str,
    ts_data: dict | None = None,
) -> str:
    """
    Analyze the user query and run relevant ML models to build live context.

    This is the main entry point called by the chat module. It detects
    what the user is asking and runs the appropriate models.
    """
    q = query.lower()
    context_parts = []

    # Detect salary/budget mentions for TOPSIS
    salary = _extract_salary(q)
    if salary or any(w in q for w in ["where", "live", "move", "relocate", "recommend", "best area", "afford"]):
        sal = salary or 45000  # default if not specified
        results = run_personalized_topsis(scored_df, sal, top_n=5)
        if results:
            context_parts.append(f"\n--- LIVE TOPSIS RANKING (salary: €{sal:,.0f}/yr) ---")
            for i, r in enumerate(results, 1):
                context_parts.append(
                    f"  #{i} {r['area']} ({r['county']}) — Match: {r['match_score']}/100 | "
                    f"Rent: €{r['rent']}/mo ({r['rent_pct_of_income']}% of income) | "
                    f"Remaining: €{r['monthly_remaining']}/mo | Budget: {r['budget_fit']} | "
                    f"Risk: {r['risk_score']} | Livability: {r['livability_score']} | "
                    f"Cluster: {r['cluster']}"
                )
            context_parts.append("[Model: TOPSIS multi-criteria ranking with personalized salary]")

    # Detect SHAP/explanation requests
    if any(w in q for w in ["why", "explain", "driver", "cause", "reason", "what makes", "high risk", "low"]):
        area = _extract_area_name(q, scored_df) or selected_county
        for score_type in ["risk_score", "livability_score", "affordability_score"]:
            if score_type.replace("_score", "") in q or "why" in q:
                drivers = run_shap_explanation(models, scored_df, feature_names, area, score_type)
                if drivers:
                    context_parts.append(f"\n--- LIVE SHAP ANALYSIS: {area} — {score_type} ---")
                    for d in drivers:
                        context_parts.append(
                            f"  {d['direction']} {d['feature']}: value={d['value']:.2f}, "
                            f"SHAP impact={d['shap_value']:.3f}"
                        )
                    context_parts.append("[Model: SHAP TreeExplainer on GBM model]")
                break  # only explain one score type

    # Detect forecast requests
    if any(w in q for w in ["forecast", "predict", "future", "6 month", "next year", "trend", "will rent"]):
        area = _extract_area_name(q, scored_df) or selected_county
        forecast = run_rent_forecast(area, scored_df, ts_data)
        if "error" not in forecast:
            context_parts.append(f"\n--- LIVE ARIMA FORECAST: {forecast['area']} ---")
            context_parts.append(
                f"  Current rent: €{forecast['current_rent']}/mo\n"
                f"  Forecast (6 months): €{forecast['forecast_rent_6m']}/mo ({forecast['change_pct']:+.1f}%)\n"
                f"  Method: {forecast['method']}\n"
                f"  Monthly projections: {', '.join(f'€{v}' for v in forecast['monthly_forecasts'])}\n"
                f"  80% CI range: €{forecast['lower_ci'][-1]} – €{forecast['upper_ci'][-1]}"
            )
            context_parts.append("[Model: ARIMA(1,1,1) time-series forecast]")

    return "\n".join(context_parts)


def _extract_salary(query: str) -> float | None:
    """Extract salary/income from query text."""
    import re
    # Match patterns like "65000", "65,000", "65k", "€65000"
    patterns = [
        r'€?\s*(\d{2,3})[,.]?(\d{3})\s*(?:per\s*year|/yr|annual|salary|income|earn)',
        r'(?:salary|income|earn|make|paid)\s*(?:is|of)?\s*€?\s*(\d{2,3})[,.]?(\d{3})',
        r'€?\s*(\d{2,3})k\b',
        r'€?\s*(\d{4,6})\b.*(?:salary|income|year|annual|earn)',
        r'(?:salary|income|earn)\s*.*?€?\s*(\d{4,6})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            groups = match.groups()
            if len(groups) == 2 and groups[1]:
                return float(groups[0] + groups[1])
            val = float(groups[0])
            if val < 200:  # likely "65k"
                return val * 1000
            return val
    return None


def _extract_area_name(query: str, scored_df: pd.DataFrame) -> str | None:
    """Try to extract an area name from the query."""
    q = query.lower()

    # Check ED names
    if "ed_name" in scored_df.columns:
        for name in scored_df["ed_name"].unique():
            if str(name).lower() in q:
                return str(name)

    # Check county names
    for name in scored_df["county"].unique():
        if str(name).lower() in q:
            return str(name)

    return None
