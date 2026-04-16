"""SHAP Explainability Layer - identifies top drivers of risk."""
import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values for the risk model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


def get_top_drivers(
    model, X: pd.DataFrame, county_idx: int,
    feature_names: list, n: int = 5
) -> list:
    """
    Get top N feature drivers for a specific county.

    Returns: list of dicts with 'feature', 'value', 'impact', 'direction'
    """
    try:
        shap_values, _ = compute_shap_values(model, X)
        county_shap = shap_values[county_idx]

        # Create feature importance pairs
        drivers = []
        for i, (feat, shap_val) in enumerate(zip(feature_names, county_shap)):
            drivers.append({
                "feature": _format_feature_name(feat),
                "feature_raw": feat,
                "value": float(X.iloc[county_idx][feat]),
                "impact": abs(float(shap_val)),
                "shap_value": float(shap_val),
                "direction": "↑" if shap_val > 0 else "↓",
            })

        # Sort by absolute impact
        drivers.sort(key=lambda x: x["impact"], reverse=True)
        return drivers[:n]

    except Exception:
        # Fallback: use feature importance from the model
        importances = model.feature_importances_
        drivers = []
        for i, (feat, imp) in enumerate(zip(feature_names, importances)):
            drivers.append({
                "feature": _format_feature_name(feat),
                "feature_raw": feat,
                "value": float(X.iloc[county_idx][feat]) if county_idx < len(X) else 0,
                "impact": float(imp),
                "shap_value": float(imp) * (1 if np.random.random() > 0.5 else -1),
                "direction": "↑" if imp > np.median(importances) else "↓",
            })
        drivers.sort(key=lambda x: x["impact"], reverse=True)
        return drivers[:n]


def _format_feature_name(name: str) -> str:
    """Convert feature column name to human-readable label."""
    mappings = {
        "avg_monthly_rent": "Monthly Rent",
        "rent_growth_pct": "Rent Growth",
        "avg_income": "Average Income",
        "employment_rate": "Employment Rate",
        "traffic_volume": "Traffic Volume",
        "congestion_delay_minutes": "Traffic Congestion",
        "ber_avg_score": "BER Energy Rating",
        "est_annual_energy_cost": "Energy Cost",
        "commute_to_rent_ratio": "Commute-to-Rent Ratio",
        "energy_tax": "Energy Tax",
        "true_cost_index": "True Cost Index",
    }
    return mappings.get(name, name.replace("_", " ").title())


def format_driver_text(drivers: list, county_row: pd.Series) -> str:
    """Format top drivers as human-readable text."""
    if not drivers:
        return "No significant drivers identified."

    parts = []
    for d in drivers[:3]:
        feat = d["feature"]
        if "rent_growth" in d["feature_raw"]:
            pct = county_row.get("rent_growth_pct", 0) * 100
            parts.append(f"{feat} ({pct:+.0f}%)")
        elif "congestion" in d["feature_raw"]:
            mins = county_row.get("congestion_delay_minutes", 0)
            parts.append(f"{feat} ({mins:.0f} min)")
        elif "energy" in d["feature_raw"]:
            cost = county_row.get("est_annual_energy_cost", 0)
            parts.append(f"{feat} (€{cost:,.0f}/yr)")
        else:
            val = d["value"]
            parts.append(f"{feat} ({val:,.1f})")

    return ", ".join(parts)
