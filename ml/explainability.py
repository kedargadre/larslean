"""
SHAP explainability module for the ML scoring pipeline.

Provides `explain_score` to surface the top-N features driving a GBM model's
prediction for a single area, using SHAP TreeExplainer.
"""

from __future__ import annotations

import pandas as pd
import shap


def explain_score(
    model,
    feature_vector: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """Return the top-N SHAP feature contributions for a single prediction.

    Parameters
    ----------
    model : GradientBoostingRegressor
        A trained GBM model (any of the 4 pipeline models).
    feature_vector : pd.DataFrame
        Single-row DataFrame containing the area's feature values.
    feature_names : list[str]
        Column names corresponding to the features.
    top_n : int, default 5
        Number of top features to return.

    Returns
    -------
    list[dict] sorted by abs(shap_value) descending, each entry:
        feature    : str   – column name
        value      : float – raw feature value
        shap_value : float – signed SHAP contribution
        impact     : float – abs(shap_value)
        direction  : str   – "↑" if shap_value > 0, else "↓"
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_vector)[0]  # shape: (n_features,)

    raw_values = feature_vector.iloc[0]

    entries = [
        {
            "feature": name,
            "value": float(raw_values[name]),
            "shap_value": float(sv),
            "impact": abs(float(sv)),
            "direction": "↑" if sv > 0 else "↓",
        }
        for name, sv in zip(feature_names, shap_values)
    ]

    entries.sort(key=lambda e: e["impact"], reverse=True)
    return entries[:top_n]


def get_top_drivers(model, X, row_idx, feature_names, n=5):
    """Compatibility wrapper for app.py. Returns top-N SHAP drivers for a single row."""
    row = X.iloc[[row_idx]]
    return explain_score(model, row, feature_names, top_n=n)
