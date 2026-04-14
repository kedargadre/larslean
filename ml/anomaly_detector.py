"""
Anomaly detection module for the Léarslán ML scoring pipeline.

Uses IsolationForest to flag anomalous areas and classifies severity.
"""

import logging

import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

_ANOMALY_FEATURES = [
    "avg_monthly_rent",
    "rental_yield",
    "affordability_score",
    "risk_score",
]


def detect_anomalies(df: pd.DataFrame, granularity: str = "auto") -> pd.DataFrame:
    out = df.copy()
    N = len(out)

    # Use only features that actually exist
    features = [f for f in _ANOMALY_FEATURES if f in out.columns]
    if len(features) < 2:
        out["anomaly_flag"] = 1
        out["anomaly_severity"] = "none"
        return out

    contamination = 0.15 if N <= 30 else 0.05

    iso = IsolationForest(contamination=contamination, random_state=42)
    out["anomaly_flag"] = iso.fit_predict(out[features].fillna(0).values)

    national_avg_rent = out["avg_monthly_rent"].mean() if "avg_monthly_rent" in out.columns else 1200

    def _severity(row: pd.Series) -> str:
        if row["anomaly_flag"] == 1:
            return "none"
        rent = row.get("avg_monthly_rent", 0)
        yield_ = row.get("rental_yield", 5.0)
        affordability = row.get("affordability_score", 50)
        if rent > 1.5 * national_avg_rent or yield_ < 3.0 or affordability < 20:
            return "high"
        if rent < 0.5 * national_avg_rent:
            return "low"
        return "medium"

    out["anomaly_severity"] = out.apply(_severity, axis=1)

    return out
