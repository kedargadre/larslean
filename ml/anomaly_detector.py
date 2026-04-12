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


def detect_anomalies(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Detect anomalous areas using IsolationForest.

    Args:
        df: Scored DataFrame containing the 4 anomaly features.
        granularity: Determines contamination rate.
                     N≤30 (county) → 0.15; N>30 (ED) → 0.05.

    Returns:
        DataFrame with two new columns:
          - anomaly_flag     : 1 = normal, -1 = anomalous
          - anomaly_severity : "none" | "high" | "medium" | "low"
    """
    out = df.copy()
    N = len(out)

    contamination = 0.15 if N <= 30 else 0.05

    iso = IsolationForest(contamination=contamination, random_state=42)
    out["anomaly_flag"] = iso.fit_predict(out[_ANOMALY_FEATURES].values)

    national_avg_rent = out["avg_monthly_rent"].mean()

    def _severity(row: pd.Series) -> str:
        if row["anomaly_flag"] == 1:
            return "none"
        rent = row["avg_monthly_rent"]
        yield_ = row["rental_yield"]
        affordability = row["affordability_score"]
        if rent > 1.5 * national_avg_rent or yield_ < 3.0 or affordability < 20:
            return "high"
        if rent < 0.5 * national_avg_rent:
            return "low"
        return "medium"

    out["anomaly_severity"] = out.apply(_severity, axis=1)

    return out
