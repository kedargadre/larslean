"""Anomaly Detection: Isolation Forest for real-time cost-of-living alerts (county + ED level)."""
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

def detect_anomalies(scores_df: pd.DataFrame, contamination: float = None) -> list:
    """
    Detect statistical outliers using Isolation Forest.
    
    Automatically adjusts contamination based on dataset size:
    - County level (≤30 rows): 0.15 (expect ~4 anomalies)
    - ED level (>30 rows): 0.05 (expect ~13 anomalies from ~260)
    
    Args:
        scores_df: DataFrame containing the latest harmonized metrics.
        contamination: Proportion of outliers expected. Auto-detected if None.
        
    Returns:
        List of dicts with anomaly details and alert messages.
    """
    if scores_df is None or len(scores_df) < 5:
        return []

    # Auto-detect contamination
    if contamination is None:
        contamination = 0.15 if len(scores_df) <= 30 else 0.05

    features = [
        "avg_monthly_rent",
        "rent_growth_pct",
        "affordability_score",
        "risk_score"
    ]
    
    missing = [f for f in features if f not in scores_df.columns]
    if missing:
        logger.warning(f"Missing features for anomaly detection: {missing}")
        return []

    X = scores_df[features].fillna(scores_df[features].mean())

    clf = IsolationForest(
        n_estimators=100, 
        contamination=contamination, 
        random_state=42
    )

    preds = clf.fit_predict(X)
    scores_df["is_anomaly"] = preds
    
    anomalies = []
    
    nat_rent = scores_df["avg_monthly_rent"].mean()
    nat_growth = scores_df["rent_growth_pct"].mean()
    
    anomaly_df = scores_df[scores_df["is_anomaly"] == -1]
    
    # Determine if this is ED-level data
    is_ed_level = "ed_id" in scores_df.columns
    
    for _, row in anomaly_df.iterrows():
        # Use ED name if available, else county
        if is_ed_level:
            area_name = f"{row.get('ed_name', row.get('ed_id', 'Unknown'))} ({row['county']})"
        else:
            area_name = row["county"]
        
        rent = row["avg_monthly_rent"]
        growth = row["rent_growth_pct"]
        afford_score = row.get("affordability_score", 50)
        
        reasons = []
        severity = "medium"
        
        if rent > nat_rent * 1.5:
            reasons.append(f"Rent is critically high (€{rent:,.0f} vs avg €{nat_rent:,.0f})")
            severity = "high"
        elif rent < nat_rent * 0.5:
            reasons.append(f"Rent is unusually low — potential data issue or market distortion")
            severity = "low"
            
        if growth > 0.10:
            reasons.append(f"Severe rent growth shock: +{growth*100:.1f}%")
            severity = "high"
        
        if afford_score < 20:
            reasons.append(f"Affordability crisis detected (Score: {afford_score:.0f}/100)")
            severity = "high"
            
        if not reasons:
            reasons.append("Unusual combination of cost and risk factors detected")
            
        anomaly_entry = {
            "county": row["county"],
            "severity": severity,
            "reasons": reasons,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Add ED-specific fields
        if is_ed_level:
            anomaly_entry["ed_id"] = row.get("ed_id", "")
            anomaly_entry["ed_name"] = row.get("ed_name", "")
            anomaly_entry["area_name"] = area_name
        else:
            anomaly_entry["area_name"] = area_name
        
        anomalies.append(anomaly_entry)
        
    return anomalies
