"""Alert banner for anomaly detection results."""
import streamlit as st
import pandas as pd


def render_alert_banner(anomalies):
    if anomalies is None:
        return

    # detect_anomalies returns a DataFrame with anomaly_flag and anomaly_severity
    if isinstance(anomalies, pd.DataFrame):
        if "anomaly_flag" not in anomalies.columns:
            return
        flagged = anomalies[anomalies["anomaly_flag"] == -1]
        if flagged.empty:
            return
        high = flagged[flagged["anomaly_severity"] == "high"]
        if not high.empty:
            name_col = "ed_name" if "ed_name" in high.columns else "county"
            names = high[name_col].tolist()[:3]
            st.warning(f"⚠️ High-severity anomalies detected: **{', '.join(str(n) for n in names)}**")
        return

    # Legacy list format
    if isinstance(anomalies, list) and len(anomalies) > 0:
        for a in anomalies[:3]:
            if isinstance(a, dict):
                st.warning(f"⚠️ Anomaly: {a.get('area', 'Unknown')} — {a.get('message', '')}")
