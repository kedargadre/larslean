"""UI: Anomaly Alert Banner for the top of the dashboard."""
import streamlit as st

def render_alert_banner(anomalies: list):
    """Render a global alert banner if critical anomalies are found."""
    if not anomalies:
        return
        
    # Check for high severity anomalies
    high_sev = [a for a in anomalies if a["severity"] == "high"]
    if not high_sev:
        return
        
    st.markdown("""
        <style>
            .alert-banner {
                background-color: rgba(239, 68, 68, 0.1);
                border-left: 4px solid #ef4444;
                padding: 16px;
                border-radius: 4px;
                margin-bottom: 24px;
                border-top: 1px solid rgba(239, 68, 68, 0.2);
                border-right: 1px solid rgba(239, 68, 68, 0.2);
                border-bottom: 1px solid rgba(239, 68, 68, 0.2);
            }
            .alert-title {
                color: #ef4444;
                font-weight: 800;
                margin-bottom: 8px;
                font-family: 'Inter', sans-serif;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .alert-text {
                color: #f8fafc;
                font-size: 0.9rem;
                margin: 0;
                padding-left: 20px;
            }
            .alert-text li {
                margin-bottom: 4px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    html = '<div class="alert-banner">'
    html += '<div class="alert-title">🚨 MARKET SHOCK DETECTED</div>'
    html += '<div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 8px; padding-left: 32px;">'
    html += 'Isolation Forest algorithm detected extreme statistical outliers in live cost-of-living data.'
    html += '</div>'
    html += '<ul class="alert-text">'
    for anomaly in high_sev:
        county = anomaly["county"]
        reason_str = "; ".join(anomaly["reasons"])
        html += f"<li><strong>{county}</strong>: {reason_str}</li>"
    html += '</ul></div>'
    
    st.markdown(html, unsafe_allow_html=True)
