"""Drill-down Side Panel for selected county."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.styles import metric_card, score_badge
from ui.charts import shap_chart, forecast_chart, radar_chart
from ml.risk_model import get_risk_label, get_affordability_label, get_risk_trend
from ml.explainability import get_top_drivers, format_driver_text
from ml.forecasting import forecast_metric
from insights.llm_generator import generate_insight


def render_sidebar(
    county: str,
    scores_df: pd.DataFrame,
    models: dict,
    feature_names: list,
    X: pd.DataFrame,
    time_series_data: dict,
):
    """Render the drill-down panel for a selected county."""
    county_mask = scores_df["county"] == county
    if not county_mask.any():
        st.warning(f"No data available for {county}")
        return

    county_row = scores_df[county_mask].iloc[0]
    county_idx = scores_df[county_mask].index[0]

    risk = county_row.get("risk_score", 50)
    livability = county_row.get("livability_score", 50)
    transport = county_row.get("transport_score", 50)
    affordability = county_row.get("affordability_score", 50)
    risk_label = get_risk_label(risk)
    risk_trend = get_risk_trend(county, scores_df)

    # ── Header ───────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom: 16px;">
        <h2 style="margin:0; font-size:1.6rem; font-weight:800; color:#e2e8f0;">
            📍 {county}
        </h2>
        <div style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;">
            {score_badge("Risk", risk)}
            {score_badge("Livability", livability)}
            {score_badge("Transport", transport)}
            {score_badge("Afford.", affordability)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score Cards ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        trend_icon = "🔴" if risk_trend == "Increasing" else "🟡" if risk_trend == "Stable" else "🟢"
        st.markdown(metric_card(
            "Risk Score",
            f"{risk:.0f}",
            f"{trend_icon} {risk_trend}",
            "up" if risk_trend == "Increasing" else "stable" if risk_trend == "Stable" else "down",
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card(
            "Affordability",
            f"{affordability:.0f}",
            f"{get_affordability_label(affordability)}",
            "down" if affordability > 66 else "stable" if affordability > 33 else "up",
        ), unsafe_allow_html=True)

    st.markdown("---")

    # ── Key Metrics ──────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        rent = county_row.get("avg_monthly_rent", 0)
        st.metric("Monthly Rent", f"€{rent:,.0f}")
        congestion = county_row.get("congestion_delay_minutes", 0)
        st.metric("Congestion", f"{congestion:.0f} min")
    with mcol2:
        rent_growth = county_row.get("rent_growth_pct", 0) * 100
        st.metric("Rent Growth", f"{rent_growth:+.1f}%")
        energy = county_row.get("est_annual_energy_cost", 0)
        st.metric("Energy/yr", f"€{energy:,.0f}")

    st.markdown("---")

    # ── SHAP Feature Importance ──────────────────────────────
    st.markdown('<div class="section-header">🔍 Why This Score?</div>', unsafe_allow_html=True)
    risk_model = models.get("risk_score")
    if risk_model is not None:
        drivers = get_top_drivers(risk_model, X, county_idx, feature_names, n=5)
        fig_shap = shap_chart(drivers)
        st.plotly_chart(fig_shap, width='stretch', config={"displayModeBar": False})
    else:
        drivers = []
        st.info("SHAP analysis not available")

    st.markdown("---")

    # ── Radar Chart ──────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ Score Profile</div>', unsafe_allow_html=True)
    fig_radar = radar_chart(risk, livability, transport, county, affordability)
    st.plotly_chart(fig_radar, width='stretch', config={"displayModeBar": False})

    st.markdown("---")

    # ── Time Series Forecast ─────────────────────────────────
    st.markdown('<div class="section-header">📈 Forecast (6 months)</div>', unsafe_allow_html=True)
    forecast_metric_name = st.selectbox(
        "Metric to forecast",
        ["avg_monthly_rent", "congestion_delay_minutes", "traffic_volume"],
        format_func=lambda x: {
            "avg_monthly_rent": "Monthly Rent (€)",
            "congestion_delay_minutes": "Congestion (min)",
            "traffic_volume": "Traffic Volume",
        }.get(x, x),
        key="forecast_select",
    )

    ts_data = time_series_data.get(county)
    if ts_data is not None and len(ts_data) > 0:
        forecast_data = forecast_metric(ts_data, forecast_metric_name, periods=6)
        display_name = {
            "avg_monthly_rent": "Rent",
            "congestion_delay_minutes": "Congestion",
            "traffic_volume": "Traffic",
        }.get(forecast_metric_name, forecast_metric_name)
        fig_forecast = forecast_chart(forecast_data, display_name)
        st.plotly_chart(fig_forecast, width='stretch', config={"displayModeBar": False})
    else:
        st.info("No time series data available for forecasting")

    st.markdown("---")

    # ── LLM Insight ──────────────────────────────────────────
    st.markdown('<div class="section-header">💡 AI Insight</div>', unsafe_allow_html=True)
    insight = generate_insight(
        county=county,
        risk_score=risk,
        risk_label=risk_label,
        risk_trend=risk_trend,
        top_drivers=drivers,
        livability_score=livability,
        transport_score=transport,
        county_row=county_row,
    )
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    return drivers  # Return drivers for use by other tabs
