"""Tab: Rent Forecasting — Time-series projections with confidence intervals."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import IRISH_COUNTIES, COLORS
from ml.forecasting import forecast_metric
from ui.styles import metric_card


def render_forecast_tab(area_name: str, scores_df: pd.DataFrame, ts_data: dict, level: str = "county"):
    """Render the Rent Forecasting tab."""
    county = area_name  # backward compat
    name_col = "ed_name" if level == "ed" and "ed_name" in scores_df.columns else "county"
    id_col = "ed_id" if level == "ed" and "ed_id" in scores_df.columns else "county"

    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="section-header">🔮 Rent Forecast — {area_name}</div>
        <span style="color:#94a3b8; font-size:0.9rem;">
            ARIMA-powered projections with 80% confidence bands — see where rents are heading
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])
    with ctrl1:
        forecast_horizon = st.slider(
            "📅 Forecast Horizon (months)",
            min_value=3, max_value=12, value=6, step=1,
            key="forecast_horizon",
        )
    with ctrl2:
        forecast_metric_choice = st.selectbox(
            "📊 Metric to Forecast",
            ["avg_monthly_rent", "employment_rate", "traffic_volume"],
            format_func=lambda x: {
                "avg_monthly_rent": "🏠 Monthly Rent (€)",
                "employment_rate": "💼 Employment Rate",
                "traffic_volume": "🚗 Traffic Volume",
            }.get(x, x),
            key="forecast_metric_choice",
        )
    with ctrl3:
        # Compare selector adapts to level
        compare_options = ["None"] + sorted([k for k in ts_data.keys() if k != area_name])
        compare_county = st.selectbox(
            "🆚 Compare With (optional)",
            compare_options,
            index=0,
            key="forecast_compare",
        )

    st.markdown("---")

    # ── Get forecast data ─────────────────────────────────────
    county_ts = ts_data.get(area_name)
    if county_ts is None or len(county_ts) < 4:
        st.warning(f"⚠️ Insufficient time-series data for {county}. Need at least 4 data points.")
        _render_fallback_forecast(county, scores_df, forecast_horizon, forecast_metric_choice)
        return

    result = forecast_metric(county_ts, forecast_metric_choice, periods=forecast_horizon)

    if not result["dates"]:
        st.warning("Unable to generate forecast. Using fallback linear projection.")
        _render_fallback_forecast(county, scores_df, forecast_horizon, forecast_metric_choice)
        return

    # ── Comparison forecast ───────────────────────────────────
    compare_result = None
    if compare_county != "None":
        compare_ts = ts_data.get(compare_county)
        if compare_ts is not None and len(compare_ts) >= 4:
            compare_result = forecast_metric(compare_ts, forecast_metric_choice, periods=forecast_horizon)

    # ── KPI Cards ─────────────────────────────────────────────
    _render_forecast_kpis(county, result, forecast_metric_choice, scores_df)

    st.markdown("---")

    # ── Main Forecast Chart ───────────────────────────────────
    try:
        fig = _build_forecast_chart(
            county, result, forecast_metric_choice,
            compare_county if compare_county != "None" else None,
            compare_result,
        )
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    except Exception as chart_err:
        st.warning(f"⚠️ Chart rendering issue: {chart_err}")
        st.info("The forecast data is available above in the KPI cards. Chart visualization is experiencing a Plotly compatibility issue.")

    # ── Move-By Recommendation ────────────────────────────────
    st.markdown("---")
    _render_move_recommendation(county, result, forecast_metric_choice, scores_df)

    # ── County Forecast Comparison Table ──────────────────────
    st.markdown("---")
    _render_forecast_table(ts_data, forecast_metric_choice, forecast_horizon, scores_df)


def _render_forecast_kpis(county: str, result: dict, metric: str, scores_df: pd.DataFrame):
    """Display KPI cards for the forecast."""

    metric_labels = {
        "avg_monthly_rent": ("Monthly Rent", "€"),
        "employment_rate": ("Employment Rate", ""),
        "traffic_volume": ("Traffic Volume", ""),
    }
    label, prefix = metric_labels.get(metric, (metric, ""))

    # Current value
    current = result["historical_values"][-1] if result["historical_values"] else 0
    # Forecast end value
    forecast_end = result["forecast"][-1] if result["forecast"] else current
    # Change
    change = forecast_end - current
    change_pct = (change / current * 100) if current != 0 else 0

    # Confidence width at end
    ci_lower = result["lower"][-1] if result["lower"] else forecast_end
    ci_upper = result["upper"][-1] if result["upper"] else forecast_end
    ci_range = ci_upper - ci_lower

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if metric == "avg_monthly_rent":
            st.markdown(metric_card(
                f"Current {label}", f"{prefix}{current:,.0f}",
                f"Latest data point", "stable"
            ), unsafe_allow_html=True)
        else:
            st.markdown(metric_card(
                f"Current {label}", f"{current:,.2f}",
                f"Latest data point", "stable"
            ), unsafe_allow_html=True)
    with k2:
        direction = "up" if change > 0 else "down"
        if metric == "avg_monthly_rent":
            st.markdown(metric_card(
                f"Forecast End", f"{prefix}{forecast_end:,.0f}",
                f"{change_pct:+.1f}% projected", direction
            ), unsafe_allow_html=True)
        else:
            st.markdown(metric_card(
                f"Forecast End", f"{forecast_end:,.2f}",
                f"{change_pct:+.1f}% projected", direction
            ), unsafe_allow_html=True)
    with k3:
        if metric == "avg_monthly_rent":
            st.markdown(metric_card(
                "Projected Change", f"{prefix}{abs(change):,.0f}",
                f"{'📈 Increasing' if change > 0 else '📉 Decreasing'}",
                "up" if change > 0 else "down"
            ), unsafe_allow_html=True)
        else:
            st.markdown(metric_card(
                "Projected Change", f"{abs(change):,.3f}",
                f"{'📈 Increasing' if change > 0 else '📉 Decreasing'}",
                "up" if change > 0 else "down"
            ), unsafe_allow_html=True)
    with k4:
        confidence_label = "Low" if ci_range > current * 0.3 else "Medium" if ci_range > current * 0.15 else "High"
        st.markdown(metric_card(
            "Forecast Confidence", confidence_label,
            f"CI Width: {prefix}{ci_range:,.0f}" if metric == "avg_monthly_rent" else f"CI Width: {ci_range:,.3f}",
            "down" if confidence_label == "High" else "stable" if confidence_label == "Medium" else "up"
        ), unsafe_allow_html=True)


def _build_forecast_chart(
    county: str, result: dict, metric: str,
    compare_county: str = None, compare_result: dict = None,
) -> go.Figure:
    """Build the main forecast Plotly chart."""

    metric_labels = {
        "avg_monthly_rent": "Monthly Rent (€)",
        "employment_rate": "Employment Rate",
        "traffic_volume": "Traffic Volume",
    }
    y_label = metric_labels.get(metric, metric)

    fig = go.Figure()

    # ── Historical line ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=result["historical_dates"],
        y=result["historical_values"],
        mode="lines+markers",
        name=f"{county} — Historical",
        line=dict(color="#10b981", width=3),
        marker=dict(size=6, color="#10b981"),
    ))

    # ── Confidence band ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=result["dates"] + result["dates"][::-1],
        y=result["upper"] + result["lower"][::-1],
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.12)",
        line=dict(color="rgba(59, 130, 246, 0)"),
        name="80% Confidence Band",
        hoverinfo="skip",
    ))

    # ── Forecast line ─────────────────────────────────────────
    # Connect historical to forecast
    transition_x = [result["historical_dates"][-1]] + result["dates"] if result["historical_dates"] else result["dates"]
    transition_y = [result["historical_values"][-1]] + result["forecast"] if result["historical_values"] else result["forecast"]

    fig.add_trace(go.Scatter(
        x=transition_x,
        y=transition_y,
        mode="lines+markers",
        name=f"{county} — Forecast",
        line=dict(color="#3b82f6", width=3, dash="dot"),
        marker=dict(size=7, color="#3b82f6", symbol="diamond"),
    ))

    # ── Upper/Lower bounds ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=result["dates"],
        y=result["upper"],
        mode="lines",
        name="Upper Bound",
        line=dict(color="rgba(59, 130, 246, 0.4)", width=1, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=result["dates"],
        y=result["lower"],
        mode="lines",
        name="Lower Bound",
        line=dict(color="rgba(59, 130, 246, 0.4)", width=1, dash="dash"),
    ))

    # ── Comparison county ─────────────────────────────────────
    if compare_county and compare_result and compare_result["dates"]:
        fig.add_trace(go.Scatter(
            x=compare_result["historical_dates"],
            y=compare_result["historical_values"],
            mode="lines",
            name=f"{compare_county} — Historical",
            line=dict(color="#8b5cf6", width=2, dash="solid"),
            opacity=0.7,
        ))
        comp_transition_x = [compare_result["historical_dates"][-1]] + compare_result["dates"]
        comp_transition_y = [compare_result["historical_values"][-1]] + compare_result["forecast"]
        fig.add_trace(go.Scatter(
            x=comp_transition_x,
            y=comp_transition_y,
            mode="lines",
            name=f"{compare_county} — Forecast",
            line=dict(color="#a78bfa", width=2, dash="dot"),
            opacity=0.7,
        ))

    # ── Divider: now vs forecast ──────────────────────────────
    if result["historical_dates"]:
        last_hist = result["historical_dates"][-1]
        # Use add_shape instead of add_vline to avoid Timestamp + int TypeError
        last_hist_str = str(pd.Timestamp(last_hist))
        fig.add_shape(
            type="line",
            x0=last_hist_str, x1=last_hist_str,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=last_hist_str, y=1.05, yref="paper",
            text="Now", showarrow=False,
            font=dict(color="#94a3b8", size=12),
        )

    fig.update_layout(
        title=dict(
            text=f"{y_label} — {county} Forecast",
            font=dict(size=18, color="#e2e8f0", family="Inter"),
        ),
        xaxis=dict(
            title="Date",
            gridcolor="rgba(30, 41, 59, 0.3)",
            zeroline=False,
            color="#94a3b8",
        ),
        yaxis=dict(
            title=y_label,
            gridcolor="rgba(30, 41, 59, 0.3)",
            zeroline=False,
            color="#94a3b8",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(17, 26, 46, 0.8)",
            bordercolor="rgba(30, 41, 59, 0.5)",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
        height=480,
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
    )

    return fig


def _render_move_recommendation(county: str, result: dict, metric: str, scores_df: pd.DataFrame):
    """Render the 'Move-By' recommendation insight box."""

    if metric != "avg_monthly_rent" or not result["forecast"]:
        return

    current = result["historical_values"][-1] if result["historical_values"] else 0
    forecast_vals = result["forecast"]
    dates = result["dates"]

    # Find when rent exceeds thresholds
    threshold_5pct = current * 1.05
    threshold_10pct = current * 1.10

    cross_5 = None
    cross_10 = None
    for i, val in enumerate(forecast_vals):
        if val >= threshold_5pct and cross_5 is None:
            cross_5 = dates[i]
        if val >= threshold_10pct and cross_10 is None:
            cross_10 = dates[i]

    total_increase = forecast_vals[-1] - current
    monthly_loss = total_increase  # per month extra you'd pay

    if total_increase > 0:
        icon = "⏰"
        urgency = "high" if total_increase > 150 else "medium" if total_increase > 50 else "low"
        urgency_text = {
            "high": "Rents are projected to rise significantly. Acting sooner saves real money.",
            "medium": "Moderate rent increases expected. You have some runway, but don't wait too long.",
            "low": "Rents are relatively stable. No urgency, but monitor the trend."
        }[urgency]

        cross_text = ""
        if cross_5:
            cross_text += f"Rent is projected to exceed +5% (€{threshold_5pct:,.0f}) by **{pd.Timestamp(cross_5).strftime('%B %Y')}**. "
        if cross_10:
            cross_text += f"It could hit +10% (€{threshold_10pct:,.0f}) by **{pd.Timestamp(cross_10).strftime('%B %Y')}**."

        st.markdown(f"""
        <div class="insight-box">
            {icon} <strong>Move-By Recommendation for {county}</strong><br><br>
            {urgency_text}<br><br>
            If you wait {len(forecast_vals)} months, you could be paying an extra
            <strong>€{total_increase:,.0f}/month</strong> — that's
            <strong>€{total_increase * 12:,.0f}/year</strong> in additional rent.<br><br>
            {cross_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
            🎉 <strong>Good news for {county}!</strong><br><br>
            Rents are projected to <strong>decrease by €{abs(total_increase):,.0f}/month</strong> over the forecast period.
            You might benefit from waiting — prices could drop further.
        </div>
        """, unsafe_allow_html=True)


def _render_forecast_table(ts_data: dict, metric: str, horizon: int, scores_df: pd.DataFrame):
    """Render a comparison table of forecasts across all counties."""

    st.markdown('<div class="section-header">📋 All-County Forecast Comparison</div>', unsafe_allow_html=True)

    metric_labels = {
        "avg_monthly_rent": "Rent (€)",
        "employment_rate": "Employment",
        "traffic_volume": "Traffic",
    }

    rows = []
    for county in IRISH_COUNTIES:
        county_ts = ts_data.get(county)
        if county_ts is None or len(county_ts) < 4:
            # Use current data as fallback
            cr = scores_df[scores_df["county"] == county]
            if len(cr) > 0:
                current_val = cr.iloc[0].get(metric, 0)
                rows.append({
                    "County": county,
                    "Current": current_val,
                    "Forecast": current_val,
                    "Change": 0,
                    "Change %": 0,
                    "Trend": "—",
                })
            continue

        result = forecast_metric(county_ts, metric, periods=horizon)
        if not result["historical_values"]:
            continue

        current = result["historical_values"][-1]
        forecast_end = result["forecast"][-1] if result["forecast"] else current
        change = forecast_end - current
        change_pct = (change / current * 100) if current != 0 else 0

        trend = "📈 Rising" if change_pct > 3 else "📉 Falling" if change_pct < -3 else "➡️ Stable"

        if metric == "avg_monthly_rent":
            rows.append({
                "County": county,
                "Current": f"€{current:,.0f}",
                "Forecast": f"€{forecast_end:,.0f}",
                "Change": f"€{change:+,.0f}",
                "Change %": f"{change_pct:+.1f}%",
                "Trend": trend,
            })
        else:
            rows.append({
                "County": county,
                "Current": f"{current:,.3f}",
                "Forecast": f"{forecast_end:,.3f}",
                "Change": f"{change:+,.3f}",
                "Change %": f"{change_pct:+.1f}%",
                "Trend": trend,
            })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            width='stretch',
            hide_index=True,
            height=400,
        )


def _render_fallback_forecast(county: str, scores_df: pd.DataFrame, horizon: int, metric: str):
    """Render a simple projection when time-series data is insufficient."""

    cr = scores_df[scores_df["county"] == county]
    if len(cr) == 0:
        st.error("No data available for this county.")
        return

    current = cr.iloc[0].get(metric, 0)
    growth_rate = cr.iloc[0].get("rent_growth_pct", 0.05) if metric == "avg_monthly_rent" else 0.02
    monthly_growth = growth_rate / 12

    dates = pd.date_range(start=pd.Timestamp.now(), periods=horizon, freq="MS")
    projections = [current * (1 + monthly_growth) ** i for i in range(horizon)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=projections,
        mode="lines+markers",
        name="Linear Projection",
        line=dict(color="#f59e0b", width=3, dash="dot"),
        marker=dict(size=6),
    ))
    fig.add_hline(y=current, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Current: €{current:,.0f}" if metric == "avg_monthly_rent" else f"Current: {current:.3f}")

    fig.update_layout(
        title=f"Projected Trend — {county} (Linear Fallback)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        height=400,
        margin=dict(l=60, r=30, t=60, b=50),
    )

    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    st.info("ℹ️ This is a simplified linear projection. More historical data would enable ARIMA forecasting with confidence intervals.")
