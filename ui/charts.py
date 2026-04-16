"""Plotly Charts V2 - SHAP, Forecast, Radar, Comparison, Histogram, Donut, Waterfall, Gauge."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ── Shared Layout ──────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(30,41,59,0.3)", zerolinecolor="rgba(30,41,59,0.3)"),
    yaxis=dict(gridcolor="rgba(30,41,59,0.3)", zerolinecolor="rgba(30,41,59,0.3)"),
)


def shap_chart(drivers: list) -> go.Figure:
    """Create a horizontal bar chart showing top feature drivers."""
    if not drivers:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No drivers available")
        return fig

    features = [d["feature"] for d in reversed(drivers)]
    impacts = [d["shap_value"] for d in reversed(drivers)]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.2f}" for v in impacts],
        textposition="outside",
        textfont=dict(size=11, color="#e2e8f0"),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="🔍 Top Risk Drivers (SHAP)", font=dict(size=14)),
        height=280,
        xaxis_title="Impact on Risk Score",
        showlegend=False,
    )

    return fig


def forecast_chart(forecast_data: dict, metric_name: str = "Rent") -> go.Figure:
    """Create time-series forecast line chart with confidence band."""
    fig = go.Figure()

    if not forecast_data or not forecast_data.get("historical_dates"):
        fig.update_layout(**DARK_LAYOUT, title="No forecast data available")
        return fig

    hist_dates = forecast_data["historical_dates"]
    hist_values = forecast_data["historical_values"]
    fut_dates = forecast_data["dates"]
    fut_forecast = forecast_data["forecast"]
    fut_lower = forecast_data["lower"]
    fut_upper = forecast_data["upper"]

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_values,
        mode="lines+markers",
        name="Historical",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=5),
    ))

    if fut_dates:
        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(fut_dates) + list(reversed(fut_dates)),
            y=list(fut_upper) + list(reversed(fut_lower)),
            fill="toself",
            fillcolor="rgba(139, 92, 246, 0.15)",
            line=dict(width=0),
            name="Confidence",
            showlegend=False,
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fut_dates, y=fut_forecast,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#8b5cf6", width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
        ))

        # Divider line
        fig.add_vline(
            x=hist_dates[-1] if hist_dates else None,
            line=dict(color="rgba(148,163,184,0.3)", width=1, dash="dot"),
        )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"📈 {metric_name} Forecast (6 months)", font=dict(size=14)),
        height=300,
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        hovermode="x unified",
    )

    return fig


def radar_chart(risk: float, livability: float, transport: float,
                county_name: str = "", affordability: float = 50) -> go.Figure:
    """Create radar/spider chart for the four scores."""
    categories = ["Risk\n(inverted)", "Livability", "Transport", "Affordability"]
    values = [100 - risk, livability, transport, affordability]
    values.append(values[0])  # close the polygon
    categories.append(categories[0])

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="#10b981", width=2),
        marker=dict(size=8, color="#10b981"),
        name=county_name,
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"⚡ Score Profile", font=dict(size=14)),
        height=320,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="rgba(30,41,59,0.3)",
                linecolor="rgba(30,41,59,0.3)",
                tickfont=dict(size=9, color="#94a3b8"),
            ),
            angularaxis=dict(
                gridcolor="rgba(30,41,59,0.3)",
                linecolor="rgba(30,41,59,0.3)",
                tickfont=dict(size=11, color="#e2e8f0"),
            ),
        ),
        showlegend=False,
    )

    return fig


def dual_radar_chart(county_a: str, scores_a: dict, county_b: str, scores_b: dict) -> go.Figure:
    """Create overlaid radar chart comparing two counties."""
    categories = ["Risk\n(inverted)", "Livability", "Transport", "Affordability"]

    vals_a = [100 - scores_a.get("risk", 50), scores_a.get("livability", 50),
              scores_a.get("transport", 50), scores_a.get("affordability", 50)]
    vals_b = [100 - scores_b.get("risk", 50), scores_b.get("livability", 50),
              scores_b.get("transport", 50), scores_b.get("affordability", 50)]

    vals_a.append(vals_a[0])
    vals_b.append(vals_b[0])
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=vals_a, theta=categories_closed,
        fill="toself", fillcolor="rgba(59, 130, 246, 0.12)",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=8, color="#3b82f6"),
        name=county_a,
    ))

    fig.add_trace(go.Scatterpolar(
        r=vals_b, theta=categories_closed,
        fill="toself", fillcolor="rgba(239, 68, 68, 0.12)",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=8, color="#ef4444"),
        name=county_b,
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"⚔️ {county_a} vs {county_b}", font=dict(size=14)),
        height=380,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="rgba(30,41,59,0.3)",
                            tickfont=dict(size=9, color="#94a3b8")),
            angularaxis=dict(gridcolor="rgba(30,41,59,0.3)",
                             tickfont=dict(size=11, color="#e2e8f0")),
        ),
        legend=dict(orientation="h", y=-0.1, font=dict(size=12)),
    )

    return fig


def county_comparison_chart(scores_df: pd.DataFrame, metric: str = "risk_score", name_col: str = "county") -> go.Figure:
    """Create horizontal bar chart comparing all areas for a metric.
    
    Args:
        scores_df: DataFrame with metric scores
        metric: column name to compare
        name_col: column to use for area names ("county" or "ed_name")
    """
    metric_labels = {
        "risk_score": "Risk Score",
        "livability_score": "Livability Score",
        "transport_score": "Transport Score",
        "affordability_score": "Affordability Score",
    }

    if name_col not in scores_df.columns:
        name_col = "county"

    sorted_df = scores_df.sort_values(metric, ascending=True)

    colors = []
    for val in sorted_df[metric]:
        if val >= 67:
            colors.append("#ef4444")
        elif val >= 34:
            colors.append("#f59e0b")
        else:
            colors.append("#10b981")

    level_label = "ED" if name_col != "county" else "County"

    fig = go.Figure(go.Bar(
        x=sorted_df[metric],
        y=sorted_df[name_col],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}" for v in sorted_df[metric]],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(
            text=f"📊 {metric_labels.get(metric, metric)} by {level_label}",
            font=dict(size=14),
        ),
        height=max(400, len(sorted_df) * 22),
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 110])

    return fig


# ── V2 Charts ──────────────────────────────────────────────────

def rent_distribution_histogram(prices: list, county: str = "", median_val: float = 0) -> go.Figure:
    """Plotly histogram of rent prices with median line overlay."""
    fig = go.Figure()

    if not prices:
        fig.update_layout(**DARK_LAYOUT, title="No price data available")
        return fig

    fig.add_trace(go.Histogram(
        x=prices,
        nbinsx=20,
        marker=dict(
            color="rgba(59, 130, 246, 0.6)",
            line=dict(color="#3b82f6", width=1),
        ),
        name="Listings",
    ))

    if median_val > 0:
        fig.add_vline(
            x=median_val,
            line=dict(color="#10b981", width=2, dash="dash"),
            annotation_text=f"Median: €{median_val:,.0f}",
            annotation_font=dict(color="#10b981", size=12),
        )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"💰 Price Distribution — {county}", font=dict(size=14)),
        xaxis_title="Price (€)",
        yaxis_title="Number of Listings",
        height=300,
        showlegend=False,
        bargap=0.1,
    )

    return fig


def property_type_donut(type_counts: dict, title: str = "Property Types") -> go.Figure:
    """Donut/pie chart of property type breakdown."""
    if not type_counts:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No data")
        return fig

    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"]

    fig = go.Figure(go.Pie(
        labels=list(type_counts.keys()),
        values=list(type_counts.values()),
        hole=0.55,
        marker=dict(colors=colors[:len(type_counts)],
                     line=dict(color="#0a1628", width=2)),
        textfont=dict(size=11, color="#e2e8f0"),
        textinfo="label+percent",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"🏠 {title}", font=dict(size=14)),
        height=300,
        showlegend=False,
    )

    return fig


def budget_waterfall(salary: float, rent: float, energy: float, commute: float,
                     county: str = "") -> go.Figure:
    """Waterfall chart showing salary deductions."""
    remaining = salary - rent - energy - commute

    fig = go.Figure(go.Waterfall(
        name="Budget",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["💼 Salary", "🏠 Rent", "⚡ Energy", "🚗 Commute", "💰 Remaining"],
        y=[salary, -rent, -energy, -commute, 0],
        connector=dict(line=dict(color="rgba(148,163,184,0.3)")),
        increasing=dict(marker=dict(color="#10b981")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#3b82f6" if remaining > 0 else "#ef4444")),
        textposition="outside",
        text=[f"€{salary:,.0f}", f"-€{rent:,.0f}", f"-€{energy:,.0f}",
              f"-€{commute:,.0f}", f"€{remaining:,.0f}"],
        textfont=dict(size=11, color="#e2e8f0"),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"💰 Monthly Budget Breakdown — {county}", font=dict(size=14)),
        height=380,
        showlegend=False,
    )
    fig.update_yaxes(title="€ per month")

    return fig


def national_comparison_gauge(county_median: float, national_median: float,
                               label: str = "Rent") -> go.Figure:
    """Gauge chart showing county median vs national median."""
    if national_median == 0:
        ratio = 1.0
    else:
        ratio = county_median / national_median

    pct = ratio * 100

    if ratio < 0.85:
        color = "#10b981"
    elif ratio < 1.15:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=county_median,
        number=dict(prefix="€", font=dict(size=28, color="#e2e8f0")),
        delta=dict(
            reference=national_median,
            relative=True,
            valueformat=".0%",
            increasing=dict(color="#ef4444"),
            decreasing=dict(color="#10b981"),
        ),
        gauge=dict(
            axis=dict(range=[0, max(county_median, national_median) * 1.5],
                      tickcolor="#94a3b8"),
            bar=dict(color=color),
            bgcolor="rgba(17,26,46,0.9)",
            bordercolor="#1e293b",
            steps=[
                dict(range=[0, national_median * 0.85], color="rgba(16,185,129,0.1)"),
                dict(range=[national_median * 0.85, national_median * 1.15], color="rgba(245,158,11,0.1)"),
                dict(range=[national_median * 1.15, max(county_median, national_median) * 1.5], color="rgba(239,68,68,0.1)"),
            ],
            threshold=dict(
                line=dict(color="#e2e8f0", width=2),
                thickness=0.75,
                value=national_median,
            ),
        ),
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"📊 {label} vs National Median", font=dict(size=13)),
        height=250,
    )

    return fig
