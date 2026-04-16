"""Tab 4: Budget Simulator — Can I afford to live in {area}?"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import IRISH_COUNTIES
from ui.styles import metric_card
from ui.charts import budget_waterfall


def render_budget_tab(area_name: str, scores_df: pd.DataFrame, level: str = "county"):
    """Render the Budget Simulator tab."""
    county = area_name  # backward compat alias
    name_col = "ed_name" if level == "ed" and "ed_name" in scores_df.columns else "county"

    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="section-header">💰 Budget Simulator — {area_name}</div>
        <span style="color:#94a3b8; font-size:0.9rem;">
            Work out if you can comfortably afford to live here
        </span>
    </div>
    """, unsafe_allow_html=True)

    area_row = scores_df[scores_df[name_col] == area_name]
    if len(area_row) == 0:
        st.error(f"No data for {area_name}.")
        return
    cr = area_row.iloc[0]

    market = {}
    if level == "county":
        try:
            from ingestion.daft_client import get_county_market_summary
            market = get_county_market_summary(area_name)
        except Exception:
            pass

    # ── User Inputs ───────────────────────────────────────────
    st.markdown('<div class="section-header">👤 Your Details</div>', unsafe_allow_html=True)

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        salary = st.slider(
            "💼 Monthly Take-Home Salary (€)",
            min_value=1500, max_value=8000, value=3500, step=100,
            key="budget_salary",
        )

        household_size = st.selectbox(
            "👨‍👩‍👧 Household Size",
            [1, 2, 3, 4, 5],
            index=0,
            key="budget_household",
        )

    with input_col2:
        has_car = st.checkbox("🚗 I own a car (adds commute costs)", value=True, key="budget_car")

        bedroom_pref = st.selectbox(
            "🛏️ Bedrooms Needed",
            [1, 2, 3, 4],
            index=1,
            key="budget_bedrooms",
        )

    st.markdown("---")

    # ── Calculate Costs ───────────────────────────────────────
    # Rent estimate (from Daft or synthetic)
    if market.get("has_live_data") and market.get("rental_median", 0) > 0:
        base_rent = market["rental_median"]
        rent_source = "Daft.ie live median"
    else:
        base_rent = cr.get("avg_monthly_rent", 1200)
        rent_source = "Synthetic estimate"

    # Adjust rent for bedrooms
    rent_multiplier = {1: 0.75, 2: 1.0, 3: 1.25, 4: 1.5}
    monthly_rent = base_rent * rent_multiplier.get(bedroom_pref, 1.0)

    # Energy
    annual_energy = cr.get("est_annual_energy_cost", 2500)
    # Scale energy slightly by household size
    monthly_energy = (annual_energy / 12) * (1 + (household_size - 1) * 0.15)

    # Commute
    if has_car:
        congestion = cr.get("congestion_delay_minutes", 10)
        monthly_commute = congestion * 2 * 22  # €2/min * 22 workdays
    else:
        monthly_commute = 120  # avg public transport monthly

    # Total deductions
    total_costs = monthly_rent + monthly_energy + monthly_commute
    remaining = salary - total_costs
    cost_pct = (total_costs / salary * 100) if salary > 0 else 100

    # ── Verdict ───────────────────────────────────────────────
    if remaining > 1000:
        verdict = "🟢 Comfortable"
        verdict_color = "#10b981"
        verdict_text = f"You'd have **€{remaining:,.0f}** remaining per month — a healthy margin."
    elif remaining > 300:
        verdict = "🟡 Tight"
        verdict_color = "#f59e0b"
        verdict_text = f"You'd have **€{remaining:,.0f}** remaining — workable but watch spending."
    elif remaining > 0:
        verdict = "🔴 Stretched"
        verdict_color = "#ef4444"
        verdict_text = f"Only **€{remaining:,.0f}** remaining — consider alternatives."
    else:
        verdict = "🔴 Unaffordable"
        verdict_color = "#ef4444"
        verdict_text = f"You'd be **€{abs(remaining):,.0f} short** per month."

    # ── KPI Cards ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Your Monthly Budget</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(metric_card(
            "Rent Estimate", f"€{monthly_rent:,.0f}",
            f"{bedroom_pref}-bed • {rent_source}", "stable"
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card(
            "Energy", f"€{monthly_energy:,.0f}",
            f"{household_size}-person household", "stable"
        ), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card(
            "Commute", f"€{monthly_commute:,.0f}",
            "Car" if has_car else "Public transport", "stable"
        ), unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card" style="border-color: {verdict_color}40;">
            <div class="metric-label">Verdict</div>
            <div class="metric-value" style="font-size:1.6rem; background: {verdict_color};
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{verdict}</div>
            <div class="metric-delta" style="color:{verdict_color};">
                €{remaining:,.0f} remaining ({cost_pct:.0f}% of salary)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Waterfall Chart ───────────────────────────────────────
    fig_waterfall = budget_waterfall(salary, monthly_rent, monthly_energy, monthly_commute, area_name)
    st.plotly_chart(fig_waterfall, width='stretch', config={"displayModeBar": False})

    st.markdown("---")

    # ── Alternative Areas ──────────────────────────────────────
    level_label = "Electoral Division" if level == "ed" else "County"
    st.markdown(f'<div class="section-header">🗺️ More Affordable {level_label}s</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <span style="color:#94a3b8; font-size:0.9rem;">
        {level_label}s where a €{salary:,}/month salary stretches further (based on similar bedroom needs)
    </span>
    """, unsafe_allow_html=True)

    alternatives = []
    for _, row in scores_df.iterrows():
        alt_name = row[name_col]
        if alt_name == area_name:
            continue
        alt_rent = row.get("avg_monthly_rent", 0) * rent_multiplier.get(bedroom_pref, 1.0)
        alt_energy = row.get("est_annual_energy_cost", 0) / 12 * (1 + (household_size - 1) * 0.15)
        alt_congestion = row.get("congestion_delay_minutes", 0)
        alt_commute = (alt_congestion * 2 * 22) if has_car else 120
        alt_total = alt_rent + alt_energy + alt_commute
        alt_remaining = salary - alt_total
        alt_livability = row.get("livability_score", 50)

        if alt_remaining > remaining:
            alternatives.append({
                "Area": alt_name,
                "Rent": f"€{alt_rent:,.0f}",
                "Energy": f"€{alt_energy:,.0f}",
                "Commute": f"€{alt_commute:,.0f}",
                "Remaining": f"€{alt_remaining:,.0f}",
                "Savings vs Here": f"+€{alt_remaining - remaining:,.0f}",
                "Livability": f"{alt_livability:.0f}/100",
                "_remaining_val": alt_remaining,
            })

    if alternatives:
        alt_df = pd.DataFrame(alternatives)
        alt_df = alt_df.sort_values("_remaining_val", ascending=False).head(8)
        display_cols = ["Area", "Rent", "Energy", "Commute", "Remaining", "Savings vs Here", "Livability"]
        st.dataframe(alt_df[display_cols], width='stretch', hide_index=True, height=350)
    else:
        st.success(f"🎉 {area_name} is already one of the most affordable options for your budget!")

    # ── Summary insight ───────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div class="insight-box">
        {verdict_text}<br><br>
        💡 <strong>Rent</strong> makes up <strong>{(monthly_rent / total_costs * 100):.0f}%</strong> of your total living costs in {area_name}.
        {"Consider <strong>" + alternatives[0]["Area"] + "</strong> where you could save <strong>" + alternatives[0]["Savings vs Here"] + "/month</strong>." if alternatives else ""}
    </div>
    """, unsafe_allow_html=True)
