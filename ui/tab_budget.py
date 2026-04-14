"""Budget Simulator tab."""
import streamlit as st


def render_budget_tab(area, scores_df, level="county"):
    st.markdown("### 💰 Budget Simulator")
    name_col = "ed_name" if level == "ed" and "ed_name" in scores_df.columns else "county"
    row = scores_df[scores_df[name_col] == area]
    if row.empty:
        row = scores_df[scores_df["county"] == area]
    if row.empty:
        st.info(f"No data for {area}")
        return

    r = row.iloc[0]
    income = st.number_input("Monthly Take-Home Income (€)", 1000, 15000, 3500, step=100)
    rent = r.get("avg_monthly_rent", 1200)
    energy = r.get("est_annual_energy_cost", 2500) / 12
    commute = r.get("congestion_delay_minutes", 10) * 2 * 22

    total = rent + energy + commute
    remaining = income - total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏠 Rent", f"€{rent:,.0f}")
    col2.metric("⚡ Energy", f"€{energy:,.0f}")
    col3.metric("🚗 Commute", f"€{commute:,.0f}")

    if remaining > 1000:
        col4.metric("✅ Remaining", f"€{remaining:,.0f}", "Comfortable")
    elif remaining > 300:
        col4.metric("🟡 Remaining", f"€{remaining:,.0f}", "Tight")
    elif remaining > 0:
        col4.metric("🟠 Remaining", f"€{remaining:,.0f}", "Stretched")
    else:
        col4.metric("🔴 Remaining", f"€{remaining:,.0f}", "Over Budget")
