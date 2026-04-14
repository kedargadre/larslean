"""Sidebar detail panel for selected county."""
import streamlit as st


def render_sidebar(county, scores_df, models=None, feature_names=None, X=None, time_series_data=None):
    row = scores_df[scores_df["county"] == county]
    if row.empty:
        st.info(f"No data for {county}")
        return

    r = row.iloc[0]
    st.markdown(f'<div class="section-header">📋 {county} Detail</div>', unsafe_allow_html=True)

    for metric, label, icon in [
        ("risk_score", "Risk", "🔴"),
        ("livability_score", "Livability", "🟢"),
        ("transport_score", "Transport", "🔵"),
        ("affordability_score", "Affordability", "💰"),
    ]:
        if metric in r.index:
            st.metric(f"{icon} {label}", f"{r[metric]:.0f}/100")

    st.markdown("---")
    stats = {}
    if "avg_monthly_rent" in r.index:
        stats["Monthly Rent"] = f"€{r['avg_monthly_rent']:,.0f}"
    if "avg_income" in r.index:
        stats["Avg Income"] = f"€{r['avg_income']:,.0f}"
    if "employment_rate" in r.index:
        stats["Employment"] = f"{r['employment_rate']*100:.1f}%"
    if "congestion_delay_minutes" in r.index:
        stats["Congestion"] = f"{r['congestion_delay_minutes']:.0f} min"
    if "ber_avg_score" in r.index:
        stats["BER Score"] = f"{r['ber_avg_score']:.1f}"

    for k, v in stats.items():
        st.markdown(f"**{k}:** {v}")
