"""Where Should I Live? — TOPSIS recommender tab."""
import streamlit as st


def render_recommender_tab(scores_df, level="county"):
    st.markdown("### 🧠 Where Should I Live?")

    if level == "ed" and "ed_type" in scores_df.columns:
        _render_ed_recommender(scores_df)
    else:
        _render_county_recommender(scores_df)


def _render_county_recommender(df):
    st.markdown("Adjust your priorities to find the best county:")
    col1, col2 = st.columns(2)
    w_afford = col1.slider("🏠 Affordability", 0, 100, 70, key="rec_afford")
    w_livability = col1.slider("🌿 Quality of Life", 0, 100, 50, key="rec_livability")
    w_transport = col2.slider("🚗 Transport", 0, 100, 40, key="rec_transport")
    w_jobs = col2.slider("📈 Job Market", 0, 100, 50, key="rec_jobs")

    weights = {"affordability_score": w_afford, "livability_score": w_livability,
               "transport_score": w_transport, "employment_rate": w_jobs}
    total_w = max(sum(max(v, 1) for v in weights.values()), 1)

    result = df.copy()
    result["match_score"] = sum(
        (max(w, 1) / total_w) * result[col].fillna(0)
        for col, w in weights.items() if col in result.columns
    )
    result = result.sort_values("match_score", ascending=False).head(10)

    for i, (_, row) in enumerate(result.iterrows()):
        name = row.get("county", "Unknown")
        score = row.get("match_score", 0)
        rent = row.get("avg_monthly_rent", 0)
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
        st.markdown(f"**{medal} {name}** — Match: {score:.0f} | Rent: €{rent:,.0f}/mo")


def _render_ed_recommender(df):
    st.markdown("Adjust your priorities:")
    col1, col2, col3 = st.columns(3)
    w_afford = col1.slider("🏠 Affordability", 0, 100, 70, key="rec_ed_afford")
    w_quality = col1.slider("🌿 Quality of Life", 0, 100, 50, key="rec_ed_quality")
    w_transport = col2.slider("🚗 Transport", 0, 100, 40, key="rec_ed_transport")
    w_energy = col2.slider("⚡ Energy Efficiency", 0, 100, 30, key="rec_ed_energy")
    w_jobs = col3.slider("📈 Job Market", 0, 100, 50, key="rec_ed_jobs")
    w_stability = col3.slider("🛡️ Stability", 0, 100, 40, key="rec_ed_stability")

    max_rent = st.slider("Max Rent Budget (€)", 500, 5000, 2000, step=100, key="rec_ed_budget")
    area_types = st.multiselect("Area Types", ["urban_core", "suburban", "town", "village", "rural"],
                                default=["suburban", "town", "village"], key="rec_ed_types")

    try:
        from ml.recommender import topsis_rank
        user_profile = {
            "max_rent_budget": max_rent,
            "selected_area_types": area_types,
            "slider_affordability": w_afford,
            "slider_quality": w_quality,
            "slider_transport": w_transport,
            "slider_energy": w_energy,
            "slider_jobs": w_jobs,
            "slider_stability": w_stability,
        }
        ranked = topsis_rank(df, user_profile)
        if ranked.empty:
            st.warning("No areas match your filters. Try relaxing budget or area types.")
            return
        for i, (_, row) in enumerate(ranked.head(10).iterrows()):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
            st.markdown(f"**{medal} {row.get('ed_name', '?')}** ({row.get('county', '')}) — "
                        f"Match: {row.get('match_score', 0):.1f} | Rent: €{row.get('avg_monthly_rent', 0):,.0f}/mo")
    except Exception as e:
        st.error(f"Recommender error: {e}")
