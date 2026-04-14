"""Property Explorer tab."""
import streamlit as st


def render_property_tab(county, scores_df):
    st.markdown("### 🏠 Property Explorer")
    row = scores_df[scores_df["county"] == county]
    if row.empty:
        st.info(f"No data for {county}")
        return
    r = row.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Rent", f"€{r.get('avg_monthly_rent', 0):,.0f}/mo")
    col2.metric("Rental Yield", f"{r.get('rental_yield', 0):.1f}%")
    col3.metric("Affordability", f"{r.get('affordability_score', 0):.0f}/100")
    st.caption("Live Daft.ie integration coming soon.")
