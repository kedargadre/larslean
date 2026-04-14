"""Area Duel tab — side-by-side comparison."""
import streamlit as st
import plotly.graph_objects as go


def render_duel_tab(scores_df, level="county"):
    st.markdown("### ⚔️ Area Duel")
    name_col = "ed_name" if level == "ed" and "ed_name" in scores_df.columns else "county"
    areas = sorted(scores_df[name_col].unique())

    col1, col2 = st.columns(2)
    a = col1.selectbox("Area A", areas, index=0, key="duel_a")
    b = col2.selectbox("Area B", areas, index=min(1, len(areas)-1), key="duel_b")

    metrics = ["risk_score", "livability_score", "transport_score", "affordability_score"]
    available = [m for m in metrics if m in scores_df.columns]

    row_a = scores_df[scores_df[name_col] == a].iloc[0] if len(scores_df[scores_df[name_col] == a]) > 0 else None
    row_b = scores_df[scores_df[name_col] == b].iloc[0] if len(scores_df[scores_df[name_col] == b]) > 0 else None

    if row_a is None or row_b is None:
        st.warning("Select two valid areas.")
        return

    fig = go.Figure()
    labels = [m.replace("_score", "").title() for m in available]
    fig.add_trace(go.Scatterpolar(r=[row_a[m] for m in available], theta=labels, fill="toself", name=a, opacity=0.6))
    fig.add_trace(go.Scatterpolar(r=[row_b[m] for m in available], theta=labels, fill="toself", name=b, opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=400,
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
    st.plotly_chart(fig, use_container_width=True)
