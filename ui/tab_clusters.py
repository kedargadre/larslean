"""Area Clusters tab — KMeans + UMAP scatter."""
import streamlit as st
import plotly.express as px


def render_clusters_tab(scores_df, level="county"):
    st.markdown("### 🏘️ Area Clusters")

    if "cluster_category" not in scores_df.columns:
        st.info("Cluster data not available. Run the ML pipeline first.")
        return

    name_col = "ed_name" if level == "ed" and "ed_name" in scores_df.columns else "county"
    x_col = "umap_x" if "umap_x" in scores_df.columns else "affordability_score"
    y_col = "umap_y" if "umap_y" in scores_df.columns else "risk_score"

    fig = px.scatter(
        scores_df, x=x_col, y=y_col, color="cluster_category",
        hover_name=name_col,
        hover_data=["risk_score", "livability_score", "affordability_score", "avg_monthly_rent"],
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis_title="UMAP-1" if "umap" in x_col else x_col.replace("_", " ").title(),
        yaxis_title="UMAP-2" if "umap" in y_col else y_col.replace("_", " ").title(),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cluster Summary")
    summary = scores_df.groupby("cluster_category").agg(
        count=("cluster_category", "size"),
        avg_rent=("avg_monthly_rent", "mean"),
        avg_risk=("risk_score", "mean"),
        avg_livability=("livability_score", "mean"),
    ).round(1)
    st.dataframe(summary, use_container_width=True)
