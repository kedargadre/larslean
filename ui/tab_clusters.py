"""Tab: Neighbourhood Clustering — 'Find Areas Like Mine'."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.clustering import generate_clusters
from ui.styles import metric_card

def render_clusters_tab(scores_df: pd.DataFrame, level: str = "county"):
    """Render the neighbourhood clustering visualization."""

    level_label = "Electoral Division" if level == "ed" else "County"
    name_col = "ed_name" if level == "ed" and "ed_name" in scores_df.columns else "county"

    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="section-header">🏘️ Find Areas Like Mine</div>
        <span style="color:#94a3b8; font-size:0.9rem;">
            AI-driven demographic and economic clustering to find hidden-gem {level_label.lower()}s with similar profiles.
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating UMAP clusters..."):
        clustered_df = generate_clusters(scores_df)

    if "umap_x" not in clustered_df.columns:
        st.error("Failed to generate clusters. Ensure umap-learn is installed and data is sufficient.")
        return

    # ── Interactive Cluster Map ───────────────────────────────
    st.markdown("### 🗺️ AI Area Constellation")
    st.markdown(f"<span style='color:#94a3b8;'>{level_label}s closer together on this map share statistically similar economic, risk, and livability profiles.</span>", unsafe_allow_html=True)
    
    # Custom color sequence
    color_map = {
        "Premium Urban 🏙️": "#ef4444", 
        "Hidden Gems 💎": "#10b981", 
        "High Risk / Volatile ⚠️": "#f59e0b",
        "Balanced Suburbs 🏡": "#3b82f6",
        "Affordable & Safe 🌿": "#34d399",
        "Budget Rural 🏡": "#8b5cf6",
    }

    fig = px.scatter(
        clustered_df,
        x="umap_x",
        y="umap_y",
        color="cluster_category",
        color_discrete_map=color_map,
        hover_name=name_col,
        hover_data={
            "umap_x": False,
            "umap_y": False,
            "avg_monthly_rent": ":,.0f",
            "livability_score": ":.0f",
            "risk_score": ":.0f",
        },
        size=[10] * len(clustered_df),
        opacity=0.8,
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        xaxis=dict(showgrid=True, gridcolor="rgba(30, 41, 59, 0.3)", zeroline=False, title="", showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(30, 41, 59, 0.3)", zeroline=False, title="", showticklabels=False),
        legend=dict(
            title="Area Profile",
            bgcolor="rgba(17, 26, 46, 0.8)",
            bordercolor="rgba(30, 41, 59, 0.5)",
            borderwidth=1,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
    )

    # Add shadow/glow effect to markers
    for trace in fig.data:
        trace.marker.line.width = 1
        trace.marker.line.color = "#f8fafc"

    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    st.markdown("---")

    # ── "If you like X, consider Y" tool ──────────────────────
    st.markdown("### 🔄 Smart Alternatives")
    
    cols = st.columns([1, 2])
    with cols[0]:
        target = st.selectbox(f"If you want to live in...", sorted(clustered_df[name_col].tolist()), key="cluster_target")
    
    # Find cluster of target
    target_row = clustered_df[clustered_df[name_col] == target]
    if not target_row.empty:
        target_cluster = target_row.iloc[0]["cluster_category"]
        peers = clustered_df[(clustered_df["cluster_category"] == target_cluster) & (clustered_df[name_col] != target)]
        
        with cols[1]:
            st.markdown(f"<div style='margin-top: 32px;'><strong>{target}</strong> is a <strong>{target_cluster}</strong>.</div>", unsafe_allow_html=True)
            
        if not peers.empty:
            st.markdown("Consider these statistically similar alternatives:")
            
            p1, p2, p3 = st.columns(3)
            peer_cols = [p1, p2, p3]
            
            for i, (_, peer) in enumerate(peers.head(3).iterrows()):
                rent_diff = peer["avg_monthly_rent"] - target_row.iloc[0]["avg_monthly_rent"]
                direction = "down" if rent_diff < 0 else "up"
                
                with peer_cols[i % 3]:
                    st.markdown(metric_card(
                        peer[name_col], 
                        f"€{peer['avg_monthly_rent']:,.0f} / mo",
                        f"{'Save' if rent_diff < 0 else 'Pay extra'} €{abs(rent_diff):,.0f} vs {target}",
                        direction
                    ), unsafe_allow_html=True)
        else:
            st.info(f"No close statistical peers found for this {level_label.lower()} in the current model.")

