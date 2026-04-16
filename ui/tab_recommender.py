"""Tab: Where Should I Live? — AI-Powered County Recommender."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import IRISH_COUNTIES, COLORS
from ml.recommender import topsis_rank, get_recommendation_narrative
from ui.styles import metric_card


def render_recommender_tab(scores_df: pd.DataFrame, level: str = "county"):
    """Render the 'Where Should I Live?' recommender tab."""
    level_label = "Electoral Division" if level == "ed" else "County"
    name_col = "ed_name" if level == "ed" and "ed_name" in scores_df.columns else "county"

    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="section-header">🧠 Where Should I Live?</div>
        <span style="color:#94a3b8; font-size:0.9rem;">
            AI-powered {level_label.lower()} ranking based on your personal profile — powered by TOPSIS multi-criteria analysis
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── User Profile Form ─────────────────────────────────────
    st.markdown('<div class="section-header">👤 Your Profile</div>', unsafe_allow_html=True)

    form_col1, form_col2, form_col3 = st.columns(3)

    with form_col1:
        budget = st.slider(
            "💰 Monthly Take-Home (€)",
            min_value=1500, max_value=8000, value=3500, step=100,
            key="rec_budget",
            help="Your monthly after-tax income",
        )

        family_size = st.selectbox(
            "👨‍👩‍👧‍👦 Household Size",
            [1, 2, 3, 4, 5],
            index=0,
            key="rec_family",
            format_func=lambda x: {
                1: "1 — Single",
                2: "2 — Couple",
                3: "3 — Small Family",
                4: "4 — Family",
                5: "5 — Large Family",
            }[x],
        )

    with form_col2:
        work_mode = st.selectbox(
            "💻 Work Mode",
            ["remote", "hybrid", "office"],
            index=0,
            key="rec_work",
            format_func=lambda x: {
                "remote": "🏠 Fully Remote",
                "hybrid": "🔄 Hybrid (2-3 days office)",
                "office": "🏢 Full-Time Office",
            }[x],
        )

        commute_tolerance = st.selectbox(
            "🚗 Commute Tolerance",
            ["low", "medium", "high"],
            index=1,
            key="rec_commute",
            format_func=lambda x: {
                "low": "😤 Low — I hate commuting",
                "medium": "😐 Medium — 30 min is fine",
                "high": "😎 High — I don't mind driving",
            }[x],
        )

    with form_col3:
        st.markdown("""
        <div style="margin-top: 8px;">
            <div class="metric-label">Priority Weights</div>
            <span style="color:#94a3b8; font-size:0.8rem;">
                Adjust what matters most to you
            </span>
        </div>
        """, unsafe_allow_html=True)

        w_afford = st.slider("💰 Affordability", 0.0, 1.0, 0.3, 0.05, key="w_afford")
        w_livability = st.slider("🏡 Livability", 0.0, 1.0, 0.25, 0.05, key="w_livability")
        w_safety = st.slider("🛡️ Low Risk", 0.0, 1.0, 0.2, 0.05, key="w_safety")
        w_transport = st.slider("🚗 Transport", 0.0, 1.0, 0.15, 0.05, key="w_transport")

    st.markdown("---")

    # ── Run TOPSIS ────────────────────────────────────────────
    priorities = {
        "affordability_score": w_afford,
        "livability_score": w_livability,
        "risk_score": w_safety,
        "transport_score": w_transport,
    }

    ranked = topsis_rank(
        scores_df,
        budget=budget,
        commute_tolerance=commute_tolerance,
        family_size=family_size,
        work_mode=work_mode,
        priorities=priorities,
    )

    # ── Top 3 Reveal ──────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Your Top Matches</div>', unsafe_allow_html=True)

    top3 = ranked.head(3)
    medals = ["🥇", "🥈", "🥉"]
    medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            score = row["match_score"]
            county_name = row["county"]
            remaining = row["monthly_remaining"]
            fit = row["budget_fit"]
            risk = row.get("risk_score", 50)
            livability = row.get("livability_score", 50)
            afford = row.get("affordability_score", 50)
            rent = row.get("avg_monthly_rent", 0)

            import textwrap
            html = textwrap.dedent(f"""
            <div class="metric-card" style="border-color: {medal_colors[i]}40; text-align:center; min-height:320px;">
                <div style="font-size:2.5rem; margin-bottom:8px;">{medals[i]}</div>
                <div class="metric-value" style="font-size:1.8rem; margin-bottom:4px;">
                    {county_name}
                </div>
                <div style="font-size:2rem; font-weight:800; color:{medal_colors[i]}; margin:8px 0;">
                    {score:.0f}/100
                </div>
                <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:12px;">Match Score</div>

                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; text-align:left; margin-top:12px;">
                    <div>
                        <span style="color:#94a3b8; font-size:0.75rem;">RENT</span><br>
                        <span style="color:#e2e8f0; font-weight:600;">€{rent:,.0f}</span>
                    </div>
                    <div>
                        <span style="color:#94a3b8; font-size:0.75rem;">REMAINING</span><br>
                        <span style="color:{'#10b981' if remaining > 500 else '#f59e0b' if remaining > 0 else '#ef4444'}; font-weight:600;">
                            €{remaining:,.0f}
                        </span>
                    </div>
                    <div>
                        <span style="color:#94a3b8; font-size:0.75rem;">RISK</span><br>
                        <span style="color:#e2e8f0; font-weight:600;">{risk:.0f}/100</span>
                    </div>
                    <div>
                        <span style="color:#94a3b8; font-size:0.75rem;">LIVABILITY</span><br>
                        <span style="color:#e2e8f0; font-weight:600;">{livability:.0f}/100</span>
                    </div>
                </div>

                <div style="margin-top:12px; font-size:0.85rem;">{fit}</div>
            </div>
            """)
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Narrative Recommendation ──────────────────────────────
    st.markdown('<div class="section-header">📝 AI Recommendation Summary</div>', unsafe_allow_html=True)

    narrative = get_recommendation_narrative(ranked, budget, family_size, work_mode, commute_tolerance)
    st.markdown(f"""
    <div class="insight-box">
        {narrative.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Radar Chart: Top 3 Comparison ─────────────────────────
    st.markdown('<div class="section-header">🕸️ Score Comparison — Top 3</div>', unsafe_allow_html=True)
    fig_radar = _build_radar_chart(top3)
    st.plotly_chart(fig_radar, width='stretch', config={"displayModeBar": False})

    st.markdown("---")

    # ── Full Rankings Table ───────────────────────────────────
    st.markdown('<div class="section-header">📋 Full County Rankings</div>', unsafe_allow_html=True)

    display_df = ranked[[
        "rank", "county", "match_score", "avg_monthly_rent",
        "monthly_remaining", "budget_fit", "risk_score",
        "livability_score", "affordability_score", "transport_score",
    ]].copy()
    display_df.columns = [
        "Rank", "County", "Match Score", "Rent (€)",
        "Monthly Remaining (€)", "Budget Fit", "Risk",
        "Livability", "Affordability", "Transport",
    ]

    # Format numeric columns
    for col in ["Rent (€)", "Monthly Remaining (€)"]:
        display_df[col] = display_df[col].apply(lambda x: f"€{x:,.0f}")
    for col in ["Match Score", "Risk", "Livability", "Affordability", "Transport"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}")

    st.dataframe(display_df, width='stretch', hide_index=True, height=500)


def _build_radar_chart(top3: pd.DataFrame) -> go.Figure:
    """Build a radar/spider chart comparing top 3 counties."""

    categories = ["Affordability", "Livability", "Transport", "Employment", "Low Risk"]
    cat_cols = ["affordability_score", "livability_score", "transport_score", "employment_rate", "risk_score"]

    colors = ["#10b981", "#3b82f6", "#8b5cf6"]

    fig = go.Figure()

    for i, (_, row) in enumerate(top3.iterrows()):
        values = []
        for col in cat_cols:
            val = row.get(col, 50)
            if col == "employment_rate":
                val = val * 100 if val < 1 else val  # normalize
            elif col == "risk_score":
                val = 100 - val  # invert: low risk = high score
            values.append(val)
        values.append(values[0])  # close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=row["county"],
            fill="toself",
            fillcolor=f"rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.1)",
            line=dict(color=colors[i], width=2),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(30, 41, 59, 0.3)",
                color="#94a3b8",
            ),
            angularaxis=dict(
                gridcolor="rgba(30, 41, 59, 0.3)",
                color="#94a3b8",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(17, 26, 46, 0.8)",
            bordercolor="rgba(30, 41, 59, 0.5)",
            borderwidth=1,
        ),
        height=450,
        margin=dict(l=60, r=60, t=40, b=40),
        showlegend=True,
    )

    return fig
