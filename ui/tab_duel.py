"""Tab 3: Area Duel — Head-to-head comparison of two areas (County or ED)."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import IRISH_COUNTIES
from ui.styles import metric_card
from ui.charts import dual_radar_chart
from ml.risk_model import get_risk_label, get_affordability_label


def render_duel_tab(scores_df: pd.DataFrame, level: str = "county"):
    """Render the Area Duel comparison tab."""

    level_label = "Electoral Division" if level == "ed" else "County"
    name_col = "ed_name" if level == "ed" and "ed_name" in scores_df.columns else "county"

    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="section-header">⚔️ {level_label} Duel — Head-to-Head Comparison</div>
        <span style="color:#94a3b8; font-size:0.9rem;">
            Compare two {level_label.lower()}s side-by-side across all metrics
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Area Selectors ───────────────────────────────────────
    area_list = sorted(scores_df[name_col].unique().tolist())
    if len(area_list) < 2:
        st.warning(f"Need at least 2 {level_label.lower()}s to compare.")
        return

    col_a, col_vs, col_b = st.columns([5, 1, 5])

    with col_a:
        default_a = area_list.index("Dublin") if "Dublin" in area_list else 0
        area_a = st.selectbox(
            f"🔵 {level_label} A",
            area_list,
            index=default_a,
            key="duel_area_a",
        )

    with col_vs:
        st.markdown("""
        <div style="text-align:center; padding-top:28px;">
            <span style="font-size:2rem; font-weight:800; 
                  background: linear-gradient(135deg, #3b82f6, #ef4444);
                  -webkit-background-clip: text;
                  -webkit-text-fill-color: transparent;">VS</span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        default_b = min(1, len(area_list) - 1)
        area_b = st.selectbox(
            f"🔴 {level_label} B",
            area_list,
            index=default_b,
            key="duel_area_b",
        )

    if area_a == area_b:
        st.warning(f"Please select two different {level_label.lower()}s to compare.")
        return

    st.markdown("---")

    # ── Get Data ──────────────────────────────────────────────
    row_a = scores_df[scores_df[name_col] == area_a]
    row_b = scores_df[scores_df[name_col] == area_b]

    if len(row_a) == 0 or len(row_b) == 0:
        st.error("Data not available for selected areas.")
        return

    ra = row_a.iloc[0]
    rb = row_b.iloc[0]

    # Market data (only at county level)
    market_a = {}
    market_b = {}
    if level == "county":
        try:
            from ingestion.daft_client import get_county_market_summary
            market_a = get_county_market_summary(area_a)
            market_b = get_county_market_summary(area_b)
        except Exception:
            pass

    # ── Radar Chart ───────────────────────────────────────────
    scores_a = {
        "risk": ra.get("risk_score", 50),
        "livability": ra.get("livability_score", 50),
        "transport": ra.get("transport_score", 50),
        "affordability": ra.get("affordability_score", 50),
    }
    scores_b = {
        "risk": rb.get("risk_score", 50),
        "livability": rb.get("livability_score", 50),
        "transport": rb.get("transport_score", 50),
        "affordability": rb.get("affordability_score", 50),
    }

    fig_radar = dual_radar_chart(area_a, scores_a, area_b, scores_b)
    st.plotly_chart(fig_radar, width='stretch', config={"displayModeBar": False})

    st.markdown("---")

    # ── Comparison Table ──────────────────────────────────────
    st.markdown('<div class="section-header">📊 Metric Comparison</div>', unsafe_allow_html=True)

    metrics = [
        ("🔴 Risk Score", "risk_score", "lower", "{:.0f}/100"),
        ("🟢 Livability Score", "livability_score", "higher", "{:.0f}/100"),
        ("🔵 Transport Score", "transport_score", "higher", "{:.0f}/100"),
        ("💰 Affordability Score", "affordability_score", "higher", "{:.0f}/100"),
        ("🏠 Monthly Rent", "avg_monthly_rent", "lower", "€{:,.0f}"),
        ("📈 Rent Growth", "rent_growth_pct", "lower", "{:.1%}"),
        ("💼 Avg Income", "avg_income", "higher", "€{:,.0f}"),
        ("👔 Employment Rate", "employment_rate", "higher", "{:.1%}"),
        ("🚗 Congestion", "congestion_delay_minutes", "lower", "{:.0f} min"),
        ("⚡ Energy Cost/yr", "est_annual_energy_cost", "lower", "€{:,.0f}"),
        ("🏗️ BER Rating", "ber_avg_score", "lower", "{:.1f}"),
    ]

    table_rows = []
    for label, col, better, fmt in metrics:
        val_a = ra.get(col, 0)
        val_b = rb.get(col, 0)

        try:
            str_a = fmt.format(val_a)
            str_b = fmt.format(val_b)
        except (ValueError, TypeError):
            str_a = str(val_a)
            str_b = str(val_b)

        if better == "higher":
            winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
        else:
            winner = "A" if val_a < val_b else "B" if val_b < val_a else "Tie"

        icon_a = "✅" if winner == "A" else ""
        icon_b = "✅" if winner == "B" else ""

        table_rows.append({
            "Metric": label,
            f"🔵 {area_a}": f"{str_a} {icon_a}",
            f"🔴 {area_b}": f"{str_b} {icon_b}",
            "Winner": area_a if winner == "A" else area_b if winner == "B" else "Tie",
        })

    # Add live Daft data rows if available (county level only)
    if market_a.get("has_live_data") or market_b.get("has_live_data"):
        daft_metrics = [
            ("🏠 Live Median Rent", "rental_median", "lower", "€{:,.0f}"),
            ("📋 Active Listings", "rental_listing_count", "higher", "{:,}"),
            ("🏷️ Price/Bedroom", "rental_price_per_bedroom", "lower", "€{:,.0f}"),
        ]
        for label, key, better, fmt in daft_metrics:
            val_a = market_a.get(key, 0)
            val_b = market_b.get(key, 0)
            try:
                str_a = fmt.format(val_a)
                str_b = fmt.format(val_b)
            except (ValueError, TypeError):
                str_a = str(val_a)
                str_b = str(val_b)
            if better == "higher":
                winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
            else:
                winner = "A" if val_a < val_b else "B" if val_b < val_a else "Tie"
            icon_a = "✅" if winner == "A" else ""
            icon_b = "✅" if winner == "B" else ""
            table_rows.append({
                "Metric": label,
                f"🔵 {area_a}": f"{str_a} {icon_a}",
                f"🔴 {area_b}": f"{str_b} {icon_b}",
                "Winner": area_a if winner == "A" else area_b if winner == "B" else "Tie",
            })

    table_df = pd.DataFrame(table_rows)
    st.dataframe(table_df, width='stretch', hide_index=True, height=500)

    # ── Scoreboard ────────────────────────────────────────────
    st.markdown("---")
    wins_a = sum(1 for r in table_rows if r["Winner"] == area_a)
    wins_b = sum(1 for r in table_rows if r["Winner"] == area_b)
    ties = sum(1 for r in table_rows if r["Winner"] == "Tie")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(metric_card(
            f"🔵 {area_a} Wins", str(wins_a), "", "down"
        ), unsafe_allow_html=True)
    with sc2:
        st.markdown(metric_card(
            "🤝 Ties", str(ties), "", "stable"
        ), unsafe_allow_html=True)
    with sc3:
        st.markdown(metric_card(
            f"🔴 {area_b} Wins", str(wins_b), "", "up"
        ), unsafe_allow_html=True)

    # ── Natural Language Summary ──────────────────────────────
    st.markdown("---")
    overall_winner = area_a if wins_a > wins_b else area_b if wins_b > wins_a else "It's a tie"

    if overall_winner != "It's a tie":
        loser = area_b if overall_winner == area_a else area_a
        afford_diff = abs(ra.get("affordability_score", 50) - rb.get("affordability_score", 50))
        transport_diff = abs(ra.get("transport_score", 50) - rb.get("transport_score", 50))

        summary = (
            f"📊 **{overall_winner}** wins the duel with **{max(wins_a, wins_b)}/{len(table_rows)}** metrics. "
            f"It has a {get_risk_label(ra.get('risk_score', 50) if overall_winner == area_a else rb.get('risk_score', 50)).lower()} "
            f"risk profile and is rated as "
            f"**{get_affordability_label(ra.get('affordability_score', 50) if overall_winner == area_a else rb.get('affordability_score', 50)).lower()}** "
            f"for affordability. "
            f"The affordability gap is **{afford_diff:.0f} points** "
            f"and the transport connectivity gap is **{transport_diff:.0f} points**."
        )
    else:
        summary = f"📊 **{area_a}** and **{area_b}** are evenly matched across all metrics — a genuine tie!"

    st.markdown(f'<div class="insight-box">{summary}</div>', unsafe_allow_html=True)

