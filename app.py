"""
Léarslán V3 — Irish Community Intelligence Dashboard
Hierarchical County → Electoral Division Architecture
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ── Path Setup ─────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv()

from config import IRISH_COUNTIES, GEOJSON_FILE, get_county_eds
from ui.styles import inject_css, metric_card
from ui.map_view import render_map
from ui.sidebar import render_sidebar
from ui.charts import county_comparison_chart
from ui.tab_property import render_property_tab
from ui.tab_duel import render_duel_tab
from ui.tab_budget import render_budget_tab
from ui.tab_forecast import render_forecast_tab
from ui.tab_recommender import render_recommender_tab
from ui.tab_clusters import render_clusters_tab
from insights.chat import render_advisor_tab

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Léarslán — Community Intelligence",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data Loading — Pre-trained Models ──────────────────────────
@st.cache_data(show_spinner="Loading pre-trained models...")
def load_and_process_data():
    """Load pre-trained scored data and models from ml/models/."""
    from ml.pipeline import load_artifacts
    scored_df, models, feature_names = load_artifacts()

    # Aggregate ED-level scored_df to county level for county view
    county_df = scored_df.groupby("county").agg(
        {col: "mean" for col in scored_df.select_dtypes(include="number").columns}
    ).reset_index()

    available_features = [c for c in feature_names if c in county_df.columns]
    X = county_df[available_features].fillna(0)

    # Build time series from raw CSVs for forecast tab
    ts_data = _load_time_series_safe()
    daft_summaries = _get_daft_summaries_safe()

    return county_df, models, feature_names, X, ts_data, daft_summaries


@st.cache_data(show_spinner="Loading Electoral Division data...")
def load_and_process_ed_data():
    """Load pre-trained ED-level scored data from ml/models/."""
    from ml.pipeline import load_artifacts
    scored_df, models, feature_names = load_artifacts()

    available_features = [c for c in feature_names if c in scored_df.columns]
    X = scored_df[available_features].fillna(0)

    # Build ED time series from raw CSVs
    ts_data = _load_ed_time_series_safe()

    return scored_df, models, feature_names, X, ts_data


@st.cache_data(show_spinner=False)
def _load_time_series_safe() -> dict:
    try:
        from ingestion.spatial_harmonizer import get_time_series_data
        return get_time_series_data()
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_ed_time_series_safe() -> dict:
    try:
        from ingestion.spatial_harmonizer import get_ed_time_series_data
        return get_ed_time_series_data()
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def _get_daft_summaries_safe() -> dict:
    try:
        from ingestion.daft_client import get_county_market_summary
        summaries = {}
        for county in IRISH_COUNTIES:
            try:
                summaries[county] = get_county_market_summary(county)
            except Exception:
                summaries[county] = {}
        return summaries
    except Exception:
        return {}


# ── Main App ───────────────────────────────────────────────────
def main():
    inject_css()

    # Load county-level data
    try:
        scores_df, models, feature_names, X, ts_data, daft_summaries = load_and_process_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Ensure ml/models/ contains pre-trained artifacts. Run `python -m ml.pipeline` to generate them.")
        return

    # Load ED-level data
    try:
        ed_scores_df, ed_models, ed_feature_names, ed_X, ed_ts_data = load_and_process_ed_data()
        ed_data_available = True
    except Exception as e:
        ed_data_available = False
        ed_scores_df = None
        ed_ts_data = None

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:24px;">
            <div class="dashboard-title">🇮🇪 Léarslán</div>
            <div class="dashboard-subtitle">Community Intelligence Engine</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Spatial level toggle
        spatial_level = st.radio(
            "🔍 Analysis Level",
            ["🏛️ County", "📍 Electoral Division"],
            index=0,
            key="spatial_level",
            help="Switch between county-level overview and fine-grained Electoral Division analysis"
        )
        is_ed_mode = "Electoral" in spatial_level

        st.markdown("---")

        # County selector (always visible)
        selected_county = st.selectbox(
            "🗺️ Select County",
            IRISH_COUNTIES,
            index=IRISH_COUNTIES.index("Dublin"),
            key="county_select",
        )

        # ED selector (only in ED mode)
        selected_ed_id = None
        selected_ed_name = None
        if is_ed_mode and ed_data_available:
            county_eds = get_county_eds(selected_county)
            if county_eds:
                ed_options = {f"{name} ({etype})": eid for eid, name, etype in county_eds}
                selected_ed_label = st.selectbox(
                    "📍 Select Electoral Division",
                    list(ed_options.keys()),
                    key="ed_select",
                )
                selected_ed_id = ed_options[selected_ed_label]
                selected_ed_name = selected_ed_label.split(" (")[0]
            else:
                st.info(f"No EDs registered for {selected_county}")
                is_ed_mode = False

        # Metric selector
        selected_metric = st.selectbox(
            "📊 Map Metric",
            ["risk_score", "livability_score", "transport_score", "affordability_score"],
            format_func=lambda x: {
                "risk_score": "🔴 Risk Score",
                "livability_score": "🟢 Livability Score",
                "transport_score": "🔵 Transport Score",
                "affordability_score": "💰 Affordability Score",
            }.get(x, x),
            key="metric_select",
        )

        st.markdown("---")

        # Quick stats — context-aware
        if is_ed_mode and ed_scores_df is not None:
            county_eds_df = ed_scores_df[ed_scores_df["county"] == selected_county]
            n_eds = len(county_eds_df)
            avg_risk = county_eds_df["risk_score"].mean() if n_eds > 0 else 0
            avg_livability = county_eds_df["livability_score"].mean() if n_eds > 0 else 0
            avg_affordability = county_eds_df["affordability_score"].mean() if "affordability_score" in county_eds_df.columns and n_eds > 0 else 50
            high_risk_count = len(county_eds_df[county_eds_df["risk_score"] >= 67]) if n_eds > 0 else 0

            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div class="metric-label">{selected_county} — {n_eds} EDs</div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Avg ED Risk", f"{avg_risk:.0f}/100")
            st.metric("Avg ED Livability", f"{avg_livability:.0f}/100")
            st.metric("Avg ED Affordability", f"{avg_affordability:.0f}/100")
            st.metric("High Risk EDs", f"{high_risk_count}")
        else:
            avg_risk = scores_df["risk_score"].mean()
            avg_livability = scores_df["livability_score"].mean()
            avg_affordability = scores_df["affordability_score"].mean() if "affordability_score" in scores_df.columns else 50
            high_risk_count = len(scores_df[scores_df["risk_score"] >= 67])

            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div class="metric-label">National Averages</div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Avg Risk", f"{avg_risk:.0f}/100")
            st.metric("Avg Livability", f"{avg_livability:.0f}/100")
            st.metric("Avg Affordability", f"{avg_affordability:.0f}/100")
            st.metric("High Risk Counties", f"{high_risk_count}")

        st.markdown("---")
        st.caption("Built for 🇮🇪 Ireland")
        st.caption("Data: CSO • TII • SEAI • RTB • Daft.ie")

    # ── Main Content — Multi-Tab ──────────────────────────────
    level_badge = "📍 Electoral Division" if is_ed_mode else "🏛️ County"
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div class="dashboard-title">🇮🇪 Léarslán — Community Intelligence Dashboard</div>
        <div class="dashboard-subtitle">
            Hyper-local cost-of-living intelligence · Viewing: {level_badge} Level
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Run Anomaly Detection ─────────────────────────────────
    from ml.anomaly_detector import detect_anomalies
    from ui.alert_banner import render_alert_banner

    active_df = ed_scores_df if (is_ed_mode and ed_scores_df is not None) else scores_df
    anomalies = detect_anomalies(active_df)
    render_alert_banner(anomalies)

    tab_overview, tab_property, tab_duel, tab_clusters, tab_budget, tab_forecast, tab_recommend, tab_advisor = st.tabs([
        "🗺️ Overview",
        "🏠 Property Explorer",
        "⚔️ Area Duel",
        "🏘️ Area Clusters",
        "💰 Budget Simulator",
        "🔮 Forecast",
        "🧠 Where to Live?",
        "🤖 AI Advisor",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────
    with tab_overview:
        if is_ed_mode and ed_scores_df is not None:
            _render_ed_overview_tab(
                selected_county, selected_ed_id, selected_ed_name,
                selected_metric, ed_scores_df, ed_models, ed_feature_names, ed_X, ed_ts_data
            )
        else:
            _render_overview_tab(
                selected_county, selected_metric,
                scores_df, models, feature_names, X, ts_data
            )

    # ── Tab 2: Property Explorer ──────────────────────────────
    with tab_property:
        render_property_tab(selected_county, scores_df)

    # ── Tab 3: Area Duel ──────────────────────────────────────
    with tab_duel:
        if is_ed_mode and ed_scores_df is not None:
            render_duel_tab(ed_scores_df, level="ed")
        else:
            render_duel_tab(scores_df, level="county")

    # ── Tab 4: Area Clusters ──────────────────────────────────
    with tab_clusters:
        if is_ed_mode and ed_scores_df is not None:
            render_clusters_tab(ed_scores_df, level="ed")
        else:
            render_clusters_tab(scores_df, level="county")

    # ── Tab 5: Budget Simulator ───────────────────────────────
    with tab_budget:
        if is_ed_mode and ed_scores_df is not None:
            selected_area = selected_ed_name or selected_county
            render_budget_tab(selected_area, ed_scores_df, level="ed")
        else:
            render_budget_tab(selected_county, scores_df)

    # ── Tab 6: Forecast ──────────────────────────────────────
    with tab_forecast:
        if is_ed_mode and ed_scores_df is not None and selected_ed_id:
            render_forecast_tab(selected_ed_id, ed_scores_df, ed_ts_data, level="ed")
        else:
            render_forecast_tab(selected_county, scores_df, ts_data)

    # ── Tab 7: Where Should I Live? ──────────────────────────
    with tab_recommend:
        if is_ed_mode and ed_scores_df is not None:
            render_recommender_tab(ed_scores_df, level="ed")
        else:
            render_recommender_tab(scores_df)

    # ── Tab 8: AI Advisor ─────────────────────────────────────
    with tab_advisor:
        from ml.explainability import get_top_drivers
        risk_model = models.get("risk_score")
        county_mask = scores_df["county"] == selected_county
        drivers = []
        if risk_model is not None and county_mask.any():
            try:
                county_idx = scores_df[county_mask].index[0]
                drivers = get_top_drivers(risk_model, X, county_idx, feature_names, n=5)
            except Exception:
                pass

        market = (daft_summaries or {}).get(selected_county, {})
        # Pass ED-level data for live TOPSIS inference
        advisor_df = ed_scores_df if (ed_data_available and ed_scores_df is not None) else scores_df
        render_advisor_tab(
            selected_county, advisor_df, drivers, market,
            models=models, feature_names=feature_names, ts_data=ts_data,
        )


def _render_overview_tab(
    selected_county, selected_metric,
    scores_df, models, feature_names, X, ts_data,
):
    """Render the Overview tab — county level (map + detail panel)."""
    from ml.risk_model import get_risk_label, get_risk_trend, get_affordability_label

    col1, col2, col3, col4, col5 = st.columns(5)

    county_row = scores_df[scores_df["county"] == selected_county]
    if len(county_row) > 0:
        cr = county_row.iloc[0]
        with col1:
            st.markdown(metric_card(
                "Selected County", selected_county,
            ), unsafe_allow_html=True)
        with col2:
            risk = cr.get("risk_score", 0)
            label = get_risk_label(risk)
            trend = get_risk_trend(selected_county, scores_df)
            st.markdown(metric_card(
                "Risk Score", f"{risk:.0f}/100",
                f"{label} • {trend}",
                "up" if trend == "Increasing" else "stable" if trend == "Stable" else "down",
            ), unsafe_allow_html=True)
        with col3:
            rent = cr.get("avg_monthly_rent", 0)
            rent_growth = cr.get("rent_growth_pct", 0) * 100
            st.markdown(metric_card(
                "Monthly Rent", f"€{rent:,.0f}",
                f"{rent_growth:+.1f}% growth",
                "up" if rent_growth > 10 else "stable" if rent_growth > 3 else "down",
            ), unsafe_allow_html=True)
        with col4:
            true_cost = cr.get("true_cost_index", 0)
            st.markdown(metric_card(
                "True Cost Index", f"{true_cost:.0f}",
                "Composite score",
                "stable",
            ), unsafe_allow_html=True)
        with col5:
            afford = cr.get("affordability_score", 50)
            st.markdown(metric_card(
                "Affordability", f"{afford:.0f}/100",
                get_affordability_label(afford),
                "down" if afford > 66 else "stable" if afford > 33 else "up",
            ), unsafe_allow_html=True)

    st.markdown("---")

    map_col, detail_col = st.columns([3, 2])

    with map_col:
        st.markdown('<div class="section-header">🗺️ Ireland Risk Map</div>', unsafe_allow_html=True)
        render_map(scores_df, selected_metric)

        st.markdown("---")
        st.markdown('<div class="section-header">📊 County Comparison</div>', unsafe_allow_html=True)
        fig_compare = county_comparison_chart(scores_df, selected_metric)
        st.plotly_chart(fig_compare, use_container_width=True, config={"displayModeBar": False})

    with detail_col:
        render_sidebar(
            county=selected_county,
            scores_df=scores_df,
            models=models,
            feature_names=feature_names,
            X=X,
            time_series_data=ts_data,
        )


def _render_ed_overview_tab(
    selected_county, selected_ed_id, selected_ed_name,
    selected_metric, ed_scores_df, ed_models, ed_feature_names, ed_X, ed_ts_data,
):
    """Render the Overview tab — ED level (ED map + detail panel)."""
    from ml.risk_model import get_risk_label, get_affordability_label

    # Filter to selected county's EDs
    county_eds = ed_scores_df[ed_scores_df["county"] == selected_county].copy()

    if len(county_eds) == 0:
        st.warning(f"No Electoral Division data available for {selected_county}")
        return

    # Top metrics for selected ED
    col1, col2, col3, col4, col5 = st.columns(5)

    ed_row = county_eds[county_eds["ed_id"] == selected_ed_id] if selected_ed_id else county_eds.head(1)
    if len(ed_row) > 0:
        er = ed_row.iloc[0]
        with col1:
            display_name = selected_ed_name or er.get("ed_name", "Unknown")
            st.markdown(metric_card(
                "Selected ED", display_name,
                f"in {selected_county}",
            ), unsafe_allow_html=True)
        with col2:
            risk = er.get("risk_score", 0)
            label = get_risk_label(risk)
            county_avg_risk = county_eds["risk_score"].mean()
            diff = risk - county_avg_risk
            st.markdown(metric_card(
                "Risk Score", f"{risk:.0f}/100",
                f"{label} • {diff:+.0f} vs county avg",
                "up" if diff > 5 else "stable" if diff > -5 else "down",
            ), unsafe_allow_html=True)
        with col3:
            rent = er.get("avg_monthly_rent", 0)
            county_avg_rent = county_eds["avg_monthly_rent"].mean()
            pct_diff = ((rent - county_avg_rent) / county_avg_rent * 100) if county_avg_rent > 0 else 0
            st.markdown(metric_card(
                "Monthly Rent", f"€{rent:,.0f}",
                f"{pct_diff:+.0f}% vs county avg",
                "up" if pct_diff > 10 else "stable" if pct_diff > -10 else "down",
            ), unsafe_allow_html=True)
        with col4:
            true_cost = er.get("true_cost_index", 0)
            st.markdown(metric_card(
                "True Cost Index", f"{true_cost:.0f}",
                "Composite score",
                "stable",
            ), unsafe_allow_html=True)
        with col5:
            afford = er.get("affordability_score", 50)
            st.markdown(metric_card(
                "Affordability", f"{afford:.0f}/100",
                get_affordability_label(afford),
                "down" if afford > 66 else "stable" if afford > 33 else "up",
            ), unsafe_allow_html=True)

    st.markdown("---")

    map_col, detail_col = st.columns([3, 2])

    with map_col:
        st.markdown(f'<div class="section-header">📍 {selected_county} — Electoral Divisions</div>', unsafe_allow_html=True)
        render_map(county_eds, selected_metric, level="ed", county=selected_county)

        st.markdown("---")
        st.markdown(f'<div class="section-header">📊 ED Comparison — {selected_county}</div>', unsafe_allow_html=True)

        # ED comparison chart
        name_col = "ed_name" if "ed_name" in county_eds.columns else "ed_id"
        fig_compare = county_comparison_chart(county_eds, selected_metric, name_col=name_col)
        st.plotly_chart(fig_compare, use_container_width=True, config={"displayModeBar": False})

    with detail_col:
        st.markdown(f'<div class="section-header">📋 ED Detail Panel</div>', unsafe_allow_html=True)

        if selected_ed_id and len(ed_row) > 0:
            er = ed_row.iloc[0]
            st.markdown(f"### {er.get('ed_name', selected_ed_id)}")
            st.markdown(f"**County:** {selected_county}")

            # Score cards
            for metric_name, label, icon in [
                ("risk_score", "Risk", "🔴"),
                ("livability_score", "Livability", "🟢"),
                ("transport_score", "Transport", "🔵"),
                ("affordability_score", "Affordability", "💰"),
            ]:
                val = er.get(metric_name, 0)
                county_avg = county_eds[metric_name].mean() if metric_name in county_eds.columns else 0
                nat_avg = ed_scores_df[metric_name].mean() if metric_name in ed_scores_df.columns else 0
                st.markdown(f"""
                {icon} **{label}:** {val:.0f}/100
                <small style="color:#94a3b8;">County avg: {county_avg:.0f} · National avg: {nat_avg:.0f}</small>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Key stats table
            stats = {
                "Monthly Rent": f"€{er.get('avg_monthly_rent', 0):,.0f}",
                "Rent Growth": f"{er.get('rent_growth_pct', 0)*100:+.1f}%",
                "Avg Income": f"€{er.get('avg_income', 0):,.0f}",
                "Employment": f"{er.get('employment_rate', 0)*100:.1f}%",
                "Energy Cost": f"€{er.get('est_annual_energy_cost', 0):,.0f}/yr",
                "BER Rating": f"{er.get('ber_avg_score', 0):.1f}",
                "Congestion": f"{er.get('congestion_delay_minutes', 0):.0f} min",
            }
            for k, v in stats.items():
                st.markdown(f"**{k}:** {v}")
        else:
            st.info("Select an Electoral Division from the sidebar to see details")

        # County-wide ED ranking table
        st.markdown("---")
        st.markdown(f"### 📊 All EDs in {selected_county}")

        display_cols = ["ed_name"]
        for col in ["risk_score", "livability_score", "affordability_score", "avg_monthly_rent"]:
            if col in county_eds.columns:
                display_cols.append(col)

        ranking = county_eds[display_cols].sort_values("risk_score", ascending=True).reset_index(drop=True)
        ranking.index = ranking.index + 1
        ranking.index.name = "Rank"

        col_config = {
            "ed_name": st.column_config.TextColumn("Electoral Division", width="medium"),
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
            "livability_score": st.column_config.ProgressColumn("Livability", min_value=0, max_value=100, format="%.0f"),
            "affordability_score": st.column_config.ProgressColumn("Affordability", min_value=0, max_value=100, format="%.0f"),
            "avg_monthly_rent": st.column_config.NumberColumn("Rent €", format="€%.0f"),
        }

        st.dataframe(
            ranking,
            column_config=col_config,
            use_container_width=True,
            height=min(400, 35 * len(ranking) + 38),
        )


if __name__ == "__main__":
    main()
