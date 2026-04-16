"""
Léarslán V3 — Irish Community Intelligence Dashboard
Hierarchical County → Electoral Division Architecture
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No frequency information")

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ── Path Setup ─────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv()

from config import IRISH_COUNTIES, GEOJSON_FILE, get_county_eds, get_province_counties, COUNTY_PROVINCE
from ui.styles import inject_css, metric_card
from ui.map_view import render_map
from ui.sidebar import render_sidebar
from ui.charts import county_comparison_chart
from ui.tab_property import render_property_tab
from ui.tab_budget import render_budget_tab
from ui.tab_forecast import render_forecast_tab
from insights.chat import render_floating_advisor
from insights.context import build_page_context

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Léarslán — Community Intelligence",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data Loading & ML Pipeline ─────────────────────────────────
@st.cache_data(show_spinner="Loading county data & training models...")
def load_and_process_data():
    """Load county-level data, run feature engineering, and train models."""
    from ingestion.spatial_harmonizer import harmonize_data, get_time_series_data
    from ml.feature_engineering import engineer_features
    from ml.risk_model import train_risk_model

    df = harmonize_data()
    daft_summaries = _get_daft_summaries_safe()
    df = engineer_features(df, daft_summaries=daft_summaries)
    models, scored_df, feature_names = train_risk_model(df)
    ts_data = get_time_series_data()

    available_features = [c for c in feature_names if c in scored_df.columns]
    X = scored_df[available_features].fillna(0)

    return scored_df, models, feature_names, X, ts_data, daft_summaries


@st.cache_data(show_spinner="Loading Neighbourhood data...")
def load_and_process_ed_data():
    """Load ED-level data, run feature engineering, and train models."""
    from config import CSO_ED_FILE
    from ingestion.spatial_harmonizer import harmonize_ed_data, get_ed_time_series_data
    from ml.feature_engineering import engineer_features
    from ml.risk_model import train_risk_model

    df = harmonize_ed_data()
    df = engineer_features(df, daft_summaries=None)  # No Daft data at ED level
    models, scored_df, feature_names = train_risk_model(df)
    ts_data = get_ed_time_series_data()

    available_features = [c for c in feature_names if c in scored_df.columns]
    X = scored_df[available_features].fillna(0)

    return scored_df, models, feature_names, X, ts_data


@st.cache_data(ttl=900, show_spinner=False)
def _get_daft_summaries_safe() -> dict:
    """Attempt to fetch Daft.ie summaries for all counties. Graceful on failure."""
    # Bypassed to prevent Streamlit from hanging indefinitely due to Daft.ie rate limiting
    return {}


# ── Matches Dialog ─────────────────────────────────────────────
@st.dialog("🏆 Your Top Matches", width="large")
def _show_matches_dialog(top3, level, full_df, selected_county=None, scope_label="all of Ireland"):
    """Modal popup: 3 winner cards + optional selected-county card + full rankings."""
    medals = ["🥇", "🥈", "🥉"]
    medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    name_col = "ed_name" if level == "ed" and "ed_name" in top3.columns else "county"

    def _match_card(row, label, label_color, medal_icon):
        name = row.get(name_col, "Unknown")
        score = row["match_score"]
        remaining = row["monthly_remaining"]
        fit = row["budget_fit"]
        risk = row.get("risk_score", 50)
        livability = row.get("livability_score", 50)
        rent = row.get("avg_monthly_rent", 0)
        in_red = remaining < 0

        remaining_color = "#10b981" if remaining > 500 else "#f59e0b" if remaining > 0 else "#ef4444"
        card_border = "rgba(239,68,68,0.5)" if in_red else f"{label_color}40"
        card_bg = "rgba(60,10,10,0.7)" if in_red else "rgba(17,26,46,0.8)"
        red_warning = (
            f'<div style="background:#ef444420;border:1px solid #ef4444;border-radius:6px;'
            f'padding:6px 8px;margin-top:8px;font-size:0.78rem;color:#ef4444;font-weight:600;">'
            f'⚠️ Rent (€{rent:,.0f}) exceeds your budget — you\'d be €{abs(remaining):,.0f}/mo in the red</div>'
            if in_red else ""
        )
        return f"""
        <div style="background:{card_bg};border:1px solid {card_border};
                    border-radius:12px;padding:16px;text-align:center;height:100%;">
            <div style="font-size:2rem;">{"🚨" if in_red else medal_icon}</div>
            <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;margin:4px 0;">{name}</div>
            <div style="font-size:1.5rem;font-weight:800;color:{label_color};margin:6px 0;">
                {score:.0f}/100
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;
                        text-align:left;font-size:0.82rem;margin-top:8px;">
                <div><span style="color:#94a3b8;">RENT</span><br><b>€{rent:,.0f}</b></div>
                <div><span style="color:#94a3b8;">LEFT</span><br>
                    <b style="color:{remaining_color};">€{remaining:,.0f}</b></div>
                <div><span style="color:#94a3b8;">RISK</span><br><b>{risk:.0f}</b></div>
                <div><span style="color:#94a3b8;">LIVABILITY</span><br><b>{livability:.0f}</b></div>
            </div>
            <div style="margin-top:8px;font-size:0.8rem;color:#94a3b8;">{fit}</div>
            {red_warning}
        </div>"""

    # ── Top 3 within scope ────────────────────────────────────
    st.markdown(f"**Best matches across {scope_label} based on your profile:**")
    top3_names = list(top3[name_col]) if name_col in top3.columns else []
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.markdown(_match_card(row, medal_colors[i], medal_colors[i], medals[i]),
                        unsafe_allow_html=True)

    # ── Your selected county (if not already in top 3) ───────
    if selected_county and level == "county" and selected_county not in top3_names:
        sel_rows = full_df[full_df[name_col] == selected_county] if name_col in full_df.columns else pd.DataFrame()
        if len(sel_rows) > 0:
            sel_row = sel_rows.iloc[0]
            sel_rank = int(full_df[full_df[name_col] == selected_county].index[0]) + 1
            st.markdown(
                f"<br>**Your selected county — {selected_county} "
                f"(ranked #{sel_rank} of {len(full_df)} in {scope_label}):**",
                unsafe_allow_html=True,
            )
            sel_col, _ = st.columns([1, 2])
            with sel_col:
                st.markdown(_match_card(sel_row, "#60a5fa", "#60a5fa", "📍"),
                            unsafe_allow_html=True)
    elif selected_county and level == "county" and selected_county in top3_names:
        st.caption(f"✓ Your selected county ({selected_county}) is already in your top matches.")

    # ── Full rankings table ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Full Rankings", expanded=False):
        display_cols = [name_col, "match_score", "avg_monthly_rent", "monthly_remaining",
                        "budget_fit", "risk_score", "livability_score"]
        display_cols = [c for c in display_cols if c in full_df.columns]
        ranked_display = full_df[display_cols].reset_index(drop=True)
        ranked_display.index = ranked_display.index + 1
        ranked_display.index.name = "Rank"
        st.dataframe(ranked_display, use_container_width=True, height=320)

    if st.button("✕ Close", key="dlg_close_matches"):
        st.rerun()


# ── Main App ───────────────────────────────────────────────────
def main():
    inject_css()

    # Load county-level data
    try:
        scores_df, models, feature_names, X, ts_data, daft_summaries = load_and_process_data()
    except Exception as e:
        st.error(f"Error loading county data: {e}")
        st.error(f"Traceback: {__import__('traceback').format_exc()}")
        return

    # Load ED-level data
    try:
        ed_scores_df, ed_models, ed_feature_names, ed_X, ed_ts_data = load_and_process_ed_data()
        ed_data_available = True
    except Exception as e:
        ed_data_available = False
        ed_scores_df = None
        ed_ts_data = None

    # ── Sidebar: Your Profile (preferences) ───────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:20px;">
            <div class="dashboard-title">🇮🇪 Léarslán</div>
            <div class="dashboard-subtitle">Community Intelligence Engine</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">👤 Tell Us About You</div>', unsafe_allow_html=True)
        st.caption("A few details about you — we'll find your best-fit areas.")

        budget = st.number_input(
            "💰 Monthly Take-Home (€)", min_value=500, max_value=50000,
            value=int(st.session_state.get("_prefs", {}).get("budget", 3500)),
            step=100, key="pref_budget",
        )
        family_size = st.selectbox(
            "👨‍👩‍👧‍👦 Household", [1, 2, 3, 4, 5], index=0, key="pref_fam",
            format_func=lambda x: {1: "Single", 2: "Couple", 3: "Small Family", 4: "Family", 5: "Large Family"}[x],
        )
        work_mode = st.selectbox(
            "💻 Work Mode", ["remote", "hybrid", "office"], key="pref_work",
            format_func=lambda x: {"remote": "🏠 Remote", "hybrid": "🔄 Hybrid", "office": "🏢 Office"}[x],
        )
        commute = st.selectbox(
            "🚗 Commute Tolerance", ["low", "medium", "high"], index=1, key="pref_comm",
            format_func=lambda x: {"low": "😤 Low", "medium": "😐 Medium", "high": "😎 High"}[x],
        )

        with st.expander("⚖️ Priority Weights", expanded=False):
            w_afford = st.slider("💰 Affordability", 0.0, 1.0, 0.3, 0.05, key="pref_wa")
            w_live = st.slider("🏡 Livability", 0.0, 1.0, 0.25, 0.05, key="pref_wl")
            w_safe = st.slider("🛡️ Low Risk", 0.0, 1.0, 0.2, 0.05, key="pref_ws")
            w_trans = st.slider("🚗 Transport", 0.0, 1.0, 0.15, 0.05, key="pref_wt")

        if st.button("🔍 Find My Matches", use_container_width=True, type="primary"):
            st.session_state["_prefs"] = {
                "budget": int(budget), "family_size": family_size, "work_mode": work_mode,
                "commute": commute, "w_afford": w_afford, "w_live": w_live,
                "w_safe": w_safe, "w_trans": w_trans,
            }
            # One-shot flag: consumed (popped) in main() so dialog only opens
            # for this button click, not on unrelated reruns.
            st.session_state["_show_matches_once"] = True

        st.markdown("---")

        # Quick stats — context-aware (will be populated after scope is chosen below)
        _sidebar_stats_slot = st.container()

    # ── Page header (shown once, above scope bar) ─────────────
    st.markdown("""
    <div style="margin-bottom:16px;">
        <div class="dashboard-title">🇮🇪 Léarslán — Community Intelligence Dashboard</div>
        <div class="dashboard-subtitle">Hyper-local cost-of-living intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Cascading scope/view bar ───────────────────────────────
    # Row 1: pill toggle (zoom level)
    zoom_level = st.radio(
        "Zoom",
        ["🇮🇪 Ireland", "🏛️ County", "🏘️ Neighbourhood"],
        index=0,
        key="zoom_level",
        horizontal=True,
        label_visibility="collapsed",
        help="Drill from national view → county → neighbourhood",
    )
    is_ireland = zoom_level == "🇮🇪 Ireland"
    is_ed_mode = "Neighbourhood" in zoom_level

    # Row 2: contextual dropdowns + metric (only appear when relevant)
    scope_cols = st.columns([1.6, 1.8, 1.6])

    with scope_cols[0]:
        if not is_ireland:
            selected_county = st.selectbox(
                "🏛️ County",
                IRISH_COUNTIES,
                index=IRISH_COUNTIES.index("Dublin"),
                key="county_select",
            )
        else:
            selected_county = "Dublin"  # default for stats, not shown in Ireland view
            st.markdown(
                '<div style="color:#64748b;font-size:0.82rem;padding-top:6px;">'
                'Select County or Neighbourhood above to drill in</div>',
                unsafe_allow_html=True,
            )

    selected_ed_id = None
    selected_ed_name = None
    with scope_cols[1]:
        if is_ed_mode and ed_data_available:
            county_eds_list = get_county_eds(selected_county)
            if county_eds_list:
                ed_options = {f"{name} ({etype})": eid for eid, name, etype in county_eds_list}
                selected_ed_label = st.selectbox(
                    "🏘️ Neighbourhood",
                    list(ed_options.keys()),
                    key="ed_select",
                )
                selected_ed_id = ed_options[selected_ed_label]
                selected_ed_name = selected_ed_label.split(" (")[0]
            else:
                st.info(f"No neighbourhoods for {selected_county}")
                is_ed_mode = False
        elif not is_ireland:
            st.markdown(
                '<div style="color:#64748b;font-size:0.82rem;padding-top:6px;">'
                'Switch to Neighbourhood above to drill deeper</div>',
                unsafe_allow_html=True,
            )

    with scope_cols[2]:
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

    # Populate sidebar quick-stats now that scope is known
    with _sidebar_stats_slot:
        if is_ed_mode and ed_scores_df is not None:
            county_eds_df = ed_scores_df[ed_scores_df["county"] == selected_county]
            n_eds = len(county_eds_df)
            avg_risk = county_eds_df["risk_score"].mean() if n_eds > 0 else 0
            avg_livability = county_eds_df["livability_score"].mean() if n_eds > 0 else 0
            avg_affordability = county_eds_df["affordability_score"].mean() if "affordability_score" in county_eds_df.columns and n_eds > 0 else 50
            high_risk_count = len(county_eds_df[county_eds_df["risk_score"] >= 67]) if n_eds > 0 else 0

            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div class="metric-label">{selected_county} — {n_eds} Neighbourhoods</div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Avg Neighbourhood Risk", f"{avg_risk:.0f}/100")
            st.metric("Avg Neighbourhood Livability", f"{avg_livability:.0f}/100")
            st.metric("Avg Neighbourhood Affordability", f"{avg_affordability:.0f}/100")
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

        st.markdown("---")
        st.caption("Built for 🇮🇪 Ireland")
        st.caption("Data: CSO • TII • SEAI • RTB • Daft.ie")

    # ── Matches dialog — only fires on the rerun triggered by the
    #    "Find My Matches" click; `pop` consumes the flag so any
    #    subsequent rerun (county change, etc.) doesn't re-open it.
    if st.session_state.pop("_show_matches_once", False) and st.session_state.get("_prefs"):
        prefs = st.session_state["_prefs"]
        from ml.recommender import topsis_rank
        _priorities = {
            "affordability_score": prefs.get("w_afford", 0.3),
            "livability_score":    prefs.get("w_live", 0.25),
            "risk_score":          prefs.get("w_safe", 0.2),
            "transport_score":     prefs.get("w_trans", 0.15),
        }

        # Scope the candidate pool by what the user is zoomed into:
        #   Ireland      → all 26 counties
        #   County zoom  → counties in the same province as the selection
        #   Neighbourhood → that county's EDs
        if is_ed_mode and ed_scores_df is not None:
            _rank_df = ed_scores_df[ed_scores_df["county"] == selected_county].copy()
            _level = "ed"
            _scope_label = f"neighbourhoods in {selected_county}"
        elif not is_ireland:
            province_counties = get_province_counties(selected_county)
            if province_counties:
                _rank_df = scores_df[scores_df["county"].isin(province_counties)].copy()
                _scope_label = f"{COUNTY_PROVINCE.get(selected_county, 'region')} counties"
            else:
                _rank_df = scores_df
                _scope_label = "all of Ireland"
            _level = "county"
        else:
            _rank_df = scores_df
            _level = "county"
            _scope_label = "all of Ireland"

        _ranked_full = topsis_rank(
            _rank_df,
            budget=prefs.get("budget", 3500),
            commute_tolerance=prefs.get("commute", "medium"),
            family_size=prefs.get("family_size", 1),
            work_mode=prefs.get("work_mode", "remote"),
            priorities=_priorities,
        )
        _show_matches_dialog(
            _ranked_full.head(3), _level, _ranked_full, selected_county, _scope_label,
        )

    # ── Main content ──────────────────────────────────────────
    level_badge = "🏘️ Neighbourhood" if is_ed_mode else ("🏛️ County" if not is_ireland else "🇮🇪 Ireland")
    st.caption(f"Viewing: **{level_badge}** level")

    tab_overview, tab_forecast = st.tabs([
        "🗺️ Overview",
        "🔮 Forecast",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────
    with tab_overview:
        st.session_state["current_tab"] = "overview"
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

    # ── Tab 2: Forecast ──────────────────────────────────────
    with tab_forecast:
        st.session_state["current_tab"] = "forecast"
        if is_ed_mode and ed_scores_df is not None and selected_ed_id:
            render_forecast_tab(selected_ed_id, ed_scores_df, ed_ts_data, level="ed")
        else:
            render_forecast_tab(selected_county, scores_df, ts_data)

    # ── Floating AI Advisor (bottom-right, all tabs) ──────────
    try:
        from ml.explainability import get_top_drivers
        risk_model = models.get("risk_score")
        county_mask = scores_df["county"] == selected_county
        advisor_drivers = []
        if risk_model is not None and county_mask.any():
            county_idx = scores_df[county_mask].index[0]
            advisor_drivers = get_top_drivers(risk_model, X, county_idx, feature_names, n=5)
    except Exception:
        advisor_drivers = []

    page_context = build_page_context(
        active_tab=st.session_state.get("current_tab", "overview"),
        selected_county=selected_county,
        selected_ed_id=selected_ed_id,
        selected_metric=selected_metric,
        spatial_level="ed" if is_ed_mode else "county",
    )

    active_advisor_df = ed_scores_df if (is_ed_mode and ed_scores_df is not None) else scores_df
    render_floating_advisor(
        selected_county=selected_county,
        scores_df=active_advisor_df,
        drivers=advisor_drivers,
        market=(daft_summaries or {}).get(selected_county, {}),
        page_context=page_context,
        models=models,
        feature_names=feature_names,
        ts_data=ts_data,
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

    _metric_labels = {
        "risk_score": "🔴 Risk Score",
        "livability_score": "🟢 Livability Score",
        "transport_score": "🔵 Transport Score",
        "affordability_score": "💰 Affordability Score",
    }
    st.markdown(f'<div class="section-header">🗺️ Ireland — {_metric_labels.get(selected_metric, selected_metric)}</div>', unsafe_allow_html=True)
    render_map(scores_df, selected_metric)

    # ── County Detail (collapsible) ───────────────────────────
    with st.expander(f"📋 {selected_county} — Detailed Analysis", expanded=False):
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
    """Render the Overview tab — Neighbourhood level (ED map + detail panel)."""
    from ml.risk_model import get_risk_label, get_affordability_label

    # Filter to selected county's neighbourhoods
    county_eds = ed_scores_df[ed_scores_df["county"] == selected_county].copy()

    if len(county_eds) == 0:
        st.warning(f"No Neighbourhood data available for {selected_county}")
        return

    # Top metrics for selected ED
    col1, col2, col3, col4, col5 = st.columns(5)

    ed_row = county_eds[county_eds["ed_id"] == selected_ed_id] if selected_ed_id else county_eds.head(1)
    if len(ed_row) > 0:
        er = ed_row.iloc[0]
        with col1:
            display_name = selected_ed_name or er.get("ed_name", "Unknown")
            st.markdown(metric_card(
                "Neighbourhood", display_name,
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

    _ed_metric_label = {"risk_score":"🔴 Risk Score","livability_score":"🟢 Livability Score","transport_score":"🔵 Transport Score","affordability_score":"💰 Affordability Score"}.get(selected_metric, selected_metric)
    st.markdown(f'<div class="section-header">📍 {selected_county} — {_ed_metric_label}</div>', unsafe_allow_html=True)
    render_map(county_eds, selected_metric, level="ed", county=selected_county)

    # ── Neighbourhood Detail (collapsible) ───────────────────
    with st.expander(f"📋 {selected_county} — Neighbourhood Details & Rankings", expanded=False):
        if selected_ed_id and len(ed_row) > 0:
            er = ed_row.iloc[0]
            st.markdown(f"### {er.get('ed_name', selected_ed_id)}")
            st.markdown(f"**County:** {selected_county}")
            for metric_name, label, icon in [
                ("risk_score", "Risk", "🔴"), ("livability_score", "Livability", "🟢"),
                ("transport_score", "Transport", "🔵"), ("affordability_score", "Affordability", "💰"),
            ]:
                val = er.get(metric_name, 0)
                county_avg = county_eds[metric_name].mean() if metric_name in county_eds.columns else 0
                st.markdown(f"{icon} **{label}:** {val:.0f}/100 <small style='color:#94a3b8;'>County avg: {county_avg:.0f}</small>", unsafe_allow_html=True)
            st.markdown("---")
            stats = {
                "Monthly Rent": f"€{er.get('avg_monthly_rent', 0):,.0f}",
                "Rent Growth": f"{er.get('rent_growth_pct', 0)*100:+.1f}%",
                "Avg Income": f"€{er.get('avg_income', 0):,.0f}",
                "Employment": f"{er.get('employment_rate', 0)*100:.1f}%",
                "Energy Cost": f"€{er.get('est_annual_energy_cost', 0):,.0f}/yr",
                "Congestion": f"{er.get('congestion_delay_minutes', 0):.0f} min",
            }
            for k, v in stats.items():
                st.markdown(f"**{k}:** {v}")
        else:
            st.info("Select a Neighbourhood from the top bar to see details")

        st.markdown("---")
        st.markdown(f"### 📊 All Neighbourhoods in {selected_county}")
        display_cols = ["ed_name"]
        for col in ["risk_score", "livability_score", "affordability_score", "avg_monthly_rent"]:
            if col in county_eds.columns:
                display_cols.append(col)
        ranking = county_eds[display_cols].sort_values("risk_score", ascending=True).reset_index(drop=True)
        ranking.index = ranking.index + 1
        ranking.index.name = "Rank"
        st.dataframe(ranking, width='stretch', height=min(400, 35 * len(ranking) + 38))


def _render_match_cards(top3, level="county"):
    """Render top 3 match result cards."""
    medals = ["🥇", "🥈", "🥉"]
    medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    name_col = "ed_name" if level == "ed" and "ed_name" in top3.columns else "county"

    st.markdown('<div class="section-header">🏆 Your Top Matches</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            name = row.get(name_col, "Unknown")
            score = row["match_score"]
            remaining = row["monthly_remaining"]
            fit = row["budget_fit"]
            risk = row.get("risk_score", 50)
            livability = row.get("livability_score", 50)
            rent = row.get("avg_monthly_rent", 0)

            st.markdown(f"""
            <div class="metric-card" style="border-color: {medal_colors[i]}40; text-align:center;">
                <div style="font-size:2rem;">{medals[i]}</div>
                <div class="metric-value" style="font-size:1.4rem;">{name}</div>
                <div style="font-size:1.6rem; font-weight:800; color:{medal_colors[i]}; margin:6px 0;">
                    {score:.0f}/100
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; text-align:left; font-size:0.8rem;">
                    <div><span style="color:#94a3b8;">RENT</span><br><b>€{rent:,.0f}</b></div>
                    <div><span style="color:#94a3b8;">LEFT</span><br>
                        <b style="color:{'#10b981' if remaining > 500 else '#f59e0b' if remaining > 0 else '#ef4444'};">€{remaining:,.0f}</b>
                    </div>
                    <div><span style="color:#94a3b8;">RISK</span><br><b>{risk:.0f}</b></div>
                    <div><span style="color:#94a3b8;">LIVABILITY</span><br><b>{livability:.0f}</b></div>
                </div>
                <div style="margin-top:8px; font-size:0.8rem;">{fit}</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
