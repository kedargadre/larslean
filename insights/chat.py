"""
AI Advisor chat module for the Léarslán dashboard.
Uses Anthropic Claude. Falls back to template responses if API unavailable.
"""
import logging
from textwrap import dedent

import streamlit as st
import pandas as pd

from insights.context import build_area_context, build_page_context
from insights.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)

@st.cache_resource
def _get_cached_rag_engine():
    return get_rag_engine()

_MAX_HISTORY = 20

_SYSTEM_PROMPT = dedent("""\
    You are **Léarslán AI**, an expert Irish community intelligence advisor built
    into the Léarslán dashboard. You help users understand cost-of-living data,
    housing affordability, risk scores, transport connectivity, and energy
    efficiency across Ireland's 26 counties and 255 Electoral Divisions.

    PERSONALITY:
    - Warm, knowledgeable, concise. You speak like a trusted local advisor.
    - Use Irish place names naturally. Reference specific data points.
    - Be opinionated when the data supports it.

    CAPABILITIES:
    - Explain ML scores (risk, livability, transport, affordability) and what drives them.
    - Compare areas using real metrics from GBM models.
    - Advise on relocation decisions based on TOPSIS multi-criteria ranking.
    - Interpret SHAP feature importance to explain why an area scores high/low.
    - Provide ARIMA-based rent forecasts.
    - Reference Irish housing policy, grants, and government schemes.

    CITATION RULES:
    - When referencing policy documents, cite with [Source: document title]
    - When referencing data metrics, cite with [Data: metric_name = value]
    - When referencing model analysis, cite with [Model: model_name]
    - Include at least one citation per answer when data is relevant.
    - If you cannot ground an answer in available data, say so honestly.

    FORMATTING:
    - Use markdown. Keep answers 3-5 paragraphs max unless asked for detail.
    - Use EUR for currency, Irish conventions for place names.
""")


def render_floating_advisor(
    selected_county,
    scores_df,
    drivers=None,
    market=None,
    page_context=None,
    models=None,
    feature_names=None,
    ts_data=None,
):
    """Render the AI Advisor as a fixed bottom-right popover on all tabs."""
    _ensure_session_state()

    # Store ML context in session for _generate_response and the fragment
    st.session_state["_advisor_models"] = models or {}
    st.session_state["_advisor_feature_names"] = feature_names or []
    st.session_state["_advisor_scored_df"] = scores_df
    st.session_state["_advisor_ts_data"] = ts_data
    st.session_state["_advisor_county"] = selected_county
    st.session_state["_advisor_area_context"] = build_area_context(
        selected_county, scores_df, drivers, market
    )
    st.session_state["_advisor_page_ctx"] = page_context or build_page_context(
        active_tab="overview", selected_county=selected_county,
    )

    # Inject CSS: floating pill button fixed bottom-right, panel opens upward
    st.markdown("""
    <style>
    /* ── Floating advisor button — fixed bottom-right ── */
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 28px !important;
        right: 28px !important;
        z-index: 999999 !important;
        width: auto !important;
    }
    /* Pill button */
    div[data-testid="stPopover"] > div:first-child > button {
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 22px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.45), 0 2px 8px rgba(0,0,0,0.3) !important;
        cursor: pointer !important;
        transition: box-shadow 0.2s ease, transform 0.2s ease !important;
        white-space: nowrap !important;
    }
    div[data-testid="stPopover"] > div:first-child > button:hover {
        box-shadow: 0 6px 28px rgba(16, 185, 129, 0.65), 0 4px 12px rgba(0,0,0,0.35) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stPopover"] > div:first-child > button:active {
        transform: translateY(0px) !important;
    }
    /* Chat panel — floats upward from the button, spans 50vw */
    div[data-testid="stPopoverBody"] {
        width: 50vw !important;
        min-width: 440px !important;
        max-width: 50vw !important;
        max-height: 80vh !important;
        overflow-y: auto !important;
        background: linear-gradient(180deg, #0d1b2a 0%, #111a2e 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.55), 0 2px 12px rgba(16,185,129,0.1) !important;
        padding: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.popover("🤖 AI Advisor ▲"):
        _advisor_chat_fragment()


@st.fragment
def _advisor_chat_fragment():
    """Fragment-isolated chat UI — only this reruns when messages are sent."""
    selected_county = st.session_state.get("_advisor_county", "Ireland")
    area_context = st.session_state.get("_advisor_area_context", "")
    page_ctx = st.session_state.get("_advisor_page_ctx", {})
    msgs = st.session_state.advisor_messages

    tab_label = page_ctx.get("active_tab", "overview").title()

    # ── Header ────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px; padding-bottom:12px;
                border-bottom:1px solid rgba(16,185,129,0.2);">
        <div style="width:42px; height:42px; border-radius:50%;
                    background:linear-gradient(135deg,#10b981,#3b82f6);
                    display:flex; align-items:center; justify-content:center;
                    font-size:20px; flex-shrink:0; box-shadow:0 2px 8px rgba(16,185,129,0.3);">
            🤖
        </div>
        <div>
            <div style="font-weight:700; font-size:1rem; color:#e2e8f0; letter-spacing:-0.2px;">
                Léarslán AI Advisor
            </div>
            <div style="font-size:0.7rem; color:#10b981; margin-top:2px;">
                🟢 Online &nbsp;·&nbsp; Powered by GBM + TOPSIS + SHAP + ARIMA
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome message or chat history ──────────────────────
    if not msgs:
        st.markdown(f"""
<div style="font-size:0.9rem; line-height:1.65; color:#e2e8f0;">
👋 Hi! I'm your <b>Léarslán AI Advisor</b>. I have live access to ML models covering
<b>255 Electoral Divisions</b> across Ireland.<br><br>
Currently looking at <b>{selected_county}</b>. Try asking me:
</div>
""", unsafe_allow_html=True)
        suggestions = [
            f"Where should I live on a €65k salary?",
            f"Why is {selected_county}'s risk score high?",
            f"What will rent be in 6 months?",
            f"What grants are available for first-time buyers?",
        ]
        for s in suggestions:
            st.markdown(f"- *{s}*")
    else:
        for msg in msgs[-6:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Input row ─────────────────────────────────────────────
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input(
            "Ask anything...", key="advisor_popover_input",
            label_visibility="collapsed",
            placeholder="Ask about rent, areas, grants...",
        )
    with col2:
        send = st.button("➤", key="advisor_send", type="primary", use_container_width=True)

    if user_input and send:
        st.session_state.advisor_messages.append({"role": "user", "content": user_input})
        response = _generate_response(user_input, area_context, page_ctx)
        st.session_state.advisor_messages.append({"role": "assistant", "content": response})
        _trim_history()
        st.rerun(scope="fragment")

    # Clear button (only once there's history)
    if msgs:
        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear conversation", key="advisor_clear", use_container_width=True):
            st.session_state.advisor_messages = []
            st.rerun(scope="fragment")


# ── Internal ──────────────────────────────────────────────────

def _ensure_session_state():
    if "advisor_messages" not in st.session_state:
        st.session_state.advisor_messages = []


def _trim_history():
    if len(st.session_state.advisor_messages) > _MAX_HISTORY:
        st.session_state.advisor_messages = st.session_state.advisor_messages[-_MAX_HISTORY:]


def _generate_response(query, area_context, page_context):
    ml_context = ""
    try:
        from insights.ml_tools import build_ml_context
        ml_context = build_ml_context(
            query=query,
            scored_df=st.session_state.get("_advisor_scored_df", pd.DataFrame()),
            models=st.session_state.get("_advisor_models", {}),
            feature_names=st.session_state.get("_advisor_feature_names", []),
            selected_county=st.session_state.get("_advisor_county", ""),
            ts_data=st.session_state.get("_advisor_ts_data"),
        )
    except Exception as e:
        logger.warning("ML inference failed: %s", e)

    rag_engine = _get_cached_rag_engine()
    rag_results = rag_engine.retrieve(query, top_k=3)

    # Add user preferences context if available
    prefs_context = _get_preferences_context()

    full_system = _assemble_system_prompt(area_context, page_context, rag_results, ml_context, prefs_context)

    try:
        return _call_claude(query, full_system)
    except Exception as e:
        logger.warning("Claude call failed: %s", e)
        return _template_fallback(query, area_context, rag_results, error=str(e))


def _get_preferences_context():
    """Pull user preferences + top matches from session state."""
    parts = []

    # ── User profile (set via 'Tell Us About You' sidebar) ────
    prefs = st.session_state.get("_advisor_prefs")
    if not prefs:
        # Fallback: read directly from Streamlit widget keys
        budget = st.session_state.get("pref_budget")
        if budget:
            prefs = {
                "budget": budget,
                "family_size": st.session_state.get("pref_fam", 1),
                "work_mode": st.session_state.get("pref_work", "remote"),
                "commute": st.session_state.get("pref_comm", "medium"),
                "w_afford": st.session_state.get("pref_wa", 0.3),
                "w_live": st.session_state.get("pref_wl", 0.25),
                "w_safe": st.session_state.get("pref_ws", 0.2),
                "w_trans": st.session_state.get("pref_wt", 0.15),
            }

    if prefs:
        work_labels = {"remote": "🏠 Remote", "hybrid": "🔄 Hybrid", "office": "🏢 Office"}
        comm_labels = {"low": "Low (prefers nearby)", "medium": "Medium", "high": "High (flexible)"}
        fam_labels = {1: "Single", 2: "Couple", 3: "Small Family", 4: "Family", 5: "Large Family"}
        parts.append(
            f"\n--- USER PROFILE (from 'Tell Us About You') ---\n"
            f"Monthly take-home budget: €{prefs['budget']:,}\n"
            f"Household: {fam_labels.get(prefs['family_size'], prefs['family_size'])} ({prefs['family_size']} person(s))\n"
            f"Work mode: {work_labels.get(prefs['work_mode'], prefs['work_mode'])}\n"
            f"Commute tolerance: {comm_labels.get(prefs['commute'], prefs['commute'])}\n"
            f"Priority weights set by user:\n"
            f"  - Affordability: {prefs['w_afford']:.0%}\n"
            f"  - Livability:    {prefs['w_live']:.0%}\n"
            f"  - Low Risk:      {prefs['w_safe']:.0%}\n"
            f"  - Transport:     {prefs['w_trans']:.0%}"
        )

    # ── Top match results (from last TOPSIS run) ──────────────
    matches = st.session_state.get("_advisor_top_matches")
    if matches:
        scope = matches.get("scope", "Ireland")
        level = matches.get("level", "county")
        top3 = matches.get("top3", [])
        medals = ["🥇", "🥈", "🥉"]
        match_lines = [
            f"\n--- LAST MATCH RESULTS (TOPSIS ranking across {scope}) ---",
            f"These are the top 3 best-fit {level}s for the user's profile:",
        ]
        for i, m in enumerate(top3):
            medal = medals[i] if i < len(medals) else f"#{i+1}"
            remaining = m.get('monthly_remaining', 0)
            remaining_str = f"+€{remaining:,.0f} surplus" if remaining >= 0 else f"€{abs(remaining):,.0f} over budget"
            match_lines.append(
                f"  {medal} {m['name']}: Match Score {m['match_score']}/100 | "
                f"Rent €{m['rent']:,.0f}/mo | {remaining_str} | "
                f"Budget Fit: {m['budget_fit']} | "
                f"Risk: {m['risk_score']} | Livability: {m['livability_score']}"
            )
        match_lines.append(
            "IMPORTANT: When the user asks about their matches or recommendations, "
            "reference these specific results. Do NOT say you lack access to match data."
        )
        parts.append("\n".join(match_lines))

    # ── Current map/area selection ────────────────────────────
    county = st.session_state.get("_advisor_county")
    zoom = st.session_state.get("zoom_level", "🇮🇪 Ireland")
    ed_name = st.session_state.get("ed_select", "")
    if county:
        parts.append(
            f"\n--- CURRENT DASHBOARD SELECTION ---\n"
            f"Zoom level: {zoom}\n"
            f"Selected county: {county}"
            + (f"\nSelected neighbourhood: {ed_name.split(' (')[0]}" if ed_name else "")
        )

    return "\n".join(parts)


def _assemble_system_prompt(area_context, page_context, rag_results, ml_context="", prefs_context=""):
    parts = [_SYSTEM_PROMPT]

    desc = page_context.get("natural_description", "browsing the dashboard")
    parts.append(f"\n--- CURRENT PAGE CONTEXT ---\nThe user is currently {desc}.")
    parts.append(f"\n--- AREA DATA (from pre-trained GBM models) ---\n{area_context}")

    if prefs_context:
        parts.append(prefs_context)

    if ml_context:
        parts.append(f"\n--- LIVE MODEL INFERENCE (run just now) ---{ml_context}")
        parts.append(
            "\nIMPORTANT: The live model results above were computed in real-time. "
            "Prioritize these over static data. Cite the model used, e.g. [Model: TOPSIS] or [Model: SHAP]."
        )

    if rag_results:
        parts.append("\n--- REFERENCE DOCUMENTS ---")
        for i, r in enumerate(rag_results, 1):
            source = r["source_title"]
            authority = r.get("authority", "")
            parts.append(
                f"[{i}] {source}" + (f" (Authority: {authority})" if authority else "")
                + f"\n    {r['content'][:400]}"
            )
        parts.append("\nCite these as [Source: document title] when relevant.")

    return "\n".join(parts)


def _call_claude(query, system_prompt):
    """Call Anthropic Claude."""
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set in .env")

    from config import ANTHROPIC_MODEL
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    api_messages = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state.get("advisor_messages", [])]
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=api_messages,
    )
    return response.content[0].text


def _template_fallback(query, area_context, rag_results, error=None):
    """Fallback when LLM is unavailable."""
    lines = area_context.split("\n")
    county = lines[0].replace("=== ", "").replace(" -- Key Metrics ===", "") if lines else "this area"

    metrics = {}
    for line in lines:
        if ":" in line and "===" not in line and "---" not in line:
            key, val = line.split(":", 1)
            metrics[key.strip()] = val.strip()

    parts = [f"## {county} — Quick Summary\n"]
    parts.append(
        f"**{county}** has a risk score of **{metrics.get('Risk Score', 'N/A')}**, "
        f"livability of **{metrics.get('Livability Score', 'N/A')}**, and affordability of "
        f"**{metrics.get('Affordability Score', 'N/A')}**. "
        f"Average monthly rent is **{metrics.get('Avg Monthly Rent', 'N/A')}**.\n"
    )

    if rag_results:
        parts.append("### Relevant Policy Context\n")
        for r in rag_results[:2]:
            parts.append(f"> {r['content'][:200]}...\n> — [Source: {r['source_title']}]\n")

    parts.append("\n*Running in template mode. Check ANTHROPIC_API_KEY for full AI.*")
    if error:
        parts.append(f"\n`Debug: {error}`")

    return "\n".join(parts)
