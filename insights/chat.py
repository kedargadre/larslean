"""
AI Advisor chat module for the Léarslán dashboard.

Renders the advisor tab (called from app.py) and the floating sidebar widget.
Powered by Gemini 2.0 Flash with hybrid RAG context injection.
Falls back to template-based responses when Gemini is unavailable.
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
    """Build RAG engine once, persist across all Streamlit reruns."""
    return get_rag_engine()

_MAX_HISTORY = 20
_MODEL_NAME = "gpt-4o-mini"

# ── System Prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = dedent("""\
    You are **Léarslán AI**, an expert Irish community intelligence advisor built
    into the Léarslán dashboard. You help users understand cost-of-living data,
    housing affordability, risk scores, transport connectivity, and energy
    efficiency across Ireland's 26 counties and 255 Electoral Divisions.

    PERSONALITY:
    - Warm, knowledgeable, concise. You speak like a trusted local advisor.
    - Use Irish place names naturally. Reference specific data points.
    - Be opinionated when the data supports it — don't hedge unnecessarily.

    CAPABILITIES:
    - Explain ML scores (risk, livability, transport, affordability) and what drives them.
    - Compare areas using real metrics.
    - Advise on relocation decisions based on user priorities.
    - Reference Irish housing policy, grants, and government schemes.
    - Interpret SHAP feature importance to explain why an area scores high/low.

    CITATION RULES (CRITICAL):
    - When referencing policy documents or reports, cite with [Source: document title]
    - When referencing data metrics, cite with [Data: metric_name = value]
    - When referencing SHAP/model analysis, cite with [Model: SHAP analysis]
    - Include at least one citation per answer when data or documents are relevant.
    - If you cannot ground an answer in the available data or documents, say so honestly.

    FORMATTING:
    - Use markdown for readability (bold, bullet points, headers).
    - Keep answers focused — 3-5 paragraphs max unless the user asks for detail.
    - Use € for currency, Irish conventions for place names.
""")


# ── Public API (called from app.py) ──────────────────────────────────────────

def render_advisor_tab(
    selected_county: str,
    scores_df: pd.DataFrame,
    drivers: list | None = None,
    market: dict | None = None,
    models: dict | None = None,
    feature_names: list | None = None,
    ts_data: dict | None = None,
):
    """
    Render the AI Advisor as a full tab.

    This is the function imported by app.py:
        from insights.chat import render_advisor_tab
    """
    st.markdown("### 🤖 Léarslán AI Advisor")
    st.caption(
        f"Context: **{selected_county}** · Powered by live ML models (GBM, TOPSIS, SHAP, ARIMA)"
    )

    _ensure_session_state()

    # Store models in session state for live inference
    st.session_state["_advisor_models"] = models or {}
    st.session_state["_advisor_feature_names"] = feature_names or []
    st.session_state["_advisor_scored_df"] = scores_df
    st.session_state["_advisor_ts_data"] = ts_data
    st.session_state["_advisor_county"] = selected_county

    # Build context
    area_context = build_area_context(selected_county, scores_df, drivers, market)
    page_ctx = build_page_context(
        active_tab="advisor",
        selected_county=selected_county,
    )

    # Render chat history
    for msg in st.session_state.advisor_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask Léarslán AI about this area..."):
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.advisor_messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = _generate_response(prompt, area_context, page_ctx)
            st.markdown(response)
        st.session_state.advisor_messages.append({"role": "assistant", "content": response})

        _trim_history()


def render_floating_advisor(
    selected_county: str,
    scores_df: pd.DataFrame,
    drivers: list | None = None,
    market: dict | None = None,
    page_context: dict | None = None,
):
    """
    Render the AI Advisor as a sidebar expander (visible on all tabs).

    Call this from app.py OUTSIDE of any tab block, in the sidebar section.
    """
    _ensure_session_state()

    area_context = build_area_context(selected_county, scores_df, drivers, market)
    page_ctx = page_context or build_page_context(
        active_tab="overview",
        selected_county=selected_county,
    )

    with st.sidebar:
        st.markdown("---")
        with st.expander("🤖 AI Advisor", expanded=False):
            tab_label = page_ctx.get("active_tab", "overview").title()
            st.caption(f"👀 Looking at: **{tab_label}** · **{selected_county}**")

            for msg in st.session_state.advisor_messages[-6:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask Léarslán AI...", key="floating_advisor_input"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.advisor_messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = _generate_response(prompt, area_context, page_ctx)
                    st.markdown(response)
                st.session_state.advisor_messages.append({"role": "assistant", "content": response})

                _trim_history()


# ── Internal ──────────────────────────────────────────────────────────────────

def _ensure_session_state():
    if "advisor_messages" not in st.session_state:
        st.session_state.advisor_messages = []


def _trim_history():
    if len(st.session_state.advisor_messages) > _MAX_HISTORY:
        st.session_state.advisor_messages = st.session_state.advisor_messages[-_MAX_HISTORY:]


def _generate_response(query: str, area_context: str, page_context: dict) -> str:
    """Generate a response using live ML inference + RAG + LLM."""
    # 1. Run live ML models based on query intent
    from insights.ml_tools import build_ml_context
    ml_context = ""
    try:
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

    # 2. Retrieve RAG context
    rag_engine = _get_cached_rag_engine()
    rag_results = rag_engine.retrieve(query, top_k=3)

    # 3. Assemble full prompt with ML context
    full_system = _assemble_system_prompt(area_context, page_context, rag_results, ml_context)

    # 4. Try LLM — single API call per message
    try:
        result = _call_gemini(query, full_system)
        return result
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        st.warning(f"LLM error: {e}")
        return _template_fallback(query, area_context, rag_results)


def _assemble_system_prompt(
    area_context: str,
    page_context: dict,
    rag_results: list[dict],
    ml_context: str = "",
) -> str:
    """Build the full system prompt with all context layers."""
    parts = [_SYSTEM_PROMPT]

    # Layer 1: Page context
    desc = page_context.get("natural_description", "browsing the dashboard")
    parts.append(f"\n--- CURRENT PAGE CONTEXT ---\nThe user is currently {desc}.")

    # Layer 2: Area data (static scores from pre-trained models)
    parts.append(f"\n--- AREA DATA (from pre-trained GBM models) ---\n{area_context}")

    # Layer 3: Live ML inference results
    if ml_context:
        parts.append(f"\n--- LIVE MODEL INFERENCE (run just now for this query) ---{ml_context}")
        parts.append(
            "\nIMPORTANT: The live model results above were computed in real-time "
            "using the trained ML models. Prioritize these over static data when answering. "
            "Always cite the model used, e.g. [Model: TOPSIS ranking] or [Model: SHAP analysis]."
        )

    # Layer 3: RAG documents
    if rag_results:
        parts.append("\n--- REFERENCE DOCUMENTS ---")
        for i, r in enumerate(rag_results, 1):
            source = r["source_title"]
            authority = r["authority"]
            method = r.get("method", "unknown")
            parts.append(
                f"[{i}] {source}"
                + (f" (Authority: {authority})" if authority else "")
                + f" [retrieval: {method}, score: {r['score']:.3f}]"
                + f"\n    {r['content'][:400]}"
            )
        parts.append(
            "\nUse these references to ground your answers. "
            "Cite them as [Source: document title] when relevant."
        )

    return "\n".join(parts)


def _call_gemini(query: str, system_prompt: str) -> str:
    """Call LLM via LiteLLM (OpenAI-compatible) endpoint."""
    import requests
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    base_url = os.getenv("LITELLM_BASE_URL", "")
    api_key = os.getenv("LITELLM_API_KEY", "").strip('"')

    if not base_url or not api_key:
        raise ValueError("LITELLM_BASE_URL or LITELLM_API_KEY not configured in .env")

    # Build messages: system + history + current query
    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.get("advisor_messages", [])[:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": _MODEL_NAME, "messages": messages, "max_tokens": 1024, "temperature": 0.7},
        verify=False,
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} {resp.text[:300]}")

    return resp.json()["choices"][0]["message"]["content"]


def _template_fallback(query: str, area_context: str, rag_results: list[dict]) -> str:
    """Rule-based fallback when Gemini is unavailable."""
    lines = area_context.split("\n")
    county = lines[0].replace("=== ", "").replace(" — Key Metrics ===", "") if lines else "this area"

    # Extract key metrics for template
    metrics = {}
    for line in lines:
        if ":" in line and "===" not in line and "---" not in line:
            key, val = line.split(":", 1)
            metrics[key.strip()] = val.strip()

    response_parts = [f"## {county} — Quick Summary\n"]

    risk = metrics.get("Risk Score", "N/A")
    livability = metrics.get("Livability Score", "N/A")
    affordability = metrics.get("Affordability Score", "N/A")
    rent = metrics.get("Avg Monthly Rent", "N/A")

    response_parts.append(
        f"**{county}** has a risk score of **{risk}**, "
        f"livability of **{livability}**, and affordability of **{affordability}**. "
        f"Average monthly rent is **{rent}**.\n"
    )

    # Add RAG context if available
    if rag_results:
        response_parts.append("### Relevant Policy Context\n")
        for r in rag_results[:2]:
            response_parts.append(
                f"> {r['content'][:200]}...\n"
                f"> — [Source: {r['source_title']}]\n"
            )

    response_parts.append(
        "\n*⚠️ AI Advisor is running in template mode. "
        "Configure GEMINI_API_KEY in .env for full conversational AI.*"
    )

    return "\n".join(response_parts)
