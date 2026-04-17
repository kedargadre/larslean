"""Custom CSS & Dark Theme V2 for Léarslán Dashboard."""
import streamlit as st


def inject_css():
    """Inject premium light theme CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    .stApp > header { background: transparent; }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.3);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #334155;
    }

    /* ── Tabs ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(203, 213, 225, 0.5);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #1e293b;
        background: rgba(59, 130, 246, 0.06);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.1)) !important;
        color: #0f172a !important;
        border-bottom: 2px solid #10b981;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, #10b981, #3b82f6) !important;
        height: 3px;
        border-radius: 2px;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Cards ────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.8));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        color: #1e293b;
    }

    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.12);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(135deg, #059669, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .metric-delta {
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 4px;
    }

    .delta-up { color: #dc2626; }
    .delta-down { color: #059669; }
    .delta-stable { color: #d97706; }

    /* ── Score Badge ──────────────────────────────── */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 24px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .badge-high {
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    .badge-medium {
        background: rgba(245, 158, 11, 0.1);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .badge-low {
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    /* ── Title ────────────────────────────────────── */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #059669 0%, #2563eb 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-top: 4px;
        font-weight: 400;
    }

    /* ── Insight Box ──────────────────────────────── */
    .insight-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.04), rgba(59, 130, 246, 0.02));
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        line-height: 1.7;
        font-size: 0.95rem;
        color: #334155;
    }

    /* ── Section Headers ─────────────────────────── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(59, 130, 246, 0.2);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Data Tables ─────────────────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Chat Messages ───────────────────────────── */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* ── Selectbox ────────────────────────────────── */
    div[data-baseweb="select"] {
        border-radius: 12px;
    }

    /* ── Plotly Charts ────────────────────────────── */
    .js-plotly-plot .plotly {
        border-radius: 12px;
    }

    /* ── Map Container ───────────────────────────── */
    iframe {
        border-radius: 16px;
        border: 1px solid rgba(203, 213, 225, 0.6);
    }

    /* ── Loading Animation ───────────────────────── */
    .stSpinner > div {
        border-top-color: #10b981;
    }

    /* ── Divider ──────────────────────────────────── */
    hr {
        border-color: rgba(203, 213, 225, 0.6);
        margin: 20px 0;
    }

    /* ── Pill Radio (zoom level selector) ────────── */
    .stRadio > div {
        gap: 0px;
        background: rgba(241, 245, 249, 0.8);
        border-radius: 10px;
        padding: 3px;
        border: 1px solid rgba(203, 213, 225, 0.6);
        display: inline-flex !important;
    }

    .stRadio [data-baseweb="radio"] {
        background: transparent;
        border-radius: 7px;
        padding: 5px 16px;
        transition: background 0.15s ease, color 0.15s ease;
    }

    /* Unselected pill — dark enough text to read over light bg */
    .stRadio [data-baseweb="radio"] label,
    .stRadio [data-baseweb="radio"] label p,
    .stRadio [data-baseweb="radio"] label div {
        color: #475569 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        opacity: 1 !important;
    }

    .stRadio [data-baseweb="radio"]:hover label,
    .stRadio [data-baseweb="radio"]:hover label p,
    .stRadio [data-baseweb="radio"]:hover label div {
        color: #1e293b !important;
    }

    /* Selected pill highlight */
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25);
    }

    .stRadio [data-baseweb="radio"][aria-checked="true"] label,
    .stRadio [data-baseweb="radio"][aria-checked="true"] label p,
    .stRadio [data-baseweb="radio"][aria-checked="true"] label div {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ── Scrollbar ────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #f8fafc;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* ── Slider Labels ────────────────────────────── */
    .stSlider label p, .stSlider label span {
        color: #1e293b !important;
        font-weight: 500;
    }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #94a3b8 !important;
    }

    /* ── Right AI Advisor Drawer ─────────────────── */
    .advisor-drawer-title {
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .advisor-welcome {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(59, 130, 246, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 16px;
        font-size: 0.9rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 40%;
    }

    /* ── Hide Streamlit Branding ──────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Info Tooltip Widget ─────────────────────── */
    .info-tooltip-wrap {
        display: inline-block;
        position: relative;
        margin-left: 5px;
        vertical-align: middle;
        cursor: help;
    }
    .info-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #cbd5e1;
        color: #475569;
        font-size: 10px;
        font-weight: 700;
        font-style: normal;
        line-height: 1;
        transition: background 0.2s ease;
        user-select: none;
    }
    .info-tooltip-wrap:hover .info-icon {
        background: #3b82f6;
        color: #ffffff;
    }
    .tooltip-box {
        visibility: hidden;
        opacity: 0;
        pointer-events: none;
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        width: 260px;
        background: #1e293b;
        color: #e2e8f0;
        font-size: 0.78rem;
        line-height: 1.55;
        padding: 10px 13px;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        border: 1px solid rgba(59,130,246,0.25);
        z-index: 99999;
        text-align: left;
        transition: opacity 0.2s ease, visibility 0.2s ease;
        white-space: normal;
    }
    .tooltip-box::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #1e293b;
    }
    .info-tooltip-wrap:hover .tooltip-box {
        visibility: visible;
        opacity: 1;
    }
    /* Map header info icon (bigger) */
    .map-metric-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .map-metric-header .info-icon {
        width: 20px;
        height: 20px;
        font-size: 12px;
    }
    .map-metric-header .tooltip-box {
        width: 300px;
        font-size: 0.82rem;
    }
    </style>
    """, unsafe_allow_html=True)


def info_tooltip(text: str) -> str:
    """Generate an inline ℹ info icon with a hover tooltip."""
    safe_text = text.replace('"', '&quot;').replace("'", "&#39;")
    return (
        f'<span class="info-tooltip-wrap">'
        f'<i class="info-icon">i</i>'
        f'<span class="tooltip-box">{text}</span>'
        f'</span>'
    )


def metric_card(label: str, value: str, delta: str = "", delta_type: str = "stable", tooltip: str = "") -> str:
    """Generate HTML for a metric card with an optional info tooltip."""
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    tip_html = info_tooltip(tooltip) if tooltip else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}{tip_html}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def score_badge(label: str, score: float) -> str:
    """Generate HTML for a score badge."""
    if score >= 67:
        badge_class = "badge-high"
    elif score >= 34:
        badge_class = "badge-medium"
    else:
        badge_class = "badge-low"
    return f'<span class="score-badge {badge_class}">● {label}: {score:.0f}</span>'
