"""Custom CSS & Dark Theme V2 for Léarslán Dashboard."""
import streamlit as st


def inject_css():
    """Inject premium dark theme CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0f1f3d 50%, #0a1628 100%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    .stApp > header { background: transparent; }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #111a2e 100%);
        border-right: 1px solid rgba(30, 41, 59, 0.5);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* ── Tabs ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(17, 26, 46, 0.6);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(30, 41, 59, 0.5);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background: rgba(59, 130, 246, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(59, 130, 246, 0.15)) !important;
        color: #e2e8f0 !important;
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
        background: linear-gradient(135deg, rgba(17, 26, 46, 0.9), rgba(26, 39, 68, 0.7));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }

    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .metric-delta {
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 4px;
    }

    .delta-up { color: #ef4444; }
    .delta-down { color: #10b981; }
    .delta-stable { color: #f59e0b; }

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
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* ── Title ────────────────────────────────────── */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 400;
    }

    /* ── Insight Box ──────────────────────────────── */
    .insight-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(59, 130, 246, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    /* ── Section Headers ─────────────────────────── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
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
        border: 1px solid rgba(30, 41, 59, 0.5);
    }

    /* ── Loading Animation ───────────────────────── */
    .stSpinner > div {
        border-top-color: #10b981;
    }

    /* ── Divider ──────────────────────────────────── */
    hr {
        border-color: rgba(30, 41, 59, 0.5);
        margin: 20px 0;
    }

    /* ── Pill Radio (zoom level selector) ────────── */
    .stRadio > div {
        gap: 0px;
        background: rgba(30, 41, 59, 0.55);
        border-radius: 10px;
        padding: 3px;
        border: 1px solid rgba(71, 85, 105, 0.5);
        display: inline-flex !important;
    }

    .stRadio [data-baseweb="radio"] {
        background: transparent;
        border-radius: 7px;
        padding: 5px 16px;
        transition: background 0.15s ease, color 0.15s ease;
    }

    /* Unselected pill — bright-enough text to read over dark bg */
    .stRadio [data-baseweb="radio"] label,
    .stRadio [data-baseweb="radio"] label p,
    .stRadio [data-baseweb="radio"] label div {
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        opacity: 1 !important;
    }

    .stRadio [data-baseweb="radio"]:hover label,
    .stRadio [data-baseweb="radio"]:hover label p,
    .stRadio [data-baseweb="radio"]:hover label div {
        color: #ffffff !important;
    }

    /* Selected pill highlight */
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.35);
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
        background: #0a1628;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* ── Slider Labels ────────────────────────────── */
    .stSlider label p, .stSlider label span {
        color: #e2e8f0 !important;
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
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", delta_type: str = "stable") -> str:
    """Generate HTML for a metric card."""
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
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
