"""UI styles — CSS injection and metric card renderer."""
import streamlit as st


def inject_css():
    st.markdown("""
    <style>
    .dashboard-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
    .dashboard-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.8;
    }
    .metric-card {
        background: rgba(100,100,100,0.08);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; }
    .metric-card .label { font-size: 0.78rem; opacity: 0.7; }
    .metric-card .sub { font-size: 0.7rem; opacity: 0.5; }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, subtitle: str = "", trend: str = "stable") -> str:
    color = {"up": "#ef4444", "down": "#10b981", "stable": "#3b82f6"}.get(trend, "#3b82f6")
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{color}">{value}</div>
        <div class="sub">{subtitle}</div>
    </div>
    """
