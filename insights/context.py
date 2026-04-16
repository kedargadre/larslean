"""
Page context collector for the Léarslán AI Advisor.

Assembles a structured dict describing what the user is currently viewing
(tab, area, metric, tab-specific data) so the LLM can give context-aware answers.
"""

import pandas as pd


_TAB_DESCRIPTIONS = {
    "overview": "viewing the national overview map coloured by {metric}",
    "property": "browsing property listings in {county}",
    "duel": "comparing two areas side-by-side",
    "clusters": "exploring neighbourhood archetype clusters (KMeans + UMAP)",
    "budget": "simulating monthly living costs in {county}",
    "forecast": "viewing ARIMA rent/metric forecasts for {county}",
    "recommender": "using the TOPSIS 'Where Should I Live?' recommender",
    "advisor": "chatting with the AI Advisor",
}


def build_page_context(
    active_tab: str,
    selected_county: str,
    selected_ed_id: str | None = None,
    selected_ed_name: str | None = None,
    selected_metric: str = "risk_score",
    spatial_level: str = "county",
    **tab_specific,
) -> dict:
    """
    Build the context dict injected into the advisor's system prompt.

    Called from app.py with the current UI state. The advisor uses this
    to understand what the user is looking at without being told explicitly.
    """
    ctx = {
        "active_tab": active_tab,
        "spatial_level": spatial_level,
        "selected_county": selected_county,
        "selected_ed_id": selected_ed_id,
        "selected_ed_name": selected_ed_name,
        "selected_metric": selected_metric,
        "tab_specific": tab_specific,
    }

    template = _TAB_DESCRIPTIONS.get(active_tab, "browsing the dashboard")
    ctx["natural_description"] = template.format(
        metric=_format_metric(selected_metric),
        county=selected_county,
    )

    return ctx


def build_area_context(
    county: str,
    scores_df: pd.DataFrame,
    drivers: list | None = None,
    market: dict | None = None,
) -> str:
    """
    Build a natural-language data summary for the selected area.

    This is the structured data block injected into the system prompt so
    the LLM can reference real numbers in its answers.
    """
    row = scores_df[scores_df["county"] == county]
    if row.empty:
        return f"No data available for {county}."

    r = row.iloc[0]
    lines = [f"=== {county} — Key Metrics ==="]

    _add(lines, r, "risk_score", "Risk Score", "/100")
    _add(lines, r, "livability_score", "Livability Score", "/100")
    _add(lines, r, "transport_score", "Transport Score", "/100")
    _add(lines, r, "affordability_score", "Affordability Score", "/100")
    _add(lines, r, "avg_monthly_rent", "Avg Monthly Rent", "", "€")
    _add(lines, r, "avg_income", "Avg Annual Income", "", "€")
    _add(lines, r, "employment_rate", "Employment Rate", "", fmt_pct=True)
    _add(lines, r, "congestion_delay_minutes", "Congestion Delay", " min")
    _add(lines, r, "ber_avg_score", "BER Avg Score", "")
    _add(lines, r, "est_annual_energy_cost", "Est Annual Energy Cost", "", "€")
    _add(lines, r, "true_cost_index", "True Cost Index", "/100")

    if "cluster_category" in r.index:
        lines.append(f"Cluster: {r['cluster_category']}")
    if "risk_label" in r.index:
        lines.append(f"Risk Label: {r['risk_label']}")

    # SHAP drivers
    if drivers:
        lines.append("\n--- Top Risk Drivers (SHAP) ---")
        for d in drivers[:5]:
            feat = d.get("feature", d.get("feature_raw", "?"))
            direction = d.get("direction", "")
            impact = d.get("impact", d.get("shap_value", 0))
            lines.append(f"  {direction} {feat}: impact={impact:.2f}")

    # Market data
    if market and market.get("has_live_data"):
        lines.append("\n--- Live Market Data (Daft.ie) ---")
        lines.append(f"  Rental listings: {market.get('rental_listing_count', 'N/A')}")
        lines.append(f"  Median rent: €{market.get('rental_median', 'N/A')}")
        lines.append(f"  Sale listings: {market.get('sale_listing_count', 'N/A')}")
        lines.append(f"  Median sale price: €{market.get('sale_median', 'N/A')}")

    return "\n".join(lines)


def _add(lines, row, col, label, suffix="", prefix="", fmt_pct=False):
    if col in row.index:
        val = row[col]
        if fmt_pct:
            lines.append(f"{label}: {val * 100:.1f}%")
        elif isinstance(val, float):
            lines.append(f"{label}: {prefix}{val:,.1f}{suffix}")
        else:
            lines.append(f"{label}: {prefix}{val}{suffix}")


def _format_metric(metric: str) -> str:
    return metric.replace("_", " ").title()
