"""Stub: generate_insight for sidebar panel. Delegates to chat module if available."""

def generate_insight(county, risk_score, risk_label, risk_trend, top_drivers,
                     livability_score, transport_score, county_row):
    """Generate a text insight for the county detail panel."""
    drivers_text = ""
    if top_drivers:
        drivers_text = " Key drivers: " + ", ".join(
            d.get("feature", "?") for d in top_drivers[:3]
        ) + "."

    trend_word = {"Increasing": "rising", "Decreasing": "improving", "Stable": "stable"}.get(risk_trend, risk_trend.lower())

    rent = county_row.get("avg_monthly_rent", 0)
    afford = county_row.get("affordability_score", 50)

    return (
        f"**{county}** has a risk score of **{risk_score:.0f}/100** ({risk_label}), "
        f"trending {trend_word}. Livability is **{livability_score:.0f}** and transport "
        f"scores **{transport_score:.0f}**. Average rent is **€{rent:,.0f}/mo** with an "
        f"affordability rating of **{afford:.0f}/100**.{drivers_text}"
    )
