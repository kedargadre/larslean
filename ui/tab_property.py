"""Tab 2: Property Explorer — Live Daft.ie listings, distributions, and market pulse."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.styles import metric_card
from ui.charts import rent_distribution_histogram, property_type_donut, national_comparison_gauge
from ingestion.daft_client import fetch_county_listings, get_county_market_summary


def render_property_tab(county: str, scores_df: pd.DataFrame):
    """Render the Property Explorer tab."""

    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="section-header">🏠 Property Explorer — {county}</div>
        <span style="color:#94a3b8; font-size:0.9rem;">
            Live property data from Daft.ie • Updates every 15 minutes
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Fetch market summary
    market = get_county_market_summary(county)

    if not market.get("has_live_data"):
        st.warning(f"⚠️ Unable to fetch live Daft.ie data for {county}. "
                   "Showing synthetic data fallback.")
        _show_synthetic_fallback(county, scores_df)
        return

    # ── Toggle: Rental vs Sale ────────────────────────────────
    view_mode = st.radio(
        "View Mode",
        ["🏠 Rental Market", "🏡 Sales Market"],
        horizontal=True,
        key="property_view_mode",
    )

    is_rental = "Rental" in view_mode

    st.markdown("---")

    if is_rental:
        _render_rental_view(county, market)
    else:
        _render_sale_view(county, market)


def _render_rental_view(county: str, market: dict):
    """Render rental market view."""

    # ── Market Pulse KPIs ─────────────────────────────────────
    st.markdown('<div class="section-header">📊 Rental Market Pulse</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(
            "Median Rent", f"€{market['rental_median']:,.0f}",
            "per month", "stable"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Active Listings", f"{market['rental_listing_count']}",
            "on Daft.ie now", "stable"
        ), unsafe_allow_html=True)
    with c3:
        ppb = market.get("rental_price_per_bedroom", 0)
        st.markdown(metric_card(
            "Price / Bedroom", f"€{ppb:,.0f}",
            "avg per bed", "stable"
        ), unsafe_allow_html=True)
    with c4:
        price_range = f"€{market['rental_min']:,.0f} — €{market['rental_max']:,.0f}"
        st.markdown(metric_card(
            "Price Range", price_range,
            "", "stable"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row ────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        fig_hist = rent_distribution_histogram(
            market["rental_prices"], county, market["rental_median"]
        )
        st.plotly_chart(fig_hist, width='stretch', config={"displayModeBar": False})

    with chart_col2:
        fig_donut = property_type_donut(market["rental_types"], "Rental Property Types")
        st.plotly_chart(fig_donut, width='stretch', config={"displayModeBar": False})

    st.markdown("---")

    # ── Live Listings Table ───────────────────────────────────
    st.markdown('<div class="section-header">📋 Live Rental Listings</div>', unsafe_allow_html=True)

    rent_df = fetch_county_listings(county, search_type="rent")
    if len(rent_df) > 0:
        display_df = rent_df[["title", "price_display", "bedrooms", "bathrooms", "property_type"]].copy()
        display_df.columns = ["Property", "Price", "Beds", "Baths", "Type"]

        st.dataframe(
            display_df,
            width='stretch',
            height=400,
            hide_index=True,
            column_config={
                "Property": st.column_config.TextColumn("Property", width="large"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Beds": st.column_config.NumberColumn("Beds", width="small"),
                "Baths": st.column_config.NumberColumn("Baths", width="small"),
                "Type": st.column_config.TextColumn("Type", width="small"),
            }
        )

        # Show Daft links
        with st.expander("🔗 View on Daft.ie"):
            for _, row in rent_df.head(10).iterrows():
                if row["daft_link"]:
                    st.markdown(f"- [{row['title'][:60]}...]({row['daft_link']})")
    else:
        st.info("No rental listings found for this county.")


def _render_sale_view(county: str, market: dict):
    """Render sales market view."""

    # ── Market Pulse KPIs ─────────────────────────────────────
    st.markdown('<div class="section-header">📊 Sales Market Pulse</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(
            "Median Price", f"€{market['sale_median']:,.0f}",
            "", "stable"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Active Listings", f"{market['sale_listing_count']}",
            "for sale on Daft.ie", "stable"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "Lowest Price", f"€{market['sale_min']:,.0f}",
            "", "down"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(
            "Highest Price", f"€{market['sale_max']:,.0f}",
            "", "up"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row ────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        fig_hist = rent_distribution_histogram(
            market["sale_prices"], county, market["sale_median"]
        )
        st.plotly_chart(fig_hist, width='stretch', config={"displayModeBar": False})

    with chart_col2:
        fig_donut = property_type_donut(market["sale_types"], "Property Types for Sale")
        st.plotly_chart(fig_donut, width='stretch', config={"displayModeBar": False})

    st.markdown("---")

    # ── Live Listings Table ───────────────────────────────────
    st.markdown('<div class="section-header">📋 Properties for Sale</div>', unsafe_allow_html=True)

    sale_df = fetch_county_listings(county, search_type="sale")
    if len(sale_df) > 0:
        display_df = sale_df[["title", "price_display", "bedrooms", "bathrooms", "property_type"]].copy()
        display_df.columns = ["Property", "Price", "Beds", "Baths", "Type"]

        st.dataframe(
            display_df,
            width='stretch',
            height=400,
            hide_index=True,
        )

        with st.expander("🔗 View on Daft.ie"):
            for _, row in sale_df.head(10).iterrows():
                if row["daft_link"]:
                    st.markdown(f"- [{row['title'][:60]}...]({row['daft_link']})")
    else:
        st.info("No sale listings found for this county.")


def _show_synthetic_fallback(county: str, scores_df: pd.DataFrame):
    """Show synthetic data when Daft.ie is unreachable."""
    county_row = scores_df[scores_df["county"] == county]
    if len(county_row) > 0:
        cr = county_row.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg Monthly Rent", f"€{cr.get('avg_monthly_rent', 0):,.0f}")
        with c2:
            st.metric("Rent Growth", f"{cr.get('rent_growth_pct', 0)*100:+.1f}%")
        with c3:
            st.metric("Energy Cost/yr", f"€{cr.get('est_annual_energy_cost', 0):,.0f}")
