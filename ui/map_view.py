"""Map view — Folium choropleth renderer."""
import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "real_data"


def render_map(scores_df, metric, level="county", county=None):
    if level == "ed":
        geo_path = DATA_DIR / "ireland_eds.geojson"
        key_col = "ed_id"
        geo_key = "feature.properties.ed_id"
    else:
        geo_path = DATA_DIR / "ireland_counties.geojson"
        key_col = "county"
        geo_key = "feature.properties.name"

    if not geo_path.exists():
        st.warning(f"GeoJSON not found: {geo_path}")
        return

    with open(geo_path, encoding="utf-8") as f:
        geo_data = json.load(f)

    m = folium.Map(location=[53.5, -7.5], zoom_start=7, tiles="CartoDB dark_matter")

    if key_col in scores_df.columns and metric in scores_df.columns:
        folium.Choropleth(
            geo_data=geo_data,
            data=scores_df,
            columns=[key_col, metric],
            key_on=geo_key,
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name=metric.replace("_", " ").title(),
        ).add_to(m)

    st_folium(m, width=700, height=450, returned_objects=[])
