"""Folium Choropleth Map Component — supports County and ED level."""
import folium
import json
import pandas as pd
from streamlit_folium import st_folium
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAP_CENTER, MAP_ZOOM, GEOJSON_FILE, ED_GEOJSON_FILE, COUNTY_CENTROIDS


def render_map(
    scores_df: pd.DataFrame,
    selected_metric: str = "risk_score",
    level: str = "county",
    county: str = None,
) -> dict:
    """
    Render Folium choropleth map.
    
    Args:
        scores_df: DataFrame with metric scores (county or ED level)
        selected_metric: column name to color by
        level: "county" or "ed"
        county: When level="ed", the county to zoom into
    """
    if level == "ed":
        return _render_ed_map(scores_df, selected_metric, county)
    else:
        return _render_county_map(scores_df, selected_metric)


def _render_county_map(scores_df: pd.DataFrame, selected_metric: str) -> dict:
    """Render county-level choropleth map."""
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB Dark",
        name="Dark",
    ).add_to(m)

    color_map = {
        "risk_score": "YlOrRd",
        "livability_score": "YlGn",
        "transport_score": "YlGnBu",
        "affordability_score": "RdYlGn",
    }
    metric_labels = {
        "risk_score": "Risk Score",
        "livability_score": "Livability Score",
        "transport_score": "Transport Score",
        "affordability_score": "Affordability Score",
    }

    fill_color = color_map.get(selected_metric, "YlOrRd")
    legend_name = metric_labels.get(selected_metric, selected_metric)

    if GEOJSON_FILE.exists():
        try:
            with open(GEOJSON_FILE, "r") as f:
                geojson_data = json.load(f)

            # ── Patch mislabelled county names ────────────────────
            # Cork's GeoJSON feature has name='NA' instead of 'Cork'
            # (name_irish='Corcaigh', iso='IE-CO'). Fix at load time.
            _GEOJSON_NAME_FIXES = {
                "NA": "Cork",  # iso=IE-CO, name_irish=Corcaigh
            }
            for feature in geojson_data["features"]:
                raw_name = feature["properties"].get("name", "")
                if raw_name in _GEOJSON_NAME_FIXES:
                    feature["properties"]["name"] = _GEOJSON_NAME_FIXES[raw_name]

            choropleth = folium.Choropleth(
                geo_data=geojson_data,
                name="choropleth",
                data=scores_df,
                columns=["county", selected_metric],
                key_on="feature.properties.name",
                fill_color=fill_color,
                fill_opacity=0.7,
                line_opacity=0.3,
                line_color="#1e293b",
                legend_name=legend_name,
                highlight=True,
            )
            choropleth.add_to(m)

            score_dict = dict(zip(scores_df["county"], scores_df[selected_metric]))
            for feature in geojson_data["features"]:
                county_name = feature["properties"]["name"]
                score = score_dict.get(county_name, 0)
                feature["properties"]["score"] = f"{score:.0f}"

            geojson_layer = folium.GeoJson(
                geojson_data,
                style_function=lambda x: {"fillOpacity": 0, "weight": 0},
                tooltip=folium.GeoJsonTooltip(
                    fields=["name", "score"],
                    aliases=["County:", f"{legend_name}:"],
                    style="background-color: #111a2e; color: #e2e8f0; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; font-family: Inter, sans-serif;",
                ),
            )
            geojson_layer.add_to(m)

        except Exception as e:
            st.warning(f"Choropleth rendering error: {e}. Using marker fallback.")
            _add_circle_markers(m, scores_df, selected_metric)
    else:
        _add_circle_markers(m, scores_df, selected_metric)

    # County labels
    for county, (lat, lon) in COUNTY_CENTROIDS.items():
        score = scores_df[scores_df["county"] == county][selected_metric].values
        score_val = score[0] if len(score) > 0 else 0

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px; font-weight:600; color:#e2e8f0; '
                     f'text-shadow: 0 0 4px #0a1628, 0 0 8px #0a1628; '
                     f'white-space:nowrap; font-family:Inter,sans-serif;">'
                     f'{county}<br><span style="color:{"#ef4444" if score_val > 66 else "#f59e0b" if score_val > 33 else "#10b981"}">'
                     f'{score_val:.0f}</span></div>',
                icon_size=(80, 30),
                icon_anchor=(40, 15),
            ),
        ).add_to(m)

    output = st_folium(
        m, width=None, height=550,
        returned_objects=["last_object_clicked"],
        key="ireland_map",
    )
    return output


def _render_ed_map(scores_df: pd.DataFrame, selected_metric: str, county: str = None) -> dict:
    """Render ED-level map — choropleth polygons or circle markers for EDs within a county."""
    # Center on county centroid
    if county and county in COUNTY_CENTROIDS:
        center = list(COUNTY_CENTROIDS[county])
        zoom = 10
    else:
        center = MAP_CENTER
        zoom = MAP_ZOOM

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB Dark",
        name="Dark",
    ).add_to(m)

    color_map = {
        "risk_score": "YlOrRd",
        "livability_score": "YlGn",
        "transport_score": "YlGnBu",
        "affordability_score": "RdYlGn",
    }
    metric_labels = {
        "risk_score": "Risk Score",
        "livability_score": "Livability Score",
        "transport_score": "Transport Score",
        "affordability_score": "Affordability Score",
    }

    fill_color = color_map.get(selected_metric, "YlOrRd")
    legend_name = metric_labels.get(selected_metric, selected_metric)

    # Circle markers at ED centroids (derived from GeoJSON polygons if available,
    # otherwise spread around the county centroid as a fallback).
    if ED_GEOJSON_FILE.exists():
        try:
            with open(ED_GEOJSON_FILE, "r") as f:
                geojson_data = json.load(f)

            # Filter to selected county
            if county:
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": [
                        f for f in geojson_data["features"]
                        if f["properties"].get("county") == county
                    ],
                }

            _add_ed_circles_from_geojson(
                m, geojson_data, scores_df, selected_metric, fill_color, legend_name,
            )
        except Exception as e:
            st.warning(f"ED circle render error: {e}. Using fallback markers.")
            _add_ed_circle_markers(m, scores_df, selected_metric, county)
    else:
        _add_ed_circle_markers(m, scores_df, selected_metric, county)

    output = st_folium(
        m, width=None, height=550,
        returned_objects=["last_object_clicked"],
        key="ed_map",
    )
    return output


def _add_ed_circles_from_geojson(
    m, geojson_data, scores_df, selected_metric, fill_color, legend_name,
):
    """Draw a circle marker at each ED polygon centroid, coloured by score."""
    import branca.colormap as cm

    palette_map = {
        "YlOrRd": ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        "YlGn":   ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"],
        "YlGnBu": ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
        "RdYlGn": ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"],
    }
    colors = palette_map.get(fill_color, palette_map["YlOrRd"])

    score_dict = dict(zip(scores_df["ed_id"], scores_df[selected_metric])) \
        if "ed_id" in scores_df.columns else {}
    name_dict = dict(zip(scores_df["ed_id"], scores_df.get("ed_name", scores_df.get("ed_id", [])))) \
        if "ed_id" in scores_df.columns else {}

    all_scores = [v for v in score_dict.values() if v is not None]
    vmin = min(all_scores) if all_scores else 0
    vmax = max(all_scores) if all_scores else 100
    if vmax == vmin:
        vmax = vmin + 1

    colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=legend_name)

    def _centroid(coords):
        # Supports Polygon (list of rings) and MultiPolygon (list of polygons)
        pts = []
        def _walk(x):
            if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(c, (int, float)) for c in x):
                pts.append(x)
            elif isinstance(x, (list, tuple)):
                for item in x:
                    _walk(item)
        _walk(coords)
        if not pts:
            return None
        lon = sum(p[0] for p in pts) / len(pts)
        lat = sum(p[1] for p in pts) / len(pts)
        return (lat, lon)

    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        ed_id = props.get("ed_id") or props.get("ED_ID") or props.get("id")
        ed_name = props.get("ed_name") or name_dict.get(ed_id) or str(ed_id or "ED")
        county_nm = props.get("county", "")
        ed_type = props.get("ed_type", "")

        geom = feature.get("geometry", {})
        centroid = _centroid(geom.get("coordinates", []))
        if not centroid:
            continue

        score = score_dict.get(ed_id)
        if score is None:
            continue

        color = colormap(score)
        tooltip_html = (
            f"<b>{ed_name}</b><br>"
            + (f"{county_nm}<br>" if county_nm else "")
            + (f"<i>{ed_type}</i><br>" if ed_type else "")
            + f"{legend_name}: {score:.0f}"
        )

        folium.CircleMarker(
            location=centroid,
            radius=10,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=folium.Tooltip(
                tooltip_html,
                style=("background-color: #111a2e; color: #e2e8f0; "
                       "border: 1px solid #1e293b; border-radius: 8px; "
                       "padding: 8px; font-family: Inter, sans-serif;"),
            ),
            popup=f"<b>{ed_name}</b><br>{legend_name}: {score:.0f}",
        ).add_to(m)

    colormap.add_to(m)


def _add_circle_markers(m, scores_df, selected_metric):
    """Fallback: add circle markers for each county."""
    for _, row in scores_df.iterrows():
        county = row["county"]
        score = row.get(selected_metric, 50)
        lat, lon = COUNTY_CENTROIDS.get(county, (53.5, -7.5))

        if score >= 67:
            color = "#ef4444"
        elif score >= 34:
            color = "#f59e0b"
        else:
            color = "#10b981"

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(8, score / 5),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"<b>{county}</b><br>{selected_metric}: {score:.0f}",
        ).add_to(m)


def _add_ed_circle_markers(m, scores_df, selected_metric, county=None):
    """Fallback: add circle markers for each ED."""
    # We need to derive lat/lon from county centroids + some offset
    if county and county in COUNTY_CENTROIDS:
        base_lat, base_lon = COUNTY_CENTROIDS[county]
    else:
        base_lat, base_lon = 53.5, -7.5

    import numpy as np
    np.random.seed(hash(county or "default") % 2**32)

    for i, (_, row) in enumerate(scores_df.iterrows()):
        ed_name = row.get("ed_name", row.get("ed_id", f"ED-{i}"))
        score = row.get(selected_metric, 50)

        # Spread markers
        angle = 2 * np.pi * i / max(len(scores_df), 1)
        radius = 0.05 + (i % 3) * 0.03
        lat = base_lat + radius * np.sin(angle) + np.random.uniform(-0.01, 0.01)
        lon = base_lon + radius * np.cos(angle) + np.random.uniform(-0.01, 0.01)

        if score >= 67:
            color = "#ef4444"
        elif score >= 34:
            color = "#f59e0b"
        else:
            color = "#10b981"

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(6, score / 6),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>{ed_name}</b><br>{selected_metric}: {score:.0f}",
            tooltip=f"{ed_name}: {score:.0f}",
        ).add_to(m)
