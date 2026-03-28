from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
SUMMARY_PATH = ARTIFACT_DIR / "build_summary.json"
ZONES_PATH = ARTIFACT_DIR / "zones.geojson"
COLLISIONS_PATH = ARTIFACT_DIR / "collisions_cleaned.csv"
COLOR_STOPS = [
    (255, 255, 204),
    (254, 217, 118),
    (253, 141, 60),
    (240, 59, 32),
    (177, 0, 38),
]


def interpolate_color(value: float, max_value: float) -> list[int]:
    if max_value <= 0:
        return [240, 240, 240, 160]
    ratio = min(max(value / max_value, 0.0), 1.0)
    scaled = ratio * (len(COLOR_STOPS) - 1)
    lower_index = int(scaled)
    upper_index = min(lower_index + 1, len(COLOR_STOPS) - 1)
    blend = scaled - lower_index
    lower = COLOR_STOPS[lower_index]
    upper = COLOR_STOPS[upper_index]
    rgb = [
        int(lower[channel] + (upper[channel] - lower[channel]) * blend)
        for channel in range(3)
    ]
    return [*rgb, int(150 + 80 * ratio)]


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_zones() -> gpd.GeoDataFrame:
    with ZONES_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")


@st.cache_data(show_spinner=False)
def load_collisions() -> pd.DataFrame:
    collisions = pd.read_csv(COLLISIONS_PATH, parse_dates=["ACCIDENTDATE"])
    collisions["year"] = collisions["year"].astype(int)
    return collisions


def prepare_map_data(
    zones: gpd.GeoDataFrame,
    collisions: pd.DataFrame,
    year_range: tuple[int, int],
    metric: str,
) -> tuple[dict, pd.DataFrame]:
    filtered = collisions[
        (collisions["year"] >= year_range[0]) & (collisions["year"] <= year_range[1])
    ].copy()
    year_span = max(year_range[1] - year_range[0] + 1, 1)

    grouped = filtered.groupby("zone_id").size().rename("collision_count").reset_index()
    display = zones[["zone_id", "zone_area_km2", "geometry"]].copy()
    display = display.merge(grouped, on="zone_id", how="left")
    display["collision_count"] = display["collision_count"].fillna(0).astype(int)
    display["collisions_per_km2"] = display["collision_count"] / display["zone_area_km2"]
    display["avg_collisions_per_year"] = display["collision_count"] / year_span

    metric_column = "collision_count" if metric == "Collision count" else "collisions_per_km2"
    max_value = float(display[metric_column].max())
    display["fill_color"] = display[metric_column].apply(
        lambda value: interpolate_color(float(value), max_value)
    )
    display["zone_area_km2"] = display["zone_area_km2"].round(3)
    display["collisions_per_km2"] = display["collisions_per_km2"].round(2)
    display["avg_collisions_per_year"] = display["avg_collisions_per_year"].round(2)

    payload = json.loads(display.to_json(drop_id=True))
    return payload, display.drop(columns=["geometry"])


def build_map(zones_geojson: dict) -> pdk.Deck:
    temp_gdf = gpd.GeoDataFrame.from_features(zones_geojson["features"], crs="EPSG:4326")
    centroid = temp_gdf.to_crs("EPSG:3347").unary_union.centroid
    center = gpd.GeoSeries([centroid], crs="EPSG:3347").to_crs("EPSG:4326").iloc[0]

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=zones_geojson,
        pickable=True,
        stroked=True,
        filled=True,
        auto_highlight=True,
        get_fill_color="properties.fill_color",
        get_line_color=[60, 60, 60, 180],
        line_width_min_pixels=1,
    )
    return pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=center.y,
            longitude=center.x,
            zoom=10.4,
            pitch=0,
        ),
        tooltip={
            "html": (
                "<b>{zone_id}</b><br/>"
                "Area: {zone_area_km2} km²<br/>"
                "Collisions: {collision_count}<br/>"
                "Collisions / km²: {collisions_per_km2}<br/>"
                "Avg / year: {avg_collisions_per_year}"
            )
        },
        map_style="light",
    )


def main() -> None:
    st.set_page_config(page_title="KW Collision Heatmap", layout="wide")
    st.title("Kitchener-Waterloo Collision Zone Heatmap")

    if not SUMMARY_PATH.exists() or not ZONES_PATH.exists() or not COLLISIONS_PATH.exists():
        st.error("Heatmap artifacts are missing. Run `python build_kw_heatmap.py` first.")
        st.stop()

    summary = load_summary()
    zones = load_zones()
    collisions = load_collisions()
    year_min = int(collisions["year"].min())
    year_max = int(collisions["year"].max())

    with st.sidebar:
        st.header("Controls")
        year_range = st.slider(
            "Year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        )
        metric = st.radio(
            "Map metric",
            options=["Collision count", "Collisions per km²"],
            index=0,
        )
        st.metric("Excluded zero-coordinate rows", summary["raw_summary"]["unmappable_rows"])
        st.caption("These records remain in the artifacts but are excluded from the map.")
        warnings = summary["validations"].get("geometry_constrained_area_band_warnings", [])
        if warnings:
            st.caption(
                "Geometry-constrained zones outside the target area band: "
                + ", ".join(warnings)
            )

    zones_geojson, table = prepare_map_data(zones, collisions, year_range, metric)
    filtered_collisions = collisions[
        (collisions["year"] >= year_range[0]) & (collisions["year"] <= year_range[1])
    ]

    stat_1, stat_2, stat_3 = st.columns(3)
    stat_1.metric("Visible collisions", f"{len(filtered_collisions):,}")
    stat_2.metric("Zone count", f"{len(zones):,}")
    stat_3.metric("Year span", f"{year_range[0]}-{year_range[1]}")

    st.pydeck_chart(build_map(zones_geojson), use_container_width=True)
    st.subheader("Zone Metrics")
    st.dataframe(
        table.sort_values(by="collision_count", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
