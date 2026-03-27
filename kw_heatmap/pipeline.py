from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize, unary_union

DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
BOUNDARY_PLACES = [
    "Kitchener, Ontario, Canada",
    "Waterloo, Ontario, Canada",
]
OUTPUT_CRS = "EPSG:4326"
AREA_CRS = "EPSG:3347"
EXCLUDED_HIGHWAY_TAGS = {
    "service",
    "parking_aisle",
    "driveway",
    "private",
    "emergency_access",
    "rest_area",
    "services",
}
MIN_SHARED_EDGE_M = 1.0
MIN_BLOCK_AREA_M2 = 250.0
NEAREST_ZONE_MAX_DISTANCE_M = 150.0


@dataclass(frozen=True)
class BuildPaths:
    raw_csv: Path
    output_dir: Path
    cache_dir: Path
    cleaned_csv: Path
    unmappable_csv: Path
    zones_geojson: Path
    metrics_csv: Path
    summary_json: Path
    boundary_geojson: Path
    streets_geojson: Path

    @classmethod
    def create(cls, raw_csv: Path, output_dir: Path, cache_dir: Path) -> "BuildPaths":
        return cls(
            raw_csv=raw_csv,
            output_dir=output_dir,
            cache_dir=cache_dir,
            cleaned_csv=output_dir / "collisions_cleaned.csv",
            unmappable_csv=output_dir / "collisions_unmappable.csv",
            zones_geojson=output_dir / "zones.geojson",
            metrics_csv=output_dir / "zone_metrics.csv",
            summary_json=output_dir / "build_summary.json",
            boundary_geojson=cache_dir / "kw_boundary.geojson",
            streets_geojson=cache_dir / "kw_streets.geojson",
        )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    payload = json.loads(gdf.to_crs(OUTPUT_CRS).to_json(drop_id=True))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_geojson(path: Path, crs: str = OUTPUT_CRS) -> gpd.GeoDataFrame:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    features = payload.get("features", [])
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    return gpd.GeoDataFrame.from_features(features, crs=crs)


def normalize_highway_tags(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        tags = value
    else:
        tags = [value]
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def is_public_street(value: object) -> bool:
    tags = normalize_highway_tags(value)
    if not tags:
        return True
    return tags.isdisjoint(EXCLUDED_HIGHWAY_TAGS)


def iter_lines(geometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            if not part.is_empty:
                yield part


def clean_collision_data(raw_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(raw_csv)
    df["source_row"] = df.index
    df["ACCIDENTDATE"] = pd.to_datetime(df["ACCIDENTDATE"], format=DATE_FORMAT)
    df["year"] = df["ACCIDENTDATE"].dt.year
    df["has_valid_coords"] = (
        df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
        & (df["LATITUDE"] != 0)
        & (df["LONGITUDE"] != 0)
    )

    valid = df[df["has_valid_coords"]].copy()
    invalid = df[~df["has_valid_coords"]].copy()
    summary = {
        "raw_rows": int(len(df)),
        "mappable_rows": int(len(valid)),
        "unmappable_rows": int(len(invalid)),
        "date_min": valid["ACCIDENTDATE"].min().isoformat(),
        "date_max": valid["ACCIDENTDATE"].max().isoformat(),
    }
    return valid, invalid, summary


def fetch_kw_boundary(cache_path: Path, refresh_cache: bool = False) -> gpd.GeoDataFrame:
    if cache_path.exists() and not refresh_cache:
        return read_geojson(cache_path)

    geocoded_frames = []
    for place in BOUNDARY_PLACES:
        place_gdf = ox.geocode_to_gdf(place)
        geocoded_frames.append(place_gdf[["display_name", "geometry"]].copy())

    boundary = gpd.GeoDataFrame(
        pd.concat(geocoded_frames, ignore_index=True),
        geometry="geometry",
        crs=geocoded_frames[0].crs,
    )
    dissolved = gpd.GeoDataFrame(
        {"name": ["Kitchener-Waterloo"], "geometry": [unary_union(boundary.geometry)]},
        crs=boundary.crs,
    )
    write_geojson(dissolved, cache_path)
    return dissolved


def fetch_kw_streets(
    boundary: gpd.GeoDataFrame,
    cache_path: Path,
    refresh_cache: bool = False,
) -> gpd.GeoDataFrame:
    if cache_path.exists() and not refresh_cache:
        return read_geojson(cache_path)

    ox.settings.use_cache = True
    ox.settings.log_console = False
    graph = ox.graph_from_polygon(
        boundary.to_crs(OUTPUT_CRS).geometry.iloc[0],
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index(drop=True)
    edges = edges[edges["highway"].apply(is_public_street)].copy()
    edges = gpd.clip(edges, boundary.to_crs(edges.crs))
    keep_cols = [column for column in ["name", "highway", "geometry"] if column in edges.columns]
    streets = edges[keep_cols].explode(index_parts=False, ignore_index=True)
    streets = streets[streets.geometry.notna() & ~streets.geometry.is_empty].copy()
    write_geojson(streets, cache_path)
    return streets


def build_street_blocks(
    boundary: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    boundary_proj = boundary.to_crs(AREA_CRS)
    streets_proj = streets.to_crs(AREA_CRS)
    study_area = boundary_proj.geometry.iloc[0]

    line_geometries = []
    for geometry in streets_proj.geometry:
        clipped = geometry.intersection(study_area)
        for line in iter_lines(clipped):
            line_geometries.append(line)
    for line in iter_lines(study_area.boundary):
        line_geometries.append(line)

    polygon_geometries = list(polygonize(unary_union(line_geometries)))
    blocks = gpd.GeoDataFrame(geometry=polygon_geometries, crs=AREA_CRS)
    blocks = blocks[blocks.geometry.notna() & ~blocks.geometry.is_empty].copy()
    blocks["geometry"] = blocks.geometry.intersection(study_area)
    blocks = blocks[blocks.geometry.notna() & ~blocks.geometry.is_empty].copy()
    blocks = blocks[blocks.area > MIN_BLOCK_AREA_M2].copy()
    blocks["inside"] = blocks.representative_point().within(study_area.buffer(1e-6))
    blocks = blocks[blocks["inside"]].drop(columns=["inside"]).copy()
    blocks = blocks.explode(index_parts=False, ignore_index=True)
    blocks["block_id"] = range(len(blocks))
    blocks["area_m2"] = blocks.area
    blocks["centroid"] = blocks.centroid
    return blocks, boundary_proj


def build_block_graph(blocks: gpd.GeoDataFrame) -> nx.Graph:
    graph = nx.Graph()
    blocks = blocks.reset_index(drop=True)
    for row in blocks.itertuples():
        graph.add_node(int(row.block_id), area=float(row.area_m2))

    sindex = blocks.sindex
    for row in blocks.itertuples():
        candidate_indexes = list(sindex.query(row.geometry))
        for candidate_index in candidate_indexes:
            other = blocks.iloc[candidate_index]
            other_id = int(other["block_id"])
            if other_id <= row.block_id:
                continue
            if not row.geometry.touches(other.geometry):
                continue
            shared_length = row.geometry.boundary.intersection(other.geometry.boundary).length
            if shared_length <= MIN_SHARED_EDGE_M:
                continue
            graph.add_edge(int(row.block_id), other_id, shared_length=float(shared_length))
    return graph


def allocate_seed_counts(
    graph: nx.Graph,
    block_areas: dict[int, float],
    target_zones: int,
) -> list[tuple[list[int], int]]:
    components = [sorted(component) for component in nx.connected_components(graph)]
    target_zones = max(target_zones, len(components))
    target_zones = min(target_zones, graph.number_of_nodes())

    component_areas = [sum(block_areas[node] for node in component) for component in components]
    total_area = sum(component_areas)
    seed_counts = [1] * len(components)

    remaining = target_zones - len(components)
    if remaining > 0:
        quotas = [
            (area / total_area) * remaining if total_area else 0.0
            for area in component_areas
        ]
        extras = []
        for component, quota in zip(components, quotas):
            cap = max(len(component) - 1, 0)
            extras.append(min(cap, math.floor(quota)))
        seed_counts = [base + extra for base, extra in zip(seed_counts, extras)]
        assigned = sum(seed_counts)

        remainders = [quota - math.floor(quota) for quota in quotas]
        while assigned < target_zones:
            candidates = [
                index
                for index, component in enumerate(components)
                if seed_counts[index] < len(component)
            ]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda index: (remainders[index], component_areas[index], -index),
            )
            seed_counts[chosen] += 1
            assigned += 1
            remainders[chosen] = 0.0

    return list(zip(components, seed_counts))


def choose_seed_nodes(
    blocks: gpd.GeoDataFrame,
    component_nodes: list[int],
    seed_count: int,
) -> list[int]:
    block_lookup = blocks.set_index("block_id")
    centroids = {node: block_lookup.at[node, "centroid"] for node in component_nodes}
    areas = {node: float(block_lookup.at[node, "area_m2"]) for node in component_nodes}
    component_centroid = unary_union([centroids[node] for node in component_nodes]).centroid

    first_seed = min(
        component_nodes,
        key=lambda node: (
            centroids[node].distance(component_centroid),
            -areas[node],
            centroids[node].x,
            centroids[node].y,
            node,
        ),
    )
    seeds = [first_seed]
    remaining = [node for node in component_nodes if node != first_seed]

    while len(seeds) < seed_count and remaining:
        next_seed = max(
            remaining,
            key=lambda node: (
                min(centroids[node].distance(centroids[seed]) for seed in seeds),
                areas[node],
                -centroids[node].x,
                -centroids[node].y,
                -node,
            ),
        )
        seeds.append(next_seed)
        remaining.remove(next_seed)
    return seeds


def shared_boundary_to_zone(
    graph: nx.Graph,
    assignments: dict[int, int],
    node: int,
    zone_id: int,
) -> float:
    shared = 0.0
    for neighbor in graph.neighbors(node):
        if assignments.get(neighbor) == zone_id:
            shared += float(graph.edges[node, neighbor].get("shared_length", 0.0))
    return shared


def grow_component_zones(
    graph: nx.Graph,
    blocks: gpd.GeoDataFrame,
    component_nodes: list[int],
    seed_nodes: list[int],
    zone_ids: list[int],
) -> dict[int, int]:
    block_lookup = blocks.set_index("block_id")
    area_lookup = block_lookup["area_m2"].to_dict()
    centroid_lookup = block_lookup["centroid"].to_dict()
    component_target_area = sum(area_lookup[node] for node in component_nodes) / max(len(zone_ids), 1)
    component_set = set(component_nodes)

    assignments = {seed: zone_id for seed, zone_id in zip(seed_nodes, zone_ids)}
    zone_areas = {zone_id: float(area_lookup[seed]) for seed, zone_id in zip(seed_nodes, zone_ids)}
    zone_frontiers = {
        zone_id: {
            neighbor
            for neighbor in graph.neighbors(seed)
            if neighbor in component_set and neighbor not in assignments
        }
        for seed, zone_id in zip(seed_nodes, zone_ids)
    }
    seed_centroids = {zone_id: centroid_lookup[seed] for seed, zone_id in zip(seed_nodes, zone_ids)}
    unassigned = component_set - set(seed_nodes)

    while unassigned:
        active_zones = [zone_id for zone_id, frontier in zone_frontiers.items() if frontier]
        if not active_zones:
            fallback_node = min(unassigned)
            fallback_zone = min(
                zone_ids,
                key=lambda zone_id: (
                    centroid_lookup[fallback_node].distance(seed_centroids[zone_id]),
                    zone_areas[zone_id],
                    zone_id,
                ),
            )
            assignments[fallback_node] = fallback_zone
            zone_areas[fallback_zone] += float(area_lookup[fallback_node])
            unassigned.remove(fallback_node)
            continue

        current_zone = min(active_zones, key=lambda zone_id: (zone_areas[zone_id], zone_id))
        candidate = min(
            zone_frontiers[current_zone],
            key=lambda node: (
                abs((zone_areas[current_zone] + area_lookup[node]) - component_target_area),
                -shared_boundary_to_zone(graph, assignments, node, current_zone),
                centroid_lookup[node].distance(seed_centroids[current_zone]),
                area_lookup[node],
                node,
            ),
        )

        assignments[candidate] = current_zone
        zone_areas[current_zone] += float(area_lookup[candidate])
        unassigned.remove(candidate)
        for zone_id in zone_frontiers:
            zone_frontiers[zone_id].discard(candidate)
        for neighbor in graph.neighbors(candidate):
            if neighbor in unassigned:
                zone_frontiers[current_zone].add(neighbor)
        zone_frontiers[current_zone].discard(candidate)

    return assignments


def source_zone_stays_connected(
    graph: nx.Graph,
    assignments: dict[int, int],
    node_to_move: int,
    donor_zone: int,
) -> bool:
    donor_nodes = [node for node, zone_id in assignments.items() if zone_id == donor_zone and node != node_to_move]
    if not donor_nodes:
        return False
    if len(donor_nodes) == 1:
        return True
    return nx.is_connected(graph.subgraph(donor_nodes))


def rebalance_assignments(
    graph: nx.Graph,
    blocks: gpd.GeoDataFrame,
    assignments: dict[int, int],
    zone_ids: list[int],
    max_passes: int = 200,
) -> dict[int, int]:
    area_lookup = blocks.set_index("block_id")["area_m2"].to_dict()
    zone_areas = {
        zone_id: sum(area_lookup[node] for node, assigned_zone in assignments.items() if assigned_zone == zone_id)
        for zone_id in zone_ids
    }
    target_area = sum(zone_areas.values()) / max(len(zone_ids), 1)

    for _ in range(max_passes):
        best_move = None
        for donor_zone in zone_ids:
            donor_nodes = [node for node, zone_id in assignments.items() if zone_id == donor_zone]
            if len(donor_nodes) <= 1:
                continue

            for node in donor_nodes:
                adjacent_zones = {assignments.get(neighbor) for neighbor in graph.neighbors(node)}
                adjacent_zones.discard(donor_zone)
                for recipient_zone in sorted(zone for zone in adjacent_zones if zone is not None):
                    if not source_zone_stays_connected(graph, assignments, node, donor_zone):
                        continue
                    donor_after = zone_areas[donor_zone] - area_lookup[node]
                    recipient_after = zone_areas[recipient_zone] + area_lookup[node]
                    improvement = (
                        abs(zone_areas[donor_zone] - target_area)
                        + abs(zone_areas[recipient_zone] - target_area)
                        - abs(donor_after - target_area)
                        - abs(recipient_after - target_area)
                    )
                    if improvement <= 1e-9:
                        continue

                    candidate = (
                        improvement,
                        shared_boundary_to_zone(graph, assignments, node, recipient_zone),
                        -abs(recipient_after - target_area),
                        -area_lookup[node],
                        recipient_zone,
                        donor_zone,
                        node,
                    )
                    if best_move is None or candidate > best_move:
                        best_move = candidate

        if best_move is None:
            break

        _, _, _, _, recipient_zone, donor_zone, node = best_move
        assignments[node] = recipient_zone
        zone_areas[donor_zone] -= float(area_lookup[node])
        zone_areas[recipient_zone] += float(area_lookup[node])

    return assignments


def build_analysis_zones(
    blocks: gpd.GeoDataFrame,
    target_zones: int,
) -> tuple[gpd.GeoDataFrame, dict[int, str], nx.Graph]:
    graph = build_block_graph(blocks)
    block_areas = blocks.set_index("block_id")["area_m2"].to_dict()
    component_seed_counts = allocate_seed_counts(graph, block_areas, target_zones)

    next_zone_numeric = 0
    assignments: dict[int, int] = {}
    zone_numeric_ids: list[int] = []
    for component_nodes, seed_count in component_seed_counts:
        seeds = choose_seed_nodes(blocks, component_nodes, seed_count)
        zone_ids = list(range(next_zone_numeric, next_zone_numeric + len(seeds)))
        next_zone_numeric += len(seeds)
        zone_numeric_ids.extend(zone_ids)
        assignments.update(grow_component_zones(graph, blocks, component_nodes, seeds, zone_ids))

    assignments = rebalance_assignments(graph, blocks, assignments, zone_numeric_ids)
    blocks_with_zones = blocks.drop(columns=["centroid"]).copy()
    blocks_with_zones["zone_numeric"] = blocks_with_zones["block_id"].map(assignments)

    zones = (
        blocks_with_zones[["zone_numeric", "area_m2", "geometry"]]
        .dissolve(by="zone_numeric", aggfunc={"area_m2": "sum"})
        .reset_index()
    )
    zones["geometry"] = zones.geometry.buffer(0)
    zones["zone_area_km2"] = zones.geometry.area / 1_000_000.0
    centroids = zones.representative_point()
    zones["_centroid_x"] = centroids.x
    zones["_centroid_y"] = centroids.y
    zones = zones.sort_values(by=["_centroid_y", "_centroid_x"], ascending=[False, True]).reset_index(drop=True)
    width = max(2, len(str(len(zones))))
    zones["zone_id"] = [f"Z{index:0{width}d}" for index in range(1, len(zones) + 1)]
    numeric_to_zone_id = dict(zip(zones["zone_numeric"], zones["zone_id"]))
    final_assignments = {block_id: numeric_to_zone_id[zone_numeric] for block_id, zone_numeric in assignments.items()}
    zones = zones.drop(columns=["_centroid_x", "_centroid_y"])
    return zones, final_assignments, graph


def assign_collisions_to_zones(
    collisions: pd.DataFrame,
    zones: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    points = gpd.GeoDataFrame(
        collisions.copy(),
        geometry=gpd.points_from_xy(collisions["LONGITUDE"], collisions["LATITUDE"]),
        crs=OUTPUT_CRS,
    )
    zone_polygons = zones[["zone_id", "zone_area_km2", "geometry"]].to_crs(OUTPUT_CRS)

    joined = gpd.sjoin(points, zone_polygons, how="left", predicate="within")
    unmatched_mask = joined["zone_id"].isna()
    nearest_assigned = 0
    if unmatched_mask.any():
        nearest = gpd.sjoin_nearest(
            points.loc[unmatched_mask].to_crs(AREA_CRS),
            zone_polygons.to_crs(AREA_CRS),
            how="left",
            max_distance=NEAREST_ZONE_MAX_DISTANCE_M,
            distance_col="distance_to_zone_m",
        )
        joined.loc[unmatched_mask, "zone_id"] = nearest["zone_id"].values
        joined.loc[unmatched_mask, "zone_area_km2"] = nearest["zone_area_km2"].values
        joined.loc[unmatched_mask, "distance_to_zone_m"] = nearest["distance_to_zone_m"].values
        nearest_assigned = int(len(nearest))

    if joined["zone_id"].isna().any():
        missing = int(joined["zone_id"].isna().sum())
        raise ValueError(f"{missing} collisions could not be assigned to a zone.")

    joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    years_in_scope = max(joined["year"].nunique(), 1)

    metrics = zones[["zone_id", "zone_area_km2"]].copy()
    counts = joined.groupby("zone_id").size().rename("collision_count").reset_index()
    metrics = metrics.merge(counts, on="zone_id", how="left")
    metrics["collision_count"] = metrics["collision_count"].fillna(0).astype(int)
    metrics["collisions_per_km2"] = metrics["collision_count"] / metrics["zone_area_km2"]
    metrics["avg_collisions_per_year"] = metrics["collision_count"] / years_in_scope

    summary = {
        "assigned_collisions": int(len(joined)),
        "nearest_assigned_collisions": nearest_assigned,
        "years_in_scope": int(years_in_scope),
    }
    return joined, metrics, summary


def zone_assignments_are_contiguous(graph: nx.Graph, assignments: dict[int, str]) -> bool:
    for zone_id in sorted(set(assignments.values())):
        members = [node for node, assigned_zone in assignments.items() if assigned_zone == zone_id]
        if len(members) <= 1:
            continue
        if not nx.is_connected(graph.subgraph(members)):
            return False
    return True


def zones_outside_area_band(
    zones: gpd.GeoDataFrame,
    boundary_proj: gpd.GeoDataFrame,
) -> list[str]:
    median_area = zones["zone_area_km2"].median()
    lower_bound = 0.5 * median_area
    upper_bound = 1.5 * median_area
    study_boundary = boundary_proj.geometry.iloc[0].boundary

    violations = []
    for row in zones.itertuples():
        touches_outer_boundary = row.geometry.boundary.intersection(study_boundary).length > MIN_SHARED_EDGE_M
        if lower_bound <= row.zone_area_km2 <= upper_bound:
            continue
        if touches_outer_boundary:
            continue
        violations.append(row.zone_id)
    return violations


def zone_has_meaningful_rebalance_move(
    zone_id: str,
    graph: nx.Graph,
    blocks: gpd.GeoDataFrame,
    assignments: dict[int, str],
    target_area_m2: float,
    improvement_ratio: float = 0.01,
) -> bool:
    area_lookup = blocks.set_index("block_id")["area_m2"].to_dict()
    zone_areas = {
        assigned_zone: sum(area_lookup[node] for node, candidate_zone in assignments.items() if candidate_zone == assigned_zone)
        for assigned_zone in set(assignments.values())
    }
    threshold = target_area_m2 * improvement_ratio
    current_zone_area = zone_areas[zone_id]

    for node, assigned_zone in assignments.items():
        if assigned_zone == zone_id:
            for neighbor in graph.neighbors(node):
                recipient_zone = assignments[neighbor]
                if recipient_zone == zone_id:
                    continue
                if not source_zone_stays_connected(graph, assignments, node, zone_id):
                    continue
                donor_after = current_zone_area - area_lookup[node]
                recipient_after = zone_areas[recipient_zone] + area_lookup[node]
                improvement = (
                    abs(current_zone_area - target_area_m2)
                    + abs(zone_areas[recipient_zone] - target_area_m2)
                    - abs(donor_after - target_area_m2)
                    - abs(recipient_after - target_area_m2)
                )
                if improvement > threshold:
                    return True
        else:
            if not any(assignments[neighbor] == zone_id for neighbor in graph.neighbors(node)):
                continue
            if not source_zone_stays_connected(graph, assignments, node, assigned_zone):
                continue
            zone_after = current_zone_area + area_lookup[node]
            donor_after = zone_areas[assigned_zone] - area_lookup[node]
            improvement = (
                abs(current_zone_area - target_area_m2)
                + abs(zone_areas[assigned_zone] - target_area_m2)
                - abs(zone_after - target_area_m2)
                - abs(donor_after - target_area_m2)
            )
            if improvement > threshold:
                return True

    return False


def classify_area_band_violations(
    zones: gpd.GeoDataFrame,
    boundary_proj: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    assignments: dict[int, str],
    graph: nx.Graph,
) -> tuple[list[str], list[str]]:
    median_area = zones["zone_area_km2"].median()
    lower_bound = 0.5 * median_area
    upper_bound = 1.5 * median_area
    study_boundary = boundary_proj.geometry.iloc[0].boundary
    target_area_m2 = blocks["area_m2"].sum() / max(len(zones), 1)

    constrained = []
    unconstrained = []
    for row in zones.itertuples():
        if lower_bound <= row.zone_area_km2 <= upper_bound:
            continue
        touches_outer_boundary = row.geometry.boundary.intersection(study_boundary).length > MIN_SHARED_EDGE_M
        if touches_outer_boundary:
            constrained.append(row.zone_id)
            continue
        if not zone_has_meaningful_rebalance_move(
            zone_id=row.zone_id,
            graph=graph,
            blocks=blocks,
            assignments=assignments,
            target_area_m2=target_area_m2,
        ):
            constrained.append(row.zone_id)
            continue
        unconstrained.append(row.zone_id)
    return constrained, unconstrained


def validate_outputs(
    raw_summary: dict,
    cleaned_collisions: pd.DataFrame,
    blocks: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
    assignments: dict[int, str],
    graph: nx.Graph,
    boundary_proj: gpd.GeoDataFrame,
) -> dict:
    validations = {
        "raw_row_count_matches": raw_summary["raw_rows"] == 8928,
        "mappable_row_count_matches": raw_summary["mappable_rows"] == 8598,
        "unmappable_row_count_matches": raw_summary["unmappable_rows"] == 330,
        "date_min_matches": raw_summary["date_min"] == "2015-01-01T00:00:00",
        "date_max_matches": raw_summary["date_max"] == "2022-07-31T17:41:00",
        "zone_count_in_expected_range": 40 <= len(zones) <= 60,
        "all_mappable_collisions_assigned": len(cleaned_collisions) == raw_summary["mappable_rows"],
        "metrics_cover_all_zones": set(metrics["zone_id"]) == set(zones["zone_id"]),
        "zone_graph_contiguous": zone_assignments_are_contiguous(graph, assignments),
    }

    constrained_violations, unconstrained_violations = classify_area_band_violations(
        zones=zones,
        boundary_proj=boundary_proj,
        blocks=blocks,
        assignments=assignments,
        graph=graph,
    )
    validations["interior_zones_within_area_band_or_geometry_constrained"] = not unconstrained_violations
    failed_checks = [name for name, passed in validations.items() if not passed]
    if failed_checks:
        details = ", ".join(failed_checks)
        if unconstrained_violations:
            details += f". Unconstrained zone band violations: {', '.join(unconstrained_violations)}"
        raise ValueError(f"Validation failed for: {details}")
    if constrained_violations:
        validations["geometry_constrained_area_band_warnings"] = constrained_violations
    return validations


def run_pipeline(
    raw_csv: Path | str = Path("Traffic_Collisions_280340447332117481.csv"),
    output_dir: Path | str = Path("artifacts"),
    cache_dir: Path | str = Path("cache"),
    target_zones: int = 50,
    refresh_cache: bool = False,
) -> dict:
    raw_csv = Path(raw_csv)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    ensure_directory(output_dir)
    ensure_directory(cache_dir)

    paths = BuildPaths.create(raw_csv=raw_csv, output_dir=output_dir, cache_dir=cache_dir)
    valid_collisions, invalid_collisions, raw_summary = clean_collision_data(paths.raw_csv)
    boundary = fetch_kw_boundary(paths.boundary_geojson, refresh_cache=refresh_cache)
    streets = fetch_kw_streets(boundary, paths.streets_geojson, refresh_cache=refresh_cache)
    blocks, boundary_proj = build_street_blocks(boundary, streets)
    zones, block_assignments, graph = build_analysis_zones(blocks, target_zones=target_zones)
    cleaned_collisions, metrics, join_summary = assign_collisions_to_zones(valid_collisions, zones)
    validations = validate_outputs(
        raw_summary=raw_summary,
        cleaned_collisions=cleaned_collisions,
        blocks=blocks,
        zones=zones,
        metrics=metrics,
        assignments=block_assignments,
        graph=graph,
        boundary_proj=boundary_proj,
    )

    zones_out = zones.merge(metrics, on=["zone_id", "zone_area_km2"], how="left")
    write_geojson(zones_out, paths.zones_geojson)
    cleaned_collisions.to_csv(paths.cleaned_csv, index=False)
    invalid_collisions.to_csv(paths.unmappable_csv, index=False)
    metrics.to_csv(paths.metrics_csv, index=False)

    summary = {
        "raw_summary": raw_summary,
        "join_summary": join_summary,
        "zone_count": int(len(zones)),
        "zone_area_km2_min": float(zones["zone_area_km2"].min()),
        "zone_area_km2_median": float(zones["zone_area_km2"].median()),
        "zone_area_km2_max": float(zones["zone_area_km2"].max()),
        "output_files": {
            "collisions_cleaned_csv": str(paths.cleaned_csv),
            "collisions_unmappable_csv": str(paths.unmappable_csv),
            "zones_geojson": str(paths.zones_geojson),
            "zone_metrics_csv": str(paths.metrics_csv),
            "build_summary_json": str(paths.summary_json),
        },
        "validations": validations,
    }
    with paths.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the KW collision heatmap artifacts.")
    parser.add_argument("--raw-csv", default="Traffic_Collisions_280340447332117481.csv", help="Path to the raw collision CSV.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts.")
    parser.add_argument("--cache-dir", default="cache", help="Directory for cached GIS inputs.")
    parser.add_argument("--target-zones", default=50, type=int, help="Target number of balanced street-bounded zones.")
    parser.add_argument("--refresh-cache", action="store_true", help="Refetch municipal boundary and OSM street data.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pipeline(
        raw_csv=args.raw_csv,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        target_zones=args.target_zones,
        refresh_cache=args.refresh_cache,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
