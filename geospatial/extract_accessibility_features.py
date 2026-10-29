"""
extract_accessibility_features.py
---------------------------------
Real accessibility features per district from OpenStreetMap (OSMnx):

  road_density_km_per_sqkm : length of MAJOR roads (motorway..tertiary) inside the
                             district / district area. Major roads only, because
                             downloading every residential lane for a whole Indian
                             district is not feasible through Overpass, and OSM
                             minor-road completeness varies a lot between states.
                             State this definition in the report.
  school_distance_km       : MEAN over ~100 sample points spread evenly inside the
                             district of the distance to the nearest OSM school.
  hospital_distance_km     : same for OSM hospitals and clinics.

PHASE 1 CHANGES
  * The old code measured distance from the single district centroid, which says
    little for large districts. A grid of sample points inside the polygon
    represents the whole district.
  * A failed or empty download is written as NaN. There are no fallback
    constants. (The CSV shipped in the repo contained 1.5 / 2.0 / 5.0 for every
    district; scripts/build_real_training_table.py now refuses such files.)
  * Road graphs keep disconnected components (retain_all=True); the default
    keeps only the largest component and under-counts roads.

LIMITATION: OSM tagging completeness differs across India. A district with few
tagged schools looks "far from schools" even if it is not. Treat the features as
OSM-based indicators, not ground truth, and say so in the report.

Usage (one state file):
    python geospatial/extract_accessibility_features.py --districts data/raw/districts_Karnataka.json
All states (resumable):
    python scripts/extract_all_india_accessibility.py
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR

N_SAMPLE_POINTS = 100
SEARCH_RADIUS_KM = 50   # if no POI is tagged inside the district, search this far around the centroid
MAJOR_ROAD_FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
METRIC_CRS = "EPSG:6933"  # world equal-area, metres


def polygon_from_geojson(geojson_geom):
    from shapely.geometry import shape
    return shape(geojson_geom)


def mean_nearest_distance_km(sample_xy_m, poi_xy_m):
    """Mean over sample points of the distance (km) to the nearest POI. Inputs are Nx2 arrays in metres."""
    from scipy.spatial import cKDTree
    sample_xy_m = np.asarray(sample_xy_m, dtype=float)
    poi_xy_m = np.asarray(poi_xy_m, dtype=float)
    if len(poi_xy_m) == 0 or len(sample_xy_m) == 0:
        return float("nan")
    dist_m, _ = cKDTree(poi_xy_m).query(sample_xy_m, k=1)
    return float(np.mean(dist_m) / 1000.0)


def sample_points_in_polygon(polygon, n=N_SAMPLE_POINTS):
    """Roughly n evenly spaced (lon, lat) points inside the polygon (regular grid over its bounding box)."""
    from shapely.geometry import Point
    from shapely.prepared import prep

    minx, miny, maxx, maxy = polygon.bounds
    prepared = prep(polygon)
    k = 8
    while k <= 512:
        xs = np.linspace(minx, maxx, k)
        ys = np.linspace(miny, maxy, k)
        pts = [(x, y) for x in xs for y in ys if prepared.contains(Point(x, y))]
        if len(pts) >= n:
            idx = np.linspace(0, len(pts) - 1, n).astype(int)
            return [pts[i] for i in idx]
        k *= 2
    if not pts:  # very small / thin polygon: fall back to its representative point
        rp = polygon.representative_point()
        return [(rp.x, rp.y)]
    return pts


def _to_metres(lon_lat_array):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    a = np.asarray(lon_lat_array, dtype=float)
    x, y = tr.transform(a[:, 0], a[:, 1])
    return np.column_stack([x, y])


def compute_road_density(polygon, road_filter=MAJOR_ROAD_FILTER):
    """Returns km of major road per km2 of district area, or NaN if the download fails/returns nothing."""
    import geopandas as gpd
    import osmnx as ox

    graph = ox.graph_from_polygon(polygon, custom_filter=road_filter,
                                  truncate_by_edge=True, retain_all=True, simplify=True)
    edges = ox.graph_to_gdfs(graph, nodes=False)
    total_km = edges["length"].sum() / 1000.0
    area_km2 = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(METRIC_CRS).area.iloc[0] / 1e6
    return float(total_km / area_km2) if area_km2 > 0 else float("nan")


def poi_points_lonlat(polygon, centroid_lat, centroid_lon, tags):
    """OSM POIs matching tags as (lon, lat) points; falls back to a radius search around the centroid."""
    import osmnx as ox

    try:
        pois = ox.features_from_polygon(polygon, tags=tags)
    except Exception:
        pois = None
    if pois is None or len(pois) == 0:
        try:
            pois = ox.features_from_point((centroid_lat, centroid_lon), tags=tags, dist=SEARCH_RADIUS_KM * 1000)
        except Exception:
            return np.empty((0, 2))
    if pois is None or len(pois) == 0:
        return np.empty((0, 2))
    c = pois.geometry.to_crs(METRIC_CRS).centroid.to_crs("EPSG:4326")
    return np.column_stack([c.x.values, c.y.values])


def mean_poi_distance_km(polygon, centroid_lat, centroid_lon, tags):
    poi = poi_points_lonlat(polygon, centroid_lat, centroid_lon, tags)
    if len(poi) == 0:
        return float("nan")
    samples = sample_points_in_polygon(polygon)
    return mean_nearest_distance_km(_to_metres(samples), _to_metres(poi))


def extract_accessibility_for_district(district):
    geom = polygon_from_geojson(district["geometry"])
    lat, lon = district["latitude"], district["longitude"]
    row = {"area_id": district["area_id"], "area_name": district["area_name"],
           "state_name": district["state_name"], "data_mode": "real"}

    for col, fn in [
        ("road_density_km_per_sqkm", lambda: compute_road_density(geom)),
        ("school_distance_km", lambda: mean_poi_distance_km(geom, lat, lon, {"amenity": "school"})),
        ("hospital_distance_km", lambda: mean_poi_distance_km(geom, lat, lon, {"amenity": ["hospital", "clinic"]})),
    ]:
        try:
            row[col] = fn()
        except Exception as e:
            print(f"  WARNING: {col} failed for {district['area_name']}: {e}")
            row[col] = float("nan")   # missing stays missing; never a placeholder constant
    return row


def main():
    parser = argparse.ArgumentParser(description="Extract OSM-based accessibility features per district")
    parser.add_argument("--districts", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with open(args.districts, "r", encoding="utf-8") as f:
        districts = json.load(f)

    rows = []
    for d in districts:
        print(f"Processing accessibility features for {d['area_name']} ...")
        rows.append(extract_accessibility_for_district(d))

    df = pd.DataFrame(rows)
    out_path = args.out or os.path.join(PROCESSED_DIR, "real_accessibility_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to: {out_path}")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
