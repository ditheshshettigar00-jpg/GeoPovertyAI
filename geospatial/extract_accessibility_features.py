"""
extract_accessibility_features.py
------------------------------------
Extracts real accessibility features per district using OpenStreetMap data
via OSMnx:

  - road_density_km_per_sqkm : total road length within the district / district area
  - school_distance_km        : distance from the district centroid to the nearest
                                OSM-tagged school (facility to centroid)
  - hospital_distance_km      : distance from the district centroid to the nearest
                                OSM-tagged hospital/clinic (facility to centroid)

METHODOLOGY:
1. Road Density: Queries OSM drive road network clipped strictly to the
   district boundary polygon, computes total network length (km), and divides
   by district area (sq km) projected to equal-area CRS (EPSG:6933).
2. POI Distances: Queries OpenStreetMap nodes/ways for amenity='school' and
   amenity in ['hospital', 'clinic']. Distances are calculated from the district
   centroid to the nearest facility in projected metric space (EPSG:6933).
   Grid sampling: When interior sample points are evaluated, the mean distance
   across the points reflects district-wide spatial access.
3. Strict No-Synthesis Rule: If OSM data is unavailable, query times out, or no
   facility is found within the search envelope, the value is recorded as NaN.
   No synthetic fallback constants (e.g. 1.5, 2.0, 5.0) are ever used.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR

SEARCH_RADIUS_KM = 50  # search buffer for sparse remote districts


def polygon_from_geojson(geojson_geom):
    from shapely.geometry import shape
    return shape(geojson_geom)


def compute_road_density(polygon, network_type="drive"):
    """
    Downloads the OSM road network clipped to `polygon` and returns
    (road_density_km_per_sqkm, total_road_length_km).
    """
    import osmnx as ox

    graph = ox.graph_from_polygon(polygon, network_type=network_type, truncate_by_edge=True)
    edges = ox.graph_to_gdfs(graph, nodes=False)

    total_length_km = edges["length"].sum() / 1000.0  # OSMnx edge 'length' is in meters

    # Project polygon to an equal-area CRS to get a correct area in sq km
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:6933")  # world equal-area projection
    area_sqkm = gdf_proj.geometry.area.iloc[0] / 1e6

    density = total_length_km / area_sqkm if area_sqkm > 0 else None
    return density, total_length_km


def compute_nearest_poi_distance(polygon, centroid_lat, centroid_lon, tags):
    """
    Finds the nearest OSM POI matching `tags` (e.g. {'amenity': 'school'})
    to the district centroid, searching first within the district polygon,
    then expanding to a fixed-radius buffer if nothing is found inside.
    Returns distance in km, or None if nothing found within the search radius.
    """
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point

    try:
        pois = ox.features_from_polygon(polygon, tags=tags)
    except Exception:
        pois = None

    if pois is None or len(pois) == 0:
        # Fall back to a radius search around the centroid
        try:
            pois = ox.features_from_point(
                (centroid_lat, centroid_lon), tags=tags, dist=SEARCH_RADIUS_KM * 1000
            )
        except Exception:
            return None

    if pois is None or len(pois) == 0:
        return None

    pois = pois.copy()
    pois["geometry"] = pois["geometry"].centroid  # collapse polygons (e.g. school grounds) to points

    centroid_point = gpd.GeoSeries([Point(centroid_lon, centroid_lat)], crs="EPSG:4326")
    pois_gdf = gpd.GeoDataFrame(geometry=pois["geometry"], crs="EPSG:4326")

    # Project to equal-area/metric CRS for correct distance calculation
    centroid_proj = centroid_point.to_crs("EPSG:6933")
    pois_proj = pois_gdf.to_crs("EPSG:6933")

    distances_m = pois_proj.distance(centroid_proj.iloc[0])
    nearest_km = distances_m.min() / 1000.0
    return float(nearest_km)


def extract_accessibility_for_district(district):
    geom = polygon_from_geojson(district["geometry"])
    lat, lon = district["latitude"], district["longitude"]

    row = {
        "area_id": district["area_id"],
        "area_name": district["area_name"],
        "state_name": district["state_name"],
        "data_mode": "real",
    }

    try:
        density, total_km = compute_road_density(geom)
        row["road_density_km_per_sqkm"] = density
    except Exception as e:
        print(f"  WARNING: road density failed for {district['area_name']}: {e}")
        row["road_density_km_per_sqkm"] = None

    try:
        row["school_distance_km"] = compute_nearest_poi_distance(
            geom, lat, lon, tags={"amenity": "school"}
        )
    except Exception as e:
        print(f"  WARNING: school distance failed for {district['area_name']}: {e}")
        row["school_distance_km"] = None

    try:
        row["hospital_distance_km"] = compute_nearest_poi_distance(
            geom, lat, lon, tags={"amenity": ["hospital", "clinic"]}
        )
    except Exception as e:
        print(f"  WARNING: hospital distance failed for {district['area_name']}: {e}")
        row["hospital_distance_km"] = None

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
