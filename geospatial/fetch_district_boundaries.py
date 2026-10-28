"""
fetch_district_boundaries.py
------------------------------
Fetches administrative (district-level) boundaries for a chosen Indian
state from the FAO GAUL dataset in Earth Engine, and saves a table of
area_id, area_name, state_name, centroid lat/lon, and boundary GeoJSON.

This becomes your list of REAL areas to loop over in every later
extraction script (nightlight, NDVI, built-up, Sentinel-2 image), replacing
the synthetic Demo_District rows from Module 1.

Usage:
    python geospatial/fetch_district_boundaries.py --state "Karnataka"
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geospatial.gee_config import GAUL_LEVEL2, initialize_ee
from config import RAW_DIR


def fetch_district_boundaries(state_name: str):
    ee = initialize_ee()

    fc = (
        ee.FeatureCollection(GAUL_LEVEL2)
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filter(ee.Filter.eq("ADM1_NAME", state_name))
    )

    count = fc.size().getInfo()
    if count == 0:
        raise ValueError(
            f"No districts found for state_name='{state_name}'. "
            "GAUL state names can differ slightly from common usage "
            "(e.g. spacing/spelling) -- check the exact ADM1_NAME by running "
            "fc = ee.FeatureCollection(GAUL_LEVEL2).filter(ee.Filter.eq('ADM0_NAME','India')) "
            "and inspecting fc.aggregate_array('ADM1_NAME').distinct().getInfo()"
        )

    features = fc.getInfo()["features"]
    print(f"Found {count} districts in {state_name}")

    rows = []
    for i, feat in enumerate(features, start=1):
        props = feat["properties"]
        geom = feat["geometry"]

        # Compute centroid using Earth Engine (server-side) for accuracy
        ee_geom = ee.Geometry(geom)
        centroid = ee_geom.centroid(maxError=1).coordinates().getInfo()
        lon, lat = centroid[0], centroid[1]

        rows.append({
            "area_id": i,
            "area_name": props.get("ADM2_NAME", f"District_{i}"),
            "state_name": state_name,
            "latitude": lat,
            "longitude": lon,
            "geometry": geom,   # full boundary polygon, GeoJSON format
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch district boundaries for a given Indian state from GEE")
    parser.add_argument("--state", type=str, required=True,
                         help='State name as it appears in GAUL, e.g. "Karnataka"')
    parser.add_argument("--out", type=str, default=None,
                         help="Output JSON path (default: data/raw/districts_<state>.json)")
    args = parser.parse_args()

    rows = fetch_district_boundaries(args.state)

    out_path = args.out or os.path.join(RAW_DIR, f"districts_{args.state.replace(' ', '_')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)

    print(f"Saved {len(rows)} districts (with boundaries) to: {out_path}")
    print("Sample:", {k: v for k, v in rows[0].items() if k != "geometry"})


if __name__ == "__main__":
    main()
