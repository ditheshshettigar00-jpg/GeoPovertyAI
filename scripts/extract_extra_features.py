"""
extract_extra_features.py
-------------------------
EXPERIMENTAL. Extracts six extra district-level features from Google Earth Engine for
the 564 districts in data/processed/final_training_table.csv, so they can be tested
with training/evaluate_extra_features.py. Nothing here changes the production
model, the database or any result file.

  elevation_mean_m      SRTM 30 m elevation, district mean
  slope_mean_deg        slope derived from the same SRTM image, district mean
  rainfall_mm_year      CHIRPS daily precipitation, mean daily rate x 365, survey window
  lst_day_c             MODIS MOD11A2 daytime land surface temperature (deg C), survey window
  cropland_pct          ESA WorldCover 2020, % of district area classed as cropland
  travel_time_city_min  Oxford MAP 2015 travel time to nearest city (minutes). 2015 vintage.

Output: data/processed/extra_geospatial_features.csv  (key: area_id of the training table)

Run a 2-district smoke test first, because Earth Engine dataset IDs and band names change:
    python scripts/extract_extra_features.py --limit 2
Then the full run (resumable; re-run it to continue or to retry districts with missing values):
    python scripts/extract_extra_features.py

Windows PowerShell: set  $env:GEE_PROJECT_ID="your-project-id"  first (see geospatial/gee_config.py).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PROCESSED_DIR, GEE_START_DATE, GEE_END_DATE

TRAINING_TABLE = os.path.join(PROCESSED_DIR, "final_training_table.csv")
OUT_CSV = os.path.join(PROCESSED_DIR, "extra_geospatial_features.csv")
MAX_WORKERS = 6
SAVE_EVERY = 25

SRTM = "USGS/SRTMGL1_003"
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"
MODIS_LST = "MODIS/061/MOD11A2"
WORLDCOVER = "ESA/WorldCover/v200"
TRAVEL = "Oxford/MAP/accessibility_to_cities_2015_v1_0"

FEATURES = ["elevation_mean_m", "slope_mean_deg", "rainfall_mm_year",
            "lst_day_c", "cropland_pct", "travel_time_city_min"]


def feature_image(ee, name):
    """Return (single-band ee.Image named `name`, scale in metres) for one feature."""
    if name == "elevation_mean_m":
        return ee.Image(SRTM).select("elevation").rename(name), 90
    if name == "slope_mean_deg":
        return ee.Terrain.slope(ee.Image(SRTM).select("elevation")).rename(name), 90
    if name == "rainfall_mm_year":
        col = ee.ImageCollection(CHIRPS).filterDate(GEE_START_DATE, GEE_END_DATE).select("precipitation")
        return col.mean().multiply(365).rename(name), 5566
    if name == "lst_day_c":
        col = ee.ImageCollection(MODIS_LST).filterDate(GEE_START_DATE, GEE_END_DATE).select("LST_Day_1km")
        return col.mean().multiply(0.02).subtract(273.15).rename(name), 1000
    if name == "cropland_pct":
        wc = ee.ImageCollection(WORLDCOVER).first().select("Map")
        return wc.eq(40).multiply(100).rename(name), 100
    if name == "travel_time_city_min":
        return ee.Image(TRAVEL).select("accessibility").rename(name), 1000
    raise ValueError(f"unknown feature {name}")


def load_districts(table):
    """Attach the boundary geometry from data/raw/districts_<state>.json to each training row.
    Matching is by (GAUL state, district name): the area_id inside those JSON files restarts in
    every state, so it must NOT be used as a key."""
    cache, rows = {}, []
    for r in table.itertuples():
        state = r.gaul_state_name
        if state not in cache:
            path = os.path.join(RAW_DIR, f"districts_{state.replace(' ', '_')}.json")
            with open(path) as f:
                cache[state] = {(d["state_name"], d["area_name"]): d for d in json.load(f)}
        d = cache[state].get((state, r.area_name))
        if d is None:
            raise KeyError(f"No boundary for {state} / {r.area_name}")
        rows.append({"area_id": int(r.area_id), "area_name": r.area_name,
                     "gaul_state_name": state, "geometry": d["geometry"]})
    return rows


def extract_one(ee, district, features):
    geom = ee.Geometry(district["geometry"])
    row = {k: district[k] for k in ("area_id", "area_name", "gaul_state_name")}
    for name in features:
        try:
            img, scale = feature_image(ee, name)
            val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=scale,
                                   maxPixels=1e9, bestEffort=True, tileScale=4).getInfo().get(name)
            row[name] = val
        except Exception as ex:  # a failed band stays missing, never filled with a constant
            row[name] = None
            print(f"    {name} error for {district['area_name']}: {ex}")
    return row


def extract_with_retry(ee, district, features, attempts=3):
    for k in range(attempts):
        try:
            return extract_one(ee, district, features)
        except Exception as ex:
            if k == attempts - 1:
                print(f"  [{district['area_name']}] ERROR after {attempts} tries: {ex}")
                return None
            time.sleep(2 ** (k + 1))


def pending_districts(districts, existing, features):
    """Districts with no row yet, or a row that still has a missing value in a requested feature."""
    if existing.empty:
        return list(districts)
    ok = existing.set_index("area_id")
    done = {int(i) for i, r in ok.iterrows() if all(f in ok.columns and pd.notna(r[f]) for f in features)}
    return [d for d in districts if d["area_id"] not in done]


def main():
    ap = argparse.ArgumentParser(description="Extract extra GEE features for the 564 districts (experimental)")
    ap.add_argument("--features", nargs="+", default=FEATURES, choices=FEATURES)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pending districts (smoke test)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    from geospatial.gee_config import initialize_ee
    ee = initialize_ee()

    table = pd.read_csv(TRAINING_TABLE)
    districts = load_districts(table)
    existing = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
    todo = pending_districts(districts, existing, args.features)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(districts)} districts, {len(todo)} to process, features: {', '.join(args.features)}")

    rows = existing.to_dict("records") if not existing.empty else []
    by_id = {int(r["area_id"]): r for r in rows}
    for start in range(0, len(todo), SAVE_EVERY):
        batch = todo[start:start + SAVE_EVERY]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for d, res in zip(batch, pool.map(lambda d: extract_with_retry(ee, d, args.features), batch)):
                if res is not None:
                    by_id[d["area_id"]] = {**by_id.get(d["area_id"], {}), **res}
        pd.DataFrame(by_id.values()).sort_values("area_id").to_csv(OUT_CSV, index=False)
        print(f"  saved {len(by_id)} districts so far", flush=True)

    out = pd.read_csv(OUT_CSV)
    print("\nMissing values per feature:\n", out[args.features].isna().sum().to_string())
    print("\nSummary:\n", out[args.features].describe().loc[["count", "min", "mean", "max"]].round(2).to_string())
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
