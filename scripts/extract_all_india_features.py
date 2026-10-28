"""
extract_all_india_features.py
------------------------------
Master script: fetches district boundaries and extracts GEE satellite
features for ALL 34 Indian states. Saves progress after each state so
it can be safely interrupted and resumed.

Outputs:
  data/raw/districts_<State>.json       -- boundary + centroid per state
  data/processed/real_geospatial_features.csv  -- merged features for all districts
"""
import json, os, sys, time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geospatial.gee_config import initialize_ee, GAUL_LEVEL2, VIIRS_MONTHLY, VIIRS_BAND, GHSL_BUILTUP, GHSL_POP, SENTINEL2_SR
from config import RAW_DIR, PROCESSED_DIR

STATES = [
    "Andaman and Nicobar","Andhra Pradesh","Arunachal Pradesh","Assam",
    "Bihar","Chandigarh","Chhattisgarh","Dadra and Nagar Haveli",
    "Daman and Diu","Delhi","Goa","Gujarat","Haryana","Himachal Pradesh",
    "Jharkhand","Karnataka","Kerala","Lakshadweep","Madhya Pradesh",
    "Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Orissa",
    "Puducherry","Punjab","Rajasthan","Sikkim","Tamil Nadu","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal"
]

START_DATE = "2023-01-01"
END_DATE   = "2023-12-31"
OUT_CSV    = os.path.join(PROCESSED_DIR, "real_geospatial_features.csv")
PROGRESS_FILE = "data/raw/gee_extraction_progress.json"

def mask_s2_clouds(image, ee):
    scl = image.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return image.updateMask(good).divide(10000)

def reduce_mean(ee, image, geometry, band, scale):
    result = image.select(band).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry,
        scale=scale, maxPixels=1e9, bestEffort=True
    ).getInfo()
    return result.get(band)

def fetch_boundaries(ee, state):
    fc = (ee.FeatureCollection(GAUL_LEVEL2)
          .filter(ee.Filter.eq("ADM0_NAME","India"))
          .filter(ee.Filter.eq("ADM1_NAME", state)))
    count = fc.size().getInfo()
    if count == 0:
        print(f"  WARNING: No districts found for {state}")
        return []
    features = fc.getInfo()["features"]
    rows = []
    for i, feat in enumerate(features, 1):
        props = feat["properties"]
        geom  = feat["geometry"]
        centroid = ee.Geometry(geom).centroid(maxError=1).coordinates().getInfo()
        rows.append({"area_id": i, "area_name": props.get("ADM2_NAME", f"District_{i}"),
                     "state_name": state, "latitude": centroid[1], "longitude": centroid[0],
                     "geometry": geom})
    return rows

def extract_features(ee, district):
    geom = ee.Geometry(district["geometry"])
    row = {k: district[k] for k in ["area_id","area_name","state_name","latitude","longitude"]}
    row["data_mode"] = "real"
    # VIIRS nighttime lights
    try:
        viirs = ee.ImageCollection(VIIRS_MONTHLY).filterDate(START_DATE, END_DATE).select(VIIRS_BAND).mean()
        row["nightlight_mean"] = reduce_mean(ee, viirs, geom, VIIRS_BAND, 500)
    except Exception as ex:
        row["nightlight_mean"] = None
        print(f"    nightlight error: {ex}")
    # Sentinel-2 NDVI
    try:
        col = (ee.ImageCollection(SENTINEL2_SR).filterBounds(geom)
               .filterDate(START_DATE, END_DATE)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
               .map(lambda img: mask_s2_clouds(img, ee)))
        n = col.size().getInfo()
        if n > 0:
            ndvi = col.median().clip(geom).normalizedDifference(["B8","B4"]).rename("NDVI")
            row["ndvi_mean"] = reduce_mean(ee, ndvi, geom, "NDVI", 10)
            row["s2_image_count"] = n
        else:
            row["ndvi_mean"] = None; row["s2_image_count"] = 0
    except Exception as ex:
        row["ndvi_mean"] = None; row["s2_image_count"] = 0
        print(f"    NDVI error: {ex}")
    # GHSL built-up
    try:
        bu = ee.ImageCollection(GHSL_BUILTUP).sort("system:time_start", False).first()
        band = bu.bandNames().get(0)
        val = bu.select([band]).reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=100, maxPixels=1e9, bestEffort=True).getInfo()
        row["builtup_pct"] = list(val.values())[0] if val else None
    except Exception as ex:
        row["builtup_pct"] = None
        print(f"    builtup error: {ex}")
    # GHSL population
    try:
        pop = ee.ImageCollection(GHSL_POP).sort("system:time_start", False).first()
        band = pop.bandNames().get(0)
        val = pop.select([band]).reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=100, maxPixels=1e9, bestEffort=True).getInfo()
        row["population_density"] = list(val.values())[0] if val else None
    except Exception as ex:
        row["population_density"] = None
        print(f"    population error: {ex}")
    return row

def main():
    ee = initialize_ee()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load progress
    done_states = set()
    all_rows = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            done_states = set(json.load(f))
        print(f"Resuming -- already done: {done_states}")
    if os.path.exists(OUT_CSV):
        all_rows = pd.read_csv(OUT_CSV).to_dict("records")
        print(f"Loaded {len(all_rows)} existing rows from {OUT_CSV}")

    for state in STATES:
        if state in done_states:
            print(f"Skipping {state} (already done)")
            continue

        print(f"\n{'='*50}")
        print(f"Processing state: {state}")
        print(f"{'='*50}")

        # Fetch boundaries
        boundary_file = os.path.join(RAW_DIR, f"districts_{state.replace(' ','_')}.json")
        if os.path.exists(boundary_file):
            with open(boundary_file) as f:
                districts = json.load(f)
            print(f"  Loaded {len(districts)} districts from cache")
        else:
            districts = fetch_boundaries(ee, state)
            if not districts:
                continue
            with open(boundary_file, "w") as f:
                json.dump(districts, f)
            print(f"  Saved {len(districts)} district boundaries")

        # Extract features
        state_rows = []
        for d in districts:
            print(f"  [{d['area_name']}]", end=" ", flush=True)
            try:
                row = extract_features(ee, d)
                state_rows.append(row)
                print(f"nightlight={row.get('nightlight_mean','N/A'):.3f}" if row.get('nightlight_mean') else "done")
            except Exception as ex:
                print(f"ERROR: {ex}")
                time.sleep(2)

        all_rows.extend(state_rows)

        # Save progress after each state
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        done_states.add(state)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(list(done_states), f)
        print(f"  Saved progress: {len(all_rows)} total districts so far")

    print(f"\nDONE! Total districts extracted: {len(all_rows)}")
    print(f"Output saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()
