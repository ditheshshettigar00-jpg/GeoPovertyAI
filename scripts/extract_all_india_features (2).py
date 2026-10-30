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
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geospatial.gee_config import initialize_ee, GAUL_LEVEL2, VIIRS_MONTHLY, VIIRS_BAND, SENTINEL2_SR
from geospatial.ghsl_features import builtup_pct_for, population_density_for
from config import RAW_DIR, PROCESSED_DIR, GEE_START_DATE, GEE_END_DATE

STATES = [
    "Andaman and Nicobar","Andhra Pradesh","Arunachal Pradesh","Assam",
    "Bihar","Chandigarh","Chhattisgarh","Dadra and Nagar Haveli",
    "Daman and Diu","Delhi","Goa","Gujarat","Haryana","Himachal Pradesh",
    "Jharkhand","Karnataka","Kerala","Lakshadweep","Madhya Pradesh",
    "Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Orissa",
    "Puducherry","Punjab","Rajasthan","Sikkim","Tamil Nadu","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal"
]

START_DATE = GEE_START_DATE   # matches NFHS-5 fieldwork (2019-2021); set in config.py
END_DATE   = GEE_END_DATE
OUT_CSV    = os.path.join(PROCESSED_DIR, "real_geospatial_features.csv")
PROGRESS_FILE = "data/raw/gee_extraction_progress.json"

# Speed settings. Earth Engine calls spend most of their time waiting on the network, so several
# districts can be processed at once. If you see "Too many concurrent" errors, lower MAX_WORKERS.
MAX_WORKERS = int(os.environ.get("EE_WORKERS", "2"))   # use 1 while the project is in restricted mode
# District-mean NDVI is practically identical at 30 m and about 9x cheaper than at 10 m.
# Mention this in the report ("NDVI aggregated at 30 m").
TILE_SCALE = int(os.environ.get("EE_TILE_SCALE", "4"))   # higher = less memory per request (fewer timeouts), a bit slower
NDVI_SCALE_M = int(os.environ.get("NDVI_SCALE_M", "100"))   # 100 m keeps compute (EECU) use low; set 30 for finer

def mask_s2_clouds(image, ee):
    scl = image.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return image.updateMask(good).divide(10000)

class RateLimited(Exception):
    """Earth Engine kept refusing requests (quota / concurrency). Data for the district is NOT usable."""


def _is_rate_limit(ex):
    m = str(ex).lower()
    return any(t in m for t in ("too many requests", "too many concurrent", "concurren", "429", "quota", "restricted mode"))


def ee_retry(fn, tries=6):
    """Runs fn(); on a rate-limit error waits 10, 20, 40, 60, 60 s and tries again. Other errors are raised."""
    for k in range(tries):
        try:
            return fn()
        except Exception as ex:
            if not _is_rate_limit(ex):
                raise
            if k == tries - 1:
                raise RateLimited(str(ex))
            wait = min(60, 10 * 2 ** k)
            print(f"    rate limited by Earth Engine; waiting {wait}s (try {k + 1}/{tries})", flush=True)
            time.sleep(wait)


def reduce_mean(ee, image, geometry, band, scale):
    result = image.select(band).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry,
        scale=scale, maxPixels=1e9, bestEffort=True, tileScale=TILE_SCALE
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
    row = {k: district[k] for k in ["area_id", "area_name", "state_name", "latitude", "longitude"]}
    row["data_mode"] = "real"
    name = district["area_name"]

    def guarded(label, fn, default=None):
        """Retries rate limits; a rate limit that does not clear aborts the district (RateLimited).
        Any other error is printed and the value is left missing."""
        try:
            return ee_retry(fn)
        except RateLimited:
            raise
        except Exception as ex:
            print(f"    {label} error for {name}: {ex}")
            return default

    def _viirs():
        img = ee.ImageCollection(VIIRS_MONTHLY).filterDate(START_DATE, END_DATE).select(VIIRS_BAND).mean()
        return reduce_mean(ee, img, geom, VIIRS_BAND, 500)

    def _s2():
        col = (ee.ImageCollection(SENTINEL2_SR).filterBounds(geom)
               .filterDate(START_DATE, END_DATE)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
               .map(lambda img: mask_s2_clouds(img, ee)))
        n = col.size().getInfo()
        if n > 0:
            ndvi = col.median().normalizedDifference(["B8", "B4"]).rename("NDVI")
            return reduce_mean(ee, ndvi, geom, "NDVI", NDVI_SCALE_M), n
        return None, 0          # a real "no usable image" district: NDVI stays missing, never filled

    row["nightlight_mean"] = guarded("nightlight", _viirs)
    # on any other error the image count is left missing so it is not confused with a real zero
    row["ndvi_mean"], row["s2_image_count"] = guarded("NDVI", _s2, default=(None, None))
    row["builtup_pct"] = guarded("built-up", lambda: builtup_pct_for(ee, geom))
    row["population_density"] = guarded("population", lambda: population_density_for(ee, geom))
    return row


def extract_with_retry(ee, district, attempts=3):
    """Retries ordinary transient failures. Returns None if the district could not be extracted."""
    for k in range(attempts):
        try:
            return extract_features(ee, district)
        except RateLimited as ex:
            print(f"  [{district['area_name']}] RATE LIMITED, giving up on this district: {ex}")
            return None
        except Exception as ex:
            if k == attempts - 1:
                print(f"  [{district['area_name']}] ERROR after {attempts} tries: {ex}")
                return None
            time.sleep(2 ** (k + 1))


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

        # Extract features (several districts at a time; results keep the original order)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = list(pool.map(lambda d: extract_with_retry(ee, d), districts))
        failed = [d["area_name"] for d, r in zip(districts, results) if r is None]
        if failed:
            print(f"\nSTOPPING: {len(failed)} district(s) of {state} could not be extracted "
                  f"({', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}).")
            print("Nothing for this state was saved, so it will be redone next time. If this is a rate limit / "
                  "restricted mode, wait or raise the project's tier, then set EE_WORKERS=1 and run again.")
            sys.exit(1)
        state_rows = [r for r in results if r is not None]
        for d, row in zip(districts, results):
            nl = row.get("nightlight_mean")
            print(f"  [{d['area_name']}] " + (f"nightlight={nl:.3f}" if nl else "done"), flush=True)

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
