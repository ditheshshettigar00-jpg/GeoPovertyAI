"""
refill_missing_features.py
--------------------------
Re-extracts ONLY the districts that have a missing value in data/processed/real_geospatial_features.csv
(nightlight, NDVI, built-up, population, or an unknown image count), using gentler Earth Engine settings
(coarser NDVI scale, larger tileScale, one worker by default) so that big districts do not time out or run
out of memory. Existing good values are never overwritten, and the CSV is saved after every district.

Why: a failed Earth Engine request leaves the value blank. Blanks are filled inside cross-validation by the
model pipeline, but a real measurement is better than an imputed one, and NDVI is the most important feature.

    python scripts/refill_missing_features.py
    python scripts/refill_missing_features.py --ndvi-scale 250 --tile-scale 16 --workers 1

Then rebuild: python scripts/build_real_training_table.py --allow-missing-accessibility
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)
from config import PROCESSED_DIR, RAW_DIR

FEATURES = ["nightlight_mean", "ndvi_mean", "builtup_pct", "population_density"]


def missing_mask(df):
    m = df[FEATURES].isna().any(axis=1)
    if "s2_image_count" in df.columns:
        m = m | df["s2_image_count"].isna()
    return m


def refill_rows(df, districts_by_key, extract_fn, workers=1, save=None):
    """Re-extract rows with missing values; fill only the blanks. Returns (df, list of districts still failing)."""
    idx = list(df.index[missing_mask(df)])
    jobs = []
    for i in idx:
        d = districts_by_key.get((df.at[i, "state_name"], df.at[i, "area_name"]))
        if d is None:
            print(f"  no cached boundary for {df.at[i, 'state_name']} / {df.at[i, 'area_name']}; skipped")
            continue
        jobs.append((i, d))
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (i, d), row in zip(jobs, pool.map(lambda j: extract_fn(j[1]), jobs)):
            if row is None:
                failed.append(d["area_name"])
                continue
            cols = FEATURES + (["s2_image_count"] if "s2_image_count" in df.columns else [])
            for c in cols:
                if pd.isna(df.at[i, c]) and row.get(c) is not None and not pd.isna(row.get(c)):
                    df.at[i, c] = row[c]
            if save:
                save(df)
            print(f"  refilled {d['area_name']}", flush=True)
    return df, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(PROCESSED_DIR, "real_geospatial_features.csv"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ndvi-scale", type=int, default=250)
    ap.add_argument("--tile-scale", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    before = df[FEATURES].isna().sum()
    n_rows = int(missing_mask(df).sum())
    print(f"{n_rows} district(s) have missing values:\n{before.to_string()}")
    if n_rows == 0:
        return

    import extract_all_india_features as ex
    from geospatial.gee_config import initialize_ee
    ex.NDVI_SCALE_M, ex.TILE_SCALE = args.ndvi_scale, args.tile_scale
    ee = initialize_ee()

    by_key = {}
    for state in df.loc[missing_mask(df), "state_name"].unique():
        path = os.path.join(RAW_DIR, f"districts_{state.replace(' ', '_')}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for d in json.load(f):
                    by_key[(d["state_name"], d["area_name"])] = d
        else:
            print(f"  boundary cache not found: {path}")

    df, failed = refill_rows(df, by_key, lambda d: ex.extract_with_retry(ee, d), args.workers,
                             save=lambda x: x.to_csv(args.csv, index=False))
    df.to_csv(args.csv, index=False)
    after = df[FEATURES].isna().sum()
    print("\nMissing values before -> after:")
    print(pd.DataFrame({"before": before, "after": after}).to_string())
    if failed:
        print(f"\n{len(failed)} district(s) still failing: {', '.join(failed[:10])}"
              f"{'...' if len(failed) > 10 else ''}\nThey stay blank and are imputed inside training.")


if __name__ == "__main__":
    main()
