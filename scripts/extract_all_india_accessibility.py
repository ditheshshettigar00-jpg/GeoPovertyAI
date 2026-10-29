"""
extract_all_india_accessibility.py
----------------------------------
Runs the OSM accessibility extraction for every district boundary file in
data/raw/districts_*.json, resumable and safe to interrupt. Results are
appended to data/processed/real_accessibility_features.csv; districts already
present (by state + name) are skipped on the next run.

This is slow (OSM downloads for ~570 districts: expect hours). Test first:
    python scripts/extract_all_india_accessibility.py --states Goa --max-districts 2

Districts that fail are written with NaN, never with a placeholder value.
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RAW_DIR
from geospatial.extract_accessibility_features import extract_accessibility_for_district
from preprocessing.name_keys import clean_key

OUT_CSV = os.path.join(PROCESSED_DIR, "real_accessibility_features.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", nargs="*", default=None, help="Only these states (as in the boundary file names)")
    p.add_argument("--max-districts", type=int, default=None, help="Limit per state (for a quick test)")
    p.add_argument("--fresh", action="store_true", help="Ignore and overwrite any existing output")
    args = p.parse_args()

    done, rows = set(), []
    if os.path.exists(OUT_CSV) and not args.fresh:
        old = pd.read_csv(OUT_CSV)
        rows = old.to_dict("records")
        done = {(clean_key(r["state_name"]), clean_key(r["area_name"])) for r in rows}
        print(f"Resuming: {len(done)} districts already in {OUT_CSV}")

    files = sorted(glob.glob(os.path.join(RAW_DIR, "districts_*.json")))
    wanted = {clean_key(s) for s in args.states} if args.states else None
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            districts = json.load(fh)
        if not districts:
            continue
        if wanted and clean_key(districts[0]["state_name"]) not in wanted:
            continue
        if args.max_districts:
            districts = districts[: args.max_districts]
        for d in districts:
            key = (clean_key(d["state_name"]), clean_key(d["area_name"]))
            if key in done:
                continue
            print(f"[{d['state_name']}] {d['area_name']}")
            rows.append(extract_accessibility_for_district(d))
            done.add(key)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)   # save after every district

    print(f"Done. {len(rows)} districts in {OUT_CSV}")


if __name__ == "__main__":
    main()
