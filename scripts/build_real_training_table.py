"""
build_real_training_table.py
----------------------------
Merges real GEE satellite features (real_geospatial_features.csv)
with real NFHS-5 survey ground truth (real_ground_truth.csv)
to create the final real training table:
    data/processed/final_training_table.csv

Matching Algorithm:
- Pass 1: Strict exact matching on cleaned district and state name.
- Pass 2: In-state fuzzy string matching using difflib with threshold >= 0.85.
  CROSS-STATE MATCHING IS STRICTLY FORBIDDEN to prevent geographic mismatching.

Outputs:
- data/processed/final_training_table.csv (matched districts with true NaN, no global imputation)
- data/processed/unmatched_districts.csv (audit list of unmatched districts on both sides)
"""

import os
import re
import difflib
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, PROCESSED_DIR, GROUND_TRUTH_DIR, FUZZY_MATCH_CUTOFF

GEO_PATH = os.path.join(PROCESSED_DIR, "real_geospatial_features.csv")
GT_PATH  = os.path.join(GROUND_TRUTH_DIR, "real_ground_truth.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "final_training_table.csv")
UNMATCHED_PATH = os.path.join(PROCESSED_DIR, "unmatched_districts.csv")

# Known official state name aliases between administrative boundary vintages (GAUL 2015 vs NFHS-5 2021)
STATE_ALIASES = {
    "orissa": "odisha",
    "pondicherry": "puducherry",
    "uttaranchal": "uttarakhand",
}

def clean_name(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def canonical_state(s):
    cleaned = clean_name(s)
    return STATE_ALIASES.get(cleaned, cleaned)

def main():
    if not os.path.exists(GEO_PATH):
        raise FileNotFoundError(f"Missing required geospatial features file: {GEO_PATH}")
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Missing required ground truth file: {GT_PATH}")

    geo_df = pd.read_csv(GEO_PATH)
    gt_df = pd.read_csv(GT_PATH)

    print(f"Loaded {len(geo_df)} satellite feature records from {GEO_PATH}.")
    print(f"Loaded {len(gt_df)} NFHS-5 ground truth records from {GT_PATH}.")

    # Deduplicate geo records if any state ran multiple times
    geo_df = geo_df.drop_duplicates(subset=["area_name", "state_name"]).copy()

    geo_df["clean_dist"] = geo_df["area_name"].apply(clean_name)
    geo_df["clean_state"] = geo_df["state_name"].apply(canonical_state)

    gt_df["clean_dist"] = gt_df["area_name"].apply(clean_name)
    gt_df["clean_state"] = gt_df["state_name"].apply(canonical_state)

    # ---------------------------------------------------------
    # Pass 1: Strict exact match on cleaned district and state
    # ---------------------------------------------------------
    pass1 = pd.merge(
        geo_df,
        gt_df[["clean_dist", "clean_state", "deprivation_score"]],
        on=["clean_dist", "clean_state"],
        how="inner"
    )
    print(f"Pass 1 (Exact in-state match): {len(pass1)} districts")

    # ---------------------------------------------------------
    # Pass 2: In-state fuzzy matching (cutoff >= FUZZY_MATCH_CUTOFF)
    # ---------------------------------------------------------
    matched_geo_keys = set(zip(pass1["clean_dist"], pass1["clean_state"]))
    rem_geo = geo_df[~geo_df.set_index(["clean_dist", "clean_state"]).index.isin(matched_geo_keys)].copy()

    fuzzy_rows = []
    matched_fuzzy_keys = set()
    for _, grow in rem_geo.iterrows():
        st_clean = grow["clean_state"]
        # Candidates strictly within the SAME state
        gt_state_cands = gt_df[gt_df["clean_state"] == st_clean]
        if len(gt_state_cands) > 0:
            cands = gt_state_cands["clean_dist"].tolist()
            matches = difflib.get_close_matches(grow["clean_dist"], cands, n=1, cutoff=FUZZY_MATCH_CUTOFF)
            if matches:
                matched_cand = matches[0]
                dep_score = gt_state_cands[gt_state_cands["clean_dist"] == matched_cand]["deprivation_score"].values[0]
                grow_dict = grow.to_dict()
                grow_dict["deprivation_score"] = dep_score
                fuzzy_rows.append(grow_dict)
                matched_fuzzy_keys.add((grow["clean_dist"], grow["clean_state"]))

    if fuzzy_rows:
        pass2 = pd.DataFrame(fuzzy_rows)
        print(f"Pass 2 (In-state fuzzy match >= {FUZZY_MATCH_CUTOFF}): {len(pass2)} districts")
        merged = pd.concat([pass1, pass2], ignore_index=True)
    else:
        merged = pass1.copy()

    # Clean up identifiers and structure
    merged = merged.drop(columns=["clean_dist", "clean_state"], errors="ignore")
    merged = merged.drop_duplicates(subset=["area_name", "state_name"]).reset_index(drop=True)
    merged["area_id"] = range(1, len(merged) + 1)
    merged["data_mode"] = "real"
    merged["data_source"] = "NFHS-5 + GEE"
    merged["data_year"] = 2021

    # ---------------------------------------------------------
    # Audit Unmatched Districts on Both Sides
    # ---------------------------------------------------------
    all_matched_geo = set(zip(merged["area_name"].apply(clean_name), merged["state_name"].apply(canonical_state)))
    
    unmatched_geo = geo_df[~geo_df.set_index(["clean_dist", "clean_state"]).index.isin(all_matched_geo)][["area_name", "state_name"]].copy()
    unmatched_geo["source"] = "GEE_satellite_unmatched"

    unmatched_gt = gt_df[~gt_df.set_index(["clean_dist", "clean_state"]).index.isin(all_matched_geo)][["area_name", "state_name"]].copy()
    unmatched_gt["source"] = "NFHS5_ground_truth_unmatched"

    unmatched_all = pd.concat([unmatched_geo, unmatched_gt], ignore_index=True)
    unmatched_all.to_csv(UNMATCHED_PATH, index=False)
    print(f"Logged {len(unmatched_geo)} unmatched satellite districts and {len(unmatched_gt)} unmatched survey districts to {UNMATCHED_PATH}")

    # Physical bounds validation (leave NaNs intact, do NOT impute globally)
    numeric_features = ["nightlight_mean", "ndvi_mean", "builtup_pct", "population_density", "deprivation_score"]
    for col in numeric_features:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Verify builtup_pct is within 0..100
    merged["builtup_pct"] = merged["builtup_pct"].clip(lower=0.0, upper=100.0)

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSUCCESS! Real training table saved to: {OUT_PATH}")
    print(f"Total matched districts in training table: {len(merged)}")
    print(f"Unique states: {merged['state_name'].nunique()}")
    print("\nDeprivation Score Summary:")
    print(merged["deprivation_score"].describe().round(4))

if __name__ == "__main__":
    main()
