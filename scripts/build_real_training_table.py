"""
build_real_training_table.py
----------------------------
Merges the real GEE satellite features (real_geospatial_features.csv)
with the real NFHS-5 survey ground truth (real_ground_truth.csv)
to create the final real training table:
    data/processed/final_training_table.csv

Uses fuzzy string matching for district names to handle slight spelling
variations between GAUL boundaries and NFHS-5 factsheets.
"""

import os
import re
import difflib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO_PATH = os.path.join(BASE_DIR, "data", "processed", "real_geospatial_features.csv")
GT_PATH  = os.path.join(BASE_DIR, "data", "ground_truth", "real_ground_truth.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "final_training_table.csv")

def clean_name(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def main():
    if not os.path.exists(GEO_PATH):
        raise FileNotFoundError(f"Missing {GEO_PATH}")
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Missing {GT_PATH}")

    geo_df = pd.read_csv(GEO_PATH)
    gt_df = pd.read_csv(GT_PATH)

    print(f"Loaded {len(geo_df)} satellite feature records.")
    print(f"Loaded {len(gt_df)} NFHS-5 ground truth records.")

    # Deduplicate geo records if any state ran multiple times
    geo_df = geo_df.drop_duplicates(subset=["area_name", "state_name"]).copy()

    geo_df["clean_dist"] = geo_df["area_name"].apply(clean_name)
    geo_df["clean_state"] = geo_df["state_name"].apply(clean_name)

    gt_df["clean_dist"] = gt_df["area_name"].apply(clean_name)
    gt_df["clean_state"] = gt_df["state_name"].apply(clean_name)

    # 1. First pass: exact match on clean district and state
    merged = pd.merge(
        geo_df,
        gt_df[["clean_dist", "clean_state", "deprivation_score"]],
        on=["clean_dist", "clean_state"],
        how="inner"
    )
    print(f"Pass 1 (Exact state + district match): {len(merged)} districts")

    # 2. Second pass: exact match on district name only for remaining
    matched_geo_dists = set(merged["clean_dist"])
    rem_geo = geo_df[~geo_df["clean_dist"].isin(matched_geo_dists)].copy()

    gt_unmatched = gt_df[~gt_df["clean_dist"].isin(matched_geo_dists)].drop_duplicates("clean_dist")
    pass2 = pd.merge(
        rem_geo,
        gt_unmatched[["clean_dist", "deprivation_score"]],
        on="clean_dist",
        how="inner"
    )
    if len(pass2) > 0:
        print(f"Pass 2 (Exact district match across states): {len(pass2)} additional districts")
        merged = pd.concat([merged, pass2], ignore_index=True)

    # 3. Third pass: fuzzy matching within state for any remaining
    matched_geo_dists = set(merged["clean_dist"])
    rem_geo = geo_df[~geo_df["clean_dist"].isin(matched_geo_dists)].copy()
    
    fuzzy_rows = []
    for _, grow in rem_geo.iterrows():
        st_clean = grow["clean_state"]
        gt_state_cands = gt_df[gt_df["clean_state"] == st_clean]
        if len(gt_state_cands) > 0:
            cands = gt_state_cands["clean_dist"].tolist()
            matches = difflib.get_close_matches(grow["clean_dist"], cands, n=1, cutoff=0.75)
            if matches:
                matched_cand = matches[0]
                dep_score = gt_state_cands[gt_state_cands["clean_dist"] == matched_cand]["deprivation_score"].values[0]
                grow_dict = grow.to_dict()
                grow_dict["deprivation_score"] = dep_score
                fuzzy_rows.append(grow_dict)

    if fuzzy_rows:
        pass3 = pd.DataFrame(fuzzy_rows)
        print(f"Pass 3 (Fuzzy spelling match): {len(pass3)} additional districts")
        merged = pd.concat([merged, pass3], ignore_index=True)

    # Clean up and re-index
    merged = merged.drop(columns=["clean_dist", "clean_state"], errors="ignore")
    merged = merged.drop_duplicates(subset=["area_name", "state_name"]).reset_index(drop=True)
    merged["area_id"] = range(1, len(merged) + 1)
    merged["data_mode"] = "REAL"
    merged["data_source"] = "NFHS-5 + GEE"
    merged["data_year"] = 2021

    # Impute or clip extreme values in features if any
    numeric_features = ["nightlight_mean", "ndvi_mean", "builtup_pct", "population_density"]
    for col in numeric_features:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            merged[col] = merged[col].fillna(merged[col].median())

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSUCCESS! Real training table saved to: {OUT_PATH}")
    print(f"Total real districts in training table: {len(merged)}")
    print("\nDeprivation Score Summary:")
    print(merged["deprivation_score"].describe().round(4))

if __name__ == "__main__":
    main()
