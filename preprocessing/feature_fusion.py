"""
feature_fusion.py
--------------------
Merges the four independently-produced feature sources into ONE final
training table, keyed by area_id:

    1. ground_truth.csv           (area_id, deprivation_score, ...)          [Module 2 note / student-provided]
    2. geospatial_features.csv    (nightlight_mean, ndvi_mean, builtup_pct,
                                     population_density)                        [Module 2 - GEE]
    3. accessibility_features.csv (road_density_km_per_sqkm,
                                     school_distance_km, hospital_distance_km)   [Module 4 - OSMnx]
    4. vgg16_features.csv         (vgg_feat_1 ... vgg_feat_512)                  [Module 3 - VGG-16]

Handles, in order:
    - inner-join validation (reports which area_ids are missing from which source)
    - duplicate row removal (by area_id)
    - missing-value imputation (median, per numeric column)
    - outlier clipping (IQR-based, per numeric geospatial/accessibility column --
      NOT applied to VGG features, which are learned embeddings rather than
      hand-scaled physical measurements)
    - final column ordering matching the structure documented in the design phase

Produces: data/processed/final_training_table.csv

NOTE ON SCALING: features are NOT standardized/scaled in this script.
Random Forest and XGBoost (Module 5) do not require feature scaling to
work correctly, so scaling is deliberately left out here to avoid losing
interpretability of raw feature values in SHAP plots. If you later add a
linear-model baseline that benefits from scaling, apply a StandardScaler
inside that specific training script instead, fit only on the training
split (not here) to avoid data leakage into the test set.

Usage:
    python preprocessing/feature_fusion.py \\
        --ground_truth data/ground_truth/real_ground_truth.csv \\
        --geospatial data/processed/real_geospatial_features.csv \\
        --accessibility data/processed/real_accessibility_features.csv \\
        --vgg16 data/processed/vgg16_features_real.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR

# Columns that should NOT be outlier-clipped or median-imputed the same way
# as physical geospatial measurements (VGG features are learned embeddings).
GEOSPATIAL_NUMERIC_COLS = [
    "nightlight_mean", "ndvi_mean", "builtup_pct", "population_density",
    "road_density_km_per_sqkm", "school_distance_km", "hospital_distance_km",
]


def load_and_validate(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    if "area_id" not in df.columns:
        raise ValueError(f"{name} file must contain an 'area_id' column: {path}")
    return df


def report_id_mismatches(dfs: dict):
    """Prints which area_ids are present in some sources but missing from others."""
    all_ids = {name: set(df["area_id"]) for name, df in dfs.items()}
    union_ids = set().union(*all_ids.values())

    print("\n--- area_id coverage check ---")
    for area_id in sorted(union_ids):
        missing_from = [name for name, ids in all_ids.items() if area_id not in ids]
        if missing_from:
            print(f"  area_id={area_id}: MISSING from {missing_from}")
    print(f"Total distinct area_ids seen across all sources: {len(union_ids)}")
    common_ids = set.intersection(*all_ids.values())
    print(f"area_ids present in ALL sources (usable for final table): {len(common_ids)}")
    return common_ids


def remove_duplicates(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset="area_id", keep="first")
    n_after = len(df)
    if n_before != n_after:
        print(f"  {source_name}: removed {n_before - n_after} duplicate area_id row(s)")
    return df


def impute_missing(df: pd.DataFrame, columns) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {n_missing} missing value(s) in '{col}' with median={median_val:.4f}")
    return df


def clip_outliers_iqr(df: pd.DataFrame, columns, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Clips (not removes) extreme outliers to [Q1 - k*IQR, Q3 + k*IQR].
    A generous multiplier (3.0, vs. the common 1.5 for boxplots) is used
    deliberately, since we want to tame extreme data errors, not discard
    genuinely poor/remote districts that are legitimately at the tail of
    the distribution -- being at the tail is exactly the signal this
    project cares about, so aggressive clipping would work against the
    project's purpose.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped > 0:
            print(f"  Clipped {n_clipped} outlier value(s) in '{col}' to [{lower:.3f}, {upper:.3f}]")
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def fuse_features(ground_truth_path, geospatial_path, accessibility_path, vgg16_path) -> pd.DataFrame:
    gt = load_and_validate(ground_truth_path, "ground_truth")
    geo = load_and_validate(geospatial_path, "geospatial_features")
    acc = load_and_validate(accessibility_path, "accessibility_features")
    vgg = load_and_validate(vgg16_path, "vgg16_features")

    print("Removing duplicate area_id rows (keeping first occurrence) ...")
    gt = remove_duplicates(gt, "ground_truth")
    geo = remove_duplicates(geo, "geospatial_features")
    acc = remove_duplicates(acc, "accessibility_features")
    vgg = remove_duplicates(vgg, "vgg16_features")

    common_ids = report_id_mismatches({
        "ground_truth": gt, "geospatial": geo, "accessibility": acc, "vgg16": vgg
    })
    if len(common_ids) == 0:
        raise ValueError(
            "No area_id is present in ALL four sources -- nothing to fuse. "
            "Check that every module's output uses the SAME area_id scheme "
            "(the one from fetch_district_boundaries.py)."
        )

    # Keep only overlapping columns needed from each source to avoid duplicate
    # column name clashes (e.g. area_name/state_name appearing in multiple files)
    geo_cols = ["area_id"] + [c for c in GEOSPATIAL_NUMERIC_COLS if c in geo.columns]
    acc_cols = ["area_id"] + [c for c in GEOSPATIAL_NUMERIC_COLS if c in acc.columns]
    vgg_cols = ["area_id"] + [c for c in vgg.columns if c.startswith("vgg_feat_")]

    merged = gt.merge(geo[geo_cols], on="area_id", how="inner")
    merged = merged.merge(acc[acc_cols], on="area_id", how="inner")
    merged = merged.merge(vgg[vgg_cols], on="area_id", how="inner")

    print(f"\nMerged table shape before cleaning: {merged.shape}")

    geospatial_present = [c for c in GEOSPATIAL_NUMERIC_COLS if c in merged.columns]

    print("\nImputing missing values (geospatial/accessibility columns only) ...")
    merged = impute_missing(merged, geospatial_present)

    print("\nClipping extreme outliers (IQR-based, geospatial/accessibility columns only) ...")
    merged = clip_outliers_iqr(merged, geospatial_present)

    # Final column ordering: identifiers, target, geospatial/accessibility, then VGG features
    id_cols = [c for c in ["area_id", "area_name", "state_name", "data_source", "data_year"] if c in merged.columns]
    target_cols = [c for c in ["deprivation_score"] if c in merged.columns]
    vgg_cols_final = [c for c in merged.columns if c.startswith("vgg_feat_")]
    ordered_cols = id_cols + target_cols + geospatial_present + vgg_cols_final
    merged = merged[ordered_cols]

    print(f"\nFinal fused table shape: {merged.shape}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Fuse ground truth + geospatial + accessibility + VGG-16 features")
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--geospatial", type=str, required=True)
    parser.add_argument("--accessibility", type=str, required=True)
    parser.add_argument("--vgg16", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    merged = fuse_features(args.ground_truth, args.geospatial, args.accessibility, args.vgg16)

    out_path = args.out or os.path.join(PROCESSED_DIR, "final_training_table.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved final training table to: {out_path}")


if __name__ == "__main__":
    main()
