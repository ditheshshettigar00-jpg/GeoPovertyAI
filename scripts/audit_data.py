"""
audit_data.py
-------------
Data-quality gate for the training table. Checks value ranges, missingness,
constant (placeholder) columns, duplicate districts, suspiciously correlated
feature groups, thin states and imagery coverage, then writes a Markdown report.

    python scripts/audit_data.py                       # audits data/processed/final_training_table.csv
    python scripts/audit_data.py --data other.csv --strict   # exit code 1 if any FAIL

Run it after every rebuild of the table and before training.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from config import PROCESSED_DIR

# column -> (min, max, unit). None = unbounded on that side.
RANGES = {
    "builtup_pct": (0, 100, "% of land"),
    "ndvi_mean": (-1, 1, "index"),
    "nightlight_mean": (0, None, "nW/cm2/sr"),
    "population_density": (0, 50000, "people per km2"),
    "road_density_km_per_sqkm": (0, 10, "km/km2"),
    "school_distance_km": (0, 100, "km"),
    "hospital_distance_km": (0, 100, "km"),
    "deprivation_score": (0, 1, "score"),
    "latitude": (6, 37.5, "deg"),
    "longitude": (68, 98, "deg"),
}
ACC = ["road_density_km_per_sqkm", "school_distance_km", "hospital_distance_km"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROCESSED_DIR, "final_training_table.csv"))
    ap.add_argument("--out", default=os.path.join(PROCESSED_DIR, "data_audit.md"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    results = []   # (status, check, detail)

    def add(status, check, detail):
        results.append((status, check, detail))

    add("INFO", "Table", f"{len(df)} districts, {df['state_name'].nunique()} states, {df.shape[1]} columns")

    # 1. ranges
    for col, (lo, hi, unit) in RANGES.items():
        if col not in df.columns:
            add("WARN", f"range: {col}", "column not present")
            continue
        v = df[col].dropna()
        bad = ((v < lo) if lo is not None else pd.Series(False, index=v.index)) | \
              ((v > hi) if hi is not None else pd.Series(False, index=v.index))
        if bad.sum():
            add("FAIL", f"range: {col}",
                f"{int(bad.sum())} of {len(v)} values outside [{lo}, {hi}] {unit}; observed min {v.min():.3f}, max {v.max():.3f}")
        else:
            add("PASS", f"range: {col}", f"all values within [{lo}, {hi}] {unit}")

    # 2. unit plausibility
    if "population_density" in df.columns:
        med = df["population_density"].median()
        if med < 50:
            add("FAIL", "units: population_density",
                f"median {med:.2f}; district medians in India are in the hundreds of people per km2, so these look like per-cell counts")
        else:
            add("PASS", "units: population_density", f"median {med:.0f} people per km2")

    # 3. missingness
    na = df.isna().mean().sort_values(ascending=False)
    for col, frac in na[na > 0].items():
        add("FAIL" if frac > 0.20 else "WARN", f"missing: {col}", f"{frac:.1%} missing ({int(frac * len(df))} rows)")
    if not (na > 0).any():
        add("PASS", "missing values", "no missing values (note: if imputation happened earlier, say where)")

    # 4. constant / placeholder columns
    num = df.select_dtypes("number")
    for col in num.columns:
        if col in ("area_id", "data_year"):
            continue
        nun = num[col].nunique(dropna=True)
        top_share = num[col].value_counts(normalize=True, dropna=True).iloc[0] if nun else 1.0
        if nun <= 1:
            add("FAIL", f"constant: {col}", "a single repeated value")
        elif top_share > 0.30 and col in ACC + ["s2_image_count"] and col != "s2_image_count":
            add("WARN", f"repeated value: {col}", f"{top_share:.0%} of rows share one value (placeholder?)")

    # 5. duplicates
    key = [c for c in ("state_name", "area_name") if c in df.columns]
    dup = df.duplicated(key).sum() if key else 0
    add("FAIL" if dup else "PASS", "duplicate districts", f"{int(dup)} duplicate (state, district) rows")

    # 6. accessibility columns that move together
    have = [c for c in ACC if c in df.columns]
    if len(have) >= 2:
        corr = df[have].corr()
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(have) for b in have[i + 1:]]
        hi_pairs = [(a, b, r) for a, b, r in pairs if abs(r) > 0.90]
        if hi_pairs:
            add("WARN", "accessibility correlation",
                "; ".join(f"{a} vs {b}: r={r:.2f}" for a, b, r in hi_pairs) +
                ". Independent OSM measurements normally correlate far less; check how these were produced.")
        else:
            add("PASS", "accessibility correlation", "accessibility features are not near-duplicates of each other")

    # 7. thin states
    sizes = df["state_name"].value_counts()
    thin = sizes[sizes < 3]
    if len(thin):
        add("WARN", "thin states", ", ".join(f"{s} ({n})" for s, n in thin.items()) +
            " have fewer than 3 districts; leave-one-state-out scores for them are very noisy")

    # 8. imagery coverage
    if "s2_image_count" in df.columns:
        z = int((df["s2_image_count"] == 0).sum())
        add("WARN" if z else "PASS", "Sentinel-2 coverage", f"{z} districts have zero usable images (NDVI is missing or filled for them)")

    # 9. visual features
    vg = [c for c in df.columns if c.startswith("vgg_feat_")]
    add("INFO", "visual features", f"{len(vg)} vgg_feat_* columns" if vg else "none present")

    order = {"FAIL": 0, "WARN": 1, "PASS": 2, "INFO": 3}
    results.sort(key=lambda r: order[r[0]])
    lines = [f"# Data audit: {os.path.basename(args.data)}", "",
             f"FAIL: {sum(r[0]=='FAIL' for r in results)} | WARN: {sum(r[0]=='WARN' for r in results)} | "
             f"PASS: {sum(r[0]=='PASS' for r in results)}", "", "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {s} | {c} | {d} |" for s, c, d in results]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for s, c, d in results:
        print(f"[{s}] {c}: {d}")
    print(f"\nReport written to {args.out}")
    if args.strict and any(r[0] == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
