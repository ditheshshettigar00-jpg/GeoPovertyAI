"""
build_real_training_table.py
----------------------------
Builds data/processed/final_training_table.csv from REAL sources only:

    geospatial features   (Sentinel-2 NDVI, VIIRS, GHSL)   <- required
    accessibility features (OSM)                            <- required unless --allow-missing-accessibility
    visual features        (VGG-16 embeddings)              <- optional
    NFHS-5 ground truth                                     <- required

PHASE 1 CHANGES
  * District matching is one-to-one and stays inside the state. The old
    "match the district name in any state" pass is gone. It was masking real
    state-name differences (Orissa/Odisha, Telangana inside Andhra Pradesh,
    Delhi, ...), which are now handled by an explicit alias table
    (data/ground_truth/district_name_aliases.csv) and one explicit fallback
    (Andhra Pradesh -> Telangana).
  * Fuzzy matching is conservative: score >= 0.90, unique, and never across
    words such as Urban/Rural/North/South/East/West. Everything else goes to a
    review file instead of being guessed.
  * Nothing is imputed here. Missing values stay NaN and are handled inside
    cross-validation folds (Phase 2) so no information leaks between states.
  * Placeholder detection: an accessibility column with a single repeated value
    stops the build.
  * Reports: data/processed/district_match_report.csv (every decision) and
    data/processed/unmatched_districts.csv (both sides, with suggestions).
"""

import argparse
import difflib
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from config import DISTRICT_ALIASES_CSV, FUZZY_MATCH_CUTOFF, GROUND_TRUTH_DIR, PROCESSED_DIR
from preprocessing.name_keys import (
    STATE_FALLBACKS, clean_key, distinguishing_tokens, state_key,
)

GEO_REQUIRED = ["area_name", "state_name", "latitude", "longitude",
                "nightlight_mean", "ndvi_mean", "builtup_pct", "population_density"]
ACC_COLS = ["road_density_km_per_sqkm", "school_distance_km", "hospital_distance_km"]
SUGGEST_CUTOFF = 0.75


def load_aliases(path):
    if not os.path.exists(path):
        print(f"  NOTE: alias file not found at {path}; continuing without aliases.")
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[(state_key(r["gaul_state"]), clean_key(r["gaul_district"]))] = (
            state_key(r["nfhs_state"]), clean_key(r["nfhs_district"]))
    return out


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_districts(geo, gt, aliases, cutoff=FUZZY_MATCH_CUTOFF):
    """Returns (report DataFrame with one row per boundary district, set of used gt indexes)."""
    gt = gt.copy()
    gt["sk"] = gt["state_name"].map(state_key)
    gt["dk"] = gt["area_name"].map(clean_key)
    gt_lookup = {(r.sk, r.dk): i for i, r in gt.iterrows()}
    used = set()
    decisions = {}

    def take(gi_geo, gi_gt, method, score):
        decisions[gi_geo] = {"gt_index": gi_gt, "method": method, "score": score}
        used.add(gi_gt)

    geo = geo.copy()
    geo["sk"] = geo["state_name"].map(state_key)
    geo["dk"] = geo["area_name"].map(clean_key)

    # Pass 1: explicit alias table
    for i, r in geo.iterrows():
        tgt = aliases.get((r["sk"], r["dk"]))
        if tgt and tgt in gt_lookup and gt_lookup[tgt] not in used:
            take(i, gt_lookup[tgt], "alias", 1.0)

    # Pass 2: exact district within the (alias-mapped) state
    for i, r in geo.iterrows():
        if i in decisions:
            continue
        gi = gt_lookup.get((r["sk"], r["dk"]))
        if gi is not None and gi not in used:
            take(i, gi, "exact", 1.0)

    # Pass 3: the one explicit state fallback (Telangana districts listed under Andhra Pradesh)
    for i, r in geo.iterrows():
        if i in decisions:
            continue
        for fb in STATE_FALLBACKS.get(r["sk"], []):
            gi = gt_lookup.get((fb, r["dk"]))
            if gi is not None and gi not in used:
                take(i, gi, "state_fallback", 1.0)
                break

    # Pass 4: conservative fuzzy matching inside the state
    suggestions = {}
    for i, r in geo.iterrows():
        if i in decisions:
            continue
        states = [r["sk"]] + STATE_FALLBACKS.get(r["sk"], [])
        cands = gt[gt["sk"].isin(states) & ~gt.index.isin(used)]
        scored = sorted(((_ratio(r["dk"], c.dk), idx) for idx, c in cands.iterrows()), reverse=True)
        if not scored:
            continue
        best, bidx = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        suggestions[i] = (gt.loc[bidx, "area_name"], gt.loc[bidx, "state_name"], round(best, 3))
        same_words = distinguishing_tokens(r["area_name"]) == distinguishing_tokens(gt.loc[bidx, "area_name"])
        if best >= cutoff and (best - second) >= 0.03 and same_words:
            take(i, bidx, "fuzzy", best)

    rows = []
    for i, r in geo.iterrows():
        d = decisions.get(i)
        sug = suggestions.get(i, (None, None, None))
        rows.append({
            "geo_index": i, "gaul_state": r["state_name"], "gaul_district": r["area_name"],
            "method": d["method"] if d else "UNMATCHED",
            "score": round(d["score"], 3) if d else None,
            "nfhs_state": gt.loc[d["gt_index"], "state_name"] if d else None,
            "nfhs_district": gt.loc[d["gt_index"], "area_name"] if d else None,
            "gt_index": d["gt_index"] if d else None,
            "suggestion_district": None if d else sug[0],
            "suggestion_state": None if d else sug[1],
            "suggestion_score": None if d else sug[2],
        })
    return pd.DataFrame(rows), used, gt


def assert_no_placeholders(df, cols, name):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"{name} is missing required column '{c}'.")
        vals = df[c].dropna()
        if len(vals) > 3 and vals.nunique() <= 1:
            raise ValueError(
                f"{name} column '{c}' has a single repeated value ({vals.iloc[0]}) across "
                f"{len(vals)} districts. That is a placeholder, not a measurement. "
                "Re-run the extraction or exclude this feature.")


def main():
    p = argparse.ArgumentParser(description="Build the real training table (real data only, no imputation)")
    p.add_argument("--geo", default=os.path.join(PROCESSED_DIR, "real_geospatial_features.csv"))
    p.add_argument("--accessibility", default=os.path.join(PROCESSED_DIR, "real_accessibility_features.csv"))
    p.add_argument("--visual", default=os.path.join(PROCESSED_DIR, "vgg16_features.csv"))
    p.add_argument("--ground-truth", default=os.path.join(GROUND_TRUTH_DIR, "real_ground_truth.csv"))
    p.add_argument("--aliases", default=DISTRICT_ALIASES_CSV)
    p.add_argument("--out", default=os.path.join(PROCESSED_DIR, "final_training_table.csv"))
    p.add_argument("--report-dir", default=PROCESSED_DIR)
    p.add_argument("--fuzzy-cutoff", type=float, default=FUZZY_MATCH_CUTOFF)
    p.add_argument("--allow-missing-accessibility", action="store_true",
                   help="Build without accessibility columns (they are then absent, never filled)")
    args = p.parse_args()

    for path, label in [(args.geo, "geospatial features"), (args.ground_truth, "ground truth")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")

    geo = pd.read_csv(args.geo)
    miss = [c for c in GEO_REQUIRED if c not in geo.columns]
    if miss:
        raise ValueError(f"Geospatial file is missing columns: {miss}")
    n0 = len(geo)
    geo = geo.drop_duplicates(subset=["state_name", "area_name"]).reset_index(drop=True)
    print(f"Geospatial rows: {n0} ({n0 - len(geo)} duplicates dropped) -> {len(geo)}")
    gt = pd.read_csv(args.ground_truth)
    print(f"Ground-truth rows: {len(gt)}")

    report, used, gt_keyed = match_districts(geo, gt, load_aliases(args.aliases), args.fuzzy_cutoff)
    print("\nMatching summary:")
    print(report["method"].value_counts().to_string())

    os.makedirs(args.report_dir, exist_ok=True)
    report.drop(columns=["gt_index"]).to_csv(os.path.join(args.report_dir, "district_match_report.csv"), index=False)

    un_geo = report[report["method"] == "UNMATCHED"]
    un_gt = gt_keyed[~gt_keyed.index.isin(used)]
    unmatched = pd.concat([
        pd.DataFrame({"side": "boundary", "state": un_geo["gaul_state"], "district": un_geo["gaul_district"],
                      "best_suggestion": un_geo["suggestion_district"], "suggestion_score": un_geo["suggestion_score"]}),
        pd.DataFrame({"side": "nfhs", "state": un_gt["state_name"], "district": un_gt["area_name"],
                      "best_suggestion": None, "suggestion_score": None}),
    ], ignore_index=True)
    unmatched.to_csv(os.path.join(args.report_dir, "unmatched_districts.csv"), index=False)
    print(f"Unmatched boundary districts: {len(un_geo)}; unmatched NFHS districts: {len(un_gt)} "
          f"(see unmatched_districts.csv; add confirmed pairs to {os.path.basename(args.aliases)})")

    m = report[report["method"] != "UNMATCHED"].copy()
    table = geo.loc[m["geo_index"].values].copy().reset_index(drop=True)
    table["gaul_state_name"] = table["state_name"]
    table["state_name"] = m["nfhs_state"].values
    table["nfhs_district_name"] = m["nfhs_district"].values
    table["deprivation_score"] = gt.loc[m["gt_index"].values, "deprivation_score"].values
    table["match_method"] = m["method"].values
    table["data_mode"] = "REAL"
    table["data_source"] = "NFHS-5 + GEE"
    table["data_year"] = 2021

    table["_sk"] = table["gaul_state_name"].map(clean_key)
    table["_dk"] = table["area_name"].map(clean_key)

    # ---- accessibility (required unless explicitly allowed to be absent) ----
    if args.allow_missing_accessibility:
        print("WARNING: building WITHOUT accessibility features (--allow-missing-accessibility passed).")
    elif os.path.exists(args.accessibility):
        acc = pd.read_csv(args.accessibility)
        assert_no_placeholders(acc, ACC_COLS, "Accessibility file")
        acc["_sk"] = acc["state_name"].map(clean_key)
        acc["_dk"] = acc["area_name"].map(clean_key)
        acc = acc.drop_duplicates(["_sk", "_dk"])[["_sk", "_dk"] + ACC_COLS]
        table = table.merge(acc, on=["_sk", "_dk"], how="left")
        print(f"Accessibility joined: {table[ACC_COLS[0]].notna().sum()} of {len(table)} districts have values")
    else:
        raise FileNotFoundError(
            f"Missing accessibility file: {args.accessibility}\n"
            "Run scripts/extract_all_india_accessibility.py, or pass --allow-missing-accessibility "
            "to build a table without those features.")

    # ---- visual features (optional; must be keyed by state + district) ----
    if os.path.exists(args.visual):
        vis = pd.read_csv(args.visual)
        if {"state_key", "district_key"}.issubset(vis.columns):
            vcols = [c for c in vis.columns if c.startswith("vgg_feat_")]
            vis = vis.drop_duplicates(["state_key", "district_key"])[["state_key", "district_key"] + vcols]
            table = table.merge(vis, left_on=["_sk", "_dk"], right_on=["state_key", "district_key"], how="left")
            table = table.drop(columns=["state_key", "district_key"])
            have = table[vcols[0]].notna().sum() if vcols else 0
            print(f"Visual features joined: {have} of {len(table)} districts have a vector ({len(vcols)}-d)")
            if have < 0.9 * len(table):
                print("WARNING: visual coverage is below 90%. Do not use vgg_feat_* in the real model.")
        else:
            print("WARNING: visual file has no state_key/district_key columns (old area_id-only format, "
                  "which collides across states). Visual features NOT included. Re-run "
                  "geospatial/download_satellite_patches.py and preprocessing/vgg16_feature_extractor.py.")
    else:
        print("NOTE: no visual feature file found; building without vgg_feat_* columns.")

    table = table.drop(columns=["_sk", "_dk"]).reset_index(drop=True)
    table["area_id"] = np.arange(1, len(table) + 1)

    lead = ["area_id", "area_name", "state_name", "gaul_state_name", "nfhs_district_name",
            "latitude", "longitude", "data_mode", "match_method"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    table.to_csv(args.out, index=False)

    print(f"\nSaved {len(table)} districts, {table['state_name'].nunique()} states -> {args.out}")
    print("\nMissing values per column (left as NaN on purpose):")
    na = table.isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "  none")
    if (table["builtup_pct"] > 100).any():
        print(f"WARNING: {(table['builtup_pct'] > 100).sum()} rows have builtup_pct > 100 "
              "(old-unit extraction). Re-run the GEE extraction.")
    print("\nNext: python scripts/audit_data.py")


if __name__ == "__main__":
    main()
