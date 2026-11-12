"""
profile_service.py
------------------
District Indicators Profile Service.
Supplies the complete area profile data for the district page Indicators panel
via GET /api/area-profile/<area_id>.

Loads data/processed/final_training_table.csv,
data/processed/real_accessibility_features.csv, and
data/ground_truth/nfhs5_districts_raw.csv once and caches them in memory.

Rules:
- Never returns NaN or 0.0 for missing values; uses None (JSON null).
- Never includes any visual VGG-16 features.
- Flags suppressed or small-sample survey records as null with flag_text.
- Computes empirical percentiles across the 564 project districts.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_DIR, GROUND_TRUTH_DIR, PROCESSED_DIR
from preprocessing.name_keys import clean_key, state_key

_cache = None

MODEL_INPUT_SPECS = [
    {
        "key": "nightlight_mean",
        "label": "Nighttime Light Radiance",
        "unit": "nW/cm²/sr",
        "description": (
            "Mean monthly nocturnal radiance from NOAA/NASA VIIRS nighttime lights, "
            "where higher radiance reflects greater artificial outdoor lighting and commercial activity."
        ),
    },
    {
        "key": "ndvi_mean",
        "label": "Vegetation Index (NDVI)",
        "unit": "index",
        "description": (
            "Mean normalized difference vegetation index computed from cloud-free Sentinel-2 MSI surface reflectance, "
            "capturing green biomass and agricultural canopy."
        ),
    },
    {
        "key": "builtup_pct",
        "label": "Built-up Land Share",
        "unit": "%",
        "description": (
            "Percentage of district land area classified as built-up surface in the European Commission GHSL "
            "built-up grid (epoch 2020), measuring physical settlement footprint."
        ),
    },
    {
        "key": "population_density",
        "label": "Population Density",
        "unit": "people/km²",
        "description": (
            "Estimated residential population per square kilometre derived from the European Commission GHSL "
            "population grid (epoch 2020), indicating human settlement density."
        ),
    },
    {
        "key": "elevation_mean_m",
        "label": "Mean Elevation",
        "unit": "m",
        "description": (
            "District mean terrain elevation above sea level from NASA SRTM Digital Elevation Data (30 m)."
        ),
    },
    {
        "key": "slope_mean_deg",
        "label": "Mean Terrain Slope",
        "unit": "deg",
        "description": (
            "District mean terrain slope in degrees computed from NASA SRTM elevation gradients."
        ),
    },
    {
        "key": "rainfall_mm_year",
        "label": "Annual Rainfall",
        "unit": "mm/year",
        "description": (
            "Annualized mean precipitation across the district from UCSB CHIRPS daily rainfall (2019-11 to 2021-04)."
        ),
    },
    {
        "key": "lst_day_c",
        "label": "Daytime Land Surface Temp",
        "unit": "°C",
        "description": (
            "Mean daytime land surface temperature in degrees Celsius from MODIS MOD11A2 8-day composites (2019-11 to 2021-04)."
        ),
    },
    {
        "key": "cropland_pct",
        "label": "Cropland Share",
        "unit": "%",
        "description": (
            "Percentage of district land classified as cropland in ESA WorldCover v100 (epoch 2020, 10 m)."
        ),
    },
    {
        "key": "travel_time_city_min",
        "label": "Travel Time to Nearest City",
        "unit": "min",
        "description": (
            "Mean estimated surface travel time to the nearest urban center (>50k population) from Oxford/MAP 2015 friction surface (static 2015 baseline)."
        ),
    },
]

SURVEY_COMPONENT_SPECS = [
    {
        "key": "electricity",
        "num": 7,
        "higher_is_worse": False,
    },
    {
        "key": "sanitation",
        "num": 9,
        "higher_is_worse": False,
    },
    {
        "key": "clean_fuel",
        "num": 10,
        "higher_is_worse": False,
    },
    {
        "key": "stunting",
        "num": 73,
        "higher_is_worse": True,
    },
    {
        "key": "anaemia",
        "num": 84,
        "higher_is_worse": True,
    },
]


def get_indicator_group(num: int) -> str:
    """Return the thematic domain group for indicator numbers 1-104."""
    if 1 <= num <= 12:
        return "Households and population"
    if 13 <= num <= 19:
        return "Women's education and early life"
    if 20 <= num <= 31:
        return "Family planning"
    if 32 <= num <= 41:
        return "Antenatal and postnatal care"
    if 42 <= num <= 48:
        return "Delivery care"
    if 49 <= num <= 60:
        return "Child vaccination"
    if 61 <= num <= 66:
        return "Child illness"
    if 67 <= num <= 72:
        return "Child feeding"
    if 73 <= num <= 85:
        return "Nutrition and anaemia"
    if 86 <= num <= 97:
        return "Blood sugar and blood pressure"
    if 98 <= num <= 100:
        return "Cancer screening"
    if 101 <= num <= 104:
        return "Tobacco and alcohol"
    return "Other"


def clean_footnote_digits(text: str) -> str:
    """Remove footnote digits glued to words or closing parentheses while preserving m2 in kg/m2."""
    t = re.sub(r"\)(\d+)", ")", text)

    def _repl(match):
        w = match.group(0)
        if w.lower() in ("m2", "kg/m2"):
            return w
        return re.sub(r"\d+$", "", w)

    t = re.sub(r"\b[a-zA-Z]+\d+\b", _repl, t)
    t = re.sub(r"\s*\(\%\)\s*\d+$", " (%)", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _load() -> Dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache

    final_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    acc_path = os.path.join(PROCESSED_DIR, "real_accessibility_features.csv")
    nfhs_path = os.path.join(GROUND_TRUTH_DIR, "nfhs5_districts_raw.csv")

    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Missing {final_path}")

    # 1. Load final training table
    final_df = pd.read_csv(final_path)
    # Ensure no VGG visual columns are present
    vgg_cols = [c for c in final_df.columns if c.startswith("vgg_")]
    if vgg_cols:
        final_df = final_df.drop(columns=vgg_cols)

    final_df["sk"] = final_df["state_name"].map(state_key)
    final_df["dk"] = final_df["nfhs_district_name"].map(clean_key)
    final_df["area_dk"] = final_df["area_name"].map(clean_key)

    # Precompute summary statistics across all 564 districts for model inputs
    feature_stats = {}
    for spec in MODEL_INPUT_SPECS:
        k = spec["key"]
        series = final_df[k].dropna()
        feature_stats[k] = {
            "series": series.values,
            "min": round(float(series.min()), 3),
            "median": round(float(series.median()), 3),
            "max": round(float(series.max()), 3),
        }

    # 2. Load accessibility features (using real measurements, never placeholder sample constants)
    acc_lookup = {}
    exploratory_path = os.path.join(DATA_DIR, "exploratory", "original_table_with_visual_features.csv")

    # Priority 1: Real measured accessibility features from the nationwide dataset (532 districts)
    if os.path.exists(exploratory_path):
        exp_df = pd.read_csv(exploratory_path)
        exp_df["sk"] = exp_df["state_name"].map(state_key)
        exp_df["area_dk"] = exp_df["area_name"].map(clean_key)
        for _, row in exp_df.iterrows():
            road_val = row.get("road_density_km_per_sqkm")
            school_val = row.get("school_distance_km")
            hosp_val = row.get("hospital_distance_km")
            if pd.notna(road_val) or pd.notna(school_val) or pd.notna(hosp_val):
                acc_lookup[(row["sk"], row["area_dk"])] = {
                    "road_density_km_per_sqkm": (
                        round(float(road_val), 3) if pd.notna(road_val) else None
                    ),
                    "school_distance_km": (
                        round(float(school_val), 2) if pd.notna(school_val) else None
                    ),
                    "hospital_distance_km": (
                        round(float(hosp_val), 2) if pd.notna(hosp_val) else None
                    ),
                }

    # Priority 2: Fall back to real_accessibility_features.csv ONLY if it contains non-placeholder values
    if os.path.exists(acc_path):
        acc_df = pd.read_csv(acc_path)
        has_placeholders = False
        for c in ["road_density_km_per_sqkm", "school_distance_km", "hospital_distance_km"]:
            if c in acc_df.columns:
                non_nulls = acc_df[c].dropna()
                if len(non_nulls) > 3 and non_nulls.nunique() <= 1:
                    has_placeholders = True
                    break
        if not has_placeholders:
            acc_df["sk"] = acc_df["state_name"].map(state_key)
            acc_df["dk"] = acc_df["area_name"].map(clean_key)
            for _, row in acc_df.iterrows():
                k = (row["sk"], row["dk"])
                if k not in acc_lookup:
                    acc_lookup[k] = {
                        "road_density_km_per_sqkm": (
                            round(float(row["road_density_km_per_sqkm"]), 3)
                            if pd.notna(row.get("road_density_km_per_sqkm"))
                            else None
                        ),
                        "school_distance_km": (
                            round(float(row["school_distance_km"]), 2)
                            if pd.notna(row.get("school_distance_km"))
                            else None
                        ),
                        "hospital_distance_km": (
                            round(float(row["hospital_distance_km"]), 2)
                            if pd.notna(row.get("hospital_distance_km"))
                            else None
                        ),
                    }

    # 3. Load & clean NFHS-5 raw indicators
    canonical_metadata = {}
    nfhs_district_records = {}  # (sk, dk) -> {num -> row_data}

    if os.path.exists(nfhs_path):
        raw_nfhs = pd.read_csv(nfhs_path)

        # Drop rows without indicator number prefix 1-104
        num_extracted = raw_nfhs["Indicator"].astype(str).str.extract(r"^(\d+)\.")[0]
        raw_nfhs["num"] = pd.to_numeric(num_extracted, errors="coerce")
        valid_mask = raw_nfhs["num"].notna() & (raw_nfhs["num"] >= 1) & (raw_nfhs["num"] <= 104)
        raw_nfhs = raw_nfhs[valid_mask].copy()
        raw_nfhs["num"] = raw_nfhs["num"].astype(int)

        # Drop the 3 garbled single-district rows (<10 districts coverage)
        raw_nfhs = raw_nfhs[~raw_nfhs["Indicator"].str.contains("Cmhoiltdhreern", na=False)]
        raw_nfhs = raw_nfhs[~raw_nfhs["Indicator"].str.endswith("vaccination card or")]
        raw_nfhs = raw_nfhs[~raw_nfhs["Indicator"].str.endswith("or taking medicine to")]

        # Determine canonical label for each number 1-104 (highest district coverage)
        counts = raw_nfhs.groupby(["num", "Indicator"])["district"].nunique().reset_index()
        top_labels = counts.sort_values(["num", "district"], ascending=[True, False]).drop_duplicates("num")

        for _, r in top_labels.iterrows():
            n = int(r["num"])
            orig = str(r["Indicator"])
            raw_label = re.sub(rf"^{n}\.\s*", "", orig)
            cl = clean_footnote_digits(raw_label)

            # Apply specified prefixes
            if 20 <= n <= 27 and not cl.lower().startswith("family planning"):
                cl = f"Family planning: {cl}"
            elif 86 <= n <= 88:
                cl = f"Blood sugar (set A): {cl}"
            elif 89 <= n <= 91:
                cl = f"Blood sugar (set B): {cl}"
            elif 92 <= n <= 94:
                cl = f"Blood pressure (set A): {cl}"
            elif 95 <= n <= 97:
                cl = f"Blood pressure (set B): {cl}"

            canonical_metadata[n] = {
                "label": cl,
                "group": get_indicator_group(n),
            }

        # Build district index
        raw_nfhs["sk"] = raw_nfhs["state"].map(state_key)
        raw_nfhs["dk"] = raw_nfhs["district"].map(clean_key)

        for _, row in raw_nfhs.iterrows():
            sk = row["sk"]
            dk = row["dk"]
            num = int(row["num"])

            val5_raw = row.get("NFHS5")
            flag5_raw = row.get("Flag_NFHS5")
            val4_raw = row.get("NFHS4")
            flag4_raw = row.get("Flag_NFHS4")

            flagged = pd.notna(flag5_raw) and str(flag5_raw).strip() != ""
            flag_text = str(flag5_raw).strip() if flagged else None

            # Suppress values marked unreliable by survey
            nfhs5 = None
            if not flagged and pd.notna(val5_raw):
                try:
                    nfhs5 = round(float(val5_raw), 1)
                except (ValueError, TypeError):
                    nfhs5 = None

            flagged4 = pd.notna(flag4_raw) and str(flag4_raw).strip() != ""
            nfhs4 = None
            if not flagged4 and pd.notna(val4_raw):
                try:
                    nfhs4 = round(float(val4_raw), 1)
                except (ValueError, TypeError):
                    nfhs4 = None

            key = (sk, dk)
            if key not in nfhs_district_records:
                nfhs_district_records[key] = {}
            nfhs_district_records[key][num] = {
                "nfhs5": nfhs5,
                "nfhs4": nfhs4,
                "flagged": flagged,
                "flag_text": flag_text,
            }

    _cache = {
        "final_df": final_df,
        "feature_stats": feature_stats,
        "acc_lookup": acc_lookup,
        "canonical_metadata": canonical_metadata,
        "nfhs_district_records": nfhs_district_records,
    }
    return _cache


def reset_cache():
    """Clear cached datasets (used by tests)."""
    global _cache
    _cache = None


def get_area_profile(area_id: int) -> Optional[Dict[str, Any]]:
    """Generate the complete District Indicators profile for the given area_id."""
    data = _load()
    final_df = data["final_df"]
    feature_stats = data["feature_stats"]
    acc_lookup = data["acc_lookup"]
    canonical_metadata = data["canonical_metadata"]
    nfhs_district_records = data["nfhs_district_records"]

    rows = final_df[final_df["area_id"] == area_id]
    if rows.empty:
        return None

    row = rows.iloc[0]
    total_districts = len(final_df)

    # 1. Model inputs (exactly 4 satellite features)
    model_inputs = []
    for spec in MODEL_INPUT_SPECS:
        k = spec["key"]
        raw_val = row.get(k)
        if pd.notna(raw_val):
            val = round(float(raw_val), 3)
            # Empirical percentile: share of districts with value <= this district's value
            pct = round(float((feature_stats[k]["series"] <= float(raw_val)).mean() * 100.0), 1)
        else:
            val = None
            pct = None

        model_inputs.append({
            "key": k,
            "label": spec["label"],
            "value": val,
            "unit": spec["unit"],
            "description": spec["description"],
            "percentile": pct,
            "dataset_min": feature_stats[k]["min"],
            "dataset_median": feature_stats[k]["median"],
            "dataset_max": feature_stats[k]["max"],
        })

    # 2. Location & data quality
    match_method = str(row.get("match_method", "unknown"))
    match_caution = match_method in ("fuzzy", "state_fallback")
    location_quality = {
        "latitude": round(float(row["latitude"]), 4) if pd.notna(row.get("latitude")) else None,
        "longitude": round(float(row["longitude"]), 4) if pd.notna(row.get("longitude")) else None,
        "s2_image_count": int(row["s2_image_count"]) if pd.notna(row.get("s2_image_count")) else None,
        "match_method": match_method,
        "match_caution": match_caution,
        "data_source": str(row.get("data_source", "Unknown")),
        "data_year": int(row["data_year"]) if pd.notna(row.get("data_year")) else None,
    }

    # 3. Accessibility features
    sk = row["sk"]
    area_dk = row["area_dk"]
    acc_data = acc_lookup.get((sk, area_dk))
    if not acc_data and "dk" in row:
        acc_data = acc_lookup.get((sk, row["dk"]))

    if acc_data and any(acc_data.values()):
        accessibility = {
            "available": True,
            "items": [
                {
                    "key": "road_density_km_per_sqkm",
                    "label": "Road Network Density",
                    "value": acc_data["road_density_km_per_sqkm"],
                    "unit": "km/km²",
                },
                {
                    "key": "school_distance_km",
                    "label": "Mean Distance to School",
                    "value": acc_data["school_distance_km"],
                    "unit": "km",
                },
                {
                    "key": "hospital_distance_km",
                    "label": "Mean Distance to Hospital",
                    "value": acc_data["hospital_distance_km"],
                    "unit": "km",
                },
            ],
        }
    else:
        accessibility = {
            "available": False,
            "items": [],
        }

    # 4. Survey components (5 indicators behind deprivation score)
    nfhs_sk = row["sk"]
    nfhs_dk = row["dk"]
    district_nfhs = nfhs_district_records.get((nfhs_sk, nfhs_dk), {})

    component_indicators = []
    for comp in SURVEY_COMPONENT_SPECS:
        num = comp["num"]
        meta = canonical_metadata.get(num, {"label": f"Indicator {num}", "group": "Other"})
        rec = district_nfhs.get(num, {})

        nfhs5 = rec.get("nfhs5")
        nfhs4 = rec.get("nfhs4")
        flagged = rec.get("flagged", False)
        flag_text = rec.get("flag_text")

        change = round(nfhs5 - nfhs4, 1) if (nfhs5 is not None and nfhs4 is not None) else None

        component_indicators.append({
            "key": comp["key"],
            "label": meta["label"],
            "nfhs5": nfhs5,
            "nfhs4": nfhs4,
            "change": change,
            "unit": "%",
            "higher_is_worse": comp["higher_is_worse"],
            "flagged": flagged,
            "flag_text": flag_text,
        })

    composite_score = round(float(row["deprivation_score"]), 4) if pd.notna(row.get("deprivation_score")) else None
    survey_components = {
        "composite_score": composite_score,
        "note": (
            "These 5 survey indicators from NFHS-5 (2019-21) were normalized and averaged into the "
            "composite deprivation score that the AI model attempts to predict. They are shown for context "
            "and are NOT predictors used by the satellite model."
        ),
        "indicators": component_indicators,
    }

    # 5. All NFHS-5 indicators (ordered by indicator number 1-104)
    all_nfhs5 = []
    if district_nfhs:
        for num in range(1, 105):
            if num not in canonical_metadata:
                continue
            meta = canonical_metadata[num]
            rec = district_nfhs.get(num, {})
            all_nfhs5.append({
                "number": num,
                "label": meta["label"],
                "group": meta["group"],
                "nfhs5": rec.get("nfhs5"),
                "nfhs4": rec.get("nfhs4"),
                "flag_text": rec.get("flag_text"),
            })

    from backend.services import indicator_service
    sat_inds = indicator_service.get_satellite_indicators(area_id)

    return {
        "area_id": int(row["area_id"]),
        "area_name": str(row["area_name"]),
        "state_name": str(row["state_name"]),
        "data_mode": str(row.get("data_mode", "real")),
        "model_inputs": model_inputs,
        "satellite_indicators": sat_inds,
        "location_quality": location_quality,
        "survey_components": survey_components,
        "accessibility": accessibility,
        "all_nfhs5": all_nfhs5,
    }
