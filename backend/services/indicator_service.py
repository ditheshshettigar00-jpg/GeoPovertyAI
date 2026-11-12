"""
indicator_service.py - Real Indicator Service for GeoPoverty AI.

Supplies complete, real satellite-derived telemetry and NFHS ground truth survey indicators
for every district in the dataset, with national/state medians, empirical percentiles,
ranks, and survey quality flags.

Rules:
- Never returns placeholder sample data.
- Never defaults missing values to 0.0.
- Joins final training table to NFHS-5 raw survey records using clean_key.
- Caches all loaded tables in memory for fast performance.
"""

import os
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import DATA_DIR, PROCESSED_DIR, GROUND_TRUTH_DIR, GEE_START_DATE, GEE_END_DATE
from preprocessing.name_keys import clean_key

# Global cache
_cache = None


def _clean_indicator_label(text: str) -> str:
    """Removes leading numbering (e.g. '7. ') and footnote superscripts without touching legitimate numbers."""
    t = re.sub(r"^\d+\.\s*", "", str(text)).strip()
    # Strip footnote digits attached to words, e.g. 'facility2', 'cooking3', '(height-for-age)18', 'anaemic22'
    t = re.sub(r"(?<=[a-zA-Z\)])\d+(?=\s*\(|\s*$)", "", t)
    return t.strip()


def _parse_indicator_number(text: str) -> int:
    """Extracts integer indicator number from indicator label."""
    m = re.match(r"^(\d+)\.", str(text).strip())
    return int(m.group(1)) if m else 9999


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache

    final_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    nfhs_path = os.path.join(GROUND_TRUTH_DIR, "nfhs5_districts_raw.csv")

    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Missing training table at {final_path}")
    if not os.path.exists(nfhs_path):
        raise FileNotFoundError(f"Missing NFHS-5 raw dataset at {nfhs_path}")

    final_df = pd.read_csv(final_path)
    final_df["sk"] = final_df["state_name"].map(clean_key)
    final_df["dk"] = final_df["nfhs_district_name"].map(clean_key)
    final_by_id = final_df.set_index("area_id")

    nfhs_df = pd.read_csv(nfhs_path)
    nfhs_df["sk"] = nfhs_df["state"].map(clean_key)
    nfhs_df["dk"] = nfhs_df["district"].map(clean_key)
    nfhs_df["num"] = nfhs_df["Indicator"].map(_parse_indicator_number)
    nfhs_df["clean_label"] = nfhs_df["Indicator"].map(_clean_indicator_label)

    # Pre-parse numeric columns for NFHS
    nfhs_df["nfhs5_num"] = pd.to_numeric(nfhs_df["NFHS5"], errors="coerce")
    nfhs_df["nfhs4_num"] = pd.to_numeric(nfhs_df["NFHS4"], errors="coerce")

    # Group NFHS records by (sk, dk)
    nfhs_records = {}
    for (sk, dk), grp in nfhs_df.groupby(["sk", "dk"]):
        sorted_grp = grp.sort_values("num")
        nfhs_records[(sk, dk)] = sorted_grp

    # Precompute survey national and state medians and series for percentiles
    survey_stats = {}
    for num, grp in nfhs_df.groupby("num"):
        valid_5 = grp["nfhs5_num"].dropna()
        state_meds = {}
        for sk, s_grp in grp.groupby("sk"):
            s_valid = s_grp["nfhs5_num"].dropna()
            if not s_valid.empty:
                state_meds[sk] = round(float(s_valid.median()), 1)
        survey_stats[num] = {
            "nat_median": round(float(valid_5.median()), 1) if not valid_5.empty else None,
            "state_medians": state_meds,
            "series": valid_5.values,
        }

    # Precompute satellite statistics across all 564 districts
    sat_cols = [
        "nightlight_mean",
        "ndvi_mean",
        "builtup_pct",
        "population_density",
        "elevation_mean_m",
        "slope_mean_deg",
        "rainfall_mm_year",
        "lst_day_c",
        "cropland_pct",
        "travel_time_city_min",
        "s2_image_count",
        "latitude",
        "longitude",
    ]
    sat_stats = {}
    for c in sat_cols:
        series = final_df[c].dropna()
        state_meds = {}
        for sk, s_grp in final_df.groupby("sk"):
            s_valid = s_grp[c].dropna()
            if not s_valid.empty:
                state_meds[sk] = float(s_valid.median())
        sat_stats[c] = {
            "nat_median": float(series.median()),
            "state_medians": state_meds,
            "series": series.values,
        }

    _cache = {
        "final_df": final_df,
        "final_by_id": final_by_id,
        "nfhs_records": nfhs_records,
        "survey_stats": survey_stats,
        "sat_stats": sat_stats,
    }
    return _cache


def get_satellite_indicators(area_id: int) -> List[Dict[str, Any]]:
    """
    Returns the 13 satellite and geospatial telemetry indicators for the given district.
    Columns: nightlight_mean, ndvi_mean, builtup_pct, population_density,
             elevation_mean_m, slope_mean_deg, rainfall_mm_year, lst_day_c,
             cropland_pct, travel_time_city_min, s2_image_count, latitude, longitude.
    """
    cache = _load_cache()
    final_by_id = cache["final_by_id"]
    if area_id not in final_by_id.index:
        return []

    row = final_by_id.loc[area_id]
    sk = row["sk"]
    sat_stats = cache["sat_stats"]

    period_str = f"{GEE_START_DATE} to {GEE_END_DATE}"

    METAS = [
        {
            "key": "nightlight_mean",
            "label": "Nighttime Light Radiance",
            "unit": "nW/cm2/sr",
            "source": "VIIRS monthly",
            "period": period_str,
            "definition": "Mean nocturnal radiance across district boundary from VIIRS Day/Night Band monthly composites.",
            "disp_decimals": 3,
            "raw_decimals": 4,
        },
        {
            "key": "ndvi_mean",
            "label": "Normalized Difference Vegetation Index (NDVI)",
            "unit": "index (-1 to 1)",
            "source": "Sentinel-2 SR median NDVI",
            "period": period_str,
            "definition": "District-wide median Normalized Difference Vegetation Index from Sentinel-2 MSI surface reflectance.",
            "disp_decimals": 3,
            "raw_decimals": 4,
        },
        {
            "key": "builtup_pct",
            "label": "Built-Up Surface Area",
            "unit": "% of land area",
            "source": "GHSL built-up surface (epoch 2020)",
            "period": "Epoch 2020",
            "definition": "Percentage of district surface area covered by built-up human settlements from European Commission GHSL.",
            "disp_decimals": 2,
            "raw_decimals": 4,
        },
        {
            "key": "population_density",
            "label": "Population Density",
            "unit": "people per km2",
            "source": "GHSL population grid (epoch 2020)",
            "period": "Epoch 2020",
            "definition": "Spatial residential population density from European Commission JRC GHSL population grid.",
            "disp_decimals": 0,
            "raw_decimals": 1,
        },
        {
            "key": "elevation_mean_m",
            "label": "Mean Elevation",
            "unit": "m",
            "source": "NASA SRTM 30 m DEM",
            "period": "Static DEM",
            "definition": "Mean terrain elevation above sea level across the district polygon from NASA SRTM.",
            "disp_decimals": 1,
            "raw_decimals": 2,
        },
        {
            "key": "slope_mean_deg",
            "label": "Mean Terrain Slope",
            "unit": "deg",
            "source": "NASA SRTM DEM slope",
            "period": "Static DEM",
            "definition": "Mean terrain slope angle derived from SRTM elevation gradients.",
            "disp_decimals": 2,
            "raw_decimals": 2,
        },
        {
            "key": "rainfall_mm_year",
            "label": "Annual Rainfall",
            "unit": "mm/year",
            "source": "UCSB CHIRPS daily precipitation",
            "period": period_str,
            "definition": "Annualized mean precipitation across the survey window from CHIRPS v2.0.",
            "disp_decimals": 0,
            "raw_decimals": 2,
        },
        {
            "key": "lst_day_c",
            "label": "Daytime Land Surface Temp",
            "unit": "deg C",
            "source": "MODIS MOD11A2 8-day LST",
            "period": period_str,
            "definition": "Mean daytime land surface temperature across the survey window from MODIS Terra.",
            "disp_decimals": 1,
            "raw_decimals": 2,
        },
        {
            "key": "cropland_pct",
            "label": "Cropland Share",
            "unit": "% of land area",
            "source": "ESA WorldCover v200 (10 m)",
            "period": "Epoch 2021",
            "definition": "Share of district land covered by cropland in ESA WorldCover v200 (2021).",
            "disp_decimals": 1,
            "raw_decimals": 2,
        },
        {
            "key": "travel_time_city_min",
            "label": "Travel Time to Nearest City",
            "unit": "minutes",
            "source": "Oxford/MAP 2015 friction surface",
            "period": "2015 baseline",
            "definition": "Estimated surface travel time to nearest city (>50k pop) from 2015 accessibility map.",
            "disp_decimals": 1,
            "raw_decimals": 2,
        },
        {
            "key": "s2_image_count",
            "label": "Cloud-Free Sentinel-2 Scenes",
            "unit": "scenes",
            "source": "usable Sentinel-2 scenes in the window",
            "period": period_str,
            "definition": "Count of usable cloud-free Sentinel-2 MSI scenes captured within the district boundary.",
            "disp_decimals": 0,
            "raw_decimals": 0,
        },
        {
            "key": "latitude",
            "label": "Centroid Latitude",
            "unit": "degrees",
            "source": "district centroid",
            "period": "GAUL 2015",
            "definition": "Geographic latitude coordinate of administrative district centroid from GAUL 2015 boundaries.",
            "disp_decimals": 4,
            "raw_decimals": 4,
        },
        {
            "key": "longitude",
            "label": "Centroid Longitude",
            "unit": "degrees",
            "source": "district centroid",
            "period": "GAUL 2015",
            "definition": "Geographic longitude coordinate of administrative district centroid from GAUL 2015 boundaries.",
            "disp_decimals": 4,
            "raw_decimals": 4,
        },
    ]

    results = []
    n_districts = len(final_by_id)

    for m in METAS:
        k = m["key"]
        raw_val = row.get(k)
        is_avail = pd.notna(raw_val)

        if is_avail:
            num_val = float(raw_val)
            # Numeric value tailored to standard precision requirements
            if m["raw_decimals"] == 0:
                val = int(round(num_val))
            else:
                val = round(num_val, m["raw_decimals"])

            # Formatted text display representation
            if m["disp_decimals"] == 0:
                formatted_value = f"{int(round(num_val)):,}"
            elif m["disp_decimals"] == 2:
                formatted_value = f"{round(num_val, 2):.2f}"
            elif m["disp_decimals"] == 3:
                formatted_value = f"{round(num_val, 3):.3f}"
            else:
                formatted_value = f"{round(num_val, 4):.4f}"

            stats = sat_stats[k]
            series = stats["series"]
            pct = round(float((series <= num_val).mean() * 100.0), 1)
            rank = int((series > num_val).sum() + 1)
            nat_med = round(float(stats["nat_median"]), m["disp_decimals"] if m["disp_decimals"] > 0 else 1)
            st_med = stats["state_medians"].get(sk)
            if st_med is not None:
                st_med = round(float(st_med), m["disp_decimals"] if m["disp_decimals"] > 0 else 1)
        else:
            val = None
            formatted_value = "N/A"
            pct = None
            rank = None
            nat_med = None
            st_med = None

        results.append({
            "key": k,
            "label": m["label"],
            "value": val,
            "formatted_value": formatted_value,
            "unit": m["unit"],
            "source": m["source"],
            "period": m["period"],
            "definition": m["definition"],
            "national_median": nat_med,
            "state_median": st_med,
            "percentile": pct,
            "rank": rank,
            "n_districts": n_districts,
            "available": is_avail,
            "reason": None if is_avail else "Data not available in source dataset",
        })

    return results


def get_survey_indicators(area_id: int) -> List[Dict[str, Any]]:
    """
    Returns the five component survey indicators of the deprivation score:
    1. Electricity (indicator 7)
    2. Improved sanitation (indicator 9)
    3. Clean cooking fuel (indicator 10)
    4. Child stunting (indicator 73)
    5. Anaemia in women (indicator 84)
    """
    cache = _load_cache()
    final_by_id = cache["final_by_id"]
    if area_id not in final_by_id.index:
        return []

    row = final_by_id.loc[area_id]
    sk = row["sk"]
    dk = row["dk"]

    nfhs_records = cache["nfhs_records"]
    district_df = nfhs_records.get((sk, dk))
    survey_stats = cache["survey_stats"]

    COMPONENTS = [
        {
            "key": "electricity",
            "indicator_number": 7,
            "label": "Population living in households with electricity (%)",
            "better_when": "higher",
        },
        {
            "key": "sanitation",
            "indicator_number": 9,
            "label": "Population living in households that use an improved sanitation facility (%)",
            "better_when": "higher",
        },
        {
            "key": "clean_fuel",
            "indicator_number": 10,
            "label": "Households using clean fuel for cooking (%)",
            "better_when": "higher",
        },
        {
            "key": "stunting",
            "indicator_number": 73,
            "label": "Children under 5 years who are stunted (height-for-age) (%)",
            "better_when": "lower",
        },
        {
            "key": "anaemia",
            "indicator_number": 84,
            "label": "All women age 15-49 years who are anaemic (%)",
            "better_when": "lower",
        },
    ]

    results = []
    for c in COMPONENTS:
        num = c["indicator_number"]
        nfhs5_val = None
        nfhs4_val = None
        flag_raw = None
        flag_text = None
        actual_label = c["label"]

        if district_df is not None:
            match = district_df[district_df["num"] == num]
            if not match.empty:
                r_ind = match.iloc[0]
                if pd.notna(r_ind["clean_label"]):
                    actual_label = r_ind["clean_label"]
                if pd.notna(r_ind["nfhs5_num"]):
                    nfhs5_val = round(float(r_ind["nfhs5_num"]), 1)
                if pd.notna(r_ind["nfhs4_num"]):
                    nfhs4_val = round(float(r_ind["nfhs4_num"]), 1)
                flag_raw = r_ind.get("Flag_NFHS5")
                if pd.isna(flag_raw):
                    flag_raw = None
                else:
                    flag_raw = str(flag_raw).strip()

        # Flag and missing value interpretation
        if nfhs5_val is None or (flag_raw and "fewer than 25" in flag_raw.lower()):
            flag_text = "not shown (fewer than 25 cases)"
            nfhs5_val = None
        elif flag_raw and "25-49" in flag_raw:
            flag_text = "Based on 25-49 unweighted cases (small sample)"
        else:
            flag_text = flag_raw

        change = round(nfhs5_val - nfhs4_val, 1) if (nfhs5_val is not None and nfhs4_val is not None) else None

        stats = survey_stats.get(num, {})
        nat_med = stats.get("nat_median")
        st_med = stats.get("state_medians", {}).get(sk)

        if nfhs5_val is not None and "series" in stats and len(stats["series"]) > 0:
            pct = round(float((stats["series"] <= nfhs5_val).mean() * 100.0), 1)
        else:
            pct = None

        results.append({
            "key": c["key"],
            "indicator_number": num,
            "label": actual_label,
            "nfhs5": nfhs5_val,
            "nfhs4": nfhs4_val,
            "change": change,
            "unit": "%",
            "better_when": c["better_when"],
            "flag_text": flag_text,
            "state_median": st_med,
            "national_median": nat_med,
            "percentile": pct,
            "available": nfhs5_val is not None,
        })

    return results


def get_all_survey_indicators(area_id: int) -> List[Dict[str, Any]]:
    """
    Returns all NFHS-5 and NFHS-4 indicators available for the district (~104),
    sorted by indicator number with survey flags and labels.
    """
    cache = _load_cache()
    final_by_id = cache["final_by_id"]
    if area_id not in final_by_id.index:
        return []

    row = final_by_id.loc[area_id]
    sk = row["sk"]
    dk = row["dk"]

    nfhs_records = cache["nfhs_records"]
    district_df = nfhs_records.get((sk, dk))
    if district_df is None or district_df.empty:
        return []

    survey_stats = cache["survey_stats"]
    results = []

    for _, r_ind in district_df.iterrows():
        num = int(r_ind["num"])
        clean_lbl = str(r_ind["clean_label"])
        raw_indicator = str(r_ind["Indicator"])

        nfhs5_val = round(float(r_ind["nfhs5_num"]), 1) if pd.notna(r_ind["nfhs5_num"]) else None
        nfhs4_val = round(float(r_ind["nfhs4_num"]), 1) if pd.notna(r_ind["nfhs4_num"]) else None

        flag_5 = r_ind.get("Flag_NFHS5")
        flag_5_str = str(flag_5).strip() if pd.notna(flag_5) else None

        flag_4 = r_ind.get("Flag_NFHS4")
        flag_4_str = str(flag_4).strip() if pd.notna(flag_4) else None

        if nfhs5_val is None or (flag_5_str and "fewer than 25" in flag_5_str.lower()):
            flag_text = "not shown (fewer than 25 cases)"
            nfhs5_val = None
        elif flag_5_str and "25-49" in flag_5_str:
            flag_text = "Based on 25-49 unweighted cases (small sample)"
        else:
            flag_text = flag_5_str

        change = round(nfhs5_val - nfhs4_val, 1) if (nfhs5_val is not None and nfhs4_val is not None) else None

        # Unit determination (almost all are %, except sex ratios)
        unit = "%"
        if "per 1,000" in raw_indicator:
            unit = "per 1,000"

        stats = survey_stats.get(num, {})
        nat_med = stats.get("nat_median")
        st_med = stats.get("state_medians", {}).get(sk)

        if nfhs5_val is not None and "series" in stats and len(stats["series"]) > 0:
            pct = round(float((stats["series"] <= nfhs5_val).mean() * 100.0), 1)
        else:
            pct = None

        results.append({
            "indicator_number": num,
            "raw_indicator": raw_indicator,
            "label": clean_lbl,
            "nfhs5": nfhs5_val,
            "nfhs4": nfhs4_val,
            "change": change,
            "unit": unit,
            "flag_nfhs5": flag_5_str,
            "flag_nfhs4": flag_4_str,
            "flag_text": flag_text,
            "state_median": st_med,
            "national_median": nat_med,
            "percentile": pct,
        })

    results.sort(key=lambda x: x["indicator_number"])
    return results


def get_district_indicators(area_id: int, full: bool = False) -> Optional[Dict[str, Any]]:
    """
    Returns full indicators payload for the API and report endpoints.
    """
    cache = _load_cache()
    final_by_id = cache["final_by_id"]
    if area_id not in final_by_id.index:
        return None

    row = final_by_id.loc[area_id]
    satellite_indicators = get_satellite_indicators(area_id)
    survey_indicators = get_survey_indicators(area_id)

    payload = {
        "area_id": area_id,
        "area_name": str(row["area_name"]),
        "state_name": str(row["state_name"]),
        "satellite_indicators": satellite_indicators,
        "survey_indicators": survey_indicators,
    }

    if full:
        payload["all_survey_indicators"] = get_all_survey_indicators(area_id)

    return payload

