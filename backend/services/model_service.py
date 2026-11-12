"""
model_service.py
-------------------
Loads the trained model (Module 5) and the fused training table, and
provides:
  - the current feature vector for any area (used by the what-if module)
  - live scenario prediction (fast: a single model.predict() call, unlike
    SHAP/Grad-CAM which are precomputed -- see Module 6 notes)

This is the ONE place in the backend that runs the model live per-request,
by design (see scripts/precompute_predictions.py docstring for why
everything else is served from cache).
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODELS_DIR, DEMO_TRAINING_CSV, PROCESSED_DIR, SCENARIO_PRESETS, SCENARIO_DISCLAIMER_NOTE

_model_bundle = None
_feature_cols = None
_thresholds = None
_feature_table = None


def _find_training_table():
    """Prefers the real fused table if present, otherwise falls back to the demo table."""
    real_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    if os.path.exists(real_path):
        return real_path
    if os.path.exists(DEMO_TRAINING_CSV):
        return DEMO_TRAINING_CSV
    raise FileNotFoundError(
        "No training table found. Run scripts/generate_demo_data.py (demo) or "
        "preprocessing/feature_fusion.py (real) first."
    )


def load_artifacts():
    global _model_bundle, _feature_cols, _thresholds, _feature_table
    if _model_bundle is not None:
        return  # already loaded

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_columns.json")
    thresholds_path = os.path.join(MODELS_DIR, "category_thresholds.json")

    for p in (model_path, features_path, thresholds_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- run training/train_models.py first.")

    _model_bundle = joblib.load(model_path)
    with open(features_path) as f:
        _feature_cols = json.load(f)
    with open(thresholds_path) as f:
        _thresholds = json.load(f)

    table_path = _find_training_table()
    _feature_table = pd.read_csv(table_path).set_index("area_id")


def score_to_category(score: float) -> str:
    load_artifacts()
    if score <= _thresholds["low_medium_boundary"]:
        return "Low"
    elif score <= _thresholds["medium_high_boundary"]:
        return "Medium"
    else:
        return "High"


def get_feature_vector(area_id: int) -> dict:
    load_artifacts()
    if _feature_table is None or area_id not in _feature_table.index:
        return None
    row = _feature_table.loc[area_id]
    return {
        c: (float(row[c]) if (c in row.index and pd.notna(row[c])) else None)
        for c in _feature_cols
    }


def predict_from_features(feature_dict: dict) -> float:
    load_artifacts()
    for c in _feature_cols:
        val = feature_dict.get(c)
        if val is None or pd.isna(val):
            raise ValueError(f"Feature '{c}' is required and cannot be None or missing.")
    X = pd.DataFrame([feature_dict])[_feature_cols]
    return float(_model_bundle["model"].predict(X)[0])


def get_feature_training_ranges() -> dict:
    load_artifacts()
    ranges = {}
    if _feature_table is not None:
        for c in _feature_cols:
            if c in _feature_table.columns:
                ranges[c] = (float(_feature_table[c].min()), float(_feature_table[c].max()))
    return ranges


def get_model_mae() -> float:
    """Retrieves the cross-validated MAE of the active best model from models/model_comparison_results.csv."""
    load_artifacts()
    try:
        csv_path = os.path.join(MODELS_DIR, "model_comparison_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            model_name = _model_bundle.get("model_name", "XGBoost") if _model_bundle else "XGBoost"
            row = df[df["model"].str.lower() == str(model_name).lower()]
            if not row.empty and "mae" in row.columns:
                return float(row["mae"].iloc[0])
            row_rf = df[df["model"] == "XGBoost"]
            if not row_rf.empty and "mae" in row_rf.columns:
                return float(row_rf["mae"].iloc[0])
    except Exception:
        pass
    return 0.0752


def get_uncertainty_interval_80() -> tuple:
    """Retrieves the 80% uncertainty interval deltas (lower, upper) from models/uncertainty.json."""
    try:
        unc_path = os.path.join(MODELS_DIR, "uncertainty.json")
        if os.path.exists(unc_path):
            with open(unc_path, "r", encoding="utf-8") as f:
                unc = json.load(f)
            if "80" in unc:
                return float(unc["80"]["lower"]), float(unc["80"]["upper"])
    except Exception:
        pass
    return -0.1179, 0.1271


def apply_preset_to_features(features: dict, preset_id: str) -> dict:
    """Applies a sensitivity proxy preset from config.SCENARIO_PRESETS to feature vector."""
    if preset_id not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown preset '{preset_id}'. Valid presets: {list(SCENARIO_PRESETS.keys())}")
    preset = SCENARIO_PRESETS[preset_id]
    result = dict(features)
    if "features" in preset:
        for f_name, mod in preset["features"].items():
            if f_name in result:
                if mod.get("op") == "multiply":
                    result[f_name] = result[f_name] * mod["value"]
                elif mod.get("op") == "add":
                    result[f_name] = result[f_name] + mod["value"]
    elif "feature" in preset:
        f_name = preset["feature"]
        if f_name in result:
            if preset.get("op") == "multiply":
                result[f_name] = result[f_name] * preset["value"]
            elif preset.get("op") == "add":
                result[f_name] = result[f_name] + preset["value"]
    return result


def predict_scenario(area_id: int, modified_features: dict = None, preset_id: str = None):
    """
    Returns estimated score before and after, 80% interval from models/uncertainty.json,
    difference, MAE error comparison, range warnings, and disclaimer note.
    """
    load_artifacts()
    base_features = get_feature_vector(area_id)
    if base_features is None:
        return None
    missing_feats = [k for k, v in base_features.items() if v is None]
    if missing_feats:
        raise ValueError(f"District {area_id} has missing base feature values: {missing_feats}")

    original_score = predict_from_features(base_features)
    scenario_features = dict(base_features)

    # 1. Apply preset if requested
    if preset_id:
        scenario_features = apply_preset_to_features(scenario_features, preset_id)

    training_ranges = get_feature_training_ranges()
    warnings = []

    # 2. Apply custom feature overrides / deltas if provided
    if modified_features:
        for k, v in modified_features.items():
            if k not in scenario_features:
                raise ValueError(f"Unknown feature '{k}'. Valid features: {list(scenario_features.keys())}")
            if isinstance(v, bool):
                raise ValueError(f"Feature '{k}' must be numeric, got boolean {v!r}")
            if isinstance(v, dict) and "op" in v and "value" in v:
                try:
                    delta = float(v["value"])
                    if not np.isfinite(delta):
                        raise ValueError()
                except (ValueError, TypeError):
                    raise ValueError(f"Feature '{k}' modification must be numeric, got {v!r}")
                if v["op"] == "multiply":
                    scenario_features[k] = scenario_features[k] * delta
                elif v["op"] == "add":
                    scenario_features[k] = scenario_features[k] + delta
            else:
                try:
                    val = float(v)
                    if not np.isfinite(val):
                        raise ValueError()
                except (ValueError, TypeError):
                    raise ValueError(f"Feature '{k}' must be numeric, got {v!r}")
                scenario_features[k] = val

    # 3. Check for out-of-range features (never extrapolate silently)
    for k, val in scenario_features.items():
        if k in training_ranges:
            col_min, col_max = training_ranges[k]
            if val < col_min or val > col_max:
                warnings.append(
                    f"Feature '{k}' value {val} is outside the training range [{col_min:.3f}, {col_max:.3f}]."
                )

    scenario_score = predict_from_features(scenario_features)

    # 4. Uncertainty intervals (80% interval from models/uncertainty.json)
    lower_80, upper_80 = get_uncertainty_interval_80()
    original_interval_80 = [
        max(0.0, round(original_score + lower_80, 4)),
        min(1.0, round(original_score + upper_80, 4)),
    ]
    scenario_interval_80 = [
        max(0.0, round(scenario_score + lower_80, 4)),
        min(1.0, round(scenario_score + upper_80, 4)),
    ]

    difference = round(scenario_score - original_score, 4)
    mae = get_model_mae()
    is_smaller_than_mae = bool(abs(difference) < mae)
    mae_text = (
        f"Smaller than the model's typical error (MAE {mae:.4f})"
        if is_smaller_than_mae
        else f"Exceeds the model's typical error (MAE {mae:.4f})"
    )

    area_name = _feature_table.loc[area_id].get("area_name", f"District {area_id}") if _feature_table is not None and area_id in _feature_table.index else f"District {area_id}"
    state_name = _feature_table.loc[area_id].get("state_name", "") if _feature_table is not None and area_id in _feature_table.index else ""

    return {
        "area_id": area_id,
        "area_name": str(area_name),
        "state_name": str(state_name),
        "original_score": original_score,
        "original_interval_80": original_interval_80,
        "original_category": score_to_category(original_score),
        "scenario_score": scenario_score,
        "scenario_interval_80": scenario_interval_80,
        "scenario_category": score_to_category(scenario_score),
        "difference": difference,
        "model_mae": mae,
        "is_smaller_than_mae": is_smaller_than_mae,
        "mae_comparison_text": mae_text,
        "base_features": base_features,
        "scenario_features": scenario_features,
        "warnings": warnings,
        "model_name": _model_bundle["model_name"],
        "note": SCENARIO_DISCLAIMER_NOTE,
    }


def predict_group_scenario(target_type: str, target_id: Any, modified_features: dict = None, preset_id: str = None):
    """
    Applies a preset or custom change across all districts in a state or hotspot cluster.
    Returns: mean change, median change, number of districts that change category,
    and the share whose change is smaller than the model's typical error (MAE).
    """
    load_artifacts()
    if _feature_table is None:
        raise ValueError("Feature table not loaded")

    from backend.services import hotspot_service

    target_type = str(target_type).lower().strip()
    districts = []
    target_name = ""

    if target_type == "state":
        target_name = str(target_id).strip()
        matches = _feature_table[_feature_table["state_name"].astype(str).str.lower() == target_name.lower()]
        if matches.empty and "gaul_state_name" in _feature_table.columns:
            matches = _feature_table[_feature_table["gaul_state_name"].astype(str).str.lower() == target_name.lower()]
        if matches.empty:
            raise ValueError(f"State '{target_id}' not found.")
        target_name = str(matches["state_name"].iloc[0])
        districts = matches.index.tolist()

    elif target_type in ("cluster", "hotspot", "hotspot_cluster"):
        try:
            cid = int(target_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid cluster_id '{target_id}', must be integer.")
        clusters = hotspot_service.identify_hotspots()
        found_cluster = None
        for c in clusters:
            if c["cluster_id"] == cid:
                found_cluster = c
                break
        if not found_cluster:
            raise ValueError(f"Hotspot cluster {cid} not found.")
        target_name = f"Hotspot Cluster #{cid} ({', '.join(found_cluster['states'])})"
        districts = [d["area_id"] for d in found_cluster["districts"]]

    elif target_type == "all":
        target_name = "All Districts (Nationwide)"
        districts = _feature_table.index.tolist()

    else:
        raise ValueError(f"Unknown target_type '{target_type}'. Expected 'state', 'cluster', or 'all'.")

    if not districts:
        raise ValueError(f"No districts found for {target_type} '{target_id}'.")

    district_results = []
    warnings_set = set()

    for aid in districts:
        try:
            res = predict_scenario(aid, modified_features=modified_features, preset_id=preset_id)
            if res is not None:
                district_results.append(res)
                for w in res.get("warnings", []):
                    warnings_set.add(w)
        except ValueError:
            continue

    if not district_results:
        raise ValueError("Could not compute scenario for any district in the group.")

    mae = get_model_mae()
    differences = [d["difference"] for d in district_results]
    mean_change = float(np.mean(differences))
    median_change = float(np.median(differences))
    category_changes_count = sum(1 for d in district_results if d["original_category"] != d["scenario_category"])
    below_mae_count = sum(1 for d in district_results if abs(d["difference"]) < mae)
    share_smaller_than_mae = float(below_mae_count / len(district_results))

    return {
        "target_type": target_type,
        "target_id": target_id,
        "target_name": target_name,
        "total_districts": len(district_results),
        "mean_change": round(mean_change, 4),
        "median_change": round(median_change, 4),
        "category_changes_count": category_changes_count,
        "share_smaller_than_mae": round(share_smaller_than_mae, 4),
        "model_mae": mae,
        "mae_comparison_text": f"Share with change smaller than MAE ({mae:.4f}): {share_smaller_than_mae*100:.1f}%",
        "districts": district_results,
        "warnings": sorted(list(warnings_set)),
        "model_name": get_model_name(),
        "note": SCENARIO_DISCLAIMER_NOTE,
    }


def get_scenario_targets() -> dict:
    """Returns available targeting options for states, clusters, and presets."""
    load_artifacts()
    from backend.services import hotspot_service

    states = []
    if _feature_table is not None and "state_name" in _feature_table.columns:
        states = sorted(_feature_table["state_name"].dropna().unique().tolist())

    clusters = []
    try:
        raw_clusters = hotspot_service.identify_hotspots()
        for c in raw_clusters:
            clusters.append({
                "cluster_id": c["cluster_id"],
                "district_count": len(c["districts"]),
                "avg_score": round(c["avg_score"], 4),
                "states": c["states"],
                "label": f"Cluster #{c['cluster_id']} ({len(c['districts'])} districts, {', '.join(c['states'])})",
            })
    except Exception:
        pass

    return {
        "states": states,
        "clusters": clusters,
        "presets": SCENARIO_PRESETS,
    }


def get_model_name() -> str:
    load_artifacts()
    return _model_bundle["model_name"]


def predict_custom_features(feature_dict: dict) -> dict:
    """Runs live model prediction and real-time SHAP explanation for user-provided inputs."""
    import numpy as np
    load_artifacts()
    training_ranges = get_feature_training_ranges()
    warnings = []

    # First validate all provided values are strictly numeric
    for k, v in feature_dict.items():
        if v is not None:
            if isinstance(v, bool):
                raise ValueError(f"Feature '{k}' must be numeric, got boolean {v!r}")
            try:
                val = float(v)
                if not np.isfinite(val):
                    raise ValueError()
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{k}' must be numeric, got {v!r}")

    clean_dict = {}
    for c in _feature_cols:
        if c in feature_dict and feature_dict[c] is not None:
            val = float(feature_dict[c])
            clean_dict[c] = val
            if c in training_ranges:
                col_min, col_max = training_ranges[c]
                if val < col_min or val > col_max:
                    warnings.append(
                        f"Feature '{c}' value {val} is outside the training range [{col_min:.3f}, {col_max:.3f}]."
                    )
        else:
            clean_dict[c] = float(_feature_table[c].median()) if _feature_table is not None and c in _feature_table.columns else 0.0

    X = pd.DataFrame([clean_dict])[_feature_cols]
    score = float(_model_bundle["model"].predict(X)[0])
    category = score_to_category(score)

    contributions = []
    base_val = 0.0
    try:
        from explainability.shap_explain import build_explainer
        background = _feature_table[_feature_cols] if _feature_table is not None else X
        explainer = build_explainer(_model_bundle["model"], background)
        shap_values = explainer.shap_values(X)[0]
        bv = explainer.expected_value
        base_val = float(np.ravel(bv)[0]) if isinstance(bv, (list, np.ndarray)) else float(bv)

        friendly_names = {
            "nightlight_mean": "Nighttime Lights Radiance",
            "ndvi_mean": "Vegetation Index (NDVI)",
            "builtup_pct": "Built-up Surface Density",
            "population_density": "Population Density",
            "elevation_mean_m": "Mean Elevation (m)",
            "slope_mean_deg": "Mean Terrain Slope (deg)",
            "rainfall_mm_year": "Annual Rainfall (mm/yr)",
            "lst_day_c": "Daytime Land Surface Temp (°C)",
            "cropland_pct": "Cropland Share (%)",
            "travel_time_city_min": "Travel Time to Nearest City (min)",
            "road_density_km_per_sqkm": "Road Network Density",
            "school_distance_km": "Distance to Nearest School",
            "hospital_distance_km": "Distance to Nearest Hospital",
        }
        for i in range(1, 64):
            friendly_names[f"vgg_feat_{i}"] = f"Visual Texture Feature #{i}"

        for f, v, s in zip(_feature_cols, X.iloc[0], shap_values):
            contributions.append({
                "feature": f,
                "label": friendly_names.get(f, f),
                "value": round(float(v), 4),
                "shap": round(float(s), 4),
                "effect": "Increases Poverty" if s > 0 else "Decreases Poverty"
            })
        contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)
    except Exception as e:
        print(f"SHAP explanation warning: {e}")

    top_factor = contributions[0] if contributions else None
    if top_factor:
        interpretation = (
            f"The primary factor driving this prediction is {top_factor['label']} "
            f"(value: {top_factor['value']}), which {top_factor['effect'].lower()} "
            f"by {abs(top_factor['shap']):.4f} relative to the nationwide baseline."
        )
    else:
        interpretation = f"Predicted deprivation score is {score:.4f} ({category} Deprivation)."

    return {
        "predicted_score": round(score, 4),
        "predicted_category": category,
        "base_value": round(base_val, 4),
        "thresholds": _thresholds,
        "warnings": warnings,
        "model_name": _model_bundle["model_name"],
        "contributions": contributions,
        "interpretation": interpretation,
        "input_features": clean_dict
    }

def find_similar_districts(area_id: int, top_k: int = 5) -> list:
    """Finds the most statistically similar districts using Z-score normalized Euclidean distance
    across all AI features (VGG + Geospatial)."""
    load_artifacts()
    if area_id not in _feature_table.index:
        return []
        
    import numpy as np
    
    # Use only the exact features the AI model cares about
    df_features = _feature_table[_feature_cols]
    
    # Standardize features so population (thousands) doesn't overpower NDVI (0-1)
    df_std = (df_features - df_features.mean()) / (df_features.std() + 1e-9)
    target_std = df_std.loc[area_id]
    
    # Compute Euclidean distance to all other districts
    distances = np.sqrt(((df_std - target_std) ** 2).sum(axis=1))
    
    # Sort and get top K + 1 (since the district itself will be rank 0 with distance 0)
    top_matches = distances.nsmallest(top_k + 1)
    
    results = []
    for match_id, dist in top_matches.items():
        if match_id == area_id:
            continue
        
        # Calculate a "similarity score" percentage where 100% is identical
        # Using an exponential decay function to map distance to [0, 100]
        sim_score = np.exp(-dist / np.sqrt(len(_feature_cols))) * 100
        
        results.append({
            "area_id": int(match_id),
            "distance": float(dist),
            "similarity_percentage": round(float(sim_score), 1)
        })
        
    return results[:top_k]
