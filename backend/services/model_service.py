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
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODELS_DIR, DEMO_TRAINING_CSV, PROCESSED_DIR
from training.transformers import IQRClipper, TrainingMedianImputer, MeanPredictor

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
    if area_id not in _feature_table.index:
        return None
    row = _feature_table.loc[area_id]
    return {c: float(row[c]) if c in row.index else 0.0 for c in _feature_cols}


def get_prediction_interval(predicted_score: float):
    """
    Returns (predicted_low, predicted_high) using empirical residual quantiles (80% confidence interval)
    computed from Leave-One-State-Out cross-validation.
    """
    load_artifacts()
    unc = _model_bundle.get("uncertainty", {})
    delta_low = unc.get("lower_delta", -0.1449)
    delta_high = unc.get("upper_delta", 0.1377)
    pred_low = max(0.0, round(float(predicted_score + delta_low), 4))
    pred_high = min(1.0, round(float(predicted_score + delta_high), 4))
    return pred_low, pred_high


def predict_from_features(feature_dict: dict) -> float:
    load_artifacts()
    # Align active features
    aligned = {c: float(feature_dict.get(c, 0.0)) for c in _feature_cols}
    X = pd.DataFrame([aligned])[_feature_cols]
    return float(_model_bundle["model"].predict(X)[0])


def predict_scenario(area_id: int, modified_features: dict):
    """
    Returns (original_score, scenario_score, category_original, category_scenario)
    for area_id, with modified_features applied on top of its current values.
    """
    base_features = get_feature_vector(area_id)
    if base_features is None:
        return None

    original_score = predict_from_features(base_features)

    scenario_features = dict(base_features)
    for k, v in modified_features.items():
        if k not in scenario_features:
            raise ValueError(f"Unknown feature '{k}'. Valid features: {list(scenario_features.keys())}")
        scenario_features[k] = float(v)
    scenario_score = predict_from_features(scenario_features)

    return {
        "original_score": original_score,
        "original_category": score_to_category(original_score),
        "scenario_score": scenario_score,
        "scenario_category": score_to_category(scenario_score),
        "difference": scenario_score - original_score,
        "model_name": _model_bundle["model_name"],
        "note": "This is a model sensitivity exploration, not a causal simulation. "
                "Real-world interventions may not produce the same change.",
    }


def get_model_name() -> str:
    load_artifacts()
    return _model_bundle["model_name"]


def predict_custom_features(feature_dict: dict) -> dict:
    """Runs live model prediction and real-time SHAP explanation for user-provided inputs."""
    import numpy as np
    load_artifacts()

    clean_dict = {}
    for c in _feature_cols:
        if c in feature_dict and feature_dict[c] is not None:
            try:
                clean_dict[c] = float(feature_dict[c])
            except (ValueError, TypeError):
                clean_dict[c] = float(_feature_table[c].median()) if _feature_table is not None and c in _feature_table.columns else 0.0
        else:
            clean_dict[c] = float(_feature_table[c].median()) if _feature_table is not None and c in _feature_table.columns else 0.0

    X = pd.DataFrame([clean_dict])[_feature_cols]
    score = float(_model_bundle["model"].predict(X)[0])
    category = score_to_category(score)
    pred_low, pred_high = get_prediction_interval(score)

    contributions = []
    base_val = 0.0
    try:
        import shap
        model_obj = _model_bundle["model"]
        if hasattr(model_obj, "named_steps"):
            tree_model = model_obj.named_steps["model"]
            X_trans = X.copy()
            if "clipper" in model_obj.named_steps:
                X_trans = model_obj.named_steps["clipper"].transform(X_trans)
            if "imputer" in model_obj.named_steps:
                X_trans = model_obj.named_steps["imputer"].transform(X_trans)
            if "scaler" in model_obj.named_steps:
                X_trans = model_obj.named_steps["scaler"].transform(X_trans)
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_trans)[0]
        else:
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X)[0]
        bv = explainer.expected_value
        base_val = float(np.ravel(bv)[0]) if isinstance(bv, (list, np.ndarray)) else float(bv)

        friendly_names = {
            "nightlight_mean": "Nighttime Lights Radiance",
            "ndvi_mean": "Vegetation Index (NDVI)",
            "builtup_pct": "Built-up Surface Density",
            "population_density": "Population Density",
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
        "predicted_low": pred_low,
        "predicted_high": pred_high,
        "predicted_category": category,
        "base_value": round(base_val, 4),
        "thresholds": _thresholds,
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
