"""
routes.py
-----------
REST API endpoints for GeoPoverty AI. Registered under the /api prefix
(see backend/app.py).

Endpoints:
    GET  /api/areas
    GET  /api/predict/<area_id>
    GET  /api/map
    GET  /api/explain/<area_id>
    GET  /api/model-performance
    POST /api/scenario
"""

import os
import sys

from flask import Blueprint, jsonify, request, send_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SATELLITE_DIR
from backend.api.auth import roles_required
from backend.services import db_service, model_service, hotspot_service, residual_service, profile_service

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _find_satellite_image(area_id: int, area_name: str = None, state_name: str = None, data_mode: str = "real"):
    """
    Returns (path, is_demo) or (None, False).
    - For real districts (data_mode == 'real'):
      Only returns an image if a real image exists for that district's state and name
      (file name pattern state_district_rgb.png).
      Never returns a demo image (name contains DEMO).
    - For demo districts (data_mode == 'demo'):
      Can return demo images (labeled as illustrative demo images).
    """
    if area_name is None or state_name is None:
        area = db_service.get_area(area_id)
        if area:
            area_name = area.get("area_name", "")
            state_name = area.get("state_name", "")
            data_mode = area.get("data_mode", data_mode)
        else:
            area_name = ""
            state_name = ""

    if data_mode == "real":
        if state_name and area_name:
            from preprocessing.name_keys import clean_key
            sk = clean_key(state_name)
            dk = clean_key(area_name)
            s_under = state_name.strip().replace(" ", "_")
            d_under = area_name.strip().replace(" ", "_")
            s_lower_under = state_name.strip().lower().replace(" ", "_")
            d_lower_under = area_name.strip().lower().replace(" ", "_")

            candidates = [
                os.path.join(SATELLITE_DIR, f"{sk}_{dk}_rgb.png"),
                os.path.join(SATELLITE_DIR, f"{s_lower_under}_{d_lower_under}_rgb.png"),
                os.path.join(SATELLITE_DIR, f"{s_under}_{d_under}_rgb.png"),
                os.path.join(SATELLITE_DIR, "real", f"{sk}_{dk}_rgb.png"),
                os.path.join(SATELLITE_DIR, "real", f"{s_lower_under}_{d_lower_under}_rgb.png"),
                os.path.join(SATELLITE_DIR, "real", f"{s_under}_{d_under}_rgb.png"),
            ]
            for p in candidates:
                if os.path.exists(p) and "DEMO" not in os.path.basename(p):
                    return p, False
        return None, False

    demo_candidates = [
        os.path.join(SATELLITE_DIR, "demo", f"district_{area_id:03d}_DEMO.png"),
        os.path.join(SATELLITE_DIR, f"district_{area_id:03d}_DEMO.png"),
    ]
    for p in demo_candidates:
        if os.path.exists(p):
            return p, True

    return None, False


@api_bp.route("/areas", methods=["GET"])
def areas():
    return jsonify(db_service.list_areas())


@api_bp.route("/predict/<int:area_id>", methods=["GET"])
def predict(area_id):
    area = db_service.get_area(area_id)
    if area is None:
        return jsonify({"error": f"area_id {area_id} not found"}), 404

    prediction = db_service.get_latest_prediction(area_id)
    if prediction is None:
        return jsonify({
            "error": f"No precomputed prediction for area_id {area_id}. "
                     f"Run scripts/precompute_predictions.py first."
        }), 404

    sat_path, is_demo = _find_satellite_image(
        area_id, area.get("area_name"), area.get("state_name"), area.get("data_mode", "real")
    )
    has_sat = bool(sat_path is not None)

    return jsonify({
        "area_id": area_id,
        "area_name": area["area_name"],
        "state_name": area["state_name"],
        "data_mode": area["data_mode"],
        "predicted_score": prediction["predicted_score"],
        "predicted_low": prediction.get("predicted_low"),
        "predicted_high": prediction.get("predicted_high"),
        "predicted_category": prediction["predicted_category"],
        "model_name": prediction["model_name"],
        "prediction_timestamp": prediction["prediction_timestamp"],
        "satellite_available": has_sat,
        "satellite_url": f"/api/satellite-image/{area_id}" if has_sat else None,
        "satellite_is_demo": is_demo,
    })


@api_bp.route("/map", methods=["GET"])
def map_data():
    return jsonify(db_service.get_all_areas_with_predictions())


@api_bp.route("/explain/<int:area_id>", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def explain(area_id):
    area = db_service.get_area(area_id)
    if area is None:
        return jsonify({"error": f"area_id {area_id} not found"}), 404

    prediction = db_service.get_latest_prediction(area_id)
    if prediction is None:
        return jsonify({"error": f"No precomputed prediction for area_id {area_id}"}), 404

    explanation = db_service.get_explanation_for_prediction(prediction["prediction_id"])
    if explanation is None:
        return jsonify({"error": f"No explanation found for area_id {area_id}"}), 404

    sat_path, is_demo = _find_satellite_image(
        area_id, area.get("area_name"), area.get("state_name"), area.get("data_mode", "real")
    )
    has_sat = bool(sat_path is not None)

    return jsonify({
        "area_id": area_id,
        "area_name": area["area_name"],
        "predicted_score": prediction["predicted_score"],
        "predicted_category": prediction["predicted_category"],
        "shap_explanation": explanation["shap_json"],
        "gradcam_available": False,
        "gradcam_reason": "The current model uses satellite statistics only, not image features.",
        "gradcam_url": None,
        "satellite_available": has_sat,
        "satellite_url": f"/api/satellite-image/{area_id}" if has_sat else None,
        "satellite_is_demo": is_demo,
    })


@api_bp.route("/satellite-image/<int:area_id>", methods=["GET"])
def satellite_image(area_id):
    area = db_service.get_area(area_id)
    if area is None:
        return jsonify({"error": "District not found"}), 404
    path, is_demo = _find_satellite_image(
        area_id, area.get("area_name"), area.get("state_name"), area.get("data_mode", "real")
    )
    if not path or not os.path.abspath(path).startswith(os.path.abspath(SATELLITE_DIR)):
        return jsonify({"error": "No satellite image available for this district"}), 404
    resp = send_file(path, mimetype="image/png")
    resp.headers["X-Image-Is-Demo"] = "true" if is_demo else "false"
    return resp


@api_bp.route("/gradcam/<int:area_id>", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def gradcam_image(area_id):
    return jsonify({"error": "Grad-CAM not available: The current model uses satellite statistics only, not image features."}), 404


@api_bp.route("/model-performance", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def model_performance():
    return jsonify(db_service.get_model_performance())


@api_bp.route("/scenario", methods=["POST"])
@roles_required("RESEARCHER", "ADMIN")
def scenario():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({
            "error": "Request body must be JSON, e.g. {\"area_id\": 1, \"modified_features\": {\"builtup_pct\": 40}}"
        }), 400

    area_id = body.get("area_id")
    state_name = body.get("state_name")
    cluster_id = body.get("cluster_id")
    group_type = body.get("group_type")
    group_id = body.get("group_id")
    preset_id = body.get("preset_id") or body.get("preset")
    modified_features = body.get("modified_features")

    if area_id is None and not state_name and cluster_id is None and not group_type:
        return jsonify({
            "error": "Request body must contain 'area_id', 'state_name', 'cluster_id', or 'group_type'."
        }), 400

    if modified_features is None and not preset_id:
        return jsonify({
            "error": "Request body must specify either 'modified_features' or 'preset_id'."
        }), 400

    try:
        # 1. Single district scenario
        if area_id is not None:
            aid = int(area_id)
            result = model_service.predict_scenario(aid, modified_features=modified_features, preset_id=preset_id)
            if result is None:
                return jsonify({"error": f"area_id {aid} not found in feature table"}), 404
            result["area_id"] = aid
            return jsonify(result)

        # 2. Group scenario (State / Cluster)
        if state_name:
            result = model_service.predict_group_scenario("state", state_name, modified_features=modified_features, preset_id=preset_id)
        elif cluster_id is not None:
            result = model_service.predict_group_scenario("cluster", cluster_id, modified_features=modified_features, preset_id=preset_id)
        elif group_type:
            result = model_service.predict_group_scenario(group_type, group_id, modified_features=modified_features, preset_id=preset_id)
        else:
            return jsonify({"error": "No valid scenario target provided"}), 400

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Scenario execution failed: {str(e)}"}), 500


@api_bp.route("/scenario-presets", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def scenario_presets():
    """Returns available sensitivity proxy presets."""
    return jsonify(model_service.SCENARIO_PRESETS)


@api_bp.route("/scenario-targets", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def scenario_targets():
    """Returns available targeting options: states, hotspot clusters, and presets."""
    return jsonify(model_service.get_scenario_targets())


@api_bp.route("/indicators/<int:area_id>", methods=["GET"])
def get_indicators(area_id):
    """
    Returns complete real indicators (13 satellite/geospatial, 5 survey) for the district.
    Query param ?full=1 adds all ~104 NFHS indicators.
    """
    from backend.services import indicator_service
    full = request.args.get("full") == "1"
    data = indicator_service.get_district_indicators(area_id, full=full)
    if data is None:
        return jsonify({"error": f"District {area_id} not found"}), 404
    return jsonify(data)


@api_bp.route("/area-features/<int:area_id>", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def area_features(area_id):
    """Returns the current feature vector for an area -- used by the
    what-if frontend page to pre-populate editable sliders/inputs."""
    features = model_service.get_feature_vector(area_id)
    if features is None:
        return jsonify({"error": f"area_id {area_id} not found"}), 404
    return jsonify(features)


@api_bp.route("/compare", methods=["POST"])
@roles_required("RESEARCHER", "ADMIN")
def compare_districts():
    body = request.get_json(silent=True)
    if not body or "area_ids" not in body or not isinstance(body["area_ids"], list):
        return jsonify({"error": "Request body must contain 'area_ids' list"}), 400

    area_ids = body["area_ids"]
    if len(area_ids) < 2 or len(area_ids) > 5:
        return jsonify({"error": "Please select between 2 and 5 districts to compare"}), 400

    results = []
    for area_id in area_ids:
        area = db_service.get_area(area_id)
        if not area:
            continue
        
        prediction = db_service.get_latest_prediction(area_id)
        features = model_service.get_feature_vector(area_id)
        
        results.append({
            "area_id": area_id,
            "area_name": area["area_name"],
            "state_name": area["state_name"],
            "predicted_score": prediction["predicted_score"] if prediction else None,
            "predicted_category": prediction["predicted_category"] if prediction else "N/A",
            "features": features or {}
        })
        
    return jsonify(results)

@api_bp.route("/hotspots", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def hotspots():
    clusters = hotspot_service.identify_hotspots()
    return jsonify(clusters)

@api_bp.route("/data-quality", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def data_quality():
    return jsonify(db_service.get_data_quality_stats())


@api_bp.route("/predict-live", methods=["POST"])
@roles_required("RESEARCHER", "ADMIN")
def predict_live():
    """Takes user-supplied feature values, runs the trained ML model, and returns
    predicted score, category, and live SHAP feature attributions."""
    body = request.get_json(silent=True) or {}
    try:
        result = model_service.predict_custom_features(body)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@api_bp.route("/similar-districts/<int:area_id>", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def similar_districts(area_id):
    """Finds the top 5 most similar districts mathematically based on AI features."""
    try:
        similar = model_service.find_similar_districts(area_id, top_k=5)
        if not similar:
            return jsonify({"error": f"Could not compute similar districts for {area_id}"}), 404
            
        results = []
        for match in similar:
            m_id = match["area_id"]
            area_data = db_service.get_area(m_id)
            pred = db_service.get_latest_prediction(m_id)
            feat = model_service.get_feature_vector(m_id)
            
            results.append({
                "area_id": m_id,
                "area_name": area_data["area_name"] if area_data else f"Unknown ({m_id})",
                "state_name": area_data["state_name"] if area_data else "Unknown",
                "similarity": match["similarity_percentage"],
                "predicted_score": pred["predicted_score"] if pred else None,
                "predicted_category": pred["predicted_category"] if pred else "N/A",
                "features": feat
            })
            
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/residual-atlas", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def residual_atlas():
    """Hidden Gap Detector: districts where survey and satellite-only prediction disagree."""
    try:
        return jsonify(residual_service.get_residual_atlas(
            direction=request.args.get("direction", "all"),
            state=request.args.get("state"),
            limit=request.args.get("limit", 25),
        ))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@api_bp.route("/area-profile/<int:area_id>", methods=["GET"])
def area_profile(area_id):
    """Complete District Indicators profile."""
    try:
        profile = profile_service.get_area_profile(area_id)
        if profile is None:
            return jsonify({"error": f"area_id {area_id} not found"}), 404
        return jsonify(profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
