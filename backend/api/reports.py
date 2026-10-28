"""
reports.py
-----------
Blueprint for all report-related endpoints.

Endpoints:
    GET /api/reports/district/<area_id>        — Download a structured PDF report
    GET /api/reports/district-data/<area_id>   — JSON payload used by the in-browser
                                                 HTML report preview page
"""

import io
import os
import sys
from datetime import datetime

from flask import Blueprint, jsonify, send_file, request, session

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.services import db_service, model_service
from backend.api.auth import roles_required
from config import SATELLITE_DIR

reports_bp = Blueprint("reports", __name__, url_prefix="/api/reports")

# ---------------------------------------------------------------------------
# Deprivation category thresholds (aligned with NFHS-5 training distribution)
# Low  : score < 0.35   — better-off districts
# Medium: 0.35 <= score < 0.60 — moderately deprived
# High : score >= 0.60   — highly deprived
# These thresholds were derived from the empirical tertile distribution of the
# NFHS-5 district-level composite deprivation scores used as the target variable.
# ---------------------------------------------------------------------------
CATEGORY_THRESHOLDS = {"Low": (0.0, 0.35), "Medium": (0.35, 0.60), "High": (0.60, 1.0)}

FEATURE_LABELS = {
    "nightlight_mean": "Nighttime Lights Radiance (VIIRS)",
    "ndvi_mean": "Vegetation Index - NDVI (Sentinel-2)",
    "builtup_pct": "Built-Up Settlement Density (GHSL)",
    "population_density": "Population Density (GHSL, /km2)",
    "road_density_km_per_sqkm": "Road Network Density (OSM, km/km2)",
    "school_distance_km": "Distance to Nearest School (OSM, km)",
    "hospital_distance_km": "Distance to Nearest Hospital (OSM, km)",
}

# SHAP top-feature canonical explanations, ordered by typical importance rank
SHAP_FEATURE_NOTES = {
    "nightlight_mean": "Nighttime radiance is one of the strongest signals of economic activity. Low values strongly push the predicted deprivation score upward.",
    "builtup_pct": "Built-up surface density reflects urbanisation. Sparse settlement patterns are associated with higher predicted deprivation.",
    "ndvi_mean": "High vegetation (NDVI) in a district with low nightlights may indicate rural, agrarian areas, which the model links to elevated deprivation.",
    "population_density": "Population density interacts with infrastructure availability. Sparse populations far from services tend to receive higher predicted scores.",
    "road_density_km_per_sqkm": "Poor road connectivity limits access to markets, healthcare and schools, contributing positively to the deprivation score.",
    "hospital_distance_km": "Greater distance to the nearest hospital is a direct infrastructure-access signal that raises the predicted score.",
    "school_distance_km": "Greater distance to the nearest school signals limited educational access, a key dimension of area-level deprivation.",
}

DISCLAIMER = (
    "Scientific Disclaimer: This report provides an area-level socioeconomic deprivation "
    "estimation based on geospatial remote-sensing features and machine-learning models. "
    "SHAP explanations describe the statistical contributions of features to the model's "
    "prediction; they do not establish causal relationships between those features and "
    "actual deprivation. Ground-truth labels are sourced from NFHS-5 (2019-21). "
    "This system is a research/demonstration prototype and is not validated for operational "
    "welfare-targeting or policy decisions."
)


def _find_satellite_image(area_id: int, data_mode: str = "real"):
    candidates = [
        os.path.join(SATELLITE_DIR, f"district_{area_id:03d}_rgb.png"),
        os.path.join(SATELLITE_DIR, f"district_{area_id:03d}_rgb.tif"),
        os.path.join(SATELLITE_DIR, f"district_{area_id}_rgb.png"),
    ]
    if data_mode == "demo":
        candidates.append(
            os.path.join(SATELLITE_DIR, "demo", f"district_{area_id:03d}_DEMO.png")
        )
    return next((p for p in candidates if os.path.exists(p)), None)


def _score_to_category(score: float) -> str:
    """Return the deprivation category for a given numeric score."""
    if score < 0.35:
        return "Low"
    elif score < 0.60:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# JSON endpoint — supplies data to the in-browser HTML report preview
# ---------------------------------------------------------------------------
@reports_bp.route("/district-data/<int:area_id>", methods=["GET"])
def district_report_data(area_id):
    """
    Returns all report data as JSON.
    Used by the /report/<area_id> frontend page to render an in-browser report.
    No authentication required so that the report page is accessible to all users.
    """
    area = db_service.get_area(area_id)
    if not area:
        return jsonify({"error": "Area not found"}), 404

    prediction = db_service.get_latest_prediction(area_id)
    if not prediction:
        return jsonify({"error": "No prediction found for this district. Please run precompute_predictions.py first."}), 404

    features = model_service.get_feature_vector(area_id)

    # Model performance metrics
    model_name = prediction.get("model_name", "Unknown")
    perf_rows = db_service.get_model_performance()
    model_perf = next((r for r in perf_rows if r["model_name"] == model_name), None)
    if not model_perf and perf_rows:
        model_perf = perf_rows[0]  # fall back to most recent evaluation

    # Explanation
    explanation = db_service.get_explanation_for_prediction(prediction["prediction_id"])
    has_gradcam = bool(
        explanation and
        explanation.get("gradcam_image_path") and
        os.path.exists(explanation["gradcam_image_path"])
    )
    shap_data = explanation["shap_json"] if explanation and explanation.get("shap_json") else None

    # Satellite image
    sat_path = _find_satellite_image(area_id, area.get("data_mode", "real"))
    has_sat = bool(sat_path)

    # Build clean feature dict for display
    display_features = {}
    if features:
        for k, v in features.items():
            if k not in ("area_id", "data_mode") and v is not None:
                display_features[k] = {
                    "label": FEATURE_LABELS.get(k, k.replace("_", " ").title()),
                    "value": round(float(v), 4)
                }

    # Build SHAP contribution list with explanatory notes
    shap_contributions = []
    if shap_data and "top_contributions" in shap_data:
        for item in shap_data["top_contributions"]:
            feat = item.get("feature", "")
            shap_contributions.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat.replace("_", " ").title()).split(" (")[0],
                "shap_value": round(float(item.get("shap_contribution", 0)), 6),
                "note": SHAP_FEATURE_NOTES.get(feat, ""),
            })

    predicted_score = float(prediction["predicted_score"])
    predicted_category = prediction.get("predicted_category") or _score_to_category(predicted_score)

    return jsonify({
        "area_id": area_id,
        "area_name": area["area_name"],
        "state_name": area["state_name"],
        "analysis_date": datetime.utcnow().strftime("%d %B %Y"),
        "data_mode": area.get("data_mode", "real"),
        "predicted_score": predicted_score,
        "predicted_category": predicted_category,
        "model_name": model_name,
        "prediction_timestamp": prediction.get("prediction_timestamp", "N/A"),
        "model_performance": {
            "model_name": model_perf["model_name"] if model_perf else model_name,
            "mae": round(float(model_perf["mae"]), 4) if model_perf and model_perf.get("mae") is not None else None,
            "rmse": round(float(model_perf["rmse"]), 4) if model_perf and model_perf.get("rmse") is not None else None,
            "r2": round(float(model_perf["r2"]), 4) if model_perf and model_perf.get("r2") is not None else None,
            "trained_on": model_perf.get("trained_on") if model_perf else None,
        },
        "features": display_features,
        "shap_contributions": shap_contributions,
        "shap_base_value": round(float(shap_data["base_value"]), 6) if shap_data and "base_value" in shap_data else None,
        "has_satellite": has_sat,
        "satellite_url": f"/api/satellite-image/{area_id}" if has_sat else None,
        "has_gradcam": has_gradcam,
        "gradcam_url": f"/api/gradcam/{area_id}" if has_gradcam else None,
        "disclaimer": DISCLAIMER,
        "category_thresholds": CATEGORY_THRESHOLDS,
        "data_sources": {
            "satellite": "ESA Sentinel-2 MSI (Bands B4/B3/B2 - RGB, 10 m resolution)",
            "nighttime_light": "NOAA/NASA VIIRS DNB Nighttime Lights",
            "built_up": "European Commission Global Human Settlement Layer (GHSL)",
            "road_osm": "OpenStreetMap (OSM) road network via OSMnx",
            "ground_truth": "National Family Health Survey 5 (NFHS-5), 2019-21",
            "deep_features": "VGG-16 (pre-trained ImageNet) - spatial feature extraction from Sentinel-2 patches",
        },
    })


# ---------------------------------------------------------------------------
# PDF download endpoint
# ---------------------------------------------------------------------------
@reports_bp.route("/district/<int:area_id>", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
def generate_district_report_pdf(area_id):
    """
    Generates and streams a structured PDF report for the specified district.
    Contains all 7 sections described in the GeoPoverty AI final report spec.
    Restricted to Researcher and Admin roles.
    """
    area = db_service.get_area(area_id)
    if not area:
        return jsonify({"error": "Area not found"}), 404

    prediction = db_service.get_latest_prediction(area_id)
    if not prediction:
        return jsonify({"error": "No prediction found for this district."}), 404

    features = model_service.get_feature_vector(area_id)
    model_name = prediction.get("model_name", "Unknown")

    perf_rows = db_service.get_model_performance()
    model_perf = next((r for r in perf_rows if r["model_name"] == model_name), None)
    if not model_perf and perf_rows:
        model_perf = perf_rows[0]

    explanation = db_service.get_explanation_for_prediction(prediction["prediction_id"])
    has_gradcam = bool(
        explanation and
        explanation.get("gradcam_image_path") and
        os.path.exists(explanation["gradcam_image_path"])
    )
    shap_data = explanation["shap_json"] if explanation and explanation.get("shap_json") else None
    sat_path = _find_satellite_image(area_id, area.get("data_mode", "real"))

    predicted_score = float(prediction["predicted_score"])
    predicted_category = prediction.get("predicted_category") or _score_to_category(predicted_score)

    # ---- ReportLab imports ------------------------------------------------
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    pdf_buffer = io.BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 2 * cm

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"GeoPoverty AI - District Report: {area['area_name']}",
        author="GeoPoverty AI System",
    )

    base_styles = getSampleStyleSheet()

    def _style(name, **kwargs):
        s = base_styles[name].clone(name + "_custom_" + str(id(kwargs)))
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    S_TITLE = _style("Heading1", fontSize=18, spaceAfter=4, textColor=colors.HexColor("#1b4332"))
    S_SUBTITLE = _style("Normal", fontSize=10, spaceAfter=14, textColor=colors.HexColor("#6b6b6b"))
    S_SECTION = _style("Heading2", fontSize=13, spaceBefore=14, spaceAfter=6,
                       textColor=colors.HexColor("#1b4332"))
    S_BODY = _style("Normal", fontSize=9.5, leading=14, spaceAfter=4)
    S_BODY_J = _style("Normal", fontSize=9.5, leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
    S_SMALL = _style("Normal", fontSize=8, leading=12, textColor=colors.HexColor("#6b6b6b"))
    S_DISCLAIMER = _style("Normal", fontSize=7.5, leading=11,
                          textColor=colors.HexColor("#6b5a1f"), alignment=TA_JUSTIFY)
    S_SCORE_BIG = _style("Normal", fontSize=28, fontName="Helvetica-Bold",
                         textColor=colors.HexColor("#1b4332"), alignment=TA_CENTER)
    S_LABEL = _style("Normal", fontSize=9, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#2b2b2b"))
    S_FOOTER = _style("Normal", fontSize=7.5, textColor=colors.HexColor("#aaaaaa"),
                      alignment=TA_CENTER)

    CAT_COLORS = {
        "Low": colors.HexColor("#52b788"),
        "Medium": colors.HexColor("#f4a261"),
        "High": colors.HexColor("#e76f51"),
    }
    cat_color = CAT_COLORS.get(predicted_category, colors.grey)

    analysis_date = datetime.utcnow().strftime("%d %B %Y")
    story = []

    # ------- Cover / Title block -------------------------------------------
    story.append(Paragraph("GeoPoverty AI", S_TITLE))
    story.append(Paragraph("Explainable AI-Based District Deprivation Analysis Report", S_SUBTITLE))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1b4332"), spaceAfter=10))

    # ------- Section 1 - District Information --------------------------------
    story.append(Paragraph("1. District Information", S_SECTION))
    info_data = [
        ["District", area["area_name"]],
        ["State", area["state_name"]],
        ["Analysis Date", analysis_date],
        ["Prediction Model", model_name],
        ["Satellite Image Source", "ESA Sentinel-2 MSI (RGB - Bands B4/B3/B2, 10 m)"],
        ["Ground Truth Source", "National Family Health Survey 5 (NFHS-5), 2019-21"],
        ["Additional Features", "VIIRS Night-Time Lights + GHSL Built-Up/Population + OSM Roads/POI"],
        ["Deep Feature Extractor", "VGG-16 (pre-trained on ImageNet) applied to Sentinel-2 patches"],
        ["Data Mode", area.get("data_mode", "real").upper()],
    ]
    info_table = Table(info_data, colWidths=[5 * cm, PAGE_W - 2 * MARGIN - 5 * cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f7f6f3"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e6e4de")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10))

    # ------- Section 2 - Predicted Deprivation Score -------------------------
    story.append(Paragraph("2. Predicted Deprivation Score", S_SECTION))
    score_data = [[
        Paragraph("Predicted Deprivation Score", S_LABEL),
        Paragraph(f"{predicted_score:.4f}", S_SCORE_BIG),
        Paragraph(
            f"Category: {predicted_category}",
            _style("Normal", fontSize=14, fontName="Helvetica-Bold",
                   textColor=cat_color, alignment=TA_CENTER)
        ),
    ]]
    score_table = Table(score_data, colWidths=[5 * cm, 5.5 * cm, 5.5 * cm])
    score_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e6e4de")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f6f3")),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Category Thresholds</b> (derived from NFHS-5 empirical tertile distribution): "
        "Low &lt; 0.35 | Medium 0.35-0.60 | High &gt;= 0.60    "
        "<i>Scale: 0.0 (Least Deprived) to 1.0 (Most Deprived)</i>",
        S_SMALL
    ))
    story.append(Spacer(1, 10))

    # ------- Section 3 - Model Performance ----------------------------------
    story.append(Paragraph("3. Model Performance Metrics", S_SECTION))
    story.append(Paragraph(
        "The following metrics were computed on the held-out test split during model evaluation. "
        "They reflect the model's generalisation to unseen districts.",
        S_BODY_J
    ))
    if model_perf:
        mae_str = f"{model_perf['mae']:.4f}" if model_perf.get("mae") is not None else "N/A"
        rmse_str = f"{model_perf['rmse']:.4f}" if model_perf.get("rmse") is not None else "N/A"
        r2_str = f"{model_perf['r2']:.4f}" if model_perf.get("r2") is not None else "N/A"
        trained_on = model_perf.get("trained_on") or "N/A"
        perf_data = [
            ["Metric", "Value", "Interpretation"],
            ["Model Algorithm", model_perf["model_name"], "Ensemble / gradient-boosted tree"],
            ["MAE (Mean Absolute Error)", mae_str, "Average absolute error on deprivation score (0-1)"],
            ["RMSE (Root Mean Squared Error)", rmse_str, "Penalises large errors more strongly than MAE"],
            ["R2 (Coefficient of Determination)", r2_str, "Fraction of variance in scores explained by model"],
            ["Training Dataset", trained_on, "Evaluated on held-out test split"],
        ]
        perf_table = Table(
            perf_data,
            colWidths=[5.5 * cm, 3 * cm, PAGE_W - 2 * MARGIN - 8.5 * cm]
        )
        perf_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b4332")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f6f3"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e6e4de")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(perf_table)
    else:
        story.append(Paragraph(
            "<i>Model performance metrics not yet recorded in the database.</i>", S_BODY))
    story.append(Spacer(1, 10))

    # ------- Section 4 - Satellite Image ------------------------------------
    story.append(Paragraph("4. Sentinel-2 Satellite Imagery", S_SECTION))
    story.append(Paragraph(
        "The following RGB composite was acquired from ESA Sentinel-2 MSI (Bands B4/B3/B2). "
        "A cloud-free patch centred on the district centroid was processed at 10 m resolution. "
        "VGG-16 spatial features were extracted from this patch prior to fusion with geospatial indicators.",
        S_BODY_J
    ))
    if sat_path:
        try:
            MAX_IMG_W = PAGE_W - 2 * MARGIN - 2 * cm
            img = RLImage(sat_path, width=MAX_IMG_W, height=MAX_IMG_W * 0.55)
            story.append(img)
            story.append(Paragraph(
                f"<i>Figure 1: ESA Sentinel-2 RGB composite patch for {area['area_name']}, "
                f"{area['state_name']}. Bands: B4 (Red), B3 (Green), B2 (Blue). "
                "Source: Google Earth Engine / ESA Copernicus.</i>",
                S_SMALL
            ))
        except Exception:
            story.append(Paragraph(
                "<i>Satellite image file found but could not be embedded. "
                "Please verify the image format.</i>", S_BODY
            ))
    else:
        story.append(Paragraph(
            "<i>No Sentinel-2 patch is currently cached for this district. "
            "Run python geospatial/download_satellite_patches.py to acquire imagery.</i>",
            S_BODY
        ))
    story.append(Spacer(1, 10))

    # ------- Section 5 - Geospatial Indicators ------------------------------
    story.append(Paragraph("5. Geospatial Indicator Values", S_SECTION))
    story.append(Paragraph(
        "The following indicator values were used as input features to the machine-learning model "
        "for this district. Values are derived from VIIRS, GHSL, and OSM data sources.",
        S_BODY_J
    ))
    if features:
        feat_rows = [["Indicator", "Value", "Data Source"]]
        for k, v in features.items():
            if k not in ("area_id", "data_mode") and v is not None:
                label = FEATURE_LABELS.get(k, k.replace("_", " ").title())
                parts = label.split(" (", 1)
                clean_label = parts[0]
                source = parts[1].rstrip(")") if len(parts) > 1 else "-"
                feat_rows.append([clean_label, f"{float(v):.4f}", source])
        feat_table = Table(
            feat_rows,
            colWidths=[5.5 * cm, 2.5 * cm, PAGE_W - 2 * MARGIN - 8 * cm]
        )
        feat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d6a4f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f6f3"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e6e4de")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(feat_table)
    else:
        story.append(Paragraph(
            "<i>No feature data available for this district.</i>", S_BODY))
    story.append(Spacer(1, 10))

    # ------- Section 6 - SHAP Explanation -----------------------------------
    story.append(Paragraph("6. SHAP Model Explanation", S_SECTION))
    story.append(Paragraph(
        "SHAP (SHapley Additive exPlanations) values quantify how much each input feature "
        "<b>influenced the model's prediction</b> for this specific district. "
        "A positive SHAP value means the feature pushed the predicted deprivation score "
        "<b>upward</b>; a negative value means it pushed the score <b>downward</b>. "
        "These are statistical attributions, not direct causal evidence.",
        S_BODY_J
    ))
    if shap_data and "top_contributions" in shap_data:
        base_val = shap_data.get("base_value", None)
        if base_val is not None:
            story.append(Paragraph(
                f"<b>Model baseline (expected value across all training districts):</b> "
                f"{float(base_val):.4f}",
                S_BODY
            ))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "<i>According to the model explanation, these features had the strongest "
            "influence on the predicted deprivation score for this district.</i>",
            S_SMALL
        ))
        story.append(Spacer(1, 6))

        shap_rows = [["Rank", "Feature", "SHAP Value", "Direction", "Interpretation"]]
        for i, item in enumerate(shap_data["top_contributions"], start=1):
            feat = item.get("feature", "")
            sv = float(item.get("shap_contribution", 0))
            direction = "Up (increases score)" if sv >= 0 else "Down (decreases score)"
            note = SHAP_FEATURE_NOTES.get(feat, "-")
            label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title()).split(" (")[0]
            shap_rows.append([str(i), label, f"{sv:+.6f}", direction, note])

        shap_table = Table(
            shap_rows,
            colWidths=[1 * cm, 4.5 * cm, 2.2 * cm, 2.8 * cm,
                       PAGE_W - 2 * MARGIN - 10.5 * cm],
        )
        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b4332")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f6f3"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e6e4de")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(shap_table)
    else:
        story.append(Paragraph(
            "<i>SHAP explanation data is not available for this district.</i>", S_BODY))
    story.append(Spacer(1, 10))

    # ------- Section 7 - Grad-CAM Explanation --------------------------------
    story.append(Paragraph("7. Grad-CAM Visual Attention Map", S_SECTION))
    story.append(Paragraph(
        "Grad-CAM (Gradient-weighted Class Activation Mapping) generates a saliency heatmap "
        "that highlights which spatial regions of the Sentinel-2 satellite image most strongly "
        "activated the VGG-16 visual feature extractor used in this pipeline. "
        "Warmer (red/yellow) regions indicate areas that contributed most to the extracted "
        "visual representation. This does not directly indicate poverty, but shows which "
        "visual patterns drove the CNN's output features.",
        S_BODY_J
    ))
    if has_gradcam:
        try:
            MAX_GC_W = PAGE_W - 2 * MARGIN - 2 * cm
            gc_img = RLImage(
                explanation["gradcam_image_path"],
                width=MAX_GC_W,
                height=MAX_GC_W * 0.55
            )
            story.append(gc_img)
            story.append(Paragraph(
                f"<i>Figure 2: Grad-CAM heatmap overlay for {area['area_name']}. "
                "Red/yellow regions had the highest activation in the VGG-16 feature layer. "
                "Generated using the final convolutional layer of the VGG-16 network.</i>",
                S_SMALL
            ))
        except Exception:
            story.append(Paragraph(
                "<i>Grad-CAM image found but could not be embedded. "
                "Please check the image format.</i>",
                S_BODY
            ))
    else:
        story.append(Paragraph(
            "<i>No Grad-CAM heatmap is currently available for this district. "
            "Run the explainability module to generate Grad-CAM outputs.</i>",
            S_BODY
        ))
    story.append(Spacer(1, 16))

    # ------- Disclaimer footer ----------------------------------------------
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor("#e6e4de"), spaceAfter=8
    ))
    story.append(Paragraph("<b>Scientific Disclaimer</b>", S_LABEL))
    story.append(Paragraph(DISCLAIMER, S_DISCLAIMER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated by GeoPoverty AI  |  {analysis_date}  |  MCA Final Year Project Prototype",
        S_FOOTER
    ))

    # ------- Build & stream -------------------------------------------------
    doc.build(story)

    # Audit log (best-effort - works when user is logged in)
    if "user_id" in session:
        try:
            db_service.log_audit_action(
                session["user_id"], "REPORT_DOWNLOAD",
                f"PDF report downloaded for area_id {area_id} ({area['area_name']})"
            )
        except Exception:
            pass

    pdf_buffer.seek(0)
    safe_name = area["area_name"].replace(" ", "_").replace("/", "-")
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"GeoPoverty_Report_{safe_name}_{analysis_date.replace(' ', '_')}.pdf",
        mimetype="application/pdf",
    )
