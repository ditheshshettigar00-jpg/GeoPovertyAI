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
import json
import logging
import os
import sys
from datetime import datetime, timezone

from flask import Blueprint, jsonify, send_file, request, session

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.services import db_service, model_service, indicator_service
from backend.api.auth import roles_required
from config import SATELLITE_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

reports_bp = Blueprint("reports", __name__, url_prefix="/api/reports")


def _load_category_thresholds():
    """
    Load dynamic tertile category thresholds from models/category_thresholds.json.
    Falls back to (0.35, 0.60) only if the file is missing or invalid, logging a warning.
    """
    path = os.path.join(MODELS_DIR, "category_thresholds.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data["low_medium_boundary"]), float(data["medium_high_boundary"])
        except Exception as exc:
            logger.warning("Could not read category thresholds from %s: %s", path, exc)
    else:
        logger.warning("category_thresholds.json not found at %s; falling back to (0.35, 0.60)", path)
    return 0.35, 0.60


_low_med_thresh, _med_high_thresh = _load_category_thresholds()
CATEGORY_THRESHOLDS = {
    "Low": (0.0, _low_med_thresh),
    "Medium": (_low_med_thresh, _med_high_thresh),
    "High": (_med_high_thresh, 1.0),
    "low_medium_boundary": _low_med_thresh,
    "medium_high_boundary": _med_high_thresh,
}


def _load_model_comparison_metrics():
    """
    Load model performance metrics evaluated via leave-one-state-out cross-validation
    from models/model_comparison_results.csv.
    """
    path = os.path.join(MODELS_DIR, "model_comparison_results.csv")
    metrics = {
        "model_name": "XGBoost",
        "r2": 0.4884,
        "mae": 0.0752,
        "rmse": 0.0954,
        "cat_accuracy": 0.6117,
        "baseline_mae": 0.1142,
        "validation_scheme": "leave-one-state-out validation: each state predicted by a model that never saw it",
    }
    meta_path = os.path.join(MODELS_DIR, "run_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if meta.get("best_model"):
                metrics["model_name"] = meta["best_model"]
        except Exception:
            pass
    if os.path.exists(path):
        try:
            import pandas as pd
            df = pd.read_csv(path)
            best_row = df[df["model"] == metrics["model_name"]]
            if best_row.empty:
                best_row = df[df["model"] == "XGBoost"]
            dummy_row = df[df["model"] == "Dummy"]
            if not best_row.empty:
                r = best_row.iloc[0]
                metrics["model_name"] = str(r["model"])
                metrics["r2"] = round(float(r["r2"]), 4)
                metrics["mae"] = round(float(r["mae"]), 4)
                metrics["rmse"] = round(float(r["rmse"]), 4)
                if "cat_accuracy" in r and pd.notna(r["cat_accuracy"]):
                    metrics["cat_accuracy"] = round(float(r["cat_accuracy"]), 4)
            if not dummy_row.empty:
                d = dummy_row.iloc[0]
                metrics["baseline_mae"] = round(float(d["mae"]), 4)
        except Exception as exc:
            logger.warning("Could not read model comparison metrics from %s: %s", path, exc)
    return metrics

FEATURE_LABELS = {
    "nightlight_mean": "Nighttime Light Radiance (VIIRS, nW/cm2/sr)",
    "ndvi_mean": "Normalized Difference Vegetation Index (NDVI, Sentinel-2)",
    "builtup_pct": "Built-Up Surface Area (GHSL, % of land area)",
    "population_density": "Population Density (GHSL, people per km2)",
    "elevation_mean_m": "Mean Elevation (SRTM, m)",
    "slope_mean_deg": "Mean Terrain Slope (SRTM, deg)",
    "rainfall_mm_year": "Annual Rainfall (CHIRPS, mm/yr)",
    "lst_day_c": "Daytime Land Surface Temperature (MODIS, deg C)",
    "cropland_pct": "Cropland Share (ESA WorldCover, %)",
    "travel_time_city_min": "Travel Time to Nearest City (Oxford/MAP 2015, min)",
}

# SHAP top-feature canonical explanations, ordered by typical importance rank
SHAP_FEATURE_NOTES = {
    "lst_day_c": "Daytime surface temperature reflects aridity and agro-climatic regimes linked with rural livelihood conditions.",
    "elevation_mean_m": "Terrain elevation separates high-altitude plateau/mountain geographies from lowland plains.",
    "builtup_pct": "Built-up surface density reflects urbanisation. Sparse settlement patterns are associated with higher predicted deprivation.",
    "ndvi_mean": "Vegetation index (NDVI) in low-nightlight areas often correlates with agrarian land cover and deprivation.",
    "nightlight_mean": "Nighttime radiance tracks electrification and economic activity; lower radiance typically pushes predicted deprivation upward.",
    "population_density": "Population density captures settlement concentration and infrastructure proximity.",
    "slope_mean_deg": "Terrain slope captures hilly topography where physical connectivity can be constrained.",
    "rainfall_mm_year": "Annual precipitation indexes agro-climatic regime and water availability.",
    "cropland_pct": "Cropland share measures agricultural land dominance versus urban or forest cover.",
    "travel_time_city_min": "Surface travel time (2015 baseline) reflects spatial remoteness from major urban centers.",
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


def _find_satellite_image(area_id: int):
    candidates = [
        os.path.join(SATELLITE_DIR, f"district_{area_id:03d}_rgb.png"),
        os.path.join(SATELLITE_DIR, f"district_{area_id:03d}_rgb.tif"),
        os.path.join(SATELLITE_DIR, f"district_{area_id}_rgb.png"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def _score_to_category(score: float) -> str:
    """Return the deprivation category for a given numeric score."""
    low_med, med_high = _load_category_thresholds()
    if score <= low_med:
        return "Low"
    elif score <= med_high:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# JSON endpoint — supplies data to the in-browser HTML report preview
# ---------------------------------------------------------------------------
@reports_bp.route("/district-data/<int:area_id>", methods=["GET"])
@roles_required("RESEARCHER", "ADMIN")
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
    sat_path = _find_satellite_image(area_id)
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

    comp_metrics = _load_model_comparison_metrics()
    low_med, med_high = _load_category_thresholds()

    return jsonify({
        "area_id": area_id,
        "area_name": area["area_name"],
        "state_name": area["state_name"],
        "analysis_date": datetime.now(timezone.utc).strftime("%d %B %Y") if hasattr(timezone, "utc") else datetime.utcnow().strftime("%d %B %Y"),
        "data_mode": area.get("data_mode", "real"),
        "predicted_score": predicted_score,
        "predicted_category": predicted_category,
        "model_name": model_name,
        "prediction_timestamp": prediction.get("prediction_timestamp", "N/A"),
        "model_performance": {
            "model_name": comp_metrics["model_name"],
            "mae": comp_metrics["mae"],
            "rmse": comp_metrics["rmse"],
            "r2": comp_metrics["r2"],
            "cat_accuracy": comp_metrics.get("cat_accuracy"),
            "baseline_mae": comp_metrics.get("baseline_mae"),
            "validation_scheme": comp_metrics["validation_scheme"],
            "trained_on": comp_metrics["validation_scheme"],
        },
        "features": display_features,
        "shap_contributions": shap_contributions,
        "shap_base_value": round(float(shap_data["base_value"]), 6) if shap_data and "base_value" in shap_data else None,
        "has_satellite": has_sat,
        "satellite_url": f"/api/satellite-image/{area_id}" if has_sat else None,
        "has_gradcam": has_gradcam,
        "gradcam_url": f"/api/gradcam/{area_id}" if has_gradcam else None,
        "disclaimer": DISCLAIMER,
        "category_thresholds": {
            "low_medium_boundary": low_med,
            "medium_high_boundary": med_high,
            "Low": (0.0, low_med),
            "Medium": (low_med, med_high),
            "High": (med_high, 1.0),
        },
        "satellite_indicators": indicator_service.get_satellite_indicators(area_id),
        "survey_indicators": indicator_service.get_survey_indicators(area_id),
        "data_sources": {
            "satellite": "true-colour Sentinel-2 composite, 2019-11-01 to 2021-04-30, for context only; the model uses tabular statistics, not image pixels",
            "nighttime_light": "NOAA/NASA VIIRS DNB Nighttime Lights",
            "built_up": "European Commission Global Human Settlement Layer (GHSL)",
            "climate_terrain": "NASA SRTM DEM, UCSB CHIRPS, MODIS MOD11A2, ESA WorldCover 2020, Oxford/MAP 2015 Travel Time",
            "ground_truth": "National Family Health Survey 5 (NFHS-5), 2019-21",
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
    sat_path = _find_satellite_image(area_id)

    predicted_score = float(prediction["predicted_score"])
    predicted_category = prediction.get("predicted_category") or _score_to_category(predicted_score)

    # ---- ReportLab imports ------------------------------------------------
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, PageBreak,
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
        pageCompression=0,
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

    analysis_date = datetime.now(timezone.utc).strftime("%d %B %Y") if hasattr(timezone, "utc") else datetime.utcnow().strftime("%d %B %Y")
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
        ["Satellite Image Source", "true-colour Sentinel-2 composite, 2019-11-01 to 2021-04-30, for context only; the model uses satellite statistics, not image pixels"],
        ["Ground Truth Source", "National Family Health Survey 5 (NFHS-5), 2019-21"],
        ["Satellite Telemetry Features", "VIIRS, Sentinel-2 NDVI, GHSL, SRTM, CHIRPS, MODIS LST, WorldCover, MAP 2015"],
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
    low_med, med_high = _load_category_thresholds()
    story.append(Paragraph(
        f"<b>Category Thresholds</b> (derived as tertiles of the NFHS-5 deprivation scores of the 564 districts used for training): "
        f"Low &lt;= {low_med:.4f} | Medium {low_med:.4f}–{med_high:.4f} | High &gt; {med_high:.4f}    "
        f"<i>Scale: 0.0 (Least Deprived) to 1.0 (Most Deprived)</i>",
        S_SMALL
    ))
    story.append(Spacer(1, 10))

    # ------- Section 3 - Model Performance ----------------------------------
    story.append(Paragraph("3. Model Performance Metrics", S_SECTION))
    story.append(Paragraph(
        "The following metrics were computed via leave-one-state-out validation: each state predicted by a model that never saw it. "
        "They reflect the model's generalisation to unseen states.",
        S_BODY_J
    ))
    comp_metrics = _load_model_comparison_metrics()
    perf_data = [
        ["Metric", "Value", "Interpretation"],
        ["Model Algorithm", comp_metrics["model_name"], "Ensemble / gradient-boosted tree"],
        ["R2 (Coefficient of Determination)", f"{comp_metrics['r2']:.4f}", "Fraction of variance in scores explained by model"],
        ["MAE (Mean Absolute Error)", f"{comp_metrics['mae']:.4f}", "Average absolute error on deprivation score (0-1)"],
        ["RMSE (Root Mean Squared Error)", f"{comp_metrics['rmse']:.4f}", "Penalises large errors more strongly than MAE"],
        ["3-Class Accuracy", f"{comp_metrics['cat_accuracy'] * 100:.2f}%" if comp_metrics.get("cat_accuracy") is not None else "N/A", "Accuracy classifying districts into Low/Medium/High"],
        ["Baseline MAE (Dummy Model)", f"{comp_metrics['baseline_mae']:.4f}" if comp_metrics.get("baseline_mae") is not None else "N/A", "Mean baseline predicting the training mean of other states"],
        ["Validation Scheme", "Leave-One-State-Out", comp_metrics["validation_scheme"]],
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
    story.append(Spacer(1, 10))

    # ------- Section 4 - Satellite Image ------------------------------------
    story.append(Paragraph("4. Sentinel-2 Satellite Imagery", S_SECTION))
    story.append(Paragraph(
        "The following image is a true-colour Sentinel-2 composite, 2019-11-01 to 2021-04-30, for context only; the model uses satellite statistics, not image pixels.",
        S_BODY_J
    ))
    if sat_path:
        try:
            MAX_IMG_W = PAGE_W - 2 * MARGIN - 2 * cm
            img = RLImage(sat_path, width=MAX_IMG_W, height=MAX_IMG_W * 0.55)
            story.append(img)
            story.append(Paragraph(
                f"<i>Figure 1: True-colour Sentinel-2 composite, 2019-11-01 to 2021-04-30, for context only; the model uses satellite statistics, not image pixels "
                f"({area['area_name']}, {area['state_name']}).</i>",
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

    # ------- Section 5 - Indicator Values (5A Satellite, 5B Survey) ----------
    story.append(Paragraph("5. Geospatial & Socioeconomic Indicator Values", S_SECTION))
    story.append(Paragraph(
        "This section details both the satellite-derived remote sensing telemetry used as inputs "
        "to the machine learning model and the ground-truth survey indicators from NFHS-5 that form the deprivation score.",
        S_BODY_J
    ))
    story.append(Spacer(1, 6))

    # 5A Satellite-derived indicators
    story.append(Paragraph("<b>5A. Satellite-Derived Indicators</b>", S_SUBTITLE))
    story.append(Spacer(1, 4))

    sat_indicators = indicator_service.get_satellite_indicators(area_id)
    if sat_indicators:
        sat_rows = [["Indicator", "Value", "Unit", "Percentile of 564", "State Median", "India Median", "Source"]]
        for s in sat_indicators:
            val_str = s.get("formatted_value", str(s["value"])) if s["value"] is not None else "-"
            unit_str = s.get("unit", "")
            pct_str = f"{s['percentile']:.1f}%" if s.get("percentile") is not None else "-"
            st_med_str = str(s["state_median"]) if s.get("state_median") is not None else "-"
            nat_med_str = str(s["national_median"]) if s.get("national_median") is not None else "-"
            source_str = s.get("source", "-")
            sat_rows.append([
                Paragraph(s["label"], S_BODY),
                Paragraph(val_str, S_BODY),
                Paragraph(unit_str, S_BODY),
                Paragraph(pct_str, S_BODY),
                Paragraph(st_med_str, S_BODY),
                Paragraph(nat_med_str, S_BODY),
                Paragraph(source_str, S_BODY),
            ])
        sat_table = Table(
            sat_rows,
            colWidths=[3.8 * cm, 1.8 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 3.6 * cm]
        )
        sat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d6a4f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f6f3"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e6e4de")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(sat_table)
    else:
        story.append(Paragraph("<i>No satellite telemetry data available for this district.</i>", S_BODY))

    story.append(Spacer(1, 10))

    # 5B Survey indicators
    story.append(Paragraph("<b>5B. Survey Indicators (NFHS-5) that Form the Deprivation Score</b>", S_SUBTITLE))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Electricity, sanitation and clean cooking fuel enter the score as 100 minus the value; stunting and anaemia enter directly.",
        S_BODY_J
    ))
    story.append(Spacer(1, 4))

    surv_indicators = indicator_service.get_survey_indicators(area_id)
    if surv_indicators:
        surv_rows = [["Indicator", "NFHS-5", "NFHS-4", "Change", "State Median", "India Median", "Note"]]
        for sv in surv_indicators:
            lbl = sv["label"]
            v5_str = f"{sv['nfhs5']:.1f}%" if sv["nfhs5"] is not None else "Not shown"
            v4_str = f"{sv['nfhs4']:.1f}%" if sv["nfhs4"] is not None else "-"
            chg_str = f"{'+' if (sv['change'] or 0) > 0 else ''}{sv['change']:.1f}%" if sv["change"] is not None else "-"
            st_med = f"{sv['state_median']:.1f}%" if sv.get("state_median") is not None else "-"
            nat_med = f"{sv['national_median']:.1f}%" if sv.get("national_median") is not None else "-"
            note = sv.get("flag_text") or "-"
            surv_rows.append([
                Paragraph(lbl, S_BODY),
                Paragraph(v5_str, S_BODY),
                Paragraph(v4_str, S_BODY),
                Paragraph(chg_str, S_BODY),
                Paragraph(st_med, S_BODY),
                Paragraph(nat_med, S_BODY),
                Paragraph(note, S_BODY),
            ])
        surv_table = Table(
            surv_rows,
            colWidths=[5.2 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm, 3.4 * cm]
        )
        surv_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d6a4f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f6f3"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e6e4de")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(surv_table)
    else:
        story.append(Paragraph("<i>No survey indicators available for this district.</i>", S_BODY))
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
        "activated the convolutional visual feature extractor. "
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
                "Red/yellow regions had the highest activation in the convolutional feature layer. "
                "Generated using the final convolutional layer of the network.</i>",
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
            "<i>Grad-CAM is not available because the production model uses tabular satellite and geospatial statistics only (not image features).</i>",
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

    # ------- Optional Appendix: All NFHS-5 Indicators -----------------------
    if request.args.get("appendix") == "1":
        story.append(PageBreak())
        story.append(Paragraph("Appendix: All NFHS-5 Indicators for this District", S_SECTION))
        story.append(Paragraph(
            f"Complete survey indicators for {area['area_name']}, {area['state_name']} "
            "from the National Family Health Survey (NFHS-5, 2019-21) and NFHS-4 (2015-16).",
            S_BODY_J
        ))
        story.append(Spacer(1, 6))
        all_indicators = indicator_service.get_all_survey_indicators(area_id)
        if all_indicators:
            app_rows = [["#", "Indicator", "NFHS-5", "NFHS-4", "Change", "Note / Flag"]]
            for ind in all_indicators:
                v5 = f"{ind['nfhs5']:.1f}" if ind["nfhs5"] is not None else "-"
                v4 = f"{ind['nfhs4']:.1f}" if ind["nfhs4"] is not None else "-"
                chg = f"{'+' if (ind['change'] or 0) > 0 else ''}{ind['change']:.1f}" if ind["change"] is not None else "-"
                fl = ind.get("flag_text") or "-"
                app_rows.append([
                    Paragraph(str(ind["indicator_number"]), S_BODY),
                    Paragraph(ind["label"], S_BODY),
                    Paragraph(v5, S_BODY),
                    Paragraph(v4, S_BODY),
                    Paragraph(chg, S_BODY),
                    Paragraph(fl, S_BODY),
                ])
            app_table = Table(
                app_rows,
                colWidths=[1.0 * cm, 8.5 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.5 * cm]
            )
            app_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d6a4f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f6f3"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e6e4de")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(app_table)

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
