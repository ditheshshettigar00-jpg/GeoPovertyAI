"""
test_phase3.py
--------------
Tests for Phase 3:
1. For every district, base_value + sum(shap_values) is within 1e-4 of predicted_score.
2. /api/explain returns gradcam_available false with reason.
3. Satellite-image endpoint does not return a demo image for a real district.
4. Grad-CAM PNG outputs are removed.
5. explainability/gradcam.py docstring notes it is for image-based models only.
6. No TensorFlow is imported in app or precompute script.
"""

import glob
import json
import os
import sqlite3
import pytest

from config import DB_PATH


def test_shap_base_plus_sum_matches_predicted_score_for_all_districts():
    """For every district in the database, base_value + sum(shap_values)
    must equal the stored predicted_score within 1e-4."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT p.area_id, p.predicted_score, e.shap_json
        FROM predictions p
        JOIN explanations e ON e.prediction_id = p.prediction_id
    """).fetchall()
    conn.close()

    assert len(rows) > 0, "No predictions found in database"

    mismatches = []
    for area_id, predicted_score, shap_json_str in rows:
        data = json.loads(shap_json_str)
        base_value = data["base_value"]
        contributions = data.get("top_contributions", [])
        shap_sum = sum(c["shap_contribution"] for c in contributions)
        calculated = base_value + shap_sum
        diff = abs(calculated - predicted_score)
        if diff > 1e-4:
            mismatches.append(f"District {area_id}: base ({base_value:.6f}) + sum ({shap_sum:.6f}) = {calculated:.6f} vs score {predicted_score:.6f} (diff={diff:.2e})")

    assert len(mismatches) == 0, f"{len(mismatches)} districts failed additive check:\n" + "\n".join(mismatches[:10])


def test_api_explain_returns_gradcam_false(client):
    """The /api/explain/<id> endpoint must return gradcam_available: false."""
    areas = client.get("/api/areas").get_json()
    assert len(areas) > 0

    for area in areas[:10]:
        aid = area["area_id"]
        res = client.get(f"/api/explain/{aid}")
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("gradcam_available") is False
        assert "gradcam_reason" in data
        assert "satellite statistics only" in data["gradcam_reason"].lower()
        assert data.get("gradcam_url") is None


def test_satellite_image_does_not_return_demo_for_real_district(client):
    """For real districts, the satellite image endpoint must never serve a demo image."""
    areas = client.get("/api/areas").get_json()
    real_areas = [a for a in areas if a.get("data_mode") == "real"]
    assert len(real_areas) > 0

    for area in real_areas[:20]:
        aid = area["area_id"]
        res = client.get(f"/api/satellite-image/{aid}")
        if res.status_code == 200:
            assert res.headers.get("X-Image-Is-Demo") == "false"
        else:
            assert res.status_code == 404
            data = res.get_json()
            assert "error" in data


def test_satellite_image_demo_file_not_served_for_real_district_1(client):
    """Area 1 is a real district. Even though demo/district_001_DEMO.png exists,
    it must never be served for area 1."""
    res = client.get("/api/satellite-image/1")
    if res.status_code == 200:
        assert res.headers.get("X-Image-Is-Demo") == "false"
    else:
        assert res.status_code == 404


def test_no_gradcam_png_files_remain():
    """Verify all old Grad-CAM PNG files were deleted from explainability/outputs/."""
    pngs = glob.glob("explainability/outputs/gradcam_*.png")
    assert len(pngs) == 0, f"Found {len(pngs)} gradcam files still present: {pngs[:5]}"


def test_gradcam_py_docstring_note():
    """Verify explainability/gradcam.py has docstring noting it is for image-based models only."""
    with open("explainability/gradcam.py", encoding="utf-8") as f:
        content = f.read()
    assert "NOTE: This module is for image-based models only" in content


def test_no_tensorflow_imports_in_app_or_precompute():
    """Verify neither backend/app.py nor scripts/precompute_predictions.py imports TensorFlow."""
    for path in ["backend/app.py", "scripts/precompute_predictions.py"]:
        with open(path, encoding="utf-8") as f:
            code = f.read()
        assert "import tensorflow" not in code
        assert "from tensorflow" not in code

