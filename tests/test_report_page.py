"""tests/test_report_page.py
-------------------------
Tests for the district report preview page (/report/<area_id>) and its data endpoint (/api/reports/district-data/<area_id>).
"""
import json
import os
import pytest
from backend.services.db_service import get_connection
from config import MODELS_DIR


def _get_valid_area_id():
    """Retrieve an area_id from the database that has a precomputed prediction."""
    conn = get_connection()
    row = conn.execute(
        "SELECT p.area_id FROM predictions p JOIN areas a ON p.area_id = a.area_id LIMIT 1"
    ).fetchone()
    return int(row[0]) if row else None


def test_report_page_renders_with_server_area_id(client):
    """GET /report/<valid id> returns 200 and the HTML contains the id as a number in the script."""
    area_id = _get_valid_area_id()
    if not area_id:
        pytest.skip("No precomputed district prediction found in the database.")

    resp = client.get(f"/report/{area_id}")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert f"const serverAreaId = {area_id};" in html


def test_report_page_and_endpoint_for_nonexistent_or_no_prediction(client):
    """GET /report/<id> for an id with no prediction still returns 200 (page loads) and

    the JSON endpoint returns 404 with an error message.
    """
    # Non-existent ID: page shell loads with 200
    resp_page = client.get("/report/999999")
    assert resp_page.status_code == 200
    html = resp_page.get_data(as_text=True)
    assert "const serverAreaId = 999999;" in html

    # Corresponding JSON endpoint returns 404 with error message
    resp_json = client.get("/api/reports/district-data/999999")
    assert resp_json.status_code == 404
    data = resp_json.get_json()
    assert "error" in data
    assert len(data["error"]) > 0


def test_report_district_data_endpoint_success(client):
    """Verify that district-data endpoint returns complete structured report data for valid district."""
    area_id = _get_valid_area_id()
    if not area_id:
        pytest.skip("No precomputed district prediction found in the database.")

    resp = client.get(f"/api/reports/district-data/{area_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["area_id"] == area_id
    assert "area_name" in data
    assert "predicted_score" in data
    assert "predicted_category" in data


def test_report_category_thresholds_match_json_and_pdf(client):
    """
    Verify that:
    1. Report JSON category_thresholds match the two values in models/category_thresholds.json.
    2. PDF text contains those dynamic threshold values.
    3. Neither '0.35' nor '0.60' appears anywhere in the generated PDF bytes.
    """
    thresh_path = os.path.join(MODELS_DIR, "category_thresholds.json")
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_data = json.load(f)

    low_med = float(thresh_data["low_medium_boundary"])
    med_high = float(thresh_data["medium_high_boundary"])

    area_id = _get_valid_area_id()
    if not area_id:
        pytest.skip("No precomputed district prediction found in the database.")

    # 1. Check report JSON
    resp_json = client.get(f"/api/reports/district-data/{area_id}")
    assert resp_json.status_code == 200
    data = resp_json.get_json()
    assert "category_thresholds" in data
    cat_thresh = data["category_thresholds"]

    assert cat_thresh["low_medium_boundary"] == thresh_data["low_medium_boundary"]
    assert cat_thresh["medium_high_boundary"] == thresh_data["medium_high_boundary"]
    assert cat_thresh["Low"][1] == thresh_data["low_medium_boundary"]
    assert cat_thresh["Medium"][0] == thresh_data["low_medium_boundary"]
    assert cat_thresh["Medium"][1] == thresh_data["medium_high_boundary"]
    assert cat_thresh["High"][0] == thresh_data["medium_high_boundary"]

    # 2. Check PDF output
    from backend.services import db_service
    existing = db_service.get_user_by_email("pdf_test_admin@example.com")
    if existing:
        uid = existing["user_id"]
    else:
        uid = db_service.create_user("Test Admin", "pdf_test_admin@example.com", "x", role="ADMIN")
    with client.session_transaction() as sess:
        sess["user_id"] = uid
        sess["role"] = "ADMIN"

    resp_pdf = client.get(f"/api/reports/district/{area_id}")
    assert resp_pdf.status_code == 200
    pdf_bytes = resp_pdf.get_data()
    assert pdf_bytes.startswith(b"%PDF")

    low_med_str = f"{low_med:.4f}".encode("latin1")
    med_high_str = f"{med_high:.4f}".encode("latin1")
    assert low_med_str in pdf_bytes, f"Expected {low_med_str} in PDF bytes"
    assert med_high_str in pdf_bytes, f"Expected {med_high_str} in PDF bytes"

    # 3. Ensure legacy hardcoded cutoffs do not appear in the PDF
    assert b"0.35" not in pdf_bytes, "Found legacy '0.35' in PDF text"
    assert b"0.60" not in pdf_bytes, "Found legacy '0.60' in PDF text"

