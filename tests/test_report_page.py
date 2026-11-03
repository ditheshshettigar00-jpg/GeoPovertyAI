"""tests/test_report_page.py
-------------------------
Tests for the district report preview page (/report/<area_id>) and its data endpoint (/api/reports/district-data/<area_id>).
"""
import pytest
from backend.services.db_service import get_connection


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
