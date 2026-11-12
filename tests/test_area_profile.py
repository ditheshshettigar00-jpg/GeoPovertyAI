"""
tests/test_area_profile.py
--------------------------
Tests for the District Indicators Profile endpoint:
    GET /api/area-profile/<area_id>
"""

import json
import os
import pytest
import pandas as pd

from config import PROCESSED_DIR, GROUND_TRUTH_DIR


def _datasets_available():
    final_p = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    nfhs_p = os.path.join(GROUND_TRUTH_DIR, "nfhs5_districts_raw.csv")
    return os.path.exists(final_p) and os.path.exists(nfhs_p)


def test_area_profile_valid_id(client):
    """GET /api/area-profile/<valid id> returns 200 with all top-level keys;

    model_inputs has 10 items; no value anywhere is NaN.
    """
    if not _datasets_available():
        pytest.skip("Required training table or NFHS ground truth CSV not found.")

    res = client.get("/api/area-profile/182")
    assert res.status_code == 200

    raw_text = res.get_data(as_text=True)
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    data = res.get_json()
    expected_keys = {
        "area_id", "area_name", "state_name", "data_mode",
        "model_inputs", "location_quality", "survey_components",
        "accessibility", "all_nfhs5"
    }
    assert expected_keys.issubset(data.keys())

    assert len(data["model_inputs"]) == 10
    input_keys = {item["key"] for item in data["model_inputs"]}
    assert input_keys == {
        "nightlight_mean", "ndvi_mean", "builtup_pct", "population_density",
        "elevation_mean_m", "slope_mean_deg", "rainfall_mm_year",
        "lst_day_c", "cropland_pct", "travel_time_city_min",
    }
    for item in data["model_inputs"]:
        assert "key" in item
        assert "label" in item
        assert "value" in item
        assert "unit" in item
        assert "description" in item
        assert "percentile" in item
        assert "dataset_min" in item
        assert "dataset_median" in item
        assert "dataset_max" in item


def test_district_without_and_with_accessibility_data(client):
    """A district without accessibility data returns accessibility.available == false

    and no numeric zeros in that group.
    A district with accessibility data (e.g. Belgaum, area_id=190) returns available == true.
    """
    if not _datasets_available():
        pytest.skip("Required CSVs not available.")

    # 1. District without accessibility: 1 (Nicobar Islands)
    res_no = client.get("/api/area-profile/1")
    assert res_no.status_code == 200
    data_no = res_no.get_json()
    acc_no = data_no["accessibility"]
    assert acc_no["available"] is False
    # Ensure no numeric zeros masquerading as real values
    for item in acc_no.get("items", []):
        assert item.get("value") != 0.0

    # 2. District with accessibility: 190 (Belgaum in Karnataka)
    res_yes = client.get("/api/area-profile/190")
    assert res_yes.status_code == 200
    data_yes = res_yes.get_json()
    acc_yes = data_yes["accessibility"]
    assert acc_yes["available"] is True
    assert len(acc_yes["items"]) == 3
    values = {it["key"]: it["value"] for it in acc_yes["items"]}
    for it in acc_yes["items"]:
        assert it["value"] is not None
    # Ensure real measured values are used, not the 1.5/2.0/5.0 dummy sample
    assert values["road_density_km_per_sqkm"] != 1.5
    assert values["school_distance_km"] != 2.0
    assert values["hospital_distance_km"] != 5.0


def test_percentiles_calculation(client):
    """Percentiles are between 0 and 100 and consistent with the data for one hand-checked district."""
    if not _datasets_available():
        pytest.skip("Required CSVs not available.")

    res = client.get("/api/area-profile/182")
    assert res.status_code == 200
    data = res.get_json()

    final_df = pd.read_csv(os.path.join(PROCESSED_DIR, "final_training_table.csv"))
    row_182 = final_df[final_df["area_id"] == 182].iloc[0]

    for item in data["model_inputs"]:
        k = item["key"]
        pct = item["percentile"]
        assert 0.0 <= pct <= 100.0

        # Hand-check calculation
        expected_pct = round(float((final_df[k] <= row_182[k]).mean() * 100.0), 1)
        assert abs(pct - expected_pct) < 0.2


def test_flagged_nfhs_value_comes_back_as_null_with_flag_text(client):
    """A flagged NFHS value comes back as null with flag text."""
    if not _datasets_available():
        pytest.skip("Required CSVs not available.")

    # In final_training_table, find a district with a flagged indicator in all_nfhs5
    from backend.services import profile_service
    profile_service.reset_cache()

    # Search for an area_id where any indicator has a flag_text
    flagged_found = False
    for aid in [1, 2, 49, 137, 182, 190, 300, 350, 400, 450]:
        prof = profile_service.get_area_profile(aid)
        if not prof:
            continue
        for ind in prof["all_nfhs5"]:
            if ind.get("flag_text"):
                assert ind["nfhs5"] is None
                assert isinstance(ind["flag_text"], str)
                assert len(ind["flag_text"]) > 0
                flagged_found = True
                break
        if flagged_found:
            break

    assert flagged_found, "Expected at least one flagged indicator record among tested districts"


def test_nonexistent_area_returns_404(client):
    """GET /api/area-profile/999999 returns 404 JSON."""
    res = client.get("/api/area-profile/999999")
    assert res.status_code == 404
    data = res.get_json()
    assert "error" in data


def test_regression_guard_area_features(client):
    """GET /api/area-features/<id> still returns the same keys as before (regression guard)."""
    res = client.get("/api/area-features/182")
    assert res.status_code == 200
    data = res.get_json()
    expected_features = [
        "nightlight_mean", "ndvi_mean", "builtup_pct", "population_density",
        "elevation_mean_m", "slope_mean_deg", "rainfall_mm_year",
        "lst_day_c", "cropland_pct", "travel_time_city_min",
    ]
    for feat in expected_features:
        assert feat in data


def test_no_vgg_features_included(client):
    """The service never includes any key starting with 'vgg_'."""
    if not _datasets_available():
        pytest.skip("Required CSVs not available.")

    res = client.get("/api/area-profile/182")
    assert res.status_code == 200
    data = res.get_json()

    # Check recursively that no key anywhere starts with vgg_
    def check_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert not str(k).startswith("vgg_")
                check_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                check_keys(item)

    check_keys(data)

