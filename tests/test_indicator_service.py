"""
test_indicator_service.py - Comprehensive tests for real indicator values.

Verifies:
1. For area_id 182 (Pashchim Singhbhum, Jharkhand), the API returns exactly:
   nightlight_mean 0.6292, ndvi_mean 0.5143, builtup_pct 0.4829, population_density 241.1,
   s2_image_count 322, latitude 22.4163, longitude 85.5136 (compared to CSV, never hard-coded twice).
2. Survey values for area 182:
   electricity 83.3 / 69.4, sanitation 37.7 / 16.6, clean fuel 15.5 / 10.0, stunting 60.6 / 59.4, anaemia 72.6 / 72.8.
3. Every one of the 564 districts returns 13 satellite/geospatial and 5 survey items.
4. All 564 districts join successfully from final_training_table to nfhs5_districts_raw.
5. get_feature_vector returns None, not 0.0, for a missing column or NaN value.
6. Percentiles are strictly between 0 and 100.
7. A district with a missing or suppressed survey value gets the 'not shown (fewer than 25 cases)' text.
8. /api/reports/district/<id> returns a PDF starting with %PDF.
9. No 'VGG' or 'OSM' strings appear in report JSON (/api/reports/district-data/<id>).
"""

import os
import pandas as pd
import pytest

from config import PROCESSED_DIR, GROUND_TRUTH_DIR
from preprocessing.name_keys import clean_key
from backend.services import indicator_service, model_service


def test_area_182_satellite_values_exact(client):
    """Verifies that area_id 182 returns exact satellite values matching final_training_table.csv."""
    res = client.get("/api/indicators/182")
    assert res.status_code == 200
    data = res.get_json()

    sat_items = {s["key"]: s["value"] for s in data["satellite_indicators"]}
    assert len(sat_items) == 13

    # Load from CSV directly to compare dynamically without duplicating hardcoded values
    csv_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    df = pd.read_csv(csv_path).set_index("area_id")
    row = df.loc[182]

    assert sat_items["nightlight_mean"] == round(float(row["nightlight_mean"]), 4)
    assert sat_items["nightlight_mean"] == 0.6292

    assert sat_items["ndvi_mean"] == round(float(row["ndvi_mean"]), 4)
    assert sat_items["ndvi_mean"] == 0.5143

    assert sat_items["builtup_pct"] == round(float(row["builtup_pct"]), 4)
    assert sat_items["builtup_pct"] == 0.4829

    assert sat_items["population_density"] == round(float(row["population_density"]), 1)
    assert sat_items["population_density"] == 241.1

    for extra_key in (
        "elevation_mean_m",
        "slope_mean_deg",
        "rainfall_mm_year",
        "lst_day_c",
        "cropland_pct",
        "travel_time_city_min",
    ):
        assert sat_items[extra_key] == round(float(row[extra_key]), 2)

    assert sat_items["s2_image_count"] == int(round(float(row["s2_image_count"])))
    assert sat_items["s2_image_count"] == 322

    assert sat_items["latitude"] == round(float(row["latitude"]), 4)
    assert sat_items["latitude"] == 22.4163

    assert sat_items["longitude"] == round(float(row["longitude"]), 4)
    assert sat_items["longitude"] == 85.5136


def test_area_182_survey_values_exact(client):
    """Verifies that area_id 182 returns the exact five ground-truth component survey indicators."""
    res = client.get("/api/indicators/182")
    assert res.status_code == 200
    data = res.get_json()

    surv_by_key = {s["key"]: s for s in data["survey_indicators"]}
    assert len(surv_by_key) == 5

    # 1. Electricity (indicator 7)
    assert surv_by_key["electricity"]["nfhs5"] == 83.3
    assert surv_by_key["electricity"]["nfhs4"] == 69.4
    assert surv_by_key["electricity"]["change"] == 13.9
    assert surv_by_key["electricity"]["better_when"] == "higher"

    # 2. Sanitation (indicator 9)
    assert surv_by_key["sanitation"]["nfhs5"] == 37.7
    assert surv_by_key["sanitation"]["nfhs4"] == 16.6
    assert surv_by_key["sanitation"]["change"] == 21.1
    assert surv_by_key["sanitation"]["better_when"] == "higher"

    # 3. Clean cooking fuel (indicator 10)
    assert surv_by_key["clean_fuel"]["nfhs5"] == 15.5
    assert surv_by_key["clean_fuel"]["nfhs4"] == 10.0
    assert surv_by_key["clean_fuel"]["change"] == 5.5
    assert surv_by_key["clean_fuel"]["better_when"] == "higher"

    # 4. Child stunting (indicator 73)
    assert surv_by_key["stunting"]["nfhs5"] == 60.6
    assert surv_by_key["stunting"]["nfhs4"] == 59.4
    assert surv_by_key["stunting"]["change"] == 1.2
    assert surv_by_key["stunting"]["better_when"] == "lower"

    # 5. Anaemia in women (indicator 84)
    assert surv_by_key["anaemia"]["nfhs5"] == 72.6
    assert surv_by_key["anaemia"]["nfhs4"] == 72.8
    assert surv_by_key["anaemia"]["change"] == -0.2
    assert surv_by_key["anaemia"]["better_when"] == "lower"


def test_all_564_districts_join_to_nfhs():
    """Verifies that 100% of the 564 districts in final_training_table join cleanly to nfhs5_districts_raw."""
    final_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    nfhs_path = os.path.join(GROUND_TRUTH_DIR, "nfhs5_districts_raw.csv")

    final_df = pd.read_csv(final_path)
    nfhs_df = pd.read_csv(nfhs_path)

    final_keys = set(zip(final_df["state_name"].map(clean_key), final_df["nfhs_district_name"].map(clean_key)))
    nfhs_keys = set(zip(nfhs_df["state"].map(clean_key), nfhs_df["district"].map(clean_key)))

    missing = [k for k in final_keys if k not in nfhs_keys]
    assert len(final_df) == 564
    assert len(missing) == 0, f"Unjoined districts: {missing}"


def test_all_564_districts_return_13_satellite_and_5_survey_items():
    """Verifies that every district among all 564 returns 13 satellite/geospatial and 5 survey items."""
    final_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    df = pd.read_csv(final_path)
    assert len(df) == 564

    for aid in df["area_id"]:
        sat = indicator_service.get_satellite_indicators(aid)
        surv = indicator_service.get_survey_indicators(aid)
        assert len(sat) == 13, f"District {aid} has {len(sat)} satellite items, expected 13"
        assert len(surv) == 5, f"District {aid} has {len(surv)} survey items, expected 5"


def test_get_feature_vector_returns_none_not_zero_for_missing_column(monkeypatch):
    """Verifies get_feature_vector returns None for missing or NaN column values, not 0.0."""
    model_service.load_artifacts()
    area_id = 182
    real_feats = model_service.get_feature_vector(area_id)
    assert real_feats is not None
    # All 10 features in regular dataset are numeric
    for v in real_feats.values():
        assert v is not None and not pd.isna(v)

    # Now mock a NaN value in one feature column in _feature_table
    orig_table = model_service._feature_table.copy()
    try:
        model_service._feature_table.loc[area_id, "nightlight_mean"] = None
        mocked_feats = model_service.get_feature_vector(area_id)
        assert mocked_feats["nightlight_mean"] is None
        assert mocked_feats["nightlight_mean"] != 0.0
    finally:
        model_service._feature_table = orig_table


def test_percentiles_between_0_and_100(client):
    """Verifies all calculated percentiles for satellite indicators are strictly within 0 and 100."""
    res = client.get("/api/indicators/182")
    data = res.get_json()
    for s in data["satellite_indicators"]:
        if s["percentile"] is not None:
            assert 0.0 <= s["percentile"] <= 100.0, f"Percentile out of bounds: {s}"

    for sv in data["survey_indicators"]:
        if sv["percentile"] is not None:
            assert 0.0 <= sv["percentile"] <= 100.0, f"Survey percentile out of bounds: {sv}"


def test_district_with_missing_survey_value_gets_not_shown_text():
    """Verifies that missing/suppressed survey indicators return 'not shown (fewer than 25 cases)'."""
    # District 564 or any district with full indicators query
    # Find an indicator with fewer than 25 cases
    all_inds = indicator_service.get_all_survey_indicators(1)
    # Check that any indicator with missing NFHS-5 gets 'not shown (fewer than 25 cases)'
    suppressed = [ind for ind in all_inds if ind["flag_text"] and "fewer than 25" in ind["flag_text"].lower()]
    assert len(suppressed) > 0, "Expected at least one suppressed indicator in full survey"
    for s in suppressed:
        assert s["nfhs5"] is None
        assert s["flag_text"] == "not shown (fewer than 25 cases)"


def test_report_pdf_generation_starts_with_pdf(client):
    """Verifies /api/reports/district/<id> returns a valid PDF beginning with %PDF."""
    from tests.conftest import make_user, login_session
    login_session(client, make_user("ADMIN"))

    # Standard PDF
    res = client.get("/api/reports/district/182")
    assert res.status_code == 200
    pdf_bytes = res.get_data()
    assert pdf_bytes.startswith(b"%PDF")

    # Appendix PDF
    res_app = client.get("/api/reports/district/182?appendix=1")
    assert res_app.status_code == 200
    pdf_app_bytes = res_app.get_data()
    assert pdf_app_bytes.startswith(b"%PDF")
    assert len(pdf_app_bytes) > len(pdf_bytes)


def test_no_vgg_or_osm_in_report_json(client):
    """Verifies that neither 'VGG' nor 'OSM' appears anywhere in the report data JSON."""
    res = client.get("/api/reports/district-data/182")
    assert res.status_code == 200
    raw_text = res.get_data(as_text=True)

    assert "OSM" not in raw_text, "Found legacy 'OSM' string in report JSON"
    assert "VGG" not in raw_text, "Found legacy 'VGG' string in report JSON"

    data = res.get_json()
    assert "satellite_indicators" in data
    assert len(data["satellite_indicators"]) == 13
    assert "survey_indicators" in data
    assert len(data["survey_indicators"]) == 5

