"""
test_scenario_explorer.py
-------------------------
Comprehensive tests for the upgraded Scenario Explorer module:
  - Preset constants in config.py (labeled as proxies, correct factors/deltas)
  - Single-district scenario prediction with 80% intervals & MAE error comparison
  - Group aggregation (State and Hotspot Cluster):
      mean change, median change, category shift count, share smaller than MAE
  - Training range bounds warnings (no silent extrapolation)
  - Mandatory model sensitivity disclaimer note
  - Route backward-compatibility (/whatif and /scenarios)
"""

import os
import json
import pytest

from config import SCENARIO_PRESETS, SCENARIO_DISCLAIMER_NOTE, PROCESSED_DIR


def _dataset_ready():
    return os.path.exists(os.path.join(PROCESSED_DIR, "final_training_table.csv"))


def test_scenario_presets_constants():
    """Requirement 1: Verify presets in config.py are defined as proxies with exact multipliers and deltas."""
    expected_keys = [
        "electrification_125",
        "electrification_150",
        "urbanisation_1pp",
        "urbanisation_2pp",
        "greening_005",
        "combined"
    ]
    for k in expected_keys:
        assert k in SCENARIO_PRESETS, f"Missing preset {k} in SCENARIO_PRESETS"
        preset = SCENARIO_PRESETS[k]
        assert "proxy" in preset["name"].lower() or "proxy" in preset["description"].lower()
        assert "intervention" not in preset["name"].lower()
        assert "intervention" not in preset["description"].lower()

    # Check exact numerical modifications
    e125 = SCENARIO_PRESETS["electrification_125"]
    assert e125["feature"] == "nightlight_mean"
    assert e125["op"] == "multiply" and e125["value"] == 1.25

    e150 = SCENARIO_PRESETS["electrification_150"]
    assert e150["feature"] == "nightlight_mean"
    assert e150["op"] == "multiply" and e150["value"] == 1.5

    u1 = SCENARIO_PRESETS["urbanisation_1pp"]
    assert u1["feature"] == "builtup_pct"
    assert u1["op"] == "add" and u1["value"] == 1.0

    u2 = SCENARIO_PRESETS["urbanisation_2pp"]
    assert u2["feature"] == "builtup_pct"
    assert u2["op"] == "add" and u2["value"] == 2.0

    g005 = SCENARIO_PRESETS["greening_005"]
    assert g005["feature"] == "ndvi_mean"
    assert g005["op"] == "add" and g005["value"] == 0.05

    comb = SCENARIO_PRESETS["combined"]
    assert "features" in comb
    assert comb["features"]["nightlight_mean"]["value"] == 1.25
    assert comb["features"]["builtup_pct"]["value"] == 1.0
    assert comb["features"]["ndvi_mean"]["value"] == 0.05


def test_single_district_preset_application(client):
    """Requirement 2 & 4: Applying a preset to one district returns score before/after, 80% interval, difference."""
    if not _dataset_ready():
        pytest.skip("Training table not available.")

    res = client.post("/api/scenario", json={
        "area_id": 204,
        "preset_id": "electrification_125"
    })
    assert res.status_code == 200
    data = res.get_json()

    assert "original_score" in data and "scenario_score" in data
    assert "original_interval_80" in data and len(data["original_interval_80"]) == 2
    assert "scenario_interval_80" in data and len(data["scenario_interval_80"]) == 2
    assert "difference" in data
    assert "model_mae" in data and data["model_mae"] > 0.0
    assert "is_smaller_than_mae" in data
    assert "mae_comparison_text" in data
    assert "warnings" in data
    assert "note" in data


def test_mae_error_comparison_text_format(client):
    """Requirement 4: When abs(change) < MAE, show 'Smaller than the model\'s typical error (MAE x)'."""
    if not _dataset_ready():
        pytest.skip("Training table not available.")

    # Small tweak on builtup_pct that produces tiny delta < MAE
    res = client.post("/api/scenario", json={
        "area_id": 204,
        "modified_features": {"builtup_pct": 0.8}
    })
    assert res.status_code == 200
    data = res.get_json()

    if data["is_smaller_than_mae"]:
        assert "Smaller than the model's typical error (MAE" in data["mae_comparison_text"]
    else:
        assert "Exceeds the model's typical error (MAE" in data["mae_comparison_text"]


def test_group_aggregation_state(client):
    """Requirement 2: Group scenario for all districts of a state returns mean, median, category changes, share < MAE."""
    if not _dataset_ready():
        pytest.skip("Training table not available.")

    res = client.post("/api/scenario", json={
        "state_name": "Goa",
        "preset_id": "urbanisation_1pp"
    })
    assert res.status_code == 200
    data = res.get_json()

    assert data["target_type"] == "state"
    assert data["total_districts"] == 2
    assert "mean_change" in data
    assert "median_change" in data
    assert "category_changes_count" in data
    assert 0 <= data["category_changes_count"] <= data["total_districts"]
    assert "share_smaller_than_mae" in data
    assert 0.0 <= data["share_smaller_than_mae"] <= 1.0
    assert "districts" in data and len(data["districts"]) == 2
    assert "model_mae" in data and data["model_mae"] > 0


def test_group_aggregation_hotspot_cluster(client):
    """Requirement 2: Group scenario for all districts of a hotspot cluster returns required aggregate statistics."""
    if not _dataset_ready():
        pytest.skip("Training table not available.")

    res = client.post("/api/scenario", json={
        "cluster_id": 0,
        "preset_id": "combined"
    })
    assert res.status_code == 200
    data = res.get_json()

    assert data["target_type"] == "cluster"
    assert data["total_districts"] > 0
    assert "mean_change" in data
    assert "median_change" in data
    assert "category_changes_count" in data
    assert "share_smaller_than_mae" in data
    assert 0.0 <= data["share_smaller_than_mae"] <= 1.0


def test_range_warnings_and_no_silent_extrapolation(client):
    """Requirement 3: Warning returned on values outside training range; never extrapolated silently."""
    if not _dataset_ready():
        pytest.skip("Training table not available.")

    res = client.post("/api/scenario", json={
        "area_id": 204,
        "modified_features": {"builtup_pct": 500.0}
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "warnings" in data
    assert len(data["warnings"]) > 0
    assert "outside the training range" in data["warnings"][0]


def test_mandatory_disclaimer_note(client):
    """Requirement 5: Always show the required exact sensitivity disclaimer note."""
    if not _dataset_ready():
        pytest.skip("Training table not available.")

    res = client.post("/api/scenario", json={
        "area_id": 204,
        "preset_id": "electrification_125"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "note" in data

    exact_required = (
        "This shows how the model's estimate responds to changes in satellite-derived features. "
        "It is not a prediction of the effect of a policy or programme. "
        "Schooling, health, nutrition and infrastructure are not in the model."
    )
    assert exact_required in data["note"]


def test_page_routes_and_navigation(client):
    """Requirement 6: Both /scenarios and /whatif return HTTP 200 with Scenario Explorer rendered."""
    r_scen = client.get("/scenarios")
    assert r_scen.status_code == 200
    assert b"Scenario Explorer" in r_scen.data

    r_whatif = client.get("/whatif")
    assert r_whatif.status_code == 200
    assert b"Scenario Explorer" in r_whatif.data

    r_presets = client.get("/api/scenario-presets")
    assert r_presets.status_code == 200
    assert "electrification_125" in r_presets.get_json()

    r_targets = client.get("/api/scenario-targets")
    assert r_targets.status_code == 200
    assert "states" in r_targets.get_json()
    assert "clusters" in r_targets.get_json()

