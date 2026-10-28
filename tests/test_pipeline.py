"""
test_pipeline.py
-------------------
Test suite covering: dataset generation, feature fusion, geography-aware
validation, model training utilities, and the Flask API. Uses the demo
(synthetic) pipeline throughout, since it requires no external network
access (GEE/OSM) and runs anywhere.

Run with:
    pytest tests/ -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_demo_data import generate_demo_dataset
from preprocessing.feature_fusion import (
    remove_duplicates, impute_missing, clip_outliers_iqr, fuse_features,
)
from training.geo_validation import grouped_train_test_split, get_group_kfold
from training.train_models import get_feature_columns


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def test_demo_dataset_shape_and_no_missing_values():
    df = generate_demo_dataset(n_areas=30, seed=1)
    assert len(df) == 30
    assert "deprivation_score" in df.columns
    assert df["deprivation_score"].between(0, 1).all()
    assert df.isna().sum().sum() == 0


def test_demo_dataset_reproducible_with_same_seed():
    df1 = generate_demo_dataset(n_areas=10, seed=5)
    df2 = generate_demo_dataset(n_areas=10, seed=5)
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Feature fusion building blocks
# ---------------------------------------------------------------------------

def test_remove_duplicates_keeps_first():
    df = pd.DataFrame({"area_id": [1, 1, 2], "val": [10, 20, 30]})
    result = remove_duplicates(df, "test")
    assert len(result) == 2
    assert result[result["area_id"] == 1]["val"].iloc[0] == 10


def test_impute_missing_fills_with_median():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    result = impute_missing(df, ["x"])
    assert result["x"].isna().sum() == 0
    assert result.loc[2, "x"] == pytest.approx(df["x"].median())


def test_clip_outliers_bounds_extreme_values():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    result = clip_outliers_iqr(df, ["x"], iqr_multiplier=1.5)
    assert result["x"].max() < 1000
    # non-outlier values should be unaffected
    assert result["x"].iloc[0] == 1


def test_fuse_features_merges_and_reports_mismatched_ids(tmp_path):
    gt = pd.DataFrame({"area_id": [1, 2, 3], "deprivation_score": [0.2, 0.5, 0.8]})
    geo = pd.DataFrame({"area_id": [1, 2, 3], "nightlight_mean": [1.0, 2.0, 3.0]})
    acc = pd.DataFrame({"area_id": [1, 2], "road_density_km_per_sqkm": [0.5, 0.6]})  # id 3 missing on purpose
    vgg = pd.DataFrame({"area_id": [1, 2, 3], "vgg_feat_1": [0.1, 0.2, 0.3]})

    gt.to_csv(tmp_path / "gt.csv", index=False)
    geo.to_csv(tmp_path / "geo.csv", index=False)
    acc.to_csv(tmp_path / "acc.csv", index=False)
    vgg.to_csv(tmp_path / "vgg.csv", index=False)

    result = fuse_features(str(tmp_path / "gt.csv"), str(tmp_path / "geo.csv"),
                            str(tmp_path / "acc.csv"), str(tmp_path / "vgg.csv"))

    assert len(result) == 2  # only area_id 1 and 2 are present in ALL sources
    assert 3 not in result["area_id"].values
    assert result.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Geography-aware validation
# ---------------------------------------------------------------------------

def test_grouped_split_holds_out_entire_states():
    df = generate_demo_dataset(n_areas=60, seed=2)
    X = df.drop(columns=["deprivation_score"])
    y = df["deprivation_score"]
    groups = df["state_name"]

    X_train, X_test, y_train, y_test, groups_train, groups_test = grouped_train_test_split(
        X, y, groups, test_size=0.3, random_state=1
    )

    train_states = set(groups_train)
    test_states = set(groups_test)
    assert train_states.isdisjoint(test_states), "A state leaked into both train and test!"
    assert len(X_train) + len(X_test) == len(df)


def test_group_kfold_rejects_too_few_groups():
    with pytest.raises(ValueError):
        get_group_kfold(n_groups_available=1, max_splits=5)


def test_group_kfold_reduces_splits_when_few_groups_available():
    cv = get_group_kfold(n_groups_available=2, max_splits=5)
    assert cv.get_n_splits() == 2


# ---------------------------------------------------------------------------
# Feature column detection
# ---------------------------------------------------------------------------

def test_get_feature_columns_excludes_identifiers_and_target():
    df = generate_demo_dataset(n_areas=5, seed=3)
    cols = get_feature_columns(df)
    assert "deprivation_score" not in cols
    assert "area_id" not in cols
    assert "state_name" not in cols
    assert "nightlight_mean" in cols
    assert "vgg_feat_1" in cols


# ---------------------------------------------------------------------------
# Flask API (requires database/geopoverty.db + models/ to already be populated
# by running Modules 1, 5, and 6's scripts first -- these tests are skipped
# gracefully if that setup hasn't been run yet, rather than failing confusingly)
# ---------------------------------------------------------------------------

def _artifacts_available():
    from config import DB_PATH, MODELS_DIR
    return os.path.exists(DB_PATH) and os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))


@pytest.fixture
def client():
    if not _artifacts_available():
        pytest.skip("Run database/init_db.py, training/train_models.py, and "
                     "scripts/precompute_predictions.py before running API tests.")
    from backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200


def test_api_areas_returns_list(client):
    r = client.get("/api/areas")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "area_id" in data[0]


def test_api_predict_valid_area(client):
    areas = client.get("/api/areas").get_json()
    area_id = areas[0]["area_id"]
    r = client.get(f"/api/predict/{area_id}")
    assert r.status_code == 200
    data = r.get_json()
    assert "predicted_score" in data
    assert data["predicted_category"] in ("Low", "Medium", "High")


def test_api_predict_invalid_area_returns_404(client):
    r = client.get("/api/predict/999999")
    assert r.status_code == 404


def test_api_explain_returns_shap_and_flags_gradcam(client):
    areas = client.get("/api/areas").get_json()
    area_id = areas[0]["area_id"]
    r = client.get(f"/api/explain/{area_id}")
    assert r.status_code == 200
    data = r.get_json()
    assert "shap_explanation" in data
    assert "top_contributions" in data["shap_explanation"]
    assert "gradcam_available" in data


def test_api_scenario_missing_body_returns_400(client):
    r = client.post("/api/scenario", json={})
    assert r.status_code == 400


def test_api_scenario_valid_request(client):
    areas = client.get("/api/areas").get_json()
    area_id = areas[0]["area_id"]
    r = client.post("/api/scenario", json={"area_id": area_id, "modified_features": {"builtup_pct": 10}})
    assert r.status_code == 200
    data = r.get_json()
    assert "original_score" in data
    assert "scenario_score" in data
    assert "not a causal simulation" in data["note"]


def test_api_scenario_unknown_feature_returns_400(client):
    areas = client.get("/api/areas").get_json()
    area_id = areas[0]["area_id"]
    r = client.post("/api/scenario", json={"area_id": area_id, "modified_features": {"totally_fake_feature": 1}})
    assert r.status_code == 400


def test_api_model_performance_returns_metrics(client):
    r = client.get("/api/model-performance")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "mae" in data[0] and "rmse" in data[0] and "r2" in data[0]


def test_api_predict_live_custom_features(client):
    payload = {
        "nightlight_mean": 4.5,
        "ndvi_mean": 0.40,
        "builtup_pct": 250.0,
        "population_density": 500.0
    }
    r = client.post("/api/predict-live", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert "predicted_score" in data
    assert data["predicted_category"] in ("Low", "Medium", "High")
    assert "contributions" in data
    
    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False
        
    if has_shap:
        assert len(data["contributions"]) > 0
        
    assert "interpretation" in data


def test_auth_register_and_login(client):
    # Register
    import time
    unique_email = f"testuser_{int(time.time())}@example.com"
    r_reg = client.post("/api/auth/register", json={
        "name": "Test User",
        "email": unique_email,
        "password": "password123"
    })
    assert r_reg.status_code == 201
    
    # Login
    r_login = client.post("/api/auth/login", json={
        "email": unique_email,
        "password": "password123"
    })
    assert r_login.status_code == 200
    assert "user" in r_login.get_json()
    
    # Profile (should be successful since logged in)
    r_prof = client.get("/api/auth/profile")
    assert r_prof.status_code == 200
    
    # Logout
    r_out = client.post("/api/auth/logout")
    assert r_out.status_code == 200
    
    # Profile should fail now
    r_prof_fail = client.get("/api/auth/profile")
    assert r_prof_fail.status_code == 401


def test_admin_route_protection(client):
    # Unauthenticated access to HTML route
    r = client.get("/admin")
    assert r.status_code == 302 # Redirect to login
    
    # Unauthenticated access to API route
    r_api = client.get("/api/admin/dashboard-stats")
    assert r_api.status_code == 401


def test_report_generation_protection(client):
    # Need researcher/admin role
    areas = client.get("/api/areas").get_json()
    if areas:
        area_id = areas[0]["area_id"]
        r = client.get(f"/api/reports/district/{area_id}")
        assert r.status_code == 401
