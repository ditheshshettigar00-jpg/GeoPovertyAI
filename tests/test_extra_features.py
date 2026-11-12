"""Plumbing tests for the experimental extra-feature scripts (no Earth Engine needed)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from config import MODELS_DIR, PROCESSED_DIR, RAW_DIR

TABLE = os.path.join(PROCESSED_DIR, "final_training_table.csv")
needs_data = pytest.mark.skipif(not os.path.exists(TABLE) or not os.path.isdir(RAW_DIR),
                                reason="training table / boundary files not present")


@needs_data
def test_every_training_district_has_a_boundary():
    from scripts.extract_extra_features import load_districts
    table = pd.read_csv(TABLE)
    districts = load_districts(table)
    assert len(districts) == len(table) == 564
    assert len({d["area_id"] for d in districts}) == 564          # global area_id, not the per-state one
    assert all(d["geometry"]["type"] in ("Polygon", "MultiPolygon", "GeometryCollection") for d in districts)


def test_pending_districts_skips_complete_rows_and_retries_missing():
    from scripts.extract_extra_features import pending_districts
    districts = [{"area_id": i} for i in (1, 2, 3, 4)]
    existing = pd.DataFrame({"area_id": [1, 2, 3], "elevation_mean_m": [10.0, np.nan, 5.0],
                             "slope_mean_deg": [1.0, 2.0, 3.0]})
    todo = pending_districts(districts, existing, ["elevation_mean_m", "slope_mean_deg"])
    assert [d["area_id"] for d in todo] == [2, 4]                 # 2 has a missing value, 4 is new
    assert len(pending_districts(districts, pd.DataFrame(), ["elevation_mean_m"])) == 4
    # asking for a feature that is not in the file yet means every district still needs it
    assert len(pending_districts(districts, existing, ["rainfall_mm_year"])) == 4


def _res(d_loso, d_rep, states):
    return pd.DataFrame({"feature_set": ["baseline (4 features)", "baseline + ALL extra"],
                         "d_loso_r2": [0.0, d_loso], "d_repeated_r2": [0.0, d_rep],
                         "states_improved": [0.0, states]})


@pytest.mark.parametrize("d_loso,d_rep,states,expected", [
    (0.03, 0.03, 0.70, True),
    (0.03, 0.03, 0.40, False),     # most states got worse
    (0.03, 0.01, 0.70, False),     # not confirmed by repeated CV
    (0.01, 0.03, 0.70, False),     # gain too small
    (-0.02, -0.02, 0.58, False),   # noise can still "improve" many states; R2 gates catch it
])
def test_verdict_requires_all_three_conditions(d_loso, d_rep, states, expected):
    from training.evaluate_extra_features import verdict
    assert verdict(_res(d_loso, d_rep, states), 0.02, 0.55)[0] is expected


@needs_data
def test_evaluation_smoke_run_with_noise_is_not_adopted():
    from training.evaluate_extra_features import run, verdict
    df = pd.read_csv(TABLE).dropna(subset=["deprivation_score"]).reset_index(drop=True)
    base = json.load(open(os.path.join(MODELS_DIR, "feature_columns.json")))
    df["noise_a"] = np.random.default_rng(0).normal(size=len(df))
    res = run(df, base, ["noise_a"], model="Ridge", repeats=1, per_feature=False)
    assert list(res["feature_set"]) == ["baseline (4 features)", "baseline + ALL extra"]
    assert res.loc[0, "d_loso_r2"] == 0
    assert verdict(res, 0.02, 0.55)[0] is False
