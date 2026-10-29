"""
test_phase2.py
--------------
Tests for the Phase 2 training code: grouped validation, fold-safe preprocessing,
metrics, baseline, prediction intervals and an end-to-end run on a small synthetic table.

Run with:  pytest tests/test_phase2.py -v
"""

import json
import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluation import _fit, loso_predictions, summarize
from training.geo_validation import leave_one_state_out, repeated_group_kfold
from training.inference import predict_with_interval
from training.pipelines import IQRClipper, build_pipeline
from training.train_models import get_feature_columns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _toy(n_states=10, per_state=6, seed=0, nan_frac=0.1):
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_states):
        shift = rng.normal(0, 0.1)
        for d in range(per_state):
            a, b = rng.rand(), rng.rand()
            rows.append({"area_id": len(rows) + 1, "area_name": f"d{len(rows)}", "state_name": f"S{s}",
                         "nightlight_mean": a, "ndvi_mean": b, "builtup_pct": rng.rand() * 50,
                         "population_density": rng.rand() * 900,
                         "deprivation_score": float(np.clip(0.6 - 0.4 * a + 0.1 * b + shift + rng.normal(0, 0.03), 0, 1))})
    df = pd.DataFrame(rows)
    mask = rng.rand(*df[["nightlight_mean", "ndvi_mean"]].shape) < nan_frac
    df.loc[:, ["nightlight_mean", "ndvi_mean"]] = df[["nightlight_mean", "ndvi_mean"]].mask(mask)
    return df


def test_leave_one_state_out_never_shares_a_state():
    g = np.array(["A", "A", "B", "B", "C"])
    splits = list(leave_one_state_out(g))
    assert len(splits) == 3
    for state, tr, te in splits:
        assert set(g[te]) == {state}
        assert state not in set(g[tr])
        assert len(set(tr) & set(te)) == 0


def test_repeated_group_kfold_each_state_tested_once_per_repeat():
    g = np.repeat(np.array([f"S{i}" for i in range(12)]), 3)
    seen = {}
    for rep, fold, tr, te in repeated_group_kfold(g, n_splits=4, n_repeats=3, seed=1):
        assert not (set(g[tr]) & set(g[te]))
        for s in set(g[te]):
            seen.setdefault(rep, []).append(s)
    for rep, states in seen.items():
        assert sorted(states) == sorted(set(g))


def test_iqr_clipper_learns_bounds_on_train_only_and_keeps_nan():
    train = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8.0], "c": [5.0] * 8})
    clip = IQRClipper(3.0).fit(train)
    out = clip.transform(pd.DataFrame({"x": [1000.0, np.nan, -1000.0], "c": [9.0, 9.0, 9.0]}))
    assert out[0, 0] < 1000 and out[2, 0] > -1000      # clipped to train-based bounds
    assert np.isnan(out[1, 0])                         # NaN preserved
    assert (out[:, 1] == 9.0).all()                    # zero-IQR column untouched


def test_pipeline_imputes_with_training_median_and_ignores_test_values():
    df = _toy(nan_frac=0.2)
    cols = get_feature_columns(df)
    tr, te = df.iloc[:40], df.iloc[40:]
    pipe = build_pipeline("Ridge", cols, "none")
    pipe.fit(tr[cols], tr["deprivation_score"])
    stats_a = pipe.named_steps["pre"].named_steps["impute"].statistics_.copy()
    te2 = te.copy()
    te2["nightlight_mean"] = 1e6                       # absurd test values must not change anything fitted
    pipe2 = build_pipeline("Ridge", cols, "none")
    pipe2.fit(tr[cols], tr["deprivation_score"])
    assert np.allclose(stats_a, pipe2.named_steps["pre"].named_steps["impute"].statistics_)
    clipped = pipe.named_steps["pre"].named_steps["clip"].transform(tr[cols])
    expected = np.nanmedian(clipped, axis=0)
    assert np.allclose(stats_a, expected)
    assert np.isfinite(pipe.predict(te[cols])).all()   # NaN in test rows is handled


def test_loso_predicts_every_row_exactly_once():
    df = _toy()
    cols = get_feature_columns(df)
    pipe = build_pipeline("Ridge", cols, "none")
    pred = loso_predictions(pipe, "Ridge", df[cols], df["deprivation_score"], df["state_name"].values)
    assert sorted(pred["row"].tolist()) == list(range(len(df)))


def test_dummy_baseline_predicts_training_mean_of_other_states():
    df = _toy(nan_frac=0)
    cols = get_feature_columns(df)
    pipe = build_pipeline("Dummy", cols, "none")
    pred = loso_predictions(pipe, "Dummy", df[cols], df["deprivation_score"], df["state_name"].values)
    r = pred[pred["fold"] == "S0"].iloc[0]
    expected = df.loc[df["state_name"] != "S0", "deprivation_score"].mean()
    assert abs(r["y_pred"] - expected) < 1e-9


def test_summarize_perfect_predictions():
    y = np.linspace(0.1, 0.9, 30)
    d = pd.DataFrame({"y_true": y, "y_pred": y, "t1": np.quantile(y, 1 / 3), "t2": np.quantile(y, 2 / 3)})
    m = summarize(d)
    assert abs(m["r2"] - 1) < 1e-9 and m["mae"] < 1e-12 and m["cat_accuracy"] == 1.0 and m["top20_recall"] == 1.0


def test_nested_tuning_runs_inside_a_training_fold():
    df = _toy(nan_frac=0)
    cols = get_feature_columns(df)
    pipe = build_pipeline("Ridge", cols, "none")
    est = _fit(pipe, "Ridge", df[cols], df["deprivation_score"], df["state_name"].values, "nested", 3, 0)
    assert np.isfinite(est.predict(df[cols])).all()


def test_get_feature_columns_skips_identifiers_text_and_match_columns():
    df = _toy(nan_frac=0)
    df["gaul_state_name"] = "x"
    df["match_score"] = 1.0
    df["s2_image_count"] = 3
    cols = get_feature_columns(df)
    for bad in ["area_id", "state_name", "deprivation_score", "gaul_state_name", "match_score", "s2_image_count"]:
        assert bad not in cols


def test_end_to_end_training_writes_all_artefacts_and_intervals_work():
    df = _toy(n_states=12, per_state=6, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "toy_real.csv")
        df.to_csv(data, index=False)
        out = os.path.join(tmp, "models")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "training", "train_models.py"), "--data", data,
                            "--models-dir", out, "--fast", "--no-db"], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-1500:]
        for f in ["best_model.pkl", "model_comparison_results.csv", "repeated_groupkfold_results.csv",
                  "per_state_errors.csv", "loso_predictions.csv", "leaky_random_kfold_results.csv",
                  "feature_columns.json", "category_thresholds.json", "uncertainty.json", "run_metadata.json"]:
            assert os.path.exists(os.path.join(out, f)), f
        res = pd.read_csv(os.path.join(out, "model_comparison_results.csv"))
        assert "Dummy" in res["model"].values and {"r2", "mae", "rmse"} <= set(res.columns)
        bundle = joblib.load(os.path.join(out, "best_model.pkl"))
        pred, low, high = predict_with_interval(bundle, df.iloc[:5])
        assert (low <= pred).all() and (pred <= high).all()
        with open(os.path.join(out, "feature_columns.json")) as fh:
            assert json.load(fh) == bundle["feature_columns"]
