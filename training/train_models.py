"""
train_models.py
------------------
Trains and compares three models for the continuous deprivation_score
target (see Section 15/17 of the project design):

    1. Linear Regression   (transparent baseline)
    2. Random Forest Regressor
    3. XGBoost Regressor

Uses a GEOGRAPHY-AWARE train/test split (whole states held out, see
training/geo_validation.py) and GEOGRAPHY-AWARE cross-validation for
hyperparameter tuning (GroupKFold by state), rather than a naive random
split, to avoid spatial leakage inflating the apparent accuracy.

Saves:
    - models/model_comparison_results.csv   (metrics for all 3 models)
    - models/best_model.pkl                  (the best model by test RMSE)
    - models/feature_columns.json             (exact feature column order the
                                                  saved model expects -- needed
                                                  by the Flask API in Module 7)
    - Logs results into the model_results table of database/geopoverty.db

Usage:
    python training/train_models.py --data data/processed/demo_final_training_table.csv
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, MODELS_DIR, RANDOM_SEED
from training.geo_validation import get_group_kfold, grouped_train_test_split

NON_FEATURE_COLS = {
    "area_id", "area_name", "state_name", "data_source", "data_year", "deprivation_score",
    "latitude", "longitude", "centroid_lat", "centroid_lon", "data_mode",
    "image_path", "s2_image_count",
}
# latitude/longitude are deliberately excluded from features: including raw
# coordinates would let the model trivially memorize location rather than
# learn from genuine visual/geospatial signal, and would undermine the
# whole point of the geography-aware validation split.


def get_feature_columns(df: pd.DataFrame):
    """All numeric columns except identifiers/target -- generic on purpose,
    so it works regardless of exactly which geospatial columns are present
    (demo vs. real data may differ slightly in accessibility column naming)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_linear_baseline(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, groups_train, seed=RANDOM_SEED):
    cv = get_group_kfold(len(set(groups_train)), max_splits=5)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 8, 12, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=seed, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=cv.split(X_train, y_train, groups=groups_train),
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best RF params: {search.best_params_}")
    return search.best_estimator_


def train_xgboost(X_train, y_train, groups_train, seed=RANDOM_SEED):
    cv = get_group_kfold(len(set(groups_train)), max_splits=5)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=seed, n_jobs=-1, objective="reg:squarederror"),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=cv.split(X_train, y_train, groups=groups_train),
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best XGBoost params: {search.best_params_}")
    return search.best_estimator_


def log_results_to_db(results: dict, trained_on: str, db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        print(f"  NOTE: database not found at {db_path} -- skipping DB logging "
              f"(run database/init_db.py first if you want results logged there).")
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for model_name, metrics in results.items():
        cur.execute(
            """INSERT INTO model_results (model_name, mae, rmse, r2, trained_on, eval_timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (model_name, metrics["mae"], metrics["rmse"], metrics["r2"],
             trained_on, datetime.now(timezone.utc).isoformat()),
        )
    conn.commit()
    conn.close()
    print(f"  Logged {len(results)} model result(s) to {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and compare Linear/RandomForest/XGBoost on the fused training table")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.25,
                         help="Fraction of STATES (not rows) held out for testing")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    data_mode = "demo" if "demo" in os.path.basename(args.data).lower() or \
                (df.get("data_source", pd.Series(dtype=str)).astype(str).str.contains("Demo").any()) else "real"

    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns.")
    if "state_name" not in df.columns:
        raise ValueError("Dataset must contain a 'state_name' column for geography-aware validation.")

    X = df[feature_cols]
    y = df["deprivation_score"]
    groups = df["state_name"]

    X_train, X_test, y_train, y_test, groups_train, groups_test = grouped_train_test_split(
        X, y, groups, test_size=args.test_size, random_state=args.seed
    )

    results = {}
    fitted_models = {}

    print("\n--- Training Linear Regression baseline ---")
    lin_model = train_linear_baseline(X_train, y_train)
    results["LinearRegression"] = evaluate(lin_model, X_test, y_test)
    fitted_models["LinearRegression"] = lin_model

    print("\n--- Training Random Forest (with geography-aware hyperparameter tuning) ---")
    rf_model = train_random_forest(X_train, y_train, groups_train, seed=args.seed)
    results["RandomForest"] = evaluate(rf_model, X_test, y_test)
    fitted_models["RandomForest"] = rf_model

    print("\n--- Training XGBoost (with geography-aware hyperparameter tuning) ---")
    xgb_model = train_xgboost(X_train, y_train, groups_train, seed=args.seed)
    results["XGBoost"] = evaluate(xgb_model, X_test, y_test)
    fitted_models["XGBoost"] = xgb_model

    print("\n=== Model comparison (held-out, unseen-state test set) ===")
    results_df = pd.DataFrame(results).T
    results_df.index.name = "model"
    print(results_df.round(4).to_string())

    os.makedirs(MODELS_DIR, exist_ok=True)
    results_csv_path = os.path.join(MODELS_DIR, "model_comparison_results.csv")
    results_df.to_csv(results_csv_path)
    print(f"\nSaved comparison table to: {results_csv_path}")

    best_model_name = results_df["rmse"].idxmin()
    best_model = fitted_models[best_model_name]
    print(f"\nBest model by test RMSE: {best_model_name} "
          f"(RMSE={results_df.loc[best_model_name, 'rmse']:.4f}, "
          f"R2={results_df.loc[best_model_name, 'r2']:.4f})")

    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump({"model": best_model, "model_name": best_model_name}, best_model_path)
    print(f"Saved best model to: {best_model_path}")

    feature_cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved feature column order to: {feature_cols_path}")

    # --- Derive Low/Medium/High category thresholds from the TRAINING
    # distribution only (Section 15 of the design doc: thresholds must be
    # data-driven, never arbitrary, and must not peek at the test set) ---
    tertiles = y_train.quantile([1 / 3, 2 / 3]).tolist()
    thresholds = {
        "low_medium_boundary": tertiles[0],
        "medium_high_boundary": tertiles[1],
        "derived_from": "training set deprivation_score tertiles",
        "note": "Low: score <= low_medium_boundary; Medium: between boundaries; "
                "High: score > medium_high_boundary. Categories are a display "
                "simplification of the continuous score, not a separately trained target.",
    }
    thresholds_path = os.path.join(MODELS_DIR, "category_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved category thresholds to: {thresholds_path} "
          f"(Low <= {tertiles[0]:.4f} < Medium <= {tertiles[1]:.4f} < High)")

    log_results_to_db(results, trained_on=data_mode)

    if data_mode == "demo":
        print("\nREMINDER: these metrics were computed on SYNTHETIC DEMO data. "
              "Do not report these numbers as real research findings.")


if __name__ == "__main__":
    main()
