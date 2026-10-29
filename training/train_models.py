"""
train_models.py
------------------
Trains, evaluates, and compares models for continuous district-level
socioeconomic deprivation estimation (deprivation_score):

Baselines and Models:
    1. Training Mean Predictor (Dummy zero-skill baseline)
    2. Ridge Regression (L2-regularized linear baseline with scaling)
    3. Random Forest Regressor (Tuned ensemble with state-grouped CV)
    4. XGBoost Regressor (Gradient-boosted decision trees)

Validation Methodology (Zero Spatial Leakage):
    - Primary: Leave-One-State-Out (LOGO) cross-validation across all 27 states
    - Secondary: 5-Fold GroupKFold cross-validation by state
    - All transformers (3.0x IQR clipping, median imputation, standardization)
      are strictly encapsulated in scikit-learn Pipelines fitted on training folds only.

Deliverables:
    - models/model_comparison_results.csv (Comprehensive cross-validation metrics)
    - models/per_state_errors.csv         (Leave-one-state-out MAE per state)
    - models/ablation_results.csv         (Feature group ablation study)
    - models/best_model.pkl               (Best pipeline refit on all data + uncertainty deltas)
    - models/feature_columns.json         (Active input feature column list)
    - models/category_thresholds.json     (Empirical tertile boundaries)
    - database/geopoverty.db              (Logged to model_results table)
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
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, MODELS_DIR, RANDOM_SEED, PROCESSED_DIR
from training.transformers import IQRClipper, TrainingMedianImputer, MeanPredictor

NON_FEATURE_COLS = {
    "area_id", "area_name", "state_name", "data_source", "data_year", "deprivation_score",
    "latitude", "longitude", "centroid_lat", "centroid_lon", "data_mode",
    "image_path", "s2_image_count",
}


# =====================================================================
# Scikit-Learn Pipeline Transformers (Fitted strictly on Training Folds)
# =====================================================================
# (Imported from training.transformers for cross-module pickling support)


def get_feature_columns(df: pd.DataFrame):
    """Returns active feature columns excluding identifiers, coordinates, and targets."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def calculate_metrics(y_true, y_pred, y_train_for_tertiles=None):
    """Computes comprehensive regression, ranking, and policy-relevant metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Spearman rank correlation
    sp_corr = spearmanr(y_true, y_pred)[0]
    if np.isnan(sp_corr):
        sp_corr = 0.0

    # 3-Class Accuracy based on training tertiles
    ref_y = y_true if y_train_for_tertiles is None else y_train_for_tertiles
    t1, t2 = np.quantile(ref_y, [1/3, 2/3])
    true_cat = np.where(y_true <= t1, 0, np.where(y_true <= t2, 1, 2))
    pred_cat = np.where(y_pred <= t1, 0, np.where(y_pred <= t2, 1, 2))
    acc3 = float(np.mean(true_cat == pred_cat))

    # Recall of Top 20% Most Deprived Districts (Targeting fidelity)
    top20_thresh = np.quantile(y_true, 0.80)
    pred20_thresh = np.quantile(y_pred, 0.80)
    top20_true = (y_true >= top20_thresh)
    top20_pred = (y_pred >= pred20_thresh)
    top20_recall = float(np.sum(top20_true & top20_pred) / max(1, np.sum(top20_true)))

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "spearman": float(sp_corr),
        "accuracy_3class": float(acc3),
        "top20_recall": float(top20_recall),
    }


def build_pipeline(model_type, seed=RANDOM_SEED):
    """Constructs an sklearn Pipeline with preprocessing encapsulated."""
    clipper = IQRClipper(iqr_multiplier=3.0)
    imputer = TrainingMedianImputer()

    if model_type == "MeanPredictor":
        return Pipeline([("imputer", imputer), ("model", MeanPredictor())])
    elif model_type == "Ridge":
        return Pipeline([
            ("clipper", clipper),
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=seed))
        ])
    elif model_type == "RandomForest":
        return Pipeline([
            ("clipper", clipper),
            ("imputer", imputer),
            ("model", RandomForestRegressor(
                n_estimators=150, max_depth=10, min_samples_leaf=3,
                random_state=seed, n_jobs=-1
            ))
        ])
    elif model_type == "XGBoost":
        return Pipeline([
            ("clipper", clipper),
            ("imputer", imputer),
            ("model", XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=seed, n_jobs=-1,
                objective="reg:squarederror"
            ))
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_leave_one_state_out(X, y, groups, model_names, seed=RANDOM_SEED):
    """Executes Leave-One-State-Out (LOGO) cross-validation across all states."""
    logo = LeaveOneGroupOut()
    out_of_fold_preds = {m: np.zeros(len(y)) for m in model_names}
    unique_groups = np.unique(groups)

    print(f"\nRunning Leave-One-State-Out Cross-Validation across {len(unique_groups)} states...")

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        for m_name in model_names:
            pipe = build_pipeline(m_name, seed=seed)
            pipe.fit(X_train, y_train)
            out_of_fold_preds[m_name][test_idx] = pipe.predict(X_test)

    results = {}
    for m_name in model_names:
        metrics = calculate_metrics(y.values, out_of_fold_preds[m_name])
        results[m_name] = metrics

    return results, out_of_fold_preds


def run_group_kfold(X, y, groups, model_names, n_splits=5, seed=RANDOM_SEED):
    """Executes 5-Fold GroupKFold cross-validation by state."""
    gkf = GroupKFold(n_splits=n_splits)
    out_of_fold_preds = {m: np.zeros(len(y)) for m in model_names}

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        for m_name in model_names:
            pipe = build_pipeline(m_name, seed=seed)
            pipe.fit(X_train, y_train)
            out_of_fold_preds[m_name][test_idx] = pipe.predict(X_test)

    results = {}
    for m_name in model_names:
        metrics = calculate_metrics(y.values, out_of_fold_preds[m_name])
        results[m_name] = metrics

    return results


def run_ablation_study(df, feature_cols, groups, y, seed=RANDOM_SEED):
    """Evaluates performance when withholding subsets of features."""
    subsets = {
        "All Features": feature_cols,
        "No Nightlight": [c for c in feature_cols if c != "nightlight_mean"],
        "No NDVI": [c for c in feature_cols if c != "ndvi_mean"],
        "No Built-up": [c for c in feature_cols if c != "builtup_pct"],
        "No Population": [c for c in feature_cols if c != "population_density"],
        "Nightlight Only": [c for c in feature_cols if c == "nightlight_mean"],
        "NDVI Only": [c for c in feature_cols if c == "ndvi_mean"],
        "Builtup & Population Only": [c for c in feature_cols if c in ("builtup_pct", "population_density")],
    }

    logo = LeaveOneGroupOut()
    ablation_rows = []

    print("\nRunning Feature Group Ablation Study (Random Forest, LOGO)...")
    for name, cols in subsets.items():
        if not cols:
            continue
        X_sub = df[cols]
        preds = np.zeros(len(y))
        for train_idx, test_idx in logo.split(X_sub, y, groups=groups):
            pipe = build_pipeline("RandomForest", seed=seed)
            pipe.fit(X_sub.iloc[train_idx], y.iloc[train_idx])
            preds[test_idx] = pipe.predict(X_sub.iloc[test_idx])
        
        m = calculate_metrics(y.values, preds)
        ablation_rows.append({
            "feature_set": name,
            "feature_count": len(cols),
            "r2": m["r2"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "spearman": m["spearman"],
            "accuracy_3class": m["accuracy_3class"],
            "top20_recall": m["top20_recall"]
        })

    return pd.DataFrame(ablation_rows)


def log_results_to_db(results: dict, trained_on: str, db_path: str = DB_PATH):
    """Persists model comparison metrics into the database."""
    if not os.path.exists(db_path):
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


def main():
    parser = argparse.ArgumentParser(description="State-grouped training and evaluation pipeline for GeoPoverty AI")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    data_path = args.data or os.path.join(PROCESSED_DIR, "final_training_table.csv")
    df = pd.read_csv(data_path)
    data_mode = "demo" if "demo" in os.path.basename(data_path).lower() else "real"

    feature_cols = get_feature_columns(df)
    print(f"Loaded dataset: {data_path} ({len(df)} records, {df['state_name'].nunique()} states)")
    print(f"Active features ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df["deprivation_score"]
    groups = df["state_name"]

    model_names = ["MeanPredictor", "Ridge", "RandomForest", "XGBoost"]

    # 1. Primary Evaluation: Leave-One-State-Out (LOGO)
    logo_results, logo_preds = run_leave_one_state_out(X, y, groups, model_names, seed=args.seed)
    logo_df = pd.DataFrame(logo_results).T
    logo_df.index.name = "model"

    print("\n" + "=" * 75)
    print("PRIMARY EVALUATION: LEAVE-ONE-STATE-OUT CROSS-VALIDATION (UNSEEN STATES)")
    print("=" * 75)
    print(logo_df.round(4).to_string())

    # 2. Secondary Evaluation: 5-Fold GroupKFold
    gkf_results = run_group_kfold(X, y, groups, model_names, n_splits=5, seed=args.seed)
    gkf_df = pd.DataFrame(gkf_results).T
    gkf_df.index.name = "model"

    print("\n" + "=" * 75)
    print("SECONDARY EVALUATION: 5-FOLD GROUPKFOLD CROSS-VALIDATION (STATE GROUPS)")
    print("=" * 75)
    print(gkf_df.round(4).to_string())

    # Save Comparison Results
    os.makedirs(MODELS_DIR, exist_ok=True)
    comparison_path = os.path.join(MODELS_DIR, "model_comparison_results.csv")
    logo_df.to_csv(comparison_path)
    print(f"\nSaved primary comparison table to: {comparison_path}")

    # 3. Per-State Error Analysis (using best model out-of-fold predictions)
    best_model_name = logo_df.loc[["Ridge", "RandomForest", "XGBoost"], "rmse"].idxmin()
    best_preds = logo_preds[best_model_name]
    
    state_errors = []
    for st in np.unique(groups):
        mask = (groups == st)
        s_mae = mean_absolute_error(y[mask], best_preds[mask])
        s_rmse = np.sqrt(mean_squared_error(y[mask], best_preds[mask]))
        state_errors.append({
            "state_name": st,
            "district_count": int(np.sum(mask)),
            "mae": round(float(s_mae), 4),
            "rmse": round(float(s_rmse), 4)
        })
    state_err_df = pd.DataFrame(state_errors).sort_values("mae", ascending=False)
    per_state_path = os.path.join(MODELS_DIR, "per_state_errors.csv")
    state_err_df.to_csv(per_state_path, index=False)
    print(f"Saved per-state errors to: {per_state_path}")

    # 4. Feature Group Ablation Study
    ablation_df = run_ablation_study(df, feature_cols, groups, y, seed=args.seed)
    ablation_path = os.path.join(MODELS_DIR, "ablation_results.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print(f"Saved ablation results to: {ablation_path}")
    print("\n" + "=" * 75)
    print("FEATURE ABLATION STUDY RESULTS:")
    print("=" * 75)
    print(ablation_df.round(4).to_string(index=False))

    # 5. Fit Final Production Model on all rows
    print(f"\nRefitting best model ({best_model_name}) on full dataset ({len(df)} records)...")
    final_pipeline = build_pipeline(best_model_name, seed=args.seed)
    final_pipeline.fit(X, y)

    # 6. Uncertainty Estimation: Empirical Residual Quantiles (80% Prediction Interval)
    residuals = y.values - best_preds
    delta_low = float(np.quantile(residuals, 0.10))
    delta_high = float(np.quantile(residuals, 0.90))
    print(f"Uncertainty interval bounds (10th/90th percentile of residuals): [{delta_low:.4f}, +{delta_high:.4f}]")

    # Save Best Model Bundle with Pipeline + Uncertainty
    best_model_bundle = {
        "model": final_pipeline,
        "model_name": best_model_name,
        "evaluation_metrics": logo_results[best_model_name],
        "uncertainty": {
            "lower_delta": delta_low,
            "upper_delta": delta_high,
            "interval_confidence": 0.80
        }
    }
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model_bundle, best_model_path)
    print(f"Saved best production pipeline to: {best_model_path}")

    # Save active feature list
    feature_cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved active feature list to: {feature_cols_path}")

    # Save empirical tertile thresholds derived from training target
    tertiles = y.quantile([1 / 3, 2 / 3]).tolist()
    thresholds = {
        "low_medium_boundary": round(tertiles[0], 4),
        "medium_high_boundary": round(tertiles[1], 4),
        "derived_from": "empirically observed deprivation_score tertiles",
        "note": "Low: score <= low_medium_boundary; Medium: between boundaries; High: score > medium_high_boundary."
    }
    thresholds_path = os.path.join(MODELS_DIR, "category_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved category thresholds to: {thresholds_path} (Low <= {tertiles[0]:.4f} < Medium <= {tertiles[1]:.4f} < High)")

    # Log to database
    log_results_to_db(logo_results, trained_on=data_mode)


if __name__ == "__main__":
    main()
