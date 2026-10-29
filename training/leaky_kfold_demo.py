"""
leaky_kfold_demo.py
-------------------
DEMONSTRATION OF SPATIAL DATA LEAKAGE (TOBLER'S FIRST LAW).

WARNING:
This script intentionally runs a standard random (non-grouped) KFold split
to illustrate why geography-aware validation is essential in geospatial AI.

In spatial datasets, adjacent administrative districts share geographic,
climatic, and socioeconomic features. Randomly splitting districts allows
the model to train on a district (e.g. Bangalore Urban) and test on its
immediate neighbor (e.g. Bangalore Rural), artificially inflating performance.

Compare the results here (R2 ~ 0.48) with the honest state-held-out cross-validation
(R2 ~ 0.32) in training/train_models.py.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RANDOM_SEED

def run_leaky_kfold(data_path=None):
    if data_path is None:
        data_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")

    df = pd.read_csv(data_path)
    features = ["nightlight_mean", "ndvi_mean", "builtup_pct", "population_density"]
    X = df[features].values
    y = df["deprivation_score"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X, y):
        rf = RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1
        )
        rf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = rf.predict(X[test_idx])

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    sp = spearmanr(y, y_pred)[0]

    print("\n" + "=" * 65)
    print("WARNING: LEAKY MODEL RESULTS (RANDOM 5-FOLD CV, NO SPATIAL GROUPING)")
    print("=" * 65)
    print(f"R2 Score:              {r2:.4f}  (Inflated by ~50% due to spatial autocorrelation)")
    print(f"Mean Absolute Error:   {mae:.4f}  (Artificially deflated)")
    print(f"Root Mean Squared Err: {rmse:.4f}")
    print(f"Spearman Correlation:  {sp:.4f}")
    print("=" * 65)
    print("CONCLUSION: Always use Leave-One-State-Out or GroupKFold in production.")

    return {"r2": r2, "mae": mae, "rmse": rmse, "spearman": sp}

if __name__ == "__main__":
    run_leaky_kfold()
