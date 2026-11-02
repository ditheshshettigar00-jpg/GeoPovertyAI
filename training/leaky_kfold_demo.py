"""
leaky_kfold_demo.py
-------------------
DEMONSTRATION OF SPATIAL DATA LEAKAGE (TOBLER'S FIRST LAW).

WARNING:
This script illustrates why geography-aware validation is essential in geospatial AI.
In spatial datasets, adjacent administrative districts share geographic,
climatic, and socioeconomic features. Randomly splitting districts allows
the model to train on a district and test on its immediate neighbor,
artificially inflating performance.

Metrics are loaded directly from models/leaky_random_kfold_results.csv and
models/model_comparison_results.csv rather than using hardcoded values.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR


def print_leakage_comparison():
    leaky_path = os.path.join(MODELS_DIR, "leaky_random_kfold_results.csv")
    honest_path = os.path.join(MODELS_DIR, "model_comparison_results.csv")

    if not os.path.exists(leaky_path) or not os.path.exists(honest_path):
        print("Error: models/leaky_random_kfold_results.csv or models/model_comparison_results.csv not found.")
        return

    leaky_df = pd.read_csv(leaky_path)
    honest_df = pd.read_csv(honest_path)

    # Pick the best model (RandomForest)
    rf_leaky = leaky_df.iloc[0]
    best_model_name = rf_leaky["model"]
    rf_honest = honest_df[honest_df["model"] == best_model_name].iloc[0]

    leaky_r2 = float(rf_leaky["r2"])
    leaky_mae = float(rf_leaky["mae"])
    leaky_rmse = float(rf_leaky["rmse"])
    leaky_sp = float(rf_leaky["spearman"])

    honest_r2 = float(rf_honest["r2"])
    honest_mae = float(rf_honest["mae"])
    honest_rmse = float(rf_honest["rmse"])
    honest_sp = float(rf_honest["spearman"])

    r2_diff = leaky_r2 - honest_r2
    r2_pct = (r2_diff / abs(honest_r2)) * 100 if honest_r2 != 0 else 0.0

    print("\n" + "=" * 70)
    print("SPATIAL DATA LEAKAGE DEMONSTRATION (TOBLER'S FIRST LAW)")
    print("=" * 70)
    print(f"Model Evaluated: {best_model_name}")
    print("-" * 70)
    print(f"{'Metric':<20} | {'Honest LOSO':<16} | {'Leaky Random 5-Fold':<18} | {'Difference':<12}")
    print("-" * 70)
    print(f"{'R2 Score':<20} | {honest_r2:<16.4f} | {leaky_r2:<18.4f} | {r2_diff:+.4f} ({r2_pct:+.1f}%)")
    print(f"{'MAE':<20} | {honest_mae:<16.4f} | {leaky_mae:<18.4f} | {leaky_mae - honest_mae:+.4f}")
    print(f"{'RMSE':<20} | {honest_rmse:<16.4f} | {leaky_rmse:<18.4f} | {leaky_rmse - honest_rmse:+.4f}")
    print(f"{'Spearman rho':<20} | {honest_sp:<16.4f} | {leaky_sp:<18.4f} | {leaky_sp - honest_sp:+.4f}")
    print("=" * 70)
    print(f"Notice: Random splitting inflates R2 from {honest_r2:.4f} to {leaky_r2:.4f} ({r2_pct:+.1f}%).")
    print("CONCLUSION: Always use Leave-One-State-Out or state-grouped CV in production.")
    print("=" * 70 + "\n")

    return {
        "honest": {"r2": honest_r2, "mae": honest_mae, "rmse": honest_rmse, "spearman": honest_sp},
        "leaky": {"r2": leaky_r2, "mae": leaky_mae, "rmse": leaky_rmse, "spearman": leaky_sp},
    }


if __name__ == "__main__":
    print_leakage_comparison()
