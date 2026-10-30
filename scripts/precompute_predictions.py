"""
precompute_predictions.py
----------------------------
Runs the trained model (Module 5) + SHAP (Module 6) + Grad-CAM (Module 3)
for every area in the fused dataset, and stores the results in the
database so the Flask API (Module 7) can serve them instantly without
recomputing on every request.

WHY PRECOMPUTE rather than predict live per API request: SHAP and Grad-CAM
are relatively expensive to compute. For a dashboard demo with a fixed,
known set of areas, precomputing once and serving cached results is the
practical choice for a student-laptop deployment (this is documented as
an explicit, reasonable design choice, not a limitation being hidden).
The one thing that DOES need to run live is the what-if/scenario endpoint
(Module 9), since it needs a fresh prediction for a user-modified feature
vector -- that's a single fast model.predict() call, not SHAP/Grad-CAM,
so it's cheap enough to run per-request.

Usage:
    python scripts/precompute_predictions.py --data data/demo/demo_training_data.csv
"""

import argparse
import json
import os
import sqlite3
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, MODELS_DIR, PROCESSED_DIR
from explainability.shap_explain import build_explainer, local_explanation
from training.train_models import get_feature_columns


def load_model_artifacts():
    model_bundle = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODELS_DIR, "category_thresholds.json")) as f:
        thresholds = json.load(f)
    return model_bundle["model"], model_bundle["model_name"], feature_cols, thresholds


def score_to_category(score: float, thresholds: dict) -> str:
    if score <= thresholds["low_medium_boundary"]:
        return "Low"
    elif score <= thresholds["medium_high_boundary"]:
        return "Medium"
    else:
        return "High"


def upsert_area(cur, row, data_mode):
    cur.execute("SELECT area_id FROM areas WHERE area_id = ?", (int(row["area_id"]),))
    exists = cur.fetchone()
    if not exists:
        cur.execute(
            """INSERT INTO areas (area_id, area_name, state_name, centroid_lat, centroid_lon, data_mode)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (int(row["area_id"]), row.get("area_name", f"Area_{int(row['area_id'])}"),
             row.get("state_name", "Unknown"),
             float(row.get("latitude", 0.0)), float(row.get("longitude", 0.0)), data_mode),
        )
    else:
        cur.execute(
            """UPDATE areas SET area_name = ?, state_name = ?, centroid_lat = ?, centroid_lon = ?, data_mode = ?
               WHERE area_id = ?""",
            (row.get("area_name", f"Area_{int(row['area_id'])}"),
             row.get("state_name", "Unknown"),
             float(row.get("latitude", 0.0)), float(row.get("longitude", 0.0)), data_mode,
             int(row["area_id"])),
        )


def main():
    default_data = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    if not os.path.exists(default_data):
        default_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo", "demo_training_data.csv")

    parser = argparse.ArgumentParser(description="Precompute predictions + SHAP explanations using out-of-state LOSO scores")
    parser.add_argument("--data", type=str, default=default_data,
                        help="Path to feature dataset (default: data/processed/final_training_table.csv)")
    parser.add_argument("--loso", type=str, default=os.path.join(MODELS_DIR, "loso_predictions.csv"),
                        help="Path to models/loso_predictions.csv (default: models/loso_predictions.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    data_mode = "demo" if "demo" in os.path.basename(args.data).lower() else "real"

    model, model_name, feature_cols, thresholds = load_model_artifacts()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns the model expects: {missing}")

    X = df[feature_cols]
    explainer = build_explainer(model, X)

    # 1. Read area_id, y_pred_out_of_state, low_80, high_80 from loso_predictions.csv
    loso_lookup = {}
    if os.path.exists(args.loso):
        loso_df = pd.read_csv(args.loso)
        for _, r in loso_df.iterrows():
            aid = int(r["area_id"])
            loso_lookup[aid] = (
                float(r["y_pred_out_of_state"]),
                float(r["low_80"]) if pd.notna(r["low_80"]) else None,
                float(r["high_80"]) if pd.notna(r["high_80"]) else None,
            )
        print(f"Loaded {len(loso_lookup)} leave-one-state-out predictions from {args.loso}")
    else:
        print(f"WARNING: {args.loso} not found. Falling back to live model inference.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Clear previous predictions/explanations so this script is safely re-runnable
    cur.execute("DELETE FROM explanations")
    cur.execute("DELETE FROM predictions")
    conn.commit()

    n_done = 0
    for i, row in df.iterrows():
        area_id = int(row["area_id"])
        upsert_area(cur, row, data_mode)

        if area_id in loso_lookup:
            score, low_80, high_80 = loso_lookup[area_id]
        else:
            score = float(model.predict(X.iloc[[i]])[0])
            low_80 = None
            high_80 = None

        category = score_to_category(score, thresholds)

        cur.execute(
            """INSERT INTO predictions (area_id, model_name, predicted_score, predicted_category, predicted_low, predicted_high, prediction_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (area_id, model_name, score, category, low_80, high_80, "leave-one-state-out"),
        )
        prediction_id = cur.lastrowid

        local_expl = local_explanation(explainer, X, i, top_n=8)
        shap_json = json.dumps(local_expl)

        # 4. Skip Grad-CAM in this script, because the model no longer uses images
        cur.execute(
            """INSERT INTO explanations (prediction_id, shap_json, gradcam_image_path)
               VALUES (?, ?, ?)""",
            (prediction_id, shap_json, None),
        )

        n_done += 1
        if n_done % 50 == 0 or n_done == len(df):
            print(f"  Processed {n_done}/{len(df)} areas ...")

    conn.commit()
    conn.close()
    print(f"\nDone. Precomputed predictions + SHAP explanations for {n_done} areas "
          f"using model '{model_name}' (data_mode='{data_mode}').")
    if data_mode == "demo":
        print("REMINDER: predictions computed on SYNTHETIC DEMO data.")
    if data_mode == "demo":
        print("REMINDER: predictions computed on SYNTHETIC DEMO data.")


if __name__ == "__main__":
    main()
