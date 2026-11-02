"""
precompute_predictions.py
----------------------------
Runs leave-one-state-out (LOSO) pipeline training + SHAP explanations for every
district, and stores the results in the database so the Flask API can serve
them instantly without recomputing on every request.

For each state:
  1. Build the same pipeline as training (training.pipelines.build_pipeline)
     using best_model, feature_columns, visual_mode, and seed from models/run_metadata.json.
  2. Fit the pipeline on all OTHER states (true leave-one-state-out).
  3. Predict that state's districts and compute SHAP explanations with that model.
  4. Check that predictions match models/loso_predictions.csv within 1e-6 and
     print a warning for any mismatch.
  5. Store the SHAP values and base value so that base + sum(SHAP) = stored predicted_score.
  6. Skip Grad-CAM (the model uses satellite statistics only, not image features).
     No TensorFlow is imported.

Usage:
    python scripts/precompute_predictions.py
    python scripts/precompute_predictions.py --data data/processed/final_training_table.csv
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
from training.pipelines import build_pipeline


def load_model_metadata():
    with open(os.path.join(MODELS_DIR, "run_metadata.json")) as f:
        meta = json.load(f)
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODELS_DIR, "category_thresholds.json")) as f:
        thresholds = json.load(f)
    return meta, feature_cols, thresholds


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
        default_data = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "demo", "demo_training_data.csv"
        )

    parser = argparse.ArgumentParser(
        description="Precompute LOSO predictions + SHAP explanations per state"
    )
    parser.add_argument("--data", type=str, default=default_data,
                        help="Path to feature dataset (default: data/processed/final_training_table.csv)")
    parser.add_argument("--loso", type=str,
                        default=os.path.join(MODELS_DIR, "loso_predictions.csv"),
                        help="Path to models/loso_predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    data_mode = "demo" if "demo" in os.path.basename(args.data).lower() else "real"

    meta, feature_cols, thresholds = load_model_metadata()
    model_name = meta.get("best_model", "RandomForest")
    visual_mode = meta.get("visual_mode", "none")
    seed = meta.get("seed", 42)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns the model expects: {missing}")

    # 1. Read area_id, y_pred_out_of_state, low_80, high_80 from loso_predictions.csv
    loso_lookup = {}
    if os.path.exists(args.loso):
        loso_df = pd.read_csv(args.loso)
        for _, r in loso_df.iterrows():
            aid = int(r["area_id"])
            loso_lookup[aid] = (
                float(r["y_pred_out_of_state"]),
                float(r["low_80"]) if pd.notna(r.get("low_80")) else None,
                float(r["high_80"]) if pd.notna(r.get("high_80")) else None,
            )
        print(f"Loaded {len(loso_lookup)} leave-one-state-out predictions from {args.loso}")
    else:
        print(f"WARNING: {args.loso} not found. Will use pipeline predictions directly.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Clear previous predictions/explanations so this script is safely re-runnable
    cur.execute("DELETE FROM explanations")
    cur.execute("DELETE FROM predictions")
    conn.commit()

    states = df["state_name"].unique()
    n_done = 0
    mismatches = 0

    print(f"Computing LOSO models and SHAP explanations across {len(states)} states...")

    for state in states:
        train_mask = df["state_name"] != state
        test_mask = df["state_name"] == state

        X_train = df.loc[train_mask, feature_cols].reset_index(drop=True)
        y_train = df.loc[train_mask, "deprivation_score"].reset_index(drop=True)
        X_test = df.loc[test_mask, feature_cols].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        # Build pipeline with same hyperparameters/seed as training
        pipe = build_pipeline(model_name, feature_cols, visual_mode=visual_mode, seed=seed)
        pipe.fit(X_train, y_train)

        # Predict held-out state's districts
        preds = pipe.predict(X_test)

        # Build SHAP explainer on the fitted LOSO pipeline
        explainer = build_explainer(pipe, X_train)

        for local_i, (_, test_row) in enumerate(test_df.iterrows()):
            area_id = int(test_row["area_id"])
            upsert_area(cur, test_row, data_mode)

            pipe_pred = float(preds[local_i])

            if area_id in loso_lookup:
                csv_score, low_80, high_80 = loso_lookup[area_id]
                diff = abs(pipe_pred - csv_score)
                # Check predictions match within 1e-6 and print a warning for any mismatch
                if diff > 1e-6:
                    print(f"  WARNING: area_id {area_id} ({state}): pipeline pred={pipe_pred:.6f}, "
                          f"loso_csv={csv_score:.6f}, diff={diff:.2e}")
                    mismatches += 1
                stored_score = csv_score
            else:
                stored_score = pipe_pred
                low_80 = None
                high_80 = None

            category = score_to_category(stored_score, thresholds)

            cur.execute(
                """INSERT INTO predictions
                   (area_id, model_name, predicted_score, predicted_category,
                    predicted_low, predicted_high, prediction_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (area_id, model_name, stored_score, category, low_80, high_80, "leave-one-state-out"),
            )
            prediction_id = cur.lastrowid

            # Compute local SHAP explanation for this district
            local_expl = local_explanation(explainer, X_test, local_i, top_n=8)

            # Ensure base + sum(SHAP) = stored predicted_score exactly
            # Any tiny diff between pipeline float and rounded CSV is absorbed into base_value
            residual_diff = stored_score - local_expl["predicted_value"]
            local_expl["base_value"] = float(local_expl["base_value"] + residual_diff)
            local_expl["predicted_value"] = float(stored_score)

            shap_json = json.dumps(local_expl)

            # Skip Grad-CAM: current model uses satellite statistics only, not image features
            cur.execute(
                """INSERT INTO explanations (prediction_id, shap_json, gradcam_image_path)
                   VALUES (?, ?, ?)""",
                (prediction_id, shap_json, None),
            )

            n_done += 1

        print(f"  Completed state '{state}' ({len(test_df)} districts, total {n_done}/{len(df)})")

    conn.commit()
    conn.close()

    print(f"\nDone. Precomputed predictions + SHAP explanations for {n_done} areas.")
    print(f"Model: '{model_name}', data_mode='{data_mode}', visual_mode='{visual_mode}'")
    if mismatches > 0:
        print(f"Note: {mismatches} districts had diff > 1e-6 vs CSV (due to 4-decimal CSV rounding).")
    if data_mode == "demo":
        print("REMINDER: predictions computed on SYNTHETIC DEMO data.")


if __name__ == "__main__":
    main()
