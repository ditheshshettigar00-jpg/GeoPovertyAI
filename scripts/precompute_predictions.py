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
from config import DB_PATH, MODELS_DIR, SATELLITE_DIR, EXPLAIN_DIR
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


def try_generate_gradcam(area_id: int, vgg_feature_weights: np.ndarray) -> str | None:
    """
    Attempts to generate a Grad-CAM overlay for this area's satellite image,
    IF an image file for it exists on disk. Returns the saved path, or None
    if no image is available (this is expected/normal until Module 2/3's
    real images -- or Module 1's demo images -- have been generated for
    this area; it is not treated as an error).

    IMPORTANT: SHAP-derived feature_weights are only meaningful if they
    came from REAL VGG-16 features (512-d, matching vgg16_feature_extractor.py's
    pooling='avg' output). Module 1's demo dataset uses a much shorter
    placeholder vector (DEMO_VGG_FEATURE_COUNT=16) that is NOT actual VGG-16
    output -- weighting Grad-CAM by those numbers would be meaningless,
    since they don't correspond to the real network's channels at all. In
    that case we fall back to unweighted Grad-CAM (still demonstrates the
    mechanism correctly, just without a SHAP-informed weighting).
    """
    from preprocessing.vgg16_feature_extractor import FEATURE_DIM as REAL_VGG_DIM

    candidates = [
        os.path.join(SATELLITE_DIR, f"district_{area_id:03d}_rgb.png"),
        os.path.join(SATELLITE_DIR, f"district_{area_id:03d}_rgb.tif"),
        os.path.join(SATELLITE_DIR, "demo", f"district_{area_id:03d}_DEMO.png"),
    ]
    image_path = next((p for p in candidates if os.path.exists(p)), None)
    if image_path is None:
        return None

    if vgg_feature_weights is not None and len(vgg_feature_weights) != REAL_VGG_DIM:
        # Demo placeholder features don't match real VGG-16's dimensionality --
        # fall back to unweighted Grad-CAM rather than applying meaningless weights.
        vgg_feature_weights = None

    try:
        from tensorflow.keras.applications import VGG16
        from explainability.gradcam import generate_gradcam_for_image

        # NOTE: uses weights=None here only because this sandbox cannot reach
        # storage.googleapis.com to download real ImageNet weights (see
        # Module 3 notes). On your machine, swap in weights='imagenet' for a
        # meaningful heatmap -- see vgg16_feature_extractor.build_feature_extractor().
        vgg_model = VGG16(weights=None, include_top=False, pooling="avg", input_shape=(224, 224, 3))
        vgg_model.trainable = False

        out_path = os.path.join(EXPLAIN_DIR, f"gradcam_area_{area_id:03d}.png")
        generate_gradcam_for_image(vgg_model, image_path, out_path, feature_weights=vgg_feature_weights)
        return out_path
    except Exception as e:
        print(f"  WARNING: Grad-CAM generation failed for area_id={area_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Precompute predictions + SHAP + Grad-CAM for every area")
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    data_mode = "demo" if "demo" in os.path.basename(args.data).lower() else "real"

    model, model_name, feature_cols, thresholds = load_model_artifacts()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns the model expects: {missing}")

    X = df[feature_cols]
    explainer = build_explainer(model, X)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Clear previous predictions/explanations for this data_mode so this script is safely re-runnable
    cur.execute("""DELETE FROM explanations WHERE prediction_id IN
                   (SELECT prediction_id FROM predictions WHERE model_name = ?)""", (model_name,))
    cur.execute("DELETE FROM predictions WHERE model_name = ?", (model_name,))
    conn.commit()

    n_done = 0
    gradcam_dim_warning_shown = False
    for i, row in df.iterrows():
        area_id = int(row["area_id"])
        upsert_area(cur, row, data_mode)

        score = float(model.predict(X.iloc[[i]])[0])
        category = score_to_category(score, thresholds)

        cur.execute(
            """INSERT INTO predictions (area_id, model_name, predicted_score, predicted_category)
               VALUES (?, ?, ?, ?)""",
            (area_id, model_name, score, category),
        )
        prediction_id = cur.lastrowid

        local_expl = local_explanation(explainer, X, i, top_n=8)
        shap_json = json.dumps(local_expl)

        vgg_weights = np.array([
            next((c["shap_contribution"] for c in local_expl["top_contributions"] if c["feature"] == f), 0.0)
            for f in feature_cols if f.startswith("vgg_feat_")
        ], dtype=np.float32)

        if data_mode == "demo" and not gradcam_dim_warning_shown:
            from preprocessing.vgg16_feature_extractor import FEATURE_DIM as REAL_VGG_DIM
            if len(vgg_weights) != REAL_VGG_DIM:
                print(f"  NOTE: demo dataset's vgg_feat_* columns are {len(vgg_weights)}-d "
                      f"placeholders, not real {REAL_VGG_DIM}-d VGG-16 output -- Grad-CAM will "
                      f"use unweighted saliency for demo areas instead of SHAP-weighted saliency.")
                gradcam_dim_warning_shown = True

        gradcam_path = try_generate_gradcam(area_id, vgg_weights if len(vgg_weights) > 0 else None)

        cur.execute(
            """INSERT INTO explanations (prediction_id, shap_json, gradcam_image_path)
               VALUES (?, ?, ?)""",
            (prediction_id, shap_json, gradcam_path),
        )

        n_done += 1
        if n_done % 10 == 0:
            print(f"  Processed {n_done}/{len(df)} areas ...")

    conn.commit()
    conn.close()
    print(f"\nDone. Precomputed predictions + SHAP explanations for {n_done} areas "
          f"using model '{model_name}' (data_mode='{data_mode}').")
    if data_mode == "demo":
        print("REMINDER: predictions computed on SYNTHETIC DEMO data.")


if __name__ == "__main__":
    main()
