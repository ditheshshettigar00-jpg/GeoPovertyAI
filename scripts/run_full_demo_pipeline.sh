#!/bin/bash
# run_full_demo_pipeline.sh
# ----------------------------
# Runs the ENTIRE pipeline end-to-end on SYNTHETIC DEMO data, from database
# setup through to trained models and precomputed explanations. Useful for:
#   - a first smoke-test that everything is installed correctly
#   - preparing for a viva demo without depending on live GEE/OSM access
#
# After this finishes, run:  python backend/app.py
# and open http://127.0.0.1:5000 in a browser.
#
# Usage:
#   bash scripts/run_full_demo_pipeline.sh

set -e  # stop immediately if any step fails

echo "=== [1/6] Initializing database ==="
python3 database/init_db.py

echo ""
echo "=== [2/6] Generating synthetic demo dataset (60 districts) ==="
python3 scripts/generate_demo_data.py --n_areas 60

echo ""
echo "=== [3/6] Generating synthetic demo satellite images ==="
python3 preprocessing/generate_demo_images.py

echo ""
echo "=== [4/6] Loading demo data into database ==="
python3 scripts/load_demo_to_db.py

echo ""
echo "=== [5/6] Training and comparing models (Linear/RandomForest/XGBoost) ==="
python3 training/train_models.py --data data/demo/demo_training_data.csv --test_size 0.3

echo ""
echo "=== [6/6] Precomputing predictions + SHAP + Grad-CAM for every district ==="
python3 scripts/precompute_predictions.py --data data/demo/demo_training_data.csv

echo ""
echo "=================================================================="
echo "DONE. Everything is trained and precomputed on SYNTHETIC DEMO data."
echo "Start the dashboard with:  python3 backend/app.py"
echo "Then open: http://127.0.0.1:5000"
echo "=================================================================="
