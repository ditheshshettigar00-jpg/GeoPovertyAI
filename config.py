"""
config.py
---------
Central configuration: file paths and constants shared across modules.
Import this from any script instead of hardcoding paths.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SATELLITE_DIR = os.path.join(DATA_DIR, "satellite")
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "ground_truth")
DEMO_DIR = os.path.join(DATA_DIR, "demo")

# Model directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Database
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DATABASE_DIR, "geopoverty.db")

# Explainability outputs
EXPLAIN_DIR = os.path.join(BASE_DIR, "explainability", "outputs")

# Demo dataset file
DEMO_TRAINING_CSV = os.path.join(DEMO_DIR, "demo_training_data.csv")

# Number of synthetic VGG-16-style feature columns used in DEMO mode.
# NOTE: Real VGG-16 extraction (Module 3) produces a longer vector (e.g. 4096-d
# from fc2, or a smaller pooled vector). DEMO mode uses a small, clearly-labeled
# placeholder (16 columns) so the fusion/training/SHAP pipeline can be built and
# tested before real imagery is processed.
DEMO_VGG_FEATURE_COUNT = 16

# Random seed for reproducibility across all demo/training scripts
RANDOM_SEED = 42

# Ensure required directories exist
for _dir in [RAW_DIR, PROCESSED_DIR, SATELLITE_DIR, GROUND_TRUTH_DIR, DEMO_DIR,
             MODELS_DIR, DATABASE_DIR, EXPLAIN_DIR]:
    os.makedirs(_dir, exist_ok=True)
