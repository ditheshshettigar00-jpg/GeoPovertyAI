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

# --------------------------------------------------------------------------
# Real-data extraction settings (Phase 1)
# NFHS-5 fieldwork ran 2019-2021, so satellite features must come from the same
# period. VIIRS monthly and Sentinel-2 SR Harmonized both cover this window.
# --------------------------------------------------------------------------
GEE_START_DATE = "2019-11-01"
GEE_END_DATE = "2021-04-30"
GHSL_EPOCH_YEAR = 2020          # GHSL epoch closest to the survey period
GHSL_CELL_AREA_M2 = 10_000.0    # one GHSL cell is 100 m x 100 m

# District-name matching between GAUL boundaries and NFHS-5 ground truth
DISTRICT_ALIASES_CSV = os.path.join(GROUND_TRUTH_DIR, "district_name_aliases.csv")
FUZZY_MATCH_CUTOFF = 0.90

# Ensure required directories exist
for _dir in [RAW_DIR, PROCESSED_DIR, SATELLITE_DIR, GROUND_TRUTH_DIR, DEMO_DIR,
             MODELS_DIR, DATABASE_DIR, EXPLAIN_DIR]:
    os.makedirs(_dir, exist_ok=True)
