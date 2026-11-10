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

# --------------------------------------------------------------------------
# Hotspot spatial clustering constants (DBSCAN with Haversine metric)
# --------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
HOTSPOT_DISTANCE_KM = 150.0
HOTSPOT_EPS = HOTSPOT_DISTANCE_KM / EARTH_RADIUS_KM  # 150 / 6371 radians (~0.02354)
HOTSPOT_MIN_SAMPLES = 3

# Ensure required directories exist
for _dir in [RAW_DIR, PROCESSED_DIR, SATELLITE_DIR, GROUND_TRUTH_DIR, DEMO_DIR,
             MODELS_DIR, DATABASE_DIR, EXPLAIN_DIR]:
    os.makedirs(_dir, exist_ok=True)

# --------------------------------------------------------------------------
# Scenario Explorer presets (proxies for satellite-derived feature changes)
# NOTE: These are sensitivity proxies, never policy interventions.
# --------------------------------------------------------------------------
SCENARIO_DISCLAIMER_NOTE = (
    "This shows how the model's estimate responds to changes in satellite-derived features. "
    "It is not a prediction of the effect of a policy or programme. "
    "Schooling, health, nutrition and infrastructure are not in the model. "
    "This is a model sensitivity exploration, not a causal simulation."
)

SCENARIO_PRESETS = {
    "electrification_125": {
        "id": "electrification_125",
        "name": "Electrification proxy (x1.25)",
        "feature": "nightlight_mean",
        "op": "multiply",
        "value": 1.25,
        "description": "Electrification proxy: scales night-light radiance by 1.25x.",
    },
    "electrification_150": {
        "id": "electrification_150",
        "name": "Electrification proxy (x1.5)",
        "feature": "nightlight_mean",
        "op": "multiply",
        "value": 1.5,
        "description": "Electrification proxy: scales night-light radiance by 1.5x.",
    },
    "urbanisation_1pp": {
        "id": "urbanisation_1pp",
        "name": "Urbanisation proxy (+1 pp)",
        "feature": "builtup_pct",
        "op": "add",
        "value": 1.0,
        "description": "Urbanisation proxy: increases built-up land share by +1.0 percentage point.",
    },
    "urbanisation_2pp": {
        "id": "urbanisation_2pp",
        "name": "Urbanisation proxy (+2 pp)",
        "feature": "builtup_pct",
        "op": "add",
        "value": 2.0,
        "description": "Urbanisation proxy: increases built-up land share by +2.0 percentage points.",
    },
    "greening_005": {
        "id": "greening_005",
        "name": "Greening proxy (+0.05)",
        "feature": "ndvi_mean",
        "op": "add",
        "value": 0.05,
        "description": "Greening proxy: increases NDVI vegetation index by +0.05.",
    },
    "combined": {
        "id": "combined",
        "name": "Combined growth proxy",
        "features": {
            "nightlight_mean": {"op": "multiply", "value": 1.25},
            "builtup_pct": {"op": "add", "value": 1.0},
            "ndvi_mean": {"op": "add", "value": 0.05},
        },
        "description": "Combined proxy: night-light x1.25, built-up +1.0 pp, and NDVI +0.05.",
    },
}

