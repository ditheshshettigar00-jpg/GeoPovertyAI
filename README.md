# GeoPoverty AI

An explainable, multimodal Geospatial Artificial Intelligence (GeoAI) platform for **area-level** socioeconomic deprivation estimation in India using satellite remote sensing, nocturnal luminosity, settlement grids, and leave-one-state-out machine learning.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-65%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-GeoPovertyAI-181717.svg?logo=github)](https://github.com/ditheshshettigar00-jpg/GeoPovertyAI)

---

## 1. Project Purpose

Socioeconomic census and demographic surveys in developing nations are expensive and infrequent, typically conducted once every 5 to 10 years. In India, regional resource allocation often relies on outdated indicators. 

**GeoPoverty AI** bridges this temporal and spatial gap through Earth Observation (EO) telemetry and machine learning:
- **4 Tabular Satellite Features Only**: The final model strictly uses 4 tabular satellite features: `nightlight_mean`, `ndvi_mean`, `builtup_pct`, and `population_density`, extracted from satellite data for 2019-11-01 to 2021-04-30 (GHSL epoch 2020). Deep visual features (VGG-16) and Grad-CAM are **NOT** part of the final model. Deep visual features (VGG-16) were evaluated only in an exploratory run on the original project dataset (data/exploratory/original_table_with_visual_features.csv; 2023 imagery). In that run, leave-one-state-out R2 was 0.3273 without the visual features (using 7 tabular features: 4 satellite + 3 accessibility columns, not the final 4-feature model) and 0.2879 with them, and visual features alone gave 0.0702 (models/exploratory_visual_ablation/ablation_results.csv). They were not re-evaluated on the final 2019-21 data, and the final model does not use them. The accessibility columns in the original dataset (road density, school and hospital distance) could not be traced to an extraction script and were almost perfectly correlated with each other (|r| about 0.96); removing the accessibility columns changed R2 by about 0.003 (0.288 to 0.291) when the visual features were present, so they were not used. An OSM extraction script is included for future work but was not run nationally.
- **Leakage-Free Spatial Validation**: Employs rigorous **Leave-One-State-Out (LOSO)** cross-validation and repeated state-grouped 5-fold CV to prevent spatial autocorrelation leakage (Tobler's First Law), testing models strictly on entire unseen administrative states. Preprocessing imputation is handled via **training-fold median inside the pipeline**.
- **Explainable AI (XAI)**: Provides game-theoretic additive TreeSHAP values for every prediction, decomposing displayed **out-of-state estimates** into intuitive directional contributions.
- **Decision Support Tools**: Includes an interactive Leaflet choropleth map, live What-If simulation with training distribution bounds checking, spatial hotspot clustering using **Haversine DBSCAN with 150 km radius** (`min_samples=3`), and automated district report generation.

> **Scope & Ethical Boundary:** This platform estimates deprivation at the aggregate administrative district level against survey-based benchmarks. It does **not** track, identify, or profile individual households or persons. Scores reflect statistical associations with satellite observations, not absolute measures or causal determinants.

---

## 2. Repository Architecture & Directory Structure

```text
GeoPovertyAI/
├── backend/                  # Flask REST API, routing, security & services
│   ├── api/                  # Endpoints: auth, prediction, explainability, reports, hotspots
│   ├── services/             # Database service, ML model service, DBSCAN hotspot service
│   └── app.py                # Application factory, session security, CORS configuration
├── config.py                 # Central configuration: paths, DBSCAN hyperparameters, secrets
├── database/                 # Database layer
│   ├── schema.sql            # SQLite schema (areas, predictions, explanations, users)
│   └── init_db.py            # Migration and database initialization script
├── data/                     # Data assets (raw boundaries, processed tables, ground truth)
│   ├── ground_truth/         # NFHS-5 survey ground truth & score_sensitivity.md
│   ├── processed/            # Final training table, district matching reports, data_audit.md
│   ├── raw/                  # Administrative boundaries (FAO GAUL 2015 GeoJSON)
│   └── satellite/            # Sentinel-2 optical RGB patches
├── documentation/            # Technical specifications, architecture, and reports
│   ├── METHODS.md            # Detailed methodology, formulas, matching logic & limitations
│   ├── RESULTS.md            # Generated benchmark and evaluation tables
│   ├── PROJECT_PROMPT.md     # Project specification & phase prompts
│   ├── SYSTEM_ARCHITECTURE.md# Architecture diagrams and system component specifications
│   └── screenshots/          # Application walkthrough captures
├── explainability/           # Explainable AI framework
│   ├── shap_explain.py       # Scikit-Learn Pipeline TreeExplainer & local SHAP attributions
│   └── gradcam.py            # Explored extension: reference Grad-CAM adapter for image models (deactivated for final tabular model)
├── frontend/                 # Web dashboard interface
│   ├── static/               # CSS styles, vanilla JavaScript dashboard scripts, icons
│   └── templates/            # Jinja2 HTML templates (overview, district, compare, whatif)
├── geospatial/               # Remote sensing data acquisition via Google Earth Engine
│   ├── extract_geospatial_features.py # Earth Engine Sentinel-2, VIIRS & GHSL extraction
│   └── download_satellite_patches.py  # Local Sentinel-2 patch retrieval
├── models/                   # Serialized model bundles and validation artifacts
│   ├── best_model.pkl        # Fitted Pipeline bundle (RandomForest, preprocessing, interval)
│   ├── loso_predictions.csv  # District-level out-of-state predictions and 80% error bounds
│   ├── model_comparison_results.csv # Leave-one-state-out model comparison metrics
│   ├── repeated_groupkfold_results.csv # Repeated state-grouped 5-fold CV metrics
│   ├── ablation_results.csv  # Feature ablation evaluation table
│   ├── per_state_errors.csv  # District error breakdowns per state
│   └── leaky_random_kfold_results.csv # Demonstration of spatial leakage inflation
├── notebooks/                # Jupyter exploration and guide tasks
│   └── guide_tasks.ipynb     # State-grouped split modeling demonstration
├── preprocessing/            # Preprocessing utilities
│   ├── name_keys.py          # State-contained canonical district key matching
│   └── feature_fusion.py     # Multimodal table merger and data cleaning
├── scripts/                  # Command-line operational scripts
│   ├── create_admin.py       # Secure interactive admin user creation
│   ├── generate_results_doc.py # Dynamic results documentation generator
│   ├── precompute_predictions.py # Offline LOSO prediction & SHAP precomputation
│   └── refill_missing_features.py # Secondary Earth Engine extraction for large districts
├── tests/                    # Automated testing suite (65 passing pytest tests)
│   ├── conftest.py           # Shared test fixtures & test client setup
│   ├── test_pipeline.py      # Core data pipelines, API endpoints, authentication
│   ├── test_phase2.py        # Spatial cross-validation, pipelines, uncertainty
│   ├── test_phase3.py        # Additive SHAP, Grad-CAM deactivation, satellite safety
│   └── test_security_hotspots.py # CORS, cookie security, rate limiting, DBSCAN
├── ui_prototype/             # Standalone UI wireframes and visual prototypes
├── requirements.txt          # Pinned Python package dependencies from pip freeze
└── README.md                 # Project documentation and quick-start guide
```

---

## 3. Installation & Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit recommended)
- Git

### Clone the Repository
```bash
git clone https://github.com/ditheshshettigar00-jpg/GeoPovertyAI.git
cd GeoPovertyAI
```

### Virtual Environment Setup
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Security & Environment Configuration
Set the Flask environment mode. For development and testing:
```bash
# Windows PowerShell:
$env:FLASK_ENV = "development"

# Linux / macOS:
export FLASK_ENV=development
```
*Note: In production environments, omit `FLASK_ENV=development` and set a strong, random key via `FLASK_SECRET_KEY`.*

### Database Initialization & Admin Setup
```bash
# 1. Initialize SQLite schema and apply migrations
python database/init_db.py

# 2. Interactively create the administrative user (no hardcoded credentials)
python scripts/create_admin.py
```

### Start the Web Application
```bash
python backend/app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 4. How to Run the Pipeline in Order

To reproduce the end-to-end data processing, spatial modeling, and precomputations:

```bash
# Step 1: Precompute Leave-One-State-Out predictions and SHAP explanations
python scripts/precompute_predictions.py

# Step 2: (Optional) Compare spatial leakage vs. honest state holdout
python training/leaky_kfold_demo.py

# Step 3: Automatically regenerate documentation benchmark tables
python scripts/generate_results_doc.py

# Step 4: Run the complete test suite
pytest -v
```

---

## 5. Empirical Results & Benchmark Tables

<!-- RESULTS START -->

### Model Evaluation and Benchmark Results

> **Dataset Scope:** Evaluated on **564 districts** across **31 states** using **4 tabular satellite/geospatial features**. Best-performing model: **RandomForest**.

#### 1. Leave-One-State-Out (LOSO) Validation Results
*Primary benchmark: each prediction is made for an entire administrative state the model never encountered during training.*

| Model | R² | MAE | RMSE | Spearman ρ | Cat. Accuracy | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy | -0.0565 | 0.1142 | 0.1371 | -0.6972 | 31.9% | 0.9% |
| Ridge | 0.2700 | 0.0914 | 0.1140 | 0.5542 | 48.6% | 32.7% |
| RandomForest | 0.3360 | 0.0883 | 0.1087 | 0.5745 | 54.6% | 46.0% |
| GradientBoosting | 0.2951 | 0.0912 | 0.1120 | 0.5539 | 52.7% | 45.1% |
| XGBoost | 0.3074 | 0.0905 | 0.1110 | 0.5643 | 52.7% | 46.9% |

#### 2. Repeated State-Grouped 5-Fold Cross-Validation
*Secondary benchmark: mean and standard deviation over repeated 5-fold cross-validation partitioned by administrative state.*

| Model | R² (mean ± std) | MAE (mean ± std) | RMSE (mean ± std) | Spearman ρ | Cat. Accuracy | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy | -0.0529 ± 0.0359 | 0.1139 ± 0.0019 | 0.1369 ± 0.0023 | -0.2812 | 32.3% | 17.5% |
| Ridge | 0.2500 ± 0.0320 | 0.0930 ± 0.0020 | 0.1155 ± 0.0025 | 0.5268 | 47.6% | 30.6% |
| RandomForest | 0.2984 ± 0.0216 | 0.0910 ± 0.0011 | 0.1117 ± 0.0017 | 0.5450 | 52.0% | 42.5% |
| GradientBoosting | 0.2662 ± 0.0217 | 0.0930 ± 0.0013 | 0.1143 ± 0.0017 | 0.5337 | 51.1% | 40.9% |
| XGBoost | 0.2806 ± 0.0197 | 0.0923 ± 0.0012 | 0.1132 ± 0.0015 | 0.5461 | 52.1% | 41.9% |

#### 3. Baseline Comparison (Predicting Training Mean)
*Comparison against the Dummy regressor that always predicts the training mean of other states.*

| Model | MAE | MAE vs Baseline (%) | Beats Baseline? |
| --- | --- | --- | --- |
| Dummy | 0.1142 | +0.0% | No |
| Ridge | 0.0914 | -20.0% | Yes |
| RandomForest | 0.0883 | -22.7% | Yes |
| GradientBoosting | 0.0912 | -20.1% | Yes |
| XGBoost | 0.0905 | -20.7% | Yes |

#### 4. Feature Ablation Study
*Systematic feature omission on leave-one-state-out evaluation to determine individual indicator contribution.*

| Feature Set | Features | R² | MAE | RMSE | Spearman ρ | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| all features | 4 | 0.3360 | 0.0883 | 0.1087 | 0.5745 | 46.0% |
| tabular only | 4 | 0.3360 | 0.0883 | 0.1087 | 0.5745 | 46.0% |
| without nightlight | 3 | 0.2894 | 0.0911 | 0.1125 | 0.5430 | 38.0% |
| without vegetation | 3 | 0.1567 | 0.0988 | 0.1225 | 0.4032 | 37.2% |
| without built_up | 3 | 0.2529 | 0.0939 | 0.1153 | 0.4978 | 39.8% |
| without population | 3 | 0.1854 | 0.0976 | 0.1204 | 0.4188 | 28.3% |

#### 5. Five States with Highest Prediction Error
*States where leave-one-state-out predictions deviate most from survey ground truth (ranked by MAE descending).*

| State Name | Districts (n) | MAE | Bias (Mean Error) | RMSE |
| --- | --- | --- | --- | --- |
| Goa | 2 | 0.1769 | +0.1769 | 0.1771 |
| Tripura | 4 | 0.1608 | -0.1608 | 0.1719 |
| Meghalaya | 6 | 0.1517 | -0.1517 | 0.1724 |
| Jharkhand | 22 | 0.1464 | -0.1389 | 0.1680 |
| Mizoram | 8 | 0.1280 | +0.1142 | 0.1374 |

#### 6. Spatial Leakage: Leaky Random Splits vs. Honest State Holdout
*Demonstration of spatial autocorrelation (Tobler's First Law): random district splits inflate R² by **+0.1268** (+37.7%), producing overly optimistic performance.*

| Validation Scheme | Model | R² | MAE | RMSE | Spearman ρ | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Leave-One-State-Out (Honest) | RandomForest | 0.3360 | 0.0883 | 0.1087 | 0.5745 | Strict state holdout; model never saw test states |
| Random 5-Fold CV (Leaky) | RandomForest | 0.4628 | 0.0779 | 0.0978 | 0.6694 | Random folds ignore spatial autocorrelation (Tobler's First Law) |

<!-- RESULTS END -->

---

## 6. Limitations & Scientific Disclaimers

1. **Moderate Predictive Accuracy & Mean Compression:** The model achieves a Leave-One-State-Out $R^2$ of **0.3360** (MAE **0.0883**). Predictions compress toward the mean: highly deprived districts are under-estimated, while the least deprived districts are over-estimated.
2. **Tabular Satellite Features Only (4 Features):** The final model uses 4 features only: `nightlight_mean`, `ndvi_mean`, `builtup_pct`, and `population_density`, extracted from satellite data for 2019-11-01 to 2021-04-30 (GHSL epoch 2020). Deep visual features (VGG-16) and Grad-CAM are NOT part of the final model. Deep visual features (VGG-16) were evaluated only in an exploratory run on the original project dataset (data/exploratory/original_table_with_visual_features.csv; 2023 imagery). In that run, leave-one-state-out R2 was 0.3273 without the visual features (using 7 tabular features: 4 satellite + 3 accessibility columns, not the final 4-feature model) and 0.2879 with them, and visual features alone gave 0.0702 (models/exploratory_visual_ablation/ablation_results.csv). They were not re-evaluated on the final 2019-21 data, and the final model does not use them. The accessibility columns in the original dataset (road density, school and hospital distance) could not be traced to an extraction script and were almost perfectly correlated with each other (|r| about 0.96); removing the accessibility columns changed R2 by about 0.003 (0.288 to 0.291) when the visual features were present, so they were not used. An OSM extraction script is included for future work but was not run nationally.
3. **Historical Boundary Coverage:** Administrative polygons are derived from FAO GAUL 2015. Consequently, 564 of the 705 NFHS-5 districts are represented; newly created administrative units or heavily reorganized territories (Jammu & Kashmir, Ladakh, and NCT Delhi) are excluded.
4. **Thin States & Small Sample Variance:** States with fewer than 3 districts (e.g., Goa with 2, Dadra & Nagar Haveli and Daman & Diu with 2, and Andaman & Nicobar Islands with 1) exhibit noisy Leave-One-State-Out error estimates.
5. **Composite Survey Target:** The ground truth label is an artificial mathematical composite of five survey indicators, not a direct measurement of consumption poverty or household wealth.
6. **Correlation vs. Causation:** SHAP feature attributions describe model behavior, not real-world causal mechanisms. Higher nighttime lights correlate with lower deprivation in the model, but artificially installing lights will not cause poverty reduction.
7. **Simulation Scope:** The What-If simulator reflects model behavior under hypothetical input perturbations, not the socioeconomic equilibrium outcomes of real-world policy interventions.
8. **Inference Discrepancy:** District scores displayed on district profile pages are out-of-state Leave-One-State-Out estimates (the model never saw that state during training), whereas live What-If scenarios use the final model refitted on all districts.

For detailed technical specifications on data sources, dates, formulas, and matching rules, see [`documentation/METHODS.md`](documentation/METHODS.md).

---

## 7. Testing & Quality Assurance

The test suite validates data integrity, spatial cross-validation, additive SHAP guarantees, security configurations, and API endpoints:

```bash
pytest -v
```

**Test Results:**
- **65 passed** in `61.01s` (100% pass rate) across:
  - `tests/test_phase2.py`: Spatial cross-validation, IQR clipping, pipeline transformers (10 tests)
  - `tests/test_phase3.py`: Additive SHAP equality ($\le 10^{-4}$), Grad-CAM deactivation, satellite image safety (7 tests)
  - `tests/test_pipeline.py`: Feature fusion, API routes, authentication, district matching (26 tests)
  - `tests/test_security_hotspots.py`: CORS origin validation, session cookie flags, rate limiting, DBSCAN clustering (22 tests)
