# GeoPoverty AI

An explainable, multimodal Geospatial Artificial Intelligence (GeoAI) platform for **area-level** socioeconomic deprivation estimation in India using satellite remote sensing, nocturnal luminosity, settlement grids, and leave-one-state-out machine learning.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-264%20Passed-brightgreen.svg)](tests/)
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
- **Hidden Gap Detector** (`/hidden-gaps`, `/api/residual-atlas`): Compares the NFHS-5 survey score with the out-of-state satellite estimate; a district is a candidate only if the survey score is outside the 80% interval AND the percentile-rank gap is at least 0.25; screening hint, not proof of hidden poverty.
- **Scenario Explorer** (the What-If page: `/scenario-explorer`, `/whatif`, `/api/scenario`): Preset changes to satellite-derived features, with before/after estimate, 80% range and the message when a change is smaller than the model's typical error; a sensitivity tool, not a prediction of policy effects.
- **District Indicators Panel** (`/district/<id>`, `/api/indicators/<id>`): All 7 satellite indicators and the 5 NFHS-5 component indicators for each district, with percentile and state/India medians.

> **Scope & Ethical Boundary:** This platform estimates deprivation at the aggregate administrative district level against survey-based benchmarks. It does **not** track, identify, or profile individual households or persons. Scores reflect statistical associations with satellite observations, not absolute measures or causal determinants.

---

## 2. Repository Architecture & Directory Structure

```text
GeoPovertyAI/
├── backend/                  # Flask REST API, routing, security & services
│   ├── api/                  # Endpoints: auth, prediction, explainability, reports, hotspots
│   ├── services/             # Application services: database, ML inference, telemetry, residuals
│   │   ├── indicator_service.py # Dynamic extraction of 7 satellite indicators & NFHS survey metrics
│   │   ├── profile_service.py   # Grouped district profile, empirical percentiles & accessibility
│   │   └── residual_service.py  # Hidden Gap Detector (NFHS-5 survey vs. out-of-state satellite residual atlas)
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
│   ├── PROJECT_BLUEPRINT.md  # Master project blueprint, task specifications & milestones
│   ├── PIPELINE_HISTORY.md   # Historical development logs (Phase 1 & Phase 2)
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
├── tests/                    # Automated testing suite (264 passing pytest tests)
│   ├── conftest.py           # Shared test fixtures & test client setup
│   ├── test_access_control.py # Role-based access control route matrix, parameter validation, self/last-admin protection (162 tests)
│   ├── test_area_profile.py  # Grouped district profile schema, percentiles & survey flags (7 tests)
│   ├── test_hidden_gaps.py   # Residual atlas, candidate criteria & screening limits (9 tests)
│   ├── test_indicator_service.py # Real district indicators, NFHS join, PDF tables & fallbacks (9 tests)
│   ├── test_phase2.py        # Spatial cross-validation, pipelines, uncertainty (10 tests)
│   ├── test_phase3.py        # Additive SHAP, Grad-CAM deactivation, satellite safety (7 tests)
│   ├── test_pipeline.py      # Core data pipelines, API endpoints, authentication (26 tests)
│   ├── test_report_page.py   # District report preview route & JSON data endpoints (4 tests)
│   ├── test_scenario_explorer.py # Scenario presets, group aggregation & error text (8 tests)
│   └── test_security_hotspots.py # CORS, cookie security, rate limiting, DBSCAN (22 tests)
├── requirements.txt          # Pinned Python package dependencies from pip freeze
└── README.md                 # Project documentation and quick-start guide
```

---

## 3. Web Pages & REST API Endpoints

### Web Interface Pages
- **`/`** — Home landing page and national overview dashboard.
- **`/prediction`** — Live AI predictor with interactive input sliders and sensitivity exploration.
- **`/map`** — Interactive Leaflet choropleth map of India with district-level deprivation overlays.
- **`/explanation`** — Model explainability center with game-theoretic TreeSHAP feature attributions.
- **`/performance`** — Empirical benchmark results, LOSO cross-validation, and ablation tables.
- **`/district/<int:area_id>`** — District Indicators panel: complete profile, 7 satellite indicators, and NFHS-5 survey components.
- **`/report/<int:area_id>`** — In-browser district executive report preview with PDF download.
- **`/compare`** — Side-by-side comparative analysis between two districts.
- **`/matchmaker`** — Multi-criteria district matching and peer comparison tool.
- **`/hidden-gaps`** — Hidden Gap Detector: residual atlas comparing survey scores vs. out-of-state satellite estimates.
- **`/hotspots`** — Spatial DBSCAN deprivation clusters (Haversine metric, 150 km radius, min 3 districts).
- **`/scenario-explorer`** (also `/whatif`, `/scenarios`) — Scenario Explorer: policy sensitivity exploration under feature shifts.
- **`/data-quality`** — Training data audit, GAUL boundary coverage, and feature range distributions.
- **`/about`** — Project methodology, remote sensing data sources, and scientific disclaimers.
- **`/admin`** — Administrative management portal with user management and audit logs (role-protected).
- **`/login`**, **`/register`** — User registration, login, and session authentication.

### REST API Endpoints
- **`GET /api/areas`** — Retrieve all 564 districts with names, states, centroid coordinates, and predictions.
- **`GET /api/predict/<int:area_id>`** — Predicted deprivation score, category, and 80% error interval for a district.
- **`GET /api/area-profile/<int:area_id>`** — Full district profile including satellite features, percentiles, real accessibility, and NFHS components.
- **`GET /api/indicators/<int:area_id>`** — 7 satellite indicators and 5 NFHS survey components (`?full=1` adds all 104 NFHS indicators).
- **`GET /api/area-features/<int:area_id>`** — Raw 4 model features for a given district.
- **`POST /api/predict-live`** — Live model prediction and real-time TreeSHAP attributions for custom user inputs.
- **`POST /api/scenario`** — Scenario Explorer simulation for an individual district, an entire state, or a hotspot cluster.
- **`GET /api/scenario-presets`** — Standardized sensitivity presets (electrification, urbanisation, greening, combined).
- **`GET /api/scenario-targets`** — Selectable target options (districts, states, and hotspot clusters) for scenarios.
- **`GET /api/residual-atlas`** — Hidden Gap Detector: candidate list filtered by direction (`?direction=hidden|resilient|all`).
- **`GET /api/hotspots`** — Spatial DBSCAN clusters of contiguous high-deprivation districts.
- **`GET /api/explain/<int:area_id>`** — Local TreeSHAP attribution waterfall for a district's out-of-state prediction.
- **`GET /api/map`** — GeoJSON boundaries and deprivation classifications for the Leaflet map.
- **`POST /api/compare`** — Compare multiple districts across features, scores, and survey indicators.
- **`GET /api/similar-districts/<int:area_id>`** — Top 5 nearest neighbor districts in feature space.
- **`GET /api/satellite-image/<int:area_id>`** — Cloud-free Sentinel-2 RGB composite satellite patch image.
- **`GET /api/model-performance`** — Leave-One-State-Out validation metrics and benchmark comparison tables.
- **`GET /api/reports/district-data/<int:area_id>`** — Structured JSON data payload for district executive reports.
- **`GET /api/reports/district/<int:area_id>`** — Generate and download official district PDF report (`?appendix=1` for full NFHS table).
- **`GET /api/data-quality`** — Data audit metrics, boundary coverage, and feature summary statistics.
- **`POST /api/auth/register`**, **`POST /api/auth/login`**, **`POST /api/auth/logout`**, **`GET /api/auth/profile`** — Authentication and user session endpoints.
- **`GET /api/admin/dashboard-stats`**, **`GET /api/admin/users`**, **`GET /api/admin/pending-researchers`**, **`GET /api/admin/audit-logs`** — Administrative management and audit logging endpoints.
- **`PUT /api/admin/users/<int:user_id>/role`**, **`PUT /api/admin/users/<int:user_id>/status`**, **`POST /api/admin/users/<int:user_id>/approve-researcher`**, **`POST /api/admin/users/<int:user_id>/reject-researcher`** — Administrative governance and role moderation endpoints.

---

## 4. Access Control & User Roles

GeoPoverty AI implements a multi-tiered Role-Based Access Control (RBAC) architecture enforced directly at the server level for all HTML pages and REST API routes.

### Role Hierarchy & Responsibilities

1. **`PUBLIC` (Public / Student / Anonymous Browsing):**
   - **No login required:** Anonymous visitors can freely view the Landing Page (`/`), About Page (`/about`), National Choropleth Map (`/map`), and District Drilldown (`/district/<id>`).
   - Public read APIs (`/api/areas`, `/api/map`, `/api/predict/<id>`, `/api/indicators/<id>`, `/api/area-profile/<id>`, `/api/satellite-image/<id>`) are open without authentication to support public data transparency and student exploration.
   - Standard registered users receive the `PUBLIC` role by default.

2. **`RESEARCHER` (Analytical Decision-Support Suite):**
   - Access to advanced modeling, explainability, sensitivity, and reporting tools:
     - Live AI Predictor (`/prediction`, `POST /api/predict-live`)
     - Model Explainability (`/explanation`, `GET /api/explain/<id>`)
     - Performance Benchmarks (`/performance`, `GET /api/model-performance`)
     - District Executive Reports & PDF Dossiers (`/report/<id>`, `GET /api/reports/district/<id>`)
     - Comparative Analysis & Matching (`/compare`, `/matchmaker`, `POST /api/compare`, `GET /api/similar-districts/<id>`)
     - Spatial Clusters & Hotspots (`/hotspots`, `GET /api/hotspots`)
     - Hidden Gap Residual Atlas (`/hidden-gaps`, `GET /api/residual-atlas`)
     - Scenario Explorer Simulator (`/scenario-explorer`, `/whatif`, `/scenarios`, `POST /api/scenario`, `GET /api/scenario-presets`, `GET /api/scenario-targets`)
     - Data Quality Audit (`/data-quality`, `GET /api/data-quality`)
   - **Approval Workflow:** Users requesting researcher access register via `/register` with "Request researcher access" checked. The account is created as `role='PUBLIC'` with `requested_role='RESEARCHER'`. The user can immediately log in with public privileges while their researcher request awaits administrator review.

3. **`ADMIN` (System Governance & Auditing):**
   - Full administrative dashboard (`/admin`) and backend management APIs.
   - Capabilities: Review and approve/reject pending researcher requests, update user roles, toggle account active/inactive status, and inspect immutable security audit logs (`audit_logs` table).
   - **Administrator Security Policy:** Administrative accounts **cannot** be registered through the web UI or public endpoints. They can only be bootstrapped via the interactive command-line utility (`python scripts/create_admin.py`) or promoted by an existing active administrator.
   - **Safety Invariants:** The system enforces server-side safeguards (`_guard_admin_change`):
     - An administrator cannot demote or deactivate their own active account.
     - An administrator cannot demote or deactivate the last remaining active administrator in the system.

### Access Control Matrix

| Route / Endpoint | Method | Anonymous | `PUBLIC` | `RESEARCHER` | `ADMIN` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `/` (Landing Page) | `GET` | Allowed | Allowed | Allowed | Allowed |
| `/about` | `GET` | Allowed | Allowed | Allowed | Allowed |
| `/map` | `GET` | Allowed | Allowed | Allowed | Allowed |
| `/district/<id>` | `GET` | Allowed | Allowed | Allowed | Allowed |
| `/login`, `/register` | `GET` | Allowed | Allowed | Allowed | Allowed |
| `/overview` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/prediction` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/explanation` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/performance` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/report/<id>` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/compare` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/matchmaker` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/hidden-gaps` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/hotspots` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/data-quality` | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/scenario-explorer` (`/scenarios`, `/whatif`) | `GET` | Redirect to Login | 403 Forbidden | Allowed | Allowed |
| `/admin` | `GET` | Redirect to Login | 403 Forbidden | 403 Forbidden | Allowed |
| `/api/areas`, `/api/map` | `GET` | 200 OK | 200 OK | 200 OK | 200 OK |
| `/api/predict/<id>`, `/api/indicators/<id>` | `GET` | 200 OK | 200 OK | 200 OK | 200 OK |
| `/api/area-profile/<id>`, `/api/satellite-image/<id>` | `GET` | 200 OK | 200 OK | 200 OK | 200 OK |
| `/api/explain/<id>`, `/api/model-performance` | `GET` | 401 Unauthorized | 403 Forbidden | 200 OK | 200 OK |
| `/api/scenario-presets`, `/api/scenario-targets` | `GET` | 401 Unauthorized | 403 Forbidden | 200 OK | 200 OK |
| `/api/area-features/<id>`, `/api/hotspots`, `/api/data-quality` | `GET` | 401 Unauthorized | 403 Forbidden | 200 OK | 200 OK |
| `/api/similar-districts/<id>`, `/api/residual-atlas` | `GET` | 401 Unauthorized | 403 Forbidden | 200 OK | 200 OK |
| `/api/reports/district-data/<id>`, `/api/reports/district/<id>` | `GET` | 401 Unauthorized | 403 Forbidden | 200 OK | 200 OK |
| `/api/compare`, `/api/scenario`, `/api/predict-live` | `POST` | 401 Unauthorized | 403 Forbidden | 200 OK | 200 OK |
| `/api/auth/register`, `/api/auth/login` | `POST` | 200 OK | 200 OK | 200 OK | 200 OK |
| `/api/auth/logout`, `/api/auth/profile` | `POST`/`GET` | 401 Unauthorized | 200 OK | 200 OK | 200 OK |
| `/api/admin/*` (All admin endpoints) | `GET`/`POST`/`PUT` | 401 Unauthorized | 403 Forbidden | 403 Forbidden | 200 OK |

### HTTP Status Code & Redirection Behavior

- **Unauthenticated Page Access:** When an unauthenticated visitor requests a protected page, the server issues an HTTP 302 redirect to `/login?next=<encoded_path>`. The `safe_next` validator ensures only relative same-origin paths are honored, blocking open-redirect exploits.
- **Unauthenticated API Access:** When an unauthenticated client requests a protected API endpoint, the server returns structured JSON `{"error": "Unauthorized"}` with HTTP status **401 Unauthorized**.
- **Forbidden Page Access:** When an authenticated user lacks the required role for a page, the server returns a styled `403.html` template with HTTP status **403 Forbidden**.
- **Forbidden API Access:** When an authenticated user lacks the required role for an API, the server returns structured JSON `{"error": "Forbidden: insufficient permissions"}` with HTTP status **403 Forbidden**.

---

## 5. Quick Start (Fresh Clone)

### Important Database Note
> **Database Exclusion:** The SQLite database file (`geopoverty.db`) and its write-ahead logs (`*.db-wal`, `*.db-shm`) are excluded from Git version control. On any fresh clone, you **must** run `python database/init_db.py` to create the database schema, followed by `python scripts/precompute_predictions.py` to populate district features, out-of-state Leave-One-State-Out predictions, and additive SHAP explanations before running the web application or pytest suite.

### Setup for Windows PowerShell

```powershell
# 1. Clone the repository and enter the directory
git clone https://github.com/ditheshshettigar00-jpg/GeoPovertyAI.git
cd GeoPovertyAI

# 2. Create and activate a Python virtual environment (Python 3.10, 3.11, or 3.12)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Initialize SQLite database schema and migrations
python database/init_db.py

# 5. Precompute out-of-state Leave-One-State-Out predictions & SHAP explanations
python scripts/precompute_predictions.py

# 6. Interactively create the initial administrator user (no hardcoded passwords)
python scripts/create_admin.py

# 7. Set development environment variable
$env:FLASK_ENV = "development"

# 8. Start the Flask application
python backend/app.py

# 9. Verify the installation by running the test suite (in a separate terminal with venv activated)
pytest -q
```

### Setup for Linux / macOS

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/ditheshshettigar00-jpg/GeoPovertyAI.git
cd GeoPovertyAI

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Initialize SQLite database schema and migrations
python database/init_db.py

# 5. Precompute out-of-state Leave-One-State-Out predictions & SHAP explanations
python scripts/precompute_predictions.py

# 6. Interactively create the initial administrator user
python scripts/create_admin.py

# 7. Set development environment variable
export FLASK_ENV=development

# 8. Start the Flask application
python backend/app.py

# 9. Verify the installation by running the test suite
pytest -q
```

Open your browser and navigate to: **`http://localhost:5000`**

---

## 6. How to Run the Pipeline in Order

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

## 7. Empirical Results & Benchmark Tables

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

## 8. Limitations & Scientific Disclaimers

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

## 9. Testing & Quality Assurance

The test suite validates data integrity, spatial cross-validation, additive SHAP guarantees, security configurations, role-based access control, and API endpoints:

```bash
pytest -v
```

**Test Results:**
- **264 passed** (100% pass rate) across:
  - `tests/test_access_control.py`: Role-based access control route matrix, parameter validation, self/last-admin protection (162 tests)
  - `tests/test_area_profile.py`: Grouped district profile schema, survey indicators, and empirical percentiles (7 tests)
  - `tests/test_hidden_gaps.py`: Residual thresholding, candidate criteria, and screening limits (9 tests)
  - `tests/test_indicator_service.py`: Real district indicators, NFHS-5 survey join, PDF tables, and graceful fallbacks (9 tests)
  - `tests/test_phase2.py`: Spatial cross-validation, IQR clipping, pipeline transformers (10 tests)
  - `tests/test_phase3.py`: Additive SHAP equality ($\le 10^{-4}$), Grad-CAM deactivation, satellite image safety (7 tests)
  - `tests/test_pipeline.py`: Feature fusion, API routes, authentication, district matching (26 tests)
  - `tests/test_report_page.py`: District report preview route, dynamic tertile category thresholds JSON & PDF verification (4 tests)
  - `tests/test_scenario_explorer.py`: Scenario presets, single & group aggregations, uncertainty, and disclaimer integrity (8 tests)
  - `tests/test_security_hotspots.py`: CORS origin validation, session cookie flags, rate limiting, DBSCAN clustering (22 tests)
