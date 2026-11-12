# PROJECT BLUEPRINT: GeoPoverty AI
## Multimodal Explainable AI System for Area-Level Socioeconomic Deprivation Estimation

---

## 1. Executive Summary & Project Identification

* **Project Title:** GeoPoverty AI: Multimodal Explainable AI System for Area-Level Socioeconomic Deprivation Estimation
* **Domain:** Geospatial Artificial Intelligence (GeoAI), Remote Sensing, Machine Learning, Explainable AI (XAI), Computational Social Science.
* **Scope & Ethics Statement:** 
  * Estimates socioeconomic deprivation at the aggregate geographic administrative level (districts across India).
  * **Ethical Boundary:** Does **not** classify, profile, or track individual persons or households.
  * **Statistical Stance:** Treats satellite imagery, nighttime luminosity, and settlement density as statistical correlates of poverty, not causal explanations or ground truth. Designed as an automated decision-support research prototype.
* **Operating Modes:** Dual-mode architecture supporting **Demo Mode** (synthetic district data with zero network/API dependency for instant reproduction) and **Real Data Mode** (Google Earth Engine, Census/NFHS-5 survey ground truth).

---

## 2. Technology Stack & Environment Requirements

### 2.1 Language & Framework Versions (Pinned to `requirements.txt`)
* **Python Runtime:** Python 3.10 – 3.12 (CPU-compatible, standard laptop deployment)
* **Web Backend:** Flask 3.0.3, Flask-CORS 4.0.1, Werkzeug 3.1.8
* **Database & Persistence:** SQLite 3 (built-in), SQLAlchemy 2.0.31
* **Machine Learning & Core Math:** Scikit-Learn 1.5.1, XGBoost 2.0.3, SciPy 1.17.1, Joblib 1.6.0, NumPy 1.26.4, Pandas 2.2.2
* **Explainability (XAI):** SHAP 0.45.1 (TreeExplainer for tree ensembles and pipelines)
* **Geospatial & Remote Sensing:** GeoPandas 0.14.4, Rasterio 1.3.10, Shapely 2.0.4, OSMnx 1.9.3, EarthEngine-API 0.1.408, Geemap 0.32.1, PyProj 3.7.2
* **Visualization & Report Generation:** ReportLab 5.0.1, Pillow 10.4.0, Matplotlib 3.9.0
* **Testing & Quality Assurance:** PyTest 8.2.2 (**102 passed** tests in automated test suite)
* **Explored Deep Learning Extension:** TensorFlow 2.16.1 (CPU build, preserved for experimental VGG-16 feature extraction)

### 2.2 Frontend Stack
* **Architecture:** Server-side Jinja2 templates coupled with dynamic Vanilla JavaScript client fetch calls.
* **Styling:** Custom CSS design system (responsive clean styling, CSS variables, structured cards).
* **Mapping Engine:** Leaflet.js (Choropleth polygons, GeoJSON layers, custom tile layers, cluster heatmaps).
* **Data Visualization:** Chart.js and customized HTML/CSS horizontal contribution bars with plain-language explanations.

---

## 3. Directory & File Organization

```text
GeoPovertyAI/
├── backend/
│   ├── app.py                      # Flask app factory, session security, CORS origin checking
│   ├── api/
│   │   ├── routes.py               # Core REST API (areas, predictions, map GeoJSON, XAI, scenarios)
│   │   ├── auth.py                 # User authentication & memory rate limiting (5 failed / 15 min)
│   │   ├── reports.py              # ReportLab automated PDF executive report generation
│   │   └── admin.py                # Admin endpoints, user roles, security audit trail
│   └── services/
│       ├── db_service.py           # Thread-safe database querying & transaction service
│       ├── model_service.py        # Model loading, inference pipeline, live what-if simulation
│       └── hotspot_service.py      # Haversine DBSCAN with 150 km radius clustering
├── frontend/
│   ├── templates/                  # Jinja2 HTML views (home, map, district, compare, whatif, etc.)
│   └── static/
│       ├── css/                    # Custom CSS design system stylesheets
│       └── js/                     # Modular client-side controllers (map.js, explain.js, compare.js)
├── geospatial/
│   ├── GEE_SETUP.md                # Google Earth Engine authentication & cloud project instructions
│   ├── gee_config.py               # Earth Engine asset IDs, projection coordinates, state configurations
│   ├── extract_geospatial_features.py # Extraction for Sentinel-2, VIIRS nightlights, GHSL population
│   ├── download_satellite_patches.py  # District-centered RGB patch downloader
│   └── extract_accessibility_features.py # Explored OSMnx extraction script
├── preprocessing/
│   ├── name_keys.py                # Canonical state-contained district key matching
│   ├── generate_demo_images.py     # Deterministic synthetic satellite patch generator
│   └── feature_fusion.py           # Multimodal merge, 3xIQR outlier clipping, pipeline imputation
├── training/
│   ├── geo_validation.py           # Leave-One-State-Out & RepeatedGroupKFold spatial splitters
│   ├── pipelines.py                # Pipeline transformers (IQRClipper, median imputer, build_pipeline)
│   ├── leaky_kfold_demo.py         # Dynamic demonstration of spatial leakage from models/*.csv
│   └── train_models.py             # Linear, Ridge, RandomForest, GradientBoosting, XGBoost training
├── explainability/
│   ├── shap_explain.py             # SHAP local additive contributions & global feature importance
│   └── gradcam.py                  # Explored extension: reference Grad-CAM adapter for image models
├── database/
│   ├── schema.sql                  # Complete relational database DDL with prediction intervals
│   └── init_db.py                  # Safe migration and database initialization script
├── scripts/
│   ├── create_admin.py             # Interactive admin creation with prompt/env options
│   ├── generate_results_doc.py     # Dynamic generator of documentation/RESULTS.md & README tables
│   ├── precompute_predictions.py   # Precompute LOSO out-of-state estimates & consistent SHAP
│   └── refill_missing_features.py  # Secondary Earth Engine extraction for large districts (250m)
├── tests/
│   ├── conftest.py                 # Shared test fixtures & client configuration
│   ├── test_pipeline.py            # Core pipeline and API endpoint tests (26 tests)
│   ├── test_phase2.py              # Spatial cross-validation & pipeline tests (10 tests)
│   ├── test_phase3.py              # Additive SHAP equality & safety tests (7 tests)
│   └── test_security_hotspots.py   # CORS, cookies, rate limiting, DBSCAN tests (22 tests)
├── documentation/
│   ├── METHODS.md                  # Comprehensive methodology, formulas, and limitations
│   ├── RESULTS.md                  # Generated benchmark tables from models/*.csv
│   └── PROJECT_BLUEPRINT.md        # System blueprint and technical specification
├── config.py                       # Global directory paths, DBSCAN constants, security settings
└── requirements.txt                # Pinned dependency specification from pip freeze
```

---

## 4. End-to-End Data Pipeline & Feature Engineering

### 4.1 Data Sources & Active Features
The final production model uses **10 Google Earth Engine features**, anchored to the NFHS-5 survey window (**2019-11-01 to 2021-04-30** where time-varying):
1. **`nightlight_mean`**: VIIRS Day/Night Band monthly radiance ($nW \cdot cm^{-2} \cdot sr^{-1}$).
2. **`ndvi_mean`**: Cloud-filtered Sentinel-2 surface reflectance (30 m / 250 m scale).
3. **`builtup_pct`**: Global Human Settlement Layer built-up land percentage (GHSL epoch 2020).
4. **`population_density`**: GHSL human population density grid (GHSL epoch 2020).
5. **`elevation_mean_m`** & **`slope_mean_deg`**: NASA SRTM 30 m terrain elevation and derived slope.
6. **`rainfall_mm_year`**: UCSB CHIRPS daily precipitation annualized across the 2019–21 survey window.
7. **`lst_day_c`**: MODIS MOD11A2 daytime land surface temperature across the 2019–21 survey window.
8. **`cropland_pct`**: ESA WorldCover v200 cropland share (epoch 2021).
9. **`travel_time_city_min`**: Oxford/MAP surface travel time to nearest >50k city (2015 baseline; deprecated in Earth Engine).

*Explored extensions not in final model:* Deep visual features (VGG-16) were evaluated only in an exploratory run on the original project dataset (data/exploratory/original_table_with_visual_features.csv; 2023 imagery). In that run, leave-one-state-out R2 was 0.3273 without the visual features (using 7 tabular features: 4 satellite + 3 accessibility columns, not the final 4-feature model) and 0.2879 with them, and visual features alone gave 0.0702 (models/exploratory_visual_ablation/ablation_results.csv). They were not re-evaluated on the final 2019-21 data, and the final model does not use them. The accessibility columns in the original dataset (road density, school and hospital distance) could not be traced to an extraction script and were almost perfectly correlated with each other (|r| about 0.96); removing the accessibility columns changed R2 by about 0.003 (0.288 to 0.291) when the visual features were present, so they were not used. An OSM extraction script is included for future work but was not run nationally.

### 4.2 Socioeconomic Ground Truth
* Derived from the National Family Health Survey 5 (NFHS-5, 2019–2021).
* Deprivation score is an unweighted composite of 5 min-max scaled indicators: sanitation, electricity, clean cooking fuel, child stunting, and maternal anaemia (see [`documentation/METHODS.md`](METHODS.md)).
* Continuous ground truth score range: $[0.0, 1.0]$.

### 4.3 Feature Fusion & Pipeline Transformation
* **ID Resolution & Merging:** State-contained canonical key matching via `preprocessing/name_keys.py`.
* **Imputation:** Executed as **training-fold median inside the pipeline** to guarantee zero data leakage between training and held-out folds.
* **Outlier Management:** Relaxed $3.0 \times \text{IQR}$ threshold clipping within the pipeline to preserve legitimate remote or dense geographical signal.

---

## 5. Machine Learning & Spatial Cross-Validation

### 5.1 Spatial Autocorrelation & Geography-Aware Validation (`geo_validation.py`)
* Standard random K-Fold cross-validation yields inflated accuracy due to spatial proximity (Tobler's First Law).
* **Primary Evaluation Scheme:** **Leave-One-State-Out (LOSO) Cross-Validation**. Entire administrative states are held out as independent test blocks (31 folds across 564 districts).
* **Secondary Scheme:** Repeated State-Grouped 5-Fold Cross-Validation (`RepeatedGroupKFold`, 5 repeats).
* **Evaluation Metrics:** Evaluated by MAE, RMSE, $R^2$, and Spearman rank correlation $\rho$. All empirical numbers are reported directly from [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) and tabulated in [`documentation/RESULTS.md`](RESULTS.md).
* **Best Model:** `XGBRegressor` (XGBoost) selected as the highest-performing 10-feature architecture by LOSO $R^2$ (`0.4884`), MAE (`0.0752`), and RMSE (`0.0954`).

---

## 6. Explainable AI (XAI) Architecture

### 6.1 Tabular Explainability: SHAP (SHapley Additive exPlanations)
* Implemented via `explainability/shap_explain.py` wrapping the Scikit-Learn Pipeline with `shap.TreeExplainer`.
* **Additive Decomposition Guarantee:** For every district $i$:
  $$\hat{y}_i = \phi_0 + \sum_{j=1}^{10} \phi_{ij}$$
  Where $\phi_0$ is the base expected value from training and $\phi_{ij}$ are the exact feature attributions.
* In [`scripts/precompute_predictions.py`](../scripts/precompute_predictions.py), SHAP values are precomputed per state using the exact LOSO pipeline fitted on other states, ensuring additive consistency with stored out-of-state predicted scores.
* District pages display plain-language explanations alongside contribution bars with the explicit notice: *"Contributions show what drove this estimate; they describe the model, not causes."*

### 6.2 Explored Vision Extension: Grad-CAM
* Grad-CAM was explored as an extension for convolutional feature extractors in `explainability/gradcam.py`.
* Because the final model relies exclusively on 10 tabular satellite and geospatial statistics, Grad-CAM is deactivated in the web interface and API (`gradcam_available: false` with reason: *"The current model uses satellite statistics only, not image features"*). The reference implementation is retained for future image-based models.

---

## 7. Spatial Clustering & Hotspot Detection (`hotspot_service.py`)

* **Algorithm:** **Haversine DBSCAN with 150 km radius** (`eps = 150 / 6371 \approx 0.02354` radians, `min_samples = 3`).
* **Target Filter:** Identifies geographic clusters among districts whose predicted deprivation tier is `'High'` (discretized from empirical training tertiles).
* Generates regional poverty corridor boundaries, filtering isolated outliers to help planners prioritize contiguous multi-district interventions.

---

## 8. Relational Database Schema (`database/schema.sql`)

1. **`users`**: User authentication, passwords (`password_hash`), RBAC roles (`PUBLIC`, `RESEARCHER`, `ADMIN`), account status.
2. **`areas`**: Geographic district records (`area_id`, `area_name`, `state_name`, `centroid_lat`, `centroid_lon`, `data_mode`).
3. **`ground_truth`**: NFHS-5 survey values (`deprivation_score`, `data_source`, `data_year`).
4. **`satellite_data`**: Metadata for Sentinel-2 optical imagery.
5. **`geospatial_features`**: Tabular remote sensing indicators per district.
6. **`predictions`**: Model predictions per district (`predicted_score`, `predicted_low`, `predicted_high`, `predicted_category`, `prediction_type='leave-one-state-out'`). Displayed district scores are **out-of-state estimates**.
7. **`explanations`**: Serialized SHAP contribution JSON strings (`shap_json`) matching out-of-state scores.
8. **`audit_logs`**: Security logging tracking administrative actions.

---

## 9. REST API Specification

| Method | Endpoint | Description | Access Tier |
|---|---|---|---|
| `GET` | `/api/areas` | List of all monitored districts with coordinates | Public |
| `GET` | `/api/predict/<area_id>` | Predicted score, 80% range, category, and satellite status | Public |
| `GET` | `/api/map` | Full district list with predictions for Leaflet choropleth | Public |
| `GET` | `/api/explain/<area_id>` | SHAP values, base value, and `gradcam_available: false` | Public |
| `GET` | `/api/satellite-image/<id>`| Real Sentinel-2 RGB patch (demo images never served for real districts) | Public |
| `GET` | `/api/hotspots` | Haversine DBSCAN clusters for High-deprivation districts | Public |
| `GET` | `/api/model-performance` | LOSO validation benchmarks from `model_comparison_results.csv` | Public |
| `POST` | `/api/scenario` | Live What-If policy simulation on user-modified features | Public |
| `POST` | `/api/auth/register` | Register new user account with hashed password | Public |
| `POST` | `/api/auth/login` | Session login with rate limiting (5 failed / 15 min) | Public |
| `GET` | `/api/auth/logout` | Terminate session and clear cookie | Public |
| `GET` | `/api/reports/district/<id>`| ReportLab PDF executive dossier generation | Public |

---

## 10. Frontend Modules

1. **Executive Dashboard (`home.html`):** National deprivation summary, top vulnerable districts, and navigation.
2. **Interactive Spatial Map (`map.html`):** Leaflet choropleth rendering out-of-state scores with dynamic filtering.
3. **District Deep-Dive (`district.html`):** Scorecard with out-of-state estimate, 80% range, plain-language SHAP bars, and Sentinel-2 satellite image (or *"No satellite image available"*).
4. **Comparative Analysis (`compare.html`):** Side-by-side evaluation of 2 to 5 districts across satellite indicators.
5. **Spatial Hotspots (`hotspots.html`):** Interactive visualization of Haversine DBSCAN clusters.
6. **Policy Scenario Simulator (`whatif.html`):** Sliders allowing live simulation of shifts across the 10 satellite and geospatial indicators using the final model.

---

## 11. Testing & Quality Assurance

* **PyTest Suite:** **278 automated unit and integration tests** passing across 11 modules (including [`tests/test_extra_features.py`](../tests/test_extra_features.py), [`tests/test_area_profile.py`](../tests/test_area_profile.py), [`tests/test_indicator_service.py`](../tests/test_indicator_service.py), [`tests/test_hidden_gaps.py`](../tests/test_hidden_gaps.py), [`tests/test_scenario_explorer.py`](../tests/test_scenario_explorer.py), [`tests/test_report_page.py`](../tests/test_report_page.py), [`tests/test_access_control.py`](../tests/test_access_control.py), plus the Phase 1–3 suites below).
  - `tests/test_phase2.py`: Spatial cross-validation integrity, pipeline imputation, prediction intervals (10 tests).
  - `tests/test_phase3.py`: Additive SHAP equality ($\le 10^{-4}$), Grad-CAM deactivation, satellite image safety (7 tests).
  - `tests/test_pipeline.py`: Feature fusion, API endpoints, authentication, district matching (26 tests).
  - `tests/test_security_hotspots.py`: CORS origin validation, session cookie flags, rate limiting, DBSCAN clustering (22 tests).
