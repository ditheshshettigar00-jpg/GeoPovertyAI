# PROJECT BLUEPRINT: GeoPoverty AI
## Multimodal Explainable AI System for Area-Level Socioeconomic Deprivation Estimation

---

## 1. Executive Summary & Project Identification

* **Project Title:** GeoPoverty AI: Multimodal Explainable AI System for Area-Level Socioeconomic Deprivation Estimation
* **Domain:** Geospatial Artificial Intelligence (GeoAI), Remote Sensing, Machine Learning, Explainable AI (XAI), Computational Social Science.
* **Scope & Ethics Statement:** 
  * Estimates socioeconomic deprivation at the aggregate geographic administrative level (districts / sub-districts).
  * **Ethical Boundary:** Does **not** classify, profile, or track individual persons or households.
  * **Statistical Stance:** Treats satellite imagery, nighttime luminosity, and infrastructure density as statistical correlates of poverty, not causal explanations or ground truth. Designed as an automated decision-support research prototype.
* **Operating Modes:** Dual-mode architecture supporting **Demo Mode** (synthetic 60-district data with zero network/API dependency for instant reproduction) and **Real Data Mode** (Google Earth Engine, OpenStreetMap, Census/NFHS ground truth).

---

## 2. Technology Stack & Environment Requirements

### 2.1 Language & Framework Versions
* **Python Runtime:** Python 3.10 – 3.11 (CPU-compatible, no discrete GPU required)
* **Web Backend:** Flask 3.0.3, Flask-CORS 4.0.1, Werkzeug
* **Database / ORM:** SQLite 3 (built-in), SQLAlchemy 2.0.31
* **Machine Learning:** Scikit-Learn 1.5.1, XGBoost 2.0.3
* **Deep Learning (Computer Vision):** TensorFlow 2.16.1 (CPU build, VGG-16 feature extraction)
* **Explainability (XAI):** SHAP 0.45.1 (TreeExplainer, LinearExplainer), Custom Layer-Activation Grad-CAM
* **Geospatial & Remote Sensing:** GeoPandas 0.14.4, Rasterio 1.3.10, Shapely 2.0.4, OSMnx 1.9.3, EarthEngine-API 0.1.408, Geemap 0.32.1
* **Report Generation & Utilities:** ReportLab 4.2.0, Pillow 10.4.0, Matplotlib 3.9.0, Pandas 2.2.2, NumPy 1.26.4
* **Testing:** PyTest 8.2.2

### 2.2 Frontend Stack
* **Architecture:** Server-side Jinja2 templates coupled with dynamic Vanilla JavaScript client fetch calls.
* **Styling:** Custom CSS design system (responsive dark/modern styling, CSS variables, glassmorphism cards).
* **Mapping Engine:** Leaflet.js (Choropleth polygons, GeoJSON layers, custom tile layers, cluster heatmaps).
* **Data Visualization:** Chart.js (SHAP feature attribution bar charts, model regression parity plots, residual histograms).

---

## 3. Directory & File Organization

```
GeoPovertyAI/
├── backend/
│   ├── app.py                      # Flask app factory, session configuration, HTML route controllers
│   ├── api/
│   │   ├── routes.py               # Core REST API (areas, predictions, map GeoJSON, XAI, scenarios)
│   │   ├── auth.py                 # User authentication (register, login, session logout)
│   │   ├── reports.py              # ReportLab automated PDF executive report generation
│   │   └── admin.py                # Admin dashboard stats, user role management, audit log viewing
│   └── services/
│       ├── db_service.py           # Thread-safe database querying & transaction service
│       ├── model_service.py        # Model loading, inference pipeline, live what-if simulation
│       └── hotspot_service.py      # DBSCAN geospatial clustering of high-deprivation regions
├── frontend/
│   ├── templates/                  # 15 Jinja2 HTML views (home, map, district, compare, whatif, etc.)
│   └── static/
│       ├── css/                    # Custom CSS design system stylesheets
│       └── js/                     # Modular client-side controllers (map.js, explain.js, compare.js)
├── geospatial/
│   ├── GEE_SETUP.md                # Google Earth Engine authentication & cloud project instructions
│   ├── gee_config.py               # Earth Engine asset IDs, projection coordinates, state configurations
│   ├── fetch_district_boundaries.py# Administrative polygon fetching via FAO GAUL
│   ├── extract_geospatial_features.py # Extraction for Sentinel-2, VIIRS nightlights, GHSL population
│   ├── download_satellite_patches.py  # District-centered RGB GeoTIFF/PNG patch downloader
│   └── extract_accessibility_features.py # OSMnx road network density & distance to amenities
├── preprocessing/
│   ├── image_preprocessing.py      # Resizing (224x224), min-max normalization, ImageNet preprocessing
│   ├── vgg16_feature_extractor.py  # Frozen VGG-16 deep feature extractor (512-d pooled vectors)
│   ├── generate_demo_images.py     # Deterministic synthetic satellite patch generator
│   └── feature_fusion.py           # Multimodal merge, 3xIQR outlier clipping, median imputation
├── training/
│   ├── geo_validation.py           # State-level GroupKFold split preventing spatial autocorrelation leakage
│   └── train_models.py             # Linear Regression, Random Forest, XGBoost training & tuning
├── explainability/
│   ├── shap_explain.py             # SHAP local additive contributions & global feature importance
│   └── gradcam.py                  # Gradient-weighted Class Activation Mapping for deep feature activations
├── database/
│   ├── schema.sql                  # Complete 8-table relational database DDL
│   ├── init_db.py                  # Database initialization script
│   └── geopoverty.db               # SQLite database file
├── scripts/
│   ├── generate_demo_data.py       # 60-district synthetic dataset generator with realistic physics
│   ├── load_demo_to_db.py          # Bulk loader of demo records into SQLite
│   ├── precompute_predictions.py   # Batch prediction, SHAP, and Grad-CAM offline precomputing
│   └── run_full_demo_pipeline.sh   # One-command full bootstrap & test runner
├── tests/
│   └── test_pipeline.py            # Comprehensive 19-test PyTest validation suite
├── documentation/
│   └── PROJECT_BLUEPRINT.md        # Complete system blueprint & synopsis reference
├── config.py                       # Global project directory paths & shared constants
└── requirements.txt                # Pinned dependency specification
```

---

## 4. End-to-End Data Pipeline & Feature Engineering

### 4.1 Data Sources
1. **Satellite Remote Sensing (Google Earth Engine):**
   * **Sentinel-2 MultiSpectral Instrument (MSI):** Cloud-masked surface reflectance composites (Bands B4, B3, B2 for RGB; B8 for NIR to calculate Normalized Difference Vegetation Index: $NDVI = \frac{NIR - Red}{NIR + Red}$).
   * **VIIRS Day/Night Band (DNB):** Monthly average radiance composite representing anthropogenic nocturnal luminosity ($nW \cdot cm^{-2} \cdot sr^{-1}$).
   * **GHSL (Global Human Settlement Layer):** Built-up surface percentage and population density grids.
2. **Infrastructure & Accessibility (OpenStreetMap via OSMnx):**
   * **Road Network Density:** Total drivable road length per unit district area ($km / km^2$).
   * **Nearest Healthcare Distance:** Euclidean/network distance from district centroid to closest hospital/clinic ($km$).
   * **Nearest Education Distance:** Distance from district centroid to nearest primary/secondary school ($km$).
3. **Socioeconomic Ground Truth:**
   * Government census statistics (e.g., Census of India 2011) or demographic health surveys (NFHS-4/5).
   * Target variable: Continuous Multidimensional Deprivation Index / Deprivation Score ($Y \in [0.0, 1.0]$).

### 4.2 Feature Fusion & Cleaning Pipeline (`feature_fusion.py`)
* **ID Resolution & Merging:** Inner join on normalized district identifier (`area_id` / `district_name`).
* **Handling Missing Data:** Median imputation calculated strictly across training folds to prevent target leakage.
* **Robust Outlier Management:** Deliberately relaxed $3.0 \times \text{IQR}$ threshold clipping. Extreme geographic values (e.g., very high remote mountain road distances or ultra-dense urban centres) represent true real-world signal rather than measurement error.
* **Feature Vector Composition:**
  * Tabular features: `nightlight_mean`, `ndvi_mean`, `builtup_pct`, `population_density`, `road_distance_km`, `school_distance_km`, `hospital_distance_km`.
  * Deep visual features: 512-dimensional bottleneck latent vector $V_1, V_2, \dots, V_{512}$ extracted from district satellite imagery.

---

## 5. Machine Learning & Spatial Cross-Validation

### 5.1 Spatial Autocorrelation & Geography-Aware Validation (`geo_validation.py`)
* **The Problem:** Standard random K-Fold cross-validation suffers from severe spatial autocorrelation. Neighboring districts share climate, topography, and governance, artificially inflating cross-validation scores.
* **Solution:** **Spatial State-Grouped Cross-Validation (`GroupKFold`)**. Entire administrative states are held out as independent test blocks. Models are evaluated on their ability to generalize to unseen geographic territories.

### 5.2 Model Architectures & Hyperparameter Optimization (`train_models.py`)
1. **Baseline Regressor:** Ordinary Least Squares (OLS) Linear Regression with standard feature scaling.
2. **Non-Linear Ensemble:** Random Forest Regressor (`n_estimators=100-300`, `max_depth=5-15`, `min_samples_split=2-10`).
3. **Gradient Boosting:** XGBoost Regressor (`learning_rate=0.01-0.1`, `max_depth=3-7`, `subsample=0.8`, `colsample_bytree=0.8`).
4. **Model Selection Criteria:** Minimized Root Mean Squared Error ($RMSE$) and Mean Absolute Error ($MAE$), alongside coefficient of determination ($R^2$).
5. **Categorical Quantile Thresholds:** Training fold tertiles derive boundary thresholds for classifying continuous scores into discrete tiers (`Low`, `Medium`, `High` deprivation).

---

## 6. Explainable AI (XAI) Architecture

### 6.1 Tabular Explainability: SHAP (SHapley Additive exPlanations)
* Implements game-theoretic Shapley values via `shap.TreeExplainer` (for tree ensembles) and `shap.LinearExplainer`.
* **Additive Decomposition:** Every individual district prediction satisfies:
  $$\hat{y}_i = \phi_0 + \sum_{j=1}^{M} \phi_{ij}$$
  Where $\phi_0$ is the base expected value across the training population, and $\phi_{ij}$ is the contribution of feature $j$.
* Precomputed into structured JSON arrays: `[{"feature": "nightlight_mean", "contribution": -0.12, "value": 14.5}, ...]`.

### 6.2 Computer Vision Explainability: Layer-Activation Grad-CAM
* Computes gradients of target latent activations with respect to the final convolutional feature maps (`block5_conv3` in VGG-16).
* Computes spatial importance weight $\alpha_k$ for each filter map $A^k$:
  $$L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$
* Generates a 2D heatmap overlaid onto the original RGB satellite patch, highlighting structural elements (e.g., informal settlements, barren terrain, dense road intersections) driving model representations.
* Stored as pre-rendered transparent overlay PNG images served via REST API.

---

## 7. Spatial Clustering & Hotspot Detection (`hotspot_service.py`)

* **Algorithm:** Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using the Haversine metric on district centroids:
  $$\text{dist}((\text{lat}_1, \text{lon}_1), (\text{lat}_2, \text{lon}_2)) \le \epsilon$$
* **Objective:** Automatically detect contiguous regional belts of severe socioeconomic deprivation (High Deprivation Clusters) requiring priority intervention, filtering out isolated outlier districts.
* Outputs cluster labels, convex hull bounding boxes, and regional vulnerability statistics.

---

## 8. Relational Database Schema (`database/schema.sql`)

1. **`users`**: User accounts, passwords (`password_hash`), RBAC roles (`PUBLIC`, `RESEARCHER`, `ADMIN`), account status.
2. **`areas`**: Geographic district entities, centroid coordinates (`lat`, `lon`), GeoJSON boundary polygons, and `data_mode` (`demo` vs `real`).
3. **`ground_truth`**: Ground-truth target metrics (`deprivation_score`), data source label, and survey year.
4. **`satellite_data`**: Metadata for satellite image patches, acquisition timestamps, cloud cover percentage, file paths.
5. **`geospatial_features`**: Tabular remote sensing & accessibility attributes per district.
6. **`predictions`**: Model output per district, model identifier (`model_name`), continuous score, and categorical class (`Low`/`Medium`/`High`).
7. **`explanations`**: Serialized SHAP contribution JSON strings and relative paths to Grad-CAM heatmap PNG files.
8. **`model_results`**: Historical training and cross-validation performance metrics ($MAE$, $RMSE$, $R^2$) per model version.
9. **`audit_logs`**: System security auditing recording user ID, action timestamps, and operation details.

---

## 9. REST API & Endpoint Specification

| Method | Endpoint | Description | Access Tier |
|---|---|---|---|
| `GET` | `/api/areas` | List of all monitored districts with metadata | Public |
| `GET` | `/api/predict/<area_id>` | Predicted deprivation score, category, and ranking | Public |
| `GET` | `/api/map` | Full GeoJSON FeatureCollection with fused predictions | Public |
| `GET` | `/api/explain/<area_id>` | Detailed SHAP breakdown and base value | Public |
| `GET` | `/api/gradcam/<area_id>` | Serves visual Grad-CAM heatmap overlay PNG image | Public |
| `GET` | `/api/model-performance` | Evaluation metrics ($R^2$, $MAE$, $RMSE$) for models | Public |
| `POST` | `/api/scenario` | Interactive "What-If" simulation on modified feature vector | Public |
| `POST` | `/api/auth/register` | Create user account with hashed credentials | Public |
| `POST` | `/api/auth/login` | Authenticate and initialize Flask session | Public |
| `GET` | `/api/auth/logout` | Terminate session | Public |
| `GET` | `/api/reports/district/<id>`| Automated PDF generation of district dossier | Researcher / Admin |
| `GET` | `/api/admin/stats` | System-wide statistics and data health monitoring | Admin |
| `POST` | `/api/admin/user/<id>/role`| Update user authorization role | Admin |
| `GET` | `/api/admin/audit-logs` | Retrieve chronological security audit records | Admin |

---

## 10. Frontend User Interface Modules

1. **Executive Dashboard (`home.html`):** High-level summary of national deprivation, top vulnerable districts, and quick navigation.
2. **Interactive Spatial Map (`map.html`):** Leaflet choropleth with dynamic color ramps, district popups, layer toggle (Satellite vs OpenStreetMap), and filtering.
3. **Unified District Deep-Dive (`district.html` / `explanation.html`):** Detailed scorecard combining satellite patch view, Grad-CAM attention heatmap, and Chart.js horizontal SHAP feature attribution bar chart.
4. **Comparative Analysis (`compare.html`):** Side-by-side radar and bar chart evaluation of up to 5 districts across all geospatial and socioeconomic metrics.
5. **Spatial Hotspot Map (`hotspots.html`):** DBSCAN cluster visualization showing prioritized intervention corridors.
6. **Policy Scenario "What-If" Simulator (`whatif.html`):** Sliders allowing planners to simulate the impact of infrastructure investments (e.g., building roads or hospitals) on estimated deprivation.
7. **Data Quality & Ingestion Health (`data_quality.html`):** Missing value audits, outlier checks, and distribution histograms.
8. **Administration Portal (`admin_dashboard.html`):** User management, role elevation, and audit logs.

---

## 11. Testing & Quality Assurance

* **PyTest Suite (`tests/test_pipeline.py`):** 19 automated unit and integration tests:
  * Synthetic dataset generation integrity (schema, ranges, null absence).
  * Feature fusion edge cases (missing attributes, duplicate handling, IQR clipping).
  * Geography-aware state partition validity (zero overlap between train and test states).
  * Model training, parameter serialization, and threshold derivation.
  * REST API response status codes, payload schemas, and error boundaries (404 on missing districts, 400 on malformed scenario inputs).
* **Reproduction Script:** `scripts/run_full_demo_pipeline.sh` executes the complete pipeline from scratch in under 3 minutes on any standard dual-core laptop.
