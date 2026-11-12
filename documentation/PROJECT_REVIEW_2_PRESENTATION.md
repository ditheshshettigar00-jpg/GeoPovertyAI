# GeoPoverty AI: Main Project – 2nd Review Presentation Guide

**Candidate Name:** Dithesh Ashlesh Kumar  
**University Seat Number (USN):** 4JK25MC036  
**Project Guide:** Mr. Sunith Kumar T  
**Degree & Department:** Master of Computer Applications (MCA)  
**Scheduled Review Window:** November 04–05, 2026  
**Generated PowerPoint File:** [`GeoPovertyAI_2nd_Review_Presentation.pptx`](../GeoPovertyAI_2nd_Review_Presentation.pptx)  
**Total Slide Count:** 12 Slides (Strictly adhering to Departmental Maximum Guidelines)

---

## Presentation Structure Overview

| Slide # | Category / Section | Topic / Slide Title | Key Visuals & Exhibits |
|:---:|:---|:---|:---|
| **1** | Title & Admin | **GeoPoverty AI: An Explainable Multimodal AI-Based System for Area-Level Socioeconomic Deprivation Estimation Using Satellite Imagery** | Navy Executive Theme, Student & Guide Credentials, Batch Details |
| **2** | SDS (System Design) | **6-Tier Modular Architecture & System Decomposition** | 6-Tier Architecture Breakdown (Ingestion, Preprocessing, ML, XAI, Backend, Presentation) |
| **3** | SDS (Component Design) | **Relational Schema Modeling & Subsystem Interfaces** | 9 SQLite Relational Tables, `area_id` Pivot, Zero-Leakage Matching |
| **4** | SDS (Behavioral Design) | **System Interaction & Sequence Workflows** | 4-Stage Runtime Sequence, User Roles (RBAC), Async Processing |
| **5** | Methodology (Data) | **Multimodal Telemetry & NFHS-5 Deprivation Target** | Sentinel-2, VIIRS, GHSL Sensors + 5-Indicator Deprivation Index & PCA Sensitivity |
| **6** | Methodology (Pipeline) | **Algorithmic Pipeline & Feature Engineering Design** | Scikit-learn Pipeline, 4-Feature Set, Median Imputation, 3.0×IQR Filter, Ablation Rationale |
| **7** | Methodology (Validation)| **Spatial Leakage Mitigation: Tobler's First Law & LOSO** | Spatial Autocorrelation, Leave-One-State-Out (31 Folds) vs Leaky Random K-Fold |
| **8** | Methodology (Progress) | **Framework Implementation & Core Modules Built to Date** | 6 Core Modules Implemented (Harmonization, ML, XAI, DBSCAN, Flask REST, GIS UI) |
| **9** | Performance (Benchmark)| **Quantitative Model Evaluation & Benchmark Results** | Full Benchmark Table (564 Districts, 31 States), RandomForest Champion ($R^2=0.3360$), Chart |
| **10** | Performance (Ablation) | **Feature Ablation Study & Spatial Leakage Analysis** | Feature Dropping Impact (NDVI #1 Driver), Spatial Error Geography, Chart |
| **11** | Performance (Qualitative)| **Qualitative Validation: XAI, Policy Simulator & GIS UI** | Interactive Leaflet Choropleth, Policy Simulator, TreeSHAP Waterfall, DBSCAN Hotspots |
| **12** | Summary & Roadmap | **Project Milestones, Deliverables & Roadmap for Final Review** | Completed Milestones vs. Final Review Objectives, Panel Q&A Invitation |

---

# Detailed Slide-by-Slide Script & Talking Points

---

### Slide 1: Title & Administrative Details
- **Header:** MAIN PROJECT – 2ND REVIEW | MCA 2026
- **Title:** GeoPoverty AI: An Explainable Multimodal AI System for Area-Level Deprivation Estimation Using Satellite Imagery
- **Subtitle:** Integrating Sentinel-2, VIIRS Nightlights, and GHSL Settlement Telemetry with Spatial ML and TreeSHAP Explainability
- **Candidate:** Dithesh Ashlesh Kumar (USN: `4JK25MC036`)
- **Guide:** Mr. Sunith Kumar T, Department of Computer Applications
- **Review Date:** November 2026

#### Presenter Spoken Script (Approx. 60 seconds):
> *"Respected Chairperson, esteemed members of the departmental review panel, and my guide Mr. Sunith Kumar T, good morning.*
>
> *I am Dithesh Ashlesh Kumar (USN: 4JK25MC036). Today, I present the 2nd Review of my MCA Main Project titled **GeoPoverty AI: An Explainable Multimodal AI-Based System for Area-Level Socioeconomic Deprivation Estimation Using Satellite Imagery**.*
>
> *Traditional ground socioeconomic surveys (such as the Census or NFHS) are conducted only every 5 to 10 years, costing hundreds of crores and experiencing severe reporting latency. GeoPoverty AI addresses this gap by utilizing free, high-revisit satellite earth observation telemetry to estimate district-level socioeconomic deprivation across India continuously.*
>
> *In this 2nd Review presentation, I will strictly adhere to the departmental guidelines covering:*
> 1. *Our Software Design Specification (SDS): System architecture, 9-table schema, and sequence workflows.*
> 2. *Our Methodology: NFHS-5 ground truth formulation, 4 core satellite features, and our spatial leakage-free validation framework.*
> 3. *Our Performance Metrics: Leave-One-State-Out (LOSO) benchmarks across 564 districts, ablation studies, and qualitative decision-support features."*

---

### Slide 2: Software Design Specification (SDS) – Architectural Design & System Modularity
- **Category:** SOFTWARE DESIGN SPECIFICATION (SDS) | ARCHITECTURE & MODULARITY
- **Title:** 6-Tier Modular Architecture & System Decomposition
- **Three Columns:**
  1. **Data & Preprocessing Tiers:**
     - Tier 1: Multi-source remote sensing ingestion (Sentinel-2, VIIRS DNB, JRC GHSL) and ground truth (NFHS-5).
     - Tier 2: Pipeline feature fusion and leakage-free preprocessing (fold median imputation, 3.0×IQR filtering).
  2. **Modeling & Explainability Tiers:**
     - Tier 3: Spatial ML inference (Random Forest, XGBoost, GBDT, Ridge) with Leave-One-State-Out CV.
     - Tier 4: Game-theoretic XAI (TreeSHAP additive attributions) and spatial clustering (150 km Haversine DBSCAN).
  3. **Application & Presentation Tiers:**
     - Tier 5: Flask 3.0.3 REST API backend and SQLite 3 relational database (`sqlite3` / `schema.sql`).
     - Tier 6: Leaflet.js interactive choropleth GIS interface, Chart.js plots, and PDF dossier generator.

#### Presenter Spoken Script (Approx. 75 seconds):
> *"Moving to the Software Design Specification: Slide 2 illustrates our 6-Tier Modular Architecture.*
>
> *To ensure industrial-grade software engineering, we enforced strict separation of concerns, high cohesion, and low coupling across all layers:*
> - *Tier 1 ingests raw multi-sensor telemetry—Sentinel-2 optical rasters, VIIRS nighttime lights, and GHSL human settlement grids—temporally synchronized to the NFHS-5 survey window (November 2019 to April 2021).*
> - *Tier 2 handles feature fusion and preprocessing inside Scikit-learn pipelines. All imputation and outlier clipping parameters are fitted exclusively on training partitions to prevent data snooping.*
> - *Tier 3 executes spatial machine learning inference using frozen models evaluated under state-grouped holdouts.*
> - *Tier 4 provides transparent game-theoretic explainability via TreeSHAP and geographic hotspot clustering via Haversine DBSCAN.*
> - *Tier 5 and Tier 6 provide the Flask REST API backend and a responsive Leaflet.js choropleth GIS client.*
>
> *Because the client communicates exclusively via structured REST JSON APIs, the machine learning models and web interface are fully decoupled."*

---

### Slide 3: Software Design Specification (SDS) – Component Design & Relational Schema Modeling
- **Category:** SOFTWARE DESIGN SPECIFICATION (SDS) | COMPONENT & DATA DESIGN
- **Title:** Relational Schema Modeling & Subsystem Interfaces
- **Database Schema (9 Tables):**
  - `areas` (PK: `area_id`): 564 districts, state names, coordinates, GAUL boundary data.
  - `ground_truth` (PK: `gt_id`, FK: `area_id`): NFHS-5 composite deprivation target and 5 raw indicators.
  - `geospatial_features` (PK: `feature_id`, FK: `area_id`): 4 engineered satellite features (`nightlight_mean`, `ndvi_mean`, `builtup_pct`, `population_density`).
  - `satellite_data` (PK: `sat_id`, FK: `area_id`): Raster metadata, acquisition dates, cloud cover fractions.
  - `predictions` (PK: `prediction_id`, FK: `area_id`): Predicted deprivation scores [0, 1] and tertile labels.
  - `explanations` (PK: `explain_id`, FK: `prediction_id`): SHAP base values, local feature attributions, and ranks.
  - `model_results` (PK: `result_id`): Benchmark metrics (R², MAE, RMSE, Spearman $\rho$, Top-20% recall).
  - `users` and `audit_logs` (PK: `user_id` / `id`): RBAC authentication and audit trail tracking.
- **Key Highlights:**
  - `area_id` acts as the single primary relational anchor.
  - State-contained boundary matching resolved 564 districts across 31 states with 0 cross-state mismatches.
  - Dual operational modes: Real-World Mode vs. Demo Mode (`data_mode='demo'`).

#### Presenter Spoken Script (Approx. 75 seconds):
> *"Slide 3 details our component design and relational data model.*
>
> *The SQLite 3 database comprises 9 normalized relational tables defined in `database/schema.sql`. The 'areas' table serves as the primary relational pivot via 'area_id', linking satellite telemetry, ground-truth surveys, model predictions, and SHAP vectors with strict referential integrity.*
>
> *A major technical accomplishment in our data tier is our state-contained boundary matching algorithm. Because FAO GAUL 2015 boundaries predate certain district bifurcations in the 2019-21 NFHS-5 survey, we enforced strict in-state entity resolution (allowing cross-state mapping only for the historical Andhra Pradesh / Telangana bifurcation). This successfully matched 564 Indian districts across 31 states and union territories with zero entity leakage.*
>
> *Furthermore, our database supports dual operating modes: Real-World Mode for empirical research, and an isolated Demo Mode marked with data_mode='demo' for non-destructive system demonstration and functional testing."*

---

### Slide 4: Software Design Specification (SDS) – Interaction & Sequence Workflows
- **Category:** SOFTWARE DESIGN SPECIFICATION (SDS) | BEHAVIORAL WORKFLOWS
- **Title:** System Interaction & Sequence Workflows
- **4-Stage Sequential Flow:**
  1. **Client Request:** User selects district on Leaflet map or adjusts sliders in Policy Simulator; async HTTP fetch dispatched to `/api/predict` or `/api/districts`.
  2. **Backend Routing:** Flask controller validates request against schema, checks session RBAC, and queries SQLite database.
  3. **Pipeline Execution:** Scikit-learn pipeline performs training-fold median imputation and 3.0×IQR clipping; Random Forest predicts continuous score and tertile; TreeSHAP calculates feature attributions.
  4. **Response & Render:** JSON payload returned with score, empirical 80% prediction interval (`[-0.1363, +0.1437]`), and SHAP vectors; Leaflet map and Chart.js waterfall plots update asynchronously.

#### Presenter Spoken Script (Approx. 70 seconds):
> *"Slide 4 illustrates our interaction sequence and activity workflows.*
>
> *When a policymaker clicks a district on the map or simulates a policy intervention in the What-If Simulator:*
> 1. *The client dispatches an asynchronous HTTP request to our Flask REST endpoints.*
> 2. *The controller validates inputs against feature_columns.json, checks authorization, and retrieves district telemetry from the database.*
> 3. *The request passes through our serialized Scikit-learn inference pipeline (best_model.pkl). The model applies fold-derived median imputation and IQR clipping, then generates the continuous deprivation score in [0, 1] and tertile label.*
> 4. *Concurrently, our TreeSHAP engine computes additive feature attributions.*
> 5. *The client receives a structured JSON payload containing the prediction, empirical 80% prediction intervals (`[-0.1363, +0.1437]` from LOSO residuals), and SHAP attribution vectors, updating the Leaflet map and Chart.js graphs asynchronously.*
>
> *The system also supports automated PDF policy dossier generation for official distribution."*

---

### Slide 5: Model & Methodology – Multimodal Telemetry & NFHS-5 Target
- **Category:** EXPLANATION OF MODEL / METHODOLOGY | DATA INGESTION & GROUND TRUTH
- **Title:** Multimodal Telemetry & NFHS-5 Deprivation Target
- **Multimodal Remote Sensing Telemetry:**
  - **Sentinel-2 MSI:** 30m resolution multispectral imagery, cloud-masked via Scene Classification Layer (SCL 4–7). Computes district-wide median NDVI = $(\text{NIR} - \text{Red})/(\text{NIR} + \text{Red})$.
  - **NOAA/NASA VIIRS DNB:** 500m nocturnal luminosity composite ($nW \cdot cm^{-2} \cdot sr^{-1}$), proxying electrification and commercial activity.
  - **JRC GHSL Built-Up (`builtup_pct`):** 100m resolution (epoch 2020), percentage of district surface area covered by human structures.
  - **JRC GHSL Population Density (`population_density`):** 100m spatial population density ($\text{people}/km^2$, epoch 2020).
  - **Temporal Synchronization:** Strictly anchored to NFHS-5 survey dates (2019-11-01 to 2021-04-30).
- **NFHS-5 Ground Truth Deprivation Target:**
  - 5 Dimensions: (1) Sanitation Deficit, (2) Lack of Electricity, (3) Solid Cooking Fuel, (4) Child Stunting ($<-2\text{ SD}$), (5) Maternal Anaemia.
  - Formula: $\text{deprivation\_score} = \frac{1}{5} \sum_{k=1}^5 I_k^{\text{norm}} \in [0, 1]$
  - PCA Sensitivity Validation: First Principal Component (PC1) explains 49% variance and maintains Spearman rank correlation $\rho = 0.996$ with equal-weight index.

#### Presenter Spoken Script (Approx. 75 seconds):
> *"Transitioning to Section 2: Model and Methodology on Slide 5.*
>
> *Here we explain our data ingestion and ground truth formulation:*
> *For our ground truth, we derived a continuous deprivation score from the official NFHS-5 survey (2019-21) published by the Ministry of Health and Family Welfare (MoHFW) and IIPS. We selected 5 indicators capturing living conditions:*
> 1. *Percentage of households without improved sanitation*
> 2. *Percentage lacking electricity access*
> 3. *Percentage lacking clean cooking fuel*
> 4. *Prevalence of child stunting*
> 5. *Prevalence of anaemia among pregnant women*
> *Each indicator is min-max normalized across all surveyed districts, and the deprivation score is their unweighted arithmetic mean.*
>
> *To verify this target against weighting bias, we performed Principal Component Analysis across all 705 surveyed districts: PC1 explains 49% of total variance and has a 0.996 Spearman rank correlation with our equal-weight score. Omitting any single indicator retains rank correlations above 0.93, confirming that the index is mathematically robust.*
>
> *On the telemetry side, our inputs comprise Sentinel-2 NDVI at 30m resolution, VIIRS nighttime light radiance at 500m, and JRC GHSL built-up area and population density at 100m, all synchronized to the survey time window."*

---

### Slide 6: Model & Methodology – Algorithmic Pipeline & Feature Engineering
- **Category:** EXPLANATION OF MODEL / METHODOLOGY | FEATURE PIPELINE & SELECTION
- **Title:** Algorithmic Pipeline & Feature Engineering Design
- **Key Metrics Highlighted:**
  - **4** Core Satellite Features | **564** Evaluated Districts across **31** States | **3.0×IQR** Outlier Clipping | **0.996** PCA PC1 Stability
- **Pipeline Architecture:**
  - All preprocessing packaged in `sklearn.pipeline.Pipeline`.
  - Missing value imputation via training-fold medians.
  - Outlier filtering via fold-derived $[Q_1 - 3\times\text{IQR}, Q_3 + 3\times\text{IQR}]$ boundaries.
  - Strictly deterministic column ordering: `['nightlight_mean', 'ndvi_mean', 'builtup_pct', 'population_density']`.
  - Calibrated empirical prediction intervals from LOSO residuals: 80% `[-0.1363, +0.1437]`, 90% `[-0.1618, +0.1905]`.
- **Scientific Feature Selection & Ablation:**
  - *VGG-16 Deep Vision Ablated:* Standalone $R^2 = 0.0702$; degraded tabular model ($R^2 = 0.3273 \rightarrow 0.2879$) due to domain gap on 30m tiles. Intentionally excluded.
  - *OSM Accessibility Ablated:* Multicollinearity $|r| \approx 0.96$ between road and hospital distances; marginal contribution ($\Delta R^2 \approx 0.003$); excluded to maintain nationwide reproducibility.

#### Presenter Spoken Script (Approx. 75 seconds):
> *"Slide 6 explains our algorithmic pipeline architecture and feature selection rationale.*
>
> *All transformations are packaged inside Scikit-learn Pipeline objects. To guarantee that test data is never snooped:*
> - *Imputations strictly use training-fold medians.*
> - *Outlier filtering applies 3.0×IQR boundaries computed solely on training partitions.*
>
> *Now, an essential technical decision made during this phase:*
> *In preliminary exploratory runs, we evaluated deep visual representations using a pretrained VGG-16 CNN and OpenStreetMap infrastructure accessibility metrics.*
> *However, our empirical ablation revealed two critical insights:*
> 1. *Pretrained VGG-16 features alone yielded an out-of-state R² of only 0.0702, and adding them to tabular models actually degraded R² from 0.3273 to 0.2879 due to domain shift on satellite tiles.*
> 2. *OSM road, school, and hospital distances had severe multicollinearity (|r| ≈ 0.96) and contributed virtually zero performance gain (ΔR² ≈ 0.003), while lacking uniform nationwide auditability.*
>
> *Therefore, following Occam's Razor and scientific best practices, we selected the parsimonious 4-feature satellite model, guaranteeing maximum nationwide auditability, stability, and zero feature redundancy."*

---

### Slide 7: Model & Methodology – Spatial Leakage Prevention & Validation
- **Category:** EXPLANATION OF MODEL / METHODOLOGY | SPATIAL VALIDATION ARCHITECTURE
- **Title:** Spatial Leakage Mitigation: Tobler's First Law & LOSO
- **The Spatial Autocorrelation Hazard:**
  - Tobler's First Law: *"Everything is related to everything else, but near things are more related than distant things."*
  - Standard random train/test splits place neighboring districts across folds. The model simply memorizes regional patterns rather than learning true socioeconomic signals.
  - **Empirical Inflation:** Random 5-Fold CV yields an artificially inflated $R^2 = 0.4628$ (+37.7% boost). This is spurious spatial leakage.
- **Rigorous Spatial Validation Schemes:**
  - **Leave-One-State-Out (LOSO) CV (Primary):** 31 iterations. The model trains on 30 states and evaluates on an unseen administrative state. Honest out-of-state $R^2 = 0.3360, \text{MAE} = 0.0883$.
  - **Repeated GroupKFold (Secondary):** 5-fold CV grouped by `state_name`, repeated 5 times with distinct seeds ($R^2 = 0.2984 \pm 0.0216$).
  - **Integrity Guarantee:** Zero geographic, feature, or target leakage.

#### Presenter Spoken Script (Approx. 80 seconds):
> *"Slide 7 addresses one of the most vital scientific contributions of our project: Spatial Leakage Prevention.*
>
> *In spatial machine learning, standard random cross-validation is fundamentally invalid due to Tobler's First Law of Geography: 'Near things are more related than distant things'. Neighboring districts share similar climate, topography, infrastructure, and administrative schemes.*
> *If district A is in the training set and neighboring district B is in the test set, the model memorizes local geography rather than learning generalizable socioeconomic signatures.*
>
> *When we ran standard random 5-fold CV, the model produced an R² of 0.4628. However, this is an illusion of accuracy—an artificial inflation of +37.7% caused by spatial leakage.*
>
> *To establish genuine generalization, we implemented Leave-One-State-Out (LOSO) cross-validation across all 31 administrative states:*
> - *In each of the 31 iterations, all districts in an entire state are held out.*
> - *The model trains on 30 states and predicts on the held-out state, simulating a real-world deployment over un-surveyed regions.*
> *Every single district prediction in our system is an out-of-state estimate from this honest LOSO scheme."*

---

### Slide 8: Model & Methodology – Framework Implementation & Core Modules Built
- **Category:** EXPLANATION OF MODEL / METHODOLOGY | IMPLEMENTATION STATUS
- **Title:** Framework Implementation & Core Modules Built to Date
- **6 Implemented Core Modules:**
  1. **Data Harmonization Engine (`preprocessing/name_keys.py`):** State-contained fuzzy entity matcher linking GAUL 2015 polygons to NFHS-5 survey (564 districts, 31 states).
  2. **ML Inference Pipeline (`training/pipelines.py`):** Serialized pipelines (`best_model.pkl`) with median imputation and 3.0×IQR clipping.
  3. **Game-Theoretic XAI (`explainability/shap_explain.py`):** TreeSHAP integration for local waterfall plots and global feature rankings (precomputed in SQLite).
  4. **Spatial DBSCAN Engine (`backend/services/hotspot_service.py`):** Haversine density-based spatial clustering (150 km radius, $\min=3$) finding cross-border poverty corridors.
  5. **Decision Support Web App (`backend/app.py`, `backend/api/routes.py`):** Flask 3.0.3 REST backend, SQLite 3 database (`sqlite3` / `schema.sql`), RBAC authentication, and audit logs.
  6. **GIS UI & Policy Simulator (`frontend/`):** Leaflet.js interactive choropleth map, live parameter predictor, What-If simulator, and ReportLab PDF dossier generator.

#### Presenter Spoken Script (Approx. 70 seconds):
> *"Slide 8 summarizes our technical implementation and the 6 core software modules built to date:*
> 1. *Data Harmonization Engine: String normalization and state-contained matching linking 564 districts with zero cross-state errors.*
> 2. *Spatial ML Inference Pipeline: Serialized Scikit-learn pipelines with frozen model artifacts (best_model.pkl).*
> 3. *Game-Theoretic Explainability Engine: TreeSHAP implementation calculating exact local Shapley values precomputed in SQLite.*
> 4. *Spatial DBSCAN Analytics Engine: Density-based clustering using Haversine distance with a 150 km search radius, successfully finding contiguous poverty pockets regardless of administrative borders.*
> 5. *Decision-Support Backend: Flask 3.0.3, SQLite 3 relational database, RBAC authentication, and audit logs.*
> 6. *GIS Client & Policy Simulator: Leaflet.js choropleth visualization, interactive What-If scenario simulations, and automated ReportLab PDF dossier generation.*
>
> *All 6 core modules are implemented, tested, and integrated into our functional prototype."*

---

### Slide 9: Performance Metrics & Evaluation – Benchmark Comparison & Model Selection
- **Category:** PERFORMANCE METRICS & EVALUATION | BENCHMARK COMPARISON
- **Title:** Quantitative Model Evaluation & Benchmark Results
- **LOSO Quantitative Benchmark (564 Districts, 31 States):**

| Model | $R^2$ | MAE | RMSE | Spearman $\rho$ | Cat. Accuracy | Top-20% Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy (Mean)** | -0.0565 | 0.1142 | 0.1371 | -0.6972 | 31.9% | 0.9% |
| **Ridge Regression** | 0.2700 | 0.0914 | 0.1140 | 0.5542 | 48.6% | 32.7% |
| **RandomForest Regressor ★** | **0.3360** | **0.0883** | **0.1087** | **0.5745** | **54.6%** | **46.0%** |
| **GradientBoosting** | 0.2951 | 0.0912 | 0.1120 | 0.5539 | 52.7% | 45.1% |
| **XGBoost Regressor** | 0.3074 | 0.0905 | 0.1110 | 0.5643 | 52.7% | 46.9% |

- **Embedded Exhibit:** High-resolution model comparison chart (`models/figures/model_comparison.png`).
- **Key Takeaways:**
  - RandomForest selected as primary production model: Highest $R^2$ (0.3360), lowest MAE (0.0883), and lowest RMSE (0.1087).
  - 22.7% error reduction over baseline Dummy regressor.
  - Strong monotonic ranking ability ($\rho = 0.5745$).
  - High Top-20% recall (46.0%): identifies nearly half of India's most deprived districts purely from satellite data.

#### Presenter Spoken Script (Approx. 80 seconds):
> *"Slide 9 presents our primary quantitative evaluation: the Leave-One-State-Out (LOSO) benchmark across 564 districts in 31 states.*
>
> *As shown in the table and chart:
> - The Dummy Regressor (predicting the training mean of other states) yields an R² of -0.0565 and MAE of 0.1142.
> - Ridge Regression achieves an R² of 0.2700, demonstrating a clear linear signal.
> - RandomForest Regressor clearly emerges as the champion model:
>   * Out-of-state R² of 0.3360
>   * MAE of 0.0883 (a 22.7% reduction in error vs. baseline)
>   * RMSE of 0.1087
>   * Spearman rank correlation of 0.5745
>   * 3-class Categorical Accuracy of 54.6%
>   * Top-20% Deprivation Recall of 46.0%
> - XGBoost and GradientBoosting also perform strongly (R² 0.3074 and 0.2951), but Random Forest's variance reduction across geographically diverse states makes it the most reliable.*
>
> *Capturing 46% of the most acutely deprived districts purely from orbital sensors demonstrates remarkable utility for targeting developmental funding."*

---

### Slide 10: Performance Metrics & Evaluation – Feature Ablation & Spatial Leakage Analysis
- **Category:** PERFORMANCE METRICS & EVALUATION | ABLATION & SPATIAL INTEGRITY
- **Title:** Feature Ablation Study & Spatial Leakage Analysis
- **LOSO Feature Ablation Table:**

| Feature Configuration | Features | $R^2$ | MAE | RMSE | Spearman $\rho$ | Top-20% Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **All 4 Features (Full Model)** | 4 | **0.3360** | **0.0883** | **0.1087** | **0.5745** | **46.0%** |
| Without Nightlight (`nightlight_mean`) | 3 | 0.2894 | 0.0911 | 0.1125 | 0.5430 | 38.0% |
| Without Built-Up (`builtup_pct`) | 3 | 0.2529 | 0.0939 | 0.1153 | 0.4978 | 39.8% |
| Without Population (`population_density`) | 3 | 0.1854 | 0.0976 | 0.1204 | 0.4188 | 28.3% |
| Without Vegetation (`ndvi_mean`) | 3 | **0.1567** | **0.0988** | **0.1225** | **0.4032** | **37.2%** |

- **Embedded Exhibit:** Feature group importance visualization (`models/figures/feature_group_importance.png`).
- **Key Takeaways:**
  - **Vegetation (NDVI) is the #1 Driver:** Dropping Sentinel-2 NDVI causes $R^2$ to collapse to 0.1567 ($-53.4\%$ drop), proving agricultural health is the primary predictor of rural deprivation.
  - **Population Density is #2:** Omitting population density causes a $-44.8\%$ drop to $R^2 = 0.1854$.
  - **Residual Error Geography:** Highest prediction errors occur in topographically distinct and heavily forested states (Goa, Tripura, Meghalaya, Jharkhand).

#### Presenter Spoken Script (Approx. 80 seconds):
> *"Slide 10 presents our systematic feature ablation study and spatial error breakdown.*
>
> *By training models while selectively omitting one satellite feature at a time under Leave-One-State-Out evaluation:
> - Omitting Sentinel-2 NDVI causes R² to plummet from 0.3360 to 0.1567—a 53.4% drop in predictive power. This proves that vegetation index, reflecting rural agricultural viability and green cover, is the single most vital predictor.
> - Omitting GHSL population density cuts R² to 0.1854 (a 44.8% drop), showing that sparse settlement patterns strongly correlate with deprivation.
> - Dropping built-up area and nighttime lights reduces R² to 0.2529 and 0.2894 respectively.*
>
> *This confirms that all four features provide non-redundant, complementary signals.
> On the right, we also show where residual errors are concentrated: predominantly northeastern and forested states (Goa, Tripura, Meghalaya, Jharkhand). In these regions, high canopy cover shields rural settlements from optical sensors, providing clear avenues for our future spatial refinement."*

---

### Slide 11: Performance Metrics & Evaluation – Qualitative Evaluation & System Validation
- **Category:** PERFORMANCE METRICS & EVALUATION | QUALITATIVE VALIDATION
- **Title:** Qualitative Validation: XAI, Policy Simulator & GIS UI
- **4 Embedded UI Screenshots:**
  1. **Interactive Choropleth GIS:** Full choropleth map of 564 districts color-coded by deprivation tertiles, with search, state filtering, and hover uncertainty bounds.
  2. **What-If Policy Simulator:** Interactive tool allowing planners to simulate hypothetical infrastructure changes (e.g. rural electrification drives) and observe estimated deprivation changes in real time.
  3. **TreeSHAP Local Attributions:** Transparent waterfall plots decomposing exactly which indicators pushed a district's score up or down.
  4. **Spatial Clusters & PDF Dossier:** Haversine DBSCAN detecting continuous cross-border poverty corridors, paired with automated ReportLab PDF dossier generation.

#### Presenter Spoken Script (Approx. 80 seconds):
> *"Slide 11 highlights our qualitative evaluation and decision-support capabilities through screenshots of our functioning system:
> 1. Interactive Choropleth Map: Displays 564 districts classified into Low, Medium, and High deprivation tertiles, enabling state-level filtering, spatial queries, and dynamic hover overlays showing empirical 80% prediction intervals.
> 2. What-If Policy Intervention Simulator: Allows policymakers to adjust parameters (such as simulating a rural electrification drive or urban settlement expansion) and observe forward model predictions and score changes in real-time.
> 3. TreeSHAP Local Attributions: Provides transparent explanations. For instance, in East Godavari or Anantapur, SHAP shows exactly how high vegetation index and low nighttime luminosity push deprivation scores up or down, corroborating known economic geography.
> 4. Hotspot Detection & PDF Reporting: Haversine DBSCAN isolates continuous regional clusters (e.g. the tribal belt in Central India) across administrative state borders, and one-click PDF dossiers provide field officers with publication-ready reports."*

---

### Slide 12: Summary, Deliverables & Roadmap for Final Review
- **Category:** SUMMARY & ROADMAP | 2ND REVIEW DELIVERABLES
- **Title:** Project Milestones, Deliverables & Roadmap for Final Review
- **Completed Work & 2nd Review Deliverables (Left Card):**
  - Complete Software Design Specification (SDS) with 6-tier architecture and 9-table schema.
  - Leakage-free spatial ML engine trained and frozen with Leave-One-State-Out CV (Random Forest $R^2 = 0.3360$).
  - TreeSHAP additive attributions and 150 km Haversine DBSCAN operational.
  - Full-stack web application with Leaflet.js GIS, Policy Simulator, and PDF dossier generator.
  - Scientifically audited dataset of 564 districts with zero cross-state errors.
- **Roadmap to Final Review (Right Card):**
  - Sub-district spatial disaggregation to tehsil/block level using 10m Sentinel-2 bands.
  - Dynamic Earth Engine tile server to stream live false-color infrared rasters onto Leaflet.
  - Multi-year longitudinal drift analysis across census and survey waves.
  - Cloud containerization (Docker) and deployment with NGINX reverse proxy.
  - Final project report and dissertation completion.
- **Bottom Banner:** "Thank You! Questions and Feedback from the Panel are Welcome."

#### Presenter Spoken Script (Approx. 60 seconds):
> *"In conclusion: for this 2nd Review milestone, we have accomplished all targeted deliverables:
> - We completed the detailed Software Design Specification (SDS).
> - We solved spatial entity reconciliation across 564 districts in 31 states.
> - We formulated and sensitivity-validated the continuous NFHS-5 deprivation index.
> - We proved the necessity of spatial cross-validation, demonstrating how random K-Fold inflates R² by +37.7%, and established an honest Leave-One-State-Out benchmark (RandomForest R² = 0.3360, MAE = 0.0883, Top-20% recall = 46.0%).
> - We integrated TreeSHAP explainability, Haversine DBSCAN spatial clustering, and an interactive GIS decision-support portal.
>
> For the Final Review phase, our roadmap includes sub-district spatial disaggregation, dynamic Earth Engine raster tile streaming, longitudinal drift analysis, and completing the comprehensive MCA dissertation.
>
> Thank you for your valuable time and attention. I am now open to questions, feedback, and suggestions from the esteemed panel."*

---

# Comprehensive Viva Q&A Guide (Departmental Panel Defense)

### Q1: Why did you drop VGG-16 deep visual features and OpenStreetMap accessibility features from your final model?
**Model Answer:**
> *"During our exploratory phase, we conducted rigorous ablation benchmarks. Pretrained VGG-16 features evaluated on satellite image patches achieved an out-of-state $R^2$ of only 0.0702 on their own. When combined with tabular features, they actually degraded model performance from $R^2 = 0.3273$ to $0.2879$. This occurs because ImageNet-pretrained CNN filters are trained on consumer RGB photographs (dogs, cars) and experience a severe domain shift when applied to top-down 30-meter satellite tiles. Furthermore, OpenStreetMap road, school, and hospital distances had severe collinearity with each other ($|r| \approx 0.96$) and contributed marginal variance ($\Delta R^2 \approx 0.003$), while lacking nationwide auditability. By ablated both, our final 4-feature satellite model is parsimonious, fully reproducible, immune to collinearity, and achieves our highest honest generalization score."*

### Q2: What is spatial autocorrelation, and how does your evaluation prevent spatial data leakage?
**Model Answer:**
> *"According to Tobler's First Law of Geography, adjacent geographic regions share similar climatic, topographical, economic, and administrative conditions. In standard random K-Fold cross-validation, neighboring districts are randomly assigned to both the training and testing sets. As a result, the model memorizes regional spatial patterns rather than learning true generalizable socioeconomic signatures. In our experiments, standard random 5-fold CV produced an artificially inflated $R^2$ of 0.4628 (+37.7% inflation). To prevent this spatial leakage, we implemented Leave-One-State-Out (LOSO) cross-validation across 31 administrative states: in every fold, an entire state is held out, forcing the model to predict on geographically unseen territory. Every score displayed in our platform is an out-of-state prediction from this honest process."*

### Q3: Why is Random Forest outperforming XGBoost and Gradient Boosting in your Leave-One-State-Out evaluation?
**Model Answer:**
> *"In Leave-One-State-Out cross-validation, the test fold is an entire unseen state with potentially distinct regional topography and economic characteristics. Gradient Boosting and XGBoost build trees sequentially to correct residuals, which makes them more prone to over-indexing on idiosyncratic state patterns present in the training folds. In contrast, Random Forest uses bagging (bootstrap aggregation) and random feature subsampling across 300 trees (`n_estimators=300`, `max_features=0.5`, `min_samples_leaf=3`), which significantly reduces model variance. This makes Random Forest more robust when generalizing to entirely new geographic states, achieving an out-of-state $R^2$ of 0.3360 compared to XGBoost's 0.3074."*

### Q4: How did you construct your ground truth deprivation score, and how do you know it is valid?
**Model Answer:**
> *"Our ground truth is formulated from 5 official socioeconomic indicators in the NFHS-5 factsheets (2019–21): sanitation deficit, lack of electricity, lack of clean cooking fuel, child stunting, and maternal anaemia. Each indicator is min-max normalized across all surveyed districts, and the deprivation score is their unweighted arithmetic mean in the interval $[0, 1]$. To validate against arbitrary weighting, we performed Principal Component Analysis across 705 surveyed districts: the first principal component (PC1) explains 49% of the total variance and maintains a 0.996 Spearman rank correlation with our equal-weight score. Furthermore, leave-one-indicator-out sensitivity analysis confirmed correlations between 0.932 and 0.988, proving the composite ranking is exceptionally stable."*

### Q5: How does your What-If Policy Simulator work under the hood?
**Model Answer:**
> *"The What-If Scenario Explorer takes the baseline 4-feature vector for a selected district and applies user-defined adjustments (for instance, an electrification proxy multiplying `nightlight_mean` by `1.25x` or `1.5x`, or custom feature deltas). This adjusted vector is fed directly into our serialized Scikit-learn pipeline in real time. The model computes forward inference, generating the simulated deprivation score, the score difference ($\Delta$) compared against the model's MAE (`0.0883`), out-of-range warnings, and updated 80% empirical intervals (`[-0.1363, +0.1437]`)."*

### Q6: What is the purpose of Haversine DBSCAN clustering in your application?
**Model Answer:**
> *"Administrative boundaries like state and district borders are artificial political constructs; poverty and economic hardship often follow continuous geographical and environmental corridors (such as the Chota Nagpur plateau or drought-prone belts). We implemented DBSCAN with the Haversine metric on district centroid coordinates with an $\epsilon$ of 150 km and $\text{min\_samples} = 3$ high-deprivation districts. This automatically discovers contiguous trans-boundary poverty clusters, enabling regional multi-state developmental coordination."*


---

# Complete Mathematical & Algorithmic Calculations Reference (Panel Defense)

Use this section if the review panel asks ***"How are the calculations done?"*** for any component in the project. Every formula and constant below comes directly from the implemented codebase.

### 1. Ground Truth Target Calculation (`scripts/prepare_real_ground_truth.py`)
- **Source:** 5 socioeconomic indicators from NFHS-5 district factsheets:
  1. **Sanitation (Indicator 9):** Reported as % *with* improved sanitation $\rightarrow$ inverted to deficit: $100 - \text{value}$.
  2. **Electricity (Indicator 7):** Reported as % *with* electricity $\rightarrow$ inverted to deficit: $100 - \text{value}$.
  3. **Clean Cooking Fuel (Indicator 10):** Reported as % *using* clean fuel $\rightarrow$ inverted to deficit: $100 - \text{value}$.
  4. **Child Stunting (Indicator 73):** Already % of children stunted (kept as-is).
  5. **Maternal Anaemia (Indicator 84):** Already % of pregnant women anaemic (kept as-is).
- **Min-Max Normalization:** Across all districts with complete data:
  $$I_k^{\text{norm}} = \frac{I_k - \min(I_k)}{\max(I_k) - \min(I_k)}$$
- **Composite Deprivation Score:** Unweighted arithmetic mean rounded to 4 decimal places:
  $$\text{deprivation\_score} = \frac{1}{5} \sum_{k=1}^{5} I_k^{\text{norm}} \in [0, 1]$$
- **Sensitivity Check (`scripts/score_sensitivity.py`):** Validated across 705 districts; PC1 (49% variance) has Spearman $\rho = 0.996$ with the equal-weight score, and leaving out any single indicator keeps $\rho \in [0.932, 0.988]$.

---

### 2. Satellite & Geospatial Feature Calculations (`geospatial/extract_geospatial_features.py`, `geospatial/ghsl_features.py`)
All telemetry is strictly synchronized to `2019-11-01` to `2021-04-30` (GHSL epoch `2020`).
- **NDVI (`ndvi_mean`):**
  - Sentinel-2 SR scenes with $<20\%$ `CLOUDY_PIXEL_PERCENTAGE`, masked via SCL band keeping classes 4 (vegetation), 5 (bare soil), 6 (water), and 7 (unclassified), scaled by `/ 10000`.
  - Median temporal composite clipped to district geometry, then:
    $$\text{NDVI} = \frac{\text{B8 (NIR)} - \text{B4 (Red)}}{\text{B8 (NIR)} + \text{B4 (Red)}}$$
  - Aggregated as the district spatial mean (`ee.Reducer.mean()`).
- **Nighttime Light (`nightlight_mean`):**
  - NOAA/NASA VIIRS monthly composite (`avg_rad`), temporal mean over the survey window, district spatial mean at 500m scale.
- **Built-Up Percentage (`builtup_pct`):**
  - GHSL grid cell area is $100\text{ m} \times 100\text{ m} = 10,000\text{ m}^2$. Since raw values represent built-up $\text{m}^2$ per cell ($0\text{–}10000$):
    $$\text{builtup\_pct} = \text{clip}\left(\frac{\text{mean\_m}^2}{10,000} \times 100,\; 0,\; 100\right)$$
- **Population Density (`population_density`):**
  - Since $1\text{ km}^2 = 1,000,000\text{ m}^2$, there are $1,000,000 / 10,000 = 100$ GHSL cells per $\text{km}^2$:
    $$\text{population\_density} = \text{mean\_per\_cell} \times 100 \quad (\text{people/km}^2)$$

---

### 3. Preprocessing & Model Inference Calculations (`training/pipelines.py`)
- **IQR Outlier Clipping (`IQRClipper(3.0)`):** Fitted *only* on each training fold:
  $$\text{IQR} = Q_3 - Q_1, \quad \text{bounds} = [Q_1 - 3.0 \times \text{IQR},\; Q_3 + 3.0 \times \text{IQR}]$$
- **Imputation:** `SimpleImputer(strategy="median")` fitted *only* on the training fold.
- **Random Forest Prediction:** `RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, random_state=42)`. The continuous prediction $\hat{y}$ is the unweighted mean across all 300 decision trees.
- **Tertile Categorization (`models/category_thresholds.json`):** Fitted on target tertiles (1/3 and 2/3 quantiles):
  - **Low:** $\hat{y} \le 0.3198$
  - **Medium:** $0.3198 < \hat{y} \le 0.4526$
  - **High:** $\hat{y} > 0.4526$

---

### 4. Evaluation Metrics & Empirical Uncertainty Intervals (`training/evaluation.py`, `training/train_models.py`, `models/uncertainty.json`)
- **Regression Metrics (computed on pooled LOSO out-of-state predictions across 564 districts):**
  - $\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$ (`0.0883` for RandomForest vs. `0.1142` Dummy = **22.7% reduction**)
  - $\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ (`0.1087`)
  - $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ (`0.3360`)
- **Ranking & Targeting Metrics:**
  - **Spearman $\rho$ (`0.5745`):** Rank correlation measuring monotonic ordering accuracy across districts.
  - **Categorical Accuracy (`54.6%`):** Match rate on Low/Medium/High tertiles (with cutoffs fitted per training fold).
  - **Top-20% Recall (`46.0%`):** Share of districts in the true top 20% most-deprived ($\ge 80\text{th}$ percentile of $y$) that also fall in the predicted top 20% ($\ge 80\text{th}$ percentile of $\hat{y}$); random guessing baseline $= 20\%$.
- **Empirical Prediction Intervals:** Derived directly from the percentiles of out-of-state residuals $r_i = y_i - \hat{y}_i$:
  - **80% Interval (used in UI & Scenario Explorer):** $[10\text{th}, 90\text{th}]$ residual percentiles $= [\hat{y} - 0.1363,\; \hat{y} + 0.1437]$ (clamped to $[0, 1]$).
  - **90% Interval:** $[5\text{th}, 95\text{th}]$ residual percentiles $= [\hat{y} - 0.1618,\; \hat{y} + 0.1905]$.

---

### 5. Explainability, Clustering, Hidden Gaps & Scenario Calculations
- **TreeSHAP Local Attribution (`explainability/shap_explain.py`):** Exact additive decomposition computed on preprocessed features:
  $$\hat{y}_i = \text{base\_value} + \sum_{j=1}^{4} \text{SHAP}_{i,j}$$
  Global importance is $\text{mean}(|\text{SHAP}_{i,j}|)$ across districts.
- **Haversine DBSCAN Hotspots (`backend/services/hotspot_service.py`, `config.py`):**
  - Filters to districts where `predicted_category == 'High'` ($\hat{y} > 0.4526$).
  - Converts `(centroid_lat, centroid_lon)` to radians and clusters with `min_samples = 3` and angular radius:
    $$\epsilon = \frac{150.0\text{ km}}{6371.0\text{ km}} \approx 0.02354\text{ radians}$$
- **Hidden Gap Detector (`backend/services/residual_service.py`):**
  - Computes raw gap $(y_{\text{true}} - \hat{y}_{\text{LOSO}})$ and percentile rank gap $(\text{pct\_rank}(y_{\text{true}}) - \text{pct\_rank}(\hat{y}_{\text{LOSO}}))$.
  - Flags a district as `Hidden-deprivation` or `Resilient` only if **both** $y_{\text{true}}$ falls outside the 80% LOSO interval **and** $|\text{rank\_gap}| \ge 0.25$.
- **What-If Scenario Explorer (`backend/services/model_service.py`):**
  - Perturbs the district's 4-feature vector (e.g., `nightlight_mean * 1.25`, `builtup_pct + 1.0`, `ndvi_mean + 0.05`) and runs live inference to compute $\Delta = \text{scenario\_score} - \text{original\_score}$.
  - Compares $|\Delta|$ against the model's LOSO MAE (`0.0883`) and flags out-of-training-range inputs.
