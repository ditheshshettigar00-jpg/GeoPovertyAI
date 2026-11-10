# Pipeline Development History: Phase 1 & Phase 2

> **Historical Context:** This document records the historical implementation notes and architectural evolution across Phase 1 (Data Integrity) and Phase 2 (Modelling & Evaluation) of GeoPoverty AI.

---

## 1. Phase 1: Data Integrity & Harmonization

### Core Objectives & Refactorings
- **GHSL Sensor Unit Harmonization:**
  - Corrected native Global Human Settlement Layer units: `builtup_pct` is calibrated to true district land percentage ($[0.0, 100.0]\%$), and `population_density` is calibrated to people per $\text{km}^2$.
  - Fixed epoch to **2020** to align with the NFHS-5 survey period.
- **Survey Window Alignment:**
  - Strict temporal bounding of Sentinel-2 MSI and VIIRS DNB monthly composites to **2019-11-01 through 2021-04-30**, synchronizing earth observation features with NFHS-5 national demographic fieldwork.
- **Geography & In-State District Boundary Matching:**
  - Built canonical district name normalization and matching rules (`preprocessing/name_keys.py`) incorporating alias lookups, exact matching, state boundary confinement, and an explicit Andhra Pradesh $\rightarrow$ Telangana historical partition fallback.
  - Reconciled **564 valid districts** (increased from 532 in earlier iterations) against the FAO GAUL administrative polygon database with 0 duplicate target mappings.
- **Integrity Validation & Audit Tooling:**
  - Added `scripts/audit_data.py` to audit feature distributions, coordinate boundaries, and range limits.
  - Added `score_sensitivity.py` to evaluate alternative composite weighting schemes against the equal-weight formulation.
- **Accessibility Feature Verification:**
  - Identified that preliminary OpenStreetMap accessibility columns (road density, school distance, hospital distance) in earlier experimental tables were almost perfectly collinear ($|r| \approx 0.96$) and lacked verifiable provenance. Confined production modeling strictly to verified satellite telemetry.

---

## 2. Phase 2: Modeling, Spatial Validation & Evaluation

### Core Objectives & Architecture
- **Leave-One-State-Out (LOSO) Cross-Validation:**
  - Designed spatial holdout partitioning to counter spatial autocorrelation leakage (Tobler's First Law). Each test fold evaluates an entire administrative state unseen during training.
  - Added repeated state-grouped 5-fold cross-validation to assess variance across geographic partitions.
- **Leakage-Free Pipeline Architecture:**
  - Integrated outlier clipping ($3.0 \times \text{IQR}$) and missing value imputation strictly inside the `scikit-learn` Pipeline transformers (`training/pipelines.py`).
  - Imputation is performed dynamically using the **training-fold median**; testing folds never inform imputation parameters.
- **Baseline Benchmarking:**
  - Evaluated a "Dummy" mean predictor (predicting the training mean of other states) as the fundamental hurdle.
  - Standardized features within linear pipelines, resolving negative $R^2$ scaling artifacts seen in unscaled linear regressions.
- **Prediction Uncertainty & Interval Estimation:**
  - Derived empirical conformal prediction error intervals (80% and 90% bounds) from out-of-state cross-validation residuals (`models/uncertainty.json`).
  - Wrapped inference in `training/inference.py:predict_with_interval(bundle, X)` to serve interval estimations alongside point predictions.
- **Ablation & Explored Visual Extensions:**
  - Conducted systematic feature ablation across tabular indicators and explored VGG-16 deep convolutional features.
  - Established that deep visual features evaluated on exploratory imagery degraded generalization performance under strict Leave-One-State-Out cross-validation (tabular-only $R^2 = 0.327$ vs. tabular + visual $R^2 = 0.288$), confirming that the 4 tabular remote-sensing indicators provide superior out-of-state generalization.
- **Categorical Tertile Boundaries:**
  - Established empirical tertile cut-offs (`low_medium_boundary`, `medium_high_boundary`) from the continuous target distribution, serializing them into `models/category_thresholds.json`.

---

## 3. Historical Reference Artifacts

### Key Output Files in `models/`
- `best_model.pkl`: Fitted Pipeline bundle (`RandomForest`, imputer, clipper, scaler, prediction interval bounds).
- `model_comparison_results.csv`: LOSO benchmark comparing Dummy, Ridge, RandomForest, GradientBoosting, and XGBoost.
- `repeated_groupkfold_results.csv`: Repeated state-grouped 5-fold cross-validation metrics.
- `leaky_random_kfold_results.csv`: Empirical demonstration of spatial autocorrelation leakage under standard random K-fold splits.
- `ablation_results.csv`: Feature omission sensitivity metrics.
- `uncertainty.json`: Conformal residual quantiles for 80% and 90% prediction intervals.
- `category_thresholds.json`: Tertile boundaries for Low, Medium, and High deprivation classifications.
