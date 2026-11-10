# Methodology & Technical Specification

This document details the data acquisition, feature engineering, ground truth construction, geographic matching, validation architecture, spatial clustering, and statistical uncertainty modeling for **GeoPoverty AI**.

---

## 1. Data Sources, Temporal Window, and Telemetry

The final production model uses **4 features only**:
- `nightlight_mean` (VIIRS nocturnal luminosity)
- `ndvi_mean` (Sentinel-2 multispectral vegetation index)
- `builtup_pct` (GHSL built-up land percentage)
- `population_density` (GHSL human settlement population density)

All remote sensing telemetry is strictly anchored to the NFHS-5 survey period (**2019-11-01 to 2021-04-30**), with GHSL at epoch **2020**.

| Feature / Target | Source & Sensor | Temporal Window / Epoch | Native Resolution | Aggregation Metric |
|---|---|---|---|---|
| **Deprivation Ground Truth** | International Institute for Population Sciences (IIPS) NFHS-5 | 2019 – 2021 | District-level survey | 5-indicator composite |
| **Vegetation Index (`ndvi_mean`)** | ESA Sentinel-2 MSI Surface Reflectance (Harmonized) | 2019-11-01 to 2021-04-30 | 10 m (scaled to 100 m / 250 m) | District spatial mean |
| **Nocturnal Radiance (`nightlight_mean`)** | NOAA/NASA VIIRS Day/Night Band (DNB) Monthly Composites | 2019-11-01 to 2021-04-30 | 500 m | District spatial mean ($nW \cdot cm^{-2} \cdot sr^{-1}$) |
| **Built-Up Percentage (`builtup_pct`)** | JRC Global Human Settlement Layer (GHSL GHS-BUILT-S R2023A) | Epoch 2020 | 100 m | Percentage of district surface area (%) |
| **Population Density (`population_density`)** | JRC Global Human Settlement Layer (GHSL GHS-POP R2023A) | Epoch 2020 | 100 m | Spatial density ($\text{people} / km^2$) |

### Remote Sensing Resolution & NDVI Scale
- **Default Sentinel-2 Processing Scale:** District-wide median composites are computed via Google Earth Engine at a **100 m** resolution using Cloud Probability masking and Scene Classification Layer (SCL) filters (retaining only vegetation, bare soil, and water; rejecting clouds, cirrus, and shadows).
- **Refilled Districts Scale:** For districts requiring feature recovery via `scripts/refill_missing_features.py`, an export scale of **250 m** was used to resolve Earth Engine server-side user memory quota and timeout limits over extensive district polygons while preserving regional vegetative gradients.

### Explored Feature Extensions
- **Visual Features (VGG-16) & Grad-CAM:** Deep visual features (VGG-16) were evaluated only in an exploratory run on the original project dataset (data/exploratory/original_table_with_visual_features.csv; 2023 imagery). In that run, leave-one-state-out R2 was 0.3273 without the visual features (using 7 tabular features: 4 satellite + 3 accessibility columns, not the final 4-feature model) and 0.2879 with them, and visual features alone gave 0.0702 (models/exploratory_visual_ablation/ablation_results.csv). They were not re-evaluated on the final 2019-21 data, and the final model does not use them.
- **Accessibility Metrics (OpenStreetMap):** The accessibility columns in the original dataset (road density, school and hospital distance) could not be traced to an extraction script and were almost perfectly correlated with each other (|r| about 0.96); removing the accessibility columns changed R2 by about 0.003 (0.288 to 0.291) when the visual features were present, so they were not used. An OSM extraction script is included for future work but was not run nationally.

---

## 2. Ground Truth Target Construction & Sensitivity

### Deprivation Score Formulation
The continuous ground truth target variable (`deprivation_score` $\in [0, 1]$) is constructed from five core socioeconomic deprivation indicators reported in the NFHS-5 district fact sheets:
1. **Sanitation:** Percentage of households without improved sanitation facilities.
2. **Electricity:** Percentage of households without electricity access.
3. **Clean Cooking Fuel:** Percentage of households without clean fuel for cooking (LPG/biogas/electricity).
4. **Child Stunting:** Percentage of children under 5 years who are stunted (height-for-age $<-2$ standard deviations).
5. **Maternal Anaemia:** Percentage of pregnant women aged 15–49 years who are anaemic ($<11.0 \text{ g/dl}$).

Each raw indicator is min-max normalized across all surveyed districts:
$$I_k^{\text{norm}} = \frac{I_k - \min(I_k)}{\max(I_k) - \min(I_k)}$$

The aggregate district deprivation score is the unweighted arithmetic mean of the five normalized components:
$$\text{deprivation\_score} = \frac{1}{5} \sum_{k=1}^5 I_k^{\text{norm}}$$

### Sensitivity Analysis (from `data/ground_truth/score_sensitivity.md`)
Evaluation across 705 districts with all five NFHS-5 indicators confirms that the equal-weighted formulation is exceptionally robust and preserves the true underlying deprivation ranking:

| Score Variant | Spearman Rank Correlation ($\rho$) vs. Equal-Weight Score |
|---|---|
| **Equal weights (current score)** | **1.000** |
| **PC1 (first principal component, explains 49% variance)** | **0.996** |
| **Leave out clean fuel** | **0.932** |
| **Leave out stunting** | **0.974** |
| **Leave out sanitation** | **0.975** |
| **Leave out anaemia** | **0.975** |
| **Leave out electricity** | **0.988** |

*Sign-aligned Principal Component 1 (PC1) Loadings:* sanitation **0.514**, electricity **0.406**, clean cooking fuel **0.468**, child stunting **0.488**, maternal anaemia **0.338**.

---

## 3. Boundary Ingestion & State-Contained District Matching

Administrative boundaries are obtained from the FAO Global Administrative Unit Layers (GAUL 2015). Because GAUL was compiled in 2015 and NFHS-5 took place in 2019–2021, entity resolution is performed with strict state-boundary enforcement to prevent false cross-state matches:

1. **Name Normalization:** All state and district names are converted to canonical keys via `preprocessing/name_keys.py` (lowercased, punctuation/spaces stripped).
2. **State Alias Resolution:** Historical and orthographic state variants are harmonized via an alias mapping (e.g., `orissa` $\rightarrow$ `odisha`, `damananddiu` $\rightarrow$ `dadranagarhavelidamandiu`).
3. **Exact In-State Match:** Candidate pairs matching on both `(state_key, district_key)` are linked first.
4. **Bifurcation Fallback (Andhra Pradesh / Telangana):** Telangana was bifurcated from Andhra Pradesh in 2014 and remains listed under Andhra Pradesh in GAUL 2015. This is the **only** permitted cross-state fallback in the pipeline.
5. **Conservative Fuzzy Matching:** Remaining unlinked districts within the same state are resolved using token overlap and distinguished word tracking (e.g., preserving distinctions between `Bangalore Urban` and `Bangalore Rural`). Matching across state lines is strictly forbidden.
6. **Result:** **564 districts** across **31 states and union territories** are matched with zero duplicates (see [`data/processed/data_audit.md`](../data/processed/data_audit.md)).

---

## 4. Spatial Validation & Modeling Architecture

### Spatial Data Leakage Prevention (Tobler's First Law)
Standard random train/test cross-validation splits suffer from severe spatial autocorrelation leakage: adjacent districts share climate, topography, infrastructure, and administrative policies. Randomly allocating neighboring districts into train and test folds leads to artificially inflated accuracy (inflating $R^2$ by $+0.1268$ as shown in [`documentation/RESULTS.md`](RESULTS.md)).

GeoPoverty AI implements two rigorous spatial validation schemes:
1. **Leave-One-State-Out (LOSO) Cross-Validation (Primary):**
   - In each of the 31 iterations, all districts belonging to one administrative state are held out entirely.
   - The model is fitted strictly on the remaining 30 states and tested on the unseen state.
   - All displayed district scores in the application are **out-of-state estimates** from this process.
2. **Repeated State-Grouped 5-Fold Cross-Validation (Secondary):**
   - Districts are partitioned into 5 folds grouped by administrative state, repeated 5 times with distinct random seeds (`RepeatedGroupKFold`).

### Leakage-Free Scikit-Learn Pipelines
All data transformations are contained inside an end-to-end `sklearn.pipeline.Pipeline` (`training/pipelines.py`):
- **IQR Outlier Clipping:** Outlier thresholds are computed exclusively on the training fold features and applied downstream.
- **Imputation:** Executed as **training-fold median inside the pipeline** to prevent any data snooping or leakage from test folds.
- **Optional Nested Tuning:** When hyperparameter tuning is activated (`--tuning nested`), randomized hyperparameter searches are run strictly within the training folds of each CV iteration.

### Empirical Prediction Intervals
To provide honest uncertainty bounds for unobserved states, prediction intervals are computed directly from the empirical residuals of the Leave-One-State-Out cross-validation ($e_i = y_i - \hat{y}_i^{\text{LOSO}}$):
- **80% Prediction Interval:** $[\hat{y} - 0.1363, \hat{y} + 0.1437]$ (clipped to $[0, 1]$).
- **90% Prediction Interval:** $[\hat{y} - 0.1618, \hat{y} + 0.1905]$ (clipped to $[0, 1]$).

---

## 5. Spatial Hotspot Clustering

Spatial hotspot identification is executed via **Haversine DBSCAN with 150 km radius** in [`backend/services/hotspot_service.py`](../backend/services/hotspot_service.py):
- **Target Selection:** Filters districts whose predicted category is `'High'` deprivation (rather than applying an arbitrary fixed threshold).
- **Distance Metric:** Haversine metric on radians of district centroids:
  $$\text{eps} = \frac{150}{6371} \approx 0.02354 \text{ radians}, \quad \text{min\_samples} = 3$$
- **Cluster Output:** Identifies contiguous multi-district regional belts of severe deprivation, filtering out isolated geographical anomalies.

---

## 6. Limitations & Scientific Disclaimers

1. **Moderate Predictive Accuracy & Mean Compression:**
   The model achieves a Leave-One-State-Out $R^2$ of **0.3360** (MAE **0.0883**, see [`documentation/RESULTS.md`](RESULTS.md)). Predictions exhibit compression toward the mean: highly deprived districts are systematically under-estimated, while the most affluent districts are over-estimated.
2. **Sparse Tabular Feature Space:**
   The final production model utilizes only 4 tabular features (NDVI, nighttime luminosity, built-up percentage, and population density). Infrastructure accessibility metrics (distance to roads, hospitals, schools) and VGG-16 deep visual embeddings were not used.
3. **Historical Boundary Limitations:**
   GAUL administrative boundaries date from 2015. Consequently, 564 of the 705 NFHS-5 districts are represented. Regions with major administrative reorganizations (notably Jammu & Kashmir, Ladakh, and NCT Delhi) are excluded from the current fused dataset.
4. **Thin States & Small Sample Variance:**
   States with fewer than 3 districts (Goa with 2, Dadra & Nagar Haveli and Daman & Diu with 2, and Andaman & Nicobar Islands with 1) exhibit noisy Leave-One-State-Out errors because a single district can heavily distort state-level metrics.
5. **Composite Survey Label:**
   The ground truth deprivation score is a mathematical aggregate of five survey indicators, not a direct census measurement of household poverty or consumption expenditure.
6. **Correlation vs. Causation:**
   Feature attributions (SHAP values) represent model association, not real-world causal mechanisms. For example, higher nighttime luminosity is associated with lower deprivation in the model, but artificially adding lights will not automatically reduce poverty.
7. **What-If Simulation Scope:**
   The What-If scenario simulator reflects model inference under perturbed inputs; it does not simulate equilibrium socioeconomic interventions or macroeconomic policy impacts.
8. **Evaluation Discrepancy:**
   District scores displayed on district profile pages are out-of-state estimates, whereas live What-If simulations are computed using the final model fitted across all districts.


---

## 7. Scenario Explorer & Sensitivity Proxies

The **Scenario Explorer** (`/scenarios`, aliased to `/whatif`) provides an interactive simulation harness to inspect how the production model responds to systematic perturbations in satellite-derived features.

### 1. Sensitivity Proxies vs. Policy Interventions
Perturbations are defined and reported strictly as **proxies**, never policy interventions or causal simulations. The production model relies solely on four remote sensing proxies (`nightlight_mean`, `ndvi_mean`, `builtup_pct`, `population_density`). Real-world socioeconomic drivers such as education, public health infrastructure, child nutrition, and governance are not captured in the feature space.

### 2. Standardized Presets (`config.py`)
Six standardized proxy scenarios are configured in `config.py`:
- **Electrification Proxy (+25%)** (`ntl_plus_25`): Nighttime luminosity multiplied by 1.25x.
- **Electrification Proxy (+50%)** (`ntl_plus_50`): Nighttime luminosity multiplied by 1.50x.
- **Urbanisation Proxy (+1 pp)** (`builtup_plus_1`): Built-up surface percentage increased by +1.0 percentage point.
- **Urbanisation Proxy (+2 pp)** (`builtup_plus_2`): Built-up surface percentage increased by +2.0 percentage points.
- **Greening Proxy (+0.05)** (`ndvi_plus_05`): Sentinel-2 NDVI increased by +0.05.
- **Combined Growth Proxy** (`combined_growth`): Nighttime lights x1.25, Built-up +1.0 pp, NDVI +0.05.

### 3. Group Aggregations (States & Hotspot Clusters)
Users can evaluate scenarios for a single district, across all districts of an administrative state (31 states/UTs), or across contiguous high-deprivation spatial clusters identified by Haversine DBSCAN (150 km radius). Group evaluations compute:
- **Mean Change & Median Change:** Distributional shifts across the target region.
- **Category Transitions:** Count and percentage of districts transitioning across deprivation tiers (Low, Medium, High).
- **Sub-MAE Insignificance Share:** Percentage of districts where |Delta| < MAE (0.0883). Changes smaller than the model's typical cross-validation error are explicitly marked as *within noise threshold*.

### 4. Bounding, Uncertainty & Mandatory Disclaimer
- **Training Range Guardrails:** Any perturbed value exceeding the empirical training range triggers explicit non-extrapolation warnings.
- **Prediction Intervals:** Every scenario estimate displays the calibrated 80% empirical prediction interval from `models/uncertainty.json` ([-0.1363, +0.1437]).
- **Mandatory System Note:** All endpoints and views enforce the explicit disclaimer:
  > *"This shows how the model's estimate responds to changes in satellite-derived features. It is not a prediction of the effect of a policy or programme. Schooling, health, nutrition and infrastructure are not in the model."*


---

## 8. District Indicator Telemetry & Ground Truth Specifications

All district reports, profile views, and APIs expose real, empirical indicator telemetry. Values are never imputed with synthetic constants, never defaulted to 0.0, and never entered manually.

### 8.1 Satellite-Derived Indicators (7)
Extracted across administrative boundaries for all 564 districts in the training table:

| Indicator (`key`) | Unit | India Median | Min | Max | Source | Period | Definition |
|---|---|---|---|---|---|---|---|
| **Nighttime Light Radiance** (`nightlight_mean`) | `nW/cm2/sr` | 0.8710 | 0.2257 | 45.0564 | VIIRS monthly | 2019-11-01 to 2021-04-30 | District-wide nocturnal luminosity spatial mean |
| **Normalized Difference Vegetation Index (NDVI)** (`ndvi_mean`) | `index (-1 to 1)` | 0.4266 | 0.0795 | 0.8023 | Sentinel-2 SR median NDVI | 2019-11-01 to 2021-04-30 | District spatial median Normalized Difference Vegetation Index |
| **Built-Up Surface Area** (`builtup_pct`) | `% of land area` | 1.2974 | 0.0075 | 40.5230 | GHSL built-up surface (epoch 2020) | Epoch 2020 | Spatial percentage of land covered by built-up settlements |
| **Population Density** (`population_density`) | `people per km2` | 424.5 | 2.5 | 44741.3 | GHSL population grid (epoch 2020) | Epoch 2020 | Spatial human residential population density |
| **Cloud-Free Sentinel-2 Scenes** (`s2_image_count`) | `scenes` | 286 | 18 | 1677 | usable Sentinel-2 scenes in window | 2019-11-01 to 2021-04-30 | Total usable cloud-free Sentinel-2 MSI observations |
| **Centroid Latitude** (`latitude`) | `degrees` | 24.2595 | 7.5191 | 32.6557 | district centroid | GAUL 2015 | Centroid administrative latitude coordinate |
| **Centroid Longitude** (`longitude`) | `degrees` | 79.4415 | 69.8004 | 96.3563 | district centroid | GAUL 2015 | Centroid administrative longitude coordinate |

### 8.2 Survey Components of Deprivation Score (NFHS-5)
The 5 component indicators joined via `clean_key` across 100% of the 564 districts (all 705 national survey districts evaluated for distribution statistics):

| # | Indicator Label | Better Direction | National Median | Min | Max | Target Formulation Description |
|---|---|---|---|---|---|---|
| **#7** | Population living in households with electricity (%) | `higher` | 98.7% | 68.4% | 100.0% | Indicator 7: Electricity access percentage |
| **#9** | Population living in households that use an improved sanitation facility (%) | `higher` | 73.7% | 29.2% | 99.9% | Indicator 9: Sanitation facility percentage |
| **#10** | Households using clean fuel for cooking (%) | `higher` | 50.0% | 8.6% | 99.8% | Indicator 10: Clean cooking fuel percentage (LPG/biogas/electricity) |
| **#73** | Children under 5 years who are stunted (height-for-age) (%) | `lower` | 32.9% | 13.2% | 60.6% | Indicator 73: Under-5 child stunting prevalence (< -2 SD) |
| **#84** | All women age 15-49 years who are anaemic (%) | `lower` | 57.2% | 14.9% | 93.5% | Indicator 84: Anaemia prevalence in reproductive-age women (< 11.0 g/dl) |

> **Target Integration Rule:** In the composite deprivation score, electricity (#7), improved sanitation (#9), and clean cooking fuel (#10) enter as $100 - x$ (inverted so higher signifies greater deprivation), whereas child stunting (#73) and maternal anaemia (#84) enter directly. Each component is min-max normalized across all districts before unweighted averaging.


---

## 9. Hidden Gap Detector (Residual Atlas)

The **Hidden Gap Detector** (`/hidden-gaps`, `/api/residual-atlas`) cross-examines the NFHS-5 survey deprivation score ($y_i$) of each district against the out-of-state satellite estimate ($\hat{y}_i^{\text{LOSO}}$) produced when the district's entire state was withheld during Leave-One-State-Out cross-validation.

### 9.1 Formulation & Two-Stage Candidate Screening Criteria
The residual gap and percentile-rank gap are defined as:
$$\text{gap}_i = y_i - \hat{y}_i^{\text{LOSO}}$$
$$\text{rank\_gap}_i = \text{rank}(y_i) - \text{rank}(\hat{y}_i^{\text{LOSO}}) \quad \in [-1, +1]$$

Because model predictions compress toward the national mean, the raw residual gap is strongly correlated with the survey score ($r = 0.787$). Relying on residual magnitude or uncertainty bounds alone would label almost every severely deprived district as "hidden deprivation" and every least-deprived district as "resilient."

To isolate structural discrepancies from mean compression, a district is classified as a **candidate** only when **BOTH** conditions hold simultaneously:
1. **Uncertainty Interval Exceedance:** The survey score lies strictly outside the model's 80% out-of-state empirical prediction interval ($y_i < \text{low\_80}_i$ or $y_i > \text{high\_80}_i$).
2. **Percentile-Rank Threshold:** The percentile-rank gap is at least 0.25 (25 percentile points) in the same direction:
   - **Hidden-deprivation candidate:** $y_i > \text{high\_80}_i$ AND $\text{rank\_gap}_i \ge +0.25$ (district is ranked at least 25 percentile points higher in deprivation than satellite telemetry predicts).
   - **Resilient candidate:** $y_i < \text{low\_80}_i$ AND $\text{rank\_gap}_i \le -0.25$ (district is ranked at least 25 percentile points lower in deprivation than satellite telemetry predicts).
   - **Consistent:** All other districts (475 of 564, or 84.2%).

### 9.2 Empirical Distribution Across India (564 Districts)
- **Total evaluated districts ($N$):** 564
- **Hidden-deprivation candidates:** 43
- **Resilient candidates:** 46
- **Consistent districts:** 475
- **Districts outside 80% prediction interval:** 114 (20.21%)
- **Residual gap standard deviation ($\sigma_{\text{gap}}$):** 0.1088
- **Survey vs. prediction correlation ($r$):** 0.581
- **Residual gap vs. survey correlation ($r$):** 0.787

### 9.3 Methodological Limits & Screening Scope
- **Screening Hint, Not Proof:** A candidate classification reflects statistical disagreement between remote sensing indicators and survey metrics. It is a screening hint for local administrative verification, not proof of hidden poverty or of a causal deficit.
- **Uncaptured Socioeconomic Dimensions:** The satellite model strictly observes physical settlement proxies (nighttime luminosity, vegetation cover, built-up surfaces, and population density). Discrepancies often stem from socioeconomic factors that satellites cannot observe, such as nutritional programs, schooling access, primary healthcare delivery, or localized social schemes.
- **Expected Statistical Variation:** For a calibrated model with an 80% interval, approximately 20% of districts naturally fall outside the bounds (here 20.21%). Interval exceedance alone is not anomalous.
- **Small-State Reliability:** States with fewer than 5 districts (e.g., Goa with 2, Dadra & Nagar Haveli and Daman & Diu with 2, and Andaman & Nicobar Islands with 1) exhibit high Leave-One-State-Out error variance and are explicitly marked as unreliable in state-level systematic bias tables.
- **Aggregate Level Only:** Analyses apply exclusively to administrative district aggregates. No household- or person-level inference is possible.
