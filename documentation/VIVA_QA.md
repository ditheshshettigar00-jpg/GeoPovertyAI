# GeoPoverty AI: Comprehensive Viva Q&A Guide

This guide prepares you to defend **GeoPoverty AI** in front of external examiners and technical evaluators. Every answer is grounded in the actual codebase, empirical benchmarks, and statistical evaluations of the repository.

---

## 📌 Numbers to Memorize

| Metric / Parameter | Value | Source / File Reference |
|---|---|---|
| **Best Model & Algorithm** | **RandomForestRegressor** (100 trees, default depth) | [`models/run_metadata.json`](../models/run_metadata.json) |
| **Primary Evaluation Scheme** | **Leave-One-State-Out (LOSO)** CV (31 folds) | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Model Performance ($R^2$)** | **0.3360** (explains 33.6% out-of-state variance) | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Mean Absolute Error (MAE)** | **0.0883** (deprivation score scale: 0.0 to 1.0) | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Root Mean Squared Error (RMSE)**| **0.1087** | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Spearman Rank Correlation ($\rho$)**| **0.5745** (monotonic ranking agreement) | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Tertile Categorical Accuracy** | **54.61%** (3-class: Low / Medium / High) | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Top-20% Deprivation Recall** | **46.02%** (identifies most deprived fifth) | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Dummy Baseline MAE** | **0.1142** (predicting training mean of other states) | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Improvement over Baseline** | **22.7%** MAE reduction ($0.0883$ vs $0.1142$) | [`models/model_comparison_results.csv`](../models/model_comparison_results.csv) |
| **Spatial Leakage Penalty ($R^2$)** | **+0.1268** (+37.7% artificial boost under random 5-fold: $0.4628$ vs $0.3360$) | [`documentation/RESULTS.md`](RESULTS.md) |
| **Dataset Scope** | **564 districts** across **31 states / UTs** | [`data/processed/data_audit.md`](../data/processed/data_audit.md) |
| **Production Features (4)** | `nightlight_mean`, `ndvi_mean`, `builtup_pct`, `population_density` | [`backend/services/model_service.py`](../backend/services/model_service.py) |
| **Tertile Category Cutoffs** | Low: $\le 0.3198$, Medium: $0.3198 - 0.4526$, High: $> 0.4526$ | [`models/category_thresholds.json`](../models/category_thresholds.json) |
| **80% Prediction Interval** | $[-0.1363, +0.1437]$ (empirical LOSO residual quantiles) | [`models/uncertainty.json`](../models/uncertainty.json) |
| **90% Prediction Interval** | $[-0.1618, +0.1905]$ | [`models/uncertainty.json`](../models/uncertainty.json) |
| **Hotspot Clustering** | Haversine DBSCAN ($\text{eps} = 150/6371 \approx 0.02354$ rad, $\text{min\_samples} = 3$) | [`backend/services/hotspot_service.py`](../backend/services/hotspot_service.py) |
| **Hidden Gap Candidates** | Outside 80% interval AND $|\text{rank\_gap}| \ge 0.25$ (43 hidden, 46 resilient) | [`backend/services/residual_service.py`](../backend/services/residual_service.py) |
| **Ground Truth Target** | Equal-weighted composite of 5 min-max NFHS-5 indicators (PC1 $\rho = 0.996$) | [`data/ground_truth/score_sensitivity.md`](../data/ground_truth/score_sensitivity.md) |
| **Temporal Remote Sensing Window** | **2019-11-01 to 2021-04-30** (matching NFHS-5 fieldwork); GHSL epoch **2020** | [`documentation/METHODS.md`](METHODS.md) |
| **Automated Test Suite** | **270 passed** tests across 10 test modules | [`tests/`](../tests/) |

---

## ⚠️ Examiner Traps to Avoid

| ❌ What NOT to Say | ✅ What to Say Instead (Scientifically Rigorous) |
|---|---|
| *"Our AI model achieves 90% accuracy in predicting poverty."* | *"Our regression model achieves an $R^2$ of 0.3360 and MAE of 0.0883 under strict Leave-One-State-Out cross-validation across 564 districts, outperforming the baseline by 22.7%. When discretized into tertiles, categorical accuracy is 54.6%."* |
| *"We used deep learning (VGG-16 CNN) to classify satellite images."* | *"We conducted an exploratory ablation on VGG-16 image embeddings on 2023 data, but visual features alone achieved only $R^2 = 0.0702$ and degraded tabular performance from 0.3273 to 0.2879. Our production model relies on 4 physical remote sensing telemetry features anchored to the 2019–2021 survey period."* |
| *"The Hidden Gap Detector proves that the government underreported poverty in certain districts."* | *"The Hidden Gap Detector is an exploratory screening tool based on statistical divergence between satellite physical proxies and survey indicators (outside the 80% prediction interval and $\ge 25$ percentile rank gap). It flags candidates for field audits, not proof of misreporting."* |
| *"The What-If tool simulates the impact of government poverty alleviation schemes."* | *"The Scenario Explorer is a sensitivity harness showing how the statistical model responds to input perturbations. It is strictly non-causal; real-world socioeconomic dimensions such as education, healthcare, and welfare schemes are not in the feature space."* |
| *"We used standard random 80/20 train/test split."* | *"Random train/test splits severely violate Tobler's First Law of Geography through spatial autocorrelation, artificially leaking regional climate and governance into test folds and inflating $R^2$ by +0.1268. We enforce Leave-One-State-Out cross-validation to guarantee geographic generalization."* |
| *"Our predictions are real-time live satellite feeds."* | *"Our models are trained on satellite data temporally aligned with NFHS-5 fieldwork (2019–2021). Predictions for all 564 districts are precomputed from out-of-state cross-validation and served from SQLite for rapid, deterministic query latency."* |

---

## Section A: Machine Learning & Methodology

### Q1. Why did you choose Leave-One-State-Out (LOSO) cross-validation instead of standard k-fold cross-validation?
**Answer:** Standard random cross-validation randomly assigns geographically adjacent districts into training and testing sets. Due to spatial autocorrelation—formalized by Tobler's First Law of Geography ("near things are more related than distant things")—neighboring districts share climate, topography, infrastructure, and state-level governance policies. This causes massive spatial data leakage, yielding artificially inflated performance. In our experiments, random 5-fold CV inflated $R^2$ by $+0.1268$ (+37.7%, from 0.3360 to 0.4628) as documented in [`documentation/RESULTS.md`](RESULTS.md). Leave-One-State-Out cross-validation completely holds out an entire administrative state in each iteration, guaranteeing that the model is evaluated strictly on regions whose state administration and localized geography it has never encountered.

### Q2. Is an $R^2$ of 0.3360 good for this task? How do you justify it?
**Answer:** Yes, an $R^2$ of 0.3360 is a respectable, honest result for cross-state socioeconomic deprivation prediction from space. We are predicting a complex multidimensional ground-truth composite (health, nutrition, sanitation, clean fuel, and electrification) across 564 districts in 31 Indian states using only four satellite-derived aggregate features. The model achieves an MAE of 0.0883, which beats the dummy baseline (predicting the mean deprivation of other states, MAE 0.1142) by 22.7% ([`models/model_comparison_results.csv`](../models/model_comparison_results.csv)). Moreover, it achieves a Spearman rank correlation of 0.5745, demonstrating strong ability to rank districts from least to most deprived for targeted resource prioritization.

### Q3. Why did RandomForest outperform Ridge regression, Gradient Boosting, and XGBoost?
**Answer:** According to our benchmark results in [`models/model_comparison_results.csv`](../models/model_comparison_results.csv), RandomForest achieved $R^2 = 0.3360$, outperforming XGBoost ($0.3074$), GradientBoosting ($0.2951$), Ridge ($0.2700$), and Dummy ($-0.0565$). Linear models like Ridge fail to capture non-linear saturation thresholds inherent in remote sensing telemetry (e.g., nighttime luminosity saturates rapidly once basic electrification is reached, while vegetation index displays non-linear seasonal correlations with agricultural poverty). Meanwhile, boosted tree ensembles (XGBoost and GBDT) are prone to subtle overfitting on smaller tabular datasets ($N = 564$) during leave-one-group-out splits, whereas Random Forest's bootstrap aggregation (bagging) over 100 decorrelated trees provides superior variance reduction across diverse state geographies.

### Q4. Why did you restrict the production model to 4 tabular features instead of using deep learning (CNN) on satellite imagery?
**Answer:** We conducted a systematic ablation study on visual features (VGG-16 embeddings) documented in [`models/exploratory_visual_ablation/ablation_results.csv`](../models/exploratory_visual_ablation/ablation_results.csv). In that exploratory evaluation, visual embeddings alone achieved an $R^2$ of only 0.0702, and concatenating visual features to tabular features actually degraded $R^2$ from 0.3273 down to 0.2879. Furthermore, the available satellite imagery patches were from 2023, creating a severe 3-year temporal mismatch with the NFHS-5 survey ground truth (2019–2021). The four physical tabular indicators (`nightlight_mean`, `ndvi_mean`, `builtup_pct`, `population_density`) provide transparent physical interpretations, align precisely with the survey window, and run reliably without heavy GPU dependencies.

### Q5. Why were the OpenStreetMap accessibility features (road density, hospital/school distance) dropped from the final model?
**Answer:** The accessibility metrics present in early exploratory data could not be traced to a reproducible, documented extraction pipeline, and an exploratory collinearity audit revealed extreme multicollinearity ($|r| \approx 0.96$) among them. In ablation testing ([`models/exploratory_visual_ablation/ablation_results.csv`](../models/exploratory_visual_ablation/ablation_results.csv)), removing accessibility columns changed $R^2$ by only 0.003 (from 0.2879 to 0.2908), indicating negligible unique signal. Because OpenStreetMap feature completeness in rural India varies widely by district volunteer activity rather than actual physical infrastructure, relying on untracked OSM data introduced uncalibrated reporting bias.

### Q6. How was the continuous ground truth deprivation score constructed, and is equal weighting justified?
**Answer:** The ground truth score $\in [0.0, 1.0]$ is constructed from five published district-level indicators in NFHS-5 (2019–2021): household electricity access (Indicator 7), improved sanitation (Indicator 9), clean cooking fuel (Indicator 10), under-5 child stunting (Indicator 73), and women's anaemia (Indicator 84). The first three are inverted ($100 - x$) so higher values signify greater deprivation, and all five are min-max normalized across all districts before equal-weighted averaging ([`data/ground_truth/GROUND_TRUTH_README.md`](../data/ground_truth/GROUND_TRUTH_README.md)). To verify that equal weighting was not arbitrary, we performed sensitivity analysis against Principal Component Analysis (PCA) in [`data/ground_truth/score_sensitivity.md`](../data/ground_truth/score_sensitivity.md): the first principal component (PC1) explains 49% of indicator variance, and its Spearman rank correlation with our equal-weighted score across 705 districts is an astounding 0.996. Leave-one-indicator-out rank correlations all exceed 0.932, confirming ranking robustness.

### Q7. How are prediction intervals computed, and what is their statistical meaning?
**Answer:** Prediction intervals are derived empirically from the Leave-One-State-Out cross-validation residuals ($e_i = y_i - \hat{y}_i^{\text{LOSO}}$), as stored in [`models/uncertainty.json`](../models/uncertainty.json). For an 80% coverage interval, we calculate the 10th and 90th percentiles of out-of-state prediction errors across all 564 districts, yielding the additive bounds $[-0.1363, +0.1437]$. Unlike theoretical Gaussian intervals that assume homoscedastic normal errors, these empirical conformal bounds reflect the actual historical distribution of errors made by the model when predicting entirely unseen states.

### Q8. What is the Hidden Gap Detector, and why does it require both an uncertainty exceedance and a rank gap $\ge 0.25$?
**Answer:** The Hidden Gap Detector ([`backend/services/residual_service.py`](../backend/services/residual_service.py)) flags districts where satellite-predicted physical development strongly diverges from survey-measured human deprivation. Because statistical regression models suffer from mean compression (under-estimating high deprivation and over-estimating low deprivation), raw residuals correlate strongly with the target ($r = 0.787$). If we only checked interval exceedance, almost every severely impoverished district would be erroneously flagged. To isolate genuine structural anomalies from mathematical shrinkage, a district must satisfy two criteria simultaneously: (1) its survey score must lie strictly outside the 80% out-of-state prediction interval, AND (2) its percentile-rank gap must be at least 25 percentile points ($|\text{rank\_gap}| \ge 0.25$). Across India, 43 districts are classified as hidden deprivation candidates and 46 as resilient candidates, while 475 districts (84.2%) remain consistent.

---

## Section B: Geospatial & Data Engineering

### Q9. Why was the remote sensing temporal window strictly fixed to 2019-11-01 through 2021-04-30?
**Answer:** This temporal window corresponds exactly to the national fieldwork phases of the National Family Health Survey (NFHS-5), which were conducted between November 2019 and April 2021. Remote sensing measurements are dynamic: nighttime lights fluctuate with economic activity, and vegetation indices vary with monsoons and drought cycles. Anchoring Sentinel-2 MSI and NOAA/NASA VIIRS satellite telemetry to the exact survey period prevents temporal misalignment where post-COVID or infrastructure changes in 2023 would be erroneously correlated with 2019–2021 survey data ([`documentation/METHODS.md`](METHODS.md)).

### Q10. Why are there 564 districts in the training table when NFHS-5 surveyed 705 districts?
**Answer:** Administrative boundary definitions in India change frequently due to district bifurcations and reorganizations. We used the Food and Agriculture Organization (FAO) Global Administrative Unit Layers (GAUL 2015) polygon dataset for spatial boundary extractions. GAUL 2015 reflects 2015 administrative borders, whereas NFHS-5 reflects 2019–2021 borders. Regions that experienced fundamental administrative restructuring (such as the reorganization of Jammu & Kashmir and Ladakh into Union Territories, and NCT Delhi's nine-district layout) could not be unambiguously linked without risking cross-boundary distortion. All 564 matched districts are verified with 100% duplicate-free integrity in [`data/processed/data_audit.md`](../data/processed/data_audit.md).

### Q11. How did you match district names between GAUL polygons and NFHS-5 survey sheets?
**Answer:** Matching is performed through a state-contained entity resolution algorithm in [`preprocessing/name_keys.py`](../preprocessing/name_keys.py). Both datasets are first converted to alphanumeric lowercase keys stripped of whitespace and punctuation. State names are harmonized across historical orthographic variants (e.g., `orissa` $\rightarrow$ `odisha`). Crucially, candidate district matches are strictly restricted to the same state to prevent false cross-state linkages (e.g., "Bilaspur" exists in both Chhattisgarh and Himachal Pradesh). The only permitted cross-state fallback is between Andhra Pradesh and Telangana, which were bifurcated in 2014 and combined in GAUL 2015.

### Q12. How was cloud masking implemented for Sentinel-2 surface reflectance?
**Answer:** As detailed in [`documentation/METHODS.md`](METHODS.md), optical Sentinel-2 scenes over each district polygon were filtered using Google Earth Engine. First, scenes with 20% or more cloudy pixels across the scene footprint were discarded. Second, the Sentinel-2 Scene Classification Layer (SCL band) was used to mask out pixels that were not classified as vegetation (class 4), non-vegetated surface (class 5), water (class 6), or unclassified land (class 7). The median NDVI across all valid, cloud-free observations within the temporal window was then computed for each district polygon.

### Q13. Why was a 250 m scale used for 48 districts while the remaining districts used 30 m native scale for NDVI?
**Answer:** Sentinel-2 Surface Reflectance has a native spatial resolution of 10–20 meters, and Google Earth Engine standard extractions for medium-sized districts operated at a 30-meter pixel scale. However, very large district polygons (such as Kutch in Gujarat, Leh in Ladakh, or extensive desert districts in Rajasthan) exceeded Google Earth Engine server-side user memory limits and triggered computational timeouts during multi-year median reductions. For the 48 large districts refilled via `scripts/refill_missing_features.py`, an export scale of 250 meters was applied. This preserved macro-level vegetative gradients while operating reliably within Earth Engine memory quotas ([`documentation/METHODS.md`](METHODS.md)).

### Q14. What are the GHSL layers, and why did you use the 2020 epoch?
**Answer:** The Global Human Settlement Layer (GHSL) is produced by the European Commission's Joint Research Centre (JRC). We used two products: `GHS-BUILT-S R2023A` (built-up surface area fraction at 100 m resolution) and `GHS-POP R2023A` (spatial population count per grid cell at 100 m resolution). We selected epoch 2020 because it falls squarely in the middle of the NFHS-5 survey period (2019–2021), providing an accurate snapshot of human settlement footprints and residential density without temporal distortion.

---

## Section C: Software Engineering & Architecture

### Q15. Why did you use Flask 3, vanilla JavaScript, and Jinja2 instead of a heavy SPA framework like React or Next.js?
**Answer:** For a policy decision-support and geospatial research platform, server-side rendering (SSR) via Flask and Jinja2 provides immediate first-contentful paint, zero client-side compilation or node tooling overhead, and robust URL shareability across administrative offices. Vanilla modern ES6 JavaScript delivers clean, dependency-free interactive visualisations (using Leaflet for choropleth mapping and Chart.js for distribution plots) without the bloat, build steps, or dependency vulnerabilities of large single-page application frameworks.

### Q16. Why did you precompute district predictions into SQLite rather than running model inference live on every page load?
**Answer:** District predictions displayed on map views, comparisons, and district profile pages are derived from Leave-One-State-Out cross-validation. In this scheme, predicting a district requires loading the specific model fold trained without that district's state. Running 31 random forest ensemble models dynamically on every HTTP request would create latency and consume server CPU memory unnecessarily. Precomputing predictions using `scripts/precompute_predictions.py` into `geopoverty.db` ensures sub-10 millisecond database query response times, full offline capability, and deterministic reproducibility. Live inference is reserved exclusively for the Scenario Explorer (`/scenarios`), where users apply custom feature perturbations.

### Q17. How does SQLite handle concurrent access in this application, and how did you prevent "database is locked" errors?
**Answer:** SQLite operates in WAL (Write-Ahead Logging) mode, which allows multiple concurrent readers while serializing write operations. In [`backend/services/db_service.py`](../backend/services/db_service.py), database connections use explicit timeout parameters and short-lived transactional scopes. Every write operation (such as user registration or status update) is wrapped in `try...except` blocks that execute explicit `conn.rollback()` on exceptions before propagating errors. This guarantees that uncommitted locks are never left lingering on connections, preventing downstream reader or writer deadlocks during automated tests or high-concurrency requests.

### Q18. Explain your Role-Based Access Control (RBAC) design and the researcher approval workflow.
**Answer:** The application implements a three-tier RBAC architecture: `PUBLIC`, `RESEARCHER`, and `ADMIN`. Public visitors can view the national choropleth map, district profiles, and executive summaries. Advanced analytical capabilities—such as the Hidden Gap Detector, DBSCAN Hotspot Analysis, District Comparison tool, and Scenario Explorer—are restricted to approved researchers. When a user registers for a researcher account, their account is placed in a `PENDING` state with access restricted to public tiers. An administrator must log into the Admin Dashboard (`/admin`) to verify the user's institutional email and credentials before approving them to `RESEARCHER` status ([`documentation/SECURITY_NOTES.md`](SECURITY_NOTES.md)).

### Q19. How did you harden the application against common security vulnerabilities like SQL injection, CSRF, and path traversal?
**Answer:** As detailed in [`documentation/SECURITY_NOTES.md`](SECURITY_NOTES.md):
1. **SQL Injection:** All SQLite queries in `backend/services/db_service.py` strictly utilize parameterized `?` placeholders; dynamic SQL column names in update queries are checked against strict hardcoded whitelists.
2. **Path Traversal:** File serving endpoints (such as satellite imagery) validate input against integer regex converters (`<int:area_id>`) and enforce canonical path prefix verification (`os.path.abspath(path).startswith(SATELLITE_DIR)`).
3. **Session & CSRF Security:** Session cookies enforce `HttpOnly`, `SameSite=Lax`, and dynamic `Secure` flags (enabled automatically in production). Password hashing utilizes PBKDF2-SHA256 with 600,000 iterations.
4. **Rate Limiting:** Login endpoints enforce a 5-failure threshold per 15-minute window per IP, with client IP extraction secured against spoofed `X-Forwarded-For` headers unless `TRUST_PROXY=1` is explicitly configured.

### Q20. What is your automated test strategy, and why do you run pytest twice?
**Answer:** Our test suite comprises **270 automated tests** across 10 modules covering unit logic, service integration, access control, security edge cases, PDF generation, and report rendering. Running `pytest` twice sequentially verifies state isolation and database connection hygiene: if any service fails to close a connection or release an SQLite transaction lock, the second test run immediately fails with a "database is locked" error. Achieving 270 passed tests back-to-back confirms complete isolation and test idempotency.

---

## Section D: System Limitations & Scientific Honesty

### Q21. What is "mean compression," and how does it affect your model's predictions?
**Answer:** Mean compression (or regression shrinkage toward the mean) is an inherent property of least-squares and mean-squared-error minimization algorithms. When faced with residual uncertainty, the model hedges its predictions toward the overall sample mean ($\approx 0.38$). Consequently, the model systematically under-estimates deprivation in the most severely impoverished districts (e.g., in parts of Jharkhand and rural Bihar) and over-estimates deprivation in the most affluent districts (e.g., in Goa and urban Kerala). We explicitly communicate this limitation in district executive reports and report uncertainty bounds so policymakers understand that point predictions for extreme districts are attenuated.

### Q22. Why do "thin states" (like Goa or Tripura) have higher prediction errors in Leave-One-State-Out CV?
**Answer:** In Leave-One-State-Out cross-validation, the entire state's error is computed as the average error across its constituent districts. In "thin states" with very few districts—such as Goa ($N=2$), Dadra & Nagar Haveli ($N=2$), or Andaman & Nicobar ($N=1$)—a single district with idiosyncratic local topography or coastal economics will heavily skew the entire state's error metric. As recorded in [`documentation/RESULTS.md`](RESULTS.md), Goa exhibits an MAE of 0.1769, largely because its two districts are atypical compared to the rest of India. In larger states (e.g., Uttar Pradesh with 71 districts), idiosyncratic errors average out.

### Q23. Are SHAP feature attribution values causal? Can we tell a policymaker that increasing nightlights will eradicate poverty?
**Answer:** Absolutely not. SHAP (SHapley Additive exPlanations) values quantify the mathematical contribution of each feature to the model's prediction relative to the expected baseline value ($\hat{y} = \phi_0 + \sum_{j=1}^M \phi_j$). They measure statistical association within the fitted random forest, not physical or economic causality. For example, nighttime luminosity has a large negative SHAP value (associating higher lights with lower deprivation), but simply distributing streetlights or floodlights will not improve household sanitation, eradicate maternal anaemia, or reduce child stunting. We prominently display this non-causal disclaimer on all explanation dashboards.

### Q24. What are the ethical implications and risks of satellite-based poverty mapping?
**Answer:** Satellite-based deprivation estimation offers tremendous benefits for rapid, low-cost screening of underserved regions between decennial census rounds. However, it carries significant ethical risks if misused:
1. **Misallocation of Welfare:** Point predictions must never be used to disqualify individual households or communities from welfare schemes, cash transfers, or food rations, because satellites cannot detect indoor living conditions, social discrimination, or micro-level food insecurity.
2. **Surveillance & Privacy:** Remote sensing must remain aggregate-level (district or block level); high-resolution commercial surveillance of vulnerable populations must be avoided.
3. **Confirmation Bias:** Algorithmic estimates should serve as an audit hint to guide physical field surveys, not as a substitute for ground truth census accountability.

### Q25. Why can't the Scenario Explorer be used as a policy impact simulation tool?
**Answer:** The Scenario Explorer allows users to perturb remote sensing features (e.g., $+25\%$ nighttime lights or $+1\text{ pp}$ built-up area) to observe model sensitivity. However, real-world socioeconomic deprivation is driven by healthcare access, school quality, social protection programs, wage structures, and agricultural commodity prices—none of which are inputs to this model. A policy intervention (such as an electrification subsidy or housing grant) alters economic fundamentals, whereas the model only calculates how its static regression trees respond to altered satellite numbers. We enforce an unavoidable system disclaimer across all scenario outputs stating that results reflect model sensitivity, not real-world policy outcomes.

### Q26. Which states had the highest prediction error, and why did the model struggle there?
**Answer:** As documented in [`documentation/RESULTS.md`](RESULTS.md), the five states with the highest Leave-One-State-Out MAE are:
1. **Goa** (MAE: 0.1769, $N=2$)
2. **Tripura** (MAE: 0.1608, $N=4$)
3. **Meghalaya** (MAE: 0.1517, $N=6$)
4. **Jharkhand** (MAE: 0.1464, $N=22$)
5. **Mizoram** (MAE: 0.1280, $N=8$)

The model struggled in these states for two distinct reasons: first, northeastern states (Tripura, Meghalaya, Mizoram) feature dense cloud cover, steep hilly terrain, and high forest canopy where optical NDVI and nighttime light reflectance behave differently than in the Indo-Gangetic plains. Second, Jharkhand has extensive mineral mining belts where heavy industrial lighting produces bright nocturnal radiance despite high multidimensional rural deprivation, confusing the satellite proxies.

### Q27. What would be the next step to improve this system if given another year of research?
**Answer:** If extended, we would pursue four concrete technical improvements:
1. **Synthetic Aperture Radar (SAR) Telemetry:** Incorporate Sentinel-1 C-band SAR radar imagery, which penetrates tropical cloud cover and provides all-weather structural information on building density and soil moisture.
2. **Sub-District (Tehsil/Block) Boundaries:** Transition from GAUL 2015 to contemporary Survey of India administrative boundaries to map deprivation at the sub-district or gram panchayat level.
3. **Temporal Panel Modeling:** Train on both NFHS-4 (2015–16) and NFHS-5 (2019–21) to learn temporal rates of change rather than a single static cross-section.
4. **Multimodal Tabular-Vision Fusion:** Use self-supervised vision backbones pre-trained specifically on earth observation imagery (such as SatMAE or Clay Foundation Model) rather than standard ImageNet models like VGG-16.
