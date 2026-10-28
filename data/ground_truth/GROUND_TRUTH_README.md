# Ground Truth Socioeconomic Deprivation Target

This document specifies the exact provenance, mathematical formulation, and sensitivity analysis of the district-level ground truth target (`deprivation_score`) used in GeoPoverty AI.

---

## 1. Survey Data Source
- **Survey:** National Family Health Survey (NFHS-5), 2019–2021.
- **Source Agency:** International Institute for Population Sciences (IIPS) and Ministry of Health and Family Welfare (MoHFW), Government of India.
- **Level of Aggregation:** District-level administrative aggregates (705 surveyed districts across 34 Indian States & Union Territories).
- **Raw File:** `data/ground_truth/nfhs5_districts_raw.csv`
- **Processed File:** `data/ground_truth/real_ground_truth.csv`

---

## 2. Constituent Indicators & Inversions
Five standardized socioeconomic and public health indicators were selected across living standards, health, and clean infrastructure:

| Indicator Key | NFHS-5 Factsheet # | Indicator Description | Directionality | Transformation |
| :--- | :--- | :--- | :--- | :--- |
| `sanitation` | **9** | Households with improved sanitation facility (%) | Higher = Better | **Inverted**: $100 - \text{value}$ |
| `electricity` | **7** | Households with electricity (%) | Higher = Better | **Inverted**: $100 - \text{value}$ |
| `clean_fuel` | **10** | Households using clean fuel for cooking (%) | Higher = Better | **Inverted**: $100 - \text{value}$ |
| `stunting` | **73** | Children under 5 years who are stunted (height-for-age) (%) | Higher = Worse | Preserved as-is |
| `anaemia` | **84** | Pregnant women aged 15-49 who are anaemic (%) | Higher = Worse | Preserved as-is |

---

## 3. Mathematical Formulation

### Step A: Min-Max Normalization
Each inverted/aligned indicator $X_k$ across all districts $i \in \{1, \dots, N\}$ is normalized to $[0, 1]$:

$$z_{i, k} = \frac{X_{i, k} - \min(X_k)}{\max(X_k) - \min(X_k)}$$

### Step B: Equal-Weighted Composite Score
The final continuous deprivation target is the arithmetic mean of the five normalized components:

$$\text{Deprivation Score}_i = \frac{1}{5} \sum_{k=1}^{5} z_{i, k} \quad \in [0.0, 1.0]$$

Higher values indicate more severe multidimensional socioeconomic deprivation.

---

## 4. Sensitivity Check: Equal Weights vs. Principal Component Analysis (PCA)
To verify that equal weighting does not distort relative district rankings compared to an empirical, data-driven weighting scheme, we performed Principal Component Analysis (PCA) across the five normalized indicators:

- **First Principal Component (PC1) Explained Variance:** **55.86%**
- **PC1 Factor Loadings:**
  - `clean_fuel`: **+0.7356**
  - `sanitation`: **+0.4760**
  - `stunting`: **+0.3746**
  - `electricity`: **+0.2274**
  - `anaemia`: **+0.2009**
  *(All factor loadings are strictly positive, confirming that PC1 represents a unified deprivation gradient).*

### Correlation Results:
- **Spearman Rank Correlation ($r_s$):** **0.9780** ($p < 10^{-15}$)
- **Pearson Linear Correlation ($r$):** **0.9761** ($p < 10^{-15}$)

**Conclusion:** The rank order of district deprivation is exceptionally robust ($r_s = 0.9780$) to the choice between equal weighting and first principal component weighting. Equal weighting is retained for superior public policy interpretability.
