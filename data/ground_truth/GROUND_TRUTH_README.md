# Ground Truth Data (real socioeconomic target)

This is the one piece of Module 2 that is **not** Earth Engine — the
deprivation *target* has to come from an actual survey/census, not a
satellite. This note tells you exactly what's realistically available and
how to bring it into this pipeline.

## What is realistically accessible

- **Census of India, 2011** (most recent full census at the time of
  writing): district-level indicator tables (e.g., housing quality,
  household amenities, literacy) are published at
  https://censusindia.gov.in/ — you'll need to browse to the relevant
  table series (e.g., House Listing and Housing Census tables) and
  download district-level CSV/Excel files yourself; exact table numbers
  and layouts change over time on that site, so I'm not going to guess a
  specific deep link here — search the site's data portal for
  "District Census Handbook" or "Houselisting and Housing Data" for your
  chosen state.
- **NFHS (National Family Health Survey)** district-level fact sheets
  are published via the Ministry of Health & Family Welfare / IIPS, and
  summarized indicators are also mirrored on the DHS Program site
  (https://dhsprogram.com/). District-level *fact sheets* (PDF/summary
  tables) are publicly accessible; household-level microdata with precise
  coordinates generally requires a formal DHS Program data request and is
  usually released with location displacement for privacy, as discussed
  in the earlier design document.

## Ground Truth Target: Deprivation Score Formula & Provenance

The continuous district-level target `deprivation_score` $\in [0.0, 1.0]$ is derived from the **National Family Health Survey (NFHS-5, 2019-2021)** district factsheets (`nfhs5_districts_raw.csv`, 705 districts).

### 1. Selected Indicators
Five published district-level indicators representing multidimensional deprivation are included:
1. **Sanitation** (Indicator 9): `% of households using improved sanitation facility` (Inverted: $100 - x$)
2. **Electricity** (Indicator 7): `% of households with electricity` (Inverted: $100 - x$)
3. **Clean Cooking Fuel** (Indicator 10): `% of households using clean fuel for cooking` (Inverted: $100 - x$)
4. **Child Stunting** (Indicator 73): `% of children under 5 years who are stunted (height-for-age)` (Direct deprivation)
5. **Child Anaemia** (Indicator 84): `% of children age 6-59 months who are anaemic` (Direct deprivation)

### 2. Normalization & Aggregation Formula
Each indicator is min-max scaled across all Indian districts to $[0, 1]$:
$$\text{norm}(X) = \frac{X - \min(X)}{\max(X) - \min(X)}$$

The composite `deprivation_score` is computed with equal weighting:
$$\text{deprivation\_score} = \frac{1}{5} \sum_{k=1}^{5} \text{norm}(X_k)$$

### 3. Sensitivity Analysis & Statistical Robustness
We validated whether equal weighting introduces arbitrary bias compared to data-driven unsupervised alternatives using Principal Component Analysis (PCA). Full details are recorded in [`score_sensitivity.md`](score_sensitivity.md):
- **Principal Component 1 (PC1)**: Explains **49%** of the joint indicator variance with positive loadings across all 5 dimensions.
- **Rank Correlation ($\rho$)**: The Spearman rank correlation between the equal-weight score and PC1 across all 705 districts is **0.996**.
- **Leave-One-Indicator-Out Stability**:
  - Leaving out Sanitation: $\rho = 0.975$
  - Leaving out Electricity: $\rho = 0.988$
  - Leaving out Clean Fuel: $\rho = 0.932$
  - Leaving out Stunting: $\rho = 0.974$
  - Leaving out Anaemia: $\rho = 0.975$

This confirms that relative district rankings are highly robust ($\rho > 0.93$) and do not hinge on any single indicator choice.


## What I will NOT do

I won't fabricate a ready-made CSV of "real" district deprivation scores,
invent specific table numbers, or guess at numbers that look plausible —
that would risk you presenting fabricated data as real research findings,
which the project's own constraints explicitly rule out. This step
genuinely requires you to go pull the actual published table for your
chosen state/districts.
