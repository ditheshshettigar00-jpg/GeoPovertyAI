# GeoPoverty AI: Data Integrity Audit Report

**Target Dataset:** `D:\geo poverty project\v2+\GeoPovertyAI_Complete\GeoPovertyAI\data\processed\final_training_table.csv`  
**Total Records:** 512  
**Total Columns:** 14  
**Unique States:** 27  
**Unique District Names:** 508

## 1. Duplicate Row Analysis

- Duplicate `area_id`: **0**
- Duplicate (`area_name`, `state_name`): **0**

## 2. Missing Value (NaN) Rates

No missing values (NaN) found across all columns.


## 3. Numeric Value Ranges & Physical Plausibility

| Column | Min | Max | Mean | Std |
| --- | --- | --- | --- | --- |
| area_id | 1.0 | 512.0 | 256.5 | 147.9459 |
| latitude | 8.306 | 32.6557 | 23.1036 | 5.5663 |
| longitude | 69.8004 | 96.3563 | 80.9174 | 6.1498 |
| nightlight_mean | 0.3693 | 52.1492 | 1.6358 | 3.1243 |
| ndvi_mean | 0.0799 | 0.7947 | 0.4625 | 0.1102 |
| s2_image_count | 0.0 | 596.0 | 178.4727 | 109.3341 |
| builtup_pct | 0.0086 | 40.57 | 2.0627 | 2.7506 |
| population_density | 2.5425 | 44366.3645 | 867.4517 | 2758.4453 |
| deprivation_score | 0.1066 | 0.8133 | 0.3956 | 0.1344 |
| data_year | 2021.0 | 2021.0 | 2021.0 | 0.0 |


## 4. Constant or Low Variance Column Check

**[CRITICAL: Constant columns found (1)]:** ['data_year']


## 5. Accessibility Columns Multicollinearity Analysis

Accessibility columns not present in table.

## 6. Visual (VGG-16) Feature Columns Audit

- Detected visual feature count: **0** (Expected: 512 for full VGG-16 pooled representations; table has 0).

