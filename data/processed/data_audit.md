# Data audit: final_training_table.csv

FAIL: 0 | WARN: 5 | PASS: 10

| Status | Check | Detail |
|---|---|---|
| WARN | range: road_density_km_per_sqkm | column not present |
| WARN | range: school_distance_km | column not present |
| WARN | range: hospital_distance_km | column not present |
| WARN | thin states | Goa (2), Dadra Nagar Haveli Daman Diu (1), Sikkim (1) have fewer than 3 districts; leave-one-state-out scores for them are very noisy |
| WARN | Sentinel-2 coverage | 28 districts have zero usable images (NDVI is missing or filled for them) |
| PASS | range: builtup_pct | all values within [0, 100] % of land |
| PASS | range: ndvi_mean | all values within [-1, 1] index |
| PASS | range: nightlight_mean | all values within [0, None] nW/cm2/sr |
| PASS | range: population_density | all values within [0, 50000] people per km2 |
| PASS | range: deprivation_score | all values within [0, 1] score |
| PASS | range: latitude | all values within [6, 37.5] deg |
| PASS | range: longitude | all values within [68, 98] deg |
| PASS | units: population_density | median 463 people per km2 |
| PASS | missing values | no missing values (note: if imputation happened earlier, say where) |
| PASS | duplicate districts | 0 duplicate (state, district) rows |
| INFO | Table | 529 districts, 30 states, 17 columns |
| INFO | visual features | none present |
