# Data audit: final_training_table.csv

FAIL: 0 | WARN: 4 | PASS: 17

| Status | Check | Detail |
|---|---|---|
| WARN | range: road_density_km_per_sqkm | column not present |
| WARN | range: school_distance_km | column not present |
| WARN | range: hospital_distance_km | column not present |
| WARN | thin states | Dadra Nagar Haveli Daman Diu (2), Goa (2), Andaman Nicobar Islands (1) have fewer than 3 districts; leave-one-state-out scores for them are very noisy |
| PASS | range: builtup_pct | all values within [0, 100] % of land |
| PASS | range: ndvi_mean | all values within [-1, 1] index |
| PASS | range: nightlight_mean | all values within [0, None] nW/cm2/sr |
| PASS | range: population_density | all values within [0, 50000] people per km2 |
| PASS | range: elevation_mean_m | all values within [-50, 6500] m |
| PASS | range: slope_mean_deg | all values within [0, 90] deg |
| PASS | range: rainfall_mm_year | all values within [0, 15000] mm/year |
| PASS | range: lst_day_c | all values within [-30, 60] deg C |
| PASS | range: cropland_pct | all values within [0, 100] % of land |
| PASS | range: travel_time_city_min | all values within [0, 5000] min |
| PASS | range: deprivation_score | all values within [0, 1] score |
| PASS | range: latitude | all values within [6, 37.5] deg |
| PASS | range: longitude | all values within [68, 98] deg |
| PASS | units: population_density | median 424 people per km2 |
| PASS | missing values | no missing values (note: if imputation happened earlier, say where) |
| PASS | duplicate districts | 0 duplicate (state, district) rows |
| PASS | Sentinel-2 coverage | 0 districts have zero usable images (NDVI is missing or filled for them) |
| INFO | Table | 564 districts, 31 states, 23 columns |
| INFO | visual features | none present |
