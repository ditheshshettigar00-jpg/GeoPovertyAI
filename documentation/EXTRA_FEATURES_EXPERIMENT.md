# Extra-feature experiment (Evaluated & Adopted on `feature/extra-features`)

**Question:** do six extra Earth Engine features improve the *honest* accuracy over the initial 4-feature RandomForest baseline (leave-one-state-out $R^2 = 0.3360$)?

## Candidate features (`scripts/extract_extra_features.py`)
| Feature | Source (Earth Engine) | Note |
| --- | --- | --- |
| `elevation_mean_m`, `slope_mean_deg` | `USGS/SRTMGL1_003` | terrain |
| `rainfall_mm_year` | `UCSB-CHG/CHIRPS/DAILY` | survey window 2019-11 to 2021-04, mean daily rate x 365 |
| `lst_day_c` | `MODIS/061/MOD11A2` | daytime land surface temperature |
| `cropland_pct` | `ESA/WorldCover/v200` | % of district classed as cropland (Epoch 2021) |
| `travel_time_city_min` | `Oxford/MAP/accessibility_to_cities_2015_v1_0` | 2015 baseline (deprecated in Earth Engine; older than 2019–21 survey) |

Districts are matched by (GAUL state, district name). The `area_id` inside `data/raw/districts_*.json` restarts in
every state and is not used as a key.

## How to run
```
$env:GEE_PROJECT_ID="your-project-id"                    # PowerShell
python scripts/extract_extra_features.py --limit 2       # smoke test: dataset IDs/bands can change
python scripts/extract_extra_features.py                 # full run, resumable
python training/evaluate_extra_features.py               # honest test, same protocol as the published results
```

## Decision rule (fixed before looking at results)
Primary test: baseline 4 features vs baseline + all extra (under RandomForest, matching the pre-adoption baseline). Adopt only if **all** hold:
1. LOSO R² gain >= 0.02
2. Repeated state-grouped 5-fold R² gain >= 0.02 (about one CV standard deviation)
3. >= 55% of states have lower MAE

Per-feature rows are exploratory. Picking a subset after seeing them would be selecting on the validation data.
The evaluation script checks that the baseline reproduces `0.3360` when run in 4-feature mode before trusting results.

## Outcome: ADOPTED
All 564 districts were extracted with 0 missing values, and the primary RandomForest comparison passed all three gates:
- **LOSO $R^2$:** `0.3360` $\rightarrow$ `0.4217` (`+0.0857 >= 0.02`, PASS)
- **Repeated state-grouped 5-fold $R^2$:** `0.2984` $\rightarrow$ `0.3543` (`+0.0559 >= 0.02`, PASS)
- **States improved (lower MAE):** `65% >= 55%` (PASS)

The 10 features were subsequently adopted end-to-end (`training/train_models.py`, `scripts/precompute_predictions.py`, `scripts/generate_results_doc.py`), where **XGBoost** achieved the best retrained LOSO performance ($R^2 = 0.4884$, MAE $= 0.0752$, Spearman $\rho = 0.7150$). Caveats retained in documentation: `travel_time_city_min` relies on the static 2015 Oxford/MAP layer, and individual drop-one ablations under XGBoost show gains are concentrated in `lst_day_c`, `elevation_mean_m`, `builtup_pct`, `ndvi_mean`, and `nightlight_mean`.
