# Extra-feature experiment (experimental, not part of the production model)

**Question:** do extra Earth Engine features improve the *honest* accuracy (leave-one-state-out R² = 0.336)?
Nothing here changes `best_model.pkl`, the database, or any published results file.

## Candidate features (`scripts/extract_extra_features.py`)
| Feature | Source (Earth Engine) | Note |
| --- | --- | --- |
| `elevation_mean_m`, `slope_mean_deg` | `USGS/SRTMGL1_003` | terrain |
| `rainfall_mm_year` | `UCSB-CHG/CHIRPS/DAILY` | survey window 2019-11 to 2021-04, mean daily rate x 365 |
| `lst_day_c` | `MODIS/061/MOD11A2` | daytime land surface temperature |
| `cropland_pct` | `ESA/WorldCover/v200` | % of district classed as cropland |
| `travel_time_city_min` | `Oxford/MAP/accessibility_to_cities_2015_v1_0` | 2015 vintage; older than the survey |

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
Primary test: baseline 4 features vs baseline + all extra. Adopt only if **all** hold:
1. LOSO R² gain >= 0.02
2. Repeated state-grouped 5-fold R² gain >= 0.02 (about one CV standard deviation)
3. >= 55% of states have lower MAE

Per-feature rows are exploratory. Picking a subset after seeing them would be selecting on the validation data.
The script first checks that the baseline reproduces the published R² (0.3360); if not, do not trust the run.

## If the verdict is "do not adopt"
Keep the 4-feature model and report it as a negative result: "six further physical/climate features were
tested under the same state-held-out protocol and did not improve generalisation to unseen states."
That is a legitimate, honest finding. If it is "adopt", retrain with `training/train_models.py`, rerun
`scripts/precompute_predictions.py` and `scripts/generate_results_doc.py`, update the README/slides, and rerun the tests.
