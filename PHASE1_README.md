# Phase 1 files (data integrity)

Copy these into your project root, keeping the folder paths. Then run:

    python scripts/audit_data.py            # shows the current table FAILS (builtup_pct, population_density)
    python geospatial/... / scripts/extract_all_india_features.py   # needs Earth Engine login (you run it)
    python scripts/extract_all_india_accessibility.py --states Goa --max-districts 2   # test, then run fully
    python scripts/build_real_training_table.py
    python scripts/audit_data.py --strict

## What changed
- GHSL units fixed (builtup_pct is a true 0..100 %, population_density is people per km2), survey-period epoch 2020.
- Imagery window is 2019-11-01..2021-04-30 (config.py), matching NFHS-5 fieldwork.
- District matching: alias table + exact + one explicit Andhra Pradesh->Telangana fallback + conservative fuzzy.
  Dry run on the 573 boundary districts: 564 matched (old table: 532), 0 duplicate targets.
- Build script refuses placeholder accessibility columns and never imputes.
- Accessibility: grid-mean distances, major-road density, NaN on failure (no constants).
- audit_data.py and score_sensitivity.py added.

## Tested here
- py_compile on every file; matching dry run (names only); audit on the current table;
  score_sensitivity.py; the distance helper (mean_nearest_distance_km).

## NOT tested here (needs network / your logins)
- Earth Engine extraction, OSMnx downloads, anything using shapely/pyproj/geopandas.
  Run the small test commands first.

## Still to do in Phase 1
- geospatial/download_satellite_patches.py names files by area_id and remaps by district name only, so
  same-named districts (Aurangabad, Bilaspur, ...) collide. Name files by state + district, make
  preprocessing/vgg16_feature_extractor.py parse that and write state_key/district_key columns
  (use preprocessing/name_keys.clean_key). The build script already joins on those columns.
- Document the score formula in data/ground_truth/GROUND_TRUTH_README.md and link score_sensitivity.md.
