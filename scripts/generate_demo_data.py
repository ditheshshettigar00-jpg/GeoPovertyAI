"""
generate_demo_data.py
----------------------
Generates a SYNTHETIC, clearly-labeled DEMO training dataset with the exact
same column structure that the real pipeline (Modules 2-4: GEE + VGG-16 +
geospatial feature extraction) will eventually produce.

WHY THIS EXISTS:
Google Earth Engine requires the student's own authenticated account, and
real satellite processing takes real compute time. This script lets every
downstream module (feature fusion, ML training, SHAP, Flask API, dashboard)
be built and demonstrated immediately, without waiting on external data
access. It also gives you a template for exactly what shape the real data
must be in once you plug it in.

THIS IS NOT REAL RESEARCH DATA.
Every row is clearly tagged data_mode = 'demo'. Do not present numbers
produced from this data as real findings in your report — only as a
working-system demonstration.

The synthetic generator does not just produce random noise: it builds in a
plausible, documented statistical relationship between the geospatial
features and the deprivation target (higher nightlight/built-up/accessibility
-> lower deprivation, with added noise), so that the ML models trained on it
behave sensibly (feature importances, SHAP directions, etc. look realistic)
during development. The exact relationship is a design choice for the demo
only and is NOT derived from any real study.

Usage:
    python scripts/generate_demo_data.py --n_areas 60
"""

import argparse
import os
import sys

import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEMO_TRAINING_CSV, DEMO_VGG_FEATURE_COUNT, RANDOM_SEED

# A small set of real Indian state names + a plausible lat/lon bounding box
# used ONLY to place synthetic "districts" at realistic-looking coordinates.
# These are illustrative area names, not sourced from an actual district list.
STATE_BOUNDS = {
    "Karnataka": (12.0, 16.5, 74.5, 78.0),
    "Maharashtra": (16.0, 21.5, 73.0, 80.0),
    "Tamil Nadu": (8.5, 13.0, 77.0, 80.3),
    "Rajasthan": (24.0, 30.0, 70.0, 78.0),
    "Uttar Pradesh": (24.5, 30.0, 77.0, 84.5),
    "West Bengal": (21.5, 27.0, 85.8, 89.5),
}


def generate_demo_dataset(n_areas: int = 60, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = list(STATE_BOUNDS.keys())

    rows = []
    for i in range(1, n_areas + 1):
        state = states[i % len(states)]
        lat_min, lat_max, lon_min, lon_max = STATE_BOUNDS[state]
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)

        # Generate a synthetic polygon boundary (approx 15x15km square)
        offset = 0.15
        coords = [
            [lon - offset, lat - offset],
            [lon + offset, lat - offset],
            [lon + offset, lat + offset],
            [lon - offset, lat + offset],
            [lon - offset, lat - offset]
        ]
        boundary_geojson = json.dumps({"type": "Polygon", "coordinates": [coords]})

        # --- synthetic geospatial features (each on a realistic-looking scale) ---
        nightlight_mean = rng.gamma(shape=2.0, scale=2.0)          # roughly 0-20
        ndvi_mean = rng.uniform(0.1, 0.8)                           # NDVI range
        builtup_pct = rng.uniform(2, 70)                            # percent
        population_density = rng.gamma(shape=2.0, scale=250)        # people/sq km
        road_distance_km = rng.exponential(scale=5.0)                # km to nearest road
        school_distance_km = rng.exponential(scale=4.0)
        hospital_distance_km = rng.exponential(scale=8.0)

        # --- synthetic VGG-16-style feature placeholder vector ---
        # Correlated loosely with builtup_pct so downstream SHAP/Grad-CAM demo
        # behaves sensibly; real vectors come from actual VGG-16 in Module 3.
        vgg_features = rng.normal(loc=builtup_pct / 100.0, scale=0.5,
                                   size=DEMO_VGG_FEATURE_COUNT)

        # --- synthetic deprivation target ---
        # Documented, hand-designed relationship for demo purposes ONLY:
        # more nightlight, more built-up, better accessibility (lower distances)
        # => lower deprivation. Not derived from any real survey.
        raw_score = (
            -0.15 * (nightlight_mean / 10.0)
            - 0.20 * (builtup_pct / 100.0)
            - 0.10 * ndvi_mean
            + 0.15 * (road_distance_km / 20.0)
            + 0.15 * (school_distance_km / 20.0)
            + 0.15 * (hospital_distance_km / 20.0)
            + rng.normal(0, 0.08)   # noise
            + 0.55                  # center the distribution
        )
        deprivation_score = float(np.clip(raw_score, 0.0, 1.0))

        row = {
            "area_id": i,
            "area_name": f"Demo_District_{i:03d}",
            "state_name": state,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "boundary_geojson": boundary_geojson,
            "deprivation_score": round(deprivation_score, 4),
            "nightlight_mean": round(nightlight_mean, 3),
            "ndvi_mean": round(ndvi_mean, 3),
            "builtup_pct": round(builtup_pct, 2),
            "population_density": round(population_density, 1),
            "road_distance_km": round(road_distance_km, 2),
            "school_distance_km": round(school_distance_km, 2),
            "hospital_distance_km": round(hospital_distance_km, 2),
        }
        for j, val in enumerate(vgg_features, start=1):
            row[f"vgg_feat_{j}"] = round(float(val), 5)

        row["data_mode"] = "demo"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo dataset for GeoPoverty AI")
    parser.add_argument("--n_areas", type=int, default=60,
                         help="Number of synthetic districts to generate (default: 60)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = generate_demo_dataset(n_areas=args.n_areas, seed=args.seed)
    os.makedirs(os.path.dirname(DEMO_TRAINING_CSV), exist_ok=True)
    df.to_csv(DEMO_TRAINING_CSV, index=False)

    print(f"Generated {len(df)} synthetic demo rows.")
    print(f"Saved to: {DEMO_TRAINING_CSV}")
    print("\nColumn summary:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nREMINDER: This is SYNTHETIC DEMO data (data_mode='demo'), "
          "not real research data. Do not report these numbers as findings.")


if __name__ == "__main__":
    main()
