"""
generate_demo_final_table.py
-------------------------------
Generates a SYNTHETIC demo dataset already in the FUSED final-training-table
format (same columns feature_fusion.py would produce), so Module 5 (model
training) can be built, tested, and demoed on its own -- without requiring
Modules 2-4 to have been run for real yet.

Spans multiple synthetic "states" so the geography-aware validation split
(Module 5) has something meaningful to hold out.

THIS IS SYNTHETIC DEMO DATA, clearly tagged data_source='Demo_Synthetic'.
Do not report metrics computed on this data as real research results --
it exists only to prove the training pipeline works correctly.

Usage:
    python scripts/generate_demo_final_table.py --n_states 8 --areas_per_state 12
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RANDOM_SEED

DEMO_VGG_DIM = 32  # smaller than the real 512-d for fast local demo training

STATE_NAMES = [
    "Karnataka", "Maharashtra", "Tamil Nadu", "Rajasthan",
    "Uttar Pradesh", "West Bengal", "Gujarat", "Kerala",
    "Punjab", "Odisha", "Bihar", "Madhya Pradesh",
]


def generate_demo_final_table(n_states: int, areas_per_state: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = STATE_NAMES[:n_states]

    rows = []
    area_id = 1
    for state in states:
        # Give each state a mild "state effect" so geography-grouped validation
        # is meaningfully different from a random split (districts within a
        # state are correlated, as in real data).
        state_effect = rng.normal(0, 0.08)

        for _ in range(areas_per_state):
            nightlight_mean = rng.gamma(2.0, 2.0)
            ndvi_mean = rng.uniform(0.1, 0.8)
            builtup_pct = rng.uniform(2, 70)
            population_density = rng.gamma(2.0, 250)
            road_density_km_per_sqkm = rng.uniform(0.2, 3.0)
            school_distance_km = rng.exponential(4.0)
            hospital_distance_km = rng.exponential(8.0)

            vgg_features = rng.normal(loc=builtup_pct / 100.0, scale=0.5, size=DEMO_VGG_DIM)

            raw_score = (
                -0.15 * (nightlight_mean / 10.0)
                - 0.20 * (builtup_pct / 100.0)
                - 0.10 * ndvi_mean
                + 0.10 * (road_density_km_per_sqkm / 3.0) * -1  # denser roads -> less deprived
                + 0.15 * (school_distance_km / 20.0)
                + 0.15 * (hospital_distance_km / 20.0)
                + state_effect
                + rng.normal(0, 0.06)
                + 0.55
            )
            deprivation_score = float(np.clip(raw_score, 0.0, 1.0))

            row = {
                "area_id": area_id,
                "area_name": f"Demo_District_{area_id:03d}",
                "state_name": state,
                "data_source": "Demo_Synthetic",
                "data_year": 2023,
                "deprivation_score": round(deprivation_score, 4),
                "nightlight_mean": round(nightlight_mean, 3),
                "ndvi_mean": round(ndvi_mean, 3),
                "builtup_pct": round(builtup_pct, 2),
                "population_density": round(population_density, 1),
                "road_density_km_per_sqkm": round(road_density_km_per_sqkm, 3),
                "school_distance_km": round(school_distance_km, 2),
                "hospital_distance_km": round(hospital_distance_km, 2),
            }
            for j, val in enumerate(vgg_features, start=1):
                row[f"vgg_feat_{j}"] = round(float(val), 5)

            rows.append(row)
            area_id += 1

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo data in fused final-table format")
    parser.add_argument("--n_states", type=int, default=8)
    parser.add_argument("--areas_per_state", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = generate_demo_final_table(args.n_states, args.areas_per_state, args.seed)
    out_path = args.out or os.path.join(PROCESSED_DIR, "demo_final_training_table.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} synthetic rows across {args.n_states} states.")
    print(f"Saved to: {out_path}")
    print(df["state_name"].value_counts())


if __name__ == "__main__":
    main()
