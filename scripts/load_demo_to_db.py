"""
load_demo_to_db.py
--------------------
Loads the synthetic demo_training_data.csv into the SQLite database
(areas, ground_truth, geospatial_features tables).

Run AFTER:
    python database/init_db.py
    python scripts/generate_demo_data.py

Usage:
    python scripts/load_demo_to_db.py
"""

import os
import sqlite3
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, DEMO_TRAINING_CSV


def load_demo_to_db(db_path: str = DB_PATH, csv_path: str = DEMO_TRAINING_CSV) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/generate_demo_data.py first."
        )
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"{db_path} not found. Run database/init_db.py first."
        )

    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Clear any previous demo rows so this script is safely re-runnable
    cur.execute("DELETE FROM geospatial_features WHERE data_mode = 'demo';")
    cur.execute("DELETE FROM ground_truth WHERE data_source = 'Demo_Synthetic';")
    cur.execute("DELETE FROM areas WHERE data_mode = 'demo';")
    conn.commit()

    inserted = 0
    for _, row in df.iterrows():
        cur.execute(
            """INSERT INTO areas (area_id, area_name, state_name, centroid_lat,
                                    centroid_lon, boundary_geojson, data_mode)
               VALUES (?, ?, ?, ?, ?, ?, 'demo')""",
            (int(row["area_id"]), row["area_name"], row["state_name"],
             float(row["latitude"]), float(row["longitude"]), row.get("boundary_geojson")),
        )

        cur.execute(
            """INSERT INTO ground_truth (area_id, deprivation_score, data_source, data_year)
               VALUES (?, ?, 'Demo_Synthetic', NULL)""",
            (int(row["area_id"]), float(row["deprivation_score"])),
        )

        cur.execute(
            """INSERT INTO geospatial_features
               (area_id, nightlight_mean, ndvi_mean, builtup_pct, population_density,
                road_distance_km, school_distance_km, hospital_distance_km, data_mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'demo')""",
            (int(row["area_id"]), float(row["nightlight_mean"]), float(row["ndvi_mean"]),
             float(row["builtup_pct"]), float(row["population_density"]),
             float(row["road_distance_km"]), float(row["school_distance_km"]),
             float(row["hospital_distance_km"])),
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"Loaded {inserted} demo areas into {db_path}")


if __name__ == "__main__":
    load_demo_to_db()
