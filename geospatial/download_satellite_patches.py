"""
download_satellite_patches.py
--------------------------------
Downloads a cloud-filtered Sentinel-2 RGB composite image patch for each
district, clipped to the district boundary, saved as a GeoTIFF. These
patches feed into Module 3 (VGG-16 feature extraction).

Uses geemap's local export (getDownloadURL under the hood), so you do NOT
need to route through Google Drive -- images download straight to your
data/satellite/ folder.

NOTE ON SCALE: districts can be large. Downloading a full-resolution (10m)
Sentinel-2 image for an entire district can be a big file. This script
exports at a reduced scale (default 30m/pixel) suitable for a 224x224-ish
patch after resizing in Module 3 -- adjust --scale if you want higher
detail for a smaller pilot area.

Usage:
    python geospatial/download_satellite_patches.py \\
        --districts data/raw/districts_Karnataka.json \\
        --start_date 2023-01-01 --end_date 2023-12-31
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geospatial.extract_geospatial_features import get_sentinel2_composite
from geospatial.gee_config import initialize_ee
from preprocessing.name_keys import clean_key
from config import SATELLITE_DIR


def download_patch(ee, district, start_date, end_date, scale, out_dir):
    import urllib.request

    geometry = ee.Geometry(district["geometry"])
    composite, n_images = get_sentinel2_composite(ee, geometry, start_date, end_date)

    if composite is None:
        print(f"  SKIPPED {district['area_name']}: no cloud-free imagery available "
              f"for {start_date}..{end_date}")
        return None

    rgb = composite.select(["B4", "B3", "B2"]).multiply(255).clamp(0, 255).uint8()

    sk = clean_key(district.get("state_name", ""))
    dk = clean_key(district.get("area_name", ""))
    if sk and dk:
        fname = f"{sk}_{dk}_rgb.png"
    else:
        fname = f"district_{district.get('area_id', 0):03d}_rgb.png"
    out_path = os.path.join(out_dir, fname)
    try:
        url = rgb.getThumbURL({
            "region": geometry,
            "dimensions": 256,
            "format": "png",
        })
        urllib.request.urlretrieve(url, out_path)
        print(f"  Saved: {out_path}  ({n_images} source images composited)")
        return out_path
    except Exception as e:
        print(f"  FAILED to export {district['area_name']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download RGB Sentinel-2 patches per district")
    parser.add_argument("--districts", type=str, required=True)
    parser.add_argument("--start_date", type=str, default="2023-01-01")
    parser.add_argument("--end_date", type=str, default="2023-12-31")
    parser.add_argument("--scale", type=int, default=30,
                         help="Export resolution in meters/pixel (default 30; use 10 for finer detail on small areas)")
    args = parser.parse_args()

    ee = initialize_ee()

    with open(args.districts, "r", encoding="utf-8") as f:
        districts = json.load(f)

    os.makedirs(SATELLITE_DIR, exist_ok=True)

    # If real final_training_table exists, map district names to real area_ids
    import pandas as pd
    from config import PROCESSED_DIR
    real_table_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    name_to_id = {}
    if os.path.exists(real_table_path):
        rdf = pd.read_csv(real_table_path)
        for _, r in rdf.iterrows():
            clean_n = str(r["area_name"]).lower().replace(" ", "").replace("-", "")
            name_to_id[clean_n] = int(r["area_id"])

    results = []
    for d in districts:
        d_copy = dict(d)
        clean_d = str(d["area_name"]).lower().replace(" ", "").replace("-", "")
        if clean_d in name_to_id:
            d_copy["area_id"] = name_to_id[clean_d]
        print(f"Downloading patch for {d_copy['area_name']} (area_id={d_copy['area_id']}) ...")
        path = download_patch(ee, d_copy, args.start_date, args.end_date, args.scale, SATELLITE_DIR)
        results.append({"area_id": d_copy["area_id"], "area_name": d_copy["area_name"], "image_path": path})

    n_ok = sum(1 for r in results if r["image_path"] is not None)
    print(f"\nDone. {n_ok}/{len(results)} patches downloaded successfully to {SATELLITE_DIR}")


if __name__ == "__main__":
    main()
