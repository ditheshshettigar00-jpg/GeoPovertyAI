"""
extract_geospatial_features.py
---------------------------------
For every district saved by fetch_district_boundaries.py, computes the
mean value of each geospatial layer WITHIN that district's boundary:

  - nightlight_mean   (VIIRS monthly average radiance)
  - ndvi_mean          (from Sentinel-2, computed as (B8-B4)/(B8+B4))
  - builtup_pct         (GHSL built-up surface fraction, as a percentage)
  - population_density   (GHSL population grid, mean per-pixel count)

This produces the same columns Module 1's demo dataset used for these
features (minus accessibility, which is Module 4 via OSMnx) -- so the
output of this script can be joined directly into the fusion pipeline
later.

Usage:
    python geospatial/extract_geospatial_features.py \\
        --districts data/raw/districts_Karnataka.json \\
        --start_date 2023-01-01 --end_date 2023-12-31
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geospatial.gee_config import (
    GHSL_BUILTUP,
    GHSL_POP,
    SENTINEL2_SR,
    VIIRS_BAND,
    VIIRS_MONTHLY,
    initialize_ee,
)
from config import PROCESSED_DIR


def mask_s2_clouds(image, ee):
    """Masks clouds in a Sentinel-2 SR image using the SCL (Scene Classification) band.
    Keeps only pixels classified as vegetation, bare soil, water, or unclassified --
    excludes cloud, cloud shadow, cirrus, and snow classes."""
    scl = image.select("SCL")
    # SCL classes to KEEP: 4=vegetation, 5=bare soil, 6=water, 7=unclassified, 11=snow(optional exclude)
    good_pixel = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return image.updateMask(good_pixel).divide(10000)  # scale reflectance to 0-1


def get_sentinel2_composite(ee, geometry, start_date, end_date, max_cloud_pct=20):
    collection = (
        ee.ImageCollection(SENTINEL2_SR)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(lambda img: mask_s2_clouds(img, ee))
    )
    n_images = collection.size().getInfo()
    if n_images == 0:
        return None, 0
    composite = collection.median().clip(geometry)
    return composite, n_images


def compute_ndvi(ee, s2_composite):
    return s2_composite.normalizedDifference(["B8", "B4"]).rename("NDVI")


def reduce_mean(ee, image, geometry, band, scale):
    result = image.select(band).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()
    return result.get(band)


def extract_features_for_district(ee, district, start_date, end_date):
    geometry = ee.Geometry(district["geometry"])

    row = {
        "area_id": district["area_id"],
        "area_name": district["area_name"],
        "state_name": district["state_name"],
        "latitude": district["latitude"],
        "longitude": district["longitude"],
        "data_mode": "real",
    }

    # --- Sentinel-2 composite + NDVI ---
    composite, n_images = get_sentinel2_composite(ee, geometry, start_date, end_date)
    if composite is not None:
        ndvi_img = compute_ndvi(ee, composite)
        row["ndvi_mean"] = reduce_mean(ee, ndvi_img, geometry, "NDVI", scale=100)
        row["s2_image_count"] = n_images
    else:
        row["ndvi_mean"] = None
        row["s2_image_count"] = 0
        print(f"  WARNING: no cloud-free Sentinel-2 images found for "
              f"{district['area_name']} in {start_date}..{end_date}")

    # --- VIIRS nighttime lights ---
    viirs = (
        ee.ImageCollection(VIIRS_MONTHLY)
        .filterDate(start_date, end_date)
        .select(VIIRS_BAND)
        .mean()
    )
    row["nightlight_mean"] = reduce_mean(ee, viirs, geometry, VIIRS_BAND, scale=500)

    # --- GHSL built-up ---
    try:
        builtup = ee.ImageCollection(GHSL_BUILTUP).sort("system:time_start", False).first()
        builtup_band = builtup.bandNames().get(0)  # take first band; verify name via catalog if this fails
        builtup_val = builtup.select([builtup_band]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=100,
            maxPixels=1e9, bestEffort=True,
        ).getInfo()
        row["builtup_pct"] = list(builtup_val.values())[0] if builtup_val else None
    except Exception as e:
        row["builtup_pct"] = None
        print(f"  WARNING: built-up extraction failed for {district['area_name']}: {e}")

    # --- GHSL population ---
    try:
        pop = ee.ImageCollection(GHSL_POP).sort("system:time_start", False).first()
        pop_band = pop.bandNames().get(0)
        pop_val = pop.select([pop_band]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=100,
            maxPixels=1e9, bestEffort=True,
        ).getInfo()
        row["population_density"] = list(pop_val.values())[0] if pop_val else None
    except Exception as e:
        row["population_density"] = None
        print(f"  WARNING: population extraction failed for {district['area_name']}: {e}")

    return row


def main():
    parser = argparse.ArgumentParser(description="Extract nightlight/NDVI/built-up/population features per district")
    parser.add_argument("--districts", type=str, required=True,
                         help="Path to JSON file from fetch_district_boundaries.py")
    parser.add_argument("--start_date", type=str, default="2023-01-01")
    parser.add_argument("--end_date", type=str, default="2023-12-31")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    ee = initialize_ee()

    with open(args.districts, "r", encoding="utf-8") as f:
        districts = json.load(f)

    rows = []
    for d in districts:
        print(f"Processing {d['area_name']} ({d['state_name']}) ...")
        row = extract_features_for_district(ee, d, args.start_date, args.end_date)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = args.out or os.path.join(PROCESSED_DIR, "real_geospatial_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to: {out_path}")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
