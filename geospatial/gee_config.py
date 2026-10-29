"""
gee_config.py
--------------
Central Earth Engine initialization + dataset ID registry, so every
extraction script uses the same, single source of truth.

Set your Cloud project ID as an environment variable befo                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       re running any
script in this module:

    export GEE_PROJECT_ID="your-project-id-here"

or edit EE_PROJECT_ID below directly.
"""

import os

EE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "geopoverty-ai-project")

# --------------------------------------------------------------------------
# Dataset IDs used in this project. These are real, current (as of this
# writing) Earth Engine Data Catalog IDs. Catalog entries occasionally
# change band names or get superseded by newer versions — if a script
# below throws a "band not found" style error, open the dataset's page on
# https://developers.google.com/earth-engine/datasets and confirm the
# current band names for your chosen dataset version.
# --------------------------------------------------------------------------

# Administrative boundaries (used as district polygons - avoids needing to
# separately download a shapefile). Level 2 = district-equivalent globally.
GAUL_LEVEL2 = "FAO/GAUL/2015/level2"

# Sentinel-2 Surface Reflectance, cloud-masked using the SCL band
SENTINEL2_SR = "COPERNICUS/S2_SR_HARMONIZED"

# VIIRS monthly nighttime lights, stray-light corrected, average radiance
VIIRS_MONTHLY = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
VIIRS_BAND = "avg_rad"

# GHSL built-up surface (fraction of built-up area per pixel)
# NOTE: verify exact band name in the GEE catalog before relying on it --
# GHSL product versions/band names have changed across releases.
GHSL_BUILTUP = "JRC/GHSL/P2023A/GHS_BUILT_S"

# GHSL population grid
GHSL_POP = "JRC/GHSL/P2023A/GHS_POP"

# Band names inside the two GHSL collections (verified at runtime by
# geospatial/ghsl_features.py, which raises a clear error if they differ).
#   built_surface    = built-up surface in square metres per 100 m cell (0..10000)
#   population_count = people per 100 m cell
GHSL_BUILTUP_BAND = "built_surface"
GHSL_POP_BAND = "population_count"


def initialize_ee():
    """
    Initializes the Earth Engine Python API. Call this once at the start
    of every script that talks to Earth Engine.
    """
    import ee

    if EE_PROJECT_ID == "REPLACE_WITH_YOUR_PROJECT_ID":
        raise RuntimeError(
            "Set GEE_PROJECT_ID environment variable or edit gee_config.py "
            "with your actual Earth Engine Cloud project ID before running."
        )

    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception as e:
        raise RuntimeError(
            "ee.Initialize() failed. Have you run 'earthengine authenticate' "
            "and registered this project for Earth Engine? "
            f"Original error: {e}"
        )

    return ee
