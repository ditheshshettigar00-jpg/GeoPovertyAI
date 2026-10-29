"""
ghsl_features.py
----------------
Shared, unit-correct extraction of GHSL built-up share and population density
for one district geometry. Used by both extract_geospatial_features.py and
scripts/extract_all_india_features.py so the two can never disagree.

WHY THIS EXISTS (Phase 1 fix): the old code took "the first band of the most
recent image" and stored the raw mean as if it were a percentage. GHSL values
are per 100 m cell:
    built_surface    -> square metres of built-up area in a 100 m x 100 m cell
                        (0..10000), so a percentage needs / 10000 * 100
    population_count -> people per 100 m cell, so people per km2 needs * 100
The old values reached 4,057 for "builtup_pct". The most recent GHSL epoch is
also a 2030 projection, so a fixed survey-period epoch is selected instead.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GHSL_CELL_AREA_M2, GHSL_EPOCH_YEAR
from geospatial.gee_config import GHSL_BUILTUP, GHSL_BUILTUP_BAND, GHSL_POP, GHSL_POP_BAND

_checked = {}


def _epoch_image(ee, collection_id, band, year):
    """Returns the single-band image for the given epoch year, verifying the band exists."""
    col = ee.ImageCollection(collection_id).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    img = col.first()
    key = (collection_id, band, year)
    if key not in _checked:
        names = ee.Image(img).bandNames().getInfo()
        if band not in names:
            raise RuntimeError(
                f"Band '{band}' not found in {collection_id} for epoch {year}. "
                f"Available bands: {names}. Update GHSL_*_BAND in geospatial/gee_config.py."
            )
        _checked[key] = True
    return img.select(band)


def _mean_over(ee, image, geometry, band):
    out = image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=100,
        maxPixels=1e9, bestEffort=True, tileScale=4,
    ).getInfo()
    return out.get(band) if out else None


def builtup_pct_for(ee, geometry, year=GHSL_EPOCH_YEAR):
    """Mean built-up share of the district as a true percentage in [0, 100]."""
    img = _epoch_image(ee, GHSL_BUILTUP, GHSL_BUILTUP_BAND, year)
    mean_m2 = _mean_over(ee, img, geometry, GHSL_BUILTUP_BAND)
    if mean_m2 is None:
        return None
    return float(min(max(mean_m2 / GHSL_CELL_AREA_M2 * 100.0, 0.0), 100.0))


def population_density_for(ee, geometry, year=GHSL_EPOCH_YEAR):
    """Mean population density of the district in people per km2."""
    img = _epoch_image(ee, GHSL_POP, GHSL_POP_BAND, year)
    mean_per_cell = _mean_over(ee, img, geometry, GHSL_POP_BAND)
    if mean_per_cell is None:
        return None
    cells_per_km2 = 1_000_000.0 / GHSL_CELL_AREA_M2
    return float(mean_per_cell * cells_per_km2)
