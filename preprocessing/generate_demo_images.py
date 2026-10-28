"""
generate_demo_images.py
--------------------------
Generates SYNTHETIC placeholder "satellite-style" RGB images for each demo
area from Module 1, so the preprocessing + VGG-16 pipeline can be built and
tested end-to-end before real Sentinel-2 patches (Module 2) are available.

These are simple procedurally generated textures (NOT real satellite
imagery) whose average color/texture is loosely tied to that area's
builtup_pct from the demo dataset, purely so Grad-CAM has something
visually structured to highlight during development.

THIS IS NOT REAL SATELLITE IMAGERY. Every file is saved with a
"_DEMO" suffix and this is documented in every place that reads it.

Usage:
    python preprocessing/generate_demo_images.py
"""

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEMO_TRAINING_CSV, SATELLITE_DIR

IMG_SIZE = 224


def make_synthetic_patch(builtup_pct: float, ndvi_mean: float, seed: int) -> np.ndarray:
    """
    Builds a very simple synthetic RGB texture:
    - more builtup_pct -> more grey/brown blocky patterns (urban-ish look)
    - more ndvi_mean -> more green tint (vegetation-ish look)
    This is a crude visual stand-in only, not a simulation of real remote
    sensing physics.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    green_tint = np.clip(ndvi_mean, 0, 1)
    grey_tint = np.clip(builtup_pct / 100.0, 0, 1)

    base_green = np.array([60, 140, 60])
    base_grey = np.array([130, 125, 120])
    base_soil = np.array([150, 120, 90])

    base_color = (green_tint * base_green
                  + grey_tint * base_grey
                  + (1 - green_tint - grey_tint).clip(0, 1) * base_soil)

    # blocky texture: random tiles for a bit of visual structure
    tile = 16
    for ty in range(0, IMG_SIZE, tile):
        for tx in range(0, IMG_SIZE, tile):
            jitter = rng.normal(0, 18, size=3)
            color = np.clip(base_color + jitter, 0, 255)
            img[ty:ty + tile, tx:tx + tile] = color

    noise = rng.normal(0, 6, size=img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def main():
    if not os.path.exists(DEMO_TRAINING_CSV):
        raise FileNotFoundError(
            f"{DEMO_TRAINING_CSV} not found. Run scripts/generate_demo_data.py first (Module 1)."
        )

    df = pd.read_csv(DEMO_TRAINING_CSV)
    out_dir = os.path.join(SATELLITE_DIR, "demo")
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for _, row in df.iterrows():
        img_arr = make_synthetic_patch(
            builtup_pct=row["builtup_pct"],
            ndvi_mean=row["ndvi_mean"],
            seed=int(row["area_id"]),
        )
        out_path = os.path.join(out_dir, f"district_{int(row['area_id']):03d}_DEMO.png")
        Image.fromarray(img_arr, mode="RGB").save(out_path)
        saved.append(out_path)

    print(f"Generated {len(saved)} synthetic demo images in: {out_dir}")
    print("REMINDER: these are synthetic placeholder textures, NOT real satellite imagery.")


if __name__ == "__main__":
    main()
