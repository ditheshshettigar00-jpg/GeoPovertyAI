"""
image_preprocessing.py
-------------------------
Shared image preprocessing utilities used before VGG-16 feature extraction.

Handles both:
  - Standard image formats (.png/.jpg) -- e.g. the demo images
  - GeoTIFF satellite patches (.tif) from Module 2's download_satellite_patches.py

Pipeline per image:
    load -> (if multi-band GeoTIFF) select RGB bands -> resize to 224x224
          -> convert to float array -> apply Keras VGG16 preprocess_input
"""

import os

import numpy as np
from PIL import Image

IMG_SIZE = 224


def load_image_as_array(path: str) -> np.ndarray:
    """
    Loads an image file and returns an (H, W, 3) uint8 RGB array.
    Supports .png/.jpg/.jpeg directly via PIL, and .tif/.tiff via rasterio
    (falls back to PIL for .tif if rasterio is unavailable or the file is
    a simple 3-band RGB GeoTIFF).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg"):
        img = Image.open(path).convert("RGB")
        return np.array(img)

    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                # Assumes band order matches export order from Module 2 (B4,B3,B2 -> R,G,B)
                n_bands = src.count
                if n_bands >= 3:
                    arr = src.read([1, 2, 3])  # (3, H, W)
                else:
                    band = src.read(1)
                    arr = np.stack([band, band, band], axis=0)
                arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr
        except ImportError:
            img = Image.open(path).convert("RGB")
            return np.array(img)

    raise ValueError(f"Unsupported image format: {ext}")


def resize_and_normalize(img_array: np.ndarray) -> np.ndarray:
    """
    Resizes an (H, W, 3) uint8 array to 224x224 and applies VGG-16-style
    preprocessing (BGR channel reorder + ImageNet mean subtraction), matching
    what tensorflow.keras.applications.vgg16.preprocess_input does, so
    pretrained VGG-16 weights behave as intended.

    Returns a float32 array of shape (224, 224, 3), ready to be batched
    and passed to the VGG-16 feature extractor.
    """
    from tensorflow.keras.applications.vgg16 import preprocess_input

    img = Image.fromarray(img_array).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)  # handles BGR conversion + ImageNet mean subtraction
    return arr


def preprocess_image_file(path: str) -> np.ndarray:
    """Convenience wrapper: file path -> preprocessed (224, 224, 3) array."""
    raw = load_image_as_array(path)
    return resize_and_normalize(raw)
