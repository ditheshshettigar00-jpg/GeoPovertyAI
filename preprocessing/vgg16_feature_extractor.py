"""
vgg16_feature_extractor.py
-----------------------------
Loads VGG-16 (ImageNet pretrained) as a FROZEN feature extractor and
converts each 224x224x3 image patch into a fixed-length feature vector.

DESIGN CHOICE (documented, not arbitrary): we use
    VGG16(weights='imagenet', include_top=False, pooling='avg')
which applies global average pooling to the last convolutional block's
output, giving a compact 512-dimensional feature vector per image --
rather than the larger 4096-d vector from the original fc2 layer. This
keeps the downstream fused feature table (Module 4) a manageable size for
Random Forest / XGBoost on a normal laptop, at the cost of losing some
spatial fine detail that the fc layers would have encoded. This tradeoff
should be stated explicitly in your project report.

Layers are frozen (not trainable) -- this is transfer learning via fixed
feature extraction, not fine-tuning. Fine-tuning is only worth attempting
later if you find you have enough labeled districts to avoid overfitting
the CNN (see documentation on this tradeoff in the design document).

Usage:
    python preprocessing/vgg16_feature_extractor.py \\
        --image_dir data/satellite/demo --out data/processed/vgg16_features_demo.csv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.image_preprocessing import preprocess_image_file
from config import PROCESSED_DIR

FEATURE_DIM = 512  # output dimensionality of the pooling='avg' VGG16 feature extractor


def build_feature_extractor():
    """
    Builds the frozen VGG-16 feature extractor.

    NOTE: on first run, Keras downloads the pretrained ImageNet weights
    (~58MB) automatically from storage.googleapis.com -- this requires a
    normal internet connection and happens transparently; no action needed
    from you beyond having internet access the first time you run this.
    """
    from tensorflow.keras.applications import VGG16

    model = VGG16(weights="imagenet", include_top=False, pooling="avg",
                   input_shape=(224, 224, 3))
    model.trainable = False  # freeze all convolutional layers
    return model


def extract_feature_vector(model, image_path: str) -> np.ndarray:
    """Runs one image through the frozen VGG-16 and returns its 512-d feature vector."""
    x = preprocess_image_file(image_path)
    x = np.expand_dims(x, axis=0)  # add batch dimension -> (1, 224, 224, 3)
    features = model.predict(x, verbose=0)  # -> (1, 512)
    return features[0]


def extract_area_id_from_filename(path: str):
    """Expects filenames like district_003_rgb.tif or district_003_DEMO.png"""
    match = re.search(r"district_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def extract_features_for_directory(image_dir: str, out_csv: str) -> pd.DataFrame:
    model = build_feature_extractor()

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.tif"))
        + glob.glob(os.path.join(image_dir, "*.tiff"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    rows = []
    for path in image_paths:
        area_id = extract_area_id_from_filename(path)
        if area_id is None:
            print(f"  SKIPPED {path}: could not parse area_id from filename "
                  f"(expected pattern 'district_<id>_...')")
            continue

        print(f"Extracting VGG-16 features for area_id={area_id} ({os.path.basename(path)}) ...")
        feat = extract_feature_vector(model, path)

        row = {"area_id": area_id, "image_path": path}
        for i, val in enumerate(feat, start=1):
            row[f"vgg_feat_{i}"] = float(val)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("area_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} feature vectors ({FEATURE_DIM}-d each) to: {out_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract VGG-16 features for all images in a directory")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_csv = args.out or os.path.join(PROCESSED_DIR, "vgg16_features.csv")
    extract_features_for_directory(args.image_dir, out_csv)


if __name__ == "__main__":
    main()
