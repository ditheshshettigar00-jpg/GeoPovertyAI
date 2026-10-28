"""
gradcam.py
------------
Grad-CAM for a VGG-16 FEATURE EXTRACTOR (not a classifier).

IMPORTANT DESIGN NOTE: textbook Grad-CAM explains a classifier's predicted
CLASS score. Our VGG-16 is used as a feature extractor (Module 3) feeding
a separate XGBoost/Random Forest model (Module 5) -- there is no single
"class score" to differentiate. We adapt Grad-CAM to this setting by
computing gradients with respect to a SCALAR target we define ourselves:

    target_score = sum_i ( feature_i * weight_i )

where `weight_i` defaults to all-ones (i.e. explains what the image
contributed to the overall pooled feature vector), but can be set to the
per-feature SHAP importances or model feature-importances for the VGG
feature columns once Module 5's model is trained -- in that case the
heatmap highlights specifically the image regions that mattered most to
the actual deprivation prediction, which is the more meaningful version
to show in the final dashboard.

This is a documented adaptation of Grad-CAM (Selvaraju et al., 2017) to a
regression/feature-attribution setting, not the original classification
formulation -- this should be described as such in your report.
"""

import os

import numpy as np
from PIL import Image

LAST_CONV_LAYER_NAME = "block5_conv3"  # VGG-16's last convolutional layer


def build_gradcam_model(vgg_model):
    """
    Builds a model that outputs both:
      - the last convolutional layer's activations (for the heatmap)
      - the global-average-pooled feature vector (our "prediction" proxy)
    from the same frozen VGG-16 base used in vgg16_feature_extractor.py.
    """
    import tensorflow as tf

    conv_layer = vgg_model.get_layer(LAST_CONV_LAYER_NAME)
    grad_model = tf.keras.models.Model(
        inputs=vgg_model.inputs,
        outputs=[conv_layer.output, vgg_model.output],
    )
    return grad_model


def compute_gradcam_heatmap(grad_model, preprocessed_image, feature_weights=None):
    """
    preprocessed_image: (224, 224, 3) float32 array, already run through
                          preprocess_image_file()
    feature_weights: optional (512,) array weighting each pooled feature's
                       contribution to the target score. Defaults to all-ones.

    Returns: (H, W) heatmap array, values in [0, 1]
    """
    import tensorflow as tf

    x = np.expand_dims(preprocessed_image, axis=0)
    x_tensor = tf.convert_to_tensor(x)

    with tf.GradientTape() as tape:
        conv_output, pooled_features = grad_model(x_tensor)
        if feature_weights is None:
            weights = tf.ones_like(pooled_features)
        else:
            weights = tf.convert_to_tensor(feature_weights.reshape(1, -1), dtype=tf.float32)
        target_score = tf.reduce_sum(pooled_features * weights)

    grads = tape.gradient(target_score, conv_output)  # (1, h, w, channels)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (channels,)

    conv_output = conv_output[0]  # (h, w, channels)
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)  # (h, w)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    original_rgb: (H, W, 3) uint8 array (the raw image, NOT preprocessed)
    heatmap: (h, w) array in [0, 1] from compute_gradcam_heatmap
    Returns a PIL Image with the heatmap overlaid using a red-yellow colormap.
    """
    import matplotlib
    import matplotlib.cm as cm

    h_img, w_img = original_rgb.shape[:2]
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w_img, h_img), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    colormap = matplotlib.colormaps["jet"] if hasattr(matplotlib, "colormaps") else cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_resized)[:, :, :3]  # drop alpha channel
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

    original = original_rgb.astype(np.float32)
    overlay = (1 - alpha) * original + alpha * colored_heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def generate_gradcam_for_image(vgg_model, image_path: str, out_path: str, feature_weights=None):
    """
    End-to-end: loads an image, computes Grad-CAM, saves the overlay PNG.
    """
    from preprocessing.image_preprocessing import load_image_as_array, resize_and_normalize

    raw = load_image_as_array(image_path)
    raw_resized = np.array(Image.fromarray(raw).resize((224, 224)))
    preprocessed = resize_and_normalize(raw)

    grad_model = build_gradcam_model(vgg_model)
    heatmap = compute_gradcam_heatmap(grad_model, preprocessed, feature_weights)
    overlay_img = overlay_heatmap_on_image(raw_resized, heatmap)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    overlay_img.save(out_path)
    return out_path
