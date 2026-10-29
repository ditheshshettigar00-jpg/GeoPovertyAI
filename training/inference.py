"""
inference.py
------------
Loads the saved model bundle and returns predictions with an uncertainty range.
The interval comes from leave-one-state-out residuals (how wrong the model was for states it had
not seen), so it reflects real out-of-state error rather than training error.

Used by the backend (Phase 4): predict_with_interval(bundle, X) -> (pred, low, high).
"""

import numpy as np


def predict_with_interval(bundle, X, level="80"):
    """X: DataFrame with bundle['feature_columns']. Returns arrays clipped to [0, 1]."""
    pred = np.asarray(bundle["model"].predict(X[bundle["feature_columns"]]), dtype=float)
    q = bundle["interval"][level]
    low = np.clip(pred + q["lower"], 0.0, 1.0)
    high = np.clip(pred + q["upper"], 0.0, 1.0)
    return np.clip(pred, 0.0, 1.0), low, high
