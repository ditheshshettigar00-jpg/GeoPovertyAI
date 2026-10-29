"""
transformers.py
---------------
Reusable Scikit-Learn transformers for GeoPoverty AI preprocessing pipelines.
Placed in a dedicated module so pickled pipelines can be deserialized reliably
across processes, test runners, and API servers.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips features to [Q1 - k*IQR, Q3 + k*IQR] calculated strictly on training split."""
    def __init__(self, iqr_multiplier=3.0):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        q1 = np.nanquantile(X_arr, 0.25, axis=0)
        q3 = np.nanquantile(X_arr, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.iqr_multiplier * iqr
        self.upper_bounds_ = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X):
        X_arr = np.asarray(X).copy()
        if self.lower_bounds_ is not None and self.upper_bounds_ is not None:
            return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)
        return X_arr


class TrainingMedianImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using column medians computed strictly on the training fold."""
    def __init__(self):
        self.medians_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.medians_ = np.nanmedian(X_arr, axis=0)
        self.medians_ = np.nan_to_num(self.medians_, nan=0.0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X).copy()
        if self.medians_ is not None:
            inds = np.where(np.isnan(X_arr))
            if len(inds[0]) > 0:
                X_arr[inds] = np.take(self.medians_, inds[1])
        return X_arr


class MeanPredictor(BaseEstimator):
    """Zero-skill baseline: predicts the mean of the training fold."""
    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full(n_samples, self.mean_)
