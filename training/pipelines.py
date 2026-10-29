"""
pipelines.py
------------
Model pipelines for GeoPoverty AI. Every data-dependent step (outlier clipping,
imputation, scaling, PCA) lives INSIDE the sklearn Pipeline, so when the pipeline
is fitted on a training fold these steps learn only from that fold. Nothing is
fitted on the full table before splitting, which is what keeps the state-held-out
scores honest.

Note on imputation: held-out states are by definition unseen, so a per-state median
could never be used for them. The imputer therefore uses the median of the TRAINING
fold. Missing values stay NaN in the table (see scripts/build_real_training_table.py).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Feature groups used for the ablation / group-importance analysis.
FEATURE_GROUPS = {
    "nightlight": ["nightlight_mean"],
    "vegetation": ["ndvi_mean"],
    "built_up": ["builtup_pct"],
    "population": ["population_density"],
    "accessibility": ["road_density_km_per_sqkm", "school_distance_km", "hospital_distance_km"],
    "visual": "prefix:vgg_feat_",
}


def group_columns(group, columns):
    spec = FEATURE_GROUPS[group]
    if isinstance(spec, str) and spec.startswith("prefix:"):
        p = spec.split(":", 1)[1]
        return [c for c in columns if c.startswith(p)]
    return [c for c in spec if c in columns]


def visual_columns(columns):
    return [c for c in columns if c.startswith("vgg_feat_")]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips each column to [Q1 - k*IQR, Q3 + k*IQR] using bounds learned on the training data only.
    NaN stays NaN. A column with zero IQR is left alone."""

    def __init__(self, factor=3.0):
        self.factor = factor

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        self.feature_names_in_ = np.array(X.columns) if hasattr(X, "columns") else np.array([f"x{i}" for i in range(A.shape[1])])
        q1 = np.nanpercentile(A, 25, axis=0)
        q3 = np.nanpercentile(A, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - self.factor * iqr, -np.inf)
        self.upper_ = np.where(iqr > 0, q3 + self.factor * iqr, np.inf)
        return self

    def transform(self, X):
        A = np.asarray(X, dtype=float)
        return np.clip(A, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def _make_estimator(name, seed, params=None):
    params = dict(params or {})
    if name == "Dummy":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return Ridge(**{"alpha": 10.0, **params})
    if name == "RandomForest":
        return RandomForestRegressor(**{"n_estimators": 300, "min_samples_leaf": 3, "max_features": 0.5,
                                        "random_state": seed, "n_jobs": -1, **params})
    if name == "GradientBoosting":
        return GradientBoostingRegressor(**{"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                                            "subsample": 0.8, "random_state": seed, **params})
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**{"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
                               "colsample_bytree": 0.8, "reg_lambda": 2.0, "random_state": seed,
                               "n_jobs": -1, "objective": "reg:squarederror", **params})
    raise ValueError(f"Unknown model: {name}")


def available_models():
    models = ["Dummy", "Ridge", "RandomForest", "GradientBoosting"]
    try:
        import xgboost  # noqa: F401
        models.append("XGBoost")
    except Exception:
        print("NOTE: xgboost is not installed; XGBoost is skipped (pip install xgboost to include it).")
    return models


# Search spaces for nested tuning (parameter names are for the final 'model' step).
PARAM_GRIDS = {
    "Ridge": {"model__alpha": [0.1, 1.0, 10.0, 30.0, 100.0]},
    "RandomForest": {"model__n_estimators": [200, 300], "model__max_depth": [None, 6, 10],
                     "model__min_samples_leaf": [2, 3, 5, 8], "model__max_features": [0.3, 0.5, 1.0]},
    "GradientBoosting": {"model__n_estimators": [100, 200, 300], "model__max_depth": [2, 3, 4],
                         "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.7, 0.85, 1.0]},
    "XGBoost": {"model__n_estimators": [100, 200, 300], "model__max_depth": [2, 3, 5],
                "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.7, 0.85, 1.0],
                "model__reg_lambda": [1.0, 2.0, 5.0]},
}


def resolve_visual_mode(mode, feature_cols):
    """auto: keep up to 32 visual columns as they are, otherwise compress with PCA inside the pipeline."""
    n_vis = len(visual_columns(feature_cols))
    if n_vis == 0:
        return "none"
    if mode == "auto":
        return "all" if n_vis <= 32 else "pca"
    return mode


def build_pipeline(model_name, feature_cols, visual_mode="all", seed=42, params=None, pca_components=16):
    """clip -> impute -> (scale for Ridge) -> [PCA on visual columns] -> model."""
    vis = visual_columns(feature_cols)
    steps = [("clip", IQRClipper(3.0)),
             ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if model_name == "Ridge":
        steps.append(("scale", StandardScaler()))
    if visual_mode == "pca" and vis:
        tab = [c for c in feature_cols if c not in vis]
        n_comp = max(1, min(pca_components, len(vis)))
        steps.append(("visual_pca", ColumnTransformer(
            [("tab", "passthrough", tab),
             ("vis", Pipeline([("z", StandardScaler()), ("pca", PCA(n_components=n_comp, random_state=seed))]), vis)],
            verbose_feature_names_out=False)))
    pre = Pipeline(steps)
    pre.set_output(transform="pandas")
    return Pipeline([("pre", pre), ("model", _make_estimator(model_name, seed, params))])
