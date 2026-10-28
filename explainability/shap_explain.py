"""
shap_explain.py
------------------
SHAP explanations for the trained model from Module 5.

Provides:
  - build_explainer(model): picks the correct SHAP explainer type for the
    model class (TreeExplainer for RF/XGBoost, LinearExplainer for the
    linear baseline)
  - global_feature_importance(explainer, X): mean absolute SHAP value per
    feature across all rows -- "what matters most overall"
  - local_explanation(explainer, X, row_index): SHAP values for ONE row --
    "why did THIS area get THIS score"

SHAP values are additive: prediction = base_value + sum(shap_values). This
lets each explanation show exactly which features pushed a specific area's
score up or down relative to the average prediction, not just relative
importance.
"""

import numpy as np
import pandas as pd


def build_explainer(model, X_background: pd.DataFrame):
    """
    X_background: a representative sample of training rows, used for
    explainers (e.g. LinearExplainer) that need a background distribution.
    Tree explainers ignore this if not needed but accept it for a uniform
    interface.
    """
    import shap
    from sklearn.linear_model import LinearRegression

    if isinstance(model, LinearRegression):
        return shap.LinearExplainer(model, X_background)
    else:
        # Works for RandomForestRegressor and XGBRegressor
        return shap.TreeExplainer(model)


def global_feature_importance(explainer, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Returns a DataFrame of feature -> mean(|SHAP value|) across all rows in X,
    sorted descending, i.e. overall feature importance according to SHAP
    (not just the tree-based impurity importance, which can be misleading).
    """
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance_df.head(top_n)


def local_explanation(explainer, X: pd.DataFrame, row_index: int, top_n: int = 8) -> dict:
    """
    Returns a dict for ONE row (one area):
        {
          "base_value": float,           # average model prediction over background
          "predicted_value": float,       # base_value + sum(all shap values for this row)
          "top_contributions": [
              {"feature": ..., "value": <feature's actual input value>,
               "shap_contribution": <how much this feature pushed the
                                       prediction up(+)/down(-)>},
              ...
          ]
        }
    """
    row = X.iloc[[row_index]]
    shap_values = explainer.shap_values(row)[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[0])

    predicted_value = float(base_value + shap_values.sum())

    contributions = pd.DataFrame({
        "feature": X.columns,
        "value": row.iloc[0].values,
        "shap_contribution": shap_values,
    })
    contributions["abs_contribution"] = contributions["shap_contribution"].abs()
    top = contributions.sort_values("abs_contribution", ascending=False).head(top_n)

    return {
        "base_value": float(base_value),
        "predicted_value": predicted_value,
        "top_contributions": [
            {
                "feature": r["feature"],
                "value": float(r["value"]),
                "shap_contribution": float(r["shap_contribution"]),
            }
            for _, r in top.iterrows()
        ],
    }


def shap_dict_to_feature_weights(local_expl: dict, all_feature_names: list) -> np.ndarray:
    """
    Converts a local SHAP explanation's contributions for the VGG-16 feature
    columns specifically into a weight vector that can be passed to
    Grad-CAM (explainability/gradcam.py) as `feature_weights`, so the
    heatmap highlights the image regions that mattered most to THIS
    specific prediction rather than an unweighted average.

    Only VGG feature contributions are used (geospatial features aren't
    tied to any pixel region, so they can't inform an image heatmap).
    Features not present in top_contributions default to weight 0.
    """
    weights = {f["feature"]: f["shap_contribution"] for f in local_expl["top_contributions"]}
    vgg_features = [f for f in all_feature_names if f.startswith("vgg_feat_")]
    return np.array([weights.get(f, 0.0) for f in vgg_features], dtype=np.float32)
