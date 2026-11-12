"""
residual_service.py
-------------------
"Hidden Gap Detector": compares the NFHS-5 survey deprivation score of each district with the
satellite-only prediction made when the district's whole state was unseen by the model
(leave-one-state-out), and lists districts where the two disagree.

    gap       = survey_score - satellite_prediction
    rank_gap  = percentile rank of survey score - percentile rank of prediction   (-1 .. +1)

A district becomes a CANDIDATE only if BOTH hold:
  1. the survey score lies outside the model's 80% out-of-state interval, and
  2. its rank gap is at least GAP_RANK_THRESHOLD (default 0.25) in the same direction.

Why two conditions: the model's predictions are compressed toward the average, so the raw gap is strongly
correlated with the survey score itself (about 0.79 on the final data). With the interval test alone, almost
every "hidden deprivation" flag is simply a very deprived district and every "resilient" flag a
least-deprived one. The rank condition asks whether the district is ordered differently from its
satellite estimate, which is less affected by that compression. It is still a screening hint.

Roughly 20% of districts fall outside an 80% interval for a well-calibrated model, so a flag is not
evidence of a cause. A gap can reflect model error, label noise, or deprivation that the ten satellite
and geospatial features (nightlight, NDVI, built-up share, population density, elevation, slope, rainfall,
LST, cropland share, and travel time to cities) cannot capture.

Reads precomputed artefacts only (models/loso_predictions.csv, data/processed/final_training_table.csv);
it never calls or changes the model.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as _config
from config import MODELS_DIR, PROCESSED_DIR

RANK_THRESHOLD = getattr(_config, "GAP_RANK_THRESHOLD", 0.25)
MIN_STATE_DISTRICTS = getattr(_config, "MIN_STATE_DISTRICTS", 5)

HIDDEN = "Hidden-deprivation candidate"
RESILIENT = "Resilient candidate"
CONSISTENT = "Consistent"

_cache = None


def _load() -> pd.DataFrame:
    global _cache
    if _cache is not None:
        return _cache

    loso_path = os.path.join(MODELS_DIR, "loso_predictions.csv")
    table_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    if not os.path.exists(loso_path) or not os.path.exists(table_path):
        raise FileNotFoundError("loso_predictions.csv / final_training_table.csv not found.")

    loso = pd.read_csv(loso_path)
    cols = ["area_id", "area_name"] + (["match_method"] if "match_method" in pd.read_csv(table_path, nrows=1).columns else [])
    names = pd.read_csv(table_path)[cols]
    df = loso.merge(names, on="area_id", how="left")
    df["area_name"] = df["area_name"].fillna("Unknown")
    if "match_method" not in df.columns:
        df["match_method"] = "unknown"
    df["match_method"] = df["match_method"].fillna("unknown")

    df["gap"] = df["y_true"] - df["y_pred_out_of_state"]
    df["rank_true"] = df["y_true"].rank(pct=True)
    df["rank_pred"] = df["y_pred_out_of_state"].rank(pct=True)
    df["rank_gap"] = df["rank_true"] - df["rank_pred"]
    df["outside_80"] = (df["y_true"] < df["low_80"]) | (df["y_true"] > df["high_80"])

    hidden = (df["y_true"] > df["high_80"]) & (df["rank_gap"] >= RANK_THRESHOLD)
    resilient = (df["y_true"] < df["low_80"]) & (df["rank_gap"] <= -RANK_THRESHOLD)
    df["gap_class"] = np.where(hidden, HIDDEN, np.where(resilient, RESILIENT, CONSISTENT))
    _cache = df
    return _cache


def reset_cache():
    global _cache
    _cache = None


def get_residual_atlas(direction: str = "all", state: str = None, limit: int = 25) -> dict:
    df = _load()
    direction = (direction or "all").lower()
    if direction not in ("all", "hidden", "resilient"):
        raise ValueError("direction must be one of: all, hidden, resilient")
    limit = max(1, min(int(limit), 200))

    view = df
    if state:
        view = view[view["state_name"].str.lower() == state.strip().lower()]
    if direction == "hidden":
        view = view[view["gap_class"] == HIDDEN]
    elif direction == "resilient":
        view = view[view["gap_class"] == RESILIENT]
    else:
        view = view[view["gap_class"] != CONSISTENT]
    view = view.reindex(view["rank_gap"].abs().sort_values(ascending=False).index).head(limit)

    # State-level systematic bias. States with very few districts are marked unreliable.
    g = df.groupby("state_name")
    states = pd.DataFrame({
        "n": g.size(),
        "mean_gap": g["gap"].mean(),
        "mae": g["gap"].apply(lambda s: s.abs().mean()),
        "hidden": g["gap_class"].apply(lambda s: int((s == HIDDEN).sum())),
        "resilient": g["gap_class"].apply(lambda s: int((s == RESILIENT).sum())),
    }).reset_index()
    states["reliable"] = states["n"] >= MIN_STATE_DISTRICTS
    states["mean_gap"] = states["mean_gap"].round(4)
    states["mae"] = states["mae"].round(4)
    states = states.assign(_abs=states["mean_gap"].abs()).sort_values(
        ["reliable", "_abs"], ascending=[False, False]).drop(columns="_abs")

    counts = df["gap_class"].value_counts().to_dict()
    items = [
        {
            "area_id": int(r.area_id),
            "area_name": r.area_name,
            "state_name": r.state_name,
            "survey_score": round(float(r.y_true), 4),
            "satellite_prediction": round(float(r.y_pred_out_of_state), 4),
            "low_80": round(float(r.low_80), 4),
            "high_80": round(float(r.high_80), 4),
            "gap": round(float(r.gap), 4),
            "rank_gap": round(float(r.rank_gap), 3),
            "class": r.gap_class,
            "match_method": r.match_method,
            # Only approximate (fuzzy) name matches are uncertain; alias and the Andhra Pradesh to
            # Telangana fallback are explicit, reviewed rules.
            "match_caution": r.match_method == "fuzzy",
        }
        for r in view.itertuples()
    ]

    return {
        "summary": {
            "n_districts": int(len(df)),
            "hidden_candidates": int(counts.get(HIDDEN, 0)),
            "resilient_candidates": int(counts.get(RESILIENT, 0)),
            "consistent": int(counts.get(CONSISTENT, 0)),
            "outside_80_interval": int(df["outside_80"].sum()),
            "share_outside_80": round(float(df["outside_80"].mean()), 4),
            "rank_threshold": RANK_THRESHOLD,
            "gap_std": round(float(df["gap"].std(ddof=1)), 4),
            "correlation_survey_vs_prediction": round(float(np.corrcoef(df["y_true"], df["y_pred_out_of_state"])[0, 1]), 4),
            "correlation_gap_vs_survey": round(float(np.corrcoef(df["gap"], df["y_true"])[0, 1]), 4),
        },
        "districts": items,
        "state_bias": states.to_dict(orient="records"),
        "min_state_districts": MIN_STATE_DISTRICTS,
        "note": (
            "Districts where the survey ranking and the satellite-based estimate disagree. A district is listed "
            "only if its survey score is outside the 80% out-of-state interval AND its rank differs by at least "
            f"{int(RANK_THRESHOLD * 100)} percentile points. A gap can reflect model error or deprivation that the "
            "ten satellite and geospatial features cannot capture (for example nutrition or schooling). About 20% of districts "
            "fall outside an 80% interval for a well-calibrated model, so these are screening hints for local "
            "verification, not proof of hidden poverty or of a cause. Aggregate district level only."
        ),
    }
