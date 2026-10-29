"""
train_models.py  (Phase 2)
--------------------------
Trains, evaluates and saves the poverty model with state-held-out validation.

WHAT IT DOES
  1. Evaluates every model with LEAVE-ONE-STATE-OUT (main result: each prediction is for a state the
     model never saw) and with repeated state-grouped 5-fold CV (second view, mean +- std).
  2. All clipping / imputation / scaling / PCA happen inside the sklearn Pipeline, so they are fitted on
     training folds only. Missing values in the table are fine.
  3. Baselines: predict-the-training-mean and Ridge, next to RandomForest, GradientBoosting and XGBoost
     (XGBoost is used when installed).
  4. Ablation: which feature groups actually help (tabular only / visual only / both, plus drop-one-group).
  5. A random K-fold run labelled LEAKY, kept apart, only to show how much leakage would inflate scores.
  6. The final model is the best non-baseline by pooled leave-one-state-out RMSE, refitted on all rows.
  7. Prediction intervals from the leave-one-state-out residuals.

OUTPUTS (in --models-dir, default models/)
  model_comparison_results.csv      main table (leave-one-state-out), columns model,r2,mae,rmse,...
  repeated_groupkfold_results.csv   second view, mean and std over repeats
  per_state_errors.csv              error of the final model per held-out state
  loso_predictions.csv              district-level out-of-state predictions of the final model
  ablation_results.csv, feature_group_importance.csv, figures/*.png
  leaky_random_kfold_results.csv    LEAKY, for illustration only
  best_model.pkl                    bundle: model (Pipeline), model_name, feature_columns, interval, ...
  feature_columns.json, category_thresholds.json, uncertainty.json, run_metadata.json

Usage:
    python training/train_models.py --data data/processed/final_training_table.csv
    python training/train_models.py --data ... --tuning nested --n-iter 6   # slower, tunes inside each fold
    python training/train_models.py --data ... --fast                        # quick smoke run
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, MODELS_DIR, RANDOM_SEED
from training.evaluation import (
    loso_predictions, per_state_table, random_kfold_scores, repeated_group_scores, summarize,
)
from training.geo_validation import get_group_kfold
from training.pipelines import (
    FEATURE_GROUPS, PARAM_GRIDS, available_models, build_pipeline, group_columns,
    resolve_visual_mode, visual_columns,
)

NON_FEATURE_COLS = {
    "area_id", "area_name", "state_name", "data_source", "data_year", "deprivation_score",
    "latitude", "longitude", "centroid_lat", "centroid_lon", "data_mode",
    "image_path", "s2_image_count", "match_score",
}
# latitude/longitude are deliberately NOT features: raw coordinates would let the model memorise location.


def get_feature_columns(df: pd.DataFrame):
    """All numeric columns except identifiers/target."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def log_results_to_db(results: pd.DataFrame, trained_on: str, db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        print(f"  NOTE: database not found at {db_path} -- skipping DB logging.")
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for name, m in results.iterrows():
        cur.execute("""INSERT INTO model_results (model_name, mae, rmse, r2, trained_on, eval_timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (name, float(m["mae"]), float(m["rmse"]), float(m["r2"]), trained_on,
                     datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    print(f"  Logged {len(results)} model result(s) to {db_path}")


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def _interval_from_residuals(pred_df):
    res = pred_df["y_true"].values - pred_df["y_pred"].values
    out = {}
    for level, (lo, hi) in {"80": (10, 90), "90": (5, 95)}.items():
        out[level] = {"lower": float(np.percentile(res, lo)), "upper": float(np.percentile(res, hi))}
    out["note"] = ("Add 'lower'/'upper' to a prediction to get the interval. Derived from leave-one-state-out "
                   "residuals, i.e. the error made for states the model had not seen.")
    return out


def _final_fit(name, X, y, groups, feature_cols, visual_mode, seed, tuning, n_iter):
    pipe = build_pipeline(name, feature_cols, visual_mode, seed)
    if tuning == "nested" and name in PARAM_GRIDS:
        cv = list(get_group_kfold(len(set(groups)), max_splits=5).split(X, y, groups=groups))
        search = RandomizedSearchCV(pipe, PARAM_GRIDS[name], n_iter=n_iter, cv=cv,
                                    scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=1)
        search.fit(X, y)
        print(f"  Final-model hyperparameters: {search.best_params_}")
        return search.best_estimator_
    est = clone(pipe)
    est.fit(X, y)
    return est


def _plots(models_dir, results, group_imp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = os.path.join(models_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    r = results.sort_values("rmse")
    ax.barh(r.index, r["mae"], color="#2a6f97")
    base = results.loc["Dummy", "mae"] if "Dummy" in results.index else None
    if base is not None:
        ax.axvline(base, color="#c1121f", linestyle="--", label="predict-the-mean baseline")
        ax.legend()
    ax.set_xlabel("MAE on unseen states (lower is better)")
    ax.set_title("Leave-one-state-out comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)

    if group_imp is not None and len(group_imp):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        g = group_imp.sort_values("r2_drop_when_removed")
        ax.barh(g.index, g["r2_drop_when_removed"], color="#38a3a5")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Drop in R2 when the group is removed (bigger = more useful)")
        ax.set_title("Feature-group importance (leave-one-state-out)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "feature_group_importance.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate the poverty model with state-held-out validation")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--tuning", choices=["none", "nested"], default="none",
                    help="'nested' tunes hyperparameters inside every training fold (slower, most rigorous)")
    ap.add_argument("--n-iter", type=int, default=6, help="random-search iterations when --tuning nested")
    ap.add_argument("--visual", choices=["auto", "none", "all", "pca"], default="auto")
    ap.add_argument("--repeats", type=int, default=5, help="repeats for repeated state-grouped 5-fold CV")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--fast", action="store_true", help="2 repeats, skip ablation (smoke run)")
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()
    if args.fast:
        args.repeats, args.skip_ablation = 2, True

    df = pd.read_csv(args.data)
    if "state_name" not in df.columns:
        raise ValueError("Dataset must contain a 'state_name' column for state-held-out validation.")
    n_before = len(df)
    df = df.dropna(subset=["deprivation_score"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with a missing target.")

    mode_col = df["data_mode"].astype(str).str.lower() if "data_mode" in df.columns else pd.Series(dtype=str)
    data_mode = "demo" if ("demo" in os.path.basename(args.data).lower() or mode_col.eq("demo").any()) else "real"

    feature_cols = get_feature_columns(df)
    visual_mode = resolve_visual_mode(args.visual, feature_cols)
    if visual_mode == "none":
        feature_cols = [c for c in feature_cols if c not in visual_columns(feature_cols)]
    X, y, groups = df[feature_cols], df["deprivation_score"], df["state_name"].values
    n_vis = len(visual_columns(feature_cols))
    print(f"{len(df)} districts, {len(set(groups))} states, {len(feature_cols)} features "
          f"({n_vis} visual, visual mode: {visual_mode}), {int(X.isna().sum().sum())} missing cells")

    os.makedirs(args.models_dir, exist_ok=True)
    names = available_models()

    # ---- 1. main evaluation: leave-one-state-out ----
    print("\n=== Leave-one-state-out evaluation ===")
    loso, loso_pred = {}, {}
    for name in names:
        pipe = build_pipeline(name, feature_cols, visual_mode, args.seed)
        pred = loso_predictions(pipe, name, X, y, groups, args.tuning, args.n_iter, args.seed)
        loso_pred[name] = pred
        loso[name] = summarize(pred)
        m = loso[name]
        print(f"  {name:<17} R2 {m['r2']:+.3f}  MAE {m['mae']:.4f}  RMSE {m['rmse']:.4f}  "
              f"Spearman {m['spearman']:.2f}  3-class {m['cat_accuracy']:.1%}  top20 {m['top20_recall']:.0%}")
    results = pd.DataFrame(loso).T
    results.index.name = "model"
    base_mae = results.loc["Dummy", "mae"]
    results["beats_mean_baseline"] = results["mae"] < base_mae
    results["mae_vs_baseline_pct"] = ((results["mae"] / base_mae - 1) * 100).round(1)
    results["cv_scheme"] = "leave-one-state-out"
    results["tuning"] = args.tuning
    results.round(4).to_csv(os.path.join(args.models_dir, "model_comparison_results.csv"))

    # ---- 2. second view: repeated state-grouped K-fold ----
    print(f"\n=== Repeated state-grouped 5-fold CV ({args.repeats} repeats) ===")
    rep_rows = {}
    for name in names:
        pipe = build_pipeline(name, feature_cols, visual_mode, args.seed)
        rep_rows[name] = repeated_group_scores(pipe, name, X, y, groups, 5, args.repeats, args.tuning, args.n_iter, args.seed)
        r = rep_rows[name]
        print(f"  {name:<17} R2 {r['r2_mean']:+.3f} +- {r['r2_std']:.3f}   MAE {r['mae_mean']:.4f} +- {r['mae_std']:.4f}")
    rep = pd.DataFrame(rep_rows).T
    rep.index.name = "model"
    rep.round(4).to_csv(os.path.join(args.models_dir, "repeated_groupkfold_results.csv"))

    # ---- 3. choose the final model ----
    candidates = results.drop(index="Dummy")
    best = candidates["rmse"].idxmin()
    print(f"\nBest model by leave-one-state-out RMSE: {best} "
          f"(R2 {results.loc[best, 'r2']:+.3f}, MAE {results.loc[best, 'mae']:.4f} vs mean-baseline MAE {base_mae:.4f})")
    if not results.loc[best, "beats_mean_baseline"]:
        print("WARNING: the best model does NOT beat predicting the mean on unseen states. Do not claim predictive skill.")

    # ---- 4. LEAKY run, kept separate ----
    leaky = random_kfold_scores(build_pipeline(best, feature_cols, visual_mode, args.seed), best, X, y, 5, args.seed)
    pd.DataFrame([{"model": best, **leaky, "WARNING": "LEAKY: random folds ignore states; not a valid accuracy estimate"}]
                 ).round(4).to_csv(os.path.join(args.models_dir, "leaky_random_kfold_results.csv"), index=False)
    print(f"\nLEAKY random 5-fold (illustration only): R2 {leaky['r2']:+.3f}, MAE {leaky['mae']:.4f}")

    # ---- 5. ablation ----
    group_imp = None
    if not args.skip_ablation:
        print(f"\n=== Ablation with {best} (leave-one-state-out) ===")
        all_cols = list(feature_cols)
        vis = visual_columns(all_cols)
        tab = [c for c in all_cols if c not in vis]
        variants = {"all features": all_cols, "tabular only": tab}
        if vis:
            variants["visual only"] = vis
        drops = {}
        for g in FEATURE_GROUPS:
            cols = group_columns(g, all_cols)
            if cols and len(cols) < len(all_cols):
                variants[f"without {g}"] = [c for c in all_cols if c not in cols]
                drops[g] = f"without {g}"
        rows = {}
        for label, cols in variants.items():
            vm = resolve_visual_mode(args.visual, cols) if visual_columns(cols) else "none"
            pipe = build_pipeline(best, cols, vm, args.seed)
            pr = loso_predictions(pipe, best, df[cols], y, groups, "none", 0, args.seed)
            rows[label] = summarize(pr)
            print(f"  {label:<22} R2 {rows[label]['r2']:+.3f}  MAE {rows[label]['mae']:.4f}")
        abl = pd.DataFrame(rows).T
        abl.index.name = "feature_set"
        abl["n_features"] = [len(variants[k]) for k in abl.index]
        abl.round(4).to_csv(os.path.join(args.models_dir, "ablation_results.csv"))
        full_r2 = abl.loc["all features", "r2"]
        group_imp = pd.DataFrame({g: {"r2_without_group": abl.loc[lbl, "r2"],
                                      "r2_drop_when_removed": full_r2 - abl.loc[lbl, "r2"]}
                                  for g, lbl in drops.items()}).T.round(4)
        group_imp.index.name = "feature_group"
        group_imp.to_csv(os.path.join(args.models_dir, "feature_group_importance.csv"))
        if "visual only" in abl.index and abl.loc["tabular only", "r2"] > abl.loc["all features", "r2"] + 0.01:
            print("\nHINT: tabular-only beats all-features, so the visual features hurt on unseen states. "
                  "Re-run with --visual none and report the visual features as a negative finding.")

    _plots(args.models_dir, results, group_imp)

    # ---- 6. final model, uncertainty, thresholds, artefacts ----
    print(f"\n=== Fitting final {best} on all {len(df)} districts ===")
    final = _final_fit(best, X, y, groups, feature_cols, visual_mode, args.seed, args.tuning, args.n_iter)
    best_pred = loso_pred[best]
    interval = _interval_from_residuals(best_pred)

    pdf = best_pred.sort_values("row")
    out = pd.DataFrame({"area_id": df["area_id"].values[pdf["row"]] if "area_id" in df.columns else pdf["row"].values,
                        "state_name": groups[pdf["row"].values], "y_true": pdf["y_true"].values,
                        "y_pred_out_of_state": pdf["y_pred"].values})
    out["low_80"] = np.clip(out["y_pred_out_of_state"] + interval["80"]["lower"], 0, 1)
    out["high_80"] = np.clip(out["y_pred_out_of_state"] + interval["80"]["upper"], 0, 1)
    out.round(4).to_csv(os.path.join(args.models_dir, "loso_predictions.csv"), index=False)
    per_state_table(best_pred, groups).to_csv(os.path.join(args.models_dir, "per_state_errors.csv"))
    covered = float(((out["y_true"] >= out["low_80"]) & (out["y_true"] <= out["high_80"])).mean())
    print(f"80% interval empirical coverage on unseen states: {covered:.1%}")

    tertiles = y.quantile([1 / 3, 2 / 3]).tolist()
    thresholds = {"low_medium_boundary": tertiles[0], "medium_high_boundary": tertiles[1],
                  "derived_from": "deprivation_score tertiles of all rows used to fit the final model",
                  "note": "Low: score <= low_medium_boundary; Medium: between; High: score > medium_high_boundary. "
                          "A display simplification of the continuous score, not a separate target."}

    bundle = {"model": final, "model_name": best, "feature_columns": feature_cols, "visual_mode": visual_mode,
              "interval": interval, "trained_on": data_mode, "estimator": final.named_steps["model"],
              "preprocessor": final.named_steps["pre"], "metrics_loso": results.loc[best, ["r2", "mae", "rmse"]].to_dict()}
    joblib.dump(bundle, os.path.join(args.models_dir, "best_model.pkl"))
    for fname, obj in [("feature_columns.json", feature_cols), ("category_thresholds.json", thresholds),
                       ("uncertainty.json", interval)]:
        with open(os.path.join(args.models_dir, fname), "w") as f:
            json.dump(obj, f, indent=2)
    meta = {"data_file": os.path.basename(args.data), "data_sha256_16": _file_sha(args.data), "n_rows": len(df),
            "n_states": int(len(set(groups))), "n_features": len(feature_cols), "visual_mode": visual_mode,
            "seed": args.seed, "tuning": args.tuning, "best_model": best, "data_mode": data_mode,
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(args.models_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if not args.no_db:
        log_results_to_db(results[["mae", "rmse", "r2"]], trained_on=data_mode)

    print(f"\nSaved artefacts to {args.models_dir}")
    if data_mode == "demo":
        print("REMINDER: these metrics come from SYNTHETIC DEMO data. Do not report them as real findings.")
    if results.loc[best, "r2"] < 0:
        print("NOTE: negative R2 on unseen states means the model is worse than the mean for some states.")


if __name__ == "__main__":
    main()
