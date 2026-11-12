"""
evaluate_extra_features.py
--------------------------
EXPERIMENTAL. Tests whether extra features improve the HONEST accuracy of the production model,
using exactly the protocol behind the published numbers (training/evaluation.py, same pipeline,
leave-one-state-out plus repeated state-grouped 5-fold). It writes only to
models/extra_feature_experiments/ and never touches best_model.pkl or the published CSVs.

Decision rule, fixed BEFORE looking at any result (so we do not pick whatever looks best):
  Primary comparison: baseline 4 features  vs  baseline + ALL extra features.
  ADOPT only if ALL three hold:
    1. LOSO R2 gain                          >= --min-gain   (default 0.02)
    2. repeated state-grouped CV R2 gain      >= --min-gain   (0.02 is about one CV std)
    3. share of states whose MAE improved    >= --min-states (default 0.55)
  The per-feature rows ("+elevation_mean_m" ...) are exploratory: they show what drives a gain
  but must not be used to cherry-pick a subset after the fact.

Usage:
    python training/evaluate_extra_features.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED
from training.evaluation import loso_predictions, repeated_group_scores, summarize, per_state_table
from training.pipelines import build_pipeline

OUT_DIR = os.path.join(MODELS_DIR, "extra_feature_experiments")


def evaluate_set(df, cols, model, repeats, seed):
    X, y, groups = df[cols], df["deprivation_score"], df["state_name"].values
    pipe = build_pipeline(model, cols, "none", seed)
    pred = loso_predictions(pipe, model, X, y, groups, "none", 0, seed)
    s = summarize(pred)
    rep = repeated_group_scores(pipe, model, X, y, groups, 5, repeats, "none", 0, seed)
    return s, rep, per_state_table(pred, groups)["mae"]


def run(df, base_cols, extra_cols, model="RandomForest", repeats=5, seed=RANDOM_SEED, per_feature=True):
    candidates = {"baseline (4 features)": base_cols}
    if len(extra_cols) > 1 or not per_feature:
        candidates["baseline + ALL extra"] = base_cols + extra_cols
    if per_feature:
        for f in extra_cols:
            candidates[f"baseline + {f}"] = base_cols + [f]
    rows, state_mae = [], {}
    for name, cols in candidates.items():
        s, rep, mae = evaluate_set(df, cols, model, repeats, seed)
        state_mae[name] = mae
        rows.append({"feature_set": name, "n_features": len(cols), "loso_r2": s["r2"], "loso_mae": s["mae"],
                     "loso_spearman": s["spearman"], "loso_top20_recall": s["top20_recall"],
                     "repeated_r2_mean": rep["r2_mean"], "repeated_r2_std": rep["r2_std"]})
    res = pd.DataFrame(rows)
    b = res.iloc[0]
    res["d_loso_r2"] = res["loso_r2"] - b["loso_r2"]
    res["d_repeated_r2"] = res["repeated_r2_mean"] - b["repeated_r2_mean"]
    base_mae = state_mae["baseline (4 features)"]
    res["states_improved"] = [float((state_mae[n].reindex(base_mae.index) < base_mae).mean())
                              for n in res["feature_set"]]
    return res


def verdict(res, min_gain, min_states, primary="baseline + ALL extra"):
    r = res[res["feature_set"] == primary]
    if r.empty:
        r = res.iloc[[1]]
    r = r.iloc[0]
    checks = {f"LOSO R2 gain {r.d_loso_r2:+.4f} >= {min_gain}": r.d_loso_r2 >= min_gain,
              f"repeated-CV R2 gain {r.d_repeated_r2:+.4f} >= {min_gain}": r.d_repeated_r2 >= min_gain,
              f"states improved {r.states_improved:.0%} >= {min_states:.0%}": r.states_improved >= min_states}
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser(description="Test extra features with the project's honest validation")
    ap.add_argument("--data", default=os.path.join(PROCESSED_DIR, "final_training_table.csv"))
    ap.add_argument("--extra", default=os.path.join(PROCESSED_DIR, "extra_geospatial_features.csv"))
    ap.add_argument("--model", default="RandomForest")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--min-states", type=float, default=0.55)
    ap.add_argument("--max-missing", type=float, default=0.20, help="skip features missing in more than this share")
    args = ap.parse_args()

    base_cols = json.load(open(os.path.join(MODELS_DIR, "feature_columns.json")))
    train = pd.read_csv(args.data).dropna(subset=["deprivation_score"])
    extra = pd.read_csv(args.extra)
    extra_cols = [c for c in extra.columns if c not in ("area_id", "area_name", "gaul_state_name")]
    df = train.merge(extra[["area_id"] + extra_cols], on="area_id", how="left").reset_index(drop=True)

    miss = df[extra_cols].isna().mean()
    print("Share missing per extra feature:\n" + miss.round(3).to_string())
    drop = [c for c in extra_cols if miss[c] > args.max_missing or df[c].nunique(dropna=True) < 2]
    if drop:
        print(f"\nSkipping (too many missing or constant): {drop}")
    extra_cols = [c for c in extra_cols if c not in drop]
    if not extra_cols:
        sys.exit("No usable extra features.")
    print(f"\n{len(df)} districts, {df['state_name'].nunique()} states; testing {extra_cols}\n")

    res = run(df, base_cols, extra_cols, args.model, args.repeats, args.seed)

    pub = os.path.join(MODELS_DIR, "model_comparison_results.csv")
    if os.path.exists(pub) and args.model in pd.read_csv(pub, index_col=0).index:
        published = pd.read_csv(pub, index_col=0).loc[args.model, "r2"]
        ok = abs(res.loc[0, "loso_r2"] - published) < 5e-4
        print(f"Sanity check: baseline LOSO R2 {res.loc[0, 'loso_r2']:.4f} vs published {published:.4f} -> "
              f"{'OK, protocol reproduced' if ok else 'MISMATCH (check data/seed before trusting results)'}\n")

    show = res.round(4).to_string(index=False)
    print(show)
    adopt, checks = verdict(res, args.min_gain, args.min_states)
    print("\nPrimary test (baseline vs baseline + ALL extra):")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\nVERDICT:", "ADOPT the extra features (then retrain, regenerate docs, re-run tests)."
          if adopt else "DO NOT ADOPT. Keep the 4-feature model; report this as a negative result.")

    os.makedirs(OUT_DIR, exist_ok=True)
    res.round(4).to_csv(os.path.join(OUT_DIR, "extra_features_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "extra_features_summary.md"), "w") as f:
        f.write("# Extra-feature experiment (experimental, not production)\n\n"
                f"Model: {args.model}. Leave-one-state-out plus repeated state-grouped 5-fold ({args.repeats} repeats).\n\n"
                f"Rule fixed in advance: adopt only if LOSO and repeated-CV R2 each gain >= {args.min_gain} "
                f"and >= {args.min_states:.0%} of states improve.\n\n```\n{show}\n```\n\n"
                f"**Verdict:** {'ADOPT' if adopt else 'DO NOT ADOPT'}\n")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
