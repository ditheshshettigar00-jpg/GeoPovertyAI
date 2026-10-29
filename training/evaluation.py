"""
evaluation.py
-------------
State-held-out evaluation helpers. All metrics are computed on predictions made
for states the model was not trained on.

Reported metrics:
    r2, mae, rmse       standard regression scores
    spearman            rank correlation (does the model order districts correctly?)
    cat_accuracy        Low/Medium/High accuracy, with tertile cut-offs fitted on each
                        fold's TRAINING labels only
    top20_recall        share of the truly most-deprived 20% of districts that land in the
                        predicted top 20% (random guessing = 0.20)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from training.geo_validation import get_group_kfold, leave_one_state_out, repeated_group_kfold
from training.pipelines import PARAM_GRIDS


def _fit(pipe, model_name, X_tr, y_tr, g_tr, tuning, n_iter, seed):
    """Fit one pipeline; with tuning='nested' the hyperparameters are chosen by a state-grouped
    search using ONLY this training fold, so the held-out state never influences them."""
    if tuning == "nested" and model_name in PARAM_GRIDS:
        n_groups = len(set(g_tr))
        if n_groups >= 3:
            cv = list(get_group_kfold(n_groups, max_splits=4).split(X_tr, y_tr, groups=g_tr))
            search = RandomizedSearchCV(clone(pipe), PARAM_GRIDS[model_name], n_iter=n_iter, cv=cv,
                                        scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=1)
            search.fit(X_tr, y_tr)
            return search.best_estimator_
    est = clone(pipe)
    est.fit(X_tr, y_tr)
    return est


def collect_predictions(pipe, model_name, X, y, groups, splits, tuning="none", n_iter=6, seed=42):
    """splits: iterable of (label, train_idx, test_idx). Returns one row per held-out prediction."""
    rows = []
    groups = np.asarray(groups)
    for label, tr, te in splits:
        est = _fit(pipe, model_name, X.iloc[tr], y.iloc[tr], groups[tr], tuning, n_iter, seed)
        pred = est.predict(X.iloc[te])
        t1, t2 = np.quantile(y.iloc[tr], [1 / 3, 2 / 3])
        rows.append(pd.DataFrame({"row": te, "fold": label, "y_true": y.iloc[te].values,
                                  "y_pred": np.asarray(pred, dtype=float), "t1": t1, "t2": t2}))
    return pd.concat(rows, ignore_index=True)


def loso_predictions(pipe, model_name, X, y, groups, tuning="none", n_iter=6, seed=42):
    return collect_predictions(pipe, model_name, X, y, groups,
                               ((s, tr, te) for s, tr, te in leave_one_state_out(groups)),
                               tuning, n_iter, seed)


def summarize(pred_df):
    yt, yp = pred_df["y_true"].values, pred_df["y_pred"].values
    cat_t = (yt > pred_df["t1"].values).astype(int) + (yt > pred_df["t2"].values).astype(int)
    cat_p = (yp > pred_df["t1"].values).astype(int) + (yp > pred_df["t2"].values).astype(int)
    top_true = yt >= np.quantile(yt, 0.8)
    top_pred = yp >= np.quantile(yp, 0.8)
    rho = spearmanr(yt, yp).correlation if np.std(yp) > 0 else float("nan")
    return {
        "r2": r2_score(yt, yp),
        "mae": mean_absolute_error(yt, yp),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "spearman": rho,
        "cat_accuracy": float((cat_t == cat_p).mean()),
        "top20_recall": float((top_true & top_pred).sum() / max(top_true.sum(), 1)),
        "n": int(len(yt)),
    }


def per_state_table(pred_df, groups):
    groups = np.asarray(groups)
    d = pred_df.copy()
    d["state_name"] = groups[d["row"].values]
    d["err"] = d["y_pred"] - d["y_true"]
    out = d.groupby("state_name").agg(n_districts=("err", "size"), mae=("err", lambda e: float(np.mean(np.abs(e)))),
                                      bias=("err", "mean"),
                                      rmse=("err", lambda e: float(np.sqrt(np.mean(np.square(e)))))).round(4)
    return out.sort_values("mae")


def repeated_group_scores(pipe, model_name, X, y, groups, n_splits=5, n_repeats=5, tuning="none", n_iter=6, seed=42):
    """Pooled metrics per repeat, then mean and std across repeats."""
    groups = np.asarray(groups)
    by_rep = {}
    for rep, fold, tr, te in repeated_group_kfold(groups, n_splits, n_repeats, seed):
        by_rep.setdefault(rep, []).append((f"r{rep}f{fold}", tr, te))
    scores = [summarize(collect_predictions(pipe, model_name, X, y, groups, sp, tuning, n_iter, seed))
              for sp in by_rep.values()]
    df = pd.DataFrame(scores)
    return {**{f"{c}_mean": df[c].mean() for c in df.columns if c != "n"},
            **{f"{c}_std": df[c].std(ddof=0) for c in ("r2", "mae", "rmse")}}


def random_kfold_scores(pipe, model_name, X, y, n_splits=5, seed=42):
    """LEAKY on purpose: ignores states. Only used to show how much leakage inflates scores."""
    from sklearn.model_selection import KFold
    splits = ((f"k{i}", tr, te) for i, (tr, te) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(X)))
    groups = np.zeros(len(X))
    return summarize(collect_predictions(pipe, model_name, X, y, groups, splits, "none", 0, seed))
