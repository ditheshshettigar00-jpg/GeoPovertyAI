"""
geo_validation.py
--------------------
Geography-aware train/test split and cross-validation utilities.

WHY THIS EXISTS: a naive random train/test split can put two neighboring
(highly similar) districts from the same state into training and test
sets respectively. Because nearby districts tend to share similar
imagery, nightlight levels, and socioeconomic conditions (spatial
autocorrelation), this leaks information and makes test performance look
better than the model would actually achieve on a genuinely unseen
region. Instead, we split by STATE, so entire states are held out for
testing/validation -- the model never sees any district from a test-set
state during training.
"""

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def grouped_train_test_split(X, y, groups, test_size=0.25, random_state=42):
    """
    Splits (X, y) into train/test sets by holding out entire groups
    (states), not individual rows.

    Returns: X_train, X_test, y_train, y_test, groups_train, groups_test
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = np.array(groups)[train_idx]
    groups_test = np.array(groups)[test_idx]

    n_train_groups = len(set(groups_train))
    n_test_groups = len(set(groups_test))
    print(f"Geography-aware split: {n_train_groups} state(s) in train "
          f"({len(X_train)} rows), {n_test_groups} state(s) in test ({len(X_test)} rows)")
    print(f"  Test states: {sorted(set(groups_test))}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def get_group_kfold(n_groups_available: int, max_splits: int = 5):
    """
    Returns a GroupKFold with a safe number of splits given how many
    distinct groups (states) are actually available in the training set.
    Falls back to fewer splits if there aren't many states -- this is a
    real constraint of geography-aware validation with a limited number
    of states, and should be reported honestly rather than silently
    replaced with a random (non-grouped) KFold.
    """
    n_splits = min(max_splits, n_groups_available)
    if n_splits < 2:
        raise ValueError(
            f"Only {n_groups_available} distinct state(s) available in the training "
            "set -- geography-aware cross-validation needs at least 2. "
            "Add more states to your study region, or use a simple held-out "
            "validation set instead of k-fold CV for hyperparameter tuning."
        )
    if n_splits < max_splits:
        print(f"NOTE: only {n_groups_available} states available for CV -- "
              f"using {n_splits}-fold GroupKFold instead of the default {max_splits}-fold.")
    return GroupKFold(n_splits=n_splits)
