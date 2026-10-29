# Phase 2: modelling and evaluation

## Apply
1. Copy `training/`, `tests/test_phase2.py` and `explainability/shap_explain.py` into your project (same paths).
   `training/geo_validation.py` keeps the old functions and adds two new ones; `train_models.py` keeps
   `get_feature_columns`, so the old tests still import.
2. Patch `backend/services/model_service.py` (one place, inside `predict_custom_features`). Replace

       import shap
       explainer = shap.TreeExplainer(_model_bundle["model"])

   with

       from explainability.shap_explain import build_explainer
       background = _feature_table[_feature_cols] if _feature_table is not None else None
       explainer = build_explainer(_model_bundle["model"], background)

   The saved model is now a sklearn Pipeline, and SHAP cannot take a Pipeline directly.
3. Optional: `pip install xgboost` (it is skipped automatically when missing).

## Run (after Phase 1 has produced the new final_training_table.csv)

    python scripts/audit_data.py --strict
    python training/train_models.py --data data/processed/final_training_table.csv
    pytest tests/test_phase2.py -v

Slower and more rigorous (hyperparameters tuned inside every training fold):

    python training/train_models.py --data data/processed/final_training_table.csv --tuning nested --n-iter 6

Quick smoke run: add `--fast`. If the ablation says tabular-only beats all features, re-run with `--visual none`.

## What changed
- Main result is leave-one-state-out: every prediction is for a state the model has not seen.
  Second view: repeated state-grouped 5-fold CV (mean +- std).
- Clipping (3 x IQR), imputation and scaling/PCA are inside the Pipeline, so they are fitted on training folds only.
  Imputation uses the training-fold median. (A per-state median is impossible for an unseen state.)
- The old Ridge/linear baseline was unscaled, which is why it showed R2 -0.78. It is now scaled.
- Baseline "Dummy" (predict the training mean) is evaluated the same way; the final model must beat it.
- Ablation + feature-group importance, a LEAKY random-fold run kept in its own file, per-state errors,
  80%/90% prediction intervals from out-of-state residuals, run metadata with seed and data hash.
- Tertile thresholds for the Low/Medium/High display are fitted on the final training labels; the
  category accuracy in the evaluation uses cut-offs fitted on each fold's training labels.

## Files written to models/
model_comparison_results.csv, repeated_groupkfold_results.csv, per_state_errors.csv, loso_predictions.csv,
ablation_results.csv, feature_group_importance.csv, leaky_random_kfold_results.csv, uncertainty.json,
run_metadata.json, best_model.pkl, feature_columns.json, category_thresholds.json, figures/*.png

`best_model.pkl` is a dict: `model` (Pipeline, works with `.predict(DataFrame)`), `model_name`, `feature_columns`,
`visual_mode`, `interval`, `estimator` and `preprocessor` (for SHAP in Phase 3), `metrics_loso`.
For prediction intervals use `training/inference.py: predict_with_interval(bundle, X)` (Phase 4 hooks it into the API).

## Sample results (sample_results_OLD_table/)
These were produced on the table from the ORIGINAL zip, before Phase 1 fixed the units. They show the output
format only. Re-run after Phase 1 and report those numbers instead.

Leave-one-state-out, 532 districts, 29 states:

| Model | R2 | MAE | 3-class acc | top-20% recall |
|---|---|---|---|---|
| Predict the mean | -0.06 | 0.115 | 32% | 0% |
| Ridge | 0.27 | 0.091 | 54% | 37% |
| Random Forest | 0.29 | 0.092 | 52% | 45% |
| Gradient Boosting | 0.25 | 0.094 | 51% | 43% |

Ablation (Random Forest): all features R2 0.288; tabular only 0.327; visual only 0.070. Removing the VGG
features improved the score, so on this table they hurt on unseen states. LEAKY random 5-fold gave R2 0.448.
(The Dummy row's negative Spearman is an artefact: leaving a state out shifts the training mean the opposite way.)

## Tested here / not tested here
Tested: all 10 tests in tests/test_phase2.py, an end-to-end run on the old table, the existing backend service
(`predict_scenario`, `predict_custom_features`) against the new model file, and the SHAP wrapper plumbing with
a stub (real `shap` is not installed in my environment). Not tested: XGBoost, `--tuning nested` on the full table
(only on small data), the Flask routes and real SHAP output. Run the old test suite on your side.
