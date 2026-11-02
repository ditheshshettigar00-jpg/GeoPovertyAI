# Empirical Results & Benchmark Tables

### Model Evaluation and Benchmark Results

> **Dataset Scope:** Evaluated on **564 districts** across **31 states** using **4 tabular satellite/geospatial features**. Best-performing model: **RandomForest**.

#### 1. Leave-One-State-Out (LOSO) Validation Results
*Primary benchmark: each prediction is made for an entire administrative state the model never encountered during training.*

| Model | R² | MAE | RMSE | Spearman ρ | Cat. Accuracy | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy | -0.0565 | 0.1142 | 0.1371 | -0.6972 | 31.9% | 0.9% |
| Ridge | 0.2700 | 0.0914 | 0.1140 | 0.5542 | 48.6% | 32.7% |
| RandomForest | 0.3360 | 0.0883 | 0.1087 | 0.5745 | 54.6% | 46.0% |
| GradientBoosting | 0.2951 | 0.0912 | 0.1120 | 0.5539 | 52.7% | 45.1% |
| XGBoost | 0.3074 | 0.0905 | 0.1110 | 0.5643 | 52.7% | 46.9% |

#### 2. Repeated State-Grouped 5-Fold Cross-Validation
*Secondary benchmark: mean and standard deviation over repeated 5-fold cross-validation partitioned by administrative state.*

| Model | R² (mean ± std) | MAE (mean ± std) | RMSE (mean ± std) | Spearman ρ | Cat. Accuracy | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy | -0.0529 ± 0.0359 | 0.1139 ± 0.0019 | 0.1369 ± 0.0023 | -0.2812 | 32.3% | 17.5% |
| Ridge | 0.2500 ± 0.0320 | 0.0930 ± 0.0020 | 0.1155 ± 0.0025 | 0.5268 | 47.6% | 30.6% |
| RandomForest | 0.2984 ± 0.0216 | 0.0910 ± 0.0011 | 0.1117 ± 0.0017 | 0.5450 | 52.0% | 42.5% |
| GradientBoosting | 0.2662 ± 0.0217 | 0.0930 ± 0.0013 | 0.1143 ± 0.0017 | 0.5337 | 51.1% | 40.9% |
| XGBoost | 0.2806 ± 0.0197 | 0.0923 ± 0.0012 | 0.1132 ± 0.0015 | 0.5461 | 52.1% | 41.9% |

#### 3. Baseline Comparison (Predicting Training Mean)
*Comparison against the Dummy regressor that always predicts the training mean of other states.*

| Model | MAE | MAE vs Baseline (%) | Beats Baseline? |
| --- | --- | --- | --- |
| Dummy | 0.1142 | +0.0% | No |
| Ridge | 0.0914 | -20.0% | Yes |
| RandomForest | 0.0883 | -22.7% | Yes |
| GradientBoosting | 0.0912 | -20.1% | Yes |
| XGBoost | 0.0905 | -20.7% | Yes |

#### 4. Feature Ablation Study
*Systematic feature omission on leave-one-state-out evaluation to determine individual indicator contribution.*

| Feature Set | Features | R² | MAE | RMSE | Spearman ρ | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| all features | 4 | 0.3360 | 0.0883 | 0.1087 | 0.5745 | 46.0% |
| tabular only | 4 | 0.3360 | 0.0883 | 0.1087 | 0.5745 | 46.0% |
| without nightlight | 3 | 0.2894 | 0.0911 | 0.1125 | 0.5430 | 38.0% |
| without vegetation | 3 | 0.1567 | 0.0988 | 0.1225 | 0.4032 | 37.2% |
| without built_up | 3 | 0.2529 | 0.0939 | 0.1153 | 0.4978 | 39.8% |
| without population | 3 | 0.1854 | 0.0976 | 0.1204 | 0.4188 | 28.3% |

#### 5. Five States with Highest Prediction Error
*States where leave-one-state-out predictions deviate most from survey ground truth (ranked by MAE descending).*

| State Name | Districts (n) | MAE | Bias (Mean Error) | RMSE |
| --- | --- | --- | --- | --- |
| Goa | 2 | 0.1769 | +0.1769 | 0.1771 |
| Tripura | 4 | 0.1608 | -0.1608 | 0.1719 |
| Meghalaya | 6 | 0.1517 | -0.1517 | 0.1724 |
| Jharkhand | 22 | 0.1464 | -0.1389 | 0.1680 |
| Mizoram | 8 | 0.1280 | +0.1142 | 0.1374 |

#### 6. Spatial Leakage: Leaky Random Splits vs. Honest State Holdout
*Demonstration of spatial autocorrelation (Tobler's First Law): random district splits inflate R² by **+0.1268** (+37.7%), producing overly optimistic performance.*

| Validation Scheme | Model | R² | MAE | RMSE | Spearman ρ | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Leave-One-State-Out (Honest) | RandomForest | 0.3360 | 0.0883 | 0.1087 | 0.5745 | Strict state holdout; model never saw test states |
| Random 5-Fold CV (Leaky) | RandomForest | 0.4628 | 0.0779 | 0.0978 | 0.6694 | Random folds ignore spatial autocorrelation (Tobler's First Law) |
