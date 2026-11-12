# Empirical Results & Benchmark Tables

### Model Evaluation and Benchmark Results

> **Dataset Scope:** Evaluated on **564 districts** across **31 states** using **10 tabular satellite/geospatial features**. Best-performing model: **XGBoost**.

#### 1. Leave-One-State-Out (LOSO) Validation Results
*Primary benchmark: each prediction is made for an entire administrative state the model never encountered during training.*

| Model | R² | MAE | RMSE | Spearman ρ | Cat. Accuracy | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy | -0.0565 | 0.1142 | 0.1371 | -0.6972 | 31.9% | 0.9% |
| Ridge | 0.3742 | 0.0846 | 0.1056 | 0.6372 | 57.5% | 51.3% |
| RandomForest | 0.4217 | 0.0817 | 0.1015 | 0.6716 | 58.0% | 45.1% |
| GradientBoosting | 0.4665 | 0.0771 | 0.0975 | 0.6973 | 60.8% | 47.8% |
| XGBoost | 0.4884 | 0.0752 | 0.0954 | 0.7150 | 61.2% | 47.8% |

#### 2. Repeated State-Grouped 5-Fold Cross-Validation
*Secondary benchmark: mean and standard deviation over repeated 5-fold cross-validation partitioned by administrative state.*

| Model | R² (mean ± std) | MAE (mean ± std) | RMSE (mean ± std) | Spearman ρ | Cat. Accuracy | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy | -0.0529 ± 0.0359 | 0.1139 ± 0.0019 | 0.1369 ± 0.0023 | -0.2812 | 32.3% | 17.5% |
| Ridge | 0.3559 ± 0.0165 | 0.0857 ± 0.0011 | 0.1071 ± 0.0014 | 0.6219 | 55.7% | 48.7% |
| RandomForest | 0.3543 ± 0.0696 | 0.0860 ± 0.0053 | 0.1071 ± 0.0057 | 0.6181 | 55.2% | 40.9% |
| GradientBoosting | 0.3805 ± 0.0951 | 0.0826 ± 0.0059 | 0.1047 ± 0.0078 | 0.6413 | 57.5% | 43.0% |
| XGBoost | 0.3888 ± 0.1048 | 0.0815 ± 0.0066 | 0.1040 ± 0.0086 | 0.6490 | 57.7% | 44.2% |

#### 3. Baseline Comparison (Predicting Training Mean)
*Comparison against the Dummy regressor that always predicts the training mean of other states.*

| Model | MAE | MAE vs Baseline (%) | Beats Baseline? |
| --- | --- | --- | --- |
| Dummy | 0.1142 | +0.0% | No |
| Ridge | 0.0846 | -25.9% | Yes |
| RandomForest | 0.0817 | -28.4% | Yes |
| GradientBoosting | 0.0771 | -32.4% | Yes |
| XGBoost | 0.0752 | -34.2% | Yes |

#### 4. Feature Ablation Study
*Systematic feature omission on leave-one-state-out evaluation to determine individual indicator contribution.*

| Feature Set | Features | R² | MAE | RMSE | Spearman ρ | Top-20% Recall |
| --- | --- | --- | --- | --- | --- | --- |
| all features | 10 | 0.4884 | 0.0752 | 0.0954 | 0.7150 | 47.8% |
| tabular only | 10 | 0.4884 | 0.0752 | 0.0954 | 0.7150 | 47.8% |
| without nightlight | 9 | 0.4674 | 0.0762 | 0.0974 | 0.7001 | 45.1% |
| without vegetation | 9 | 0.4557 | 0.0779 | 0.0984 | 0.6998 | 50.4% |
| without built_up | 9 | 0.4189 | 0.0804 | 0.1017 | 0.6634 | 47.8% |
| without population | 9 | 0.4943 | 0.0747 | 0.0949 | 0.7184 | 49.6% |
| without elevation | 9 | 0.4031 | 0.0810 | 0.1031 | 0.6523 | 47.8% |
| without slope | 9 | 0.4906 | 0.0759 | 0.0952 | 0.7190 | 46.9% |
| without rainfall | 9 | 0.5044 | 0.0750 | 0.0939 | 0.7220 | 50.4% |
| without land_surface_temp | 9 | 0.3452 | 0.0869 | 0.1080 | 0.5987 | 47.8% |
| without cropland | 9 | 0.4926 | 0.0751 | 0.0950 | 0.7205 | 50.4% |
| without travel_time | 9 | 0.4903 | 0.0752 | 0.0953 | 0.7165 | 48.7% |

#### 5. Five States with Highest Prediction Error
*States where leave-one-state-out predictions deviate most from survey ground truth (ranked by MAE descending).*

| State Name | Districts (n) | MAE | Bias (Mean Error) | RMSE |
| --- | --- | --- | --- | --- |
| Meghalaya | 6 | 0.1632 | -0.1632 | 0.1829 |
| Tripura | 4 | 0.1306 | -0.1306 | 0.1494 |
| Andhra Pradesh | 13 | 0.1255 | +0.1225 | 0.1414 |
| Mizoram | 8 | 0.1253 | +0.1190 | 0.1396 |
| Jharkhand | 22 | 0.1197 | -0.1059 | 0.1374 |

#### 6. Spatial Leakage: Leaky Random Splits vs. Honest State Holdout
*Demonstration of spatial autocorrelation (Tobler's First Law): random district splits inflate R² by **+0.1722** (+35.3%), producing overly optimistic performance.*

| Validation Scheme | Model | R² | MAE | RMSE | Spearman ρ | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Leave-One-State-Out (Honest) | XGBoost | 0.4884 | 0.0752 | 0.0954 | 0.7150 | Strict state holdout; model never saw test states |
| Random 5-Fold CV (Leaky) | XGBoost | 0.6606 | 0.0610 | 0.0777 | 0.8219 | Random folds ignore spatial autocorrelation (Tobler's First Law) |
