"""
generate_results_doc.py
-----------------------
Generates documentation/RESULTS.md and updates the results section in README.md
between <!-- RESULTS START --> and <!-- RESULTS END --> markers.

NEVER hardcodes metrics: all numbers are read dynamically from:
  - models/model_comparison_results.csv
  - models/repeated_groupkfold_results.csv
  - models/ablation_results.csv
  - models/per_state_errors.csv
  - models/leaky_random_kfold_results.csv
  - models/run_metadata.json
"""

import json
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DOCS_DIR = os.path.join(ROOT, "documentation")
README_PATH = os.path.join(ROOT, "README.md")


def format_table(df: pd.DataFrame, headers: list = None, col_formats: dict = None) -> str:
    """Formats a pandas DataFrame as a clean Markdown table with custom formatting."""
    df_copy = df.copy()
    if col_formats:
        for col, fmt in col_formats.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(lambda x: fmt.format(x) if pd.notna(x) and isinstance(x, (int, float)) else str(x))

    if headers:
        df_copy.columns = headers

    cols = list(df_copy.columns)
    header_line = "| " + " | ".join(cols) + " |"
    separator_line = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df_copy.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        rows.append(row_str)

    return "\n".join([header_line, separator_line] + rows)


def generate_results_markdown() -> str:
    # 1. Load run metadata
    meta_path = os.path.join(MODELS_DIR, "run_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_districts = meta.get("n_rows", 564)
    n_states = meta.get("n_states", 31)
    best_model = meta.get("best_model", "RandomForest")
    n_features = meta.get("n_features", 4)

    # 2. Load CSV files
    loso_df = pd.read_csv(os.path.join(MODELS_DIR, "model_comparison_results.csv"))
    repeated_df = pd.read_csv(os.path.join(MODELS_DIR, "repeated_groupkfold_results.csv"))
    ablation_df = pd.read_csv(os.path.join(MODELS_DIR, "ablation_results.csv"))
    per_state_df = pd.read_csv(os.path.join(MODELS_DIR, "per_state_errors.csv"))
    leaky_df = pd.read_csv(os.path.join(MODELS_DIR, "leaky_random_kfold_results.csv"))

    # Section 1: LOSO Comparison Table
    loso_disp = loso_df[["model", "r2", "mae", "rmse", "spearman", "cat_accuracy", "top20_recall"]].copy()
    loso_table = format_table(
        loso_disp,
        headers=["Model", "R²", "MAE", "RMSE", "Spearman ρ", "Cat. Accuracy", "Top-20% Recall"],
        col_formats={"r2": "{:.4f}", "mae": "{:.4f}", "rmse": "{:.4f}", "spearman": "{:.4f}",
                     "cat_accuracy": "{:.1%}", "top20_recall": "{:.1%}"}
    )

    # Section 2: Repeated 5-Fold Grouped CV Table
    rep_disp = pd.DataFrame({
        "Model": repeated_df["model"],
        "R² (mean ± std)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(repeated_df["r2_mean"], repeated_df["r2_std"])],
        "MAE (mean ± std)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(repeated_df["mae_mean"], repeated_df["mae_std"])],
        "RMSE (mean ± std)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(repeated_df["rmse_mean"], repeated_df["rmse_std"])],
        "Spearman ρ": repeated_df["spearman_mean"].apply(lambda x: f"{x:.4f}"),
        "Cat. Accuracy": repeated_df["cat_accuracy_mean"].apply(lambda x: f"{x:.1%}"),
        "Top-20% Recall": repeated_df["top20_recall_mean"].apply(lambda x: f"{x:.1%}"),
    })
    repeated_table = format_table(rep_disp)

    # Section 3: Baseline Comparison Table
    dummy_row = loso_df[loso_df["model"] == "Dummy"].iloc[0]
    dummy_mae = dummy_row["mae"]
    base_disp = loso_df[["model", "mae", "mae_vs_baseline_pct", "beats_mean_baseline"]].copy()
    base_disp["beats_mean_baseline"] = base_disp["beats_mean_baseline"].apply(lambda x: "Yes" if x is True or str(x).lower() == "true" else "No")
    baseline_table = format_table(
        base_disp,
        headers=["Model", "MAE", "MAE vs Baseline (%)", "Beats Baseline?"],
        col_formats={"mae": "{:.4f}", "mae_vs_baseline_pct": "{:+.1f}%"}
    )

    # Section 4: Ablation Study Table
    abl_disp = ablation_df[["feature_set", "n_features", "r2", "mae", "rmse", "spearman", "top20_recall"]].copy()
    ablation_table = format_table(
        abl_disp,
        headers=["Feature Set", "Features", "R²", "MAE", "RMSE", "Spearman ρ", "Top-20% Recall"],
        col_formats={"n_features": "{:.0f}", "r2": "{:.4f}", "mae": "{:.4f}", "rmse": "{:.4f}",
                     "spearman": "{:.4f}", "top20_recall": "{:.1%}"}
    )

    # Section 5: Five Worst Performing States
    worst_5 = per_state_df.sort_values("mae", ascending=False).head(5).copy()
    worst_table = format_table(
        worst_5[["state_name", "n_districts", "mae", "bias", "rmse"]],
        headers=["State Name", "Districts (n)", "MAE", "Bias (Mean Error)", "RMSE"],
        col_formats={"n_districts": "{:d}", "mae": "{:.4f}", "bias": "{:+.4f}", "rmse": "{:.4f}"}
    )

    # Section 6: Leaky vs Honest Comparison
    rf_honest = loso_df[loso_df["model"] == best_model].iloc[0]
    rf_leaky = leaky_df.iloc[0]
    leaky_comp = pd.DataFrame([
        {
            "Validation Scheme": "Leave-One-State-Out (Honest)",
            "Model": best_model,
            "R²": f"{rf_honest['r2']:.4f}",
            "MAE": f"{rf_honest['mae']:.4f}",
            "RMSE": f"{rf_honest['rmse']:.4f}",
            "Spearman ρ": f"{rf_honest['spearman']:.4f}",
            "Note": "Strict state holdout; model never saw test states"
        },
        {
            "Validation Scheme": "Random 5-Fold CV (Leaky)",
            "Model": best_model,
            "R²": f"{rf_leaky['r2']:.4f}",
            "MAE": f"{rf_leaky['mae']:.4f}",
            "RMSE": f"{rf_leaky['rmse']:.4f}",
            "Spearman ρ": f"{rf_leaky['spearman']:.4f}",
            "Note": "Random folds ignore spatial autocorrelation (Tobler's First Law)"
        }
    ])
    leaky_table = format_table(leaky_comp)

    r2_diff = rf_leaky['r2'] - rf_honest['r2']
    r2_pct_inflation = (r2_diff / abs(rf_honest['r2'])) * 100

    md = f"""### Model Evaluation and Benchmark Results

> **Dataset Scope:** Evaluated on **{n_districts} districts** across **{n_states} states** using **{n_features} tabular satellite/geospatial features**. Best-performing model: **{best_model}**.

#### 1. Leave-One-State-Out (LOSO) Validation Results
*Primary benchmark: each prediction is made for an entire administrative state the model never encountered during training.*

{loso_table}

#### 2. Repeated State-Grouped 5-Fold Cross-Validation
*Secondary benchmark: mean and standard deviation over repeated 5-fold cross-validation partitioned by administrative state.*

{repeated_table}

#### 3. Baseline Comparison (Predicting Training Mean)
*Comparison against the Dummy regressor that always predicts the training mean of other states.*

{baseline_table}

#### 4. Feature Ablation Study
*Systematic feature omission on leave-one-state-out evaluation to determine individual indicator contribution.*

{ablation_table}

#### 5. Five States with Highest Prediction Error
*States where leave-one-state-out predictions deviate most from survey ground truth (ranked by MAE descending).*

{worst_table}

#### 6. Spatial Leakage: Leaky Random Splits vs. Honest State Holdout
*Demonstration of spatial autocorrelation (Tobler's First Law): random district splits inflate R² by **{r2_diff:+.4f}** ({r2_pct_inflation:+.1f}%), producing overly optimistic performance.*

{leaky_table}
"""
    return md


def update_results():
    results_md = generate_results_markdown()

    # Write documentation/RESULTS.md
    os.makedirs(DOCS_DIR, exist_ok=True)
    results_doc_path = os.path.join(DOCS_DIR, "RESULTS.md")
    with open(results_doc_path, "w", encoding="utf-8") as f:
        f.write("# Empirical Results & Benchmark Tables\n\n" + results_md)
    print(f"Written: {results_doc_path}")

    # Update README.md between markers
    start_marker = "<!-- RESULTS START -->"
    end_marker = "<!-- RESULTS END -->"

    if os.path.exists(README_PATH):
        with open(README_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        if start_marker in content and end_marker in content:
            pre = content.split(start_marker)[0]
            post = content.split(end_marker)[1]
            new_content = pre + start_marker + "\n\n" + results_md + "\n" + end_marker + post
            with open(README_PATH, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated: {README_PATH} between {start_marker} and {end_marker}")
        else:
            print(f"Markers {start_marker} not found in README.md; will be populated when README is rewritten.")


if __name__ == "__main__":
    update_results()

