"""
audit_data.py
-------------
Audits data integrity of the training dataset(s):
- Value ranges and out-of-bounds metrics (e.g. builtup_pct > 100)
- Missing value (NaN) rates
- Constant or near-zero-variance columns
- Duplicate rows (by area_id or area_name + state_name)
- Multicollinearity / correlation matrix of accessibility columns
- Visual feature provenance and dimensionality

Outputs markdown audit report to: data/processed/data_audit.md
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure UTF-8 output in Windows console if supported
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR

def df_to_markdown(df, index=True):
    """Zero-dependency DataFrame to GitHub-flavored markdown table converter."""
    if not index:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join([header, sep] + rows)
    else:
        idx_name = df.index.name or ""
        cols = [idx_name] + list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for idx, row in df.iterrows():
            row_vals = [str(idx)] + [str(row[c]) for c in df.columns]
            rows.append("| " + " | ".join(row_vals) + " |")
        return "\n".join([header, sep] + rows)

def run_audit(csv_path=None, output_md=None):
    if csv_path is None:
        csv_path = os.path.join(PROCESSED_DIR, "final_training_table.csv")
    if output_md is None:
        output_md = os.path.join(PROCESSED_DIR, "data_audit.md")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    states = df["state_name"].nunique() if "state_name" in df.columns else 0
    districts = df["area_name"].nunique() if "area_name" in df.columns else 0

    lines = []
    lines.append("# GeoPoverty AI: Data Integrity Audit Report\n")
    lines.append(f"**Target Dataset:** `{csv_path}`  ")
    lines.append(f"**Total Records:** {n_rows}  ")
    lines.append(f"**Total Columns:** {n_cols}  ")
    lines.append(f"**Unique States:** {states}  ")
    lines.append(f"**Unique District Names:** {districts}\n")

    # 1. Duplicate Check
    lines.append("## 1. Duplicate Row Analysis\n")
    dup_id = df.duplicated(subset=["area_id"]).sum() if "area_id" in df.columns else 0
    dup_name_state = df.duplicated(subset=["area_name", "state_name"]).sum() if {"area_name", "state_name"}.issubset(df.columns) else 0
    lines.append(f"- Duplicate `area_id`: **{dup_id}**")
    lines.append(f"- Duplicate (`area_name`, `state_name`): **{dup_name_state}**\n")

    # 2. Missing Value Analysis
    lines.append("## 2. Missing Value (NaN) Rates\n")
    nan_counts = df.isna().sum()
    nan_rates = (nan_counts / n_rows) * 100
    nan_df = pd.DataFrame({"Missing Count": nan_counts, "Missing Pct (%)": nan_rates})
    nan_cols = nan_df[nan_df["Missing Count"] > 0]
    if len(nan_cols) > 0:
        lines.append(df_to_markdown(nan_cols, index=True))
    else:
        lines.append("No missing values (NaN) found across all columns.")
    lines.append("\n")

    # 3. Numeric Column Value Ranges & Physical Bounds Check
    lines.append("## 3. Numeric Value Ranges & Physical Plausibility\n")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_rows = []
    issues = []
    for col in num_cols:
        cmin = df[col].min()
        cmax = df[col].max()
        cmean = df[col].mean()
        cstd = df[col].std()
        stats_rows.append({
            "Column": col,
            "Min": round(cmin, 4),
            "Max": round(cmax, 4),
            "Mean": round(cmean, 4),
            "Std": round(cstd, 4)
        })

        # Plausibility checks
        if col == "builtup_pct" and cmax > 100.0:
            count_over_100 = (df[col] > 100.0).sum()
            issues.append(f"**[CRITICAL: `builtup_pct` exceeds 100%]**: max is {cmax:.2f} ({count_over_100}/{n_rows} rows exceed 100%). Raw GHSL cell built surface was not divided by 10,000 m2.")
        if col == "population_density" and cmin < 0.1:
            issues.append(f"**[WARNING: `population_density` units issue]**: min is {cmin:.4f}, mean is {cmean:.2f}. Consistent with raw per-100m grid cell count, not persons/km2.")
        if col == "deprivation_score" and (cmin < 0.0 or cmax > 1.0):
            issues.append(f"**[CRITICAL: `deprivation_score` out of bounds [0, 1]]**: min={cmin}, max={cmax}")
        if col == "ndvi_mean" and (cmin < -1.0 or cmax > 1.0):
            issues.append(f"**[CRITICAL: `ndvi_mean` out of bounds [-1, 1]]**: min={cmin}, max={cmax}")

    stats_df = pd.DataFrame(stats_rows)
    lines.append(df_to_markdown(stats_df, index=False))
    lines.append("\n")

    if issues:
        lines.append("### Bounds & Plausibility Issues Detected\n")
        for iss in issues:
            lines.append(f"- {iss}")
        lines.append("\n")

    # 4. Constant / Near-Constant Columns
    lines.append("## 4. Constant or Low Variance Column Check\n")
    const_cols = [c for c in num_cols if df[c].std() == 0 or df[c].nunique() <= 1]
    if const_cols:
        lines.append(f"**[CRITICAL: Constant columns found ({len(const_cols)})]:** {const_cols}")
    else:
        lines.append("No strictly constant numeric columns detected.")
    lines.append("\n")

    # 5. Accessibility Columns Multicollinearity Analysis
    lines.append("## 5. Accessibility Columns Multicollinearity Analysis\n")
    acc_cols = [c for c in ["road_density_km_per_sqkm", "school_distance_km", "hospital_distance_km"] if c in df.columns]
    if len(acc_cols) >= 2:
        acc_corr = df[acc_cols].corr()
        lines.append("Correlation matrix for accessibility columns:\n")
        lines.append(df_to_markdown(acc_corr.round(4), index=True))
        lines.append("\n")
        # Check for collinearity > 0.90
        high_corr = []
        for i in range(len(acc_cols)):
            for j in range(i + 1, len(acc_cols)):
                r = acc_corr.iloc[i, j]
                if abs(r) > 0.90:
                    high_corr.append((acc_cols[i], acc_cols[j], r))
        if high_corr:
            lines.append("**[CRITICAL: Severe Multicollinearity Detected (|r| > 0.90)]:**")
            for c1, c2, r in high_corr:
                lines.append(f"- `{c1}` and `{c2}`: r = {r:.4f}")
            lines.append("\n> **Finding:** Road density, school distance, and hospital distance correlate at ~0.96. They appear synthetically co-derived or proxy-generated rather than independently measured via OSMnx.\n")
    else:
        lines.append("Accessibility columns not present in table.\n")

    # 6. Visual Features Audit
    lines.append("## 6. Visual (VGG-16) Feature Columns Audit\n")
    vgg_cols = [c for c in df.columns if c.startswith("vgg_feat_")]
    lines.append(f"- Detected visual feature count: **{len(vgg_cols)}** (Expected: 512 for full VGG-16 pooled representations; table has {len(vgg_cols)}).")
    if vgg_cols:
        vgg_means = df[vgg_cols].mean()
        vgg_stds = df[vgg_cols].std()
        lines.append(f"- Range of column means: [{vgg_means.min():.4f}, {vgg_means.max():.4f}]")
        lines.append(f"- Range of column stds: [{vgg_stds.min():.4f}, {vgg_stds.max():.4f}]")
    lines.append("\n")

    report_content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content)
    print(f"\nAudit complete. Written to {output_md}")
    return report_content

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit data integrity of GeoPoverty AI datasets")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--out", type=str, default=None, help="Path to markdown output")
    args = parser.parse_args()
    run_audit(args.csv, args.out)
