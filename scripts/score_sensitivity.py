"""
score_sensitivity.py
--------------------
Checks how much the deprivation_score depends on the choices made in
prepare_real_ground_truth.py (five NFHS-5 indicators, min-max scaling, equal
weights). It compares the current score with:
  * a data-driven score (first principal component of the five scaled indicators)
  * five leave-one-indicator-out scores
using Spearman rank correlation across districts. High correlations mean the
district ranking does not hinge on one arbitrary choice.

    python scripts/score_sensitivity.py
Writes data/ground_truth/score_sensitivity.md
"""

import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from config import GROUND_TRUTH_DIR

RAW = os.path.join(GROUND_TRUTH_DIR, "nfhs5_districts_raw.csv")
OUT = os.path.join(GROUND_TRUTH_DIR, "score_sensitivity.md")
# NFHS-5 indicator number -> (name, True if a HIGH raw value means LESS deprivation)
INDICATORS = {
    "sanitation": ("9", True), "electricity": ("7", True), "clean_fuel": ("10", True),
    "stunting": ("73", False), "anaemia": ("84", False),
}


def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx != mn else pd.Series(0.5, index=s.index)


def main():
    raw = pd.read_csv(RAW)
    frames = []
    for name, (num, _) in INDICATORS.items():
        sub = raw[raw["Indicator"].str.startswith(num + ".")][["state", "district", "NFHS5"]].copy()
        sub["NFHS5"] = pd.to_numeric(sub["NFHS5"], errors="coerce")
        frames.append(sub.dropna().rename(columns={"NFHS5": name}).set_index(["state", "district"]))
    m = frames[0].join(frames[1:], how="inner").dropna()

    scaled = pd.DataFrame(index=m.index)
    for name, (_, higher_is_better) in INDICATORS.items():
        col = 100.0 - m[name] if higher_is_better else m[name]
        scaled[name] = minmax(col)

    base = scaled.mean(axis=1)

    # PC1 of the standardised indicators, sign-aligned with the equal-weight score
    z = (scaled - scaled.mean()) / scaled.std(ddof=0)
    _, svals, vt = np.linalg.svd(z.values, full_matrices=False)
    pc1 = z.values @ vt[0]
    if np.corrcoef(pc1, base)[0, 1] < 0:
        pc1 = -pc1
    explained = svals[0] ** 2 / (svals ** 2).sum()

    rows = [("Equal weights (current score)", 1.0),
            (f"PC1 (explains {explained:.0%} of indicator variance)",
             pd.Series(pc1, index=m.index).corr(base, method="spearman"))]
    for name in INDICATORS:
        alt = scaled.drop(columns=[name]).mean(axis=1)
        rows.append((f"Leave out {name}", alt.corr(base, method="spearman")))

    weights = pd.Series(vt[0] * (1 if np.corrcoef(z.values @ vt[0], base)[0, 1] > 0 else -1),
                        index=scaled.columns).round(3)

    lines = ["# Deprivation score sensitivity", "",
             f"{len(m)} districts with all five NFHS-5 indicators. Spearman rank correlation with the current equal-weight score:", "",
             "| Variant | Spearman rho |", "|---|---|"]
    lines += [f"| {n} | {r:.3f} |" for n, r in rows]
    lines += ["", "PC1 loadings (sign-aligned): " + ", ".join(f"{k} {v}" for k, v in weights.items()), "",
              "Reading: rho close to 1 means the district ranking is robust to this choice."]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
