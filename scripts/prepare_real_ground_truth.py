import pandas as pd
import os

INPUT = "data/ground_truth/nfhs5_districts_raw.csv"
OUTPUT = "data/ground_truth/real_ground_truth.csv"

INDICATORS = {"sanitation": "9", "electricity": "7", "stunting": "73", "anaemia": "84", "clean_fuel": "10"}

def norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx != mn else pd.Series(0.5, index=s.index)

df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows, {df['district'].nunique()} districts")

frames = {}
for name, num in INDICATORS.items():
    sub = df[df["Indicator"].str.startswith(num + ".")][["state","district","NFHS5"]].copy()
    sub["NFHS5"] = pd.to_numeric(sub["NFHS5"], errors="coerce")
    sub = sub.dropna(subset=["NFHS5"]).rename(columns={"NFHS5": name}).set_index(["state","district"])
    frames[name] = sub
    print(f"  {name}: {len(sub)} districts")

merged = list(frames.values())[0]
for idf in list(frames.values())[1:]:
    merged = merged.join(idf, how="outer")
merged = merged.reset_index().dropna()
print(f"Districts with all 5: {len(merged)}")

for col in ["sanitation","electricity","clean_fuel"]:
    merged[col] = 100.0 - merged[col]
for col in INDICATORS:
    merged[col+"_n"] = norm(merged[col])

merged["deprivation_score"] = merged[[c+"_n" for c in INDICATORS]].mean(axis=1).round(4)

out = pd.DataFrame({"area_id": range(1, len(merged)+1), "area_name": merged["district"].str.strip(), "state_name": merged["state"].str.strip(), "deprivation_score": merged["deprivation_score"], "data_source": "NFHS-5", "data_year": 2021})
out.to_csv(OUTPUT, index=False)
print(f"Saved {len(out)} rows to {OUTPUT}")
print(out["deprivation_score"].describe().round(4))
print("Most deprived:")
print(out.nlargest(5,"deprivation_score")[["area_name","state_name","deprivation_score"]].to_string(index=False))
print("Least deprived:")
print(out.nsmallest(5,"deprivation_score")[["area_name","state_name","deprivation_score"]].to_string(index=False))
