# Deprivation score sensitivity

705 districts with all five NFHS-5 indicators. Spearman rank correlation with the current equal-weight score:

| Variant | Spearman rho |
|---|---|
| Equal weights (current score) | 1.000 |
| PC1 (explains 49% of indicator variance) | 0.996 |
| Leave out sanitation | 0.975 |
| Leave out electricity | 0.988 |
| Leave out clean_fuel | 0.932 |
| Leave out stunting | 0.974 |
| Leave out anaemia | 0.975 |

PC1 loadings (sign-aligned): sanitation 0.514, electricity 0.406, clean_fuel 0.468, stunting 0.488, anaemia 0.338

Reading: rho close to 1 means the district ranking is robust to this choice.
