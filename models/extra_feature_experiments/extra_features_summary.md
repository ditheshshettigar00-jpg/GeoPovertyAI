# Extra-feature experiment (experimental, not production)

Model: RandomForest. Leave-one-state-out plus repeated state-grouped 5-fold (10 repeats).

Rule fixed in advance: adopt only if LOSO and repeated-CV R2 each gain >= 0.02 and >= 55% of states improve.

```
                    feature_set  n_features  loso_r2  loso_mae  loso_spearman  loso_top20_recall  repeated_r2_mean  repeated_r2_std  d_loso_r2  d_repeated_r2  states_improved
          baseline (4 features)           4   0.3398    0.0883         0.5787             0.4336            0.3074           0.0359     0.0000         0.0000           0.0000
           baseline + ALL extra          10   0.4248    0.0815         0.6769             0.4602            0.3888           0.0412     0.0850         0.0814           0.6129
    baseline + elevation_mean_m           5   0.3793    0.0847         0.6119             0.4867            0.3440           0.0418     0.0395         0.0365           0.5806
      baseline + slope_mean_deg           5   0.3504    0.0878         0.5915             0.4425            0.3147           0.0358     0.0106         0.0073           0.5161
    baseline + rainfall_mm_year           5   0.2982    0.0906         0.5509             0.4336            0.2544           0.0560    -0.0416        -0.0530           0.4516
           baseline + lst_day_c           5   0.4052    0.0838         0.6420             0.4602            0.3694           0.0284     0.0654         0.0620           0.6774
        baseline + cropland_pct           5   0.3510    0.0874         0.5819             0.4779            0.3351           0.0311     0.0112         0.0277           0.5161
baseline + travel_time_city_min           5   0.3434    0.0887         0.5883             0.4513            0.3123           0.0362     0.0036         0.0048           0.4839
```

**Verdict:** ADOPT
