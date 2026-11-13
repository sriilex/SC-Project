# Compact Sugeno — chronological validation confirmation

No frozen test targets or predictions were read. Nomination uses earlier tuning folds only.
This is exploratory: validation was used in previous experiments and base-model selection.

Predeclared exploratory success: **False**.
Nominated pipeline mean horizon-relative pinball change vs constant: **+0.46%**.

| h | model | fit n | score n | parameters | pinball | Δ vs constant | coverage | Winkler | width multiplier SD | shift SD |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | base | 0 | 195 | 0 | 0.008090 | +11.28% | 0.923 | 0.1074 | 0.00000 | 0.00000 |
| 1 | constant | 2310 | 195 | 2 | 0.007270 | +0.00% | 0.887 | 0.1036 | 0.00000 | 0.00000 |
| 1 | volatility | 2310 | 195 | 3 | 0.007294 | +0.33% | 0.897 | 0.1042 | 0.04969 | 0.00000 |
| 1 | residual | 2310 | 195 | 4 | 0.007159 | -1.53% | 0.867 | 0.1002 | 0.03639 | 0.00000 |
| 1 | nominated | 2310 | 195 | 4 | 0.007159 | -1.53% | 0.867 | 0.1002 | 0.03639 | 0.00000 |
| 1 | legacy_sugeno0 | 2310 | 195 | 42 | 0.007100 | -2.34% | 0.774 | 0.1008 | 0.09017 | 0.00258 |
| 1 | legacy_sugeno1_split | 2310 | 195 | 105 | 0.007193 | -1.06% | 0.774 | 0.1020 | 0.11848 | 0.00375 |
| 7 | base | 0 | 237 | 0 | 0.034597 | +19.91% | 0.827 | 0.4298 | 0.00000 | 0.00000 |
| 7 | constant | 2775 | 237 | 2 | 0.028854 | +0.00% | 0.890 | 0.4218 | 0.00000 | 0.00000 |
| 7 | volatility | 2775 | 237 | 3 | 0.028981 | +0.44% | 0.895 | 0.4258 | 0.03353 | 0.00000 |
| 7 | residual | 2775 | 237 | 6 | 0.028858 | +0.01% | 0.886 | 0.4208 | 0.00000 | 0.00252 |
| 7 | nominated | 2775 | 237 | 6 | 0.028858 | +0.01% | 0.886 | 0.4208 | 0.00000 | 0.00252 |
| 7 | legacy_sugeno0 | 2775 | 237 | 42 | 0.028859 | +0.02% | 0.823 | 0.4194 | 0.18893 | 0.02900 |
| 7 | legacy_sugeno1_split | 2775 | 237 | 105 | 0.027947 | -3.14% | 0.793 | 0.4021 | 0.26669 | 0.02531 |
| 30 | base | 0 | 144 | 0 | 0.066341 | +39.15% | 0.660 | 0.8605 | 0.00000 | 0.00000 |
| 30 | constant | 1650 | 144 | 2 | 0.047677 | +0.00% | 0.924 | 0.7450 | 0.00000 | 0.00000 |
| 30 | volatility | 1650 | 144 | 4 | 0.047743 | +0.14% | 0.938 | 0.7528 | 0.06449 | 0.00000 |
| 30 | residual | 1650 | 144 | 3 | 0.047677 | +0.00% | 0.924 | 0.7450 | 0.00000 | 0.00000 |
| 30 | nominated | 1650 | 144 | 4 | 0.047743 | +0.14% | 0.938 | 0.7528 | 0.06449 | 0.00000 |
| 30 | legacy_sugeno0 | 1650 | 144 | 30 | 0.046733 | -1.98% | 0.951 | 0.7432 | 0.09059 | 0.07630 |
| 30 | legacy_sugeno1_split | 1650 | 144 | 57 | 0.047249 | -0.90% | 0.944 | 0.7493 | 0.19173 | 0.06441 |
| 90 | base | 0 | 198 | 0 | 0.476654 | +237.18% | 0.000 | 9.0485 | 0.00000 | 0.00000 |
| 90 | constant | 2118 | 198 | 2 | 0.141365 | +0.00% | 0.626 | 2.4541 | 0.00000 | 0.00000 |
| 90 | volatility | 2118 | 198 | 6 | 0.145931 | +3.23% | 0.621 | 2.5089 | 0.00000 | 0.03037 |
| 90 | residual | 2118 | 198 | 4 | 0.143329 | +1.39% | 0.626 | 2.4933 | 0.00144 | 0.00000 |
| 90 | nominated | 2118 | 198 | 6 | 0.145931 | +3.23% | 0.621 | 2.5089 | 0.00000 | 0.03037 |
| 90 | legacy_sugeno0 | 2118 | 198 | 30 | 0.173012 | +22.39% | 0.394 | 2.9240 | 0.38546 | 0.26520 |
| 90 | legacy_sugeno1_split | 2118 | 198 | 57 | 0.246401 | +74.30% | 0.313 | 4.2263 | 0.27049 | 0.38862 |

## Frozen nominations

- h=1: `residual_r3_constloc_l0`; tuning change vs constant -0.52%.
- h=7: `residual_r3_fuzzyloc_l0.1`; tuning change vs constant -0.40%.
- h=30: `volatility_r3_constloc_l0`; tuning change vs constant -0.26%.
- h=90: `volatility_r3_fuzzyloc_l0.01`; tuning change vs constant -2.14%.

## Limits and next data expansion

Fitting counts are rows, not independent observations. Same-date vessel classes are correlated and long-horizon targets overlap. At h=90 the confirmation window contains very few independent blocks; bootstrap intervals must not be treated as precise significance evidence.
To fit a larger Sugeno system, generate additional walk-forward OUT-OF-SAMPLE base forecasts or reserve a longer calibration period. Do not pad this dataset with in-sample training errors or test outcomes. More epochs/duplicated rows do not add information.
The constant comparator uses an exact convex fit in positive-multiplier/location coordinates, not the old Adam/tanh optimizer; these results do not replace the paper's historical test table.
