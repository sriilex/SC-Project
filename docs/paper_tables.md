# Paper tables — generated from run outputs

Do not edit by hand. Regenerate with `uv run python -m ml.paper_tables`.

> **Note:** All evaluations use the 3-class panel (Capesize, Panamax, Supramax) on a balanced common-date calendar. Handysize is excluded due to a 7.5-month BHSI data gap in 2025 (see paper Section 3.2 and Appendix C). Pooled `n` reflects 3-class row totals only.

## Table 1 — Main test results (pooled, common support)

Delta is the relative change in mean pinball against the base model, computed here.

| h | model | n | mean pinball | Δ vs base | cover-80 | Winkler-80 |
|---|---|---|---|---|---|---|
| 1 | Base Forecaster | 620 | **0.00839** | — | 0.819 | **0.13748** |
| 1 | Sugeno (0th-order) | 620 | 0.00869 | +3.5% | 0.732 | 0.14195 |
| 1 | Sugeno (1st-order) | 620 | 0.00889 | +5.9% | 0.734 | 0.14506 |
| 7 | Base Forecaster | 726 | **0.02526** | — | 0.902 | **0.37863** |
| 7 | Sugeno (0th-order) | 726 | 0.02804 | +11.0% | 0.803 | 0.43494 |
| 7 | Sugeno (1st-order) | 726 | 0.02898 | +14.7% | 0.793 | 0.45428 |
| 30 | Base Forecaster | 413 | 0.05185 | — | 0.944 | 0.81158 |
| 30 | Sugeno (0th-order) | 413 | **0.04979** | -3.9% | 0.797 | **0.74116** |
| 30 | Sugeno (1st-order) | 413 | 0.05323 | +2.6% | 0.746 | 0.75070 |
| 90 | Base Forecaster | 473 | 0.07533 | — | 0.989 | 1.19035 |
| 90 | Sugeno (0th-order) | 473 | 0.06259 | -16.9% | 0.772 | 0.84158 |
| 90 | Sugeno (1st-order) | 473 | **0.06245** | **-17.1%** | 0.793 | **0.83235** |

## Table 2 — h=90 per class (test)

Bold marks the lowest mean pinball in each class. The Sugeno layer improves all three classes at h=90.

| class | model | n | mean pinball | cover-80 | Winkler-80 |
|---|---|---|---|---|---|
| Capesize | Base Forecaster | 157 | 0.11791 | 0.987 | 1.83621 |
| Capesize | Sugeno (0th-order) | 157 | 0.10200 | 0.726 | 1.34240 |
| Capesize | Sugeno (1st-order) | 157 | **0.09683** | 0.796 | **1.23337** |
| Panamax | Base Forecaster | 158 | 0.05586 | 0.981 | 0.93714 |
| Panamax | Sugeno (0th-order) | 158 | **0.04112** | 0.823 | **0.59433** |
| Panamax | Sugeno (1st-order) | 158 | 0.04620 | 0.797 | 0.66006 |
| Supramax | Base Forecaster | 158 | 0.05249 | 1.000 | 0.80180 |
| Supramax | Sugeno (0th-order) | 158 | 0.04490 | 0.766 | **0.59116** |
| Supramax | Sugeno (1st-order) | 158 | **0.04454** | 0.785 | 0.60617 |

## Table 3 — Attribution baselines (test, pooled)

All arms use the validation split, the same pinball objective and the same tanh bounds.
Median loss is the P50 pinball component; Winkler is the endpoint score.

| h | calibrator | n | mean pinball | median loss | cover-80 | Winkler-80 |
|---|---|---|---|---|---|---|
| 1 | Constant width | 620 | 0.00835 | 0.01155 | 0.798 | 0.1364 |
| 1 | Constant width + location | 620 | **0.00830** | **0.01150** | 0.794 | **0.1354** |
| 1 | Constant width + rolling bias | 620 | 0.00834 | 0.01155 | 0.803 | 0.1362 |
| 1 | Linear state-dependent | 620 | 0.00832 | 0.01150 | 0.815 | 0.1361 |
| 7 | Constant width | 726 | **0.02472** | 0.03793 | 0.882 | 0.3624 |
| 7 | Constant width + location | 726 | 0.02473 | 0.03793 | 0.880 | 0.3629 |
| 7 | Constant width + rolling bias | 726 | 0.02472 | 0.03793 | 0.880 | 0.3621 |
| 7 | Linear state-dependent | 726 | 0.02491 | 0.03793 | 0.883 | 0.3679 |
| 30 | Constant width | 413 | 0.04885 | 0.07507 | 0.891 | 0.7217 |
| 30 | Constant width + location | 413 | 0.04889 | 0.07507 | 0.889 | 0.7211 |
| 30 | Constant width + rolling bias | 413 | 0.04853 | 0.07507 | 0.889 | 0.7165 |
| 30 | Linear state-dependent | 413 | **0.04757** | **0.07507** | 0.872 | **0.7014** |
| 90 | Constant width | 473 | 0.06397 | 0.10418 | 0.833 | 0.8496 |
| 90 | Constant width + location | 473 | **0.05743** | **0.09041** | 0.867 | **0.7957** |
| 90 | Constant width + rolling bias | 473 | 0.06469 | 0.10418 | 0.829 | 0.8550 |
| 90 | Linear state-dependent | 473 | 0.05881 | 0.09041 | 0.837 | 0.8030 |

## Table 5 — Validation-only calibration ablation (pooled)

Every arm is fitted on `valid` with frozen LSTM quantiles and scored on `valid`;
the test split is not involved. Lower pinball and Winkler are better.

| h | arm | n | mean pinball | cover-80 | Winkler-80 |
|---|---|---|---|---|---|
| 1 | Base (no calibration) | 982 | 0.00770 | 0.776 | 0.1217 |
| 1 | Constant width + location | 982 | 0.00768 | 0.743 | 0.1213 |
| 1 | Linear state-dependent | 982 | 0.00767 | 0.751 | 0.1211 |
| 1 | Sugeno (0th-order) | 982 | 0.00750 | 0.753 | 0.1181 |
| 1 | Sugeno (1st-order) | 982 | **0.00740** | 0.755 | **0.1151** |
| 7 | Base (no calibration) | 1190 | 0.02610 | 0.833 | 0.3843 |
| 7 | Constant width + location | 1190 | 0.02599 | 0.785 | 0.3808 |
| 7 | Linear state-dependent | 1190 | 0.02596 | 0.787 | 0.3793 |
| 7 | Sugeno (0th-order) | 1190 | 0.02559 | 0.784 | 0.3714 |
| 7 | Sugeno (1st-order) | 1190 | **0.02517** | 0.790 | **0.3651** |
| 30 | Base (no calibration) | 740 | 0.04855 | 0.908 | 0.7462 |
| 30 | Constant width + location | 740 | 0.04695 | 0.849 | 0.7120 |
| 30 | Linear state-dependent | 740 | 0.04697 | 0.850 | 0.7138 |
| 30 | Sugeno (0th-order) | 740 | **0.04612** | 0.845 | **0.7023** |
| 30 | Sugeno (1st-order) | 740 | 0.04640 | 0.846 | 0.7072 |
| 90 | Base (no calibration) | 540 | 0.09085 | 0.981 | 1.4584 |
| 90 | Constant width + location | 540 | 0.07438 | 0.946 | 1.1716 |
| 90 | Linear state-dependent | 540 | 0.07502 | 0.933 | 1.1875 |
| 90 | Sugeno (0th-order) | 540 | 0.07028 | 0.924 | 1.0787 |
| 90 | Sugeno (1st-order) | 540 | **0.06982** | 0.924 | **1.0636** |
