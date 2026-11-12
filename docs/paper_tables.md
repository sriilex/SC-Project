# Paper tables — generated from run outputs

Do not edit by hand. Regenerate with `uv run python -m ml.paper_tables`.

## Table 1 — Main test results (pooled, common support)

Delta is the relative change in mean pinball against the base model, computed here.

| h | model | n | mean pinball | Δ vs base | cover-80 | Winkler-80 |
|---|---|---|---|---|---|---|
| 1 | Base | 765 | 0.00715 | — | 0.837 | 0.11636 |
| 1 | Sugeno | 765 | 0.00751 | +5.1% | 0.745 | 0.12172 |
| 1 | CQR | 765 | 0.00716 | +0.2% | 0.871 | 0.11726 |
| 1 | ACI | 765 | **0.00707** | -1.1% | 0.816 | **0.11408** |
| 7 | Base | 895 | 0.02200 | — | 0.908 | 0.32929 |
| 7 | Sugeno | 895 | 0.02451 | +11.4% | 0.820 | 0.37806 |
| 7 | CQR | 895 | 0.02179 | -1.0% | 0.886 | 0.32294 |
| 7 | ACI | 895 | **0.02144** | -2.5% | 0.807 | **0.31254** |
| 30 | Base | 505 | 0.04833 | — | 0.939 | 0.74866 |
| 30 | Sugeno | 505 | 0.04645 | -3.9% | 0.796 | 0.68275 |
| 30 | CQR | 505 | 0.04834 | +0.0% | 0.899 | 0.71287 |
| 30 | ACI | 505 | **0.04577** | -5.3% | 0.796 | **0.67209** |
| 90 | Base | 571 | 0.06914 | — | 0.991 | 1.11435 |
| 90 | Sugeno | 571 | 0.06080 | -12.1% | 0.730 | **0.80893** |
| 90 | CQR | 571 | **0.05953** | -13.9% | 0.905 | 0.88011 |
| 90 | ACI | 571 | 0.06070 | -12.2% | 0.627 | 0.86142 |

## Table 2 — h=90 per class (test)

Bold marks the lowest mean pinball in each class. The Sugeno layer is **not** best in all four.

| class | model | n | mean pinball | cover-80 | Winkler-80 |
|---|---|---|---|---|---|
| Capesize | Base | 157 | 0.11791 | 0.987 | 1.83621 |
| Capesize | Sugeno | 157 | **0.10200** | 0.726 | **1.34240** |
| Capesize | CQR | 157 | 0.10607 | 0.962 | 1.59590 |
| Capesize | ACI | 157 | 0.10295 | 0.675 | 1.38756 |
| Panamax | Base | 158 | 0.05586 | 0.981 | 0.93714 |
| Panamax | Sugeno | 158 | **0.04112** | 0.823 | **0.59433** |
| Panamax | CQR | 158 | 0.04711 | 0.861 | 0.72789 |
| Panamax | ACI | 158 | 0.05343 | 0.538 | 0.86427 |
| Supramax | Base | 158 | 0.05249 | 1.000 | 0.80180 |
| Supramax | Sugeno | 158 | 0.04490 | 0.766 | 0.59116 |
| Supramax | CQR | 158 | **0.04471** | 0.880 | **0.56598** |
| Supramax | ACI | 158 | 0.04492 | 0.715 | 0.57488 |
| Handysize | Base | 98 | 0.03924 | 1.000 | 0.74750 |
| Handysize | Sugeno | 98 | 0.05220 | 0.531 | 0.65135 |
| Handysize | CQR | 98 | **0.02888** | 0.929 | 0.48524 |
| Handysize | ACI | 98 | 0.03018 | 0.551 | **0.47589** |

## Table 3 — Attribution baselines (test, pooled)

All arms use the validation split, the same pinball objective and the same tanh bounds.
Median loss is the P50 pinball component; Winkler is the endpoint score.

| h | calibrator | n | mean pinball | median loss | cover-80 | Winkler-80 |
|---|---|---|---|---|---|---|
| 1 | Constant width | 765 | 0.00711 | 0.00981 | 0.816 | 0.1154 |
| 1 | Constant width + location | 765 | **0.00707** | 0.00976 | 0.813 | 0.1145 |
| 1 | Constant width + rolling bias | 765 | 0.00710 | 0.00979 | 0.820 | 0.1152 |
| 1 | Linear state-dependent | 765 | 0.00709 | 0.00976 | 0.835 | 0.1151 |
| 7 | Constant width | 895 | **0.02152** | 0.03306 | 0.880 | 0.3151 |
| 7 | Constant width + location | 895 | 0.02154 | 0.03307 | 0.879 | 0.3156 |
| 7 | Constant width + rolling bias | 895 | 0.02155 | 0.03315 | 0.879 | 0.3149 |
| 7 | Linear state-dependent | 895 | 0.02173 | 0.03316 | 0.886 | 0.3202 |
| 30 | Constant width | 505 | 0.04573 | 0.07011 | 0.879 | 0.6707 |
| 30 | Constant width + location | 505 | 0.04576 | 0.07026 | 0.877 | 0.6700 |
| 30 | Constant width + rolling bias | 505 | 0.04547 | 0.06973 | 0.877 | 0.6667 |
| 30 | Linear state-dependent | 505 | **0.04464** | 0.06855 | 0.861 | 0.6538 |
| 90 | Constant width | 571 | 0.05810 | 0.09597 | 0.851 | 0.7832 |
| 90 | Constant width + location | 571 | **0.05207** | 0.08309 | 0.886 | 0.7313 |
| 90 | Constant width + rolling bias | 571 | 0.05872 | 0.09741 | 0.846 | 0.7874 |
| 90 | Linear state-dependent | 571 | 0.05302 | 0.08576 | 0.863 | 0.7331 |

## Table 4 — Fuzzy-on-CQR stack (test, pooled)

Post-hoc arm. The fuzzy stage does **not** inherit CQR's coverage guarantee.

| h | model | n | mean pinball | cover-80 | Winkler-80 |
|---|---|---|---|---|---|
| 1 | Base | 765 | **0.00715** | 0.837 | **0.11636** |
| 1 | CQR | 765 | 0.00716 | 0.871 | 0.11726 |
| 1 | Sugeno on CQR | 765 | 0.00788 | 0.739 | 0.12949 |
| 7 | Base | 895 | 0.02200 | 0.908 | 0.32929 |
| 7 | CQR | 895 | **0.02179** | 0.886 | **0.32294** |
| 7 | Sugeno on CQR | 895 | 0.02414 | 0.798 | 0.36644 |
| 30 | Base | 505 | **0.04833** | 0.939 | 0.74866 |
| 30 | CQR | 505 | 0.04834 | 0.899 | **0.71287** |
| 30 | Sugeno on CQR | 505 | 0.05334 | 0.735 | 0.76086 |
| 90 | Base | 571 | 0.06914 | 0.991 | 1.11435 |
| 90 | CQR | 571 | 0.05953 | 0.905 | 0.88011 |
| 90 | Sugeno on CQR | 571 | **0.05776** | 0.793 | **0.80843** |

## Appendix A — Pairwise uncertainty (date-block bootstrap)

Differences are Sugeno minus competitor in mean pinball (negative favours Sugeno).
The DM p-value is shown only for completeness; the date-block interval is the primary evidence.
Block sensitivity uses half and twice the primary block length.

| split | h | vs | mean diff | DM p | date-block 95% | block/2 95% | block×2 95% |
|---|---|---|---|---|---|---|---|
| test | 1 | aci | +0.00044 | 0.000 | [+0.00013, +0.00085] | [+0.00013, +0.00085] | [+0.00009, +0.00094] |
| test | 1 | const_shift | +0.00051 | 0.000 | [+0.00021, +0.00087] | [+0.00021, +0.00087] | [+0.00016, +0.00100] |
| test | 1 | cqr | +0.00035 | 0.000 | [+0.00007, +0.00071] | [+0.00007, +0.00071] | [+0.00002, +0.00082] |
| test | 1 | lstm | +0.00037 | 0.000 | [+0.00010, +0.00070] | [+0.00010, +0.00070] | [+0.00005, +0.00081] |
| test | 7 | aci | +0.00307 | 0.000 | [+0.00122, +0.00518] | [+0.00136, +0.00491] | [+0.00089, +0.00556] |
| test | 7 | const_shift | +0.00248 | 0.000 | [+0.00063, +0.00472] | [+0.00076, +0.00445] | [+0.00026, +0.00525] |
| test | 7 | cqr | +0.00272 | 0.000 | [+0.00085, +0.00500] | [+0.00101, +0.00469] | [+0.00053, +0.00545] |
| test | 7 | lstm | +0.00251 | 0.000 | [+0.00064, +0.00478] | [+0.00078, +0.00451] | [+0.00030, +0.00531] |
| test | 30 | aci | +0.00067 | 0.769 | [-0.00448, +0.00516] | [-0.00411, +0.00513] | [-0.00520, +0.00476] |
| test | 30 | const_shift | -0.00179 | 0.485 | [-0.00743, +0.00307] | [-0.00688, +0.00332] | [-0.00808, +0.00376] |
| test | 30 | cqr | -0.00190 | 0.434 | [-0.00655, +0.00222] | [-0.00635, +0.00268] | [-0.00715, +0.00236] |
| test | 30 | lstm | -0.00188 | 0.438 | [-0.00624, +0.00234] | [-0.00623, +0.00289] | [-0.00652, +0.00247] |
| test | 90 | aci | +0.00010 | 0.980 | [-0.00565, +0.00248] | [-0.00792, +0.00518] | [-0.00565, +0.00248] |
| test | 90 | const_shift | -0.00126 | 0.669 | [-0.00282, +0.00137] | [-0.00350, +0.00158] | [-0.00282, +0.00137] |
| test | 90 | cqr | +0.00128 | 0.612 | [-0.00084, +0.00293] | [-0.00075, +0.00303] | [-0.00084, +0.00293] |
| test | 90 | lstm | -0.00833 | 0.002 | [-0.01014, -0.00632] | [-0.01041, -0.00624] | [-0.01014, -0.00632] |
