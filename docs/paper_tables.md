# Paper tables — generated from run outputs

Do not edit by hand. Regenerate with `uv run python -m ml.paper_tables`.

## Table 1 — Main test results (pooled, common support)

Delta is the relative change in mean pinball against the base model, computed here.

| h | model | n | mean pinball | Δ vs base | cover-80 | Winkler-80 |
|---|---|---|---|---|---|---|
| 1 | Base | 765 | 0.00715 | — | 0.837 | 0.11636 |
| 1 | Sugeno (zero-order) | 765 | 0.00751 | +5.1% | 0.745 | 0.12172 |
| 1 | Sugeno (selected) | 765 | 0.00774 | +8.3% | 0.749 | 0.12548 |
| 1 | CQR | 765 | 0.00716 | +0.2% | 0.871 | 0.11726 |
| 1 | ACI | 765 | **0.00707** | -1.1% | 0.816 | **0.11408** |
| 7 | Base | 895 | 0.02200 | — | 0.908 | 0.32929 |
| 7 | Sugeno (zero-order) | 895 | 0.02451 | +11.4% | 0.820 | 0.37806 |
| 7 | Sugeno (selected) | 895 | 0.02561 | +16.4% | 0.793 | 0.40181 |
| 7 | CQR | 895 | 0.02179 | -1.0% | 0.886 | 0.32294 |
| 7 | ACI | 895 | **0.02144** | -2.5% | 0.807 | **0.31254** |
| 30 | Base | 505 | 0.04833 | — | 0.939 | 0.74866 |
| 30 | Sugeno (zero-order) | 505 | 0.04645 | -3.9% | 0.796 | 0.68275 |
| 30 | Sugeno (selected) | 505 | 0.05138 | +6.3% | 0.709 | 0.72816 |
| 30 | CQR | 505 | 0.04834 | +0.0% | 0.899 | 0.71287 |
| 30 | ACI | 505 | **0.04577** | -5.3% | 0.796 | **0.67209** |
| 90 | Base | 571 | 0.06914 | — | 0.991 | 1.11435 |
| 90 | Sugeno (zero-order) | 571 | 0.06080 | -12.1% | 0.730 | 0.80893 |
| 90 | Sugeno (selected) | 571 | 0.05957 | -13.8% | 0.793 | **0.80010** |
| 90 | CQR | 571 | **0.05953** | -13.9% | 0.905 | 0.88011 |
| 90 | ACI | 571 | 0.06070 | -12.2% | 0.627 | 0.86142 |

## Table 2 — h=90 per class (test)

Bold marks the lowest mean pinball in each class. The Sugeno layer is **not** best in all four.

| class | model | n | mean pinball | cover-80 | Winkler-80 |
|---|---|---|---|---|---|
| Capesize | Base | 157 | 0.11791 | 0.987 | 1.83621 |
| Capesize | Sugeno (zero-order) | 157 | 0.10200 | 0.726 | 1.34240 |
| Capesize | Sugeno (selected) | 157 | **0.09683** | 0.796 | **1.23337** |
| Capesize | CQR | 157 | 0.10607 | 0.962 | 1.59590 |
| Capesize | ACI | 157 | 0.10295 | 0.675 | 1.38756 |
| Panamax | Base | 158 | 0.05586 | 0.981 | 0.93714 |
| Panamax | Sugeno (zero-order) | 158 | **0.04112** | 0.823 | **0.59433** |
| Panamax | Sugeno (selected) | 158 | 0.04620 | 0.797 | 0.66006 |
| Panamax | CQR | 158 | 0.04711 | 0.861 | 0.72789 |
| Panamax | ACI | 158 | 0.05343 | 0.538 | 0.86427 |
| Supramax | Base | 158 | 0.05249 | 1.000 | 0.80180 |
| Supramax | Sugeno (zero-order) | 158 | 0.04490 | 0.766 | 0.59116 |
| Supramax | Sugeno (selected) | 158 | **0.04454** | 0.785 | 0.60617 |
| Supramax | CQR | 158 | 0.04471 | 0.880 | **0.56598** |
| Supramax | ACI | 158 | 0.04492 | 0.715 | 0.57488 |
| Handysize | Base | 98 | 0.03924 | 1.000 | 0.74750 |
| Handysize | Sugeno (zero-order) | 98 | 0.05220 | 0.531 | 0.65135 |
| Handysize | Sugeno (selected) | 98 | 0.04565 | 0.796 | 0.64444 |
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

## Table 5 — Validation-only calibration ablation (pooled)

Every arm is fitted on `valid` with frozen LSTM quantiles and scored on `valid`;
the test split is not involved. Lower pinball and Winkler are better.

| h | arm | n | mean pinball | cover-80 | Winkler-80 |
|---|---|---|---|---|---|
| 1 | Base (no calibration) | 982 | 0.00770 | 0.776 | 0.1217 |
| 1 | Constant width + location | 982 | 0.00768 | 0.743 | 0.1213 |
| 1 | Linear state-dependent | 982 | 0.00767 | 0.751 | 0.1211 |
| 1 | Sugeno zero-order | 982 | 0.00750 | 0.753 | 0.1181 |
| 1 | Sugeno first-order | 982 | 0.00748 | 0.757 | 0.1176 |
| 1 | Sugeno split width | 982 | 0.00747 | 0.761 | 0.1167 |
| 1 | Sugeno first-order + split width | 982 | **0.00740** | 0.755 | 0.1151 |
| 1 | Sugeno + class shrinkage | 982 | 0.00750 | 0.751 | 0.1181 |
| 1 | Sugeno + early stop / weight decay | 982 | 0.00775 | 0.761 | 0.1221 |
| 1 | Sugeno full combination | 982 | 0.00773 | 0.764 | 0.1216 |
| 7 | Base (no calibration) | 1190 | 0.02610 | 0.833 | 0.3843 |
| 7 | Constant width + location | 1190 | 0.02599 | 0.785 | 0.3808 |
| 7 | Linear state-dependent | 1190 | 0.02590 | 0.788 | 0.3799 |
| 7 | Sugeno zero-order | 1190 | 0.02478 | 0.766 | 0.3619 |
| 7 | Sugeno first-order | 1190 | 0.02459 | 0.792 | 0.3585 |
| 7 | Sugeno split width | 1190 | 0.02459 | 0.798 | 0.3548 |
| 7 | Sugeno first-order + split width | 1190 | **0.02429** | 0.792 | 0.3503 |
| 7 | Sugeno + class shrinkage | 1190 | 0.02478 | 0.766 | 0.3618 |
| 7 | Sugeno + early stop / weight decay | 1190 | 0.02565 | 0.818 | 0.3781 |
| 7 | Sugeno full combination | 1190 | 0.02563 | 0.823 | 0.3776 |
| 30 | Base (no calibration) | 717 | 0.06065 | 0.838 | 0.8914 |
| 30 | Constant width + location | 717 | 0.05960 | 0.757 | 0.8602 |
| 30 | Linear state-dependent | 717 | 0.05722 | 0.773 | 0.8197 |
| 30 | Sugeno zero-order | 717 | 0.05185 | 0.796 | 0.7391 |
| 30 | Sugeno first-order | 717 | 0.04928 | 0.794 | 0.6866 |
| 30 | Sugeno split width | 717 | 0.05210 | 0.774 | 0.7499 |
| 30 | Sugeno first-order + split width | 717 | **0.04853** | 0.805 | 0.6806 |
| 30 | Sugeno + class shrinkage | 717 | 0.05178 | 0.792 | 0.7385 |
| 30 | Sugeno + early stop / weight decay | 717 | 0.05523 | 0.795 | 0.7774 |
| 30 | Sugeno full combination | 717 | 0.05460 | 0.823 | 0.7756 |
| 90 | Base (no calibration) | 970 | 0.07188 | 0.916 | 1.1748 |
| 90 | Constant width + location | 970 | 0.06361 | 0.773 | 0.9113 |
| 90 | Linear state-dependent | 970 | 0.05694 | 0.781 | 0.7995 |
| 90 | Sugeno zero-order | 970 | 0.05109 | 0.854 | 0.7288 |
| 90 | Sugeno first-order | 970 | 0.04785 | 0.849 | 0.6964 |
| 90 | Sugeno split width | 970 | 0.05067 | 0.869 | 0.7301 |
| 90 | Sugeno first-order + split width | 970 | **0.04781** | 0.861 | 0.6976 |
| 90 | Sugeno + class shrinkage | 970 | 0.05057 | 0.854 | 0.7246 |
| 90 | Sugeno + early stop / weight decay | 970 | 0.05570 | 0.858 | 0.7765 |
| 90 | Sugeno full combination | 970 | 0.05480 | 0.903 | 0.7801 |

## Table 5b — Validation ablation at h=90, per class

Same arms and protocol as Table 5, restricted to the long horizon and split by class.

| class | arm | n | mean pinball | cover-80 | Winkler-80 |
|---|---|---|---|---|---|
| Capesize | Base (no calibration) | 242 | 0.12043 | 0.946 | 2.0663 |
| Capesize | Constant width + location | 242 | 0.09649 | 0.818 | 1.4968 |
| Capesize | Linear state-dependent | 242 | 0.09139 | 0.822 | 1.4035 |
| Capesize | Sugeno zero-order | 242 | 0.09314 | 0.810 | 1.3595 |
| Capesize | Sugeno first-order + split width | 242 | 0.08757 | 0.843 | 1.2722 |
| Panamax | Base (no calibration) | 242 | 0.06033 | 0.967 | 0.9606 |
| Panamax | Constant width + location | 242 | 0.05225 | 0.777 | 0.7130 |
| Panamax | Linear state-dependent | 242 | 0.04972 | 0.748 | 0.6779 |
| Panamax | Sugeno zero-order | 242 | 0.03875 | 0.860 | 0.5521 |
| Panamax | Sugeno first-order + split width | 242 | 0.03696 | 0.909 | 0.5313 |
| Supramax | Base (no calibration) | 242 | 0.05597 | 0.888 | 0.8507 |
| Supramax | Constant width + location | 242 | 0.05474 | 0.723 | 0.7276 |
| Supramax | Linear state-dependent | 242 | 0.04360 | 0.764 | 0.5640 |
| Supramax | Sugeno zero-order | 242 | 0.03430 | 0.893 | 0.4783 |
| Supramax | Sugeno first-order + split width | 242 | 0.03447 | 0.822 | 0.4870 |
| Handysize | Base (no calibration) | 244 | 0.05095 | 0.865 | 0.8244 |
| Handysize | Constant width + location | 244 | 0.05107 | 0.775 | 0.7095 |
| Handysize | Linear state-dependent | 244 | 0.04317 | 0.791 | 0.5547 |
| Handysize | Sugeno zero-order | 244 | 0.03828 | 0.852 | 0.5270 |
| Handysize | Sugeno first-order + split width | 244 | **0.03234** | 0.869 | 0.5013 |

## Table 6 — Frozen test: selected variant vs base and frozen simple calibrators

The Sugeno row is the only arm scored on test in the new run (one-shot, after
selection on validation). Base and simple-calibrator test numbers are the
previously frozen reference run, not new test reads.

| h | model | n | mean pinball | cover-80 | Winkler-80 |
|---|---|---|---|---|---|
| 1 | Sugeno (selected) | 765 | 0.00774 | 0.749 | 0.12548 |
| 1 | Base | 765 | 0.00715 | 0.837 | 0.11636 |
| 1 | Constant width | 765 | 0.00711 | 0.816 | 0.11536 |
| 1 | Constant width + location | 765 | **0.00707** | 0.813 | **0.11449** |
| 1 | Constant width + rolling bias | 765 | 0.00710 | 0.820 | 0.11518 |
| 1 | Linear state-dependent | 765 | 0.00709 | 0.835 | 0.11510 |
| 7 | Sugeno (selected) | 895 | 0.02561 | 0.793 | 0.40181 |
| 7 | Base | 895 | 0.02200 | 0.908 | 0.32929 |
| 7 | Constant width | 895 | **0.02152** | 0.880 | 0.31507 |
| 7 | Constant width + location | 895 | 0.02154 | 0.879 | 0.31561 |
| 7 | Constant width + rolling bias | 895 | 0.02155 | 0.879 | **0.31493** |
| 7 | Linear state-dependent | 895 | 0.02173 | 0.886 | 0.32023 |
| 30 | Sugeno (selected) | 505 | 0.05138 | 0.709 | 0.72816 |
| 30 | Base | 505 | 0.04833 | 0.939 | 0.74866 |
| 30 | Constant width | 505 | 0.04573 | 0.879 | 0.67074 |
| 30 | Constant width + location | 505 | 0.04576 | 0.877 | 0.67002 |
| 30 | Constant width + rolling bias | 505 | 0.04547 | 0.877 | 0.66669 |
| 30 | Linear state-dependent | 505 | **0.04464** | 0.861 | **0.65381** |
| 90 | Sugeno (selected) | 571 | 0.05957 | 0.793 | 0.80010 |
| 90 | Base | 571 | 0.06914 | 0.991 | 1.11435 |
| 90 | Constant width | 571 | 0.05810 | 0.851 | 0.78323 |
| 90 | Constant width + location | 571 | **0.05207** | 0.886 | **0.73128** |
| 90 | Constant width + rolling bias | 571 | 0.05872 | 0.846 | 0.78736 |
| 90 | Linear state-dependent | 571 | 0.05302 | 0.863 | 0.73310 |

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
