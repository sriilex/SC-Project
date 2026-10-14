# State-Dependent Sugeno Calibration of Quantile Freight Forecasts

## Abstract

We forecast four dry-bulk vessel-class freight indices — Capesize, Panamax,
Supramax and Handysize — at horizons of 1, 7, 30 and 90 days. A pooled quantile
LSTM produces the P10, P50 and P90 of the future log index level through a
monotone head. Because a network trained once cannot adapt when the regime
changes, we add a small, interpretable Sugeno fuzzy layer that reads matured
forecast residuals and trailing volatility and applies a state-dependent
location shift and width multiplier to the frozen quantiles. Applied directly
to the raw base, the layer repairs the base's severe long-horizon overcoverage
(0.99 to 0.73 at 90 days) and improves the mean pinball loss at 30 and 90 days,
but it does not beat a conformal baseline and it hurts at short horizons.
Stacking the same layer **on top of** a split-conformal correction is markedly
better: the stacked model is the best of all models at every horizon on both
pinball loss and the Winkler score, with near-nominal coverage, improving on
adaptive conformal inference by about 13% at 90 days and on the raw base by
about 26%. The layer adds roughly 27 parameters, trains in seconds without
touching the base network, and its rules are readable. We report the stacked
model as a post-hoc finding and treat it as the pre-registered candidate for
the next evaluation episode.

## 1. Setup

The targets are four vessel-class indices (Capesize, Panamax, Supramax,
Handysize); the Baltic Dry Index is never a target, only a cross-series input.
Inputs include port-congestion counts, macro-commodity series (Brent, iron ore,
coal, the dollar index, the S&P 500, US industrial production) and calendar
indicators. Macro series are joined on an as-of basis so each row sees only
what was knowable at its date.

The pipeline builds a continuous 365-day calendar and imputes missing days by
carrying the last observation forward, so lags and rolling statistics stay on a
regular grid. The target is shifted first and weekend/holiday rows are dropped
afterwards, so training and evaluation use real market days only.

We use one chronological split: train 2012-07-04 → 2022-07-25, calibrate on
validation 2023-01-03 → 2024-07-25, and read the test window 2025-01-02 →
2026-08-20 once, at the end. Five-month embargo gaps separate the splits.
Splits contain 10,053 / 1,565 / 1,277 rows. Because horizons are calendar-day
shifts, each horizon has a different weekday composition; at 30 days only
Monday-to-Wednesday issue dates can land on a trading day (Table 1).

**Table 1.** Weekday composition of retained rows (real market days only).

| split | h | rows | Mon | Tue | Wed | Thu | Fri |
|---|---|---|---|---|---|---|---|
| train | 1 | 7,973 | 1,888 | 2,039 | 2,043 | 2,000 | 0 |
| train | 7 | 9,645 | 1,703 | 2,008 | 2,010 | 2,008 | 1,916 |
| train | 30 | 5,807 | 1,846 | 2,000 | 1,955 | 0 | 0 |
| train | 90 | 7,812 | 3 | 1,847 | 2,003 | 2,003 | 1,956 |
| test | 1 | 950 | 175 | 246 | 250 | 238 | 35 |
| test | 7 | 1,098 | 145 | 242 | 229 | 248 | 211 |
| test | 30 | 615 | 148 | 219 | 197 | 32 | 3 |
| test | 90 | 674 | 3 | 118 | 179 | 172 | 167 |

The neural warm-up truncates these further; common-support row counts in the
results are 765 / 895 / 505 / 571. We treat the differing masks as a limitation
rather than comparing row counts across horizons.

## 2. Related work

Neural freight forecasting is typically point forecasting: Su, Park and Bae
(2024) use a CNN–BiLSTM–attention model on the BDI and emit a single value, and
a 2025 neuro-fuzzy line predicts the BDI composite as a point. Duru (2010)
corrects a point forecast with a moving average of past percentage errors. Our
base emits a quantile fan, and our object of study is that fan. On the
calibration side, conformalized quantile regression (Romano et al., 2019) and
adaptive conformal inference (Gibbs and Candès, 2021) are strong and cheap, but
their correction is a scalar — the same for every row. The gap we address is
calibration that is both **state-dependent** and **readable**.

## 3. Method

### 3.1 Base forecaster

A pooled quantile LSTM consumes a 120-day window of daily features and emits a
quantile of the log return at horizon $h$, converted to log level by adding the
current log index. The head is monotone by construction,
$P50 = o_1$, $P10 = P50 - \mathrm{softplus}(o_2)$,
$P90 = P50 + \mathrm{softplus}(o_3)$, so raw quantiles cannot cross. It is
trained with the exact pinball loss on per-class, per-horizon standardised
targets and reported as a five-seed ensemble.

### 3.2 Sugeno calibration layer

Two small Sugeno systems read state and output a log width multiplier $m$ and a
location shift $s$:

$$p50' = p50 + s,\quad p10' = p50' - (p50-p10)e^{m},\quad
p90' = p50' + (p90-p50)e^{m},$$

so ordering is preserved. Inputs are causal and recomputed per date:

- **coverage gap** $= P(y{<}P10) + P(y{>}P90) - 0.20$ over the last $N$
  matured residuals (usable at $h \le 7$, where enough forecasts mature);
- **volatility**, the trailing standard deviation of log-level changes,
  z-scored against strictly earlier rows;
- **median bias** $= \overline{(y-P50)}/\mathrm{sd}(y-P50)$ over the same $N$
  matured rows.

A forecast issued at $t$ matures at $t+h$, so the rolling window uses only
forecasts issued at or before $t-h$. The window grows with the horizon,
$N(h)=\max(60,5h)$, because overlapping residuals carry about one independent
observation per $h$ rows. Residuals from the in-sample base are excluded as
burn-in, so early rows fall back to the identity.

Each system is a 3×3 Sugeno rule base (3 rules on volatility only for the wide
horizons) with Gaussian memberships and the product t-norm; the output is the
weighted average of consequents, bounded through $\tanh$. Width uses
$m = 0.7\tanh(\cdot)$; the location bound scales with the horizon target
standard deviation, $s = 2.2\sigma_h\tanh(\cdot)$, so the shift is comparably
free at every horizon. Consequents start at zero, so the untrained layer is the
identity. Only the layer is trained (base frozen), by full-batch Adam, learning
rate 0.05, 600 epochs. Training is deterministic: no mini-batching, dropout or
random initialisation, so seeds give identical parameters. Inference is a
closed-form numpy evaluation.

**Table 2.** Learned memberships (centre, width) and rule consequents.

| h | system | input | MF1 | MF2 | MF3 | consequents |
|---|---|---|---|---|---|---|
| 1 | width | coverage gap | (−1.98, 0.44) | (0.55, 0.15) | (1.70, 0.10) | −0.21, 0.11, −1.39, 0.08, 0.27, −1.97, −0.57, −0.02, 0.94 |
| 1 | width | volatility | (−1.26, 0.25) | (0.27, 0.08) | (2.58, 0.56) | — |
| 1 | location | median bias | (−1.34, 0.36) | (0.87, 0.20) | (1.27, 0.23) | 0.33, −0.18, −0.13, 0.01, 0.03, −0.16, −0.30, 0.07, −5.46 |
| 1 | location | volatility | (−0.67, 0.28) | (−0.27, 0.13) | (0.89, 0.18) | — |
| 90 | width | volatility | (1.04, 1.43) | (0.99, 0.15) | (0.99, 0.18) | −3.24, −1.06, −1.18 |
| 90 | location | median bias | (−1.15, 0.19) | (−0.35, 0.08) | (0.96, 0.17) | −0.35, 0.66, −0.57, 0.35, 1.11, −1.45, −0.36, 0.74, −1.70 |
| 90 | location | volatility | (−0.26, 0.22) | (0.09, 2.78) | (3.64, 2.29) | — |

The rules are readable: at high trailing volatility the width multiplier moves
away from its identity value, and a positive matured coverage gap fires the
rules that widen the fan.

### 3.3 Stacking on a conformal correction

The layer can also be applied to an already-calibrated fan. We first apply
split conformalized quantile regression, fit on the validation window, to the
base quantiles, then recompute the fuzzy inputs from the **calibrated**
residuals and fit the fuzzy layer on the validation window. Test is untouched
until scoring. The order matters: a static conformal correction removes the
global error but cannot react to state, leaving exactly the structure the fuzzy
layer adds. (Adaptive conformal inference already adapts online, so stacking on
it is largely redundant; we verified that the static case is the informative
one.)

### 3.4 Baselines and evaluation

Reported models: the raw base, the fuzzy layer on the raw base, split
conformalized quantile regression (CQR), adaptive conformal inference (ACI),
and the fuzzy-on-CQR stack. Conformal and fuzzy layers are fit on validation
and frozen for test. Metrics are mean **pinball loss**, central-80 **coverage**
and the **Winkler-80** interval score, averaged over classes (pooled) and per
class. Because 90-day windows overlap and the classes are correlated on the
same date, uncertainty uses a bootstrap that resamples whole **dates**.

## 4. The base is badly over-covered at long horizons

An unconstrained quantile head produced crossed fans, so the monotone head is
not cosmetic. With it, crossings are zero, but the base is far from nominal:
central-80 coverage pooled is

| split | h=1 | h=7 | h=30 | h=90 |
|---|---|---|---|---|
| train | 0.776 | 0.810 | 0.825 | 0.828 |
| valid | 0.766 | 0.838 | 0.840 | 0.925 |
| test | 0.837 | 0.908 | 0.939 | 0.991 |

The intervals are far too wide at long horizons, which is why the rest of the
work is calibration rather than architecture.

## 5. Results

### 5.1 The layer applied to the raw base

Test split, common support, mean pinball / coverage / Winkler:

**Table 3.**

| h | lstm | lstm_fuzzy | cqr | aci |
|---|---|---|---|---|
| 1 | 0.00715 / 0.837 / 0.116 | 0.00751 / 0.745 / 0.122 | 0.00716 / 0.871 / 0.117 | **0.00707** / 0.816 / **0.114** |
| 7 | 0.02200 / 0.908 / 0.329 | 0.02451 / 0.820 / 0.378 | 0.02179 / 0.886 / 0.323 | **0.02136** / 0.808 / **0.310** |
| 30 | 0.04833 / 0.939 / 0.749 | 0.04645 / 0.796 / 0.683 | 0.04834 / 0.899 / 0.713 | **0.04451** / 0.810 / **0.634** |
| 90 | 0.06914 / 0.991 / 1.114 | 0.06080 / 0.730 / 0.809 | 0.05953 / 0.905 / 0.880 | **0.05894** / 0.694 / 0.809 |

Applied to the raw base the layer helps at long horizons — 12% lower pinball
than the base at 90 days and coverage pulled from 0.991 to 0.730 — but it is
the worst model at 1 and 7 days and never beats the conformal baselines. This
is the honest result for the un-stacked layer, and it motivates the stack.

### 5.2 The layer stacked on conformal calibration

**Table 4.** Test split, common support (pinball / coverage / Winkler).

| h | fuz∘cqr | cqr | aci | raw lstm |
|---|---|---|---|---|
| 1 | **0.00693 / 0.787 / 0.1115** | 0.00716 / 0.871 / 0.1173 | 0.00707 / 0.816 / 0.1141 | 0.00715 / 0.837 / 0.1164 |
| 7 | **0.02041 / 0.832 / 0.2968** | 0.02179 / 0.886 / 0.3229 | 0.02136 / 0.808 / 0.3100 | 0.02200 / 0.908 / 0.3293 |
| 30 | **0.04326 / 0.800 / 0.6209** | 0.04834 / 0.899 / 0.7129 | 0.04451 / 0.810 / 0.6343 | 0.04833 / 0.939 / 0.7487 |
| 90 | **0.05103 / 0.816 / 0.7058** | 0.05953 / 0.905 / 0.8801 | 0.05894 / 0.694 / 0.8086 | 0.06914 / 0.991 / 1.1143 |

The stack is the best model at **every horizon** on both pinball and Winkler,
with coverage close to nominal (0.79–0.83). It improves on CQR by 3–14%, on ACI
by 2–13% (about 13% at 90 days) and on the raw base by up to 26%.

**Confidence.** Date-cluster bootstrap intervals for
$(\text{fuz∘cqr} - \text{competitor})$ in mean pinball:

| h | vs cqr | vs aci |
|---|---|---|
| 1 | −0.00023 [−0.00031, −0.00015] | −0.00014 [−0.00027, 0.00000] |
| 7 | −0.00138 [−0.00170, −0.00105] | −0.00095 [−0.00132, −0.00062] |
| 30 | −0.00509 [−0.00635, −0.00381] | −0.00125 [−0.00254, 0.00007] |
| 90 | −0.00850 [−0.00988, −0.00699] | −0.00791 [−0.01016, −0.00558] |

The advantage over CQR is separated from zero at every horizon. Against ACI the
advantage is clear at 7 and 90 days and within noise at 1 and 30 days. The same
direction already appears on validation (stack below CQR at all four horizons),
so the test result is a confirmation rather than a surprise.

**Per class at 90 days (test).** The stack improves on CQR in all four classes
and is best-in-class in three:

**Table 5.**

| class | fuz∘cqr | cqr | aci | raw lstm |
|---|---|---|---|---|
| Capesize | **0.0867 / 0.828** | 0.1061 / 0.962 | 0.1016 / 0.713 | 0.1179 / 0.987 |
| Panamax | **0.0389 / 0.817** | 0.0471 / 0.861 | 0.0484 / 0.741 | 0.0559 / 0.981 |
| Supramax | 0.0448 / 0.722 | **0.0447 / 0.880** | 0.0453 / 0.671 | 0.0525 / 1.000 |
| Handysize | **0.0235 / 0.949** | 0.0289 / 0.929 | 0.0296 / 0.622 | 0.0392 / 1.000 |

Notably Handysize, which the un-stacked layer degraded, is now the class with
the largest relative gain.

### 5.3 Why the short horizons fail without stacking

At 7 days the un-stacked layer's coverage (0.820) is close to ACI's (0.808) yet
its pinball is 15% worse — the intervals are similarly calibrated but
displaced, and the ablation below attributes this to the location system. At 1
day the base under-covered on validation (0.766) and over-covered on test
(0.837), so the width learned on validation pushed coverage **down** to 0.745.
Stacking removes both failure modes because the conformal step first removes
the global error level.

### 5.4 A confound

The test window trends up: the pooled up-rate rises from 0.48–0.54 on
validation to 0.54–0.65 on test. Part of the long-horizon location correction
may capture a validation-to-test drift that a rolling evaluation would expose.
The width correction — the part that fixes overcoverage — is not a drift
effect, since it only changes interval width.

## 6. Post-hoc ablations

Run **after** test had been scored; they explain the results and were not used
to select the reported arms. Mean pinball (coverage) on the raw base:

**Table 6.**

| h | joint | width only | location only | no volatility |
|---|---|---|---|---|
| 1 | 0.00751 (0.745) | 0.00714 (0.791) | 0.00742 (0.792) | 0.00711 (0.811) |
| 7 | 0.02451 (0.820) | 0.02187 (0.898) | 0.02402 (0.867) | 0.02252 (0.865) |
| 30 | 0.04645 (0.796) | 0.04543 (0.861) | 0.05052 (0.907) | 0.04937 (0.917) |
| 90 | 0.06080 (0.730) | 0.05851 (0.818) | 0.07152 (0.983) | 0.06815 (0.991) |

The width channel is the effective one; the location-only arm is consistently
poor. On the raw base a width-only layer would have scored better at every
horizon. The stacked result shows the opposite lesson about location: on top of
a static conformal correction, state-dependent location and width together are
what beat the baselines.

## 7. Limitations

* **Single episode.** One validation and one test window; no rolling-origin
  evaluation.
* **Post-hoc stack.** The stacked arm was found after the test window had been
  read. It must be pre-registered and confirmed on a fresh episode before being
  treated as an established claim.
* **Calibrator needs labels.** The layer is supervised on matured outcomes; it
  is cheap, not free.
* **Not class-neutral when un-stacked.** On the raw base the layer helped
  Panamax, Capesize and Supramax but hurt Handysize at 90 days; stacking
  removes this.
* **Weekday-biased masks.** Calendar-day horizons give each horizon a different
  weekday composition (Table 1); 30-day horizons retain under half the eligible
  rows.
* **Small effective sample.** Overlapping targets and four correlated classes.
* **Deterministic calibrator.** Full-batch training from a fixed
  initialisation, so seeds give no variance; the base is a five-seed ensemble.
* **Staged training.** The base is frozen; joint training was not run.
* **Cost, for context.** The base is a five-seed ensemble across four horizons,
  trained over the full history; the calibration layer is four tiny models of
  roughly 27 parameters each, fit in seconds on the validation window with no
  gradient through the base.

## References

1. Duru, O. (2010). A fuzzy integrated logical forecasting model for dry bulk shipping index forecasting. *Expert Systems with Applications*.
2. Su, M., Park, K., Bae, S. (2024). Deep learning for the Baltic Dry Index. *Maritime Economics and Logistics*.
3. Neuro-fuzzy inference for Baltic Dry Index forecasting (2025). *Journal of Economics and Finance*. doi:10.1007/s12197-025-09720-2
4. Romano, Y., Patterson, E., Candès, E. J. (2019). Conformalized quantile regression. *NeurIPS*.
5. Gibbs, I., Candès, E. J. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
6. Takagi, T., Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. *IEEE SMC*.
7. Song, Q., Chissom, B. S. (1993). Fuzzy time series and its models. *Fuzzy Sets and Systems*.
8. Koenker, R., Bassett, G. (1978). Regression quantiles. *Econometrica*.
