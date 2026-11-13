# Compact Sugeno experiment protocol

This protocol is fixed before running the new candidates. Existing paper test
results have already influenced the redesign; they are not a fresh confirmation
set. Neither `samples_test.parquet` nor cached test predictions are read here.

## Models and budget

Use the frozen five-seed ensemble cache `69e158d2977b`, the three vessel classes
Capesize/Panamax/Supramax, and identical common-date support for every competitor.
The comparator is an exactly fitted, bounded constant width + location model.

Primary trial: fixed Gaussian fuzzy memberships on daily volatility regime
(log 14-day / 90-day realized volatility). Fallback trial: memberships on the
log trailing 80th percentile of absolute matured median errors divided by base
interval half-width. Both trials are declared now, not invented after viewing
the confirmation results.

Each trial has three architectures: two width rules + constant location (3
coefficients), three width rules + constant location (4), three width rules +
three location rules (6). For the fallback's fuzzy location, use a separate
median-residual / half-width diagnostic. Each architecture tries rule-difference
L1 penalties {0, 0.01, 0.1}, applied to dimensionless mean pinball plus mean
absolute consequent deviations. Total budget: 18 configurations per horizon.
Membership centres, widths, robust normalization, and location/output bounds
are determined solely from each fit window; no antecedent gradient fitting.

## Temporal protocol

Tuning fold 1: forecast issuance 2023-10-01 <= t < 2023-12-01.
Tuning fold 2: 2023-12-01 <= t < 2024-01-01.
For each fold/horizon, fit only earlier validation forecasts with t+h strictly
before the scoring window. Never use in-sample base-model training errors.

Confirmation: 2024-04-01 <= t through the end of available validation forecasts
(2024-07-25). All tuning outcomes mature before confirmation starts, including
h=90. Refit each frozen configuration on validation outcomes matured strictly
before 2024-04-01. Diagnostics may update from outcomes that mature during
scoring, but fitted coefficients and normalization never update.

Select one configuration per horizon by the equally weighted mean relative
pinball difference against constant W+L across the two tuning folds. Break
near ties by fewer coefficients and stronger shrinkage. Select the primary and
fallback finalists separately, then nominate the better tuning finalist per
horizon BEFORE confirmation. Record both trial finalists and the nominated
pipeline; do not switch nomination after confirmation.

## User-approved longer calibration amendment (before any new scoring)

The operator approved reserving more calibration data. A separate configuration
`src/config/samples_long_calibration.toml` trains the base on 2012-07 through
2019-07-25, leaves the 160-day embargo, and reserves 2020-01 through 2024-07-25
for calibration. Samples are rebuilt from the continuous master calendar into
`src/data/runs/long_calibration_v1/`; neither canonical samples nor test files
are overwritten. Base fitting uses one seed (0), hidden size 32, maximum 100
epochs, patience 8. This is a new exploratory base, not the original five-seed
ensemble. All compared calibrators share its frozen forecasts.

For this larger dataset, the two tuning issuance windows are 2022-01-01 to
2022-07-01 and 2023-01-01 to 2023-07-01 (ends exclusive). Each fit is expanding
from the beginning of available 2020 calibration forecasts, with the same
horizon-maturity purge. Confirmation remains 2024-04-01 through 2024-07-25.
All tuning labels are available before confirmation. Candidate grids and success
threshold are unchanged. A fixed original zero-order and first-order/split-width
Sugeno reference may additionally be fitted on the same earlier rows; no tuning
or nomination is based on their confirmation results.

## Evidence and success

The predeclared exploratory success threshold for the nominated pipeline is:
at least 1% reduction in the equally weighted average horizon-relative pinball
versus constant W+L, improvements on at least two horizons, and no horizon
worse by more than 5%. Record every horizon, class, coverage, Winkler score,
coefficient, output variability, and date-block uncertainty. A point-score win
is not a claim of statistical significance or a pristine held-out test victory.
The two finalist trials are not independent replications.

Save input SHA-256 fingerprints, protocol, solver status, all tuning results,
frozen selection, confirmation predictions and results, and deployment/refit
parameters under a new run directory. Do not overwrite old runs. If unsuccessful,
retain the failed results and request new untouched data before further tuning
against confirmation. Update the paper only with evidence actually obtained.
