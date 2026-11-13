# Compact Sugeno and longer-calibration experiment

## Outcome

**The frozen nominated compact pipeline did not meet the predeclared exploratory
success criterion. `docs/paper.md` is deliberately unchanged.**

Both proposed fuzzy families were implemented and tested: fixed volatility
memberships, then a predeclared matured-error-ratio fallback. All 18 candidate
configurations per horizon were ranked using earlier chronological tuning
windows. One finalist from each family and the per-horizon nominations were
locked before confirmation losses were computed. No switching to whichever
model later scored best on confirmation is permitted.

The nominated pipeline's equally weighted mean horizon-relative pinball change
against constant W+L was **+0.46%** (worse). The declared target was at least 1%
improvement, improvements at two or more horizons, and no horizon degrading by
more than 5%.

| Calendar horizon | Fit rows | Confirmation rows | Constant W+L pinball | Nominated compact Sugeno pinball | Change vs constant |
|---|---:|---:|---:|---:|---:|
| 1 day | 2,310 | 195 | 0.007270 | 0.007159 | -1.53% |
| 7 days | 2,775 | 237 | 0.028854 | 0.028858 | +0.01% |
| 30 days | 1,650 | 144 | 0.047677 | 0.047743 | +0.14% |
| 90 days | 2,118 | 198 | 0.141365 | 0.145931 | +3.23% |

There is a **real partial win at one day**: the nominated four-parameter
residual-gated width system varies its width multiplier (SD 0.0364), rather than
merely impersonating a scalar adjustment. Its mean pinball and Winkler scores
beat the constant comparator. The exploratory date-block difference interval
is [-0.000188, -0.000020]; it does not account for model search or historical
inspection of validation. Coverage remains above nominal (0.867 versus 0.80).
This does not establish superiority across horizons or on the original test set.

## Larger calibration dataset

Following the operator's approval, the base training cutoff moved to July 25,
2019. After the existing 160-day embargo, calibration spans January 2020 through
July 2024, including the freight spike. The isolated sample files contain 5,293
base-training rows and 3,419 calibration rows across the three vessel classes.
Forecasts start April 30, 2020 after the base model's sequence warm-up.

The base architecture was retrained with hidden size 32, seed 0, maximum 100
epochs and patience 8. All four horizons completed on GPU. This is a separate
one-seed exploratory base, **not** the original paper's five-seed ensemble.
Therefore the experiment is not an isolated causal test of calibration sample
size; the base training period, seed ensemble size, and generated feature panel
also differ from the historical cached run.

Original Sugeno references were also fitted on exactly the same mature fit rows:

| Horizon | Original zero-order change vs constant | Original first-order/split-width change vs constant |
|---|---:|---:|
| 1 | -2.34% | -1.06% |
| 7 | +0.02% | -3.14% |
| 30 | -1.98% | -0.90% |
| 90 | +22.39% | +74.30% |

These are fixed reference comparisons, not candidates retroactively eligible for
nomination. The favorable short/medium-horizon results are promising, but they
cannot be used to assemble a post-selected "winning" pipeline.

The earlier-cutoff base itself transfers poorly at 90 days: confirmation base
pinball is 0.476654, median MAE is 1.0502 log units, and interval coverage is 0.
Constant calibration repairs much of this, but only reaches coverage 0.626.
More calibration observations did not remove this long-horizon base-model
problem. A future experiment could collect walk-forward out-of-sample residuals
without permanently giving up recent base-training data, with a new untouched
evaluation window.

## Implementation and protocol

- `src/ml/compact_sugeno.py`: stable normalized Gaussian firing; two/three rules;
  positive width consequents; constant or fuzzy location; convex sparse pinball
  fitting via HiGHS; L1 rule-difference shrinkage; causal daily-volatility and
  matured-residual inputs; effect audits.
- `src/ml/compact_sugeno_experiment.py`: bounded candidate search, maturity purges,
  balanced common-date support, frozen selection, per-class scoring and bootstrap
  uncertainty. The two long-data tuning windows are January–June 2022 and
  January–June 2023; confirmation is April–July 2024.
- `src/ml/long_calibration.py` and
  `src/config/samples_long_calibration.toml`: isolated split creation and earlier
  base retraining. Existing sample files, caches and test results are preserved.
- `docs/compact_sugeno_protocol.md`: complete protocol and success definition,
  fixed before any new scoring.

The width coefficients are constrained directly to [exp(-0.7), exp(0.7)], avoiding
learned antecedent collapse and exp/tanh optimization. Constant W+L uses the same
convex fitter and bounds. Fixed memberships do not imply static outputs.

## Limitations and failures retained

- Calibration and confirmation are historical development data, not a pristine
  final test. **No `samples_test.parquet` or cached test predictions were read.**
- At 90 days the confirmation window has fewer than two requested 65-date blocks;
  bootstrap limits are left unavailable rather than reporting a degenerate CI.
- There are 44 genuinely nonpositive Capesize index values in early 2020. They
  cannot be represented by a log target. The first attempt stopped before tuning;
  it is retained under `compact_long_v1/failure.json`. The corrected loader warns,
  omits these from logged daily inputs and uses past-only feature LOCF. Nonfinite
  targets are never filled and never scored.
- Larger row counts are not independent sample counts: classes co-move and
  multi-day targets overlap.
- Repository-wide testing exposes four pre-existing optimizer failures involving
  removed Handysize enum values and `lock_wait_text`. They reproduce against an
  immutable archive of HEAD; evidence is saved with the experiment. Optimizer
  APIs were not changed for this ML task.

## Reproducible artifacts

- `src/data/runs/long_calibration_v1/`: separate samples, split/base metadata,
  per-horizon trained models and frozen predictions (key `1ac93085070a`).
- `src/data/runs/compact_long_v2/`: full protocol, hashes, all tuning results and
  coefficients, locked selections, confirmation metrics/predictions, per-class
  scores, and baseline test-failure evidence.

To reproduce scoring using the saved base forecasts, choose a NEW run name:

```sh
uv run python -m ml.compact_sugeno_experiment --run-name compact_long_replication --prediction-key 1ac93085070a --source-dir src/data/runs/long_calibration_v1
```

This reproduces historical development results; it does not create new evidence.
The sample builder and base trainer refuse to overwrite their existing outputs.
