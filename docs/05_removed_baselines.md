# Removed forecasting baselines

The ML evaluation code was reduced to the neural + fuzzy forecasting models and
the conformal calibration layers. Every non-neural, non-fuzzy *forecasting*
baseline was removed:

| removed model | what it was |
|---------------|-------------|
| `rw`       | Persistence (random-walk) baseline: the point forecast is the last observed log level, and the central-80 band is `±1.2816 · σ` where `σ` is the per-class train residual std (pooled fallback). |
| `rw_emp`   | Random walk with per-class empirical return quantiles: `P50` is exactly the current log level and the `P10`/`P90` come from the empirical distribution of historical `h`-step returns (per class, pooled fallback). |
| `ar1`      | Pooled AR(1) with drift fit on log returns, giving a momentum-aware median level with `±1.2816 · σ` bands from the residual std. |
| `xgb`      | Pooled XGBoost quantile models (`reg:quantileerror`), one booster per horizon. |
| `lgbm`     | Pooled LightGBM quantile models (`objective="quantile"`), one booster per horizon × quantile level. |

Kept in the evaluation harness: `lstm`, `lstm_fuzzy`, `lstm_fuzzy_old`, and the
calibration layers `const_shift`, `cqr`, `aci`. The directional `naive_sign`
diagnostic was also kept (it is a hit-rate diagnostic, not a quantile
forecaster).

## Frozen numbers (common support)

Copied from `src/data/baseline_metrics_common.csv` (scope `POOLED`;
`mean pinball` = `pinball_mean`, `cover_80`, `winkler_80`). The common-support
table only ever contained `ar1`, `xgb` and `rw_emp` among the removed models.

### `ar1`

| split | h | n | mean pinball | cover_80 | winkler_80 |
|-------|---|----|--------------|----------|------------|
| test  | 1  | 765  | 0.006653 | 0.9242 | 0.12247 |
| test  | 7  | 895  | 0.022108 | 0.9318 | 0.35703 |
| test  | 30 | 505  | 0.052209 | 0.9386 | 0.86114 |
| test  | 90 | 571  | 0.074638 | 0.9930 | 1.27304 |
| valid | 1  | 982  | 0.007358 | 0.8544 | 0.12304 |
| valid | 7  | 1190 | 0.026076 | 0.8445 | 0.38384 |
| valid | 30 | 717  | 0.064223 | 0.8396 | 0.94945 |
| valid | 90 | 970  | 0.081086 | 0.9021 | 1.32312 |

### `xgb`

| split | h | n | mean pinball | cover_80 | winkler_80 |
|-------|---|----|--------------|----------|------------|
| test  | 1  | 765  | 0.004250 | 0.7660 | 0.06301 |
| test  | 7  | 895  | 0.018761 | 0.8313 | 0.26091 |
| test  | 30 | 505  | 0.057115 | 0.5030 | 0.89056 |
| test  | 90 | 571  | 0.124653 | 0.1068 | 2.25987 |
| valid | 1  | 982  | 0.005224 | 0.7057 | 0.07755 |
| valid | 7  | 1190 | 0.022310 | 0.7782 | 0.32362 |
| valid | 30 | 717  | 0.065079 | 0.3780 | 1.09738 |
| valid | 90 | 970  | 0.096683 | 0.3515 | 1.66180 |

### `rw_emp`

| split | h | n | mean pinball | cover_80 | winkler_80 |
|-------|---|----|--------------|----------|------------|
| test  | 1  | 765  | 0.005947 | 0.8693 | 0.08941 |
| test  | 7  | 895  | 0.020963 | 0.8972 | 0.31046 |
| test  | 30 | 505  | 0.049652 | 0.9307 | 0.78623 |
| test  | 90 | 571  | 0.072983 | 0.9842 | 1.23611 |
| valid | 1  | 982  | 0.007530 | 0.7709 | 0.10906 |
| valid | 7  | 1190 | 0.026737 | 0.7866 | 0.38183 |
| valid | 30 | 717  | 0.063392 | 0.8173 | 0.92095 |
| valid | 90 | 970  | 0.081136 | 0.8918 | 1.32293 |

### `rw` and `lgbm`

Neither `rw` nor `lgbm` is present in `baseline_metrics_common.csv` (they were
only scored on the full-support table `baseline_metrics.csv`, i.e. *before* the
common-support restriction), so they have no common-support numbers. Their
frozen full-table `POOLED` numbers, for reference only, were:

| model | split | h | n | mean pinball | cover_80 | winkler_80 |
|-------|-------|---|----|--------------|----------|------------|
| rw    | test  | 1  | 950  | 0.007901 | 0.8821 | 0.13773 |
| rw    | test  | 7  | 1098 | 0.025752 | 0.8843 | 0.40246 |
| rw    | test  | 30 | 615  | 0.058127 | 0.8992 | 0.91626 |
| rw    | test  | 90 | 674  | 0.079493 | 0.9837 | 1.29670 |
| rw    | valid | 1  | 1242 | 0.009282 | 0.8180 | 0.14918 |
| rw    | valid | 7  | 1506 | 0.031318 | 0.8068 | 0.46813 |
| rw    | valid | 30 | 909  | 0.076906 | 0.7932 | 1.16082 |
| rw    | valid | 90 | 1222 | 0.093456 | 0.8224 | 1.44503 |
| lgbm  | test  | 1  | 950  | 0.004765 | 0.6853 | 0.07219 |
| lgbm  | test  | 7  | 1098 | 0.021039 | 0.8106 | 0.30164 |
| lgbm  | test  | 30 | 615  | 0.062711 | 0.4472 | 1.08242 |
| lgbm  | test  | 90 | 674  | 0.093404 | 0.3160 | 1.70744 |
| lgbm  | valid | 1  | 1242 | 0.005916 | 0.6876 | 0.08866 |
| lgbm  | valid | 7  | 1506 | 0.025934 | 0.7663 | 0.38722 |
| lgbm  | valid | 30 | 909  | 0.076271 | 0.5193 | 1.30348 |
| lgbm  | valid | 90 | 1222 | 0.087986 | 0.6367 | 1.37047 |

## Implementing code

- `rw` — `src/ml/baselines.py`: `fit_rw_sigmas`, `predict_rw`, and the
  `ZSCORES` constant. Also referenced by `src/ml/interval_diagnostics.py` and
  the Streamlit runner.
- `rw_emp` — `src/ml/baselines.py`: `fit_rw_emp`, `predict_rw_emp`,
  `_rw_return_quantiles`, and the `fit_rw_empirical` / `predict_rw_empirical`
  aliases. Referenced by `src/ml/interval_diagnostics.py`,
  `src/ml/main_table.py` (skill-vs-`rw_emp` column) and
  `tests/ml/test_lstm_targets.py`.
- `ar1` — `src/ml/baselines.py`: `fit_ar1`, `predict_ar1`.
- `xgb` — `src/ml/models/xgb.py` (`XGBModel`, `XGB_PARAMS`), the training/export
  path `src/ml/export.py`, the inference wrapper `src/ml/inference.py`
  (`FreightPredictor`, later rewired to the LSTM path), and the SHAP feature
  selection `src/ml/feature_selection.py`. Wired into the harness via
  `src/ml/baselines.py::main`.
- `lgbm` — `src/ml/models/lgbm.py` (`LGBMModel`, `LGB_PARAMS`). Wired into the
  harness via `src/ml/baselines.py::main`.

All of the above were deleted except the shared harness plumbing, which now
serves only the LSTM ensemble and the calibration layers (`src/ml/baselines.py`,
`src/ml/predictions.py`, `src/ml/conformal.py`, `src/ml/fuzzy_calibration.py`,
`src/ml/interval_diagnostics.py`, `src/ml/main_table.py`, `src/ml/inference.py`).

## Remaining references (resolved)

The follow-ups listed in the original removal pass have since been cleared:

- `docs/03_model_guide.md` now describes the neural + conformal pipeline and
  its reference table lists `lstm`, `lstm_fuzzy`, `cqr`, `aci` only.
- `docs/04_fuzzy_lstm_report.md` cites only fuzzy/neural models.
- `ui/README.md` and `ui/views/playground.py` now label the live tab as neural
  inference and no longer reference the removed export path.
- Regenerated artifacts (`baseline_metrics.csv`, `baseline_metrics_common.csv`,
  `main_table.csv`/`main_table.md`, `baseline_dm.csv`, `fuzzy_ablations.csv`,
  `interval_diagnostics*.csv`, `naive_sign_metrics.csv`) contain no removed-model
  rows. `src/data/main_run/` and `src/data/logs/` are historical run archives.
- `scratch_refactor.py` remains a tracked scratch file with dead references by
  design.

Also note: `src/ml/inference.py::FreightPredictor` now serves the cached
`train`/`valid` LSTM ensemble, so calling `predict()` on the raw frozen test
split returns no rows; a test-split backtest would need the cached test
predictions instead.
