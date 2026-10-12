# 06 — Library swaps in the evaluation layer

Replaces hand-rolled scoring / inference code with standard libraries, keeping
the public signatures (`ml.baselines.evaluate`, `ml.baselines.pinball`,
`ml.evaluation.aligned_losses`, `ml.evaluation.newey_west_se`,
`ml.evaluation.dm_test`, `ml.evaluation.block_bootstrap_ci`) unchanged.

| Swap | Library (version) | Status |
|------|-------------------|--------|
| 1. Pinball + Winkler | `scoringrules` 0.11.0 | applied |
| 2. CQR / ACI conformal | `mapie` 1.5.0 | **skipped** (API does not fit pre-computed predictions) |
| 3a. Moving-block bootstrap | `arch` 8.0.0 | applied |
| 3b. Newey-West HAC | `statsmodels` 0.15.0 | applied |

## Environment

`scoringrules`, `arch` and `statsmodels` are recorded as project dependencies
with `uv add scoringrules arch statsmodels`, which updates both `pyproject.toml`
and `uv.lock`; `uv sync` therefore keeps them. `mapie` was **not** added: it is
skipped for the API/parity reasons below. Re-locking also pruned packages that
had already been removed from the project (`lightgbm`, `xgboost`,
`nvidia-nccl-cu13`).

## Swap 1 — `scoringrules` 0.11.0 (quantile + interval score)

Replaced functions
(`quantile_score(obs, fct, alpha)`, `interval_score(obs, lower, upper, alpha)`):

| Site | Replaced hand-rolled code | Library call |
|------|---------------------------|--------------|
| `src/ml/baselines.py::pinball` | 3 lines: `q * err` / `(q - 1) * err` via a `pl.when().then().otherwise()` select | `sr.quantile_score` |
| `src/ml/baselines.py::evaluate` | 3 lines: `below = (p10 - y).clip(...)`, `above = (y - p90).clip(...)`, `winkler = width + 10.0 * (below + above).mean()` | `sr.interval_score(y, p10, p90, INTERVAL_ALPHA)` |
| `src/ml/evaluation.py::aligned_losses._losses` | 4 lines of `np.maximum(q * err, (q - 1) * err)` per quantile | `sr.quantile_score` per quantile |

The magic `10.0` penalty multiplier is gone; the central interval's miscoverage
level now has a name (`INTERVAL_ALPHA = 0.2` in `src/ml/baselines.py`).

Parity (no convention change — the two formulas are algebraically identical):
`tests/ml/test_baselines.py::test_pinball_matches_scoringrules_and_legacy_formula`
and `::test_evaluate_winkler_matches_scoringrules_interval_score` assert
`abs=1e-12` against both the inline legacy formula and the library call;
`tests/ml/test_evaluation.py::test_aligned_losses_use_scoringrules_quantile_score`
does the same for the per-row losses.

Cross-check against the pre-swap code from git HEAD on the real `valid` split
(cached LSTM predictions, h ∈ {1, 7, 30, 90}, pooled + 4 classes): every numeric
column of `evaluate` — including `pinball_*`, `pinball_mean` and `winkler_80` —
agrees to ≤ 6.7e-16; `pinball` and `aligned_losses` agree **bit for bit** (max
diff 0.0, worst 3.5e-18).

Lines: 6 hand-rolled metric lines removed, 3 library lines added by this swap
in `src/ml/baselines.py` (my edit hunks there are +9 / −6; the rest is the 2 new
imports and the named constant — the file also carries unrelated pre-existing
working-tree edits); 4 hand-rolled lines removed, 2 library lines added in
`src/ml/evaluation.py`.

## Swap 2 — `mapie` 1.5.0 (CQR / ACI): **skipped**

`src/ml/conformal.py` (`fit_cqr`, `apply_cqr`, `aci_adjust`) was left unchanged.
MAPIE does not fit this data shape:

1. **No estimator, no features.** `ConformalizedQuantileRegressor(prefit=True)`
   requires a list of three *fitted sklearn quantile regressors* for
   `(1-α)/2`, `(1+α)/2`, `0.5`, and both
   `conformalize(X_conformalize, y_conformalize)` and `predict_interval(X)`
   need a 2-D feature matrix. Our inputs are pre-computed quantile *frames*
   (`date`, `target_class`, `p_0.1/0.5/0.9`) — no model, no `X`. Using MAPIE
   would mean writing lookup-table sklearn shims plus a synthetic `X`, i.e.
   forcing it.
2. **Different conformity-score level, so no parity is possible.**
   MAPIE uses the finite-sample corrected level
   `((n + 1) * (1 - alpha)) / n` with `method="higher"`
   (`mapie/utils.py::_compute_regression_quantile`); `fit_cqr` uses the plain
   empirical `quantile(1 - alpha)`. On the real calibration frame
   (valid split, h = 7, n = 1190) that moves the CQR widening from
   `e_q = -0.0080337` to `-0.0079020`, a 1.6 % change — the frozen `cqr`
   baseline row and every stored metric would shift, not just be re-implemented.
3. **Extra outputs MAPIE does not expose.** `fit_cqr` also returns the median
   residual shift (`med_q`) and the asymmetric variant (`lo_q`, `hi_q`), and
   `_order` (P10 ≤ P50 ≤ P90) is enforced afterwards; MAPIE's
   `symmetric_correction` flag covers only the first of these.
4. **ACI is not available for a prefit frame.** MAPIE's adaptive conformal
   inference lives in `TimeSeriesRegressor(estimator, method="aci", cv=...)`
   and is driven by `update(X, y, gamma=...)`, which re-runs an underlying
   estimator on features. `aci_adjust` instead tracks `alpha_t` and a rolling
   buffer of the last 1000 conformity scores *per vessel class* on frozen
   quantiles, warm-started across splits — no estimator is involved at all.

The current implementation is documented in `src/ml/conformal.py` and stays.

## Swap 3a — `arch` 8.0.0 (`MovingBlockBootstrap`)

`src/ml/evaluation.py::block_bootstrap_ci` (its callers `dm_test` and
`write_markdown_summary` are unchanged): the 7-line resampling loop
(`np.random.default_rng` + `rng.integers` + `np.concatenate` per replicate) is
replaced by
`MovingBlockBootstrap(block, d, seed=seed).apply(lambda x: np.mean(x), reps=n_boot)`.

The schemes are identical — `arch` draws `ceil(T / block)` block start indices
uniformly on `[0, T - block]`, concatenates the blocks and truncates to `T`, and
seeds `np.random.default_rng(seed)` exactly like the removed loop. Parity is
therefore **exact**:
`tests/ml/test_evaluation.py::test_block_bootstrap_ci_matches_legacy_loop`
asserts `abs=1e-12` for `boot_lo`, `boot_hi` and `boot_mean` (measured diff
0.0); on real data (valid split, h ∈ {1, 7, 30, 90}) all three values are
bit-identical to the pre-swap code.

Lines: 7 hand-rolled lines replaced by 2 (−5 net).

## Swap 3b — `statsmodels` 0.15.0 (Newey-West HAC)

`src/ml/evaluation.py::newey_west_se` (used by `dm_test`): the 10-line
autocovariance loop is replaced by the HAC standard error of a constant-only
regression,

```python
OLS(d, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}).bse[0]
```

**One deliberate behaviour change.** The removed loop divided each cross term by
`T - k` (`np.mean(e[k:] * e[:-k])`); `statsmodels` uses the textbook Newey-West
normalisation with `1 / T` for every term. The two estimators are identical at
lag 0 (White) and differ by O(k / T) above it:

| Input | legacy SE | statsmodels SE | rel. diff |
|-------|-----------|----------------|-----------|
| fixed 60-obs series, lag 0 | 0.11586095 | 0.11586095 | 7e-16 |
| fixed 60-obs series, lag 6 | 0.16154507 | 0.16013442 | 0.87 % |
| real valid split, h = 7 (lag 6), DM stat | −2.19768 | −2.19944 | 0.08 % |
| real valid split, h = 30 (lag 29), DM stat | −0.51216 | −0.51524 | 0.60 % |
| real valid split, h = 90 (lag 89), DM stat | −0.53223 | −0.53849 | 1.18 % |

`dm_mean_diff` is unaffected (identical to the last bit); `dm_p` moves by at
most 0.0043 (h = 90); h = 1 uses lag 0 and is unchanged. The legacy code also
returned `NaN` when `lag ≥ T` (mean of an empty slice) — `statsmodels`
truncates the kernel instead, which is the intended behaviour.

Parity:
`tests/ml/test_evaluation.py::test_newey_west_se_matches_legacy_hac` asserts
`abs=1e-12` at lag 0, `rel=0.02` at lag 6 (measured 0.87 %), and an exact match
against the `statsmodels` call itself (so a hand-rolled re-implementation cannot
creep back in). The tolerance encodes the normalisation difference above, not
numerical noise.

Lines: 10 hand-rolled lines replaced by 5 (−5 net). `src/ml/evaluation.py` as a
whole diffs +32 / −27; the three hand-rolled statistics account for 21 removed
lines replaced by 9 library lines, the remainder of the churn being the 3 new
imports and expanded docstrings.

## Verification

```
uv run --no-sync ruff check src/ml tests/ml   # All checks passed
uv run --no-sync pytest tests/ml -q           # 18 passed
```

`uv run --no-sync ruff check src tests` reports 43 findings, **none of them in
the files touched here** (all pre-existing, in `src/opt/*`, `src/fetchers/*`,
`src/data_prep/*`, `tests/optimizer/*`); `src/ml` and `tests/ml` are clean.

## Follow-ups

* `mapie` is intentionally not a dependency; the reasons above are the record
  of that decision.
* `src/data/baseline_dm.csv` / `baseline_summary.md` have been regenerated with
  the current HAC and date-cluster bootstrap; no legacy-HAC rows remain.
* `tests/ml/test_evaluation.py` keeps the two removed implementations inline as
  references for the parity assertions.
