# Agent handoff — finish the fuzzy-LSTM paper and clean up the code

Repo: `C:/Users/arkma/Documents/GitHub/SC-Project`. Everything is an early draft.

## What the project is

Quantile forecasting of four dry-bulk **vessel-class indices** (Capesize, Panamax,
Supramax, Handysize) at horizons h ∈ {1, 7, 30, 90}. Target = P10/P50/P90 of the
future log index level. The method must be **fuzzy + neural**.

- Never call the targets "BDI". They are class indices; the BDI is only a
  cross-series input.
- The paper must discuss **only fuzzy / neural** models in its narrative. Tabular
  and statistical forecasting baselines (xgb, lgbm, ar1, rw, rw_emp) have been
  removed and must not return.
- The paper is `docs/04_fuzzy_lstm_report.md`. It must read like a finished paper:
  no code references, no file paths, no "Task A/B" language.

## Evaluation protocol (fixed, accepted)

Single chronological split, base trained once, calibration fit on valid, test
read once:

| split | dates |
|---|---|
| train | 2012-07-04 → 2022-07-25 |
| valid | 2023-01-03 → 2024-07-25 |
| test  | 2025-01-02 → 2026-08-20 |

~5-month embargo gaps between splits. **Single-episode is accepted for now** —
do not build a rolling-origin backtest. Test is read only through
`ml.frozen_test` (`allow_test_set_access` / `load_frozen_test`); never set
`BASELINES_INCLUDE_TEST=1` for tuning.

## State of the repo (already done — do not redo)

- `docs/05_removed_baselines.md` — removed xgb/lgbm/ar1/rw/rw_emp and their files
  (`models/xgb.py`, `models/lgbm.py`, `export.py`, `feature_selection.py`).
  `inference.py` now serves the LSTM + fuzzy path. `MODELS = (lstm, lstm_fuzzy,
  cqr, aci)`.
- `docs/06_library_swaps.md` — `scoringrules` (metrics), `arch` (bootstrap),
  `statsmodels` (HAC) swapped in; MAPIE skipped for a documented parity reason.
- `tests/ml` passes (18 tests). `src/ml tests/ml` are ruff-clean. `ruff check src
  tests` still fails with 43 **pre-existing** errors in `opt`/`data_prep`/
  `fetchers`/`tests/optimizer` — out of scope, leave them.
- Target architecture for the paper (keep it this simple):
  1. **Forecast:** one pooled quantile LSTM with a monotone head → P10/P50/P90.
  2. **Calibrate:** one Sugeno FIS, inputs `(bias, vol)` → outputs
     `(location shift s, width multiplier m)`.

## Code problems to fix

1. **Dependencies not durable.** The new libs were installed with `uv pip
   install`, not recorded in `pyproject.toml`/`uv.lock`. Add `scoringrules`,
   `arch`, `statsmodels` (and either add or drop `mapie`) via `uv add` / re-lock.
2. **Calendar-day horizons.** `datasets/build_samples.py` defines `y_step_h{h}`
   as a *calendar-day* shift masked by "is that date a trading date". Each
   horizon therefore keeps a different weekday-biased subset (h=30 loses ~2/5 of
   rows, which is why its test mask 505 < h=90's 571). Either redefine horizons
   in trading days, or at minimum report the weekday composition and state the
   limitation.
3. **Matured window.** `fuzzy_calibration.py` uses `N_MATURED=60` /
   `MIN_MATURED=5` at every horizon. At h=90 residuals overlap ~90 days, so the
   window holds ~1 independent observation. Make N horizon-aware and say so.
4. **`novol` ablation mislabeled.** At h≥30 the width system is vol-only and
   `with_vol=False` does not remove vol from the width input. Fix the arm logic
   or rename the arm.
5. **Unscaled bounds.** `M_SCALE=0.7`, `S_SCALE=0.15` are constant across
   horizons (0.15 is 2.2× the h=1 target SD but only 0.28× the h=90 SD). Scale
   per horizon and report the effect — this tests whether the "horizon-dependent
   benefit" is a bounds artifact.
6. **Single calibrator seed.** `fit_fuzzy(seed=0)`. Run ≥5 seeds and report the
   spread.
7. **Early-valid inputs.** The first valid rows' `cov_gap`/`bias_z` are computed
   from in-sample train base residuals. Add a burn-in or document it.
8. **ACI at h=90 under-covers** (0.694). Report its step/bounds; fix only as a
   baseline correction, never by tuning on test.
9. **Statistics.** DM HAC at lag h−1 and the row-level bootstrap ignore
   cross-class same-date correlation and overlap. Replace with a bootstrap that
   resamples whole **dates**; demote DM p-values to an appendix and report
   effect sizes with the effective-sample caveat.
10. **Regenerate** `baseline_dm.csv` (HAC changed h>1 slightly) and any generated
    CSV that still lists removed models.

## Paper problems to fix (`docs/04_fuzzy_lstm_report.md`)

- Rewrite in finished-paper prose, no code/file references. Structure: problem &
  setup → related work → method → diagnostic → results → limitations.
- **Arithmetic / text-vs-table:** every percentage must be computed from the
  actual table. No claim of an unsupported improvement.
- Do **not** say the base is "close to nominal": base coverage is 0.908 (h=7) and
  0.939 (h=30) — substantial overcoverage.
- At h=7 the fuzzy layer's coverage equals ACI (0.808) but its pinball is ~14%
  worse, so its failure there is **not** miscalibration. Diagnose it (width
  adaptation vs the location shift) and say so.
- Explain the h=1 narrowing: valid base undercovered (0.766) yet on test the
  fuzzy layer pushed coverage down 0.837 → 0.741.
- Do not assert "readable rules" without showing them: include a table of learned
  membership-function centres and rule consequents.
- Drop the "width-only is the safer default" recommendation if it comes from
  ablations run after the test split was scored; label all such numbers post-hoc.
- Include a per-class h=90 table (the gain holds in all four classes; show it).
- Add a data-source paragraph (indices, congestion, macro), and the calibrator
  optimizer / learning rate / epochs / initialisation.
- State the key limitations plainly: single episode; no seed variance for the
  calibrator before fix #6; effective sample at h=90; the valid→test up-rate
  shift (0.48→0.54, 0.51→0.58, 0.50→0.60, 0.54→0.65) as a confound for the h=90
  location gain.
- Related work to keep: Su/Park/Bae 2024 (CNN-BiLSTM-AM point forecast), 2025
  ANFIS-on-BDI, Duru 2010 E-FILF, Romano et al. 2019 (CQR), Gibbs & Candès 2021
  (ACI). Add a **neural** state-dependent calibrator baseline (linear/MLP on the
  same inputs) so the fuzzy structure is shown to matter. Fix the placeholder
  reference.

## Rules

- Report numbers exactly; claim nothing the DM/bootstrap (or effect size) does
  not support.
- Do not tune anything on the test split.
- At the end, list every file changed and every assumption made.
