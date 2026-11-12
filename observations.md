# Observations — validation-only Sugeno ablation (round 2)

Working notes appended while executing `TODO.md`. Repo state at start: the
previous round had already frozen run `v1` (zero-order symmetric Sugeno), the
attribution baselines, the ACI fix, and the rule-artifact audit.

## What was built

- `src/ml/fuzzy_calibration.py`
  - `fit_fuzzy` now accepts `linear` (first-order Sugeno, per-rule linear
    consequents), `split_width` (two width outputs, lower/upper), `weight_decay`,
    `early_stopping_patience` + `val_fraction` (chronological holdout on the last
    20% of fit dates, restore best validation pinball), and `class_shrinkage`
    (per-class constant location/width offsets with an L2 penalty toward zero).
  - `fis_forward` is now multi-output and numpy-inference supports first-order
    consequents. Ordering is preserved by construction in all cases.
  - All new options default to off, so the v1 arm is unchanged.
- `src/ml/calibration_ablation.py` (new): validation-only ablation runner.
  Ten arms, fit on `valid` with frozen LSTM predictions, scored on `valid`;
  selection is lowest mean pooled validation pinball (tie: Winkler). The frozen
  test split is read only after selection and only for the selected arm.
  Writes `src/data/runs/v2/` (valid table, per-arm params, bootstrap-vs-base,
  selection JSON, one-shot test table).
- `src/ml/paper_tables.py`: added Table 5 (validation ablation), Table 5b
  (validation h=90 per class), Table 6 (frozen test, selected vs frozen base and
  simple calibrators); Table 1/2 now include the selected arm.
- `docs/paper.md`: abstract, method, ablation protocol, results, discussion,
  limitations and plot descriptions updated. Appendix A relabelled as the
  zero-order reference fit.
- `tests/ml/test_fuzzy_calibration.py`: four new tests (first-order, split
  width, class shrinkage, early stopping + weight decay).

## Selection result (validation only, frozen LSTM key 69e158d2977b)

Selected arm: **`sugeno_lin_split`** = first-order consequents + separate
lower/upper width multipliers.

Mean pooled validation pinball (mean over horizons):

| arm | mean pinball |
|---|---:|
| base | 0.04158 |
| constant width + location | 0.03922 |
| linear state-dependent | 0.03693 |
| Sugeno zero-order | 0.03381 |
| Sugeno first-order | 0.03230 |
| Sugeno split width | 0.03371 |
| **Sugeno first-order + split width** | **0.03201** |
| Sugeno + class shrinkage (0.5) | 0.03366 |
| Sugeno + early stop / weight decay | 0.03608 |
| Sugeno full combination | 0.03569 |

Per horizon (pooled validation pinball, base → selected):
h=1 0.00770 → 0.00740 (-4.0%); h=7 0.02610 → 0.02429 (-6.9%);
h=30 0.06065 → 0.04853 (-20.0%); h=90 0.07188 → 0.04781 (-33.5%).
Selected validation coverage: 0.755 / 0.792 / 0.805 / 0.861.

Notes from the ablation:
- First-order consequents are the largest single gain over zero-order.
- Split width alone helps modestly and fixes the h=90 coverage collapse.
- Per-class offsets shrink to almost zero at penalty 0.5; they do not help.
- Early stopping + weight decay is the worst Sugeno arm. The chronological
  holdout is the last 20% of valid dates; at long horizons that window has very
  few matured outcomes, so it stops too early. Negative result for this holdout
  design, not for regularisation in general.

## One-shot frozen test (selected arm only)

Pooled: pinball 0.00774 / 0.02561 / 0.05138 / 0.05957; coverage 0.749 / 0.793 /
0.709 / 0.793; Winkler 0.1255 / 0.4018 / 0.7282 / 0.8001.

- vs base: 8.3% worse (h=1), 16.4% worse (h=7), 6.3% worse (h=30), 13.8% better
  (h=90). Winkler better only at h=30 and h=90; coverage much better at h=90
  (0.793 vs 0.991).
- vs the previously frozen simple calibrators on test: the selected arm is worse
  at every horizon (best simple: 0.00707 / 0.02152 / 0.04464 / 0.05207).
- vs the previous zero-order Sugeno on test: selected is slightly worse at h=1,
  7, 30 and slightly better at h=90 (0.05957 vs 0.06080), with much better h=90
  coverage (0.793 vs 0.730).
- h=90 per class (selected): Capesize 0.0968 / Panamax 0.0462 / Supramax 0.0445
  / Handysize 0.0457. Beats base in three classes, loses in Handysize; coverage
  is 0.785–0.796 in every class (zero-order had Handysize 0.531).

## Honest read

The validation ablation gives the intended positive result: the enriched Sugeno
(first-order + split width) beats the base, the constant calibrator, and the
linear state-dependent calibrator at every validation horizon. It does **not**
transfer to the single test episode. The selection procedure itself was honest
(test read once, after selection), so the discrepancy is a real finding about a
one-year calibration window and a regime shift, not a leak. The paper now says
this plainly rather than claiming a test win.

## Checks

- `uv run ruff check src/ml tests/ml` → all checks passed.
- `uv run pytest tests/ml -q` → 26 passed.
- Repo-wide `ruff check src tests` still has the pre-existing errors in
  `opt`/`data_prep`/`fetchers`/`tests/optimizer` noted in the earlier handoff;
  none are in the files touched here.

## Numerical reproducibility note (after the first v2 run)

Refactoring `fit_fuzzy` to support multi-output/multi-consequent FIS initially
changed the last-bit float order inside the training loop (two `exp` nodes from
one shared multiplier, and a post-step rather than pre-step final loss). Over
600 Adam epochs that last-bit difference grew into a visibly different zero-order
fit (~0.15% relative on validation pinball). The shared-`exp` order was restored,
so the default zero-order arm now reproduces the frozen v1 parameters exactly;
the ten-arm v2 run was regenerated afterwards. The selected arm's numbers did not
change; the zero-order/first-order/class-offset validation means moved in the
third decimal, and the tables above already reflect the regenerated run.
