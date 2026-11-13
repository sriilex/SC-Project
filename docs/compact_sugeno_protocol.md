# Compact Sugeno — Simple, Prediction-Focused Design

This is the second, simpler round of the bounded calibration experiment. It
replaces the first design (which leaned on a SciPy linear program and a
fuzzy location system) with something small enough to read in a page.

## The model

A zero-order Sugeno system with **fixed Gaussian memberships** (never
learned) over one normalized input:

```
IF x is Low / Mid / High  THEN  lower multiplier = k_r^lo,  upper multiplier = k_r^hi
                            AND  location shift   = s  (one global shift)
```

- `x` is either the volatility regime (`vol_ratio = z(14d vol) − z(90d vol)`)
  or the matured forecast-error severity (`error_ratio = trailing 80th
  percentile of |y − P50| / half-width`).
- Multipliers act **separately on the lower and upper interval bounds**:
  `L' = P50 + k^lo·(L − P50)`, `U' = P50 + k^hi·(U − P50)`, with
  `k ∈ [0.25, 4]` enforced exactly by the solver.
- `s` is a single log-unit shift applied to all three quantiles.
- Everything else (the base interval, the memberships, the objective) is
  fixed.

### Parameters

| Configuration | Parameters |
|---|---:|
| 1 rule (the Constant W+L baseline) | 2–3 |
| 2 rules | 5–6 |
| 3 rules | 8–9 |

So the fuzzy model is **at most 7 parameters larger** than the constant
baseline it must beat.

## The fitter (no scipy, no LP, no gradient descent)

Objective: **weighted pinball** on matured out-of-sample forecast errors,
plus optional L1 shrinkage of rule differences from their means.

The objective is piecewise linear and convex in each parameter:

- each width multiplier `k` appears in the objective only through the
  breakpoints `±k` of `max(0, y·k − k)`, which is monotone in `k`;
- the location shift `s` appears through the breakpoints `±s` of
  `max(0, y − s)`;
- with everything else fixed, the loss in one parameter is a 1-D convex
  piecewise-linear function.

So the fitter is an **exact coordinate (cyclic) descent**: for each parameter,
it evaluates the loss at the sorted candidate breakpoints plus the bounds and
picks the best. The result is the *exact global minimum* of the convex
subproblem — no scipy, no HiGHS, no `tanh`, no Adam, no local minima. The
cyclic sweep is coordinate exact, so it can only stop at a joint optimum of
this small separable-convex objective.

Optional **recency weighting**: rows get weight `2^(−age_months/h)`, so a
12-month half-life doubles the weight of the most recent year relative to
2020. This is the *same* weighting applied to the constant baseline, so the
comparison stays fair.

## Experiment

- **Data**: the isolated long-calibration split
  (`src/data/runs/long_calibration_v1`): base model trained through
  `2019-07-25`; calibration span `2020-01-02 → 2024-07-25`. **No test data.**
- **Fits**: 1, 2, or 3 rules × {symmetric, asymmetric} × {flat, 12m, 24m
  recency} × {λ ∈ 0, 0.01, 0.1} = 54 candidates per input family
  (volatility, residual), plus 6 one-rule constant variants.
- **Tuning folds** (fit on, score on): `2022-01-01 → 2022-06-30`,
  `2023-01-01 → 2023-06-30` (purged by horizon).
- **Selection**: best tuning-average pinball per input family, tie-breaks by
  fewer parameters then higher λ. Same for the constants.
- **Confirmation** (pre-locked): refit the finalists on all matured rows
  before `2024-04-01`, score on `2024-04-01 → 2024-07-25`.
- **Comparator of record**: the **best constant** (of the 6 variants), not
  just the flat one. If the fuzzy pipeline cannot beat the *best-tuned*
  constant, that is an honest negative result.

### Success rule (predeclared, exploratory)

The nominated fuzzy pipeline beats the best constant on the confirmation
window when:

1. average horizon-relative pinball reduction ≥ 1%,
2. it wins on at least 2 of 4 horizons,
3. no horizon is worse by more than 5%.

This is validation-only evidence. It does not replace the frozen test set.

## Why this is still genuinely fuzzy

Fixed antecedents (Gaussian memberships), product inference, normalized
weighted-average defuzzification, rule-based consequents. What was removed is
only what was unstable: learned membership geometry, nonconvex `tanh`
transforms, a separate fuzzy location system, and an external LP solver.
