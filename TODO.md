# TODO — repo work before the paper is written

Ordered by dependency. `[ ]` = open, `[x]` = done. Do not tune on test;
read `samples_test.parquet` once, at the end, via `ml.frozen_test`.

## 0. Freeze the experimental baseline

- [x] Record which cached run each number came from (`src/data/preds/*_meta.json`
      key + `fuzzy_params/*.json` + config constants).
- [x] Move the current `src/data/*.csv` outputs to `src/data/runs/<name>/` so new
      runs stop overwriting the numbers already quoted in `current paper.txt`.
- [x] Freeze the original configuration as run `v1` (reference), and treat
      window / bound / burn-in changes as separate named variants.
- [x] Make `M_SCALE`, `S_SCALE_PER_SD`, `N_MATURED`, and the burn-in explicit
      config values that appear in saved run metadata.

**Done when:** any reported number traces to a named run directory + config.

## 1. Fix ACI delayed feedback (correctness bug)

- [x] In `src/ml/conformal.py::aci_adjust`, resolve only forecasts whose outcome
      is observable at the current date (maturity queue), not the latest row.
- [x] Expose `gamma`, warm-up and score-buffer length in run metadata.
- [x] Add a regression test: changing an unmatured outcome must not change any
      forecast issued before its maturity date.
- [x] Re-run ACI and record the new numbers; do not silently inherit old ones.

**Evidence:** a synthetic check already shows a single changed h=90 outcome
alters 34 subsequent forecasts before it matures.

## 2. Attribution baselines (the main scientific ask)

Fit on validation, same objective and output bounds as the fuzzy layer:

- [x] constant width multiplier only;
- [x] constant width multiplier + constant location shift;
- [x] constant width + causal rolling-bias correction;
- [x] linear (and optionally small MLP) state-dependent calibrator on the same
      inputs, with the same `tanh` bounds.

- [x] Report mean pinball, **median-loss and Winkler/endpoint loss separately**,
      so gains are not credited only through P50.
- [x] State the selection/tuning budget for each baseline.

**Done when:** the paper can answer "does the fuzzy structure beat global
correction and ordinary state-dependent regression?"

## 3. Uncertainty and statistics

- [x] Replace independent date bootstrap with a **block bootstrap over
      consecutive dates**, keeping all four classes together in each block.
- [x] Report sensitivity to block length.
- [x] Keep DM tests secondary; do not lead with `p < 0.001` on pooled
      overlapping rows.
- [x] Remove the "lowest pinball in all four classes" claim unless a per-class
      table supports it (the v1 h=90 Handysize result does not).

## 4. Rule interpretability artifacts

- [x] Export membership centres/widths and consequent matrices as a paper table
      from the saved JSON params.
- [x] Plot width and location response surfaces, and activation weights for
      representative forecasts.
- [x] Check whether learned terms are actually distinguishable (the v1 h=90
      width centres are nearly identical); report honestly either way.

## 5. Reproduce the stacked arm

`src/data/fuzzy_cqr_stack.csv` has promising numbers but **no runner exists**.

- [x] Add a dedicated, reproducible fuzzy-on-CQR experiment script.
- [x] Specify the chronological fit protocol (valid fit, test frozen).
- [x] Regenerate the stack table + per-class table from the script.
- [x] State explicitly that the fuzzy stage does not inherit CQR's coverage
      guarantee.

## 6. Table and number generation

- [x] One script that emits all paper tables from run directories.
- [x] Compute every percentage in code; no hand-typed deltas.
      (v1 text says "13.9% below CQR"; the table implies 12.9%.)
- [x] Generate the per-class h=90 table.
- [ ] Fix references: ref 3 authors, ref 2 volume/pages.
- [x] Add data-source paragraph (indices, PortWatch congestion, macro series
      from `raw_data/sources.md`).
- [x] Add calibrator optimizer / LR / epochs / init and code-availability note.

## 7. Paper-supporting method text (can start immediately, parallel)

Not blocked by any experiment above:

- [x] Method section: causal information flow, matured windows, bounded
      location/width, zero-order Takagi–Sugeno terminology.
- [x] Structural propositions: ordering preservation, identity at zero
      consequents, positive bounded width scaling, bounded location shift.
- [x] Related work: EnbPI, AgACI, Barber et al. 2023, Kuleshov et al. 2018.
- [x] Protocol section: splits, masks, maturity, leakage guard.
- [x] One concise limitations section (single episode, small effective sample,
      calendar-day horizons).

## 8. Claims to drop or rewrite before submission

- [x] "Statistically indistinguishable from ACI" → "inconclusive at this
      sample size".
- [x] "Width-only is the safer default" from post-hoc ablation → mark as
      exploratory or remove.
- [x] Repetitive short-horizon failure narration → one technical observation
      on calibration strength vs. horizon.
- [x] "Readable rules" → only after §4 artifacts exist.

---

## Suggested split

**Agent A (repo/evidence, blocks results writing):** §1 → §2 → §3 → §5 → §6.
**Agent B (paper prose, no new numbers):** §7, plus §0/§8 housekeeping.

Handoff point: do not write Results, Discussion or the Abstract until §1–§3
and §6 are done. Everything else can proceed in parallel.
