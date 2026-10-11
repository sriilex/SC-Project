# Data quality notes

## Negative Baltic Capesize index in 2020 (44 train rows)

`BC_INDEX` (Baltic Capesize) is negative for 44 train rows between
2020-01-31 and 2020-05-14, the only negative raw values in the dataset. The
pipeline computes `log_value = log(target_value)`, so those rows get
`log_value = NaN`; every h-step target derived from them is also undefined.

### Recommendation: drop the 44 rows

Reasons:

1. **The target is a log level and a log return.** `log_value` is the model
   input and `y_step_h{h} - log_value` is the training target. Neither is
   defined for a non-positive level, so no meaningful return can be formed for
   this episode.
2. **A different transform would change the whole series, not just these rows.**
   `asinh(x)` or `sign(x) * log1p(|x|)` are defined for negatives, but they are
   not equal to `log(x)` for the remaining 99.6% of rows, so every feature,
   target and previously reported number would have to be recomputed. The
   negative episode is not a reason to redefine the dependent variable for the
   entire sample.
3. **The affected period is a known structural break.** The Baltic Capesize
   index briefly printed negative during the COVID-19 demand collapse, a regime
   that is not representative of the rest of the train/valid/test period. Its 44
   rows are 0.4% of Capesize train rows.
4. **It is cheap and transparent.** Dropping is exactly what the data loader
   already does: `load_split` normalises `NaN` to null and drops rows with null
   `log_value`. No imputation is introduced and no artificial value is invented.

The 44 rows are dropped from all reported results. If a negative-capable
transform is ever adopted, it must be applied to the raw level before any
feature construction, all samples must be rebuilt, and all results rerun and
reported separately.

## Congestion publication lag

PortWatch publication timing is undocumented. `CongestionFeature` accepts
`lag_days`; the robustness run sets `CONGESTION_LAG_DAYS=7` and rebuilds the
samples, and the resulting metric changes are reported separately from the
main table.

## Macro trailing change

`macro_*_chg_90d` is now the direct level difference `level(t) - level(t-90)`
computed on the continuous daily grid, replacing the percentage change that was
computed on each series' own bounded grid. This only takes effect after
`uv run build-samples`; existing samples retain the old definition.
