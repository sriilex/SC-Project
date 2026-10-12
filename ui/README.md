# Sugeno Fuzzy Calibration — Interactive UI

A Streamlit app for the paper *“Sugeno Fuzzy Calibration of Quantile Forecasts
for Dry Bulk Vessel-Class Indices”*. It is a read-and-run console over the
project's own `src/ml` pipeline: it never re-implements the model, the metrics,
or the fuzzy inference — it imports them, so what you see matches the paper.

## Run

From the repository root:

```bash
uv run streamlit run ui/app.py
```

or, with the environment already active:

```bash
.venv/Scripts/python -m streamlit run ui/app.py    # Windows
.venv/bin/python -m streamlit run ui/app.py        # Linux/macOS
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

Dependencies: `streamlit` is already a project dependency; the fuzzy surface plots
use `matplotlib`, which lives in the `dev` dependency group and is installed by
`uv run`. If you run outside `uv`, install both with `uv sync`.

## What is inside

| Page | Purpose |
|---|---|
| **Overview** | Abstract, contributions, method math, headline findings. |
| **Data Explorer** | Vessel-class index levels, log space, correlations, feature drill-down, split ranges. |
| **Base Model** | Train/valid coverage, the quantile fan on any date range, seed spread, crossing diagnostics. |
| **Results & Significance** | Main table, metric bars, DM forest plots with bootstrap intervals, ablation heatmap, robustness. |
| **Fuzzy Rule Lab** | Learn/see/edit the Sugeno systems; membership curves and rule surfaces; a what-if scanner for a single forecast; live re-scoring of a whole split. |
| **Run Pipeline** | Stream the project's CLI steps, and refit the fuzzy layer in-process with custom hyperparameters. |
| **Playground** | Live neural inference on a chosen date/class, a manual fuzzy calculator, and a baseline ↔ calibrator comparison. |

## Data and the frozen-test rule

The app reads the artifacts the pipeline already produced
(`src/data/baseline_*.csv`, `fuzzy_ablations.csv`, `robustness_metrics.csv`,
`interval_diagnostics.csv`, cached LSTM predictions under `src/data/preds/`,
and the learned JSON parameters under `src/data/fuzzy_params/`).

The **frozen test split** is treated as the project requires. The *Results*
page only shows the already-sanctioned `test` rows in the precomputed CSVs.
The *Fuzzy Rule Lab* scores the **validation** split by default; scoring the
test split requires an explicit confirmation and is disabled as soon as you
edit any parameter, so the UI cannot be used to tune on test.

## Layout

```
ui/
├── app.py                 # entry point + navigation
├── .streamlit/config.toml # theme
├── lib/                   # shared helpers (paths, loaders, metrics, fuzzy, viz, runner)
└── views/                 # one module per page
```
