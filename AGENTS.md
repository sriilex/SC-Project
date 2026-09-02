## Current State
- Everything in this repository is currently in an early draft state.
- Nothing is formalized, and all code, structure, and architecture is subject to change. Don't worry about it.
- The project is highly experimental at this phase.

## Pipeline Architecture & Data Principles
- **Continuous Features, Discrete Evaluation:** The ML data pipeline uses a Master 365-day continuous calendar. Missing data (weekends, holidays) is imputed using Forward-Fill (LOCF) to preserve time-series logic (lags, rolling means). 
- **Training Mask:** When generating ML training samples, weekend and non-trading holiday rows must be dropped *after* targets ($y_{t+h}$) are shifted, to ensure we only train and evaluate on real market days.

## Folder Layout
- `docs/`: Project documentation.
- `raw_data/`: Raw source data (not processed) and reference materials.
- `src/config/`: Configuration files (e.g., sample generation settings).
- `src/fetchers/`: The ingestion layer. Scripts that hit APIs/scrapers and dump to `raw_data/`.
- `src/data_prep/`: The foundation layer. Builds the 365-day master tables with LOCF.
- `src/data/`: Processed parquets, continuous master tables, and intermediate outputs.
- `src/ml/`: The machine learning layer.
  - `features/`: Feature engineering logic applied to the continuous 365-day tables.
  - `datasets/`: Scripts that shift targets, drop weekends/holidays, and build final `train.parquet`.
  - `models/`: PyTorch/Tree models and training loops.
- `src/opt/`: Optimizer and operations research models (operates on daily/hourly granularity).

## Python Style Guide
- Use Polars over Pandas.
- Use PyTorch over TensorFlow.
- Use `uv` instead of `pip` as package manager, and runner.