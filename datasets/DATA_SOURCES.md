# Candidate Data Sources — Energy Consumption Prediction (India focus)

Reviewed sources for training a deep learning model on energy consumption.
The current working dataset is #1; the rest are alternatives / complements.

## Indian datasets

### 1. Grid-India national demand + generation mix (CURRENT)
- **Link:** https://data.mendeley.com/datasets/y58jknpgs8/2
- **What:** All-India electricity demand, thermal/hydro/gas/nuclear/wind/solar
  at 5-minute resolution, Sep 2021 - Jun 2025 (~240k+ rows). Hosted on
  Mendeley Data, CC BY 4.0. Already downloaded to `datasets/india_grid_demand/`.
- **Caveat:** Generation-mix columns are components of demand itself
  (target leakage if used as features); usable target + lags only.

### 2. IEX active power load dataset (load + market prices)
- **Link:** https://data.mendeley.com/datasets/jxm8d4w4cv/1
- **What:** 23,352 hourly load samples from the Indian Energy Exchange,
  Jan 2021 - Aug 2023, with electricity prices (INR/MWh), lagged-load feature
  layout. CC BY 4.0.
- **Caveat:** Pre-featurized (lags already baked in); smaller.

### 3. Kaggle: Hourly Load India — Electrical Load Forecasting
- **Link:** https://www.kaggle.com/datasets/shubhamvashisht/hourly-load-india-electrical-load-forecasting
- **What:** Hourly all-India load demand for short-term load forecasting.
- **Caveat:** Requires Kaggle account; provenance less documented than Mendeley.

### 4. Punjab SLDC real-time / SCADA reports
- **Link:** https://www.punjabsldc.org/ (see e.g. `/realtimepbGen.aspx`)
- **What:** State-level 15-minute SCADA data (generation, demand) published by
  the State Load Despatch Centre. Common source in Indian STLF papers.
- **Caveat:** Web-report format, needs scraping; no bulk archive API.

### 5. IIT Bombay residential smart-meter dataset
- **Link:** https://figshare.com/s/a4aedbe2ae2aadc5618b (file 13842431)
- **What:** 60 instrumented apartments in an IIT Bombay high-rise, Dec 2016 -
  Jun 2018, downsampled to 1-hour granularity.
- **Caveat:** Single building; great for building-scale papers.

### 6. CEEW India Residential Energy Survey (IRES) 2020
- **Link:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/U8NYUP
- **What:** ~15,000 households pan-India: consumption, expenditure,
  appliances. Mirror: https://www.kaggle.com/datasets/pythonafroz/electrical-consumption-data-in-indian-homes/data
- **Caveat:** Cross-sectional survey, not a time series — no LSTM use;
  useful for context/regression chapters only.

## International benchmarks (for comparison baselines)

### 7. UCI Individual Household Electric Power Consumption
- **Link:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- **What:** French household, 1-minute resolution, 47 months. The classic
  LSTM forecasting benchmark.

### 8. ASHRAE Great Energy Predictor III (Kaggle)
- **Link:** https://www.kaggle.com/c/ashrae-energy-prediction
- **What:** Building-level hourly meters + weather for 1,000+ buildings.
  Includes weather data, which fixes the "no exogenous features" problem of #1.

## Weather complement (pairs well with #1)
- NASA POWER: https://power.larc.nasa.gov/data-access-viewer/
- Open-Meteo historical API (free, no key): https://open-meteo.com/en/docs/historical-weather-api

## Decision log
- 2026-08-22: Downloaded #1 (`india_grid_demand/`, ~713 MB). Kept as primary
  time-series corpus; weather from Open-Meteo planned as exogenous features.
