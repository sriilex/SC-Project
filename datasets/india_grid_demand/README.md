# india_grid_demand

All-India electricity demand and generation at **5-minute resolution**,
sourced from Grid-India (National Load Despatch Centre) SCADA/PMU telemetry
via Mendeley Data: DOI `10.17632/y58jknpgs8.2` (CC BY 4.0).

Citation:
Mukherjee, D., Kalita, K., Kumar, S. (2025). *Electricity Demand, Solar and
Wind Generation Data (September 2021 - June 2025) of India at 1-hour interval*.
Mendeley Data, V2. https://doi.org/10.17632/y58jknpgs8.2

## Contents

- One `.xlsx` per month from September 2021 through December 2023
  (`Sheet1` holds the raw 5-minute time series).
- `January 2024- June 2025.xlsx` covers the remaining period (hourly summary;
  to be verified during preprocessing).

## Schema (`Sheet1`, two header rows then data)

| Column | Meaning |
| --- | --- |
| Time | Timestamp, 5-minute steps |
| TOTAL_THM_ONLY\|P | Thermal generation (MW) |
| TOTAL_NLDC_HYD\|P | Hydro generation (MW) |
| TOTAL_NLDC_GAS\|P | Gas generation (MW) |
| TOTAL_NLDC_NPC\|P | Nuclear generation (MW) |
| ALL_INDIA_WIND\|P | Wind generation (MW) |
| ALL_IND_SOLAR\|P | Solar generation (MW) |
| Solar / Solar+Wind / Total | Formula-derived columns |
| NLDC_DEMAND\|P | All-India demand (MW) — primary prediction target |
| INTRA_SYSTEM_INTERTIA\|P | System inertia |

Note: some cells are Excel formulas (`=MAX(...)`, `=SUM(...)`) rather than
stored values; preprocessing must recompute them or use derived columns.

Re-download / refresh anytime with:

```pwsh
uv run datasets\india_grid_demand\download.py
```
