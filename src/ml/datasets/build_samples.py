"""Orchestrate the ML sample generation pipeline."""
from __future__ import annotations

import logging
import os
import tomllib
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl

from ml.features import (
    MACRO_SERIES_STEMS,
    CongestionFeature,
    CrossSeriesFeature,
    LagsFeature,
    MacroFeature,
    ReturnsFeature,
    RollingFeature,
    add_calendar_features,
    calendar_feature_names,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def _setup_logging() -> None:
    """Configure root logging once for CLI runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_config(path: Path) -> dict[str, Any]:
    """Load the samples.toml configuration."""
    with path.open("rb") as f:
        return tomllib.load(f)


def main() -> None:
    """Run the sample generation pipeline."""
    _setup_logging()
    
    config_path = REPO_ROOT / "src" / "config" / "samples.toml"
    config = load_config(config_path)
    
    master_path = REPO_ROOT / config["paths"]["master_parquet"]
    LOGGER.info(f"Loading master table from {master_path}")
    
    master_long = pl.read_parquet(master_path)
    
    LOGGER.info("Pivoting to wide format")
    master_wide = master_long.pivot(
        values="value",
        index="date",
        on="series_id"
    ).sort("date")
    
    LOGGER.info("Creating continuous calendar grid and forward-filling missing values")
    min_date = master_wide["date"].min()
    max_date = master_wide["date"].max()
    
    date_grid = pl.DataFrame({"date": pl.date_range(min_date, max_date, "1d", eager=True)})
    
    # Keep track of which days had valid targets before we forward-fill
    valid_flags = [pl.col(c).is_not_null().alias(f"{c}_is_valid") for c in master_wide.columns if c != "date"]
    valid_dates = master_wide.select([pl.col("date")] + valid_flags)
    
    master_wide = date_grid.join(master_wide, on="date", how="left")
    valid_dates = date_grid.join(valid_dates, on="date", how="left").fill_null(False)
    
    # Impute missing values by carrying the last known observation forward
    master_wide = master_wide.fill_null(strategy="forward")
    
    LOGGER.info("Adding calendar features")
    cny_dates = config["calendar"]["cny_dates"]
    master_wide = add_calendar_features(master_wide, cny_dates)
    
    LOGGER.info("Adding congestion features")
    congestion_lag_days = int(
        os.environ.get("CONGESTION_LAG_DAYS", config["features"].get("congestion_lag_days", 0))
    )
    congestion_feat = CongestionFeature(
        origin_ports=config["features"]["ports"]["origin"],
        dest_ports=config["features"]["ports"]["dest"],
        lag_days=congestion_lag_days,
    )
    master_wide = congestion_feat.transform(master_wide)

    # Macro / commodity features (Brent, iron ore, coal, USD/INR, US IP).
    # macro_long.parquet carries real World Bank Pink Sheet / FRED series and is
    # joined as-of (backward) so each row only sees values knowable as of its date.
    macro_features: list[MacroFeature] = []
    macro_path = REPO_ROOT / "src" / "data" / "macro_long.parquet"
    if macro_path.exists():
        LOGGER.info("Adding macro/commodity features")
        macro_long = pl.read_parquet(macro_path)
        loaded_series = set(macro_long["series_id"].unique().to_list())
        for series_id in MACRO_SERIES_STEMS:
            if series_id in loaded_series:
                macro_features.append(MacroFeature(macro_long, series_id))
        for macro_feat in macro_features:
            master_wide = macro_feat.transform(master_wide)
        LOGGER.info(
            f"Added {sum(len(m.feature_names_out()) for m in macro_features)} "
            f"macro features from {len(macro_features)} series"
        )
    else:
        LOGGER.warning(f"Macro series file not found at {macro_path}; skipping macro features")
    
    target_classes = config["targets"]["classes"]
    horizons = config["targets"]["horizons"]["days"]
    lags = config["features"]["lags"]
    windows = config["features"]["windows"]
    returns = config["features"]["returns"]
    bdi_col = config["features"]["bdi_col"]
    
    all_samples: list[pl.DataFrame] = []
    
    for target_col, class_name in target_classes.items():
        if target_col not in master_wide.columns:
            LOGGER.warning(f"Target column {target_col} not found, skipping {class_name}")
            continue
            
        LOGGER.info(f"Processing target: {class_name} ({target_col})")
        
        # Select target on the continuous grid. Do NOT drop nulls yet!
        df_target = master_wide.select(
            "date", 
            pl.lit(class_name).alias("target_class"),
            pl.col(target_col).alias("target_value"),
            valid_dates.get_column(f"{target_col}_is_valid").alias("is_valid_target_date")
        )
        
        df_target = df_target.with_columns(
            pl.col("target_value").log().alias("log_value")
        ).sort("date")
        
        # Calculate horizons on the continuous grid (so h=7 is exactly 7 calendar days)
        target_exprs = []
        target_cols = []
        for h in horizons:
            col_name = f"y_step_h{h}"
            mean_col_name = f"y_mean_h{h}"
            is_target_valid = pl.col("is_valid_target_date").shift(-h)
            
            target_exprs.extend([
                pl.when(is_target_valid).then(pl.col("log_value").shift(-h)).otherwise(None).alias(col_name),
                pl.when(is_target_valid).then(
                    pl.col("target_value").shift(-1).rolling_mean(window_size=h).shift(-(h-1)).log()
                ).otherwise(None).alias(mean_col_name)
            ])
            target_cols.extend([col_name, mean_col_name])
            
        df_target = df_target.with_columns(target_exprs)
        
        # Compute lags, rolling, and returns on the continuous grid
        lags_feat = LagsFeature(lags)
        df_target = lags_feat.transform(df_target)
        
        rolling_feat = RollingFeature(windows)
        df_target = rolling_feat.transform(df_target)
        
        returns_feat = ReturnsFeature(returns)
        df_target = returns_feat.transform(df_target)
        
        other_classes = [c for c in target_classes.keys() if c != target_col]
        cross_feat = CrossSeriesFeature(bdi_col=bdi_col, other_classes=other_classes)
        wide_with_cross = cross_feat.transform(master_wide)
        
        cols_to_join = ["date"]
        cols_to_join.extend(calendar_feature_names())
        cols_to_join.extend(congestion_feat.feature_names_out())
        cols_to_join.extend(cross_feat.feature_names_out())
        
        for macro_feat in ["MACRO_SP500", "MACRO_DXY"]:
            if macro_feat in master_wide.columns:
                cols_to_join.append(macro_feat)

        for mf in macro_features:
            cols_to_join.extend(mf.feature_names_out())
        
        join_df = wide_with_cross.select(cols_to_join)
        df_target = df_target.join(join_df, on="date", how="left")
        
        # Clean up: Only now do we drop days that weren't originally published
        df_target = df_target.filter(pl.col("is_valid_target_date")).drop("is_valid_target_date")
        
        # We don't drop null targets entirely because some horizons might be valid while others are not
        # e.g., t+1 is weekend (null target), but t+7 is a weekday.
        
        all_samples.append(df_target)
        
    if not all_samples:
        LOGGER.error("No samples generated!")
        return
        
    final_df = pl.concat(all_samples, how="diagonal").sort(["date", "target_class"])
    
    valid_start = date.fromisoformat(config["splits"]["valid_start"])
    test_start = date.fromisoformat(config["splits"]["test_start"])
    embargo_days = config["splits"]["embargo_days"]
    
    max_lag = max(lags)
    max_horizon = max(horizons)
    
    assert embargo_days >= max_lag + max_horizon, f"Embargo gap {embargo_days} must be >= {max_lag + max_horizon}"
    
    train_end = valid_start - timedelta(days=embargo_days)
    valid_end = test_start - timedelta(days=embargo_days)
    
    train_df = final_df.filter(pl.col("date") <= train_end)
    valid_df = final_df.filter((pl.col("date") >= valid_start) & (pl.col("date") <= valid_end))
    test_df = final_df.filter(pl.col("date") >= test_start)
    
    out_dir = REPO_ROOT / "src" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = REPO_ROOT / config["paths"]["samples_train"]
    valid_path = REPO_ROOT / config["paths"]["samples_valid"]
    test_path = REPO_ROOT / config["paths"]["samples_test"]
    manifest_path = REPO_ROOT / config["paths"]["samples_manifest"]
    
    train_df.write_parquet(train_path)
    valid_df.write_parquet(valid_path)
    test_df.write_parquet(test_path)
    
    LOGGER.info(f"Wrote train set: {train_path} ({train_df.height} rows)")
    LOGGER.info(f"Wrote valid set: {valid_path} ({valid_df.height} rows)")
    LOGGER.info(f"Wrote test set: {test_path} ({test_df.height} rows)")
    
    train_counts = train_df.group_by("target_class").agg(pl.len().alias("train_rows"))
    valid_counts = valid_df.group_by("target_class").agg(pl.len().alias("valid_rows"))
    test_counts = test_df.group_by("target_class").agg(pl.len().alias("test_rows"))
    
    counts = train_counts.join(valid_counts, on="target_class", how="outer_coalesce")
    counts = counts.join(test_counts, on="target_class", how="outer_coalesce")
    
    LOGGER.info(f"Row counts per class:\n{counts}")
    
    feat_cols = []
    feat_cols.extend(LagsFeature(lags).feature_names_out())
    feat_cols.extend(RollingFeature(windows).feature_names_out())
    feat_cols.extend(ReturnsFeature(returns).feature_names_out())
    feat_cols.extend(calendar_feature_names())
    feat_cols.extend(congestion_feat.feature_names_out())
    for mf in macro_features:
        feat_cols.extend(mf.feature_names_out())
    
    feat_cols.extend(["cross_BDI_log_level", "cross_BDI_ret_1"])
    for cls in target_classes.keys():
        feat_cols.append(f"cross_{cls}_ret_1")
        
    manifest_data = {
        "split": ["train", "valid", "test"],
        "start_date": [
            train_df["date"].min(),
            valid_df["date"].min(),
            test_df["date"].min(),
        ],
        "end_date": [
            train_df["date"].max(),
            valid_df["date"].max(),
            test_df["date"].max(),
        ],
        "rows": [train_df.height, valid_df.height, test_df.height],
        "features": ["|".join(feat_cols)] * 3,
    }
    
    manifest_df = pl.DataFrame(manifest_data)
    manifest_df.write_csv(manifest_path)
    LOGGER.info(f"Wrote manifest: {manifest_path}")

if __name__ == "__main__":
    main()
