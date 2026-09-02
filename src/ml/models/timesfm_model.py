"""TimesFM baseline model."""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import timedelta
from typing import Final
import gc
import tomllib

import numpy as np
import polars as pl
import torch

from ml.baselines import QUANTILES, EXCLUDE
from ml.features.calendar import add_calendar_features, calendar_feature_names

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# Try importing TimesFM
try:
    from timesfm3 import TimesFM3Evaluator, ModelConfig
except ImportError:
    TimesFM3Evaluator = None
    ModelConfig = None


def predict(
    train: pl.DataFrame, eval_frames: dict[str, pl.DataFrame], h: int
) -> dict[str, pl.DataFrame]:
    """Zero-shot predictions using TimesFM 3.0 with dynamic covariates."""
    if TimesFM3Evaluator is None:
        raise ImportError("timesfm is not installed.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=32,
        device=device
    )
    forecaster = TimesFM3Evaluator(config)

    # 1. Identify feature columns
    feature_cols = [c for c in train.columns if c not in EXCLUDE and c != "target_class"]
    future_cov_names = calendar_feature_names()
    past_cov_names = [c for c in feature_cols if c not in future_cov_names]

    # Load CNY dates for calendar features
    repo_root = Path(__file__).resolve().parents[3]
    with open(repo_root / "src" / "config" / "samples.toml", "rb") as f:
        toml_conf = tomllib.load(f)
    cny_dates = toml_conf["calendar"]["cny_dates"]

    # 2. Normalize features based on train
    mu = {c: train[c].mean() for c in feature_cols}
    sd = {c: train[c].std() for c in feature_cols}
    
    def preprocess(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
        exprs = []
        for c in cols:
            if c not in df.columns:
                continue
            m = mu[c] if mu[c] is not None and not np.isnan(mu[c]) else 0.0
            s = sd[c] if sd[c] is not None and sd[c] > 0 and not np.isnan(sd[c]) else 1.0
            exprs.append(
                ((pl.col(c) - m) / s).fill_nan(0.0).fill_null(0.0).alias(c)
            )
        return df.with_columns(exprs)

    # We need the full history to construct contexts
    all_dfs = [train]
    eval_names = list(eval_frames.keys())
    for name in eval_names:
        all_dfs.append(eval_frames[name])
        
    full_df = pl.concat(all_dfs, how="vertical_relaxed").sort(["target_class", "date"])
    full_df_norm = preprocess(full_df, cols=feature_cols)
    
    # Pre-filter full_df per class for efficient context building
    class_series = {}
    for c, df_c in full_df_norm.group_by("target_class"):
        class_name = c[0] if isinstance(c, tuple) else c
        class_series[class_name] = df_c.sort("date")

    out: dict[str, pl.DataFrame] = {}
    
    # Max context length to avoid OOM or slow inference
    MAX_CONTEXT = 512 
    
    for split_name, frame in eval_frames.items():
        frame = frame.sort(["target_class", "date"])
        
        contexts = []
        past_covariates_list = []
        future_covariates_list = []
        dates_list = []
        classes_list = []
        
        for row in frame.iter_rows(named=True):
            d = row["date"]
            c = row["target_class"]
            
            # Get historical DataFrame up to current date
            hist_df = class_series[c].filter(pl.col("date") <= d)
            if hist_df.height > MAX_CONTEXT:
                hist_df = hist_df.tail(MAX_CONTEXT)
                
            hist = hist_df["log_value"].to_numpy().astype(np.float32)
            
            # Past-only covariates: shape (num_channels, context_len)
            past_covs = hist_df.select(past_cov_names).to_numpy().astype(np.float32).T
            
            # Past-and-Future covariates: shape (num_channels, context_len + horizon)
            dates_past = hist_df["date"].to_list()
            last_date = dates_past[-1]
            dates_future = [last_date + timedelta(days=i) for i in range(1, h + 1)]
            all_dates = dates_past + dates_future
            
            cal_df = pl.DataFrame({"date": all_dates})
            cal_df = add_calendar_features(cal_df, cny_dates)
            cal_df = preprocess(cal_df, cols=future_cov_names)
            future_covs = cal_df.select(future_cov_names).to_numpy().astype(np.float32).T
            
            contexts.append(hist)
            past_covariates_list.append(past_covs)
            future_covariates_list.append(future_covs)
            dates_list.append(d)
            classes_list.append(c)
            
        preds_list = []
        batch_size = 32
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            batch_past_covs = past_covariates_list[i:i+batch_size]
            batch_future_covs = future_covariates_list[i:i+batch_size]
            
            outputs = list(forecaster.predict_batch(
                contexts=batch_ctx, 
                horizon=h, 
                past_only_covariates=batch_past_covs,
                past_future_covariates=batch_future_covs,
                return_quantiles=True, 
                use_symmetric_averaging=False
            ))
            for out_pred in outputs:
                # We want the h-th step (index h-1)
                # For QUANTILES = (0.1, 0.5, 0.9), indices are 0, 4, 8
                q_01 = out_pred.quantiles[h-1, 0]
                q_05 = out_pred.quantiles[h-1, 4]
                q_09 = out_pred.quantiles[h-1, 8]
                preds_list.append((q_01, q_05, q_09))
                
        preds_arr = np.array(preds_list)
        
        pred_df = pl.DataFrame({
            "date": dates_list,
            "target_class": classes_list,
            "h": np.full(len(dates_list), h, dtype=np.int64),
            "p_0.1": preds_arr[:, 0],
            "p_0.5": preds_arr[:, 1],
            "p_0.9": preds_arr[:, 2],
        })
        
        out[split_name] = pred_df
        
    del forecaster
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return out
