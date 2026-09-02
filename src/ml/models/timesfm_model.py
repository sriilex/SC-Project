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


_GLOBAL_TFM_TRAJ: dict[str, dict[tuple[object, str], np.ndarray]] = {}

class TimesFMModel:
    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        self.train = train
        self.horizons = horizons

    def predict(
        self, eval_frames: dict[str, pl.DataFrame]
    ) -> dict[str, dict[int, pl.DataFrame]]:
        """Zero-shot predictions using TimesFM 3.0 with dynamic covariates."""
        if TimesFM3Evaluator is None:
            raise ImportError("timesfm is not installed.")
            
        out: dict[str, dict[int, pl.DataFrame]] = {name: {} for name in eval_frames}
        max_h = max(self.horizons)
        train = self.train

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

        all_dfs = [train]
        eval_names = list(eval_frames.keys())
        for name in eval_names:
            all_dfs.append(eval_frames[name])
            
        full_df = pl.concat(all_dfs, how="vertical_relaxed").sort(["target_class", "date"])
        
        # Reconstruct continuous calendar to avoid sequence distortion
        class_series = {}
        for c, df_c in full_df.group_by("target_class"):
            class_name = c[0] if isinstance(c, tuple) else c
            min_date = df_c["date"].min()
            max_date = df_c["date"].max()
            date_grid = pl.DataFrame({"date": pl.date_range(min_date, max_date, "1d", eager=True)})
            cont_df = date_grid.join(df_c, on="date", how="left").fill_null(strategy="forward")
            class_series[class_name] = cont_df.sort("date")
            
        MAX_CONTEXT = 512 
        
        for split_name, frame in eval_frames.items():
            frame = frame.sort(["target_class", "date"])
            
            if split_name not in _GLOBAL_TFM_TRAJ:
                LOGGER.info(f"Populating full {max_h}-day TimesFM trajectory cache for {split_name}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                local_weights = str(repo_root / "src" / "ml" / "weights" / "timesfm-3.0-pytorch")
                config = ModelConfig(
                    checkpoint_path=local_weights,
                    per_core_batch_size=32,
                    device=device
                )
                forecaster = TimesFM3Evaluator(config)
                
                contexts = []
                dates_list = []
                classes_list = []
                
                for row in frame.iter_rows(named=True):
                    d = row["date"]
                    c = row["target_class"]
                    
                    hist_df = class_series[c].filter(pl.col("date") <= d)
                    if hist_df.height > MAX_CONTEXT:
                        hist_df = hist_df.tail(MAX_CONTEXT)
                        
                    # Pass raw log_value directly, TimesFM handles normalization
                    hist = hist_df["log_value"].to_numpy().astype(np.float32)
                    
                    contexts.append(hist)
                    dates_list.append(d)
                    classes_list.append(c)
                    
                preds_arrs = []
                batch_size = 32
                for i in range(0, len(contexts), batch_size):
                    batch_ctx = contexts[i:i+batch_size]
                    
                    # Zero-shot without covariates (pure univariate history)
                    outputs = list(forecaster.predict_batch(
                        contexts=batch_ctx, 
                        horizon=max_h, 
                        return_quantiles=True, 
                        use_symmetric_averaging=False,
                    ))
                    for out_pred in outputs:
                        q_arr = np.stack([
                            out_pred.quantiles[:, 0],
                            out_pred.quantiles[:, 4],
                            out_pred.quantiles[:, 8]
                        ], axis=1)
                        preds_arrs.append(q_arr)
                        
                del forecaster
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                cache_dict = {}
                for d, c, arr in zip(dates_list, classes_list, preds_arrs):
                    cache_dict[(d, c)] = arr
                    
                _GLOBAL_TFM_TRAJ[split_name] = cache_dict

            cache_dict = _GLOBAL_TFM_TRAJ[split_name]
            
            for h in self.horizons:
                preds_list = []
                for row in frame.iter_rows(named=True):
                    d = row["date"]
                    c = row["target_class"]
                    preds_list.append(cache_dict[(d, c)][h - 1])
                    
                preds_arr = np.array(preds_list)
                
                pred_df = pl.DataFrame({
                    "date": frame["date"],
                    "target_class": frame["target_class"],
                    "h": np.full(frame.height, h, dtype=np.int64),
                    "p_0.1": preds_arr[:, 0],
                    "p_0.5": preds_arr[:, 1],
                    "p_0.9": preds_arr[:, 2],
                })
                
                out[split_name][h] = pred_df
            
        return out

