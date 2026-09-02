"""Hybrid model: XGBoost trained on TimesFM residuals."""
from __future__ import annotations

import logging
import gc
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch
import xgboost as xgb

from ml.baselines import QUANTILES, make_matrix
from ml.models.xgb import XGB_PARAMS

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

try:
    from timesfm3 import TimesFM3Evaluator, ModelConfig
except ImportError:
    TimesFM3Evaluator = None
    ModelConfig = None

_GLOBAL_TFM_TRAJ: dict[str, dict[tuple[object, str], np.ndarray]] = {}

def get_cached_preds(df_to_pred: pl.DataFrame, name: str, h: int) -> pl.DataFrame:
    """Fetch cached TimesFM predictions for univariate history."""
    if name not in _GLOBAL_TFM_TRAJ:
        max_h = 90
        LOGGER.info(f"Populating full {max_h}-day TimesFM trajectory cache for {name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        repo_root = Path(__file__).resolve().parents[3]
        local_weights = str(repo_root / "src" / "ml" / "weights" / "timesfm-3.0-pytorch")
        config = ModelConfig(
            checkpoint_path=local_weights,
            per_core_batch_size=32,
            device=device
        )
        forecaster = TimesFM3Evaluator(config)
        
        MAX_CONTEXT = 512
        contexts = []
        dates_list = []
        classes_list = []
        last_vals_list = []

        class_series = {}
        for c, df_c in df_to_pred.group_by("target_class"):
            class_name = c[0] if isinstance(c, tuple) else c
            class_series[class_name] = df_c.sort("date")
            
        for row in df_to_pred.iter_rows(named=True):
            d = row["date"]
            c = row["target_class"]
            
            hist_df = class_series[c].filter(pl.col("date") <= d)
            if hist_df.height > MAX_CONTEXT:
                hist_df = hist_df.tail(MAX_CONTEXT)
                
            last_val = hist_df["log_value"].tail(1).item()
            hist = (hist_df["log_value"] - last_val).to_numpy().astype(np.float32)
            
            contexts.append(hist)
            dates_list.append(d)
            classes_list.append(c)
            last_vals_list.append(last_val)

        preds_arrs = []
        batch_size = 32
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            batch_last_vals = last_vals_list[i:i+batch_size]
            
            outputs = list(forecaster.predict_batch(
                contexts=batch_ctx, 
                horizon=max_h, 
                return_quantiles=True, 
                use_symmetric_averaging=False
            ))
            
            for j, out_pred in enumerate(outputs):
                lv = batch_last_vals[j]
                # Extract median (index 4) for all max_h steps
                medians = out_pred.quantiles[:, 4] + lv
                preds_arrs.append(medians)
                
        del forecaster
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cache_dict = {}
        for d, c, arr in zip(dates_list, classes_list, preds_arrs):
            cache_dict[(d, c)] = arr
            
        _GLOBAL_TFM_TRAJ[name] = cache_dict

    cache_dict = _GLOBAL_TFM_TRAJ[name]
    
    # Extract the h-th step
    preds_list = []
    for row in df_to_pred.iter_rows(named=True):
        d = row["date"]
        c = row["target_class"]
        preds_list.append(cache_dict[(d, c)][h - 1])
        
    return pl.DataFrame({
        "date": df_to_pred["date"],
        "target_class": df_to_pred["target_class"],
        "tfm_pred": preds_list
    })

class XGBPlusTFMModel:
    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        if TimesFM3Evaluator is None:
            raise ImportError("timesfm is not installed.")
        
        self.horizons = horizons
        self.boosters = {}
        self.names = None
        self.feature_types = None
        
        for h in horizons:
            valid_train = train.drop_nulls([f"y_step_h{h}"])
            core_n = max(int(valid_train.height * 0.85), valid_train.height - 60)
            tr = valid_train.sort("date").head(core_n)
            es = valid_train.sort("date").tail(valid_train.height - core_n)
            
            tfm_tr = get_cached_preds(tr, "train_core", h)
            tfm_es = get_cached_preds(es, "train_es", h)
            
            tr = tr.join(tfm_tr, on=["date", "target_class"])
            es = es.join(tfm_es, on=["date", "target_class"])
            
            names, x_tr = make_matrix(tr, h)
            _, x_es = make_matrix(es, h)
            
            y_tr_actual = tr[f"y_step_h{h}"].to_numpy()
            y_es_actual = es[f"y_step_h{h}"].to_numpy()
            
            residuals_tr = y_tr_actual - tr["tfm_pred"].to_numpy()
            residuals_es = y_es_actual - es["tfm_pred"].to_numpy()
            
            feature_types = ["c" if n == "target_class" else "q" for n in names]
            self.names = names
            self.feature_types = feature_types
            
            dtrain = xgb.DMatrix(x_tr, label=residuals_tr, feature_names=names, feature_types=feature_types)
            dval = xgb.DMatrix(x_es, label=residuals_es, feature_names=names, feature_types=feature_types)
            
            LOGGER.info(f"Training XGBoost on TimesFM residuals (h={h})")
            booster = xgb.train(
                XGB_PARAMS,
                dtrain,
                num_boost_round=2000,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            self.boosters[h] = booster

    def predict(
        self, eval_frames: dict[str, pl.DataFrame]
    ) -> dict[str, dict[int, pl.DataFrame]]:
        out: dict[str, dict[int, pl.DataFrame]] = {name: {} for name in eval_frames}
        
        for h in self.horizons:
            booster = self.boosters[h]
            for split_name, frame in eval_frames.items():
                tfm_eval = get_cached_preds(frame, split_name, h)
                frame_joined = frame.join(tfm_eval, on=["date", "target_class"])
                
                _, x = make_matrix(frame_joined, h)
                dx = xgb.DMatrix(x, feature_names=self.names, feature_types=self.feature_types)
                
                xgb_resid_preds = booster.predict(dx)
                if len(QUANTILES) == 1:
                    xgb_resid_preds = xgb_resid_preds.reshape(-1, 1)
                    
                exprs = [
                    pl.col("date"), 
                    pl.col("target_class"),
                    pl.lit(h).cast(pl.Int64).alias("h")
                ]
                
                base_pred = frame_joined["tfm_pred"].to_numpy()
                for i, q in enumerate(QUANTILES):
                    p_q = base_pred + xgb_resid_preds[:, i]
                    exprs.append(pl.lit(p_q).alias(f"p_{q}"))
                    
                out[split_name][h] = frame.select(*exprs)
                
        return out
