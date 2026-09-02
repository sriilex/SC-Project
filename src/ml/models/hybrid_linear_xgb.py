"""Hybrid model: XGBoost trained on Ridge Regression residuals."""
from __future__ import annotations

import logging
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import Ridge

from ml.baselines import QUANTILES, make_matrix
from ml.models.xgb import XGB_PARAMS

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

class XGBPlusLinearModel:
    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        self.horizons = horizons
        self.boosters = {}
        self.lin_models = {}
        self.names = None
        self.feature_types = None
        self.numeric_idx = None
        
        for h in horizons:
            valid_train = train.drop_nulls([f"y_step_h{h}"])
            core_n = max(int(valid_train.height * 0.85), valid_train.height - 60)
            tr = valid_train.sort("date").head(core_n)
            es = valid_train.sort("date").tail(valid_train.height - core_n)
            
            names, x_tr = make_matrix(tr, h)
            _, x_es = make_matrix(es, h)
            
            y_tr_actual = (tr[f"y_step_h{h}"] - tr["log_value"]).to_numpy()
            y_es_actual = (es[f"y_step_h{h}"] - es["log_value"]).to_numpy()
            
            numeric_idx = [i for i, n in enumerate(names) if n != "target_class"]
            self.numeric_idx = numeric_idx
            
            x_tr_lin = x_tr[:, numeric_idx]
            x_es_lin = x_es[:, numeric_idx]
            
            x_tr_lin = np.nan_to_num(x_tr_lin)
            x_es_lin = np.nan_to_num(x_es_lin)
            
            lin_model = Ridge(alpha=1.0)
            lin_model.fit(x_tr_lin, y_tr_actual)
            
            lin_tr_pred = lin_model.predict(x_tr_lin)
            lin_es_pred = lin_model.predict(x_es_lin)
            
            residuals_tr = y_tr_actual - lin_tr_pred
            residuals_es = y_es_actual - lin_es_pred
            
            feature_types = ["c" if n == "target_class" else "q" for n in names]
            self.names = names
            self.feature_types = feature_types
            
            dtrain = xgb.DMatrix(x_tr, label=residuals_tr, feature_names=names, feature_types=feature_types)
            dval = xgb.DMatrix(x_es, label=residuals_es, feature_names=names, feature_types=feature_types)
            
            LOGGER.info(f"Training XGBoost on Ridge residuals (h={h})")
            booster = xgb.train(
                XGB_PARAMS,
                dtrain,
                num_boost_round=2000,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            self.boosters[h] = booster
            self.lin_models[h] = lin_model

    def predict(
        self, eval_frames: dict[str, pl.DataFrame]
    ) -> dict[str, dict[int, pl.DataFrame]]:
        out: dict[str, dict[int, pl.DataFrame]] = {name: {} for name in eval_frames}
        
        for h in self.horizons:
            booster = self.boosters[h]
            lin_model = self.lin_models[h]
            
            for split_name, frame in eval_frames.items():
                _, x = make_matrix(frame, h)
                x_lin = np.nan_to_num(x[:, self.numeric_idx])
                
                lin_pred = lin_model.predict(x_lin)
                
                dx = xgb.DMatrix(x, feature_names=self.names, feature_types=self.feature_types)
                xgb_resid_preds = booster.predict(dx)
                
                if len(QUANTILES) == 1:
                    xgb_resid_preds = xgb_resid_preds.reshape(-1, 1)
                    
                exprs = [
                    pl.col("date"), 
                    pl.col("target_class"),
                    pl.lit(h).cast(pl.Int64).alias("h")
                ]
                
                log_val = frame["log_value"].to_numpy()
                for i, q in enumerate(QUANTILES):
                    p_q = log_val + lin_pred + xgb_resid_preds[:, i]
                    exprs.append(pl.lit(p_q).alias(f"p_{q}"))
                    
                out[split_name][h] = frame.select(*exprs)
                
        return out
