"""Pooled LightGBM quantile model over the sample splits.

Mirrors :class:`ml.models.xgb.XGBModel` (same fit/predict contract) but with
LightGBM's native ``quantile`` objective, one booster per horizon x quantile.
Targets are log-returns; predictions are added back to the current log level.
"""
from __future__ import annotations

import logging
from typing import Final

import lightgbm as lgb
import polars as pl

from ml.baselines import QUANTILES, make_matrix

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

LGB_PARAMS: Final[dict[str, object]] = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "verbosity": -1,
}


class LGBMModel:
    """One LightGBM quantile booster per horizon and quantile level."""

    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        self.horizons = horizons
        self.boosters: dict[int, dict[float, lgb.Booster]] = {}
        self.names: list[str] | None = None

        for h in horizons:
            valid_train = train.drop_nulls([f"y_step_h{h}"])
            core_n = max(int(valid_train.height * 0.85), valid_train.height - 60)
            tr = valid_train.sort("date").head(core_n)
            es = valid_train.sort("date").tail(valid_train.height - core_n)

            names, x_tr = make_matrix(tr, h)
            _, x_es = make_matrix(es, h)
            y_tr = (tr[f"y_step_h{h}"] - tr["log_value"]).to_numpy()
            y_es = (es[f"y_step_h{h}"] - es["log_value"]).to_numpy()

            cat_idx = [names.index("target_class")]
            self.names = names

            per_quantile: dict[float, lgb.Booster] = {}
            for q in QUANTILES:
                params = dict(LGB_PARAMS)
                params["alpha"] = q
                dtrain = lgb.Dataset(
                    x_tr, label=y_tr, feature_name=names,
                    categorical_feature=cat_idx, free_raw_data=True,
                )
                dval = lgb.Dataset(
                    x_es, label=y_es, reference=dtrain,
                    feature_name=names, categorical_feature=cat_idx,
                    free_raw_data=True,
                )
                per_quantile[q] = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=2000,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            self.boosters[h] = per_quantile

            gains = sorted(
                zip(names, per_quantile[0.5].feature_importance("gain").tolist()),
                key=lambda kv: kv[1], reverse=True,
            )[:12]
            LOGGER.info(f"h={h} top gains: {[k for k, _ in gains]}")

    def predict(
        self, eval_frames: dict[str, pl.DataFrame]
    ) -> dict[str, dict[int, pl.DataFrame]]:
        out: dict[str, dict[int, pl.DataFrame]] = {name: {} for name in eval_frames}

        for h in self.horizons:
            for split_name, frame in eval_frames.items():
                _, x = make_matrix(frame, h)
                log_val = frame["log_value"].to_numpy()
                exprs = [
                    pl.col("date"),
                    pl.col("target_class"),
                    pl.lit(h).cast(pl.Int64).alias("h"),
                ]
                for q in QUANTILES:
                    p_q = log_val + self.boosters[h][q].predict(x)
                    exprs.append(pl.lit(p_q).alias(f"p_{q}"))
                out[split_name][h] = frame.select(*exprs)

        return out
