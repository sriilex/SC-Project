from typing import Protocol

import polars as pl


class ForecastingModel(Protocol):
    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        ...
    def predict(self, eval_frames: dict[str, pl.DataFrame]) -> dict[str, dict[int, pl.DataFrame]]:
        ...

from .ar_lstm import ARLSTMModel
from .lgbm import LGBMModel
from .lstm import LSTMModel
from .xgb import XGBModel

# Add new models here to have them automatically picked up by baselines
MODELS = {
    "lstm": LSTMModel,
    "ar_lstm": ARLSTMModel,
    "xgb": XGBModel,
    "lgbm": LGBMModel,
}
