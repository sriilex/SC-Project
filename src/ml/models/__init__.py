from typing import Protocol
import polars as pl

class ForecastingModel(Protocol):
    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        ...
    def predict(self, eval_frames: dict[str, pl.DataFrame]) -> dict[str, dict[int, pl.DataFrame]]:
        ...

from .lstm import LSTMModel
from .ar_lstm import ARLSTMModel

# Add new models here to have them automatically picked up by baselines
MODELS = {
    "lstm": LSTMModel,
    "ar_lstm": ARLSTMModel,
}
