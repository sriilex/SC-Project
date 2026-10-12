"""Live quantile inference backed by the cached LSTM ensemble.

The exported-XGBoost path was removed together with the tree baselines.
:class:`FreightPredictor` now serves the five-seed monotone quantile LSTM
(via :mod:`ml.predictions`) and can apply the trained Sugeno fuzzy layer
(via :mod:`ml.fuzzy_calibration`).
"""
from __future__ import annotations

from typing import Final

import polars as pl

from ml.fuzzy_calibration import calibrate as fuzzy_calibrate

#: Base quantile columns emitted by the cached LSTM ensemble.
QUANTILE_COLS: Final[tuple[str, ...]] = ("p_0.1", "p_0.5", "p_0.9")


class FreightPredictor:
    """Serve the cached LSTM quantile ensemble for a single horizon."""

    def __init__(self, h: int) -> None:
        self.h = h
        self._base: pl.DataFrame | None = None

    def _base_predictions(self) -> pl.DataFrame:
        """Base LSTM quantiles over the cached train/valid splits (lazy)."""
        if self._base is None:
            from ml.baselines import load_split
            from ml.predictions import ensure_lstm_predictions

            splits = {name: load_split(name) for name in ("train", "valid")}
            mean_preds, _, _ = ensure_lstm_predictions(splits)
            self._base = pl.concat([mean_preds[name][self.h] for name in splits])
        return self._base

    def predict(
        self,
        df: pl.DataFrame,
        history_df: pl.DataFrame | None = None,
        params: dict | None = None,
    ) -> pl.DataFrame:
        """Quantile predictions for the ``(date, target_class)`` rows of ``df``.

        ``df`` supplies the row keys only; the quantiles come from the cached
        LSTM ensemble, so the frame matches the ``date | target_class | h |
        p_0.1 | p_0.5 | p_0.9`` contract. Sugeno calibration is applied only
        when *both* ``history_df`` and ``params`` are given; otherwise the raw
        base quantiles are returned unchanged.
        """
        keys = df.select("date", "target_class").unique()
        base = (
            self._base_predictions()
            .join(keys, on=["date", "target_class"], how="inner")
            .select(
                "date",
                "target_class",
                pl.lit(self.h).cast(pl.Int64).alias("h"),
                *QUANTILE_COLS,
            )
        )
        if history_df is not None and params is not None:
            return fuzzy_calibrate(base, history_df, self.h, params)
        return base
