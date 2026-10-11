"""Regression tests for LSTM target construction and the RW comparison.

These lock in the Phase-0 bug fixes:

* ``y_step_h1`` / ``y_mean_h1`` used to be fed to the model as features, directly
  leaking the h=1 target.
* ``create_windows`` used to compute the target as ``y_step_h - log_value`` on a
  frame where ``log_value`` had already been standardised but ``y_step_h`` had
  not, turning the h>=7 target into the raw log level (~7) instead of a return.
"""
from __future__ import annotations

import polars as pl
import torch

from ml.baselines import evaluate, fit_rw_emp, load_split, predict_rw_emp
from ml.models.lstm import EXCLUDE, LSTMModel, add_target_returns, create_windows


def test_target_columns_are_excluded_from_features() -> None:
    """No target-derived column may leak into the feature matrix."""
    for col in (
        "y_step_h1",
        "y_step_h7",
        "y_step_h30",
        "y_step_h90",
        "y_mean_h1",
        "y_mean_h90",
        "target_ret_h1",
        "target_ret_h7",
    ):
        assert col in EXCLUDE, f"{col} must be excluded from model features"


def test_create_windows_target_is_return_scale() -> None:
    """h=7 targets are returns (O(0.1)), not log levels (O(7))."""
    train = load_split("train")
    model = LSTMModel()
    model._fit_scalers(train)

    aug = add_target_returns(train, [7])
    norm = model._preprocess(aug)
    _, _, _, y, meta = create_windows(
        norm, 7, model.seq_cols, model.cross_cols, target_col="target_ret_h7"
    )
    assert len(meta) > 0
    mask = ~torch.isnan(y)
    valid = y[mask]
    meta_valid = [m for m, keep in zip(meta, mask.tolist()) if keep]

    # A level-scale regression target would be ~mu(log_value) ~= 6.9.
    assert valid.abs().mean() < 0.5
    assert valid.std() < 1.0

    # Targets must equal the raw return at each emitted date exactly (i.e. no
    # stale forward-filled values from the continuous-calendar reconstruction).
    got = pl.DataFrame(
        {
            "date": [m[0] for m in meta_valid],
            "target_class": [m[1] for m in meta_valid],
            "y": valid.numpy(),
        }
    )
    raw = aug.select("date", "target_class", "target_ret_h7").drop_nulls()
    joined = got.join(raw, on=["date", "target_class"], how="left")
    assert joined["target_ret_h7"].null_count() == 0
    assert (joined["y"] - joined["target_ret_h7"]).abs().max() < 1e-5


def test_lstm_h7_median_pinball_within_order_of_random_walk() -> None:
    """The trained LSTM h=7 P50 must stay within [0.5x, 2x] of rw_emp.

    Scored on the shared per-horizon rows. The pre-fix level-scale bug produced
    pinball ~3.4 against a random-walk baseline of ~0.05, so this bound fails
    loudly on regression while allowing normal training noise.
    """
    train = load_split("train")
    valid = load_split("valid")

    model = LSTMModel(max_epochs=25, patience=5, hidden=16, seeds=(0,), checkpoint=False)
    model.fit(train, [7])
    lstm_pred = model.predict({"valid": valid})["valid"][7]
    lstm_rows = evaluate(lstm_pred, valid, "valid", "lstm")
    lstm_pinball = next(r["pinball_0.5"] for r in lstm_rows if r["scope"] == "POOLED")
    lstm_cross = next(r["x_p10_gt_p50"] for r in lstm_rows if r["scope"] == "POOLED")

    rw_pred = predict_rw_emp(valid, 7, fit_rw_emp(train, 7))
    rw_rows = evaluate(rw_pred, valid, "valid", "rw_emp")
    rw_pinball = next(r["pinball_0.5"] for r in rw_rows if r["scope"] == "POOLED")

    assert rw_pinball > 0
    assert 0.5 * rw_pinball <= lstm_pinball <= 2.0 * rw_pinball
    # Monotone head: P10 > P50 can never happen.
    assert lstm_cross == 0.0
    assert (lstm_pred["p_0.1"] <= lstm_pred["p_0.5"]).all()
    assert (lstm_pred["p_0.5"] <= lstm_pred["p_0.9"]).all()
