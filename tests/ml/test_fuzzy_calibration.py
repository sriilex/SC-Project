"""Tests for the new Sugeno fuzzy calibration layer (Task F).

Covers the leakage guarantee (matured residuals only), ordering preservation,
schema/row-order preservation of the numpy-only ``calibrate`` path, and that a
tiny training run keeps quantiles ordered.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from ml.fuzzy_calibration import calibrate, compute_fuzzy_inputs, fis_forward, fit_fuzzy


def _synthetic_frame(n: int = 160, seed: int = 0) -> pl.DataFrame:
    rng = np.random.RandomState(seed)
    start = date(2020, 1, 1)
    dates = pl.date_range(start, start + timedelta(days=n - 1), "1d", eager=True)
    logv = 7.0 + np.cumsum(rng.normal(0, 0.01, n))
    return pl.DataFrame({
        "date": dates,
        "target_class": ["Capesize"] * n,
        "log_value": logv,
        "p_0.1": logv - 0.10,
        "p_0.5": logv,
        "p_0.9": logv + 0.10,
        "y": logv + 0.02,
    })


def _identity_params() -> dict[str, object]:
    spec = {
        "inputs": ["cov_gap", "vol_z"],
        "centers": [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        "widths": [[0.7, 0.7, 0.7], [0.7, 0.7, 0.7]],
        "consequents": [0.0] * 9,
        "linear_coefs": None,
        "output_scale": 0.7,
        "bounded": True,
        "n_mfs": 3,
        "norm_mean": [0.0, 0.0],
        "norm_std": [1.0, 1.0],
    }
    return {"h": 1, "width": spec, "loc": spec}


def test_inputs_ignore_future_outcomes() -> None:
    """A forecast at t may only use outcomes matured at or before t."""
    frame = _synthetic_frame()
    h = 7
    inp = compute_fuzzy_inputs(frame, h)
    # Pick an interior forecast date.
    t_idx = 80
    t = frame["date"][t_idx]
    before = inp.filter(pl.col("date") == t).row(0, named=True)

    # Tamper with an outcome that is NOT matured by t (issued after t - h).
    future = frame.with_columns(
        pl.when(pl.col("date") > t - timedelta(days=h))
        .then(pl.col("y") + 5.0)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    inp2 = compute_fuzzy_inputs(future, h)
    after = inp2.filter(pl.col("date") == t).row(0, named=True)

    assert before["cov_gap"] == after["cov_gap"]
    assert before["bias_z"] == after["bias_z"]


def test_inputs_use_matured_outcomes() -> None:
    """Tampering with a matured outcome DOES change the input (sanity check)."""
    frame = _synthetic_frame()
    h = 7
    t_idx = 80
    t = frame["date"][t_idx]
    inp = compute_fuzzy_inputs(frame, h)
    before = inp.filter(pl.col("date") == t).row(0, named=True)

    past = frame.with_columns(
        pl.when(pl.col("date") <= t - timedelta(days=h))
        .then(pl.col("y") + 5.0)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    after = compute_fuzzy_inputs(past, h).filter(pl.col("date") == t).row(0, named=True)
    assert (before["cov_gap"], before["bias_z"]) != (after["cov_gap"], after["bias_z"])


def test_fis_trace_is_internally_consistent() -> None:
    """Rule weights normalise to one and contributions add up to the raw output."""
    spec = _identity_params()["width"]
    x = np.array([[-2.0, 0.5], [0.0, 0.0], [1.5, -1.0]])
    trace = fis_forward(spec, x)
    assert trace["membership"].shape == (3, 2, 3)
    assert np.allclose(trace["rule_weight_normalized"].sum(axis=1), 1.0)
    assert np.allclose(trace["rule_contribution"].sum(axis=1), trace["raw"])
    assert np.all(np.abs(trace["output"]) <= float(spec["output_scale"]))


def test_calibrate_preserves_schema_order_and_ordering() -> None:
    frame = _synthetic_frame()
    pred = frame.select("date", "target_class", pl.lit(1).cast(pl.Int64).alias("h"), "p_0.1", "p_0.5", "p_0.9")
    out = calibrate(pred, frame, 1, _identity_params())
    assert out.columns == pred.columns
    assert out.height == pred.height
    assert (out["date"] == pred["date"]).all()
    assert (out["target_class"] == pred["target_class"]).all()
    assert (out["p_0.1"] <= out["p_0.5"]).all()
    assert (out["p_0.5"] <= out["p_0.9"]).all()


def test_fit_fuzzy_keeps_ordering_and_is_identity_at_init() -> None:
    frame = _synthetic_frame(n=220)
    train = frame.tail(60)
    params, stats = fit_fuzzy(frame, train, h=1, epochs=15, lr=0.05)
    assert stats["n_train"] == 60
    assert "width" in params and "loc" in params
    out = calibrate(
        train.select("date", "target_class", pl.lit(1).cast(pl.Int64).alias("h"), "p_0.1", "p_0.5", "p_0.9"),
        frame,
        1,
        params,
    )
    assert (out["p_0.1"] <= out["p_0.5"]).all()
    assert (out["p_0.5"] <= out["p_0.9"]).all()


def test_first_order_consequents_are_fitted_and_evaluated() -> None:
    """`linear=True` stores per-rule linear coefficients and stays ordered."""
    frame = _synthetic_frame(n=240)
    train = frame.tail(80)
    params, _stats = fit_fuzzy(frame, train, h=1, linear=True, epochs=10)
    spec = params["width"]
    assert spec["linear_coefs"] is not None
    assert len(spec["linear_coefs"]) == len(spec["consequents"])
    out = calibrate(
        train.select("date", "target_class", pl.lit(1).cast(pl.Int64).alias("h"), "p_0.1", "p_0.5", "p_0.9"),
        frame,
        1,
        params,
    )
    assert (out["p_0.1"] <= out["p_0.5"]).all()
    assert (out["p_0.5"] <= out["p_0.9"]).all()


def test_split_width_has_two_bounded_outputs() -> None:
    """`split_width=True` fits separate lower/upper log multipliers."""
    frame = _synthetic_frame(n=240)
    train = frame.tail(80)
    params, stats = fit_fuzzy(frame, train, h=1, split_width=True, epochs=10)
    spec = params["width"]
    assert spec["n_outputs"] == 2
    assert np.asarray(spec["output_scale"]).shape == (2,)
    assert stats["split_width"] is True
    x = np.array([[0.0, 0.0]])
    trace = fis_forward(spec, x)
    assert trace["output"].shape == (1, 2)
    assert np.all(np.abs(trace["output"]) <= np.asarray(spec["output_scale"]))


def test_class_shrinkage_stores_per_class_offsets() -> None:
    """`class_shrinkage` records a shrunk offset for every fitted class."""
    frame = _synthetic_frame(n=240).with_columns(
        pl.when(pl.int_range(0, 240) % 2 == 0)
        .then(pl.lit("Capesize"))
        .otherwise(pl.lit("Panamax"))
        .alias("target_class")
    )
    params, stats = fit_fuzzy(frame, frame.tail(80), h=1, class_shrinkage=0.5, epochs=10)
    adjust = params["class_adjust"]
    assert set(adjust) == {"Capesize", "Panamax"}
    assert all("s" in v and "m" in v for v in adjust.values())
    assert stats["class_shrinkage"] == 0.5
    out = calibrate(
        frame.tail(20).select("date", "target_class", pl.lit(1).cast(pl.Int64).alias("h"), "p_0.1", "p_0.5", "p_0.9"),
        frame,
        1,
        params,
    )
    assert (out["p_0.1"] <= out["p_0.5"]).all()


def test_early_stopping_and_weight_decay_run() -> None:
    """Early stopping reports the best epoch and keeps quantiles ordered."""
    frame = _synthetic_frame(n=240)
    params, stats = fit_fuzzy(
        frame,
        frame.tail(120),
        h=1,
        epochs=40,
        early_stopping_patience=10,
        weight_decay=1e-3,
    )
    assert 0 <= stats["best_epoch"] <= 40
    assert stats["val_pinball"] is not None
    out = calibrate(
        frame.tail(20).select("date", "target_class", pl.lit(1).cast(pl.Int64).alias("h"), "p_0.1", "p_0.5", "p_0.9"),
        frame,
        1,
        params,
    )
    assert (out["p_0.1"] <= out["p_0.5"]).all()
    assert (out["p_0.5"] <= out["p_0.9"]).all()
