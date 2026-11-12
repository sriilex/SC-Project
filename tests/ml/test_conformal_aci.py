"""Regression test for the ACI delayed-feedback fix.

A forecast issued at ``t`` for horizon ``h`` only becomes observable at
``t + h``. Changing an unmatured outcome must therefore not change any forecast
issued before its maturity date. The old implementation used every row's
outcome immediately and failed this test.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from ml.conformal import aci_adjust


def _frame(n: int = 200, h: int = 7) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(0)
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + timedelta(days=n - 1), "1d", eager=True)
    level = 7.0 + np.cumsum(rng.normal(0, 0.01, n))
    pred = pl.DataFrame({
        "date": dates,
        "target_class": ["Capesize"] * n,
        "h": [h] * n,
        "p_0.1": level - 0.5,
        "p_0.5": level,
        "p_0.9": level + 0.5,
    })
    source = pl.DataFrame({
        "date": dates,
        "target_class": ["Capesize"] * n,
        "log_value": level,
        f"y_step_h{h}": level + rng.normal(0, 0.7, n),
    })
    return pred, source


def test_unmatured_outcome_cannot_change_earlier_forecasts() -> None:
    """/slow path: tamper with one outcome and check the pre-maturity forecasts."""
    h = 7
    pred, source = _frame(h=h)
    base_out, _ = aci_adjust(pred, source, h)

    tampered_at = pred["date"][100]
    tampered = source.with_columns(
        pl.when(pl.col("date") == tampered_at)
        .then(pl.col(f"y_step_h{h}") + 5.0)
        .otherwise(pl.col(f"y_step_h{h}"))
        .alias(f"y_step_h{h}")
    )
    tampered_out, _ = aci_adjust(pred, tampered, h)

    maturity = tampered_at + timedelta(days=h)
    before_a = base_out.filter(pl.col("date") < maturity)
    before_b = tampered_out.filter(pl.col("date") < maturity)
    assert before_a.equals(before_b)

    after_a = base_out.filter(pl.col("date") >= maturity)
    after_b = tampered_out.filter(pl.col("date") >= maturity)
    assert not after_a.equals(after_b)


def test_aci_preserves_quantile_ordering() -> None:
    """The adjusted fan must stay ordered after every update."""
    pred, source = _frame()
    out, _ = aci_adjust(pred, source, 7)
    assert (out["p_0.1"] <= out["p_0.5"]).all()
    assert (out["p_0.5"] <= out["p_0.9"]).all()


def test_aci_metadata_is_exposed() -> None:
    """Gamma, warm-up and score-buffer length are part of the returned state."""
    pred, source = _frame()
    _, state = aci_adjust(pred, source, 7, gamma=0.02, warmup=5, score_buffer=50)
    st = state["Capesize"]
    assert st["gamma"] == 0.02
    assert st["warmup"] == 5
    assert st["score_buffer"] == 50
    assert "pending" in st
