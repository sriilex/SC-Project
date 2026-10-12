"""Parity tests for the library-backed statistics in ``ml.evaluation``.

Each test pins the current implementation to the hand-rolled version it
replaced (kept inline as a reference) or to the library call itself, on a small
fixed input.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import scoringrules as sr
from statsmodels.regression.linear_model import OLS

from ml.evaluation import (
    QUANTILES,
    aligned_losses,
    block_bootstrap_ci,
    dm_test,
    newey_west_se,
)


def _legacy_newey_west_se(d: np.ndarray, lag: int) -> float:
    """The removed hand-rolled Newey-West SE (cross terms divided by ``T - k``)."""
    T = d.size
    dbar = d.mean()
    e = d - dbar
    var = float(np.mean(e * e))
    for k in range(1, max(lag, 0) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * float(np.mean(e[k:] * e[:-k]))
    return math.sqrt(max(var, 0.0) / T)


def _legacy_block_bootstrap_ci(
    d: np.ndarray, block: int, n_boot: int, seed: int
) -> dict[str, float]:
    """The removed hand-rolled moving-block bootstrap loop."""
    T = d.size
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(T / block)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T - block + 1, size=n_blocks)
        sample = np.concatenate([d[s : s + block] for s in starts])[:T]
        means[i] = sample.mean()
    return {
        "boot_lo": float(np.quantile(means, 0.025)),
        "boot_hi": float(np.quantile(means, 0.975)),
        "boot_mean": float(means.mean()),
    }


def _loss_frames(n: int = 8, h: int = 3) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Small fixed (pred_a, pred_b, source) triple for horizon ``h``."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, n), interval="1d", eager=True)
    rng = np.random.default_rng(11)
    level = 6.0 + rng.normal(scale=0.1, size=n).cumsum()
    truth = level + rng.normal(scale=0.4, size=n)

    def _pred(shift: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "date": dates,
                "target_class": ["Capesize"] * n,
                "p_0.1": truth - 0.5 + shift,
                "p_0.5": truth + shift,
                "p_0.9": truth + 0.5 + shift,
            }
        )

    source = pl.DataFrame(
        {
            "date": dates,
            "target_class": ["Capesize"] * n,
            "log_value": level,
            f"y_step_h{h}": truth,
        }
    )
    return _pred(0.0), _pred(0.1), source


def test_newey_west_se_matches_legacy_hac() -> None:
    d = np.random.default_rng(0).normal(size=60)

    # lag 0 is White's estimator under both conventions -> exact parity.
    assert newey_west_se(d, 0) == pytest.approx(_legacy_newey_west_se(d, 0), abs=1e-12)

    # lag > 0: statsmodels uses the textbook ``/T`` autocovariance
    # normalisation, the legacy loop used ``/(T - k)``, so the estimators differ
    # by O(k / T). On this fixed input the gap is 0.9%; 2% is the bound.
    assert newey_west_se(d, 6) == pytest.approx(_legacy_newey_west_se(d, 6), rel=0.02)

    # And it is the statsmodels HAC SE of a constant-only regression.
    expected = OLS(d, np.ones(d.size)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False}
    ).bse[0]
    assert newey_west_se(d, 6) == pytest.approx(float(expected), abs=1e-12)


def test_block_bootstrap_ci_matches_legacy_loop() -> None:
    d = np.random.default_rng(7).normal(size=100) * 0.1 + 0.3

    got = block_bootstrap_ci(d, block=20, n_boot=500, seed=3)
    legacy = _legacy_block_bootstrap_ci(d, block=20, n_boot=500, seed=3)
    for key in ("boot_lo", "boot_hi", "boot_mean"):
        assert got[key] == pytest.approx(legacy[key], abs=1e-12)

    # Too few observations to resample: the degenerate result is unchanged.
    tiny = block_bootstrap_ci(d[:4], block=2)
    assert all(math.isnan(v) for v in tiny.values())


def test_dm_test_reports_the_plain_mean_pinball_difference() -> None:
    """The DM statistic keeps the same mean difference; only the SE source changed."""
    pred_a, pred_b, source = _loss_frames()
    la, lb, _ = aligned_losses(pred_a, pred_b, source, 3)
    out = dm_test(la, lb, h=3)
    assert out["dm_mean_diff"] == pytest.approx(float(np.mean(la - lb)), abs=1e-12)
    assert out["dm_n"] == la.size

    # A zero differential series has no HAC variance: the guard still returns NaN.
    flat = dm_test(la, la, h=3)
    assert flat["dm_mean_diff"] == 0.0
    assert math.isnan(flat["dm_stat"]) and math.isnan(flat["dm_p"])


def test_aligned_losses_use_scoringrules_quantile_score() -> None:
    pred_a, pred_b, source = _loss_frames()
    for q in (0.5, None):
        la, lb, joined = aligned_losses(pred_a, pred_b, source, 3, q)
        y = np.asarray(joined["y"])
        if q is None:
            ref_a = sum(
                np.asarray(sr.quantile_score(y, np.asarray(joined[f"p_{qq}"]), qq))
                for qq in QUANTILES
            ) / len(QUANTILES)
        else:
            ref_a = np.asarray(sr.quantile_score(y, np.asarray(joined[f"p_{q}"]), q))
        ref_b = sum(
            np.asarray(sr.quantile_score(y, np.asarray(joined[f"b_p_{qq}"]), qq))
            for qq in QUANTILES
        ) / len(QUANTILES)
        assert la == pytest.approx(ref_a, abs=1e-12)
        if q is None:
            assert lb == pytest.approx(ref_b, abs=1e-12)
        else:
            assert lb == pytest.approx(
                np.asarray(sr.quantile_score(y, np.asarray(joined[f"b_p_{q}"]), q)), abs=1e-12
            )
