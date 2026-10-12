import numpy as np
import polars as pl
import pytest
import scoringrules as sr

from ml.baselines import _clean_sigma, evaluate, pinball

H = 7


def _legacy_pinball(y: pl.Series, p: pl.Series, q: float) -> float:
    """The removed hand-rolled pinball loss."""
    err = y - p
    return float(
        pl.select(pl.when(err >= 0).then(q * err).otherwise((q - 1.0) * err).mean()).item()
    )


def _frames(n: int = 6) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A small fixed (prediction, source) pair for horizon ``H``."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, n), interval="1d", eager=True)
    rng = np.random.default_rng(0)
    level = 6.0 + rng.normal(scale=0.1, size=n).cumsum()
    p50 = level + 0.05
    p10, p90 = p50 - 0.4, p50 + 0.6
    pred = pl.DataFrame(
        {
            "date": dates,
            "target_class": ["Capesize"] * n,
            "h": [H] * n,
            "p_0.1": p10,
            "p_0.5": p50,
            "p_0.9": p90,
        }
    )
    source = pl.DataFrame(
        {
            "date": dates,
            "target_class": ["Capesize"] * n,
            "log_value": level,
            f"y_step_h{H}": level + rng.normal(scale=0.5, size=n),
        }
    )
    return pred, source


def test_pinball_loss():
    # True values
    y = pl.Series("y", [10.0, 10.0, 10.0])
    # Predictions
    p = pl.Series("p", [8.0, 10.0, 12.0])

    # Errors: y - p = [2.0, 0.0, -2.0]
    # For q=0.5: mean of [0.5*2, 0.5*0, -0.5*-2] = [1.0, 0.0, 1.0] -> mean = 2/3 = 0.6666...
    loss = pinball(y, p, 0.5)
    assert loss == pytest.approx(0.6666666666666666)

    # For q=0.1: mean of [0.1*2, 0, -0.9*-2] = [0.2, 0, 1.8] -> mean = 2.0 / 3 = 0.6666...
    loss_01 = pinball(y, p, 0.1)
    assert loss_01 == pytest.approx(0.6666666666666666)

    # For q=0.9: mean of [0.9*2, 0, -0.1*-2] = [1.8, 0, 0.2] -> mean = 2.0 / 3 = 0.6666...
    loss_09 = pinball(y, p, 0.9)
    assert loss_09 == pytest.approx(0.6666666666666666)


def test_pinball_matches_scoringrules_and_legacy_formula():
    """``pinball`` is ``scoringrules.quantile_score`` averaged, exactly."""
    pred, source = _frames()
    y = source[f"y_step_h{H}"]
    for q in (0.1, 0.5, 0.9):
        got = pinball(y, pred[f"p_{q}"], q)
        reference = float(
            np.asarray(sr.quantile_score(np.asarray(y), np.asarray(pred[f"p_{q}"]), q)).mean()
        )
        assert got == pytest.approx(reference, abs=1e-12)
        assert got == pytest.approx(_legacy_pinball(y, pred[f"p_{q}"], q), abs=1e-12)


def test_evaluate_winkler_matches_scoringrules_interval_score():
    """``winkler_80`` is the mean interval score of the P10-P90 interval."""
    pred, source = _frames()
    pooled = next(r for r in evaluate(pred, source, "valid", "lstm") if r["scope"] == "POOLED")

    y = np.asarray(source[f"y_step_h{H}"])
    p10, p90 = np.asarray(pred["p_0.1"]), np.asarray(pred["p_0.9"])
    assert pooled["winkler_80"] == pytest.approx(
        float(np.asarray(sr.interval_score(y, p10, p90, 0.2)).mean()), abs=1e-12
    )

    # The removed hand-rolled form: width + (2 / alpha) * mean breach penalty.
    below = np.clip(p10 - y, 0.0, None)
    above = np.clip(y - p90, 0.0, None)
    assert pooled["winkler_80"] == pytest.approx(
        float((p90 - p10).mean()) + 10.0 * float((below + above).mean()), abs=1e-12
    )


def test_clean_sigma():
    # Valid float
    assert _clean_sigma(1.5, 0.5) == 1.5
    # None
    assert _clean_sigma(None, 0.5) == 0.5
    # NaN
    assert _clean_sigma(float('nan'), 0.5) == 0.5
    # Negative / Zero
    assert _clean_sigma(0.0, 0.5) == 0.5
    assert _clean_sigma(-1.0, 0.5) == 0.5
