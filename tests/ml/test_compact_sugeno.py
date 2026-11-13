from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.compact_sugeno import (
    MAX_WIDTH_MULTIPLIER,
    MIN_WIDTH_MULTIPLIER,
    apply_compact,
    compute_compact_inputs,
    fit_compact,
    membership_spec,
    rule_weights,
)


def _frame():
    rng = np.random.default_rng(7)
    n = 1_100
    y = rng.normal(0, 1, n)
    p50 = y + rng.normal(0, 0.25, n)
    half = np.abs(rng.normal(0.7, 0.2, n))
    vol = np.clip(rng.normal(0, 1, n), -3, 3)
    return pl.DataFrame(
        {
            "date": np.repeat(np.arange(n) // 5, 5),
            "target_class": np.repeat(["A", "B"], n // 2),
            "y": y,
            "p_0.5": p50,
            "p_0.9": p50 + half,
            "p_0.1": p50 - half,
            "vol_ratio": vol,
        }
    )


def test_membership_is_fixed_gaussian_and_neutral():
    spec = membership_spec(3)
    assert spec["centers"] == [-1.0, 0.0, 1.0]
    assert spec["widths"] == [1.0] * 3
    weights = np.array(rule_weights(np.array([0.0, None]), 3))
    # Zero (and the neutral fallback for missing) sits exactly on the middle rule.
    assert np.allclose(weights[0], [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(weights[1], weights[0])
    assert np.allclose(weights.sum(axis=1), 1.0)
    two = np.array(rule_weights(np.array([0.0]), 2))
    assert np.allclose(two, [0.5, 0.5])


def test_missing_input_maps_to_neutral_membership_input():
    frame = pl.DataFrame(
        {"date": [0, 1, 2], "target_class": ["A"] * 3, "y": [1.0] * 3}
    )
    inputs = compute_compact_inputs(frame, h=1, width_feature="vol_ratio")
    assert np.array_equal(inputs["vol_ratio"], np.zeros(3))
    assert np.array_equal(inputs["error_ratio"], np.zeros(3))
    assert np.array_equal(inputs["median_bias"], np.zeros(3))


def test_matured_inputs_are_delayed_by_horizon():
    frame = pl.DataFrame(
        {
            "date": [0, 1, 2, 3, 4],
            "target_class": ["A"] * 5,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "p_0.5": [0.0, 0.0, 0.0, 0.0, 0.0],
            "p_0.9": [2.0, 2.0, 2.0, 2.0, 2.0],
            "p_0.1": [-2.0, -2.0, -2.0, -2.0, -2.0],
        }
    )
    inputs = compute_compact_inputs(frame, h=2, width_feature="error_ratio")
    assert np.all(np.isnan(inputs["error_ratio"][:2]))
    assert np.isclose(inputs["error_ratio"][2], 0.5)


def test_fit_is_bounded_serializable_and_nests_constant():
    frame = _frame()
    inputs = compute_compact_inputs(frame, h=1, width_feature="vol_ratio")
    params = fit_compact(frame, inputs, h=1, n_rules=3, shrinkage=0.0)
    assert params["version"] == "compact-sugeno-v2"
    assert params["n_parameters"] == 4
    for key in ("width_consequents_lo", "width_consequents_hi"):
        values = params[key]
        assert MIN_WIDTH_MULTIPLIER <= min(values) <= max(values) <= MAX_WIDTH_MULTIPLIER
    # Symmetric fit keeps identical lower/upper multipliers.
    lo = np.asarray(params["width_consequents_lo"])
    hi = np.asarray(params["width_consequents_hi"])
    assert np.array_equal(lo, hi)
    # One rule nests the constant fit under the identical objective.
    constant = fit_compact(frame, inputs, h=1, n_rules=1, shrinkage=0.0)
    assert constant["n_parameters"] == 2


def test_asymmetric_fit_keeps_bounded_separate_multipliers():
    frame = _frame()
    inputs = compute_compact_inputs(frame, h=1, width_feature="vol_ratio")
    params = fit_compact(
        frame, inputs, h=1, n_rules=2, asymmetric=True, shrinkage=0.0
    )
    assert params["asymmetric"] is True
    assert params["n_parameters"] == 5
    lo = np.asarray(params["width_consequents_lo"])
    hi = np.asarray(params["width_consequents_hi"])
    assert lo.shape == hi.shape == (2,)
    assert (lo >= MIN_WIDTH_MULTIPLIER).all()
    assert (lo <= MAX_WIDTH_MULTIPLIER).all()
    assert (hi >= MIN_WIDTH_MULTIPLIER).all()
    assert (hi <= MAX_WIDTH_MULTIPLIER).all()
    predicted = apply_compact(
        frame.select(["p_0.1", "p_0.5", "p_0.9"]), inputs, params
    )
    widths = predicted["p_0.9"] - predicted["p_0.1"]
    base_widths = frame["p_0.9"] - frame["p_0.1"]
    ratios = widths.to_numpy() / base_widths.to_numpy()
    assert MIN_WIDTH_MULTIPLIER * 0.999 <= ratios.min()
    assert ratios.max() <= MAX_WIDTH_MULTIPLIER * 1.001


def test_fit_reduces_pinball_relative_to_uncalibrated_frame():
    frame = _frame()
    inputs = compute_compact_inputs(frame, h=1, width_feature="vol_ratio")
    base = 0.9 * (frame["p_0.9"] - frame["y"]).clip(0.0) + 0.1 * (
        frame["y"] - frame["p_0.1"]
    ).clip(0.0)
    params = fit_compact(frame, inputs, h=1, n_rules=3, shrinkage=0.0)
    predicted = apply_compact(frame.select(["p_0.1", "p_0.5", "p_0.9"]), inputs, params)
    fit_loss = 0.9 * (predicted["p_0.9"] - frame["y"]).clip(0.0) + 0.1 * (
        frame["y"] - predicted["p_0.1"]
    ).clip(0.0)
    assert fit_loss.mean() <= base.mean() + 1e-8


def test_recency_half_life_weights_recent_rows_more():
    frame = _frame()
    inputs = compute_compact_inputs(frame, h=1, width_feature="vol_ratio")
    flat = fit_compact(
        frame, inputs, h=1, n_rules=1, shrinkage=0.0, recency_half_life_months=None
    )
    recent = fit_compact(
        frame, inputs, h=1, n_rules=1, shrinkage=0.0, recency_half_life_months=6
    )
    assert flat["stats"]["recency_half_life_months"] is None
    assert recent["stats"]["recency_half_life_months"] == 6
    assert abs(flat["width_consequents_lo"][0] - recent["width_consequents_lo"][0]) > 0


def test_recency_and_asymmetry_rejected_invalid_values():
    frame = _frame()
    inputs = compute_compact_inputs(frame, h=1, width_feature="vol_ratio")
    with pytest.raises(ValueError):
        fit_compact(frame, inputs, h=1, n_rules=0)
    with pytest.raises(ValueError):
        fit_compact(frame, inputs, h=1, recency_half_life_months=0)
    with pytest.raises(ValueError):
        fit_compact(frame, inputs, h=1, shrinkage=-1)
    with pytest.raises(ValueError):
        apply_compact(
            frame.select(["p_0.1", "p_0.5", "p_0.9"]),
            inputs,
            {
                "width_consequents_lo": [1.0],
                "width_consequents_hi": [1.0],
                "location_shift": 0.0,
            },
        )


def test_shrinkage_pulls_rule_differences_toward_their_mean():
    frame = _frame()
    inputs = compute_compact_inputs(frame, h=1, width_feature="vol_ratio")
    weak = fit_compact(frame, inputs, h=1, n_rules=3, shrinkage=0.0)
    strong = fit_compact(frame, inputs, h=1, n_rules=3, shrinkage=100.0)
    spread = lambda p: float(  # noqa: E731
        np.max(p["width_consequents_lo"]) - np.min(p["width_consequents_lo"])
    )
    assert spread(strong) <= spread(weak) + 1e-9
