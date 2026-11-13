"""Tests for the predeclared compact calibration experiment protocol."""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from ml.compact_sugeno_experiment import (
    CONFIRM_START,
    HORIZONS,
    LONG_TUNING_FOLDS,
    TUNING_FOLDS,
    balanced_support,
    candidate_grid,
    constant_grid,
    exploratory_success,
    row_loss,
    score,
    select_candidate,
)


def test_tuning_outcomes_mature_before_confirmation():
    for h in HORIZONS:
        for _, end in (*TUNING_FOLDS, *LONG_TUNING_FOLDS):
            assert end - timedelta(days=1) + timedelta(days=h) < CONFIRM_START


def test_candidate_budget_and_selection_ignore_confirmation():
    candidates = candidate_grid("volatility") + candidate_grid("residual")
    assert len(candidates) == 108
    rows = [
        {
            "candidate": c["name"],
            "relative_vs_constant": -0.1 if c["name"] == candidates[0]["name"] else 0.0,
        }
        for c in candidates
        for _ in TUNING_FOLDS
    ]
    assert select_candidate(rows, candidates)["name"] == candidates[0]["name"]
    with pytest.raises(ValueError, match="every tuning fold"):
        select_candidate(rows[:-1], candidates)
    constants = constant_grid()
    assert len(constants) == 6
    assert all(c["kwargs"]["n_rules"] == 1 for c in constants)
    assert {c["name"] for c in candidates}.isdisjoint(
        {c["name"] for c in constants}
    )


def test_predeclared_success_requires_average_and_horizon_guards():
    def rows(values):
        return [{"relative_vs_best_constant": v} for v in values]

    assert exploratory_success(rows([-0.02] * 4))
    assert not exploratory_success(rows([-0.10, 0.02, 0.02, 0.02]))
    assert not exploratory_success(rows([-0.10, -0.10, 0.06, -0.10]))
    assert not exploratory_success(rows([-0.005] * 4))


def test_common_support_and_manual_metrics():
    frame = pl.DataFrame(
        {
            "date": [1, 1, 1, 2, 2, 2],
            "target_class": ["Capesize", "Panamax", "Supramax"] * 2,
            "y": [0.0, 0.0, 0.0, 0.0, None, 0.0],
            "log_value": [0.0] * 6,
            "p_0.1": [-1.0] * 6,
            "p_0.5": [0.0] * 6,
            "p_0.9": [1.0] * 6,
        }
    )
    balanced = balanced_support(frame)
    assert balanced.height == 3
    assert balanced["date"].unique().to_list() == [1]
    assert np.isclose(score(balanced, balanced)["pinball"], 0.2 / 3)
    assert score(balanced, balanced)["winkler_80"] == 2
    assert score(balanced, balanced)["cover_80"] == 1
    with pytest.raises(ValueError, match="aligned"):
        row_loss(balanced, balanced.reverse())
