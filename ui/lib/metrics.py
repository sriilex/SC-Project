"""Metric helpers.

The canonical definitions live in ``ml.baselines.evaluate`` and
``ml.evaluation``; this module only adapts them to the frames the UI holds so
the UI and the paper cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lib.paths import QUANTILES


def _as_source(truth: pl.DataFrame, h: int) -> pl.DataFrame:
    """Present a ``(date, class, log_value, y)`` frame as an ``evaluate`` source."""
    return truth.with_columns(pl.col("y").alias(f"y_step_h{h}"))


def score(
    pred: pl.DataFrame, truth: pl.DataFrame, model: str, split: str = "valid"
) -> pl.DataFrame:
    """Per-scope metrics (POOLED + each class) using the project definition."""
    if pred.is_empty() or truth.is_empty():
        return pl.DataFrame()
    h = int(pred["h"][0])
    from ml.baselines import evaluate

    rows = evaluate(pred, _as_source(truth, h), split, model)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def pooled(
    pred: pl.DataFrame, truth: pl.DataFrame, model: str, split: str = "valid"
) -> dict:
    df = score(pred, truth, model, split)
    if df.is_empty():
        return {}
    sub = df.filter(pl.col("scope") == "POOLED")
    return sub.row(0, named=True) if not sub.is_empty() else {}


def per_class(
    pred: pl.DataFrame, truth: pl.DataFrame, model: str, split: str = "valid"
) -> pl.DataFrame:
    df = score(pred, truth, model, split)
    if df.is_empty():
        return df
    return df.filter(pl.col("scope") != "POOLED").sort("scope")


def compare(
    pred_a: pl.DataFrame,
    pred_b: pl.DataFrame,
    truth: pl.DataFrame,
    h: int,
    q: float | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Diebold-Mariano + moving-block bootstrap for ``A - B`` on shared rows.

    ``q=None`` uses the mean pinball over the three levels, matching the paper.
    Negative values favour model A.
    """
    from ml.evaluation import aligned_losses, block_bootstrap_ci, dm_test

    source = _as_source(truth, h)
    la, lb, joined = aligned_losses(pred_a, pred_b, source, h, q=q)
    if la.size == 0:
        return {}
    dm = dm_test(la, lb, h)
    boot = block_bootstrap_ci(la - lb, block=max(h, 5), n_boot=n_boot, seed=seed)
    return {**dm, **boot, "quantile": "mean" if q is None else str(q)}


def calibration_band(pred: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """Coverage, width and one-sided tail rates for one prediction frame."""
    if pred.is_empty() or truth.is_empty():
        return {}
    j = truth.join(pred, on=["date", "target_class"], how="inner").drop_nulls("y")
    if j.is_empty():
        return {}
    y = j["y"].to_numpy()
    p10 = j["p_0.1"].to_numpy()
    p50 = j["p_0.5"].to_numpy()
    p90 = j["p_0.9"].to_numpy()
    return {
        "n": int(j.height),
        "below_p10": float((y < p10).mean()),
        "above_p90": float((y > p90).mean()),
        "cover_80": float(((y >= p10) & (y <= p90)).mean()),
        "mean_width": float(np.mean(p90 - p10)),
        "mean_p50_shift": float(np.mean(p50)),
        "cross_p10_p50": float((p10 > p50).mean()),
    }


def pinball_by_quantile(pred: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Mean pinball loss per quantile for one prediction frame."""
    if pred.is_empty() or truth.is_empty():
        return pl.DataFrame()
    j = truth.join(pred, on=["date", "target_class"], how="inner").drop_nulls("y")
    if j.is_empty():
        return pl.DataFrame()
    y = j["y"].to_numpy()
    rows = []
    for q in QUANTILES:
        err = y - j[f"p_{q}"].to_numpy()
        loss = np.maximum(q * err, (q - 1.0) * err)
        rows.append({"quantile": q, "pinball": float(loss.mean())})
    return pl.DataFrame(rows)


def pooled_dm_from_file(
    dm: pl.DataFrame, split: str, quantile: str = "mean"
) -> pl.DataFrame:
    """Filter the precomputed DM table for a forest plot."""
    if dm.is_empty():
        return dm
    return dm.filter(
        (pl.col("split") == split) & (pl.col("quantile") == quantile)
    ).sort("h")


def coverage_gap(pred: pl.DataFrame, truth: pl.DataFrame) -> float:
    band = calibration_band(pred, truth)
    if not band:
        return float("nan")
    return band["cover_80"] - 0.80
