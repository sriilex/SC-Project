"""Evaluation statistics for the quantile forecast comparison (Task G).

* Diebold-Mariano tests on pinball-loss differentials with a Newey-West HAC
  variance at lag ``h - 1``.
* A moving-block bootstrap confidence interval for the mean pinball difference,
  because 90-day windows overlap and the four vessel classes are highly
  correlated, so effective sample sizes are much smaller than the row count.
* A short markdown summary writer.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)


def aligned_losses(
    pred_a: pl.DataFrame,
    pred_b: pl.DataFrame,
    source: pl.DataFrame,
    h: int,
    q: float | None = None,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Per-row pinball losses of two models on their shared rows.

    ``q=None`` uses the mean of the three quantile losses per row. Returns
    ``(loss_a, loss_b, joined)`` with rows sorted by date then class.
    """
    truth = source.select(
        "date", "target_class", pl.col(f"y_step_h{h}").alias("y")
    )
    a = pred_a.join(truth, on=["date", "target_class"], how="inner").drop_nulls("y")
    b = (
        pred_b.join(truth, on=["date", "target_class"], how="inner")
        .drop_nulls("y")
        .select(
            "date", "target_class", "y",
            *[pl.col(f"p_{q}").alias(f"b_p_{q}") for q in QUANTILES],
        )
    )
    j = a.join(b, on=["date", "target_class", "y"], how="inner").sort(["date", "target_class"])
    y = j["y"].to_numpy()

    def _losses(prefix: str) -> np.ndarray:
        if q is None:
            acc = np.zeros(j.height)
            for qq in QUANTILES:
                err = y - j[f"{prefix}p_{qq}"].to_numpy()
                acc += np.maximum(qq * err, (qq - 1.0) * err)
            return acc / len(QUANTILES)
        err = y - j[f"{prefix}p_{q}"].to_numpy()
        return np.maximum(q * err, (q - 1.0) * err)

    return _losses(""), _losses("b_"), j


def newey_west_se(d: np.ndarray, lag: int) -> float:
    """HAC (Newey-West) standard error of the mean of ``d``."""
    T = d.size
    if T < 2:
        return float("nan")
    dbar = d.mean()
    e = d - dbar
    gamma0 = float(np.mean(e * e))
    var = gamma0
    for k in range(1, max(lag, 0) + 1):
        w = 1.0 - k / (lag + 1.0)
        cov = float(np.mean(e[k:] * e[:-k]))
        var += 2.0 * w * cov
    var = max(var, 0.0)
    return math.sqrt(var / T)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> dict[str, float]:
    """Two-sided Diebold-Mariano statistic and normal-approximation p-value."""
    d = loss_a - loss_b
    if d.size < 3:
        return {"dm_mean_diff": float("nan"), "dm_stat": float("nan"), "dm_p": float("nan")}
    se = newey_west_se(d, max(h - 1, 0))
    stat = float(d.mean() / se) if se > 0 else float("nan")
    p = math.erfc(abs(stat) / math.sqrt(2.0)) if math.isfinite(stat) else float("nan")
    return {"dm_mean_diff": float(d.mean()), "dm_stat": stat, "dm_p": p, "dm_n": d.size}


def block_bootstrap_ci(
    d: np.ndarray,
    block: int | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Moving-block bootstrap CI for the mean of a dependent differential series."""
    T = d.size
    if T < 5:
        return {"boot_lo": float("nan"), "boot_hi": float("nan"), "boot_mean": float("nan")}
    block = block or max(T // 20, 5)
    block = min(block, T)
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


def write_markdown_summary(
    metrics: pl.DataFrame,
    dm: pl.DataFrame,
    out_path: Path,
    title: str = "Baseline metrics summary",
) -> None:
    """Write a compact markdown summary of pooled metrics and DM tests."""
    lines: list[str] = [f"# {title}", ""]
    if "test" in metrics["split"].unique().to_list():
        lines.append(
            "Test-split numbers below are the headline; valid numbers are in-sample "
            "for the fuzzy/conformal calibration layers (parameters fit on valid)."
        )
        lines.append("")
    pooled = metrics.filter(pl.col("scope") == "POOLED").sort(["split", "h", "model"])
    for split in pooled["split"].unique(maintain_order=True).to_list():
        lines.append(f"## {split} (pooled)")
        lines.append("")
        lines.append("| h | model | n | pinball 0.1 | pinball 0.5 | pinball 0.9 | mean | cover80 | width | dir_hit |")
        lines.append("|---|-------|---|-------------|-------------|-------------|------|---------|-------|---------|")
        for r in pooled.filter(pl.col("split") == split).iter_rows(named=True):
            dh = "N/A" if r["dir_hit"] is None or (isinstance(r["dir_hit"], float) and math.isnan(r["dir_hit"])) else f"{r['dir_hit']:.3f}"
            lines.append(
                f"| {r['h']} | {r['model']} | {r['n']} | {r['pinball_0.1']:.4f} | {r['pinball_0.5']:.4f} "
                f"| {r['pinball_0.9']:.4f} | {r['pinball_mean']:.4f} | {r['cover_80']:.3f} | "
                f"{r['width']:.4f} | {dh} |"
            )
        lines.append("")
    if dm.height:
        lines.append("## Diebold-Mariano vs lstm_fuzzy (pinball, HAC lag h-1)")
        lines.append("")
        lines.append("| split | h | competitor | quantile | mean diff | DM stat | p | boot 2.5% | boot 97.5% |")
        lines.append("|-------|---|------------|----------|-----------|---------|---|-----------|------------|")
        for r in dm.sort(["split", "h", "quantile", "competitor"]).iter_rows(named=True):
            lines.append(
                f"| {r['split']} | {r['h']} | {r['competitor']} | {r['quantile']} | {r['dm_mean_diff']:+.5f} "
                f"| {r['dm_stat']:+.2f} | {r['dm_p']:.4f} | {r['boot_lo']:+.5f} | {r['boot_hi']:+.5f} |"
            )
        lines.append("")
        lines.append(
            "Negative mean diff means lstm_fuzzy has lower pinball than the competitor. "
            "Because 90-day windows overlap and the four classes are correlated, the "
            "effective sample size is small; treat the block-bootstrap interval, not the "
            "DM p-value alone, as the primary evidence."
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
