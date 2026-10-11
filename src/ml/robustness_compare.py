"""Compare the main valid metrics with the congestion-lag / macro-data rebuild.

The main run used the original samples (same-day congestion, percentage-change
macro features). The robustness build applies a 7-day congestion lag and the
direct ``level(t) - level(t-90)`` macro change. Both are reported as deltas on
the valid split; test is not re-read.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"


def _pooled(path: Path) -> pl.DataFrame:
    m = pl.read_csv(path)
    return m.filter((pl.col("split") == "valid") & (pl.col("scope") == "POOLED")).select(
        "h", "model", "n", "pinball_0.1", "pinball_0.5", "pinball_0.9",
        "pinball_mean", "cover_80", "width", "dir_hit",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main_df = _pooled(DATA / "main_run" / "baseline_metrics.csv")
    lag_df = _pooled(DATA / "baseline_metrics.csv")
    joined = main_df.join(lag_df, on=["h", "model"], how="inner", suffix="_lag7")
    out = joined.with_columns(
        (pl.col("pinball_mean_lag7") - pl.col("pinball_mean")).alias("d_pinball_mean"),
        (pl.col("cover_80_lag7") - pl.col("cover_80")).alias("d_cover_80"),
        (pl.col("width_lag7") - pl.col("width")).alias("d_width"),
        (pl.col("pinball_0.5_lag7") - pl.col("pinball_0.5")).alias("d_pinball_0.5"),
    ).sort(["h", "model"])
    path = DATA / "robustness_metrics.csv"
    out.write_csv(path)
    LOGGER.info("wrote %s (%d rows)", path, out.height)

    lines = [
        "# Data robustness: 7-day congestion lag + direct macro difference (valid)",
        "",
        "Deltas are `rebuild - main`. Positive pinball/width means the rebuild is worse.",
        "Both data changes are applied together in this single rebuild.",
        "",
        "| h | model | d pinball mean | d pinball 0.5 | d cover80 | d width |",
        "|---|-------|----------------|---------------|-----------|---------|",
    ]
    for r in out.iter_rows(named=True):
        lines.append(
            f"| {r['h']} | {r['model']} | {r['d_pinball_mean']:+.5f} | {r['d_pinball_0.5']:+.5f} "
            f"| {r['d_cover_80']:+.4f} | {r['d_width']:+.5f} |"
        )
    (DATA / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
