"""Main results table (common support) + h=90 per-class breakdown.

Reads ``baseline_metrics_common.csv`` (produced by ``ml.baselines``) so every
model is scored on exactly the same rows, and writes a compact markdown table
for the paper.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"

MODELS: Final[tuple[str, ...]] = ("lstm", "lstm_fuzzy", "cqr", "aci")
CLASSES: Final[tuple[str, ...]] = ("Capesize", "Panamax", "Supramax", "Handysize")


def _fmt(value: float | None, nd: int = 5) -> str:
    return "—" if value is None else f"{value:.{nd}f}"


def _load(scope: str) -> pl.DataFrame:
    m = pl.read_csv(DATA / "baseline_metrics_common.csv")
    return m.filter((pl.col("split") == "test") & (pl.col("model").is_in(MODELS)) & (pl.col("scope") == scope))


def pooled_table() -> tuple[pl.DataFrame, list[str]]:
    m = _load("POOLED")
    lines = [
        "| h | model | n | mean pinball | cover-80 | Winkler-80 |",
        "|---|-------|---|--------------|----------|------------|",
    ]
    for h in (1, 7, 30, 90):
        sub = m.filter(pl.col("h") == h)
        for model in MODELS:
            r = sub.filter(pl.col("model") == model)
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            lines.append(
                f"| {h} | {model} | {row['n']:.0f} | {_fmt(row['pinball_mean'])} | "
                f"{_fmt(row['cover_80'], 4)} | {_fmt(row['winkler_80'], 4)} |"
            )
    return m.select("h", "model", "n", "pinball_mean", "cover_80", "winkler_80"), lines


def h90_by_class() -> list[str]:
    full = pl.read_csv(DATA / "baseline_metrics_common.csv")
    by = full.filter(
        (pl.col("split") == "test") & (pl.col("h") == 90)
        & (pl.col("model").is_in(MODELS)) & (pl.col("scope").is_in(CLASSES))
    )
    lines = [
        "| class | model | n | mean pinball | cover-80 | Winkler-80 |",
        "|-------|-------|---|--------------|----------|------------|",
    ]
    for cls in CLASSES:
        for model in MODELS:
            r = by.filter((pl.col("scope") == cls) & (pl.col("model") == model))
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            lines.append(
                f"| {cls} | {model} | {row['n']:.0f} | {_fmt(row['pinball_mean'])} | "
                f"{_fmt(row['cover_80'], 4)} | {_fmt(row['winkler_80'], 4)} |"
            )
    return lines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pooled, lines = pooled_table()
    (DATA / "main_table.csv").write_text(pooled.write_csv(), encoding="utf-8")
    md = [
        "# Main table — test split, common support",
        "",
        (
            "All models are scored on the intersection of rows where every model emitted a "
            "prediction (the LSTM needs a warm-up window). Lower pinball is better."
        ),
        "",
        *lines,
        "",
        "## h=90 per class",
        "",
        *h90_by_class(),
        "",
    ]
    (DATA / "main_table.md").write_text("\n".join(md), encoding="utf-8")
    LOGGER.info("wrote %s and %s", DATA / "main_table.csv", DATA / "main_table.md")


if __name__ == "__main__":
    main()
