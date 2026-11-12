"""Recompute POOLED metrics for the 3-class panel (Capesize, Panamax, Supramax).

The existing baseline_metrics.csv has per-class rows that were computed on
individual rows, and a POOLED row computed over all 4 classes (including
Handysize). This script produces a 3-class POOLED by weighting per-class
means by n, then saves it back as an updated column.

Usage:
    uv run python src/brain/.../scratch/recompute_3class_pooled.py
"""
from __future__ import annotations

import polars as pl
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]  # SC-Project root
DATA = REPO / "src" / "data"

CLASSES_3 = ("Capesize", "Panamax", "Supramax")

# Metric columns that are means (can be re-pooled as weighted mean)
MEAN_COLS = [
    "pinball_0.1", "pinball_0.5", "pinball_0.9", "pinball_mean",
    "below_p10", "below_p90", "cover_80", "width", "winkler_80",
    "mae_p50", "dir_hit", "x_p10_gt_p50", "x_p10_gt_p90",
]

def recompute(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    # Per-class rows for the 3 classes
    cls_rows = df.filter(pl.col("scope").is_in(CLASSES_3))

    # Weighted mean: sum(metric_i * n_i) / sum(n_i) for each (split, h, model)
    weighted = (
        cls_rows
        .with_columns([
            (pl.col(c) * pl.col("n")).alias(f"_w_{c}") for c in MEAN_COLS
        ])
        .group_by(["split", "h", "model"])
        .agg(
            pl.col("n").sum().alias("n"),
            *[(pl.col(f"_w_{c}").sum() / pl.col("n").sum()).alias(c) for c in MEAN_COLS],
        )
        .with_columns(pl.lit("POOLED_3CLS").alias("scope"))
    )

    return weighted


def main() -> None:
    for name in ("baseline_metrics.csv", "baseline_metrics_common.csv"):
        path = DATA / name
        if not path.exists():
            print(f"  skipping {name} (not found)")
            continue
        df = pl.read_csv(path)
        new_pooled = recompute(path)

        # Print comparison: old POOLED vs new POOLED_3CLS
        old_pooled = df.filter(
            (pl.col("scope") == "POOLED") & (pl.col("split") == "test")
        ).sort(["h", "model"])

        new_p = new_pooled.filter(pl.col("split") == "test").sort(["h", "model"])

        print(f"\n=== {name} ===")
        print("\n--- OLD POOLED (4 classes) ---")
        print(old_pooled.select(["h", "model", "n", "pinball_mean", "cover_80", "winkler_80"]))
        print("\n--- NEW POOLED_3CLS (Capesize, Panamax, Supramax) ---")
        print(new_p.select(["h", "model", "n", "pinball_mean", "cover_80", "winkler_80"]))

        # Delta
        joined = old_pooled.select(["h", "model", "pinball_mean", "cover_80", "winkler_80"]).join(
            new_p.select(["h", "model", "pinball_mean", "cover_80", "winkler_80"]),
            on=["h", "model"], suffix="_3cls"
        )
        print("\n--- Δ (3cls - 4cls) ---")
        print(
            joined.with_columns([
                (pl.col("pinball_mean_3cls") - pl.col("pinball_mean")).alias("Δ_pinball"),
                (pl.col("cover_80_3cls") - pl.col("cover_80")).alias("Δ_cover"),
                (pl.col("winkler_80_3cls") - pl.col("winkler_80")).alias("Δ_winkler"),
            ]).select(["h", "model", "Δ_pinball", "Δ_cover", "Δ_winkler"])
        )


if __name__ == "__main__":
    main()
