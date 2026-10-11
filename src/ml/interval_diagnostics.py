"""Interval-calibration diagnostics for the raw quantile LSTM vs rw_emp.

For each split (train, valid), horizon and scope (POOLED + the four classes)
reports below-P10, below-P90, central-80, mean interval width, P50 reliability,
quantile-crossing rates and per-quantile pinball loss. The LSTM and rw_emp are
scored on the *same* per-horizon rows (the horizon target is non-null and the
LSTM emitted a prediction), so widths and losses are like-for-like.

A second, common-support table reproduces the old all-horizons-non-null filter
as a robustness output.

Run from repo root:  uv run python -m ml.interval_diagnostics
Writes src/data/interval_diagnostics.csv and interval_diagnostics_common.csv
"""
from __future__ import annotations

import logging

import polars as pl

from ml.baselines import (
    CLASSES,
    DATA,
    HORIZONS,
    QUANTILES,
    fit_rw_emp,
    load_split,
    pinball,
    predict_rw_emp,
)
from ml.predictions import ensure_lstm_predictions

LOGGER = logging.getLogger(__name__)
SCOPES = ("POOLED", *CLASSES)


def _stats(j: pl.DataFrame) -> dict[str, float]:
    """Coverage / width / reliability / pinball stats for one horizon-scope slice."""
    y = j["y"]
    p10, p50, p90 = j["p_0.1"], j["p_0.5"], j["p_0.9"]
    losses = [pinball(y, j[f"p_{q}"], q) for q in QUANTILES]
    return {
        "n": float(j.height),
        "below_p10": float((y < p10).mean()),
        "below_p90": float((y <= p90).mean()),
        "cover_80": float(((y >= p10) & (y <= p90)).mean()),
        "width": float((p90 - p10).mean()),
        "rel_p50": float((y < p50).mean()),
        "x_p10_gt_p50": float((p10 > p50).mean()),
        "x_p10_gt_p90": float((p10 > p90).mean()),
        "pinball_0.1": losses[0],
        "pinball_0.5": losses[1],
        "pinball_0.9": losses[2],
    }


def _base_rate(j: pl.DataFrame) -> float:
    """Fraction of rows whose h-step outcome is above the current level."""
    return float((j["y"] > j["log_value"]).mean())


def _score_split(
    split: str,
    source: pl.DataFrame,
    lstm_preds: dict[int, pl.DataFrame],
    train_for_rw: pl.DataFrame,
    common_only: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for h in HORIZONS:
        truth = source.select(
            "date", "target_class", "log_value", pl.col(f"y_step_h{h}").alias("y")
        ).drop_nulls("y")
        lp = lstm_preds[h]
        if common_only:
            common = source.drop_nulls([f"y_step_h{h2}" for h2 in HORIZONS])
            truth = truth.join(common.select("date", "target_class"), on=["date", "target_class"], how="inner")
        # LSTM support = rows it predicted AND for which the target exists.
        base = truth.join(lp.select("date", "target_class"), on=["date", "target_class"], how="inner")
        rp = predict_rw_emp(source, h, fit_rw_emp(train_for_rw, h))
        for model_name, pred in (("lstm", lp), ("rw_emp", rp)):
            j_all = base.join(pred, on=["date", "target_class"], how="inner").drop_nulls("y")
            for scope in SCOPES:
                sub = j_all if scope == "POOLED" else j_all.filter(pl.col("target_class") == scope)
                if sub.is_empty():
                    continue
                rows.append({"split": split, "h": h, "scope": scope, "model": model_name, **_stats(sub)})
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    splits = {name: load_split(name) for name in ("train", "valid")}
    preds, per_seed, _meta = ensure_lstm_predictions(splits)

    rows: list[dict[str, object]] = []
    for split in ("train", "valid"):
        rows.extend(_score_split(split, splits[split], preds[split], splits["train"], common_only=False))
    out = pl.DataFrame(rows).sort(["split", "h", "scope", "model"])
    path = DATA / "interval_diagnostics.csv"
    out.write_csv(path)
    LOGGER.info("wrote %s (%d rows)", path, out.height)

    # Robustness: the old all-horizons-non-null common support.
    common_rows: list[dict[str, object]] = []
    for split in ("train", "valid"):
        common_rows.extend(_score_split(split, splits[split], preds[split], splits["train"], common_only=True))
    common = pl.DataFrame(common_rows).sort(["split", "h", "scope", "model"])
    common_path = DATA / "interval_diagnostics_common.csv"
    common.write_csv(common_path)
    LOGGER.info("wrote %s (%d rows)", common_path, common.height)

    for split in ("train", "valid"):
        LOGGER.info("\n=== %s pooled ===\n%s", split, out.filter(
            (pl.col("split") == split) & (pl.col("scope") == "POOLED")
        ).to_pandas().to_string(index=False))

    # Single-seed spread (Task A): crossing rate and central-80 per seed.
    spread_rows: list[dict[str, object]] = []
    for split in ("train", "valid"):
        for h in HORIZONS:
            for seed, frame in per_seed[split][h].items():
                truth = splits[split].select(
                    "date", "target_class", "log_value", pl.col(f"y_step_h{h}").alias("y")
                ).drop_nulls("y")
                j = truth.join(frame, on=["date", "target_class"], how="inner").drop_nulls("y")
                if j.is_empty():
                    continue
                spread_rows.append({
                    "split": split, "h": h, "seed": seed, "n": j.height,
                    "cover_80": float(((j["y"] >= j["p_0.1"]) & (j["y"] <= j["p_0.9"])).mean()),
                    "width": float((j["p_0.9"] - j["p_0.1"]).mean()),
                    "x_p10_gt_p50": float((j["p_0.1"] > j["p_0.5"]).mean()),
                    "pinball_0.5": pinball(j["y"], j["p_0.5"], 0.5),
                })
    spread = pl.DataFrame(spread_rows)
    spread_path = DATA / "interval_diagnostics_seed_spread.csv"
    spread.write_csv(spread_path)
    if not spread.is_empty():
        agg = spread.group_by(["split", "h"]).agg(
            pl.col("cover_80").min().alias("cover_min"),
            pl.col("cover_80").max().alias("cover_max"),
            pl.col("pinball_0.5").std().alias("pinball_p50_sd"),
        ).sort(["split", "h"])
        LOGGER.info("\n=== single-seed spread ===\n%s", agg.to_pandas().to_string(index=False))

    # Always-up base rate on the per-horizon support.
    LOGGER.info("\n=== always-up rate P(y_t+h > y_t) ===")
    br_rows: list[dict[str, object]] = []
    for split in ("train", "valid"):
        for h in HORIZONS:
            support = splits[split].select(
                "date", "target_class", "log_value", pl.col(f"y_step_h{h}").alias("y")
            ).drop_nulls("y")
            support = support.join(preds[split][h].select("date", "target_class"), on=["date", "target_class"], how="inner")
            for scope in SCOPES:
                sub = support if scope == "POOLED" else support.filter(pl.col("target_class") == scope)
                if sub.is_empty():
                    continue
                br_rows.append({"split": split, "h": h, "scope": scope, "n": sub.height, "up_rate": _base_rate(sub)})
    br = pl.DataFrame(br_rows)
    br.write_csv(DATA / "interval_diagnostics_always_up.csv")
    for split in ("train", "valid"):
        LOGGER.info("\n%s\n%s", split, br.filter(pl.col("split") == split).pivot(
            on="scope", index="h", values="up_rate"
        ).to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
