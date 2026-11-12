"""Reproducible fuzzy-on-CQR stacked experiment.

Protocol (chronological, test frozen):

1. Fit split-CQR on the validation split and apply it to train, validation and
   test. CQR is a static adjustment, so the same parameters are used everywhere.
2. Recompute the fuzzy signals from the *CQR-calibrated* residuals. Forecasts
   issued at ``t`` only see residuals matured by ``t``.
3. Fit the Sugeno layer on the validation split with the base (here, the
   CQR-calibrated fan) frozen.
4. Score validation and test. Test is read once.

The fuzzy stage does **not** inherit CQR's coverage guarantee: CQR is valid
under exchangeability, and the learned fuzzy transform is a further data-driven
map fitted on a finite validation window.

Run from the repository root:

    BASELINES_INCLUDE_TEST=1 uv run python -m ml.stack_experiment
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Final

import polars as pl

from ml.baselines import CLASSES, HORIZONS, evaluate, load_split
from ml.conformal import apply_cqr, fit_cqr
from ml.frozen_test import allow_test_set_access, load_frozen_test
from ml.fuzzy_calibration import calibrate as fuzzy_calibrate
from ml.fuzzy_calibration import fit_fuzzy, save_params
from ml.predictions import ensure_lstm_predictions

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"


def _context_frame(
    preds: dict[str, dict[int, pl.DataFrame]],
    splits: dict[str, pl.DataFrame],
    names: list[str],
    h: int,
) -> pl.DataFrame:
    frames = []
    for name in names:
        truth = splits[name].select(
            "date", "target_class", "log_value",
            pl.col(f"y_step_h{h}").alias("y"),
        )
        frames.append(preds[name][h].join(truth, on=["date", "target_class"], how="left"))
    return pl.concat(frames, how="diagonal")


def main() -> None:
    """Run the documented fuzzy-on-CQR protocol and write its tables."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    include_test = os.environ.get("BASELINES_INCLUDE_TEST", "").strip().lower() in ("1", "true", "yes")

    splits: dict[str, pl.DataFrame] = {name: load_split(name) for name in ("train", "valid")}
    if include_test:
        with allow_test_set_access("reproducible fuzzy-on-CQR stack experiment"):
            splits["test"] = load_frozen_test()
        LOGGER.warning("BASELINES_INCLUDE_TEST=1: test split is being scored")
    eval_splits = ["valid"] + (["test"] if "test" in splits else [])
    valid_start = splits["valid"]["date"].min()

    raw, _per_seed, _meta = ensure_lstm_predictions(
        {s: splits[s] for s in ["train", *eval_splits]}
    )

    rows: list[dict[str, object]] = []
    for h in HORIZONS:
        calib = _context_frame(raw, splits, ["valid"], h)
        cqr_fit = fit_cqr(calib)
        cqr_preds: dict[str, dict[int, pl.DataFrame]] = {s: {} for s in ["train", *eval_splits]}
        for split_name in ["train", *eval_splits]:
            cqr_preds[split_name][h] = apply_cqr(raw[split_name][h], cqr_fit)

        context_names = ["train", "valid"] + (["test"] if "test" in splits else [])
        context = _context_frame(cqr_preds, splits, context_names, h)
        train_frame = _context_frame(cqr_preds, splits, ["valid"], h)
        params, stats = fit_fuzzy(context, train_frame, h, matured_from=valid_start)
        save_params(params, f"stack_fuzzy_h{h}")

        for split_name in eval_splits:
            hist_names = ["train", "valid"]
            if split_name == "test" and "test" in splits:
                hist_names.append("test")
            history = _context_frame(cqr_preds, splits, hist_names, h)

            rows.extend(evaluate(raw[split_name][h], splits[split_name], split_name, "lstm"))
            rows.extend(evaluate(cqr_preds[split_name][h], splits[split_name], split_name, "cqr"))
            fuz = fuzzy_calibrate(
                cqr_preds[split_name][h], history, h, params, matured_from=valid_start
            )
            rows.extend(evaluate(fuz, splits[split_name], split_name, "fuz_cqr"))
        LOGGER.info(
            "h=%d stack done (n_train=%d, width_inputs=%s, loc_inputs=%s)",
            h, stats["n_train"], stats["width_inputs"], stats["loc_inputs"],
        )

    metrics = pl.DataFrame(rows).sort(["split", "h", "model", "scope"])
    path = DATA / "fuzzy_cqr_stack.csv"
    metrics.write_csv(path)
    LOGGER.info("wrote %s (%d rows)", path, metrics.height)

    h90 = metrics.filter(
        (pl.col("h") == 90)
        & (pl.col("model").is_in(["fuz_cqr", "cqr"]))
        & (pl.col("scope").is_in(CLASSES))
    ).sort(["scope", "model"])
    lines = [
        "# Fuzzy-on-CQR stack — h=90 per class (test)",
        "",
        "The fuzzy stage does **not** inherit CQR's coverage guarantee.",
        "",
        "| class | model | n | mean pinball | cover-80 | Winkler-80 |",
        "|-------|-------|---|--------------|----------|------------|",
    ]
    for r in h90.iter_rows(named=True):
        lines.append(
            f"| {r['scope']} | {r['model']} | {r['n']} | {r['pinball_mean']:.5f} "
            f"| {r['cover_80']:.4f} | {r['winkler_80']:.4f} |"
        )
    (DATA / "fuzzy_cqr_stack_h90.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_included": include_test,
        "horizons": list(HORIZONS),
        "cqr_split": "valid",
        "fuzzy_split": "valid",
        "fuzzy_base": "CQR-calibrated frozen LSTM fan",
        "matured_from": str(valid_start),
        "coverage_guarantee": "fuzzy stage does not inherit CQR's guarantee",
        "note": "post-hoc stacked arm; pre-register before treating as established",
    }
    (DATA / "fuzzy_cqr_stack_metadata.json").write_text(json.dumps(metadata, indent=2))
    pooled = metrics.filter(pl.col("scope") == "POOLED").select(
        "split", "h", "model", "n", "pinball_mean", "cover_80", "winkler_80"
    ).sort(["split", "h", "model"])
    print(pooled.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
