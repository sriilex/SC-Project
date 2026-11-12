"""Run the attribution baselines against the frozen LSTM quantile fan.

Each baseline is fit on the validation split with the same pinball objective and
the same bounded width/location maps as the Sugeno layer. The test split is read
once, in an explicit ``allow_test_set_access`` block, and only for final scoring.

Run from the repository root:

    BASELINES_INCLUDE_TEST=1 uv run python -m ml.attribution_baselines

Writes ``src/data/attribution_metrics.csv``, per-parameter JSON files and
``src/data/attribution_run_metadata.json``.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Final

import polars as pl

from ml.attribution import KINDS, apply_attribution, fit_attribution
from ml.baselines import evaluate, load_split
from ml.frozen_test import allow_test_set_access, load_frozen_test
from ml.predictions import ensure_lstm_predictions

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"
PARAM_DIR: Final[Path] = DATA / "attribution_params"
HORIZONS: Final[tuple[int, ...]] = (1, 7, 30, 90)


def _context_frame(
    base_preds: dict[str, dict[int, pl.DataFrame]],
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
        frames.append(base_preds[name][h].join(truth, on=["date", "target_class"], how="left"))
    return pl.concat(frames, how="diagonal")


def main() -> None:
    """Fit and score every attribution baseline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    include_test = os.environ.get("BASELINES_INCLUDE_TEST", "").strip().lower() in ("1", "true", "yes")

    splits: dict[str, pl.DataFrame] = {name: load_split(name) for name in ("train", "valid")}
    if include_test:
        with allow_test_set_access("final attribution-baseline table for the paper"):
            splits["test"] = load_frozen_test()
        LOGGER.warning("BASELINES_INCLUDE_TEST=1: test split is being scored")
    eval_splits = ["valid"] + (["test"] if "test" in splits else [])
    valid_start = splits["valid"]["date"].min()

    mean_preds, _per_seed, _meta = ensure_lstm_predictions(
        {s: splits[s] for s in ["train", *eval_splits]}
    )

    all_rows: list[dict[str, object]] = []
    params_out: dict[str, object] = {}
    PARAM_DIR.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        context_names = ["train", "valid"] + (["test"] if "test" in splits else [])
        context = _context_frame(mean_preds, splits, context_names, h)
        train_frame = _context_frame(mean_preds, splits, ["valid"], h)
        for kind in KINDS:
            params = fit_attribution(context, train_frame, h, kind, matured_from=valid_start)
            (PARAM_DIR / f"attr_{kind}_h{h}.json").write_text(json.dumps(params, indent=2))
            params_out[f"{kind}_h{h}"] = {
                k: params[k] for k in
                ("kind", "h", "n_train", "epochs", "lr", "m", "s", "rollbias_coef",
                 "width_coefs", "loc_coefs", "final_train_pinball")
            }
            for split_name in eval_splits:
                hist_names = ["train", "valid"]
                if split_name == "test" and "test" in splits:
                    hist_names.append("test")
                history = _context_frame(mean_preds, splits, hist_names, h)
                cal = apply_attribution(
                    mean_preds[split_name][h], history, h, params, matured_from=valid_start
                )
                all_rows.extend(evaluate(cal, splits[split_name], split_name, f"attr_{kind}"))
            LOGGER.info("h=%d %s: done", h, kind)

    metrics = pl.DataFrame(all_rows).sort(["split", "h", "model", "scope"])
    out_path = DATA / "attribution_metrics.csv"
    metrics.write_csv(out_path)
    LOGGER.info("wrote %s (%d rows)", out_path, metrics.height)

    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_included": include_test,
        "horizons": list(HORIZONS),
        "kinds": list(KINDS),
        "objective": "pinball_0.1_0.5_0.9",
        "fit_split": "valid",
        "base": "frozen LSTM quantile predictions",
        "bounds": "same tanh-bounded width and location maps as the Sugeno layer",
        "tuning_budget": "one fixed configuration per baseline (600 full-batch Adam epochs, lr 0.05); no per-baseline hyperparameter search",
        "params": params_out,
    }
    meta_path = DATA / "attribution_run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("wrote %s", meta_path)

    pooled = metrics.filter(pl.col("scope") == "POOLED").select(
        "split", "h", "model", "n", "pinball_mean", "pinball_0.5", "cover_80", "winkler_80"
    ).sort(["split", "h", "model"])
    print(pooled.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
