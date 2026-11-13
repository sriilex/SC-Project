"""Reserve a longer calibration period and generate genuinely OOS forecasts.

uv run python -m ml.long_calibration --build-only
uv run python -m ml.long_calibration --train-only

Base: 2012-07 through 2019-07; calibration: 2020-01 through 2024-07.
The existing feature builder shifts on a continuous calendar before masking.
No frozen test split is produced, loaded, or scored. This is a separate one-seed
exploratory base, not the five-seed base in the original paper's test table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import timedelta
from pathlib import Path

import polars as pl
import torch

from ml.datasets.build_samples import load_config
from ml.datasets.build_samples import main as build_samples
from ml.models.lstm import LSTMModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src/config/samples_long_calibration.toml"
LOG = logging.getLogger(__name__)


def run(*, build_only: bool = False, train_only: bool = False) -> Path:
    if build_only and train_only:
        raise ValueError("build_only and train_only are mutually exclusive")
    cfg = load_config(CONFIG)
    train_path = ROOT / cfg["paths"]["samples_train"]
    valid_path = ROOT / cfg["paths"]["samples_valid"]
    out = train_path.parent
    if not train_only:
        if train_path.exists() or valid_path.exists():
            raise FileExistsError(
                "long calibration samples already exist; use --train-only"
            )
        build_samples(CONFIG, include_test=False)
    train = pl.read_parquet(train_path).drop_nulls("log_value")
    valid = pl.read_parquet(valid_path).drop_nulls("log_value")
    if train["date"].max() + timedelta(days=90) >= valid["date"].min():
        raise ValueError("base-training labels overlap calibration")
    if set(train["target_class"].unique().to_list()) != {
        "Capesize",
        "Panamax",
        "Supramax",
    }:
        raise ValueError("expected the three-class panel")
    metadata = {
        "config": str(CONFIG.relative_to(ROOT)),
        "test_read": False,
        "base_train_start": str(train["date"].min()),
        "base_train_end": str(train["date"].max()),
        "calibration_start": str(valid["date"].min()),
        "calibration_end": str(valid["date"].max()),
        "base_train_n": train.height,
        "calibration_n": valid.height,
        "base_seeds": [0],
        "base_max_epochs": 100,
        "base_patience": 8,
        "base_hidden": 32,
        "train_sha256": hashlib.sha256(
            train.sort(["date", "target_class"]).serialize()
        ).hexdigest(),
        "valid_sha256": hashlib.sha256(
            valid.sort(["date", "target_class"]).serialize()
        ).hexdigest(),
        "status": "samples_built",
    }
    if build_only:
        (out / "split_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return out
    pred_dir = out / "preds"
    pred_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:12]
    metadata["prediction_key"] = key
    metadata["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    meta_path = out / "base_metadata.json"
    if meta_path.exists():
        raise FileExistsError(
            "base training has already been attempted; do not overwrite its evidence"
        )
    metadata["status"] = "training"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.monotonic()
    for h in (1, 7, 30, 90):
        LOG.info("Long-calibration base: starting h=%s", h)
        model = LSTMModel(
            max_epochs=100, patience=8, hidden=32, seeds=(0,), checkpoint=False
        )
        model.fit(train, [h])
        pred = model.predict({"valid": valid})["valid"][h]
        pred.write_parquet(pred_dir / f"lstm_{key}_valid_h{h}.parquet")
        torch.save(model, pred_dir / f"lstm_{key}_h{h}_model.pt")
        metadata[f"h{h}"] = {
            "n_predictions": pred.height,
            "first_issue": str(pred["date"].min()),
            "last_issue": str(pred["date"].max()),
            "elapsed_s": time.monotonic() - started,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOG.info(
            "Long-calibration base: finished h=%s after %.1fs",
            h,
            time.monotonic() - started,
        )
    metadata["status"] = "completed"
    metadata["elapsed_s"] = time.monotonic() - started
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(run(build_only=args.build_only, train_only=args.train_only))


if __name__ == "__main__":
    main()
