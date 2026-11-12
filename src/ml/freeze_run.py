"""Freeze the current experimental outputs as a named reference run.

Copies the current CSVs and fuzzy parameter files into ``src/data/runs/v1/`` and
writes a metadata JSON with the configuration constants that produced them.
Later runs can write to ``runs/<name>/`` instead of overwriting the numbers that
are quoted in the paper.

Run from the repository root:

    uv run python -m ml.freeze_run
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Final

import polars as pl

from ml.conformal import ACI_GAMMA, ACI_SCORE_BUFFER, ACI_WARMUP
from ml.fuzzy_calibration import (
    M_SCALE,
    MIN_MATURED,
    N_MATURED,
    N_MATURED_INDEPENDENT,
    S_SCALE_PER_SD,
    VOL_WINDOW,
    matured_window,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"
RUN_DIR: Final[Path] = DATA / "runs" / "v1"

ARTIFACTS: Final[tuple[str, ...]] = (
    "baseline_metrics.csv",
    "baseline_metrics_common.csv",
    "baseline_dm.csv",
    "baseline_summary.md",
    "main_table.csv",
    "main_table.md",
    "aci_diagnostics.csv",
    "fuzzy_ablations.csv",
    "fuzzy_seed_spread.csv",
    "fuzzy_cqr_stack.csv",
    "naive_sign_metrics.csv",
    "interval_diagnostics.csv",
    "interval_diagnostics_common.csv",
    "interval_diagnostics_seed_spread.csv",
    "interval_diagnostics_always_up.csv",
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ARTIFACTS:
        src = DATA / name
        if src.exists():
            shutil.copy2(src, RUN_DIR / name)
            copied.append(name)

    params_src = DATA / "fuzzy_params"
    params_dst = RUN_DIR / "fuzzy_params"
    if params_src.exists():
        shutil.copytree(params_src, params_dst, dirs_exist_ok=True)
        copied.append("fuzzy_params/")

    valid_start = None
    valid_path = DATA / "samples_valid.parquet"
    if valid_path.exists():
        valid_start = str(pl.read_parquet(valid_path, columns=["date"])["date"].min())

    metadata = {
        "run": "v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": "Reference run quoted in the first paper draft. New runs get a new named directory.",
        "lstm_prediction_key": "69e158d2977b",
        "splits": {
            "train": ["2012-07-04", "2022-07-25"],
            "valid": ["2023-01-03", "2024-07-25"],
            "test": ["2025-01-02", "2026-08-20"],
            "embargo_days": 160,
        },
        "horizons": [1, 7, 30, 90],
        "seeds": [0, 1, 2, 3, 4],
        "lstm": {"max_epochs": 100, "hidden": 32, "ensemble_seeds": 5},
        "calibration": {
            "M_SCALE": M_SCALE,
            "S_SCALE_PER_SD": S_SCALE_PER_SD,
            "N_MATURED": N_MATURED,
            "N_MATURED_INDEPENDENT": N_MATURED_INDEPENDENT,
            "matured_window_by_h": {str(h): matured_window(h) for h in (1, 7, 30, 90)},
            "MIN_MATURED": MIN_MATURED,
            "VOL_WINDOW": VOL_WINDOW,
            "burn_in_valid_start": valid_start,
            "optimizer": "full-batch Adam",
            "lr": 0.05,
            "epochs": 600,
            "initial_consequents": 0.0,
        },
        "conformal": {
            "cqr_alpha": 0.2,
            "aci": {
                "gamma": ACI_GAMMA,
                "warmup": ACI_WARMUP,
                "score_buffer": ACI_SCORE_BUFFER,
                "feedback": "matured only (maturity queue)",
            },
        },
        "bootstrap": {
            "primary": "date-block bootstrap over consecutive dates, all classes together",
            "block_length": "max(h, 5)",
            "sensitivity": ["half the primary length", "twice the primary length"],
        },
        "artifact_files": copied,
    }
    meta_path = RUN_DIR / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("froze run v1 in %s (%d artifacts)", RUN_DIR, len(copied))
    LOGGER.info("wrote %s", meta_path)


if __name__ == "__main__":
    main()
