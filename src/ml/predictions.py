"""Prediction cache for the quantile LSTM.

Training the five-seed ensemble for four horizons is expensive, and the
diagnostics, baseline suite and fuzzy calibration all consume the *same* raw
predictions. This module fits once and caches the ensemble-mean and per-seed
quantile frames under ``src/data/preds/``. The fitted model itself is cached too,
so adding a split (e.g. the frozen test set) predicts only the missing frames
instead of retraining.

The cache key hashes the model/training configuration *and* the value
fingerprint of the supplied frames, so changing seeds, epochs, hidden size or
rebuilding ``samples_*.parquet`` forces a refit instead of silently reusing
stale predictions.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Final

import polars as pl
import torch

from ml.models.lstm import DEFAULT_SEEDS, HORIZONS, LSTMModel

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PRED_DIR: Final[Path] = REPO_ROOT / "src" / "data" / "preds"

#: Bump when the model definition or training semantics change.
CACHE_VERSION: Final[str] = "lstm-v3-monotone-standardised"


def _data_fingerprint(splits: dict[str, pl.DataFrame]) -> str:
    """Order-independent value fingerprint of the frames used for prediction."""
    parts: list[str] = []
    for name in sorted(splits):
        df = splits[name]
        parts.append(f"{name}:{df.height}x{df.width}")
        for c in sorted(df.columns):
            parts.append(f"{c}:{df[c].dtype}")
            try:
                parts.append(str(int(df[c].hash().sum())))
            except Exception:  # noqa: BLE001 - unsupported dtype must not kill the cache key
                parts.append("0")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _config_key(
    horizons: tuple[int, ...],
    seeds: tuple[int, ...],
    max_epochs: int,
    hidden: int,
    data_hash: str,
) -> str:
    payload = json.dumps(
        {
            "version": CACHE_VERSION,
            "horizons": list(horizons),
            "seeds": list(seeds),
            "max_epochs": max_epochs,
            "hidden": hidden,
            "data": data_hash,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _frame_path(key: str, split: str, h: int, seed: int | None = None) -> Path:
    suffix = "" if seed is None else f"_seed{seed}"
    return PRED_DIR / f"lstm_{key}_{split}_h{h}{suffix}.parquet"


def _load_model(key: str) -> LSTMModel | None:
    path = PRED_DIR / f"lstm_{key}_model.pt"
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must not kill the run
        LOGGER.warning("could not load cached LSTM model %s: %s", path, exc)
        return None


def ensure_lstm_predictions(
    splits: dict[str, pl.DataFrame],
    horizons: tuple[int, ...] = HORIZONS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_epochs: int = 100,
    hidden: int = 32,
    refresh: bool = False,
) -> tuple[
    dict[str, dict[int, pl.DataFrame]],
    dict[str, dict[int, dict[int, pl.DataFrame]]],
    dict[str, object],
]:
    """Return ``(ensemble_mean, per_seed, meta)`` for the requested splits.

    Only ``splits["train"]`` is used for fitting; every supplied split is
    predicted. Missing split predictions are filled from the cached model when
    possible.
    """
    key = _config_key(tuple(horizons), tuple(seeds), max_epochs, hidden, _data_fingerprint(splits))
    meta_path = PRED_DIR / f"lstm_{key}_meta.json"

    def _all_present() -> bool:
        return all(_frame_path(key, s, h).exists() for s in splits for h in horizons)

    if refresh or not _all_present():
        model = None if refresh else _load_model(key)
        if model is None:
            LOGGER.info("fitting LSTM (key=%s, seeds=%s)", key, list(seeds))
            model = LSTMModel(max_epochs=max_epochs, hidden=hidden, seeds=tuple(seeds))
            model.fit(splits["train"], list(horizons))
            PRED_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model, PRED_DIR / f"lstm_{key}_model.pt")
        else:
            LOGGER.info("reusing cached LSTM model (key=%s) for missing splits", key)

        PRED_DIR.mkdir(parents=True, exist_ok=True)
        need = [s for s in splits if not all(_frame_path(key, s, h).exists() for h in horizons)]
        mean_new = model.predict({s: splits[s] for s in need})
        per_seed_new = model.predict_per_seed({s: splits[s] for s in need})
        for s in need:
            for h in horizons:
                mean_new[s][h].write_parquet(_frame_path(key, s, h))
                for seed, frame in per_seed_new[s][h].items():
                    frame.write_parquet(_frame_path(key, s, h, seed))

        meta = {
            "key": key,
            "version": CACHE_VERSION,
            "horizons": list(horizons),
            "seeds": list(seeds),
            "max_epochs": max_epochs,
            "hidden": hidden,
            "target_stats": {
                str(h): {c: list(v) for c, v in stats.items()}
                for h, stats in model.target_stats.items()
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    else:
        LOGGER.info("loaded cached LSTM predictions (key=%s)", key)
        meta = json.loads(meta_path.read_text())

    mean_preds: dict[str, dict[int, pl.DataFrame]] = {s: {} for s in splits}
    per_seed: dict[str, dict[int, dict[int, pl.DataFrame]]] = {s: {} for s in splits}
    for s in splits:
        for h in horizons:
            mean_preds[s][h] = pl.read_parquet(_frame_path(key, s, h))
            per_seed[s][h] = {
                seed: pl.read_parquet(_frame_path(key, s, h, seed))
                for seed in seeds
                if _frame_path(key, s, h, seed).exists()
            }
    return mean_preds, per_seed, meta
