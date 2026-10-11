"""Cached data loaders.

Everything the UI reads comes from files the pipeline already wrote. The frozen
test split is only ever touched through :func:`load_split_cached` with an
explicit, non-empty reason, which is then handed to the project's
``allow_test_set_access`` guard. Read-only results CSVs need no guard.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import polars as pl
import streamlit as st

from lib.paths import CLASS_SERIES, DATA, PARAM_DIR, PRED_DIR

# ---------------------------------------------------------------------------
# Read-only result tables
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics() -> pl.DataFrame:
    """Full per-split/per-class metrics table (``baseline_metrics.csv``)."""
    return _read_csv(DATA / "baseline_metrics.csv")


@st.cache_data(show_spinner=False)
def load_metrics_common() -> pl.DataFrame:
    """Common-support metrics used by the paper's headline table."""
    return _read_csv(DATA / "baseline_metrics_common.csv")


@st.cache_data(show_spinner=False)
def load_dm() -> pl.DataFrame:
    """Diebold-Mariano + block-bootstrap comparisons vs ``lstm_fuzzy``."""
    return _read_csv(DATA / "baseline_dm.csv")


@st.cache_data(show_spinner=False)
def load_ablations() -> pl.DataFrame:
    """Exploratory fuzzy ablation arms."""
    return _read_csv(DATA / "fuzzy_ablations.csv")


@st.cache_data(show_spinner=False)
def load_robustness() -> pl.DataFrame:
    """Original vs lag-7 sample robustness run."""
    return _read_csv(DATA / "robustness_metrics.csv")


@st.cache_data(show_spinner=False)
def load_diagnostics() -> pl.DataFrame:
    """Interval diagnostics: coverage, width, crossings, per-quantile pinball."""
    return _read_csv(DATA / "interval_diagnostics.csv")


@st.cache_data(show_spinner=False)
def load_main_table() -> pl.DataFrame:
    """Compact common-support table written by ``ml.main_table``."""
    return _read_csv(DATA / "main_table.csv")


@st.cache_data(show_spinner=False)
def load_naive_sign() -> pl.DataFrame:
    return _read_csv(DATA / "naive_sign_metrics.csv")


@st.cache_data(show_spinner=False)
def load_manifest() -> pl.DataFrame:
    return _read_csv(DATA / "samples_manifest.csv")


@st.cache_data(show_spinner=False)
def load_shap() -> dict:

    path = DATA / "shap_feature_importance.json"
    if not path.exists():
        return {}
    import json as _json

    try:
        return _json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - malformed json must not kill the UI
        return {}


# ---------------------------------------------------------------------------
# Raw / master series
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_master_long() -> pl.DataFrame:
    path = DATA / "master_long.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_class_levels(log: bool = True) -> pl.DataFrame:
    """Wide frame ``[date, Capesize, Panamax, Supramax, Handysize, BDI]``.

    Raw ``master_long`` stores index *levels*; when ``log=True`` we return the
    natural log, which is the modelling space.
    """
    long = load_master_long()
    if long.is_empty():
        return pl.DataFrame()
    wanted = {**CLASS_SERIES, "BDI": "BD_INDEX"}
    frames = []
    for name, series in wanted.items():
        sub = (
            long.filter(pl.col("series_id") == series)
            .select("date", pl.col("value").alias(name))
            .sort("date")
        )
        if sub.is_empty():
            continue
        if log:
            sub = sub.with_columns(
                pl.when(pl.col(name) > 0)
                .then(pl.col(name).log())
                .otherwise(None)
                .alias(name)
            )
        frames.append(sub)
    if not frames:
        return pl.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="date", how="full", coalesce=True)
    return out.sort("date")


@st.cache_data(show_spinner=False)
def load_sample_features(split: str = "valid") -> pl.DataFrame:
    """Feature rows for ``train``/``valid``. Test is deliberately excluded."""
    if split not in ("train", "valid"):
        raise ValueError("load_sample_features only serves train/valid (test is gated)")
    path = DATA / f"samples_{split}.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Model predictions and learned parameters
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def prediction_cache_key() -> str | None:
    """Pick the LSTM cache key that has test predictions (the paper's run)."""
    keys = []
    for meta in glob.glob(str(PRED_DIR / "lstm_*_meta.json")):
        key = Path(meta).name.split("_")[1]
        keys.append(key)
    if not keys:
        return None
    # Prefer the run that produced the frozen test predictions.
    for key in sorted(keys):
        if (PRED_DIR / f"lstm_{key}_test_h1.parquet").exists():
            return key
    return max(keys)


@st.cache_data(show_spinner=False)
def load_base_pred(split: str, h: int, seed: int | None = None) -> pl.DataFrame:
    """Ensemble-mean (``seed=None``) or per-seed cached base quantiles."""
    key = prediction_cache_key()
    if key is None:
        return pl.DataFrame()
    suffix = "" if seed is None else f"_seed{seed}"
    path = PRED_DIR / f"lstm_{key}_{split}_h{h}{suffix}.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


@st.cache_data(show_spinner=False)
def available_seeds(h: int) -> tuple[int, ...]:
    key = prediction_cache_key()
    if key is None:
        return ()
    seeds = []
    for seed in range(10):
        if (PRED_DIR / f"lstm_{key}_valid_h{h}_seed{seed}.parquet").exists():
            seeds.append(seed)
    return tuple(seeds)


@st.cache_data(show_spinner=False)
def load_fuzzy_params(h: int) -> dict:
    """Learned, frozen Sugeno parameters for horizon ``h``."""
    path = PARAM_DIR / f"lstm_fuzzy_h{h}.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text())


def list_saved_param_files() -> list[Path]:
    if not PARAM_DIR.exists():
        return []
    return sorted(PARAM_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Splits (train/valid free, test gated)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_split_cached(name: str) -> pl.DataFrame:
    """Load a sample split.

    ``test`` goes through the project's frozen-test guard with an explicit
    reason. Callers must obtain user confirmation before reaching this branch.
    """
    if name == "test":
        from ml.frozen_test import allow_test_set_access, load_frozen_test

        with allow_test_set_access(
            "Streamlit UI: user explicitly requested frozen test scoring"
        ):
            return load_frozen_test()
    from ml.baselines import load_split

    return load_split(name)


def build_pred_frame(split: str, h: int) -> pl.DataFrame:
    return load_base_pred(split, h)


@st.cache_data(show_spinner=False)
def build_history(h: int, include_test: bool = False) -> pl.DataFrame:
    """Context frame (base quantiles + realised outcome + level) for fuzzy inputs.

    The fuzzy input computation only looks at forecast dates strictly before
    ``t``, so concatenating splits is causal rather than leaky.
    """
    names = ["train", "valid"] + (["test"] if include_test else [])
    frames = []
    for name in names:
        pred = load_base_pred(name, h)
        if pred.is_empty():
            continue
        split = load_split_cached(name)
        truth = split.select(
            "date", "target_class", "log_value", pl.col(f"y_step_h{h}").alias("y")
        )
        frames.append(pred.join(truth, on=["date", "target_class"], how="left"))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


@st.cache_data(show_spinner=False)
def truth_frame(split: str, h: int) -> pl.DataFrame:
    split_df = load_split_cached(split)
    if split_df.is_empty():
        return split_df
    return split_df.select(
        "date",
        "target_class",
        "log_value",
        pl.col(f"y_step_h{h}").alias("y"),
    ).drop_nulls("y")


@st.cache_data(show_spinner=False)
def data_health() -> dict:
    """Small dictionary of pipeline facts shown on the overview page."""
    health: dict[str, object] = {}
    health["pred_key"] = prediction_cache_key()
    for split in ("train", "valid"):
        try:
            df = load_split_cached(split)
            health[f"{split}_rows"] = df.height
            health[f"{split}_cols"] = df.width
        except Exception:  # noqa: BLE001 - health panel is best effort
            health[f"{split}_rows"] = None
    try:
        test_path = DATA / "samples_test.parquet"
        if test_path.exists():
            health["test_rows"] = (
                pl.scan_parquet(test_path).select(pl.len()).collect().item()
            )
    except Exception:  # noqa: BLE001
        health["test_rows"] = None
    for name in ("baseline_metrics.csv", "baseline_dm.csv", "fuzzy_ablations.csv"):
        health[name] = (DATA / name).exists()
    health["param_files"] = [p.name for p in list_saved_param_files()]
    health["env_test_flag"] = os.environ.get("BASELINES_INCLUDE_TEST", "")
    return health
