"""Pooled quantile baselines over the sample splits.

Benchmarks and calibration layers:
  rw       persistence: point = last log level, bands from train RW residual std
  rw_emp   random walk with per-class empirical return quantiles (historical sim)
  ar1      pooled AR(1) with drift on log returns (momentum-aware median)
  lgbm     pooled LightGBM quantile models (one per horizon x quantile)
  xgb      pooled XGBoost quantile models
  lstm     five-seed monotone quantile LSTM ensemble
  lstm_fuzzy       new Sugeno layer (Task F)
  lstm_fuzzy_old   original constant-shift fuzzy layer (ablation)
  const_shift      raw LSTM + median residual shift (calibration split)
  cqr              split conformalized quantile regression
  aci              adaptive conformal inference / online quantile tracking

Each horizon is scored on its own mask (that horizon's target non-null and a
prediction present). The harness only reads the frozen test split when
``BASELINES_INCLUDE_TEST=1`` and wrapped in ``allow_test_set_access``.

Writes src/data/baseline_metrics.csv and src/data/naive_sign_metrics.csv.
Run with `uv run baselines` from the repository root.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import polars as pl

from ml.conformal import (
    aci_adjust,
    apply_const_shift,
    apply_cqr,
    fit_const_shift,
    fit_cqr,
)
from ml.evaluation import (
    aligned_losses,
    block_bootstrap_ci,
    dm_test,
    write_markdown_summary,
)
from ml.frozen_test import allow_test_set_access, load_frozen_test
from ml.fuzzy_calibration import calibrate as fuzzy_calibrate
from ml.fuzzy_calibration import describe_params, fit_fuzzy, plot_fuzzy, save_params
from ml.fuzzy_calibration_old import calibrate as fuzzy_calibrate_old

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"
HORIZONS: Final[tuple[int, ...]] = (1, 7, 30, 90)
QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)
ZSCORES: Final[dict[float, float]] = {0.1: -1.2816, 0.5: 0.0, 0.9: 1.2816}
CLASSES: Final[tuple[str, ...]] = ("Capesize", "Panamax", "Supramax", "Handysize")
CLASS_CODES: Final[dict[str, int]] = {c: i for i, c in enumerate(CLASSES)}
EXCLUDE: Final[frozenset[str]] = frozenset(
    {"date", "target_value", "log_value", "target_class"}
    | {f"y_step_h{h}" for h in HORIZONS}
    | {f"y_mean_h{h}" for h in HORIZONS}
    | {f"target_ret_h{h}" for h in HORIZONS}
)

# Tree models keep the encoded class as a feature, so they exclude one column less.
TREE_EXCLUDE: Final[frozenset[str]] = frozenset(
    {"date", "target_value", "log_value"}
    | {f"y_step_h{h}" for h in HORIZONS}
    | {f"y_mean_h{h}" for h in HORIZONS}
    | {f"target_ret_h{h}" for h in HORIZONS}
)


def _setup_logging() -> None:
    """Configure root logging once for CLI runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_split(name: str) -> pl.DataFrame:
    """Load one sample split, normalize NaN to null.

    Only ``log_value`` is required non-null. Rows whose *other* horizon targets
    are null are kept: dropping rows for all horizons at once discarded ~64% of
    the validation set. Each horizon is scored on its own mask via
    :func:`evaluate`.
    """
    df = pl.read_parquet(DATA / f"samples_{name}.parquet")
    floats = [c for c, t in df.schema.items() if t == pl.Float64]
    df = df.with_columns([pl.col(c).fill_nan(None) for c in floats])
    return df.drop_nulls(["log_value"])


def horizon_mask(df: pl.DataFrame, h: int) -> pl.DataFrame:
    """Rows usable for horizon ``h``: a non-null h-step target."""
    return df.drop_nulls([f"y_step_h{h}"])


def pinball(y: pl.Series, p: pl.Series, q: float) -> float:
    """Mean pinball (quantile) loss for quantile level q."""
    err = y - p
    return float(
        pl.select(pl.when(err >= 0).then(q * err).otherwise((q - 1.0) * err).mean()).item()
    )


def _clean_sigma(s: object, fallback: float) -> float:
    """Return a usable sigma or the fallback; handles None and NaN."""
    if s is None:
        return fallback
    s_f = float(s)
    return s_f if s_f == s_f and s_f > 0 else fallback


def fit_rw_sigmas(train: pl.DataFrame, h: int) -> dict[str, float]:
    """Per-class random-walk residual sigma on train, pooled fallback."""
    res = train.select(
        pl.col("target_class"),
        (pl.col(f"y_step_h{h}") - pl.col("log_value")).alias("e"),
    ).drop_nulls()
    pooled = float(res["e"].std())
    out = {
        row["target_class"]: _clean_sigma(row["s"], pooled)
        for row in res.group_by("target_class").agg(pl.col("e").std().alias("s"))
        .iter_rows(named=True)
    }
    out["POOLED"] = pooled
    return out


def predict_rw(df: pl.DataFrame, h: int, sigmas: dict[str, float]) -> pl.DataFrame:
    """Point + quantile columns for the persistence baseline."""
    sigma = pl.col("target_class").replace_strict(sigmas, default=sigmas["POOLED"])
    return df.select(
        pl.col("date"),
        pl.col("target_class"),
        pl.lit(h).cast(pl.Int64).alias("h"),
        pl.col("log_value").alias("p_0.5"),
        (pl.col("log_value") + ZSCORES[0.1] * sigma).alias("p_0.1"),
        (pl.col("log_value") + ZSCORES[0.9] * sigma).alias("p_0.9"),
    )


def _rw_return_quantiles(df: pl.DataFrame, h: int) -> dict[float, float]:
    """Empirical quantiles of h-step returns, pooled across classes."""
    e = (df[f"y_step_h{h}"] - df["log_value"]).drop_nulls()
    if e.is_empty():
        return dict.fromkeys(QUANTILES, 0.0)
    return {q: float(e.quantile(q)) for q in QUANTILES}


def fit_rw_emp(train: pl.DataFrame, h: int) -> dict[str, dict[float, float]]:
    """Per-class empirical h-step return quantiles observed in train."""
    d = train.select(
        pl.col("target_class"),
        (pl.col(f"y_step_h{h}") - pl.col("log_value")).alias("e"),
    ).drop_nulls("e")
    qmap: dict[str, dict[float, float]] = {}
    for row in (
        d.group_by("target_class")
        .agg([pl.col("e").quantile(q).alias(f"q{q}") for q in QUANTILES])
        .iter_rows(named=True)
    ):
        qmap[str(row["target_class"])] = {q: float(row[f"q{q}"]) for q in QUANTILES}
    qmap["POOLED"] = _rw_return_quantiles(train, h)
    return qmap


def predict_rw_emp(
    df: pl.DataFrame, h: int, qmap: dict[str, dict[float, float]]
) -> pl.DataFrame:
    """Random-walk with empirical return quantiles: P50 = current log level."""
    class_map = {k: v for k, v in qmap.items() if k != "POOLED"}
    exprs = []
    for q in QUANTILES:
        per_class = {k: v[q] for k, v in class_map.items()}
        offset = pl.col("target_class").replace_strict(
            per_class, default=qmap["POOLED"][q]
        )
        exprs.append((pl.col("log_value") + offset).alias(f"p_{q}"))
    return df.select(
        pl.col("date"),
        pl.col("target_class"),
        pl.lit(h).cast(pl.Int64).alias("h"),
        *exprs,
    )


fit_rw_empirical = fit_rw_emp
predict_rw_empirical = predict_rw_emp


def naive_sign_metrics(df: pl.DataFrame, split: str, h: int) -> list[dict[str, object]]:
    """Hit rate of persisting sign(r_t) as a forecast for sign(r_{t+h})."""
    d = (
        df.sort(["target_class", "date"])
        .with_columns(pl.col("log_value").diff().over("target_class").alias("r_t"))
        .drop_nulls(["r_t", "log_value", f"y_step_h{h}"])
    )
    rows: list[dict[str, object]] = []
    for scope in ("POOLED", *CLASSES):
        sub = d if scope == "POOLED" else d.filter(pl.col("target_class") == scope)
        if sub.is_empty():
            continue
        hit = ((sub["r_t"] > 0) == (sub[f"y_step_h{h}"] > sub["log_value"])).mean()
        rows.append(
            {
                "split": split,
                "h": h,
                "model": "naive_sign",
                "scope": scope,
                "n": sub.height,
                "dir_hit": float(hit),
            }
        )
    return rows


def fit_ar1(train: pl.DataFrame, h: int) -> dict[str, object]:
    """Pooled AR(1) with drift on *log returns*, plus per-class residual sigma."""
    d = (
        train.sort(["target_class", "date"])
        .with_columns(pl.col("log_value").diff().over("target_class").alias("r_t"))
        .drop_nulls(["r_t", f"y_step_h{h}"])
    )
    x = d["r_t"].to_numpy()
    y = (d[f"y_step_h{h}"] - d["log_value"]).to_numpy()
    b = float(((x * y).mean() - x.mean() * y.mean()) / ((x * x).mean() - x.mean() ** 2))
    a = float(y.mean() - b * x.mean())
    resid = y - (a + b * x)
    pooled = float(resid.std())
    sigmas = {
        row["target_class"]: _clean_sigma(row["s"], pooled)
        for row in d.with_columns(pl.lit(resid).alias("r"))
        .group_by("target_class").agg(pl.col("r").std().alias("s"))
        .iter_rows(named=True)
    }
    sigmas["POOLED"] = pooled
    return {"a": a, "b": b, "sigmas": sigmas}


def predict_ar1(df: pl.DataFrame, h: int, model: dict[str, object]) -> pl.DataFrame:
    """Point + quantile columns for the return-space AR(1) baseline."""
    a, b = float(model["a"]), float(model["b"])
    sigmas: dict[str, float] = model["sigmas"]  # type: ignore[assignment]
    r_t = pl.col("log_value").diff().over("target_class")
    mean_ret = pl.lit(a) + pl.lit(b) * r_t
    mean = pl.col("log_value") + mean_ret
    sigma = pl.col("target_class").replace_strict(sigmas, default=sigmas["POOLED"])
    return df.select(
        pl.col("date"),
        pl.col("target_class"),
        pl.lit(h).cast(pl.Int64).alias("h"),
        mean.alias("p_0.5"),
        (mean + ZSCORES[0.1] * sigma).alias("p_0.1"),
        (mean + ZSCORES[0.9] * sigma).alias("p_0.9"),
    )


def make_matrix(df: pl.DataFrame, h: int) -> tuple[list[str], object]:
    """Feature matrix (numpy) plus feature names for tree models at horizon h."""
    feats = [c for c in df.columns if c not in TREE_EXCLUDE]
    mat = (
        df.with_columns(
            pl.col("target_class").replace_strict(CLASS_CODES).alias("target_class")
        )
        .select(feats)
        .to_numpy()
    )
    return feats, mat


def evaluate(
    pred: pl.DataFrame, source: pl.DataFrame, split: str, model: str
) -> list[dict[str, object]]:
    """Per-horizon interval / pinball / coverage metrics; pooled + per class.

    The horizon mask is ``y_step_h{h}`` non-null *and* a prediction present.
    ``dir_hit`` is ``None`` for ``rw_emp``: its P50 is exactly the current
    level, so a directional hit rate is structurally meaningless.
    """
    h = int(pred["h"][0])
    truth = source.select(
        "date", "target_class",
        pl.col(f"y_step_h{h}").alias("y"), pl.col("log_value"),
    )
    j = truth.join(pred, on=["date", "target_class"], how="inner").drop_nulls("y")
    if j.is_empty():
        return []

    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str | None]] = [("POOLED", None)]
    scopes += [(c, c) for c in CLASSES]
    for scope, cls in scopes:
        sub = j if cls is None else j.filter(pl.col("target_class") == cls)
        if sub.is_empty():
            continue
        y = sub["y"]
        p10, p50, p90 = sub["p_0.1"], sub["p_0.5"], sub["p_0.9"]
        q_losses = [pinball(y, sub[f"p_{q}"], q) for q in QUANTILES]
        width = float((p90 - p10).mean())
        below = (p10 - y).clip(lower_bound=0.0)
        above = (y - p90).clip(lower_bound=0.0)
        winkler = float(width + 10.0 * (below + above).mean())
        row: dict[str, object] = {
            "split": split, "h": h, "model": model, "scope": scope, "n": sub.height,
        }
        for q, loss in zip(QUANTILES, q_losses):
            row[f"pinball_{q}"] = loss
        row["pinball_mean"] = float(sum(q_losses) / len(q_losses))
        row["below_p10"] = float((y < p10).mean())
        row["below_p90"] = float((y <= p90).mean())
        row["cover_80"] = float(((y >= p10) & (y <= p90)).mean())
        row["width"] = width
        row["winkler_80"] = winkler
        row["mae_p50"] = float((y - p50).abs().mean())
        if model == "rw_emp":
            row["dir_hit"] = None
        else:
            hits = (p50 > sub["log_value"]) == (y > sub["log_value"])
            row["dir_hit"] = float(hits.mean())
        row["x_p10_gt_p50"] = float((p10 > p50).mean())
        row["x_p10_gt_p90"] = float((p10 > p90).mean())
        rows.append(row)
    return rows


def _history_for(split_name: str, splits: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Data known before ``split_name`` for calibration.

    Valid is calibrated against train; test against train+valid. Train is
    calibrated against train (in-sample) rather than leaking the future.
    """
    if split_name in ("train", "valid"):
        return splits["train"]
    return pl.concat([splits["train"], splits["valid"]], how="diagonal")


def _context_frame(
    base_preds: dict[str, dict[int, pl.DataFrame]],
    splits: dict[str, pl.DataFrame],
    names: list[str],
    h: int,
) -> pl.DataFrame:
    """Base quantiles + realised outcome + level for the fuzzy input history."""
    frames = []
    for n in names:
        truth = splits[n].select(
            "date", "target_class", "log_value",
            pl.col(f"y_step_h{h}").alias("y"),
        )
        frames.append(base_preds[n][h].join(truth, on=["date", "target_class"], how="left"))
    return pl.concat(frames, how="diagonal")


def _selected_models() -> set[str] | None:
    """Optional model filter from ``BASELINES_MODELS`` (comma-separated)."""
    raw = os.environ.get("BASELINES_MODELS", "").strip()
    if not raw:
        return None
    return {name.strip() for name in raw.split(",") if name.strip()}


def _include_test() -> bool:
    return os.environ.get("BASELINES_INCLUDE_TEST", "").strip().lower() in ("1", "true", "yes")


def common_support_frame(
    pred_store: dict[tuple[str, int, str], pl.DataFrame],
    split_name: str,
    h: int,
    model_names: list[str],
) -> pl.DataFrame:
    """Row keys ``(date, target_class)`` present for every model at ``(split, h)``."""
    keys: pl.DataFrame | None = None
    for model_name in model_names:
        frame = pred_store.get((split_name, h, model_name))
        if frame is None:
            return pl.DataFrame()
        k = frame.select("date", "target_class").unique()
        keys = k if keys is None else keys.join(k, on=["date", "target_class"], how="inner")
    return keys if keys is not None else pl.DataFrame()


def common_support_rows(
    pred_store: dict[tuple[str, int, str], pl.DataFrame],
    splits: dict[str, pl.DataFrame],
    split_names: list[str],
    model_names: list[str],
) -> list[dict[str, object]]:
    """Re-score every model on the intersection of their available rows.

    The LSTM needs a warm-up window, so its frame is a strict subset of the tree
    and statistical frames. Restricting to the intersection makes the pooled and
    per-class numbers directly comparable across model families.
    """
    rows: list[dict[str, object]] = []
    for split_name in split_names:
        for h in HORIZONS:
            keys = common_support_frame(pred_store, split_name, h, model_names)
            if keys.is_empty():
                continue
            for model_name in model_names:
                frame = pred_store.get((split_name, h, model_name))
                if frame is None:
                    continue
                sub = frame.join(keys, on=["date", "target_class"], how="inner")
                rows.extend(evaluate(sub, splits[split_name], split_name, model_name))
    return rows


def main() -> None:
    """Run all baselines, evaluate on valid/test, write metrics csv."""
    from ml.models.lgbm import LGBMModel
    from ml.models.xgb import XGBModel
    from ml.predictions import ensure_lstm_predictions

    _setup_logging()
    selected = _selected_models()
    splits: dict[str, pl.DataFrame] = {name: load_split(name) for name in ("train", "valid")}
    if _include_test():
        with allow_test_set_access("final baseline metrics for the writeup"):
            splits["test"] = load_frozen_test()
        LOGGER.warning("BASELINES_INCLUDE_TEST=1: test split is being scored")
    eval_splits = ["valid"] + (["test"] if "test" in splits else [])

    all_rows: list[dict[str, object]] = []
    sign_rows: list[dict[str, object]] = []
    pred_store: dict[tuple[str, int, str], pl.DataFrame] = {}

    def record(model_name: str, split_name: str, pred: pl.DataFrame) -> None:
        h = int(pred["h"][0])
        all_rows.extend(evaluate(pred, splits[split_name], split_name, model_name))
        pred_store[(split_name, h, model_name)] = pred

    # 1. Statistical baselines, per horizon on that horizon's mask.
    for h in HORIZONS:
        candidates: dict[str, dict[str, pl.DataFrame]] = {}
        if selected is None or "rw" in selected:
            candidates["rw"] = {
                s: predict_rw(splits[s], h, fit_rw_sigmas(splits["train"], h))
                for s in eval_splits
            }
        if selected is None or "rw_emp" in selected:
            candidates["rw_emp"] = {
                s: predict_rw_emp(splits[s], h, fit_rw_emp(splits["train"], h))
                for s in eval_splits
            }
        if selected is None or "ar1" in selected:
            candidates["ar1"] = {
                s: predict_ar1(splits[s], h, fit_ar1(splits["train"], h))
                for s in eval_splits
            }
        for model, split_preds in candidates.items():
            for split_name, pred in split_preds.items():
                record(model, split_name, pred)
        if selected is None or "naive_sign" in selected:
            for split_name in eval_splits:
                sign_rows.extend(naive_sign_metrics(splits[split_name], split_name, h))
        LOGGER.info("h=%d: statistical baselines done", h)

    # 2. Tree models.
    for model_name, ModelClass in (("xgb", XGBModel), ("lgbm", LGBMModel)):
        if selected is not None and model_name not in selected:
            continue
        LOGGER.info("Training and evaluating %s...", model_name)
        model = ModelClass()
        model.fit(splits["train"], list(HORIZONS))
        preds_dict = model.predict({s: splits[s] for s in eval_splits})
        for split_name, h_preds in preds_dict.items():
            for h, pred_df in h_preds.items():
                record(model_name, split_name, pred_df)
        LOGGER.info("%s: done", model_name)

    # 3. LSTM ensemble + calibration layers.
    lstm_wanted = selected is None or any(
        m in (selected or set()) for m in ("lstm", "lstm_fuzzy", "lstm_fuzzy_old", "const_shift", "cqr", "aci")
    )
    if lstm_wanted:
        lstm_splits = {s: splits[s] for s in ["train", *eval_splits]}
        mean_preds, per_seed, meta = ensure_lstm_predictions(lstm_splits)
        if selected is None or "lstm" in selected:
            for split_name in eval_splits:
                for h in HORIZONS:
                    record("lstm", split_name, mean_preds[split_name][h])

        # 3a. Old fuzzy layer (ablation).
        if selected is None or "lstm_fuzzy_old" in selected:
            for split_name in eval_splits:
                history = _history_for(split_name, splits)
                for h in HORIZONS:
                    cal = fuzzy_calibrate_old(mean_preds[split_name][h], history, h)
                    record("lstm_fuzzy_old", split_name, cal)

        # 3b. Conformal baselines, calibrated on valid.
        if selected is None or any(m in (selected or set()) for m in ("const_shift", "cqr", "aci")):
            for h in HORIZONS:
                calib_frame = _context_frame(mean_preds, splits, ["valid"], h)
                const_fit = fit_const_shift(calib_frame)
                cqr_fit = fit_cqr(calib_frame)
                aci_state: dict[str, object] | None = None
                for split_name in eval_splits:
                    raw = mean_preds[split_name][h]
                    if selected is None or "const_shift" in selected:
                        record("const_shift", split_name, apply_const_shift(raw, const_fit))
                    if selected is None or "cqr" in selected:
                        record("cqr", split_name, apply_cqr(raw, cqr_fit))
                    if selected is None or "aci" in selected:
                        adj, aci_state = aci_adjust(raw, splits[split_name], h, state=aci_state)
                        record("aci", split_name, adj)

        # 3c. New Sugeno fuzzy layer, trained on valid, frozen for test.
        if selected is None or "lstm_fuzzy" in selected:
            for h in HORIZONS:
                context_names = ["train", "valid"] + (["test"] if "test" in splits else [])
                context = _context_frame(mean_preds, splits, context_names, h)
                train_frame = _context_frame(mean_preds, splits, ["valid"], h)
                params, stats = fit_fuzzy(context, train_frame, h)
                save_params(params, f"lstm_fuzzy_h{h}")
                LOGGER.info("h=%d fuzzy params:\n%s", h, describe_params(params))
                try:
                    plot_fuzzy(params, f"h{h}")
                except Exception as exc:  # noqa: BLE001 - plotting is best-effort
                    LOGGER.warning("fuzzy plot failed: %s", exc)
                LOGGER.info(
                    "h=%d fuzzy: n_train=%d median_matured=%.1f frac_matured=%.2f",
                    h, stats["n_train"], stats["n_matured_median"], stats["frac_rows_with_matured"],
                )
                for split_name in eval_splits:
                    hist_names = ["train", "valid"] + (["test"] if (split_name == "test" and "test" in splits) else [])
                    history = _context_frame(mean_preds, splits, hist_names, h)
                    cal = fuzzy_calibrate(mean_preds[split_name][h], history, h, params)
                    record("lstm_fuzzy", split_name, cal)

    metrics = pl.DataFrame(all_rows).sort(["split", "h", "model", "scope"])
    out_path = DATA / "baseline_metrics.csv"
    metrics.write_csv(out_path)
    LOGGER.info("wrote %s (%d rows)", out_path, metrics.height)

    # Common-support table: the same rows for every non-ablation model.
    common_models = [
        m for m in ("lstm", "lstm_fuzzy", "cqr", "aci", "ar1", "xgb", "rw_emp")
        if (eval_splits[0], HORIZONS[0], m) in pred_store
    ]
    if len(common_models) >= 2:
        common_rows = common_support_rows(pred_store, splits, eval_splits, common_models)
        if common_rows:
            common_path = DATA / "baseline_metrics_common.csv"
            pl.DataFrame(common_rows).sort(["split", "h", "model", "scope"]).write_csv(common_path)
            LOGGER.info("wrote %s (%d rows)", common_path, len(common_rows))

    pooled = metrics.filter(pl.col("scope") == "POOLED").select(
        "split", "h", "model", "n", "pinball_0.1", "pinball_0.5", "pinball_0.9",
        "pinball_mean", "cover_80", "width", "dir_hit",
    ).sort(["split", "model", "h"])
    LOGGER.info("pooled metrics:\n%s", pooled.to_pandas().to_string(index=False))

    # Fuzzy ablation arms (width-only, loc-only, both, no-vol); separate table.
    if lstm_wanted and os.environ.get("BASELINES_ABLATIONS", "").strip().lower() in ("1", "true", "yes"):
        abl_rows: list[dict[str, object]] = []
        arms = {
            "both": {"use_width": True, "use_loc": True, "with_vol": True},
            "widthonly": {"use_width": True, "use_loc": False, "with_vol": True},
            "loconly": {"use_width": False, "use_loc": True, "with_vol": True},
            "novol": {"use_width": True, "use_loc": True, "with_vol": False},
        }
        for h in HORIZONS:
            context_names = ["train", "valid"] + (["test"] if "test" in splits else [])
            context = _context_frame(mean_preds, splits, context_names, h)
            train_frame = _context_frame(mean_preds, splits, ["valid"], h)
            for arm, kw in arms.items():
                params, _ = fit_fuzzy(context, train_frame, h, **kw)
                for split_name in eval_splits:
                    hist_names = ["train", "valid"] + (["test"] if (split_name == "test" and "test" in splits) else [])
                    history = _context_frame(mean_preds, splits, hist_names, h)
                    cal = fuzzy_calibrate(mean_preds[split_name][h], history, h, params)
                    for r in evaluate(cal, splits[split_name], split_name, "lstm_fuzzy"):
                        abl_rows.append({**r, "arm": arm})
        abl_path = DATA / "fuzzy_ablations.csv"
        pl.DataFrame(abl_rows).write_csv(abl_path)
        LOGGER.info("wrote %s (%d rows)", abl_path, len(abl_rows))

    if sign_rows:
        sign_metrics = pl.DataFrame(sign_rows).sort(["split", "h", "scope"])
        sign_path = DATA / "naive_sign_metrics.csv"
        sign_metrics.write_csv(sign_path)
        LOGGER.info("wrote %s (%d rows)", sign_path, sign_metrics.height)

    # Diebold-Mariano + block bootstrap: lstm_fuzzy vs its competitors.
    dm_rows: list[dict[str, object]] = []
    for split_name in eval_splits:
        for h in HORIZONS:
            base_key = (split_name, h, "lstm_fuzzy")
            if base_key not in pred_store:
                continue
            for comp in ("lstm", "const_shift", "cqr", "aci"):
                comp_key = (split_name, h, comp)
                if comp_key not in pred_store:
                    continue
                for q in (0.1, 0.5, 0.9, None):
                    la, lb, _ = aligned_losses(pred_store[base_key], pred_store[comp_key], splits[split_name], h, q)
                    dm = dm_test(la, lb, h)
                    boot = block_bootstrap_ci(la - lb, block=max(h, 5))
                    dm_rows.append({
                        "split": split_name, "h": h, "competitor": comp,
                        "quantile": "mean" if q is None else q, **dm, **boot,
                    })
    if dm_rows:
        dm_df = pl.DataFrame(dm_rows)
        dm_path = DATA / "baseline_dm.csv"
        dm_df.write_csv(dm_path)
        LOGGER.info("wrote %s (%d rows)", dm_path, dm_df.height)
        write_markdown_summary(metrics, dm_df, DATA / "baseline_summary.md")
    else:
        write_markdown_summary(metrics, pl.DataFrame(), DATA / "baseline_summary.md")


if __name__ == "__main__":
    main()
