"""Pooled quantile baselines over the sample splits.

Benchmarks and calibration layers:
  lstm             five-seed monotone quantile LSTM ensemble
  lstm_fuzzy       new Sugeno layer (Task F)
  lstm_fuzzy_old   original constant-shift fuzzy layer (ablation)
  const_shift      raw LSTM + median residual shift (calibration split)
  cqr              split conformalized quantile regression
  aci              adaptive conformal inference / online quantile tracking

A directional ``naive_sign`` diagnostic (persist sign(r_t)) is also reported;
it is not a quantile forecaster.

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

import numpy as np
import polars as pl
import scoringrules as sr

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
    date_cluster_bootstrap_ci,
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
#: Miscoverage level of the reported central interval (P10-P90, i.e. 80%).
INTERVAL_ALPHA: Final[float] = 0.2
CLASSES: Final[tuple[str, ...]] = ("Capesize", "Panamax", "Supramax", "Handysize")


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
    """Mean pinball (quantile) loss for quantile level q (``scoringrules.quantile_score``)."""
    loss = sr.quantile_score(np.asarray(y), np.asarray(p), q)
    return float(np.asarray(loss).mean())


def _clean_sigma(s: object, fallback: float) -> float:
    """Return a usable sigma or the fallback; handles None and NaN."""
    if s is None:
        return fallback
    s_f = float(s)
    return s_f if s_f == s_f and s_f > 0 else fallback


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


def evaluate(
    pred: pl.DataFrame, source: pl.DataFrame, split: str, model: str
) -> list[dict[str, object]]:
    """Per-horizon interval / pinball / coverage metrics; pooled + per class.

    The horizon mask is ``y_step_h{h}`` non-null *and* a prediction present.
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
        y = sub["y"].to_numpy()
        p10, p50, p90 = (sub[c].to_numpy() for c in ("p_0.1", "p_0.5", "p_0.9"))
        level = sub["log_value"].to_numpy()
        q_losses = [pinball(y, sub[f"p_{q}"], q) for q in QUANTILES]
        width = float((p90 - p10).mean())
        # Winkler / interval score of the central interval (scoringrules).
        winkler = float(np.asarray(sr.interval_score(y, p10, p90, INTERVAL_ALPHA)).mean())
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
        row["mae_p50"] = float(np.abs(y - p50).mean())
        hits = (p50 > level) == (y > level)
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

    The LSTM needs a warm-up window, so its frame is a strict subset of the
    calibration-layer frames. Restricting to the intersection makes the pooled
    and per-class numbers directly comparable across model families.
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
    from ml.predictions import ensure_lstm_predictions

    _setup_logging()
    selected = _selected_models()
    splits: dict[str, pl.DataFrame] = {name: load_split(name) for name in ("train", "valid")}
    if _include_test():
        with allow_test_set_access("final baseline metrics for the writeup"):
            splits["test"] = load_frozen_test()
        LOGGER.warning("BASELINES_INCLUDE_TEST=1: test split is being scored")
    eval_splits = ["valid"] + (["test"] if "test" in splits else [])
    # Burn-in date for the fuzzy matured-residual windows: residuals from the
    # in-sample train base are excluded when calibrating/scoring, so the first
    # out-of-sample rows simply fall back to the identity window.
    valid_start = splits["valid"]["date"].min() if "valid" in splits else None

    all_rows: list[dict[str, object]] = []
    sign_rows: list[dict[str, object]] = []
    aci_rows: list[dict[str, object]] = []
    fuzzy_spread_rows: list[dict[str, object]] = []
    pred_store: dict[tuple[str, int, str], pl.DataFrame] = {}

    def record(model_name: str, split_name: str, pred: pl.DataFrame) -> None:
        h = int(pred["h"][0])
        all_rows.extend(evaluate(pred, splits[split_name], split_name, model_name))
        pred_store[(split_name, h, model_name)] = pred

    # 1. Naive-sign diagnostic, per horizon on that horizon's mask.
    for h in HORIZONS:
        if selected is None or "naive_sign" in selected:
            for split_name in eval_splits:
                sign_rows.extend(naive_sign_metrics(splits[split_name], split_name, h))
        LOGGER.info("h=%d: naive-sign diagnostic done", h)

    # 2. LSTM ensemble + calibration layers.
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

        # 2a. Old fuzzy layer (ablation).
        if selected is None or "lstm_fuzzy_old" in selected:
            for split_name in eval_splits:
                history = _history_for(split_name, splits)
                for h in HORIZONS:
                    cal = fuzzy_calibrate_old(mean_preds[split_name][h], history, h)
                    record("lstm_fuzzy_old", split_name, cal)

        # 2b. Conformal baselines, calibrated on valid.
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
                        for cls, st in (aci_state or {}).items():
                            if not isinstance(st, dict):
                                continue
                            a = float(st["alpha"])
                            sc = np.asarray(st["scores"], dtype=float)
                            w = float(np.quantile(sc, 1.0 - a)) if sc.size else float("nan")
                            aci_rows.append({
                                "split": split_name, "h": h, "target_class": cls,
                                "alpha_final": a, "w_next": w,
                                "score_min": float(sc.min()) if sc.size else float("nan"),
                                "score_max": float(sc.max()) if sc.size else float("nan"),
                                "n_scores": int(sc.size),
                            })

        # 2c. New Sugeno fuzzy layer, trained on valid, frozen for test.
        if selected is None or "lstm_fuzzy" in selected:
            for h in HORIZONS:
                context_names = ["train", "valid"] + (["test"] if "test" in splits else [])
                context = _context_frame(mean_preds, splits, context_names, h)
                train_frame = _context_frame(mean_preds, splits, ["valid"], h)
                params, stats = fit_fuzzy(context, train_frame, h, matured_from=valid_start)
                save_params(params, f"lstm_fuzzy_h{h}")
                LOGGER.info("h=%d fuzzy params:\n%s", h, describe_params(params))
                try:
                    plot_fuzzy(params, f"h{h}")
                except Exception as exc:  # noqa: BLE001 - plotting is best-effort
                    LOGGER.warning("fuzzy plot failed: %s", exc)
                LOGGER.info(
                    "h=%d fuzzy: n_train=%d window=%d median_matured=%.1f "
                    "frac_matured=%.2f s_scale=%.3f width_inputs=%s loc_inputs=%s",
                    h, stats["n_train"], stats["n_matured_window"], stats["n_matured_median"],
                    stats["frac_rows_with_matured"], stats["s_scale"],
                    stats["width_inputs"], stats["loc_inputs"],
                )
                for split_name in eval_splits:
                    hist_names = ["train", "valid"] + (["test"] if (split_name == "test" and "test" in splits) else [])
                    history = _context_frame(mean_preds, splits, hist_names, h)
                    cal = fuzzy_calibrate(mean_preds[split_name][h], history, h, params, matured_from=valid_start)
                    record("lstm_fuzzy", split_name, cal)

                # Calibrator seed spread: the reported model is seed 0; seeds
                # 1-4 show how sensitive the layer is to its initialisation.
                seed_models: dict[int, object] = {0: params}
                for seed in (1, 2, 3, 4):
                    seed_params, _ = fit_fuzzy(
                        context, train_frame, h, seed=seed, matured_from=valid_start
                    )
                    seed_models[seed] = seed_params
                for seed, seed_params in seed_models.items():
                    for split_name in eval_splits:
                        hist_names = ["train", "valid"] + (["test"] if (split_name == "test" and "test" in splits) else [])
                        history = _context_frame(mean_preds, splits, hist_names, h)
                        cal = fuzzy_calibrate(
                            mean_preds[split_name][h], history, h, seed_params,
                            matured_from=valid_start,
                        )
                        for r in evaluate(cal, splits[split_name], split_name, "lstm_fuzzy"):
                            if r["scope"] != "POOLED":
                                continue
                            fuzzy_spread_rows.append({
                                "split": split_name, "h": h, "seed": seed, "n": r["n"],
                                "pinball_mean": r["pinball_mean"],
                                "cover_80": r["cover_80"], "width": r["width"],
                            })

    metrics = pl.DataFrame(all_rows).sort(["split", "h", "model", "scope"])
    out_path = DATA / "baseline_metrics.csv"
    metrics.write_csv(out_path)
    LOGGER.info("wrote %s (%d rows)", out_path, metrics.height)

    # Common-support table: the same rows for every non-ablation model.
    common_models = [
        m for m in ("lstm", "lstm_fuzzy", "cqr", "aci")
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
                params, _ = fit_fuzzy(context, train_frame, h, matured_from=valid_start, **kw)
                for split_name in eval_splits:
                    hist_names = ["train", "valid"] + (["test"] if (split_name == "test" and "test" in splits) else [])
                    history = _context_frame(mean_preds, splits, hist_names, h)
                    cal = fuzzy_calibrate(mean_preds[split_name][h], history, h, params, matured_from=valid_start)
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

    if aci_rows:
        aci_path = DATA / "aci_diagnostics.csv"
        aci_df = pl.DataFrame(aci_rows).sort(["split", "h", "target_class"])
        aci_df.write_csv(aci_path)
        LOGGER.info("wrote %s (%d rows)", aci_path, aci_df.height)
        LOGGER.info("ACI final state:\n%s", aci_df)

    if fuzzy_spread_rows:
        spread_path = DATA / "fuzzy_seed_spread.csv"
        spread_df = pl.DataFrame(fuzzy_spread_rows).sort(["split", "h", "seed"])
        spread_df.write_csv(spread_path)
        agg = spread_df.group_by(["split", "h"]).agg(
            pl.col("pinball_mean").min().alias("pinball_min"),
            pl.col("pinball_mean").max().alias("pinball_max"),
            pl.col("pinball_mean").std().alias("pinball_sd"),
            pl.col("cover_80").min().alias("cover_min"),
            pl.col("cover_80").max().alias("cover_max"),
        ).sort(["split", "h"])
        LOGGER.info("wrote %s (%d rows)", spread_path, spread_df.height)
        LOGGER.info("fuzzy calibrator seed spread:\n%s", agg)

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
                    la, lb, j = aligned_losses(pred_store[base_key], pred_store[comp_key], splits[split_name], h, q)
                    dm = dm_test(la, lb, h)
                    boot = date_cluster_bootstrap_ci(la - lb, j["date"].to_numpy())
                    block = block_bootstrap_ci(la - lb, block=max(h, 5))
                    dm_rows.append({
                        "split": split_name, "h": h, "competitor": comp,
                        "quantile": "mean" if q is None else q, **dm, **boot,
                        "block_lo": block["boot_lo"], "block_hi": block["boot_hi"],
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
