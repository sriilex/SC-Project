"""Export detailed Sugeno-layer behaviour for paper visualisations.

The exports are intentionally data-only: membership curves, rule firing,
per-forecast corrections, component decompositions, partial dependence and
two-input response surfaces. No test metrics or comparisons against other
calibrators are produced here.

Run from the repository root::

    uv run python -m ml.sugeno_interpretability

Writes CSV/JSON artefacts under ``src/data/sugeno_interpretability/``.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from ml.baselines import load_split
from ml.frozen_test import allow_test_set_access, load_frozen_test
from ml.fuzzy_calibration import compute_fuzzy_inputs, fis_forward, load_params

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"
PRED_DIR: Final[Path] = DATA / "preds"
OUT_DIR: Final[Path] = DATA / "sugeno_interpretability"
HORIZONS: Final[tuple[int, ...]] = (1, 7, 30, 90)
SYSTEMS: Final[tuple[str, ...]] = ("width", "loc")
CLASSES: Final[tuple[str, ...]] = ("Capesize", "Panamax", "Supramax")
QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)
MEMBERSHIP_GRID: Final[np.ndarray] = np.linspace(-4.0, 4.0, 321)
PDP_GRID: Final[np.ndarray] = np.linspace(-3.0, 3.0, 61)
SURFACE_GRID: Final[np.ndarray] = np.linspace(-3.0, 3.0, 31)


def _prediction_key() -> str | None:
    """Frozen v1 prediction key, when the reference run metadata exists."""
    path = DATA / "runs" / "v1" / "run_metadata.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = payload.get("lstm_prediction_key")
    return str(key) if key else None


def _cached_predictions(
    splits: dict[str, pl.DataFrame],
    key: str,
    horizons: tuple[int, ...],
) -> dict[str, dict[int, pl.DataFrame]] | None:
    """Load cached prediction frames for every split/horizon, or ``None``."""
    out: dict[str, dict[int, pl.DataFrame]] = {}
    for split_name in splits:
        out[split_name] = {}
        for h in horizons:
            path = PRED_DIR / f"lstm_{key}_{split_name}_h{h}.parquet"
            if not path.exists():
                return None
            out[split_name][h] = pl.read_parquet(path)
    return out


def _load_base_predictions(
    splits: dict[str, pl.DataFrame],
) -> tuple[dict[str, dict[int, pl.DataFrame]], str]:
    """Prefer the frozen cached predictions; fall back to a fresh fit."""
    env_key = os.environ.get("SUGENO_INTERP_PRED_KEY", "").strip()
    key = env_key or _prediction_key()
    if key:
        cached = _cached_predictions(splits, key, HORIZONS)
        if cached is not None:
            LOGGER.info("using cached predictions (key=%s)", key)
            return cached, key
        LOGGER.warning("cached prediction key %s is incomplete; refitting", key)
    from ml.predictions import ensure_lstm_predictions

    preds, _per_seed, meta = ensure_lstm_predictions(splits)
    LOGGER.info("fitted predictions (key=%s)", meta.get("key"))
    return preds, str(meta.get("key"))


def _context_frame(
    base_preds: dict[str, dict[int, pl.DataFrame]],
    splits: dict[str, pl.DataFrame],
    names: list[str],
    h: int,
) -> pl.DataFrame:
    """Base quantiles, realised outcome and level for the fuzzy input history."""
    frames: list[pl.DataFrame] = []
    for name in names:
        truth = splits[name].select(
            "date",
            "target_class",
            "log_value",
            pl.col(f"y_step_h{h}").alias("y"),
        )
        frames.append(
            base_preds[name][h].join(truth, on=["date", "target_class"], how="left")
        )
    return pl.concat(frames, how="diagonal")


def _standardized(spec: dict[str, object], frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw and standardised inputs for one FIS specification."""
    columns = list(spec["inputs"])  # type: ignore[arg-type]
    if not columns:
        return np.zeros((frame.height, 0)), np.zeros((frame.height, 0))
    raw = np.stack(
        [frame[column].to_numpy().astype(float) for column in columns],
        axis=1,
    )
    mean = np.asarray(spec["norm_mean"], dtype=float).reshape(-1)
    std = np.asarray(spec["norm_std"], dtype=float).reshape(-1)
    return raw, (raw - mean) / (std + 1e-9)


def _adjust(base: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Apply a log-width multiplier and location shift to ``[p10, p50, p90]``."""
    p10, p50, p90 = base[:, 0], base[:, 1], base[:, 2]
    p50_new = p50 + s
    mult = np.exp(m)
    p10_new = p50_new - (p50 - p10) * mult
    p90_new = p50_new + (p90 - p50) * mult
    out = np.stack([p10_new, p50_new, p90_new], axis=1)
    out[:, 0] = np.minimum(out[:, 0], out[:, 1])
    out[:, 2] = np.maximum(out[:, 2], out[:, 1])
    return out


def _pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Mean pinball loss for one quantile level."""
    diff = y - pred
    return float(np.maximum(q * diff, (q - 1.0) * diff).mean())


def _pinball_mean(y: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Row-wise mean pinball loss across the three modelled quantiles."""
    losses = [
        np.maximum(q * (y - quantiles[:, i]), (q - 1.0) * (y - quantiles[:, i]))
        for i, q in enumerate(QUANTILES)
    ]
    return np.mean(losses, axis=0)


def _rule_labels(inputs: list[str], n_mfs: int) -> list[str]:
    """Human-readable antecedent labels in the same order as ``fis_forward``."""
    labels: list[str] = []
    for combo in itertools.product(range(n_mfs), repeat=len(inputs)):
        labels.append("; ".join(f"{name}=MF{idx}" for name, idx in zip(inputs, combo)))
    return labels


def _trace_or_zero(
    spec: dict[str, object], x: np.ndarray, n: int
) -> dict[str, np.ndarray]:
    """Evaluate a FIS, returning zeros when it has no inputs."""
    if not list(spec["inputs"]):  # type: ignore[arg-type]
        return {
            "membership": np.zeros((n, 0, 0)),
            "rule_weight": np.zeros((n, 0)),
            "rule_weight_normalized": np.zeros((n, 0)),
            "rule_contribution": np.zeros((n, 0)),
            "raw": np.zeros(n),
            "output": np.zeros(n),
        }
    return fis_forward(spec, x)


def _membership_rows(h: int, params: dict[str, object]) -> list[dict[str, object]]:
    """Dense Gaussian membership curves for every input and MF."""
    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        spec = params[system]  # type: ignore[index]
        centers = np.asarray(spec["centers"], dtype=float)
        widths = np.asarray(spec["widths"], dtype=float)
        for i, name in enumerate(spec["inputs"]):  # type: ignore[union-attr]
            for k in range(centers.shape[1]):
                mu = np.exp(
                    -0.5
                    * ((MEMBERSHIP_GRID - centers[i, k]) / (widths[i, k] + 1e-9)) ** 2
                )
                rows.extend(
                    {
                        "h": h,
                        "system": system,
                        "input": name,
                        "mf": k,
                        "center": float(centers[i, k]),
                        "width": float(widths[i, k]),
                        "x": float(x),
                        "membership": float(value),
                    }
                    for x, value in zip(MEMBERSHIP_GRID, mu)
                )
    return rows


def _pdp_rows(
    h: int,
    system: str,
    spec: dict[str, object],
    standardized: np.ndarray,
) -> list[dict[str, object]]:
    """One-input-at-a-time partial dependence over observed standardised rows."""
    inputs = list(spec["inputs"])  # type: ignore[arg-type]
    if not inputs or standardized.size == 0:
        return []
    n = standardized.shape[0]
    step = max(1, n // 4000)
    sample = standardized[::step]
    rows: list[dict[str, object]] = []
    for idx, input_name in enumerate(inputs):
        for value in PDP_GRID:
            modified = sample.copy()
            modified[:, idx] = value
            output = fis_forward(spec, modified)["output"]
            rows.append(
                {
                    "h": h,
                    "system": system,
                    "input": input_name,
                    "grid_value": float(value),
                    "output_mean": float(np.mean(output)),
                    "output_q10": float(np.quantile(output, 0.1)),
                    "output_q90": float(np.quantile(output, 0.9)),
                    "n_rows": int(sample.shape[0]),
                }
            )
    return rows


def _surface_rows(h: int, system: str, spec: dict[str, object]) -> list[dict[str, object]]:
    """Two-input response surface on the standardised input scale."""
    inputs = list(spec["inputs"])  # type: ignore[arg-type]
    if len(inputs) != 2:
        return []
    xx, yy = np.meshgrid(SURFACE_GRID, SURFACE_GRID)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    output = fis_forward(spec, points)["output"].reshape(xx.shape)
    return [
        {
            "h": h,
            "system": system,
            "input_x": inputs[0],
            "input_y": inputs[1],
            "x": float(xx[i, j]),
            "y": float(yy[i, j]),
            "output": float(output[i, j]),
        }
        for i in range(xx.shape[0])
        for j in range(xx.shape[1])
    ]


def _effect_rows(
    split_name: str,
    h: int,
    frame: pl.DataFrame,
    params: dict[str, object],
    standardized: dict[str, np.ndarray],
    traces: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    """Per-forecast inputs, corrections and width/location decompositions."""
    base = np.stack(
        [
            frame["p_0.1"].to_numpy().astype(float),
            frame["p_0.5"].to_numpy().astype(float),
            frame["p_0.9"].to_numpy().astype(float),
        ],
        axis=1,
    )
    y = frame["y"].to_numpy().astype(float)
    m = traces["width"]["output"]
    s = traces["loc"]["output"]
    width_only = _adjust(base, m, np.zeros_like(s))
    loc_only = _adjust(base, np.zeros_like(m), s)
    both = _adjust(base, m, s)
    base_width = base[:, 2] - base[:, 0]
    adjusted_width = both[:, 2] - both[:, 0]
    s_scale = float(params["stats"]["s_scale"])  # type: ignore[index]
    rows: list[dict[str, object]] = []
    for i in range(frame.height):
        row: dict[str, object] = {
            "split": split_name,
            "h": h,
            "date": frame["date"][i],
            "target_class": frame["target_class"][i],
            "y": float(y[i]),
            "log_value": float(frame["log_value"][i]),
            "n_matured": int(frame["n_matured"][i]),
            "vol_z": float(frame["vol_z"][i]),
            "cov_gap": float(frame["cov_gap"][i]),
            "bias_z": float(frame["bias_z"][i]),
            "m": float(m[i]),
            "width_multiplier": float(np.exp(m[i])),
            "s": float(s[i]),
            "s_relative": float(s[i] / s_scale),
            "p10_base": float(base[i, 0]),
            "p50_base": float(base[i, 1]),
            "p90_base": float(base[i, 2]),
            "width_base": float(base_width[i]),
            "p10_adjusted": float(both[i, 0]),
            "p50_adjusted": float(both[i, 1]),
            "p90_adjusted": float(both[i, 2]),
            "width_adjusted": float(adjusted_width[i]),
            "delta_p10": float(both[i, 0] - base[i, 0]),
            "delta_p50": float(both[i, 1] - base[i, 1]),
            "delta_p90": float(both[i, 2] - base[i, 2]),
            "delta_width": float(adjusted_width[i] - base_width[i]),
            "width_only_delta_p10": float(width_only[i, 0] - base[i, 0]),
            "width_only_delta_p90": float(width_only[i, 2] - base[i, 2]),
            "loc_only_delta_p10": float(loc_only[i, 0] - base[i, 0]),
            "loc_only_delta_p50": float(loc_only[i, 1] - base[i, 1]),
            "loc_only_delta_p90": float(loc_only[i, 2] - base[i, 2]),
            "cover_base": float(base[i, 0] <= y[i] <= base[i, 2]),
            "cover_adjusted": float(both[i, 0] <= y[i] <= both[i, 2]),
            "pinball_base": float(_pinball_mean(np.array([y[i]]), base[i : i + 1])[0]),
            "pinball_adjusted": float(_pinball_mean(np.array([y[i]]), both[i : i + 1])[0]),
            "pinball_width_only": float(_pinball_mean(np.array([y[i]]), width_only[i : i + 1])[0]),
            "pinball_loc_only": float(_pinball_mean(np.array([y[i]]), loc_only[i : i + 1])[0]),
        }
        for system in SYSTEMS:
            for idx, name in enumerate(params[system]["inputs"]):  # type: ignore[index]
                row[f"{system}_{name}_z"] = float(standardized[system][i, idx])
        rows.append(row)
    return rows


def _rule_rows(
    split_name: str,
    h: int,
    frame: pl.DataFrame,
    params: dict[str, object],
    traces: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    """Per-forecast normalised firing and output contribution of every rule."""
    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        spec = params[system]  # type: ignore[index]
        inputs = list(spec["inputs"])  # type: ignore[arg-type]
        labels = _rule_labels(inputs, int(spec["n_mfs"]))
        normalized = traces[system]["rule_weight_normalized"]
        contribution = traces[system]["rule_contribution"]
        for rule, label in enumerate(labels):
            for i in range(frame.height):
                rows.append(
                    {
                        "split": split_name,
                        "h": h,
                        "date": frame["date"][i],
                        "target_class": frame["target_class"][i],
                        "system": system,
                        "rule": rule,
                        "antecedents": label,
                        "activation": float(normalized[i, rule]),
                        "contribution": float(contribution[i, rule]),
                    }
                )
    return rows


def _summary(effects: pl.DataFrame) -> pl.DataFrame:
    """Pooled and per-class summaries of the layer's behaviour."""
    aggregations = [
        pl.len().alias("n"),
        pl.col("m").mean().alias("m_mean"),
        pl.col("m").std().alias("m_sd"),
        pl.col("width_multiplier").mean().alias("width_multiplier_mean"),
        pl.col("width_multiplier").min().alias("width_multiplier_min"),
        pl.col("width_multiplier").max().alias("width_multiplier_max"),
        pl.col("s").mean().alias("s_mean"),
        pl.col("s").std().alias("s_sd"),
        pl.col("s_relative").mean().alias("s_relative_mean"),
        pl.col("delta_p50").abs().mean().alias("abs_location_effect_mean"),
        pl.col("delta_width").abs().mean().alias("abs_width_effect_mean"),
        pl.col("width_base").mean().alias("width_base_mean"),
        pl.col("width_adjusted").mean().alias("width_adjusted_mean"),
        pl.col("delta_width").mean().alias("delta_width_mean"),
        (pl.col("width_multiplier") > 1.0).mean().alias("frac_widened"),
        (pl.col("width_multiplier") < 1.0).mean().alias("frac_narrowed"),
        (pl.col("s") > 0.0).mean().alias("frac_shift_up"),
        pl.col("cover_base").mean().alias("cover_base_mean"),
        pl.col("cover_adjusted").mean().alias("cover_adjusted_mean"),
        pl.col("pinball_base").mean().alias("pinball_base_mean"),
        pl.col("pinball_adjusted").mean().alias("pinball_adjusted_mean"),
        pl.col("pinball_width_only").mean().alias("pinball_width_only_mean"),
        pl.col("pinball_loc_only").mean().alias("pinball_loc_only_mean"),
        pl.col("n_matured").mean().alias("n_matured_mean"),
    ]
    pooled = (
        effects.group_by(["split", "h"])
        .agg(aggregations)
        .with_columns(pl.lit("POOLED").alias("scope"))
    )
    by_class = (
        effects.group_by(["split", "h", "target_class"])
        .agg(aggregations)
        .rename({"target_class": "scope"})
    )
    columns = ["split", "h", "scope", *[expr.meta.output_name() for expr in aggregations]]
    return pl.concat([pooled, by_class], how="diagonal").select(columns).sort(
        ["split", "h", "scope"]
    )


def _rule_summary(rules: pl.DataFrame, params_by_h: dict[int, dict[str, object]]) -> pl.DataFrame:
    """Aggregate rule firing and join the learned consequent table."""
    aggregate = (
        rules.group_by(["h", "system", "split", "rule", "antecedents"])
        .agg(
            pl.len().alias("n"),
            pl.col("activation").mean().alias("activation_mean"),
            pl.col("activation").max().alias("activation_max"),
            (pl.col("activation") > 0.05).mean().alias("activation_gt_005_frac"),
            pl.col("contribution").mean().alias("contribution_mean"),
        )
        .sort(["h", "system", "split", "rule"])
    )
    meta_rows: list[dict[str, object]] = []
    for h, params in params_by_h.items():
        for system in SYSTEMS:
            spec = params[system]  # type: ignore[index]
            inputs = list(spec["inputs"])  # type: ignore[arg-type]
            labels = _rule_labels(inputs, int(spec["n_mfs"]))
            consequents = np.asarray(spec["consequents"], dtype=float).reshape(-1)
            linear = spec.get("linear_coefs")
            for rule, label in enumerate(labels):
                row: dict[str, object] = {
                    "h": h,
                    "system": system,
                    "rule": rule,
                    "antecedents": label,
                    "consequent": float(consequents[rule]),
                    "output_scale": float(spec["output_scale"]),
                }
                if linear is not None:
                    coefs = np.asarray(linear, dtype=float).reshape(len(labels), -1)
                    for idx, name in enumerate(inputs):
                        row[f"linear_{name}"] = float(coefs[rule, idx])
                meta_rows.append(row)
    meta = pl.DataFrame(meta_rows)
    return aggregate.join(meta, on=["h", "system", "rule", "antecedents"], how="left")


def main() -> None:
    """Generate the interpretability tables from frozen params and predictions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits: dict[str, pl.DataFrame] = {
        "train": load_split("train"),
        "valid": load_split("valid"),
    }
    with allow_test_set_access("post-hoc Sugeno interpretability export"):
        splits["test"] = load_frozen_test()
    base_preds, prediction_key = _load_base_predictions(splits)
    valid_start = splits["valid"]["date"].min()

    effect_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    pdp_rows: list[dict[str, object]] = []
    surface_rows: list[dict[str, object]] = []
    standardized_by_h: dict[tuple[int, str], list[np.ndarray]] = {}
    params_by_h: dict[int, dict[str, object]] = {}

    for h in HORIZONS:
        params = load_params(f"lstm_fuzzy_h{h}")
        params_by_h[h] = params
        context = _context_frame(
            base_preds,
            splits,
            ["train", "valid", "test"],
            h,
        )
        inputs = compute_fuzzy_inputs(context, h, matured_from=valid_start)
        membership_rows.extend(_membership_rows(h, params))
        for system in SYSTEMS:
            spec = params[system]  # type: ignore[index]
            surface_rows.extend(_surface_rows(h, system, spec))
            standardized_by_h[(h, system)] = []

        for split_name in ("valid", "test"):
            truth = splits[split_name].select(
                "date",
                "target_class",
                "log_value",
                pl.col(f"y_step_h{h}").alias("y"),
            )
            frame = (
                base_preds[split_name][h]
                .join(
                    inputs.select("date", "target_class", "vol_z", "cov_gap", "bias_z", "n_matured"),
                    on=["date", "target_class"],
                    how="left",
                )
                .join(truth, on=["date", "target_class"], how="left")
                .drop_nulls(["y", "p_0.1", "p_0.5", "p_0.9", "log_value"])
            )
            standardized: dict[str, np.ndarray] = {}
            traces: dict[str, dict[str, np.ndarray]] = {}
            for system in SYSTEMS:
                spec = params[system]  # type: ignore[index]
                _raw, z = _standardized(spec, frame)
                standardized[system] = z
                traces[system] = _trace_or_zero(spec, z, frame.height)
                if z.size:
                    standardized_by_h[(h, system)].append(z)
            effect_rows.extend(
                _effect_rows(split_name, h, frame, params, standardized, traces)
            )
            rule_rows.extend(_rule_rows(split_name, h, frame, params, traces))
            LOGGER.info("h=%d split=%s rows=%d", h, split_name, frame.height)

    for (h, system), chunks in standardized_by_h.items():
        if not chunks:
            continue
        spec = params_by_h[h][system]  # type: ignore[index]
        pdp_rows.extend(_pdp_rows(h, system, spec, np.vstack(chunks)))

    effects = pl.DataFrame(effect_rows).sort(["split", "h", "date", "target_class"])
    rules = pl.DataFrame(rule_rows).sort(["split", "h", "date", "target_class", "system", "rule"])
    memberships = pl.DataFrame(membership_rows)
    pdp = pl.DataFrame(pdp_rows)
    surfaces = pl.DataFrame(surface_rows)
    summary = _summary(effects)
    rule_summary = _rule_summary(rules, params_by_h)

    effects.write_csv(OUT_DIR / "forecast_effects.csv")
    rules.write_csv(OUT_DIR / "rule_activations.csv")
    memberships.write_csv(OUT_DIR / "membership_curves.csv")
    pdp.write_csv(OUT_DIR / "partial_dependence.csv")
    surfaces.write_csv(OUT_DIR / "response_surfaces.csv")
    summary.write_csv(OUT_DIR / "effect_summary.csv")
    rule_summary.write_csv(OUT_DIR / "rule_summary.csv")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prediction_key": prediction_key,
        "horizons": list(HORIZONS),
        "splits": ["valid", "test"],
        "parameters": "src/data/fuzzy_params/lstm_fuzzy_h*.json",
        "equations": {
            "membership": "mu_ik(x) = exp(-0.5 * ((x - c_ik) / w_ik)^2)",
            "rule_weight": "w_r(x) = prod_i mu_i,k_i(x_i)",
            "defuzzified": "raw(x) = sum_r w_r(x) f_r(x) / sum_r w_r(x)",
            "bounded_output": "output(x) = scale * tanh(raw(x))",
            "width_application": "p50' = p50 + s; p10' = p50' - (p50 - p10) * exp(m); p90' = p50' + (p90 - p50) * exp(m)",
        },
        "files": {
            "membership_curves.csv": "Gaussian membership values on a dense x grid, with centres and widths.",
            "rule_summary.csv": "Learned consequents plus mean/max normalised rule firing and contribution by split.",
            "rule_activations.csv": "Per-forecast normalised firing and contribution for every rule.",
            "forecast_effects.csv": "Per-forecast inputs, m, exp(m), s, base/adjusted quantiles, component deltas, coverage and pinball losses.",
            "effect_summary.csv": "Pooled and per-class summaries of layer behaviour.",
            "partial_dependence.csv": "One-input-at-a-time partial dependence over observed standardised inputs.",
            "response_surfaces.csv": "Two-input FIS response on a standardised grid (h=1 and h=7, width and location).",
        },
        "row_counts": {
            "forecast_effects": effects.height,
            "rule_activations": rules.height,
            "membership_curves": memberships.height,
            "partial_dependence": pdp.height,
            "response_surfaces": surfaces.height,
            "effect_summary": summary.height,
            "rule_summary": rule_summary.height,
        },
        "column_notes": {
            "forecast_effects": {
                "m": "Width FIS output: log multiplier on both half-widths.",
                "width_multiplier": "exp(m); values below one narrow the interval, above one widen it.",
                "s": "Location FIS output added to P50; P10 and P90 move with it.",
                "s_relative": "s divided by the horizon's bounded location scale.",
                "width_only_delta_*": "Quantile change from the width system alone.",
                "loc_only_delta_*": "Quantile change from the location system alone.",
                "delta_*": "Combined quantile/width change.",
                "*_z": "Standardised FIS inputs actually seen by the membership functions.",
            },
            "rule_summary": {
                "antecedents": "Input membership indices in the learned rule order.",
                "consequent": "Constant rule output before weighting and tanh.",
                "activation_mean": "Mean normalised rule firing; all rules sum to one per forecast.",
                "contribution_mean": "Mean weighted rule contribution to the raw FIS output.",
            },
        },
        "notes": [
            "The width FIS uses cov_gap and vol_z at h=1 and h=7, and vol_z only at h=30 and h=90.",
            "The location FIS uses bias_z and vol_z at every horizon.",
            "Within-interval movement comes from s; exp(m) scales the lower and upper half-widths around the shifted median.",
            "Rule activation is normalised by the sum of all rule weights for that forecast.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "wrote %d effect rows, %d rule rows, %d membership rows, %d PDP rows, %d surface rows to %s",
        effects.height,
        rules.height,
        memberships.height,
        pdp.height,
        surfaces.height,
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
