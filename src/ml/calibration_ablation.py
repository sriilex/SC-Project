"""Validation-only calibration ablation for the Sugeno layer.

Every arm is fitted on the validation split with the frozen LSTM quantiles and
scored on validation; the test split is **never** read here unless
``CALIB_INCLUDE_TEST=1`` is set, and then only for the single variant selected on
validation. The raw LSTM is scored as ``base``; two ordinary calibrators
(constant width+location, linear state-dependent) provide the "simpler things"
comparison.

Run::

    uv run python -m ml.calibration_ablation

Writes ``src/data/runs/<run>/`` with the validation table, per-arm parameters,
the selected variant, and (when enabled) the one-shot test table.

Arms
----
``base``            frozen LSTM quantiles, no calibration.
``const_wl``        constant log-width multiplier + constant location shift.
``linear_state``    linear state-dependent width/location on the FIS inputs.
``sugeno0``         original zero-order Sugeno (3x3 rules, constants).
``sugeno1``         first-order Sugeno (linear consequents).
``sugeno_reg``      first-order + weight decay + early stopping.
``sugeno_split``    zero-order + separate lower/upper width multipliers.
``sugeno_lin_split``first-order + separate lower/upper width multipliers.
``sugeno_cls``      zero-order + shrunk per-class location/width offsets.
``sugeno_full``     first-order + split width + class offsets + regularisation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from ml.attribution import apply_attribution, fit_attribution
from ml.baselines import HORIZONS, evaluate, load_split
from ml.evaluation import date_block_bootstrap_ci
from ml.fuzzy_calibration import calibrate, fit_fuzzy

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"
PRED_DIR: Final[Path] = DATA / "preds"
ARMS: Final[dict[str, dict[str, object]]] = {
    "base": {},
    "const_wl": {"kind": "constant_width_location"},
    "linear_state": {"kind": "linear"},
    "sugeno0": {},
    "sugeno1": {"linear": True},
    "sugeno_reg": {
        "linear": True,
        "weight_decay": 1e-3,
        "early_stopping_patience": 50,
    },
    "sugeno_split": {"split_width": True},
    "sugeno_lin_split": {"linear": True, "split_width": True},
    "sugeno_cls": {"class_shrinkage": 0.5},
    "sugeno_full": {
        "linear": True,
        "split_width": True,
        "class_shrinkage": 0.5,
        "weight_decay": 1e-3,
        "early_stopping_patience": 50,
    },
}
SUGENO_ARMS: Final[tuple[str, ...]] = tuple(a for a in ARMS if a.startswith("sugeno"))


# ---------------------------------------------------------------------------
# Cached LSTM predictions (no test read)
# ---------------------------------------------------------------------------

def _prediction_key() -> str:
    """Key of the frozen LSTM run quoted in the paper metadata."""
    meta = json.loads((DATA / "runs" / "v1" / "run_metadata.json").read_text())
    return str(meta["lstm_prediction_key"])


def _load_preds(key: str, split: str, h: int) -> pl.DataFrame:
    path = PRED_DIR / f"lstm_{key}_{split}_h{h}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing cached LSTM predictions {path}. Run the baseline suite once "
            "to populate src/data/preds/, or set LSTM_PRED_KEY."
        )
    return pl.read_parquet(path)


def _context_frame(
    preds: dict[str, dict[int, pl.DataFrame]],
    splits: dict[str, pl.DataFrame],
    names: list[str],
    h: int,
) -> pl.DataFrame:
    frames = []
    for n in names:
        truth = splits[n].select(
            "date", "target_class", "log_value", pl.col(f"y_step_h{h}").alias("y")
        )
        frames.append(preds[n][h].join(truth, on=["date", "target_class"], how="left"))
    return pl.concat(frames, how="diagonal")


# ---------------------------------------------------------------------------
# Fitting / scoring
# ---------------------------------------------------------------------------

def _fit_arm(
    arm: str,
    context: pl.DataFrame,
    fit_rows: pl.DataFrame,
    h: int,
    matured_from,
) -> dict[str, object]:
    kw = ARMS[arm]
    if arm in ("base",):
        return {}
    if "kind" in kw:
        return fit_attribution(context, fit_rows, h, str(kw["kind"]), matured_from=matured_from)
    params, _stats = fit_fuzzy(context, fit_rows, h, matured_from=matured_from, **kw)  # type: ignore[arg-type]
    return params


def _apply_arm(
    arm: str,
    pred: pl.DataFrame,
    history: pl.DataFrame,
    h: int,
    params: dict[str, object],
    matured_from,
) -> pl.DataFrame:
    if arm == "base":
        return pred
    if "kind" in ARMS[arm]:
        return apply_attribution(pred, history, h, params, matured_from=matured_from)
    return calibrate(pred, history, h, params, matured_from=matured_from)


def _row_mean_pinball(pred: pl.DataFrame, source: pl.DataFrame, h: int) -> tuple[np.ndarray, pl.DataFrame]:
    """Per-row mean pinball loss aligned to ``(date, target_class)``."""
    truth = source.select("date", "target_class", pl.col(f"y_step_h{h}").alias("y"))
    j = truth.join(pred, on=["date", "target_class"], how="inner").drop_nulls("y").sort(
        ["date", "target_class"]
    )
    y = j["y"].to_numpy()
    loss = np.zeros(j.height)
    for q in (0.1, 0.5, 0.9):
        e = y - j[f"p_{q}"].to_numpy()
        loss += np.maximum(q * e, (q - 1.0) * e)
    return loss / 3.0, j


def run(include_test: bool = False, run_name: str = "v2") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    key = os.environ.get("LSTM_PRED_KEY", "").strip() or _prediction_key()
    LOGGER.info("using cached LSTM predictions key=%s", key)

    splits: dict[str, pl.DataFrame] = {name: load_split(name) for name in ("train", "valid")}
    preds: dict[str, dict[int, pl.DataFrame]] = {
        name: {h: _load_preds(key, name, h) for h in HORIZONS} for name in splits
    }
    valid_start = splits["valid"]["date"].min()

    run_dir = DATA / "runs" / run_name
    param_dir = run_dir / "calibration_params"
    run_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)

    valid_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    fitted: dict[tuple[str, int], dict[str, object]] = {}

    for h in HORIZONS:
        context = _context_frame(preds, splits, ["train", "valid"], h)
        fit_rows = _context_frame(preds, splits, ["valid"], h)
        history = _context_frame(preds, splits, ["train", "valid"], h)
        base_loss, base_join = _row_mean_pinball(preds["valid"][h], splits["valid"], h)
        for arm in ARMS:
            params = _fit_arm(arm, context, fit_rows, h, valid_start)
            fitted[(h, arm)] = params
            if arm in SUGENO_ARMS:
                (param_dir / f"{arm}_h{h}.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
            cal = _apply_arm(arm, preds["valid"][h], history, h, params, valid_start)
            valid_rows.extend({**r, "arm": arm} for r in evaluate(cal, splits["valid"], "valid", arm))
            if arm != "base":
                loss, j = _row_mean_pinball(cal, splits["valid"], h)
                diff = loss - base_loss
                ci = date_block_bootstrap_ci(diff, j["date"].to_numpy(), block_length=max(h, 5))
                bootstrap_rows.append({
                    "h": h, "arm": arm,
                    "d_pinball_vs_base": float(diff.mean()),
                    "boot_lo": ci["boot_lo"], "boot_hi": ci["boot_hi"],
                })
            LOGGER.info("h=%d %s: ok", h, arm)

    valid_df = pl.DataFrame(valid_rows).sort(["h", "arm", "scope"])
    valid_csv = run_dir / "calibration_ablation_valid.csv"
    valid_df.write_csv(valid_csv)
    boot_df = pl.DataFrame(bootstrap_rows)
    boot_df.write_csv(run_dir / "calibration_bootstrap_valid.csv")

    # --- selection: pooled validation pinball, then Winkler. ---
    pooled = valid_df.filter(pl.col("scope") == "POOLED")
    ranking = (
        pooled.group_by("arm")
        .agg(
            pl.col("pinball_mean").mean().alias("mean_pinball"),
            pl.col("winkler_80").mean().alias("mean_winkler"),
            pl.col("cover_80").mean().alias("mean_cover"),
        )
        .sort(["mean_pinball", "mean_winkler"])
    )
    selected = str(ranking["arm"][0])
    LOGGER.info("selected on validation: %s", selected)

    summary = {
        "run": run_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lstm_prediction_key": key,
        "protocol": (
            "all arms fitted on the valid split using frozen LSTM predictions; "
            "selection on valid pooled pinball only; test scored once for the "
            "selected variant when enabled"
        ),
        "selection_rule": "lowest mean pooled validation pinball, ties by mean Winkler",
        "selected": selected,
        "arms": {a: ARMS[a] for a in ARMS},
        "fit": {"epochs": 600, "lr": 0.05, "seed": 0, "split": "valid"},
        "test_included": include_test,
        "validation_ranking": ranking.to_dicts(),
    }
    (run_dir / "calibration_selected.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = _markdown_table(valid_df, boot_df, ranking, selected)
    (run_dir / "calibration_ablation.md").write_text(markdown, encoding="utf-8")
    LOGGER.info("wrote %s", valid_csv)

    if include_test:
        from ml.frozen_test import allow_test_set_access, load_frozen_test

        with allow_test_set_access("one-shot scoring of the validation-selected calibration variant"):
            splits["test"] = load_frozen_test()
        preds["test"] = {h: _load_preds(key, "test", h) for h in HORIZONS}
        LOGGER.warning("test split read once for the selected variant %r", selected)
        test_rows: list[dict[str, object]] = []
        for h in HORIZONS:
            if selected == "base":
                cal = preds["test"][h]
            else:
                history_test = _context_frame(preds, splits, ["train", "valid", "test"], h)
                cal = _apply_arm(selected, preds["test"][h], history_test, h, fitted[(h, selected)], valid_start)
            test_rows.extend({**r, "arm": selected} for r in evaluate(cal, splits["test"], "test", selected))
        test_df = pl.DataFrame(test_rows).sort(["h", "scope"])
        test_df.write_csv(run_dir / "calibration_test_selected.csv")
        (run_dir / "calibration_test_selected.md").write_text(_test_markdown(test_df, selected), encoding="utf-8")
        LOGGER.info("wrote %s", run_dir / "calibration_test_selected.csv")

    print(ranking.to_pandas().to_string(index=False))


def _fmt_arm(arm: str) -> str:
    return arm.replace("_", r"\_")


def _markdown_table(
    valid_df: pl.DataFrame,
    boot_df: pl.DataFrame,
    ranking: pl.DataFrame,
    selected: str,
) -> str:
    lines = [
        "# Calibration ablation — validation split only",
        "",
        "All arms fitted on `valid` with the frozen LSTM quantiles. Lower pinball and",
        "Winkler are better; `cover-80` should be near 0.80.",
        "",
        f"Selected variant: **{selected}** (lowest mean pooled validation pinball).",
        "",
        "## Pooled",
        "",
        "| arm | h | n | pinball | cover-80 | Winkler-80 | width | delta pinball vs base [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    pooled = valid_df.filter(pl.col("scope") == "POOLED").sort(["h", "arm"])
    boot = {(int(r["h"]), str(r["arm"])): r for r in boot_df.to_dicts()}
    for r in pooled.to_dicts():
        b = boot.get((int(r["h"]), str(r["arm"])))
        if b is None:
            delta = "—"
        else:
            delta = f"{b['d_pinball_vs_base']:+.5f} [{b['boot_lo']:+.5f}, {b['boot_hi']:+.5f}]"
        lines.append(
            f"| {_fmt_arm(str(r['arm']))} | {r['h']} | {r['n']} | {r['pinball_mean']:.5f} | "
            f"{r['cover_80']:.4f} | {r['winkler_80']:.4f} | {r['width']:.4f} | {delta} |"
        )

    lines += [
        "",
        "## Per class",
        "",
        "| arm | h | class | n | pinball | cover-80 | Winkler-80 |",
        "|---|---|---|---|---|---|---|",
    ]
    per = valid_df.filter(pl.col("scope") != "POOLED").sort(["h", "scope", "arm"])
    for r in per.to_dicts():
        lines.append(
            f"| {_fmt_arm(str(r['arm']))} | {r['h']} | {r['scope']} | {r['n']} | "
            f"{r['pinball_mean']:.5f} | {r['cover_80']:.4f} | {r['winkler_80']:.4f} |"
        )

    lines += [
        "",
        "## Selection ranking (mean over horizons)",
        "",
        "| rank | arm | mean pinball | mean cover-80 | mean Winkler-80 |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranking.to_dicts(), start=1):
        lines.append(
            f"| {i} | {_fmt_arm(str(r['arm']))} | {r['mean_pinball']:.5f} | "
            f"{r['mean_cover']:.4f} | {r['mean_winkler']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _test_markdown(test_df: pl.DataFrame, selected: str) -> str:
    lines = [
        f"# Frozen test — selected variant `{selected}`",
        "",
        "Scored once, after selection on validation only.",
        "",
        "| h | scope | n | pinball | cover-80 | Winkler-80 | width |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in test_df.sort(["h", "scope"]).to_dicts():
        lines.append(
            f"| {r['h']} | {r['scope']} | {r['n']} | {r['pinball_mean']:.5f} | "
            f"{r['cover_80']:.4f} | {r['winkler_80']:.4f} | {r['width']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    include_test = os.environ.get("CALIB_INCLUDE_TEST", "").strip().lower() in ("1", "true", "yes")
    run_name = os.environ.get("CALIB_RUN", "").strip() or "v2"
    run(include_test=include_test, run_name=run_name)


if __name__ == "__main__":
    main()
