"""Reproducible bounded compact-Sugeno trials; NEVER reads the test split.

Run: uv run python -m ml.compact_sugeno_experiment --run-name compact_v1
Protocol: docs/compact_sugeno_protocol.md. All nominations are saved before
confirmation scoring. Existing run directories are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ml.compact_sugeno import (
    apply_compact,
    compact_effects,
    compute_compact_inputs,
    fit_compact,
    matured_fit_rows,
)
from ml.evaluation import date_block_bootstrap_ci
from ml.fuzzy_calibration import calibrate as apply_legacy
from ml.fuzzy_calibration import fit_fuzzy

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("Capesize", "Panamax", "Supramax")
HORIZONS = (1, 7, 30, 90)
TUNING_FOLDS = (
    (date(2023, 10, 1), date(2023, 12, 1)),
    (date(2023, 12, 1), date(2024, 1, 1)),
)
LONG_TUNING_FOLDS = (
    (date(2022, 1, 1), date(2022, 7, 1)),
    (date(2023, 1, 1), date(2023, 7, 1)),
)
CONFIRM_START = date(2024, 4, 1)
LOG = logging.getLogger(__name__)


def candidate_grid(family: str) -> list[dict[str, object]]:
    if family not in ("volatility", "residual"):
        raise ValueError("unknown trial family")
    feature = "vol_ratio" if family == "volatility" else "error_ratio"
    return [
        {
            "name": f"{family}_r{rules}_{'asym' if asym else 'sym'}_rec{rec_name}_l{penalty:g}",
            "family": family,
            "kwargs": {
                "n_rules": rules,
                "asymmetric": asym,
                "width_feature": feature,
                "shrinkage": penalty,
                "recency_half_life_months": rec,
            },
        }
        for rules in (1, 2, 3)
        for asym in (False, True)
        for rec, rec_name in ((None, "flat"), (12, "12"), (24, "24"))
        for penalty in (0.0, 0.01, 0.1)
    ]


def constant_grid() -> list[dict[str, object]]:
    """Constant W+L comparators: one rule, same fitter, bounds, and weights."""
    return [
        {
            "name": f"constant_{('asym' if asym else 'sym')}_rec{rec_name}",
            "family": "constant",
            "kwargs": {
                "n_rules": 1,
                "asymmetric": asym,
                "width_feature": "vol_ratio",
                "shrinkage": 0.0,
                "recency_half_life_months": rec,
            },
        }
        for asym in (False, True)
        for rec, rec_name in ((None, "flat"), (12, "12"), (24, "24"))
    ]


def balanced_support(frame: pl.DataFrame) -> pl.DataFrame:
    """Identical finite, three-class date support for every scored model."""
    finite = frame.filter(
        pl.all_horizontal(
            [
                pl.col(c).is_finite()
                for c in ("y", "log_value", "p_0.1", "p_0.5", "p_0.9")
            ]
        )
    )
    keys = (
        finite.group_by("date")
        .agg(pl.col("target_class").n_unique().alias("n_class"))
        .filter(pl.col("n_class") == len(CLASSES))
        .select("date")
    )
    return finite.join(keys, on="date", how="inner").sort(["date", "target_class"])


def load_validation(
    key: str,
    source_dir: Path | None = None,
) -> tuple[dict[int, pl.DataFrame], pl.DataFrame, dict[str, str]]:
    """Explicit validation-only loader; daily feature scan stops at its last date."""
    data = source_dir or ROOT / "src/data"
    source = pl.read_parquet(data / "samples_valid.parquet").filter(
        pl.col("target_class").is_in(CLASSES)
    )
    hashes = {
        "samples_valid": hashlib.sha256(
            source.sort(["date", "target_class"]).serialize()
        ).hexdigest()
    }
    contexts = {}
    for h in HORIZONS:
        path = data / f"preds/lstm_{key}_valid_h{h}.parquet"
        pred = pl.read_parquet(path).filter(pl.col("target_class").is_in(CLASSES))
        if pred["h"].unique().to_list() != [h]:
            raise ValueError(f"wrong forecast horizon in {path}")
        truth = source.select(
            "date", "target_class", "log_value", pl.col(f"y_step_h{h}").alias("y")
        )
        context = pred.join(
            truth, on=["date", "target_class"], how="left", validate="1:1"
        )
        if context["log_value"].null_count():
            raise ValueError("cached forecast keys are absent from validation truth")
        contexts[h] = context
        hashes[f"valid_predictions_h{h}"] = hashlib.sha256(
            pred.sort(["date", "target_class"]).serialize()
        ).hexdigest()
    mapping = {"BC_INDEX": "Capesize", "BPI_INDEX": "Panamax", "BSI_INDEX": "Supramax"}
    daily = (
        pl.scan_parquet(ROOT / "src/data/master_long.parquet")
        .filter(
            pl.col("series_id").is_in(mapping)
            & (pl.col("date") <= source["date"].max())
        )
        .select(
            "date",
            pl.col("series_id").replace_strict(mapping).alias("target_class"),
            pl.when(pl.col("value") > 0)
            .then(pl.col("value").log())
            .otherwise(None)
            .alias("log_value"),
        )
        .collect()
    )
    excluded = daily.filter(
        pl.col("log_value").is_null() | ~pl.col("log_value").is_finite()
    ).height
    if excluded:
        LOG.warning(
            "Excluding %d daily observations outside the positive log-index domain; feature gaps use past-only LOCF",
            excluded,
        )
        daily = daily.filter(pl.col("log_value").is_finite())
    hashes["daily_levels_through_validation_end"] = hashlib.sha256(
        daily.sort(["date", "target_class"]).serialize()
    ).hexdigest()
    return contexts, daily, hashes


def row_loss(frame: pl.DataFrame, pred: pl.DataFrame) -> np.ndarray:
    if (
        frame.select("date", "target_class").to_dicts()
        != pred.select("date", "target_class").to_dicts()
    ):
        raise ValueError("scoring rows are not aligned")
    error = (
        frame["y"].to_numpy()[:, None]
        - pred.select("p_0.1", "p_0.5", "p_0.9").to_numpy()
    )
    tau = np.array([0.1, 0.5, 0.9])
    return np.maximum(tau * error, (tau - 1) * error).mean(axis=1)


def score(frame: pl.DataFrame, pred: pl.DataFrame) -> dict[str, float | int]:
    loss = row_loss(frame, pred)
    y = frame["y"].to_numpy()
    lo, median, hi = (pred[c].to_numpy() for c in ("p_0.1", "p_0.5", "p_0.9"))
    winkler = hi - lo + 10 * np.maximum(lo - y, 0) + 10 * np.maximum(y - hi, 0)
    return {
        "n": len(y),
        "n_dates": frame["date"].n_unique(),
        "pinball": float(loss.mean()),
        "cover_80": float(((y >= lo) & (y <= hi)).mean()),
        "winkler_80": float(winkler.mean()),
        "median_mae": float(np.abs(y - median).mean()),
    }


def pred_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select("date", "target_class", "h", "p_0.1", "p_0.5", "p_0.9")


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def select_candidate(
    rows: list[dict[str, object]], candidates: list[dict[str, object]]
) -> dict[str, object]:
    ranking = []
    for candidate in candidates:
        sub = [r for r in rows if r["candidate"] == candidate["name"]]
        if len(sub) != len(TUNING_FOLDS):
            raise ValueError("each candidate needs every tuning fold")
        kwargs = candidate["kwargs"]
        n_parameters = int(kwargs["n_rules"]) + (
            int(kwargs["n_rules"]) if kwargs["asymmetric"] else 0
        ) + 1
        ranking.append(
            (
                float(np.mean([r["relative_vs_constant"] for r in sub])),
                n_parameters,
                -float(kwargs["shrinkage"]),
                str(candidate["name"]),
                candidate,
            )
        )
    ranking.sort(key=lambda x: (round(x[0], 10), x[1], x[2], x[3]))
    return {**ranking[0][-1], "tuning_relative_vs_constant": ranking[0][0]}


def exploratory_success(rows: list[dict[str, object]]) -> bool:
    deltas = np.array([r["relative_vs_best_constant"] for r in rows])
    return bool(
        deltas.size == 4
        and deltas.mean() <= -0.01
        and (deltas < 0).sum() >= 2
        and deltas.max() <= 0.05
    )


def run(
    run_name: str, key: str = "69e158d2977b", source_dir: Path | None = None
) -> Path:
    if (
        not run_name
        or Path(run_name).name != run_name
        or run_name in (".", "..")
        or "\\" in run_name
    ):
        raise ValueError("run_name must be a single directory name")
    out = ROOT / "src/data/runs" / run_name
    out.mkdir(parents=True, exist_ok=False)
    (out / "params").mkdir()
    protocol = (ROOT / "docs/compact_sugeno_protocol.md").read_text(encoding="utf-8")
    (out / "protocol.md").write_text(protocol, encoding="utf-8")
    contexts, daily, hashes = load_validation(key, source_dir)
    tuning_folds = LONG_TUNING_FOLDS if source_dir is not None else TUNING_FOLDS
    all_candidates = candidate_grid("volatility") + candidate_grid("residual")
    constants = constant_grid()
    metadata = {
        "prediction_key": key,
        "test_read": False,
        "classes": CLASSES,
        "log_domain_note": "Nonpositive daily indices cannot be logged and are omitted before causal feature-grid LOCF; nonfinite targets are never filled and are excluded on common support.",
        "input_sha256": hashes,
        "protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(),
        "tuning_folds": [[str(a), str(b)] for a, b in tuning_folds],
        "source_dir": str(source_dir or ROOT / "src/data"),
        "confirmation_start": str(CONFIRM_START),
        "candidate_grid": all_candidates,
        "constant_grid": constants,
        "legacy_references": ["legacy_sugeno0", "legacy_sugeno1_split"]
        if source_dir is not None
        else [],
        "uncertainty_note": "Exploratory date-block intervals do not adjust for model selection; the existing validation data were previously inspected.",
        "status": "tuning",
    }
    dump(out / "metadata.json", metadata)
    inputs = {h: compute_compact_inputs(contexts[h], daily, h) for h in HORIZONS}
    support = {h: balanced_support(contexts[h]) for h in HORIZONS}
    tuning = []
    selection = {}
    finalist_params = {}
    # Complete ALL fitting and nomination before any confirmation losses are computed.
    for h in HORIZONS:
        h_rows = []
        for fold, (start, end) in enumerate(tuning_folds, 1):
            fit = matured_fit_rows(support[h], start, h)
            validation = support[h].filter(
                (pl.col("date") >= start) & (pl.col("date") < end)
            )
            if fit.height < 30 or validation.height < 15:
                raise ValueError(
                    f"insufficient rows h={h}, fold={fold}: fit={fit.height}, score={validation.height}"
                )
            constant = fit_compact(fit, inputs[h], h, n_rules=1, shrinkage=0)
            constant_pred = apply_compact(pred_columns(validation), inputs[h], constant)
            reference = fit_compact(fit, inputs[h], h, n_rules=1, shrinkage=0)
            constant_pred = apply_compact(pred_columns(validation), inputs[h], reference)
            ref_metrics = score(validation, constant_pred)
            for candidate in [*all_candidates, *constants]:
                params = fit_compact(fit, inputs[h], h, **candidate["kwargs"])
                pred = apply_compact(pred_columns(validation), inputs[h], params)
                metrics = score(validation, pred)
                row = {
                    "h": h,
                    "fold": fold,
                    "candidate": candidate["name"],
                    "family": candidate["family"],
                    "fit_n": fit.height,
                    "fit_n_dates": fit["date"].n_unique(),
                    "fit_last_issue": str(fit["date"].max()),
                    "score_start": str(start),
                    "score_end_exclusive": str(end),
                    "n_parameters": params["n_parameters"],
                    **metrics,
                    "constant_pinball": ref_metrics["pinball"],
                    "base_pinball": score(validation, pred_columns(validation))["pinball"],
                    "relative_vs_constant": metrics["pinball"] / ref_metrics["pinball"] - 1,
                }
                tuning.append(row)
                h_rows.append(row)
                dump(
                    out / "params" / f"tuning_h{h}_fold{fold}_{candidate['name']}.json",
                    params,
                )
        finalists = {
            family: select_candidate(h_rows, candidate_grid(family))
            for family in ("volatility", "residual")
        }
        nominated = min(
            finalists.values(),
            key=lambda c: (round(c["tuning_relative_vs_constant"], 10), c["name"]),
        )
        constant_best = min(
            (
                (
                    round(
                        float(
                            np.mean(
                                [
                                    r["pinball"]
                                    for r in h_rows
                                    if r["candidate"] == c["name"]
                                ]
                            )
                        ),
                        10,
                    ),
                    int(c["kwargs"]["n_rules"]),
                    str(c["name"]),
                    c,
                )
                for c in constants
            ),
            key=lambda x: x[:3],
        )
        selection[str(h)] = {
            "finalists": finalists,
            "nominated": nominated,
            "constant_best": {
                "name": constant_best["name"],
                "kwargs": constant_best["kwargs"],
                "tuning_pinball": constant_best[0],
            },
        }
        fit = matured_fit_rows(support[h], CONFIRM_START, h)
        for family, candidate in finalists.items():
            params = fit_compact(fit, inputs[h], h, **candidate["kwargs"])
            finalist_params[(h, family)] = params
            dump(out / "params" / f"confirmation_h{h}_{family}.json", params)
        constant = fit_compact(fit, inputs[h], h, n_rules=1, shrinkage=0)
        finalist_params[(h, "constant")] = constant
        dump(out / "params" / f"confirmation_h{h}_constant.json", constant)
        constant_best_params = fit_compact(fit, inputs[h], h, **constant_best["kwargs"])
        finalist_params[(h, "constant_best")] = constant_best_params
        dump(
            out / "params" / f"confirmation_h{h}_constant_best.json",
            constant_best_params,
        )
        if source_dir is not None:
            for name, kwargs in (
                ("legacy_sugeno0", {}),
                ("legacy_sugeno1_split", {"linear": True, "split_width": True}),
            ):
                params, _ = fit_fuzzy(contexts[h], fit, h, **kwargs)
                params["stats"]["n_fit"] = fit.height
                params["stats"]["class_pinball"] = (
                    None  # per-class offsets are disabled
                )
                params["n_parameters"] = sum(
                    sum(
                        np.asarray(params[system][field]).size
                        for field in ("centers", "widths", "consequents")
                    )
                    + (
                        0
                        if params[system]["linear_coefs"] is None
                        else np.asarray(params[system]["linear_coefs"]).size
                    )
                    for system in ("width", "loc")
                )
                finalist_params[(h, name)] = params
                dump(out / "params" / f"confirmation_h{h}_{name}.json", params)
        LOG.info("h=%s locked nomination=%s", h, nominated["name"])
    pl.DataFrame(tuning).write_csv(out / "tuning.csv")
    dump(out / "selection.json", selection)
    metadata["status"] = "selection_locked_before_confirmation"
    dump(out / "metadata.json", metadata)

    confirmation = []
    prediction_frames = []
    for h in HORIZONS:
        held = support[h].filter(pl.col("date") >= CONFIRM_START)
        base = pred_columns(held)
        constant = apply_compact(base, inputs[h], finalist_params[(h, "constant")])
        best_constant = apply_compact(
            base, inputs[h], finalist_params[(h, "constant_best")]
        )
        ref = score(held, constant)
        best_ref = score(held, best_constant)
        constant_loss = row_loss(held, constant)
        best_loss = row_loss(held, best_constant)
        for family in (
            "base",
            "constant",
            "constant_best",
            "volatility",
            "residual",
            "nominated",
            *metadata["legacy_references"],
        ):
            actual_family = (
                selection[str(h)]["nominated"]["family"]
                if family == "nominated"
                else family
            )
            params = finalist_params.get((h, actual_family))
            if family.startswith("legacy_"):
                pred = apply_legacy(base, contexts[h], h, params)
                effect = base.select("date", "target_class").with_columns(
                    pl.Series(
                        "width_multiplier",
                        (pred["p_0.9"] - pred["p_0.1"]).to_numpy()
                        / (base["p_0.9"] - base["p_0.1"]).to_numpy(),
                    ),
                    pl.Series(
                        "location_shift", (pred["p_0.5"] - base["p_0.5"]).to_numpy()
                    ),
                )
            else:
                pred = (
                    base
                    if family == "base"
                    else constant
                    if family == "constant"
                    else best_constant
                    if family == "constant_best"
                    else apply_compact(base, inputs[h], params)
                )
                effect = (
                    compact_effects(base, inputs[h], params)
                    if params is not None
                    else None
                )
            metrics = score(held, pred)
            losses = row_loss(held, pred)
            diff = losses - constant_loss
            # Do not report degenerate intervals when the window holds <2 blocks.
            block = max(5, int(np.ceil(h * 5 / 7)))
            estimable = held["date"].n_unique() >= 2 * block
            ci = (
                date_block_bootstrap_ci(
                    diff,
                    held["date"].to_numpy(),
                    block_length=block,
                    n_boot=2000,
                    seed=0,
                )
                if estimable
                else {"boot_lo": None, "boot_hi": None}
            )
            row = {
                "h": h,
                "model": family,
                "actual_family": actual_family,
                "fit_n": params["stats"]["n_fit"] if params else 0,
                "n_parameters": params["n_parameters"] if params else 0,
                **metrics,
                "relative_vs_constant": metrics["pinball"] / ref["pinball"] - 1,
                "difference_vs_constant": float(diff.mean()),
                "relative_vs_best_constant": metrics["pinball"] / best_ref["pinball"] - 1,
                "difference_vs_best_constant": float((losses - best_loss).mean()),
                "boot_lo": ci["boot_lo"],
                "boot_hi": ci["boot_hi"],
                "block_dates": block,
                "bootstrap_estimable": estimable,
                "width_multiplier_sd": float(effect["width_multiplier"].std(ddof=0))
                if effect is not None
                else 0.0,
                "location_shift_sd": float(effect["location_shift"].std(ddof=0))
                if effect is not None
                else 0.0,
            }
            confirmation.append(row)
            audit = held.select("date", "target_class", "h", "y").join(
                pred.drop("h"), on=["date", "target_class"]
            )
            audit = audit.with_columns(
                pl.lit(family).alias("model"),
                pl.Series("row_pinball", losses),
                pl.Series("constant_row_pinball", constant_loss),
            )
            if effect is not None:
                audit = audit.join(
                    effect.select(
                        "date", "target_class", "width_multiplier", "location_shift"
                    ),
                    on=["date", "target_class"],
                )
            prediction_frames.append(audit)
            LOG.info(
                "confirmation h=%s model=%s pinball=%.6f delta=%+.2f%%",
                h,
                family,
                metrics["pinball"],
                100 * row["relative_vs_constant"],
            )
    pl.DataFrame(confirmation).write_csv(out / "confirmation.csv")
    predictions = pl.concat(prediction_frames, how="diagonal_relaxed")
    predictions.write_parquet(out / "confirmation_predictions.parquet")
    # Per-class metrics are retained even if the pooled score loses.
    class_rows = []
    for group in predictions.partition_by(["h", "model", "target_class"]):
        h, model, cls = group["h"][0], group["model"][0], group["target_class"][0]
        class_rows.append(
            {
                "h": h,
                "model": model,
                "target_class": cls,
                **score(group, pred_columns(group)),
            }
        )
    pl.DataFrame(class_rows).write_csv(out / "confirmation_by_class.csv")
    nominated = [r for r in confirmation if r["model"] == "nominated"]
    metadata.update(
        status="completed",
        exploratory_success=exploratory_success(nominated),
        mean_horizon_relative_vs_constant=float(
            np.mean([r["relative_vs_constant"] for r in nominated])
        ),
        mean_horizon_relative_vs_best_constant=float(
            np.mean([r["relative_vs_best_constant"] for r in nominated])
        ),
        confirmation_end=str(max(contexts[h]["date"].max() for h in HORIZONS)),
    )
    dump(out / "metadata.json", metadata)
    write_summary(out, confirmation, metadata, selection)
    return out


def write_summary(
    out: Path,
    rows: list[dict[str, object]],
    meta: dict[str, object],
    selection: dict[str, object],
) -> None:
    lines = [
        "# Compact Sugeno — chronological validation confirmation",
        "",
        "No frozen test targets or predictions were read. Nomination uses earlier tuning folds only.",
        "This is exploratory: validation was used in previous experiments and base-model selection.",
        "",
        f"Predeclared exploratory success (vs best constant): **{meta['exploratory_success']}**.",
        f"Nominated pipeline mean horizon-relative pinball change vs constant: **{100 * meta['mean_horizon_relative_vs_constant']:+.2f}%**; vs best constant: **{100 * meta['mean_horizon_relative_vs_best_constant']:+.2f}%**.",
        "",
        "| h | model | fit n | score n | parameters | pinball | Δ vs constant | Δ vs best constant | coverage | Winkler | width multiplier SD | shift SD |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['h']} | {r['model']} | {r['fit_n']} | {r['n']} | {r['n_parameters']} | {r['pinball']:.6f} | {100 * r['relative_vs_constant']:+.2f}% | {100 * r['relative_vs_best_constant']:+.2f}% | {r['cover_80']:.3f} | {r['winkler_80']:.4f} | {r['width_multiplier_sd']:.5f} | {r['location_shift_sd']:.5f} |"
        )
    lines += ["", "## Frozen nominations", ""]
    for h, s in selection.items():
        lines.append(
            f"- h={h}: `{s['nominated']['name']}`; tuning change vs constant {100 * s['nominated']['tuning_relative_vs_constant']:+.2f}%."
        )
    lines += [
        "",
        "## Limits and next data expansion",
        "",
        "Fitting counts are rows, not independent observations. Same-date vessel classes are correlated and long-horizon targets overlap. At h=90 the confirmation window contains very few independent blocks; bootstrap intervals must not be treated as precise significance evidence.",
        "To fit a larger Sugeno system, generate additional walk-forward OUT-OF-SAMPLE base forecasts or reserve a longer calibration period. Do not pad this dataset with in-sample training errors or test outcomes. More epochs/duplicated rows do not add information.",
        "The constant comparators (constant, constant_best) are one-rule models fitted by the SAME exact 1-D piecewise-linear search, bounds, and recency options — no scipy, no LP, no learned antecedents. These results do not replace the paper's historical test table.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="compact_v1")
    parser.add_argument("--prediction-key", default="69e158d2977b")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="isolated long-calibration data/predictions directory",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(run(args.run_name, args.prediction_key, args.source_dir))


if __name__ == "__main__":
    main()
