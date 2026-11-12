"""Generate every paper table from saved metric CSVs.

No number in the paper should be hand-typed: this script reads the run outputs
and computes every percentage, delta and bold marker in code. Missing files are
skipped with a note, so the script is safe to run before an experiment finishes.

Run from the repository root:

    uv run python -m ml.paper_tables
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"
OUT: Final[Path] = REPO_ROOT / "docs" / "paper_tables.md"
HORIZONS: Final[tuple[int, ...]] = (1, 7, 30, 90)
CLASSES: Final[tuple[str, ...]] = ("Capesize", "Panamax", "Supramax", "Handysize")
MAIN_MODELS: Final[dict[str, str]] = {
    "lstm": "Base",
    "lstm_fuzzy": "Sugeno (zero-order)",
    "sugeno_sel": "Sugeno (selected)",
    "cqr": "CQR",
    "aci": "ACI",
}
VALIDATION_ARMS: Final[dict[str, str]] = {
    "base": "Base (no calibration)",
    "const_wl": "Constant width + location",
    "linear_state": "Linear state-dependent",
    "sugeno0": "Sugeno zero-order",
    "sugeno1": "Sugeno first-order",
    "sugeno_split": "Sugeno split width",
    "sugeno_lin_split": "Sugeno first-order + split width",
    "sugeno_cls": "Sugeno + class shrinkage",
    "sugeno_reg": "Sugeno + early stop / weight decay",
    "sugeno_full": "Sugeno full combination",
}
ATTR_LABELS: Final[dict[str, str]] = {
    "attr_constant_width": "Constant width",
    "attr_constant_width_location": "Constant width + location",
    "attr_constant_width_rollbias": "Constant width + rolling bias",
    "attr_linear": "Linear state-dependent",
}
STACK_LABELS: Final[dict[str, str]] = {
    "lstm": "Base",
    "cqr": "CQR",
    "fuz_cqr": "Sugeno on CQR",
}


def _read(name: str) -> pl.DataFrame | None:
    path = DATA / name
    if not path.exists():
        return None
    return pl.read_csv(path)


def _pct(value: float, reference: float) -> str:
    if reference == 0 or reference != reference:
        return "—"
    return f"{100.0 * (value / reference - 1.0):+.1f}%"


def _bold(value: float, best: float) -> str:
    return f"**{value:.5f}**" if abs(value - best) < 1e-12 else f"{value:.5f}"


def main_table(metrics: pl.DataFrame) -> list[str]:
    m = metrics.filter(
        (pl.col("split") == "test")
        & (pl.col("scope") == "POOLED")
        & (pl.col("model").is_in(list(MAIN_MODELS)))
    )
    lines = [
        "## Table 1 — Main test results (pooled, common support)",
        "",
        "Delta is the relative change in mean pinball against the base model, computed here.",
        "",
        "| h | model | n | mean pinball | Δ vs base | cover-80 | Winkler-80 |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in HORIZONS:
        sub = m.filter(pl.col("h") == h)
        if sub.is_empty():
            continue
        base = sub.filter(pl.col("model") == "lstm")
        base_pin = float(base["pinball_mean"][0]) if not base.is_empty() else float("nan")
        best_pin = float(sub["pinball_mean"].min())
        best_win = float(sub["winkler_80"].min())
        for model in MAIN_MODELS:
            r = sub.filter(pl.col("model") == model)
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            pin = float(row["pinball_mean"])
            lines.append(
                f"| {h} | {MAIN_MODELS[model]} | {row['n']} | {_bold(pin, best_pin)} "
                f"| {_pct(pin, base_pin) if model != 'lstm' else '—'} "
                f"| {row['cover_80']:.3f} | {_bold(float(row['winkler_80']), best_win)} |"
            )
    lines.append("")
    return lines


def per_class_h90(metrics: pl.DataFrame) -> list[str]:
    m = metrics.filter(
        (pl.col("split") == "test")
        & (pl.col("h") == 90)
        & (pl.col("scope").is_in(list(CLASSES)))
        & (pl.col("model").is_in(list(MAIN_MODELS)))
    )
    lines = [
        "## Table 2 — h=90 per class (test)",
        "",
        "Bold marks the lowest mean pinball in each class. The Sugeno layer is **not** best in all four.",
        "",
        "| class | model | n | mean pinball | cover-80 | Winkler-80 |",
        "|---|---|---|---|---|---|",
    ]
    for cls in CLASSES:
        sub = m.filter(pl.col("scope") == cls)
        if sub.is_empty():
            continue
        best_pin = float(sub["pinball_mean"].min())
        best_win = float(sub["winkler_80"].min())
        for model in MAIN_MODELS:
            r = sub.filter(pl.col("model") == model)
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            lines.append(
                f"| {cls} | {MAIN_MODELS[model]} | {row['n']} "
                f"| {_bold(float(row['pinball_mean']), best_pin)} "
                f"| {row['cover_80']:.3f} | {_bold(float(row['winkler_80']), best_win)} |"
            )
    lines.append("")
    return lines


def attribution_table(metrics: pl.DataFrame | None) -> list[str]:
    lines = [
        "## Table 3 — Attribution baselines (test, pooled)",
        "",
        "All arms use the validation split, the same pinball objective and the same tanh bounds.",
        "Median loss is the P50 pinball component; Winkler is the endpoint score.",
        "",
        "| h | calibrator | n | mean pinball | median loss | cover-80 | Winkler-80 |",
        "|---|---|---|---|---|---|---|",
    ]
    if metrics is None:
        lines += ["| — | (run `ml.attribution_baselines` first) | — | — | — | — | — |", ""]
        return lines
    m = metrics.filter((pl.col("split") == "test") & (pl.col("scope") == "POOLED"))
    for h in HORIZONS:
        sub = m.filter(pl.col("h") == h)
        if sub.is_empty():
            continue
        best_pin = float(sub["pinball_mean"].min())
        for model, label in ATTR_LABELS.items():
            r = sub.filter(pl.col("model") == model)
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            lines.append(
                f"| {h} | {label} | {row['n']} | {_bold(float(row['pinball_mean']), best_pin)} "
                f"| {row['pinball_0.5']:.5f} | {row['cover_80']:.3f} | {row['winkler_80']:.4f} |"
            )
    lines.append("")
    return lines


def stack_table(metrics: pl.DataFrame | None) -> list[str]:
    lines = [
        "## Table 4 — Fuzzy-on-CQR stack (test, pooled)",
        "",
        "Post-hoc arm. The fuzzy stage does **not** inherit CQR's coverage guarantee.",
        "",
        "| h | model | n | mean pinball | cover-80 | Winkler-80 |",
        "|---|---|---|---|---|---|",
    ]
    if metrics is None:
        lines += ["| — | (run `ml.stack_experiment` first) | — | — | — | — |", ""]
        return lines
    m = metrics.filter((pl.col("split") == "test") & (pl.col("scope") == "POOLED"))
    for h in HORIZONS:
        sub = m.filter(pl.col("h") == h)
        if sub.is_empty():
            continue
        best_pin = float(sub["pinball_mean"].min())
        best_win = float(sub["winkler_80"].min())
        for model, label in STACK_LABELS.items():
            r = sub.filter(pl.col("model") == model)
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            lines.append(
                f"| {h} | {label} | {row['n']} | {_bold(float(row['pinball_mean']), best_pin)} "
                f"| {row['cover_80']:.3f} | {_bold(float(row['winkler_80']), best_win)} |"
            )
    lines.append("")
    return lines


def bootstrap_table(dm: pl.DataFrame | None) -> list[str]:
    lines = [
        "## Appendix A — Pairwise uncertainty (date-block bootstrap)",
        "",
        "Differences are Sugeno minus competitor in mean pinball (negative favours Sugeno).",
        "The DM p-value is shown only for completeness; the date-block interval is the primary evidence.",
        "Block sensitivity uses half and twice the primary block length.",
        "",
        "| split | h | vs | mean diff | DM p | date-block 95% | block/2 95% | block×2 95% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if dm is None:
        lines += ["| — | — | — | — | — | — | — | — |", ""]
        return lines
    m = dm.filter((pl.col("quantile").cast(pl.Utf8) == "mean") & (pl.col("split") == "test"))
    for r in m.sort(["h", "competitor"]).iter_rows(named=True):
        lines.append(
            f"| {r['split']} | {r['h']} | {r['competitor']} | {r['dm_mean_diff']:+.5f} "
            f"| {r['dm_p']:.3f} | [{r['boot_lo']:+.5f}, {r['boot_hi']:+.5f}] "
            f"| [{r['block_small_lo']:+.5f}, {r['block_small_hi']:+.5f}] "
            f"| [{r['block_large_lo']:+.5f}, {r['block_large_hi']:+.5f}] |"
        )
    lines.append("")
    return lines


def _append_selected(main_metrics: pl.DataFrame | None) -> pl.DataFrame | None:
    """Add the validation-selected Sugeno variant's one-shot test rows."""
    path = DATA / "runs" / "v2" / "calibration_test_selected.csv"
    if not path.exists():
        return main_metrics
    sel = pl.read_csv(path).with_columns(pl.lit("sugeno_sel").alias("model")).drop("arm")
    if main_metrics is None:
        return sel
    return pl.concat([main_metrics, sel], how="diagonal_relaxed")


def validation_ablation_table(valid: pl.DataFrame | None) -> list[str]:
    """Table 5: validation-only calibration ablation, pooled."""
    lines = [
        "## Table 5 — Validation-only calibration ablation (pooled)",
        "",
        "Every arm is fitted on `valid` with frozen LSTM quantiles and scored on `valid`;",
        "the test split is not involved. Lower pinball and Winkler are better.",
        "",
        "| h | arm | n | mean pinball | cover-80 | Winkler-80 |",
        "|---|---|---|---|---|---|",
    ]
    if valid is None:
        lines += ["| — | (run `ml.calibration_ablation` first) | — | — | — | — |", ""]
        return lines
    m = valid.filter(pl.col("scope") == "POOLED")
    for h in HORIZONS:
        sub = m.filter(pl.col("h") == h)
        if sub.is_empty():
            continue
        best_pin = float(sub["pinball_mean"].min())
        for arm, label in VALIDATION_ARMS.items():
            r = sub.filter(pl.col("arm") == arm)
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            lines.append(
                f"| {h} | {label} | {row['n']} | {_bold(float(row['pinball_mean']), best_pin)} "
                f"| {row['cover_80']:.3f} | {row['winkler_80']:.4f} |"
            )
    lines.append("")
    return lines


def validation_per_class_table(valid: pl.DataFrame | None) -> list[str]:
    """Table 5b: validation ablation per class at h=90."""
    keep = ("base", "const_wl", "linear_state", "sugeno0", "sugeno_lin_split")
    lines = [
        "## Table 5b — Validation ablation at h=90, per class",
        "",
        "Same arms and protocol as Table 5, restricted to the long horizon and split by class.",
        "",
        "| class | arm | n | mean pinball | cover-80 | Winkler-80 |",
        "|---|---|---|---|---|---|",
    ]
    if valid is None:
        lines += ["| — | (run `ml.calibration_ablation` first) | — | — | — | — |", ""]
        return lines
    m = valid.filter((pl.col("h") == 90) & (pl.col("scope").is_in(list(CLASSES))))
    for cls in CLASSES:
        sub = m.filter(pl.col("scope") == cls)
        if sub.is_empty():
            continue
        best_pin = float(sub["pinball_mean"].min())
        for arm in keep:
            r = sub.filter(pl.col("arm") == arm)
            if r.is_empty():
                continue
            row = r.row(0, named=True)
            lines.append(
                f"| {cls} | {VALIDATION_ARMS[arm]} | {row['n']} | {_bold(float(row['pinball_mean']), best_pin)} "
                f"| {row['cover_80']:.3f} | {row['winkler_80']:.4f} |"
            )
    lines.append("")
    return lines


def selected_test_table(selected: pl.DataFrame | None, attr: pl.DataFrame | None, main_m: pl.DataFrame | None) -> list[str]:
    """Table 6: one-shot test score of the validation-selected variant."""
    lines = [
        "## Table 6 — Frozen test: selected variant vs base and frozen simple calibrators",
        "",
        "The Sugeno row is the only arm scored on test in the new run (one-shot, after",
        "selection on validation). Base and simple-calibrator test numbers are the",
        "previously frozen reference run, not new test reads.",
        "",
        "| h | model | n | mean pinball | cover-80 | Winkler-80 |",
        "|---|---|---|---|---|---|",
    ]
    if selected is None:
        lines += ["| — | (run `ml.calibration_ablation` with CALIB_INCLUDE_TEST=1) | — | — | — | — |", ""]
        return lines
    sel = selected.filter(pl.col("scope") == "POOLED")
    for h in HORIZONS:
        rows: list[tuple[str, dict[str, object]]] = []
        r = sel.filter(pl.col("h") == h)
        if not r.is_empty():
            rows.append(("Sugeno (selected)", r.row(0, named=True)))
        if main_m is not None:
            b = main_m.filter(
                (pl.col("split") == "test") & (pl.col("h") == h)
                & (pl.col("scope") == "POOLED") & (pl.col("model") == "lstm")
            )
            if not b.is_empty():
                rows.append(("Base", b.row(0, named=True)))
        if attr is not None:
            a = attr.filter((pl.col("split") == "test") & (pl.col("h") == h) & (pl.col("scope") == "POOLED"))
            for model, label in ATTR_LABELS.items():
                ar = a.filter(pl.col("model") == model)
                if not ar.is_empty():
                    rows.append((label, ar.row(0, named=True)))
        if not rows:
            continue
        best_pin = min(float(rw["pinball_mean"]) for _, rw in rows)
        best_win = min(float(rw["winkler_80"]) for _, rw in rows)
        for label, rw in rows:
            lines.append(
                f"| {h} | {label} | {rw['n']} | {_bold(float(rw['pinball_mean']), best_pin)} "
                f"| {rw['cover_80']:.3f} | {_bold(float(rw['winkler_80']), best_win)} |"
            )
    lines.append("")
    return lines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main_metrics = _append_selected(_read("baseline_metrics_common.csv"))
    attr_metrics = _read("attribution_metrics.csv")
    stack_metrics = _read("fuzzy_cqr_stack.csv")
    dm = _read("baseline_dm.csv")
    valid_ablation = _read("runs/v2/calibration_ablation_valid.csv")
    selected_test = _read("runs/v2/calibration_test_selected.csv")

    lines: list[str] = [
        "# Paper tables — generated from run outputs",
        "",
        "Do not edit by hand. Regenerate with `uv run python -m ml.paper_tables`.",
        "",
    ]
    if main_metrics is None:
        lines += ["_Missing `baseline_metrics_common.csv`; main tables unavailable._", ""]
    else:
        lines += main_table(main_metrics)
        lines += per_class_h90(main_metrics)
    lines += attribution_table(attr_metrics)
    lines += stack_table(stack_metrics)
    lines += validation_ablation_table(valid_ablation)
    lines += validation_per_class_table(valid_ablation)
    lines += selected_test_table(selected_test, attr_metrics, main_metrics)
    lines += bootstrap_table(dm)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
