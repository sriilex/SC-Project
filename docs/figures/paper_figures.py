"""Generate the result charts embedded in docs/paper.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "src" / "data" / "runs" / "v2"
DATA = ROOT / "src" / "data"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

VALID_ARMS = [
    ("base", "Base"),
    ("const_wl", "Constant W+L"),
    ("linear_state", "Linear state"),
    ("sugeno0", "Sugeno zero"),
    ("sugeno_lin_split", "Sugeno selected"),
]
VALID_COLORS = ["#6c757d", "#f0ad4e", "#5bc0de", "#9b59b6", "#2e8b57"]
SIMPLE_LABELS = {
    "attr_constant_width": "Constant width",
    "attr_constant_width_location": "Constant W+L",
    "attr_constant_width_rollbias": "Constant W+rolling bias",
    "attr_linear": "Linear state",
}


def _row(frame: pl.DataFrame, **filters: object) -> dict[str, object]:
    """Return one row as a dictionary after applying equality filters."""
    out = frame
    for column, value in filters.items():
        out = out.filter(pl.col(column) == value)
    return out.row(0, named=True)


def validation_pinball() -> None:
    """Plot pooled validation pinball by horizon for the main arms."""
    d = pl.read_csv(RUN / "calibration_ablation_valid.csv").filter(pl.col("scope") == "POOLED")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2))
    for ax, h in zip(axes.ravel(), (1, 7, 30, 90)):
        values = [float(_row(d, h=h, arm=arm, scope="POOLED")["pinball_mean"]) for arm, _ in VALID_ARMS]
        labels = [label for _, label in VALID_ARMS]
        ax.bar(np.arange(len(values)), values, color=VALID_COLORS)
        ax.set_title(f"$h={h}$")
        ax.set_xticks(np.arange(len(values)), labels, rotation=18, ha="right")
        ax.set_ylabel("mean pinball")
        ax.set_ylim(0, max(values) * 1.18)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.5f}", ha="center", va="bottom", fontsize=6)
    fig.suptitle("Validation mean pinball loss (lower is better)")
    fig.tight_layout()
    fig.savefig(OUT / "paper_valid_pinball.png", dpi=220)
    plt.close(fig)


def test_pinball() -> None:
    """Plot pooled test pinball for base, selected Sugeno, and best simple arm."""
    base = pl.read_csv(DATA / "baseline_metrics_common.csv").filter(
        (pl.col("split") == "test") & (pl.col("scope") == "POOLED") & (pl.col("model") == "lstm")
    )
    selected = pl.read_csv(RUN / "calibration_test_selected.csv").filter(
        (pl.col("split") == "test") & (pl.col("scope") == "POOLED")
    )
    simple = pl.read_csv(DATA / "attribution_metrics.csv").filter(
        (pl.col("split") == "test") & (pl.col("scope") == "POOLED")
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2))
    for ax, h in zip(axes.ravel(), (1, 7, 30, 90)):
        best = simple.filter(pl.col("h") == h).sort("pinball_mean").row(0, named=True)
        values = [
            float(_row(base, h=h, model="lstm")["pinball_mean"]),
            float(_row(selected, h=h)["pinball_mean"]),
            float(best["pinball_mean"]),
        ]
        labels = ["Base", "Sugeno selected", SIMPLE_LABELS[str(best["model"])]]
        ax.bar(np.arange(3), values, color=["#6c757d", "#2e8b57", "#f0ad4e"])
        ax.set_title(f"$h={h}$")
        ax.set_xticks(np.arange(3), labels, rotation=12, ha="right")
        ax.set_ylabel("mean pinball")
        ax.set_ylim(0, max(values) * 1.18)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.5f}", ha="center", va="bottom", fontsize=6.5)
    fig.suptitle("Test mean pinball loss (lower is better)")
    fig.tight_layout()
    fig.savefig(OUT / "paper_test_pinball.png", dpi=220)
    plt.close(fig)


def test_coverage() -> None:
    """Plot pooled test coverage against the nominal 80 percent level."""
    base = pl.read_csv(DATA / "baseline_metrics_common.csv").filter(
        (pl.col("split") == "test") & (pl.col("scope") == "POOLED") & (pl.col("model") == "lstm")
    )
    selected = pl.read_csv(RUN / "calibration_test_selected.csv").filter(
        (pl.col("split") == "test") & (pl.col("scope") == "POOLED")
    )
    horizons = [1, 7, 30, 90]
    base_cov = [float(_row(base, h=h, model="lstm")["cover_80"]) for h in horizons]
    sel_cov = [float(_row(selected, h=h)["cover_80"]) for h in horizons]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(horizons, base_cov, marker="o", color="#6c757d", label="Base")
    ax.plot(horizons, sel_cov, marker="o", color="#2e8b57", label="Sugeno selected")
    ax.axhline(0.80, color="#c0392b", linestyle="--", linewidth=1, label="Nominal 0.80")
    ax.set_xticks(horizons, [f"$h={h}$" for h in horizons])
    ax.set_ylim(0.65, 1.02)
    ax.set_ylabel("coverage of the 80 percent interval")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "paper_test_coverage.png", dpi=220)
    plt.close(fig)


def h90_class() -> None:
    """Plot base and selected pinball by vessel class at the 90 day horizon."""
    base = pl.read_csv(DATA / "baseline_metrics_common.csv").filter(
        (pl.col("split") == "test") & (pl.col("h") == 90) & (pl.col("model") == "lstm") & (pl.col("scope") != "POOLED")
    )
    selected = pl.read_csv(RUN / "calibration_test_selected.csv").filter(
        (pl.col("split") == "test") & (pl.col("h") == 90) & (pl.col("scope") != "POOLED")
    )
    order = ["Capesize", "Panamax", "Supramax", "Handysize"]
    base_vals = [float(_row(base, scope=cls)["pinball_mean"]) for cls in order]
    sel_vals = [float(_row(selected, scope=cls)["pinball_mean"]) for cls in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - 0.19, base_vals, width=0.36, color="#6c757d", label="Base")
    ax.bar(x + 0.19, sel_vals, width=0.36, color="#2e8b57", label="Sugeno selected")
    ax.set_xticks(x, order)
    ax.set_ylabel("mean pinball")
    ax.set_ylim(0, max(base_vals + sel_vals) * 1.18)
    ax.set_title("$h=90$ test pinball by vessel class")
    ax.legend()
    for i, (b, s) in enumerate(zip(base_vals, sel_vals)):
        ax.text(i - 0.19, b, f"{b:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(i + 0.19, s, f"{s:.3f}", ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "paper_h90_class.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    validation_pinball()
    test_pinball()
    test_coverage()
    h90_class()
    print(f"wrote 4 figures to {OUT}")
