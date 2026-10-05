"""Generate the fuzzy-rule diagrams embedded in 04_fuzzy_lstm_report.md.

Run from the repository root:
    uv run python docs/figures/fuzzy_figures.py

Outputs three PNGs next to this script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.fuzzy_calibration import (
    _mpe_negative,
    _mpe_neutral,
    _mpe_positive,
    _vol_high,
    _vol_low,
    _vol_medium,
    sugeno_shift,
)

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def membership_figure() -> None:
    e = np.linspace(-0.20, 0.20, 800)
    v = np.linspace(0.0, 0.20, 800)

    fig, (ax_e, ax_v) = plt.subplots(1, 2, figsize=(11.0, 3.8))

    for fn, label, colour in (
        (_mpe_negative, "negative (under-forecast)", "#c0392b"),
        (_mpe_neutral, "neutral", "#7f8c8d"),
        (_mpe_positive, "positive (over-forecast)", "#2471a3"),
    ):
        ax_e.plot(e, np.vectorize(fn)(e), color=colour, label=label)
    ax_e.set_title("mean_pct_error membership")
    ax_e.set_xlabel("mean_pct_error")
    ax_e.set_ylabel("degree of membership")
    ax_e.set_ylim(-0.02, 1.05)
    ax_e.legend(loc="upper right", fontsize=7)

    for fn, label, colour in (
        (_vol_low, "low", "#27ae60"),
        (_vol_medium, "medium", "#f39c12"),
        (_vol_high, "high", "#8e44ad"),
    ):
        ax_v.plot(v, np.vectorize(fn)(v), color=colour, label=label)
    ax_v.set_title("recent_volatility membership")
    ax_v.set_xlabel("recent_volatility")
    ax_v.set_ylabel("degree of membership")
    ax_v.set_ylim(-0.02, 1.05)
    ax_v.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fuzzy_membership.png", dpi=160)
    plt.close(fig)


def rule_matrix_figure() -> None:
    consequents = np.array(
        [
            [0.030, 0.050, 0.080],
            [0.000, 0.000, 0.010],
            [-0.030, -0.050, -0.080],
        ]
    )
    rule_ids = [["R1", "R2", "R3"], ["R4", "R5", "R6"], ["R7", "R8", "R9"]]
    bias_labels = ["negative", "neutral", "positive"]
    vol_labels = ["low", "medium", "high"]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(consequents, cmap="coolwarm", vmin=-0.08, vmax=0.08)

    ax.set_xticks(range(3), vol_labels)
    ax.set_yticks(range(3), bias_labels)
    ax.set_xlabel("recent_volatility")
    ax.set_ylabel("mean_pct_error")
    ax.set_title("Sugeno rule consequents (log-space shift)")

    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, f"{rule_ids[i][j]}\n{consequents[i, j]:+.3f}",
                ha="center", va="center", color="black", fontsize=9,
            )

    fig.colorbar(im, ax=ax, label="shift", shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "fuzzy_rule_matrix.png", dpi=160)
    plt.close(fig)


def surface_figure() -> None:
    e = np.linspace(-0.20, 0.20, 161)
    v = np.linspace(0.0, 0.20, 161)
    E, V = np.meshgrid(e, v)
    S = np.vectorize(sugeno_shift)(E, V)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    cf = ax.contourf(E, V, S, levels=21, cmap="coolwarm", vmin=-0.08, vmax=0.08)
    cs = ax.contour(E, V, S, levels=[-0.06, -0.03, 0.0, 0.03, 0.06], colors="black", linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%+.2f")
    ax.set_xlabel("mean_pct_error")
    ax.set_ylabel("recent_volatility")
    ax.set_title("Defuzzified shift surface")
    fig.colorbar(cf, ax=ax, label="shift", shrink=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fuzzy_rule_surface.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    membership_figure()
    rule_matrix_figure()
    surface_figure()
    print(f"wrote 3 figures to {OUT}")
