"""Chart helpers (Altair for interaction, Matplotlib for fuzzy surfaces)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.figure import Figure

from lib import fuzzy
from lib.paths import MODEL_LABELS

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
TERM_COLORS = ["#2E86AB", "#F5A623", "#C0392B"]


def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas() if df is not None and not df.is_empty() else pd.DataFrame()


def _with_date(df: pl.DataFrame) -> pl.DataFrame:
    if "date" not in df.columns:
        return df
    if df.schema.get("date") == pl.Date:
        return df
    return df.with_columns(pl.col("date").cast(pl.Utf8).str.to_date(strict=False))


def index_lines(
    df: pl.DataFrame, value_cols: list[str], height: int = 340
) -> alt.Chart:
    pdf = _to_pandas(
        _with_date(df)
        .select(["date", *value_cols])
        .melt("date", variable_name="series", value_name="value")
    )
    return (
        alt.Chart(pdf)
        .mark_line(strokeWidth=1.3)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title="log index level", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("series:N"),
                alt.Tooltip("value:Q", format=".4f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )


def fan_chart(
    base: pl.DataFrame,
    calibrated: pl.DataFrame | None,
    truth: pl.DataFrame | None,
    title: str,
    height: int = 360,
) -> alt.Chart:
    """Quantile fan for one class: area for P10-P90, line for P50."""
    b = _with_date(base).select("date", "p_0.1", "p_0.5", "p_0.9").sort("date")
    frame = b.rename({"p_0.1": "base_lo", "p_0.5": "base_mid", "p_0.9": "base_hi"})
    if calibrated is not None and not calibrated.is_empty():
        c = (
            _with_date(calibrated)
            .select("date", "p_0.1", "p_0.5", "p_0.9")
            .sort("date")
            .rename({"p_0.1": "cal_lo", "p_0.5": "cal_mid", "p_0.9": "cal_hi"})
        )
        frame = frame.join(c, on="date", how="left")
    if truth is not None and not truth.is_empty():
        t = _with_date(truth).select("date", pl.col("y").alias("realised")).sort("date")
        frame = frame.join(t, on="date", how="left")
    pdf = _to_pandas(frame)

    layers: list[alt.Chart] = []
    base_band = (
        alt.Chart(pdf)
        .mark_area(opacity=0.16, color=PALETTE[0])
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("base_lo:Q", title="log level", scale=alt.Scale(zero=False)),
            y2="base_hi:Q",
        )
    )
    base_mid = (
        alt.Chart(pdf)
        .mark_line(color=PALETTE[0], strokeWidth=1.4, strokeDash=[4, 3])
        .encode(x="date:T", y="base_mid:Q")
    )
    layers += [base_band, base_mid]
    if "cal_lo" in pdf.columns:
        cal_band = (
            alt.Chart(pdf)
            .mark_area(opacity=0.22, color=PALETTE[1])
            .encode(x="date:T", y="cal_lo:Q", y2="cal_hi:Q")
        )
        cal_mid = (
            alt.Chart(pdf)
            .mark_line(color=PALETTE[1], strokeWidth=1.8)
            .encode(x="date:T", y="cal_mid:Q")
        )
        layers += [cal_band, cal_mid]
    if "realised" in pdf.columns:
        truth_line = (
            alt.Chart(pdf)
            .mark_line(color="#222", strokeWidth=1.1)
            .encode(x="date:T", y="realised:Q")
        )
        layers.append(truth_line)
    return (
        alt.layer(*layers)
        .resolve_scale(y="shared")
        .properties(height=height, title=title)
        .interactive()
    )


def metric_bars(
    df: pl.DataFrame,
    metric: str,
    title: str,
    height: int = 320,
    models: list[str] | None = None,
) -> alt.Chart:
    if df.is_empty():
        return alt.Chart(pd.DataFrame()).mark_bar()
    d = df.filter(pl.col("scope") == "POOLED")
    if models:
        d = d.filter(pl.col("model").is_in(models))
    d = d.with_columns(pl.col("h").cast(pl.Utf8).alias("horizon"))
    pdf = _to_pandas(d)
    pdf["model_label"] = pdf["model"].map(lambda m: MODEL_LABELS.get(m, m))
    return (
        alt.Chart(pdf)
        .mark_bar()
        .encode(
            x=alt.X("model_label:N", title=None, sort="-y"),
            y=alt.Y(f"{metric}:Q", title=title),
            color=alt.Color(
                "horizon:N", title="horizon (days)", scale=alt.Scale(scheme="tableau10")
            ),
            column=alt.Column("horizon:N", title=None),
            tooltip=[
                alt.Tooltip("model_label:N"),
                alt.Tooltip("horizon:N"),
                alt.Tooltip(f"{metric}:Q", format=".4f"),
            ],
        )
        .properties(height=height)
    )


def coverage_chart(df: pl.DataFrame, h: int, height: int = 300) -> alt.Chart:
    d = df.filter((pl.col("scope") == "POOLED") & (pl.col("h") == h))
    pdf = _to_pandas(d)
    if pdf.empty:
        return alt.Chart(pdf).mark_bar()
    pdf["model_label"] = pdf["model"].map(lambda m: MODEL_LABELS.get(m, m))
    bars = (
        alt.Chart(pdf)
        .mark_bar()
        .encode(
            x=alt.X("model_label:N", title=None, sort="-y"),
            y=alt.Y(
                "cover_80:Q",
                title="central-80 coverage",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.condition(
                alt.datum.cover_80 > 0.80, alt.value("#F58518"), alt.value("#4C78A8")
            ),
            tooltip=[
                alt.Tooltip("model_label:N"),
                alt.Tooltip("cover_80:Q", format=".3f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [0.80]}))
        .mark_rule(strokeDash=[5, 4], color="#333")
        .encode(y="y:Q")
    )
    return (bars + rule).properties(
        height=height, title=f"Coverage vs nominal 0.80 (h={h})"
    )


def forest_plot(dm: pl.DataFrame, split: str, height: int = 320) -> alt.Chart:
    d = dm.filter((pl.col("split") == split) & (pl.col("quantile") == "mean"))
    pdf = _to_pandas(d)
    if pdf.empty:
        return alt.Chart(pdf).mark_point()
    pdf["h_label"] = (
        pdf["h"].astype(str)
        + "d vs "
        + pdf["competitor"].map(lambda m: MODEL_LABELS.get(m, m))
    )
    pdf["favours"] = np.where(
        pdf["dm_mean_diff"] < 0, "Sugeno better", "competitor better"
    )
    base = alt.Chart(pdf).encode(
        y=alt.Y("h_label:N", title=None, sort="-x"),
        x=alt.X(
            "dm_mean_diff:Q", title="mean pinball difference (Sugeno − competitor)"
        ),
        tooltip=[
            alt.Tooltip("h_label:N"),
            alt.Tooltip("dm_mean_diff:Q", format=".5f"),
            alt.Tooltip("dm_p:Q", format=".4f"),
            alt.Tooltip("boot_lo:Q", format=".5f"),
            alt.Tooltip("boot_hi:Q", format=".5f"),
        ],
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="#333").encode(x="x:Q")
    err = base.mark_errorbar().encode(
        x=alt.X("boot_lo:Q"), x2="boot_hi:Q", color=alt.Color("favours:N", title=None)
    )
    pts = base.mark_point(filled=True, size=70).encode(
        color=alt.Color("favours:N", legend=None)
    )
    return (rule + err + pts).properties(
        height=height, title=f"Diebold-Mariano forest ({split})"
    )


def ablation_heatmap(
    abl: pl.DataFrame, split: str = "test", height: int = 280
) -> alt.Chart:
    d = abl.filter((pl.col("split") == split) & (pl.col("scope") == "POOLED"))
    if d.is_empty():
        return alt.Chart(pd.DataFrame()).mark_rect()
    pdf = _to_pandas(d)
    return (
        alt.Chart(pdf)
        .mark_rect()
        .encode(
            x=alt.X(
                "arm:N", title=None, sort=["both", "widthonly", "loconly", "novol"]
            ),
            y=alt.Y("h:O", title="horizon"),
            color=alt.Color("pinball_mean:Q", scale=alt.Scale(scheme="tealblues")),
            tooltip=[
                alt.Tooltip("arm:N"),
                alt.Tooltip("h:O"),
                alt.Tooltip("pinball_mean:Q", format=".5f"),
            ],
        )
        .properties(height=height, title=f"Fuzzy ablation mean pinball ({split})")
    )


# ---------------------------------------------------------------------------
# Fuzzy-specific matplotlib figures
# ---------------------------------------------------------------------------


def membership_figure(spec: dict, system: str) -> Figure:
    inputs = list(spec.get("inputs") or [])
    k = max(len(inputs), 1)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 3.2), squeeze=False)
    grid = fuzzy.input_grid(spec, 0)
    for i, name in enumerate(inputs):
        ax = axes[0][i]
        curves = fuzzy.membership_curve(spec, i, grid)
        for term in range(curves.shape[1]):
            ax.plot(
                grid,
                curves[:, term],
                color=TERM_COLORS[term % len(TERM_COLORS)],
                label=fuzzy.LINGUISTIC[term]
                if term < len(fuzzy.LINGUISTIC)
                else f"MF{term}",
                linewidth=1.8,
            )
        ax.set_title(fuzzy.INPUT_LABELS.get(name, name), fontsize=10)
        ax.set_xlabel("standardised input")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel("membership")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(fuzzy.SYSTEM_LABELS.get(system, system), fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


def surface_figure(spec: dict, system: str) -> Figure:
    inputs = list(spec.get("inputs") or [])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    if not inputs:
        ax.text(0.5, 0.5, "no active inputs", ha="center")
        return fig
    if len(inputs) == 1:
        grid, z = fuzzy.surface(spec, 0)
        ax.plot(grid, z, color="#2E86AB", linewidth=2.0)
        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"standardised {fuzzy.INPUT_LABELS.get(inputs[0], inputs[0])}")
        ax.set_ylabel("defuzzified output")
    else:
        xx, yy, z = fuzzy.surface(spec, 0, 1)
        cf = ax.contourf(xx, yy, z, levels=21, cmap="RdBu_r")
        cs = ax.contour(xx, yy, z, levels=9, colors="black", linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%+.2f")
        fig.colorbar(cf, ax=ax, label="output")
        ax.set_xlabel(f"standardised {fuzzy.INPUT_LABELS.get(inputs[0], inputs[0])}")
        ax.set_ylabel(f"standardised {fuzzy.INPUT_LABELS.get(inputs[1], inputs[1])}")
    ax.set_title(fuzzy.SYSTEM_LABELS.get(system, system), fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def consequence_heatmap(spec: dict, system: str) -> Figure | None:
    cons = np.asarray(spec.get("consequents") or [], dtype=float).reshape(-1)
    if cons.size == 0:
        return None
    if len(spec.get("inputs") or []) == 2 and cons.size == 9:
        grid = cons.reshape(3, 3)
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        im = ax.imshow(
            grid, cmap="coolwarm", vmin=-np.abs(grid).max(), vmax=np.abs(grid).max()
        )
        ax.set_xticks(range(3), fuzzy.LINGUISTIC)
        ax.set_yticks(range(3), fuzzy.LINGUISTIC)
        ax.set_xlabel(fuzzy.INPUT_LABELS.get(spec["inputs"][1], spec["inputs"][1]))
        ax.set_ylabel(fuzzy.INPUT_LABELS.get(spec["inputs"][0], spec["inputs"][0]))
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=8
                )
        fig.colorbar(im, ax=ax, label="consequent")
        ax.set_title(f"Rule consequents — {system}", fontsize=10)
        fig.tight_layout()
        return fig
    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * cons.size), 2.8))
    ax.bar([f"R{i + 1}" for i in range(cons.size)], cons, color="#4C78A8")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_ylabel("consequent")
    ax.set_title(f"Rule consequents — {system}", fontsize=10)
    fig.tight_layout()
    return fig


def firing_chart(weights: np.ndarray, spec: dict) -> alt.Chart:
    idx = fuzzy.rule_index(spec)
    inputs = list(spec.get("inputs") or [])
    labels = [
        f"R{i + 1}: "
        + ", ".join(fuzzy.LINGUISTIC[m] for m in rule)
        + (f" ({', '.join(inputs)})" if inputs else "")
        for i, rule in enumerate(idx)
    ]
    df = pd.DataFrame(
        {"rule": labels, "firing": np.asarray(weights, dtype=float).reshape(-1)}
    )
    return (
        alt.Chart(df)
        .mark_bar(color="#54A24B")
        .encode(
            x=alt.X("firing:Q", title="firing strength (product t-norm)"),
            y=alt.Y("rule:N", title=None, sort="-x"),
            tooltip=[alt.Tooltip("rule:N"), alt.Tooltip("firing:Q", format=".4f")],
        )
        .properties(height=max(180, 26 * len(labels)))
    )
