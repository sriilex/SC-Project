"""Base Model page: calibration diagnostics and the quantile fan."""

from __future__ import annotations

import altair as alt
import matplotlib.pyplot as plt
import polars as pl
import streamlit as st
from lib import loaders, metrics, viz
from lib.paths import CLASSES, HORIZONS


def _coverage_table() -> pl.DataFrame:
    diag = loaders.load_diagnostics()
    if diag.is_empty():
        return diag
    d = diag.filter((pl.col("model") == "lstm") & (pl.col("scope") == "POOLED"))
    return (
        d.pivot(on="h", index="split", values="cover_80")
        .rename({"split": "Split"})
        .with_columns(pl.col("Split").str.to_titlecase())
    )


def _crossing_table() -> pl.DataFrame:
    diag = loaders.load_diagnostics()
    if diag.is_empty():
        return diag
    d = diag.filter((pl.col("model") == "lstm") & (pl.col("scope") == "POOLED"))
    return d.select(
        pl.col("split").alias("Split"),
        pl.col("h"),
        pl.col("cover_80").alias("cover-80"),
        pl.col("width").alias("width"),
        pl.col("x_p10_gt_p50").alias("P10>P50"),
        pl.col("x_p10_gt_p90").alias("P10>P90"),
        pl.col("pinball_0.1").alias("pin 0.1"),
        pl.col("pinball_0.5").alias("pin 0.5"),
        pl.col("pinball_0.9").alias("pin 0.9"),
    ).sort(["Split", "h"])


def render() -> None:
    st.title("Base Model Calibration")
    st.caption(
        "The object being calibrated: a five-seed monotone quantile LSTM. "
        "The head is monotone by construction, so raw quantiles can never cross."
    )

    cov = _coverage_table()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Central-80 coverage by split")
        if cov.is_empty():
            st.info("interval_diagnostics.csv not found.")
        else:
            st.dataframe(cov, hide_index=True, width="stretch")
            st.caption(
                "Training coverage stays inside the pre-registered 0.70–0.90 gate; "
                "validation coverage rises with the horizon and overshoots nominal at h = 90, "
                "which is exactly the regime the fuzzy width system contracts."
            )
    with right:
        st.subheader("Monotone head")
        st.latex(
            r"q_{0.5}=o_1,\quad q_{0.1}=q_{0.5}-\mathrm{softplus}(o_2),\quad "
            r"q_{0.9}=q_{0.5}+\mathrm{softplus}(o_3)"
        )
        st.markdown(
            """
The earlier unconstrained head produced `P10 > P50` in up to **66.5%** of rows at
h = 1 and **57.5%** at h = 7, with central-80 coverage of only 18–28%. The monotone
head plus exact pinball loss removed every crossing.
"""
        )

    st.divider()
    st.subheader("Quantile fan")
    c1, c2, c3, c4 = st.columns(4)
    split = c1.selectbox("Split", ["valid", "train"], index=0)
    h = c2.selectbox("Horizon (days)", HORIZONS, index=3)
    cls = c3.selectbox("Vessel class", CLASSES, index=0)
    show_seeds = c4.toggle("Overlay seed spread", value=False)

    base = loaders.load_base_pred(split, h)
    truth = loaders.truth_frame(split, h)
    if base.is_empty() or truth.is_empty():
        st.info("Cached predictions not found for this split/horizon.")
        return
    b = base.filter(pl.col("target_class") == cls).sort("date")
    t = truth.filter(pl.col("target_class") == cls).sort("date")

    if show_seeds:
        layers = []
        for seed in loaders.available_seeds(h):
            s = loaders.load_base_pred(split, h, seed).filter(
                pl.col("target_class") == cls
            )
            s = s.with_columns(pl.lit(f"seed {seed}").alias("series"))
            layers.append(s.select("date", "p_0.5", "series"))
        if layers:
            seeds = pl.concat(layers)
            st.altair_chart(_seed_chart(seeds, b, t, cls, h), width="stretch")

    st.altair_chart(
        viz.fan_chart(
            b, None, t.select("date", "y"), title=f"Base fan — {cls}, h={h} ({split})"
        ),
        width="stretch",
    )

    st.divider()
    a, bb = st.columns(2, gap="large")
    with a:
        st.subheader("Loss by quantile")
        pb = metrics.pinball_by_quantile(b, t)
        if not pb.is_empty():
            pdf = pb.to_pandas()
            pdf["quantile"] = pdf["quantile"].astype(str)
            st.altair_chart(
                alt.Chart(pdf)
                .mark_bar(color="#4C78A8")
                .encode(
                    x=alt.X("quantile:N", title="τ"),
                    y=alt.Y(
                        "pinball:Q", title="mean pinball", scale=alt.Scale(zero=False)
                    ),
                    tooltip=[
                        alt.Tooltip("quantile:N"),
                        alt.Tooltip("pinball:Q", format=".5f"),
                    ],
                )
                .properties(height=260),
                width="stretch",
            )
        band = metrics.calibration_band(b, t)
        if band:
            st.write(
                {
                    "n": band["n"],
                    "coverage": round(band["cover_80"], 3),
                    "below P10": round(band["below_p10"], 3),
                    "above P90": round(band["above_p90"], 3),
                    "mean width": round(band["mean_width"], 4),
                }
            )
    with bb:
        st.subheader("Residual diagnostics")
        j = t.join(b, on=["date", "target_class"], how="inner").drop_nulls("y")
        if not j.is_empty():
            resid = (j["y"] - j["p_0.5"]).to_numpy()
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            ax.hist(resid, bins=45, color="#72B7B2", edgecolor="white")
            ax.axvline(0, color="#C0392B", linewidth=1.2, linestyle="--")
            ax.set_xlabel("y − P50 (log units)")
            ax.set_ylabel("count")
            ax.set_title(f"Median residual — {cls}, h={h}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"mean residual = {resid.mean():+.4f} · sd = {resid.std():.4f}. "
                "The location system consumes exactly the rolling standardised version of this signal."
            )

    st.divider()
    st.subheader("Crossing and interval diagnostics")
    ct = _crossing_table()
    if ct.is_empty():
        st.info("interval_diagnostics.csv not found.")
    else:
        st.dataframe(ct, hide_index=True, width="stretch")
        st.caption(
            "Zero crossings at every horizon and split is the gate the monotone head was built to pass."
        )


def _seed_chart(
    seeds: pl.DataFrame, base: pl.DataFrame, truth: pl.DataFrame, cls: str, h: int
) -> alt.Chart:
    pdf = seeds.with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False)
    ).to_pandas()
    ensemble = (
        base.select("date", pl.col("p_0.5").alias("ensemble"))
        .with_columns(pl.col("date").cast(pl.Utf8).str.to_date(strict=False))
        .to_pandas()
    )
    real = (
        truth.select("date", pl.col("y").alias("realised"))
        .with_columns(pl.col("date").cast(pl.Utf8).str.to_date(strict=False))
        .to_pandas()
    )
    lines = (
        alt.Chart(pdf)
        .mark_line(opacity=0.55, strokeWidth=1.0)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("p_0.5:Q", title="P50 (log level)", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title="seed"),
        )
        .properties(height=340, title=f"Seed spread vs ensemble mean — {cls}, h={h}")
    )
    ens = (
        alt.Chart(ensemble)
        .mark_line(color="#111", strokeWidth=2.0)
        .encode(x="date:T", y="ensemble:Q")
    )
    tr = (
        alt.Chart(real)
        .mark_line(color="#C0392B", strokeWidth=1.0, strokeDash=[3, 2])
        .encode(x="date:T", y="realised:Q")
    )
    return (lines + ens + tr).interactive()
