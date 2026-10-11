"""Data Explorer: index levels, correlations, split ranges, feature drill-down."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from lib import loaders, viz
from lib.paths import CLASSES


def _split_ranges() -> dict[str, tuple[str, str]]:
    man = loaders.load_manifest()
    out: dict[str, tuple[str, str]] = {}
    if man.is_empty():
        return out
    for row in man.iter_rows(named=True):
        out[row["split"]] = (row["start_date"], row["end_date"])
    return out


def render() -> None:
    st.title("Data Explorer")
    st.caption(
        "Continuous 365-day master calendar with last-observation-carried-forward; "
        "weekends and holidays exist as rows but are dropped from training/evaluation."
    )

    levels = loaders.load_class_levels(log=True)
    if levels.is_empty():
        st.error("master_long.parquet not found — run the data-prep pipeline first.")
        return

    dates = levels["date"]
    lo, hi = dates.min(), dates.max()

    top = st.columns([2, 1, 1, 2])
    with top[0]:
        window = st.date_input(
            "Date range",
            value=(max(lo, hi.replace(year=hi.year - 5)), hi),
            min_value=lo,
            max_value=hi,
        )
    with top[1]:
        log_space = st.toggle(
            "Log space", value=True, help="Log levels are the modelling space."
        )
    with top[2]:
        show_bdi = st.toggle(
            "Show BDI", value=True, help="BDI is a covariate, never a target."
        )
    with top[3]:
        series = list(CLASSES) + (["BDI"] if show_bdi else [])
        chosen = st.multiselect("Series", series, default=series)

    if isinstance(window, (tuple, list)) and len(window) == 2:
        start, end = window
    else:
        start, end = lo, hi

    view = levels.filter((pl.col("date") >= start) & (pl.col("date") <= end)).sort(
        "date"
    )
    if not log_space:
        view = view.with_columns(
            [
                pl.when(pl.col(c).is_not_null())
                .then(pl.col(c).exp())
                .otherwise(None)
                .alias(c)
                for c in CLASSES + ["BDI"]
            ]
        )
    cols = [c for c in view.columns if c != "date" and c in chosen]
    if cols:
        st.altair_chart(viz.index_lines(view, cols, height=380), width="stretch")
    else:
        st.info("Pick at least one series.")

    st.divider()
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Daily log-return correlations")
        rets = (
            levels.select(["date", *CLASSES])
            .sort("date")
            .with_columns([pl.col(c).diff().alias(c) for c in CLASSES])
            .drop_nulls()
        )
        mat = rets.select(CLASSES).to_numpy()
        corr = np.corrcoef(mat.T)
        cdf = pd.DataFrame(corr, index=CLASSES, columns=CLASSES).reset_index(
            names="row"
        )
        long = cdf.melt(id_vars="row", var_name="col", value_name="corr")
        chart = (
            alt.Chart(long)
            .mark_rect()
            .encode(
                x=alt.X("col:N", title=None, sort=list(CLASSES)),
                y=alt.Y("row:N", title=None, sort=list(CLASSES)),
                color=alt.Color(
                    "corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])
                ),
                tooltip=[
                    alt.Tooltip("row:N"),
                    alt.Tooltip("col:N"),
                    alt.Tooltip("corr:Q", format=".3f"),
                ],
            )
            .properties(height=300, title="Four vessel classes move together")
        )
        text = chart.mark_text().encode(
            text=alt.Text("corr:Q", format=".2f"),
            color=alt.condition(
                abs(alt.datum.corr) > 0.6, alt.value("white"), alt.value("black")
            ),
        )
        st.altair_chart(chart + text, width="stretch")
        st.caption(
            "Strong cross-class correlation is why the paper's Diebold–Mariano effective "
            "sample size is far below the row count."
        )

    with right:
        st.subheader("Split ranges")
        ranges = _split_ranges()
        if ranges:
            rows = [{"split": k, "start": v[0], "end": v[1]} for k, v in ranges.items()]
            st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("samples_manifest.csv not found.")
        st.markdown(
            """
- **Train** fits the base model; the fuzzy/conformal layers are fit on **valid**.
- **Test** is scored once, after configuration is frozen.
- 44 Capesize train rows with a negative index value (undefined log) are removed.
- Evaluation masks each horizon to real market days where `y(t+h)` is observed.
"""
        )

    st.divider()
    st.subheader("Feature drill-down (validation split)")
    feat = loaders.load_sample_features("valid")
    if feat.is_empty():
        st.info("samples_valid.parquet not found.")
        return
    numeric = [
        c
        for c, t in zip(feat.columns, feat.dtypes)
        if c not in ("date",) and t in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    default = "ret_1" if "ret_1" in numeric else numeric[0]
    c1, c2 = st.columns([2, 1])
    with c1:
        feature = st.selectbox("Feature", numeric, index=numeric.index(default))
    with c2:
        cls = st.selectbox("Class", CLASSES, index=0)

    sub = (
        feat.filter(pl.col("target_class") == cls)
        .select("date", feature)
        .drop_nulls()
        .pipe(
            lambda d: d.with_columns(
                pl.col("date").cast(pl.Utf8).str.to_date(strict=False)
            )
        )
        .sort("date")
    )
    if sub.is_empty():
        st.info("No rows for this class/feature.")
        return
    null_share = 1.0 - sub.height / feat.filter(pl.col("target_class") == cls).height
    pdf = sub.to_pandas()
    line = (
        alt.Chart(pdf)
        .mark_line(color="#4C78A8", strokeWidth=1.2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y(f"{feature}:Q", title=feature, scale=alt.Scale(zero=False)),
        )
        .properties(height=260, title=f"{feature} — {cls}")
        .interactive()
    )
    hist = (
        alt.Chart(pdf)
        .mark_bar(color="#F58518", opacity=0.85)
        .encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=40), title=feature),
            y=alt.Y("count():Q"),
        )
        .properties(height=260, title="distribution")
    )
    st.altair_chart(line, width="stretch")
    st.altair_chart(hist, width="stretch")
    st.caption(
        f"n = {sub.height} · null share after drop = {null_share:.3%} · modelling uses forward-filled calendar rows."
    )
