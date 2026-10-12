"""Results & Significance: main table, metric charts, DM tests, ablations, robustness."""

from __future__ import annotations

import polars as pl
import streamlit as st
from lib import loaders, viz
from lib.paths import MODEL_LABELS, PAPER_MODELS


def _present_models(df: pl.DataFrame, split: str) -> list[str]:
    if df.is_empty():
        return []
    present = set(df.filter(pl.col("split") == split)["model"].unique().to_list())
    ordered = [m for m in PAPER_MODELS if m in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _winner_summary(df: pl.DataFrame, split: str) -> pl.DataFrame:
    d = df.filter(pl.col("split") == split)
    if d.is_empty():
        return d
    idx = d.group_by(["h", "scope"]).agg(pl.col("pinball_mean").min().alias("best"))
    best = d.join(idx, on=["h", "scope"], how="inner").filter(
        pl.col("pinball_mean") == pl.col("best")
    )
    return (
        best.group_by(["h", "scope"])
        .agg(pl.col("model").sort().str.join(", ").alias("lowest pinball"))
        .sort(["h", "scope"])
    )


def render() -> None:
    st.title("Results & Significance")
    st.caption(
        "Test rows below are the frozen, one-shot evaluation. Validation rows are in-sample "
        "for the fuzzy and conformal layers, so only test is a clean comparison."
    )

    common = loaders.load_metrics_common()
    full = loaders.load_metrics()
    if common.is_empty() and full.is_empty():
        st.error("No baseline metrics found — run the baselines step first.")
        return

    c1, c2 = st.columns([1, 3])
    split = c1.radio("Split", ["test", "valid"], index=0, horizontal=True)
    models = _present_models(common if not common.is_empty() else full, split)
    picked = c2.multiselect(
        "Models",
        models,
        default=[m for m in ("lstm", "lstm_fuzzy", "cqr", "aci") if m in models],
        format_func=lambda m: MODEL_LABELS.get(m, m),
    )
    if not picked:
        picked = models

    src = common if not common.is_empty() else full
    table = src.filter((pl.col("split") == split) & (pl.col("scope") == "POOLED"))

    st.subheader("Common-support main table")
    show = (
        table.filter(pl.col("model").is_in(picked))
        .select(
            "h",
            pl.col("model").replace_strict({m: MODEL_LABELS.get(m, m) for m in picked}),
            "n",
            pl.col("pinball_mean").round(5),
            pl.col("cover_80").round(4),
            pl.col("winkler_80").round(4),
            pl.col("dir_hit").round(3),
        )
        .sort(["h", "model"])
    )
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(
        "All models in this table are scored on exactly the same rows; "
        "lower pinball is better."
    )

    st.divider()
    st.subheader("Metric charts")
    t1, t2, t3 = st.tabs(
        ["Mean pinball by model × horizon", "Coverage vs nominal", "Winkler-80"]
    )
    with t1:
        st.altair_chart(
            viz.metric_bars(
                table.filter(pl.col("model").is_in(picked)),
                "pinball_mean",
                "mean pinball",
            ),
            width="stretch",
        )
    with t2:
        h_cov = st.select_slider("Horizon", options=[1, 7, 30, 90], value=90)
        st.altair_chart(
            viz.coverage_chart(table.filter(pl.col("model").is_in(picked)), h_cov),
            width="stretch",
        )
    with t3:
        st.altair_chart(
            viz.metric_bars(
                table.filter(pl.col("model").is_in(picked)), "winkler_80", "Winkler-80"
            ),
            width="stretch",
        )

    st.divider()
    st.subheader("Where each method wins")
    wins = _winner_summary(
        src.filter(
            pl.col("scope").is_in(
                [
                    "POOLED",
                    *[c for c in ["Capesize", "Panamax", "Supramax", "Handysize"]],
                ]
            )
        ),
        split,
    )
    if not wins.is_empty():
        st.dataframe(wins, hide_index=True, width="stretch")
    st.caption(
        "Ties are listed together; the model with the lowest mean pinball wins "
        "at each horizon and scope."
    )

    st.divider()
    st.subheader("Diebold–Mariano tests vs the Sugeno layer")
    dm = loaders.load_dm()
    if dm.is_empty():
        st.info("baseline_dm.csv not found.")
    else:
        q = st.radio(
            "Quantile",
            sorted(dm["quantile"].unique().to_list()),
            index=0,
            horizontal=True,
        )
        sub = dm.filter((pl.col("split") == split) & (pl.col("quantile") == q))
        st.altair_chart(viz.forest_plot(dm, split), width="stretch")
        st.dataframe(
            sub.select(
                "h",
                pl.col("competitor").replace_strict(
                    {
                        m: MODEL_LABELS.get(m, m)
                        for m in sub["competitor"].unique().to_list()
                    }
                ),
                pl.col("dm_mean_diff").round(5),
                pl.col("dm_stat").round(3),
                pl.col("dm_p").round(4),
                pl.col("boot_lo").round(5),
                pl.col("boot_hi").round(5),
                "dm_n",
            ).sort(["h", "competitor"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Negative mean difference favours Sugeno. The DM p-value uses a Newey–West HAC "
            "variance at lag h−1; the block-bootstrap interval is the primary evidence because "
            "the four classes are strongly correlated. Twelve comparisons are made without "
            "multiplicity correction, so the h = 90 CQR result is suggestive, not conclusive."
        )

    st.divider()
    st.subheader("h = 90 per-class breakdown")
    pc = src.filter(
        (pl.col("split") == split) & (pl.col("h") == 90) & (pl.col("scope") != "POOLED")
    )
    if pc.is_empty():
        st.info("No per-class rows.")
    else:
        st.dataframe(
            pc.filter(pl.col("model").is_in(picked))
            .select(
                "scope",
                pl.col("model"),
                "n",
                pl.col("pinball_mean").round(5),
                pl.col("cover_80").round(3),
                pl.col("winkler_80").round(4),
            )
            .sort(["scope", "model"]),
            hide_index=True,
            width="stretch",
        )

    st.divider()
    st.subheader("Exploratory ablation")
    abl = loaders.load_ablations()
    if abl.is_empty():
        st.info("fuzzy_ablations.csv not found.")
    else:
        a_split = st.radio(
            "Ablation split",
            ["test", "valid"],
            index=0,
            horizontal=True,
            key="abl_split",
        )
        st.altair_chart(viz.ablation_heatmap(abl, a_split), width="stretch")
        st.dataframe(
            abl.filter((pl.col("split") == a_split) & (pl.col("scope") == "POOLED"))
            .pivot(on="arm", index="h", values="pinball_mean")
            .sort("h"),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Arms: both = full layer, widthonly, loconly, novol. The location system alone is "
            "worse than base at h = 1, 7 and 30; at h = 90 both components help."
        )

    st.divider()
    st.subheader("Data robustness: 7-day congestion lag rebuild")
    rob = loaders.load_robustness()
    if rob.is_empty():
        st.info("robustness_metrics.csv not found.")
    else:
        st.dataframe(
            rob.select(
                "h",
                pl.col("model").replace_strict(
                    {m: MODEL_LABELS.get(m, m) for m in rob["model"].unique().to_list()}
                ),
                pl.col("pinball_mean").round(5).alias("main pinball"),
                pl.col("pinball_mean_lag7").round(5).alias("lag-7 pinball"),
                pl.col("d_pinball_mean").round(5).alias("Δ pinball"),
                pl.col("d_cover_80").round(4).alias("Δ cover-80"),
                pl.col("d_width").round(4).alias("Δ width"),
            ).sort(["h", "model"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Deltas are rebuild − main on the validation split. Small deltas mean the conclusions are not an artefact of the congestion timing."
        )
