"""Playground: live quantile-LSTM inference, a manual Sugeno calculator, and per-class comparison."""

from __future__ import annotations

import altair as alt
import numpy as np
import polars as pl
import streamlit as st
from lib import fuzzy, loaders, metrics, viz
from lib.paths import CLASSES, HORIZONS

from ml.conformal import aci_adjust, apply_cqr, fit_cqr
from ml.fuzzy_calibration import calibrate as fuzzy_calibrate


@st.cache_resource(show_spinner=False)
def _predictor(h: int):
    from ml.inference import FreightPredictor

    return FreightPredictor(h)


@st.cache_data(show_spinner=False)
def _methods(h: int, cls: str):
    base = loaders.load_base_pred("valid", h)
    context = loaders.build_history(h, include_test=False)
    if base.is_empty() or context.is_empty():
        return None
    frozen = loaders.load_fuzzy_params(h)
    sug = fuzzy_calibrate(base, context, h, frozen) if frozen else base
    from ml.baselines import _context_frame

    preds = {"train": {h: loaders.load_base_pred("train", h)}, "valid": {h: base}}
    splits = {s: loaders.load_split_cached(s) for s in ("train", "valid")}
    calib = _context_frame(preds, splits, ["valid"], h)
    cqr = apply_cqr(base, fit_cqr(calib))
    aci, _ = aci_adjust(base, loaders.load_split_cached("valid"), h)

    def _filt(d: pl.DataFrame) -> pl.DataFrame:
        return d.filter(pl.col("target_class") == cls).sort("date")

    return {
        "base": _filt(base),
        "cqr": _filt(cqr),
        "aci": _filt(aci),
        "sugeno": _filt(sug),
    }


def _live_tab() -> None:
    st.caption(
        "Runs the frozen quantile LSTM live on cached validation features and "
        "overlays its central-80 fan for the selected date and class."
    )
    c1, c2, c3, c4 = st.columns(4)
    h = c1.selectbox("Horizon", HORIZONS, index=1, key="pg_h")
    cls = c2.selectbox("Class", CLASSES, index=0, key="pg_cls")
    window = c3.select_slider(
        "Lookback window (days)", options=[30, 60, 120, 240], value=60
    )
    feat = loaders.load_sample_features("valid")
    if feat.is_empty():
        st.info("samples_valid.parquet not found.")
        return
    cls_rows = feat.filter(pl.col("target_class") == cls).sort("date")
    dates = cls_rows["date"].to_list()
    picked = c4.selectbox("End date", dates, index=len(dates) - 1, key="pg_date")
    picked_s = str(picked)
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    start_s = (_date.fromisoformat(picked_s[:10]) - _timedelta(days=window)).isoformat()
    window_df = cls_rows.filter(
        (pl.col("date") <= picked_s) & (pl.col("date") >= start_s)
    )

    try:
        predictor = _predictor(h)
        lstm_pred = predictor.predict(window_df)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        st.error(f"LSTM inference failed: {exc}")
        return

    lstm = (
        loaders.load_base_pred("valid", h)
        .filter(pl.col("target_class") == cls)
        .join(window_df.select("date"), on="date", how="inner")
        .sort("date")
    )
    truth = loaders.truth_frame("valid", h).filter(pl.col("target_class") == cls)
    st.altair_chart(
        viz.fan_chart(
            lstm_pred,
            lstm,
            truth.select("date", "y"),
            title=f"LSTM central-80 (band) vs LSTM median (line) — {cls}, h={h}",
        ),
        width="stretch",
    )
    st.dataframe(lstm_pred.tail(12), hide_index=True, width="stretch")


def _calculator_tab() -> None:
    st.caption(
        "A transparent version of the layer: move the three causal inputs and the base fan, "
        "and read the memberships, the firing rules and the calibrated quantiles."
    )
    c1, c2 = st.columns([1, 3])
    h = c1.selectbox("Horizon", HORIZONS, index=3, key="calc_h")
    params = loaders.load_fuzzy_params(h)
    if not params:
        st.info(f"No learned parameters for h={h}.")
        return
    st.caption("Base fan (log level). Defaults are a small ±0.1 fan around 7.0.")
    q1, q2, q3 = st.columns(3)
    p10 = q1.number_input("P10", value=6.9, step=0.01, format="%.2f")
    p50 = q2.number_input("P50", value=7.0, step=0.01, format="%.2f")
    p90 = q3.number_input("P90", value=7.1, step=0.01, format="%.2f")
    if not (p10 <= p50 <= p90):
        st.warning("Require P10 ≤ P50 ≤ P90.")

    x1, x2, x3 = st.columns(3)
    x_raw = {
        "vol_z": x1.slider("volatility (z)", -3.0, 3.0, 0.5, 0.05),
        "cov_gap": x2.slider("coverage gap", -0.5, 0.5, 0.0, 0.01),
        "bias_z": x3.slider("median bias (z)", -3.0, 3.0, 0.0, 0.05),
    }

    m = 0.0
    s = 0.0
    lcol, rcol = st.columns(2)
    for system, col in (("width", lcol), ("loc", rcol)):
        spec = params.get(system, {})
        cols = list(spec.get("inputs") or [])
        col.markdown(f"**{fuzzy.SYSTEM_LABELS.get(system, system)}**")
        if not cols:
            continue
        x = np.array([[x_raw[c] for c in cols]])
        mu = fuzzy.memberships(spec, x)[0]
        table = [
            {
                "input": fuzzy.INPUT_LABELS.get(iname, iname),
                **{
                    fuzzy.LINGUISTIC[k]: round(float(mu[i, k]), 4)
                    for k in range(mu.shape[1])
                },
            }
            for i, iname in enumerate(cols)
        ]
        col.dataframe(pl.DataFrame(table), hide_index=True, width="stretch")
        col.altair_chart(
            viz.firing_chart(fuzzy.rule_weights(spec, x)[0], spec),
            width="stretch",
        )
        out = float(fuzzy.evaluate(spec, x)[0])
        col.metric("m" if system == "width" else "s", f"{out:+.4f}")
        if system == "width":
            m = out
        else:
            s = out

    p50_new = p50 + s
    mult = float(np.exp(m))
    rows = [
        {"quantile": "P10", "base": p10, "calibrated": p50_new - (p50 - p10) * mult},
        {"quantile": "P50", "base": p50, "calibrated": p50_new},
        {"quantile": "P90", "base": p90, "calibrated": p50_new + (p90 - p50) * mult},
    ]
    st.dataframe(
        pl.DataFrame(rows).with_columns(
            pl.col("base").round(4), pl.col("calibrated").round(4)
        ),
        hide_index=True,
        width="stretch",
    )
    st.write(
        {
            "exp(m)": round(mult, 4),
            "width change": f"{(mult - 1) * 100:+.1f}%",
            "shift s": round(s, 4),
        }
    )
    st.caption(
        "Ordering is preserved: the P10 and P90 are scaled about the shifted centre by the same positive factor."
    )


def _compare_tab() -> None:
    st.caption("Per-class comparison of the four methods on the validation split.")
    c1, c2 = st.columns(2)
    h = c1.selectbox("Horizon", HORIZONS, index=3, key="cmp_h")
    cls = c2.selectbox("Class", CLASSES, index=0, key="cmp_cls")
    methods = _methods(h, cls)
    if methods is None:
        st.info("Prediction cache missing.")
        return
    truth = loaders.truth_frame("valid", h).filter(pl.col("target_class") == cls)
    rows = []
    for name, pred in methods.items():
        p = metrics.pooled(pred, truth, name)
        if p:
            rows.append(
                {
                    "method": name,
                    "n": p["n"],
                    "pinball": p["pinball_mean"],
                    "cover-80": p["cover_80"],
                    "width": p["width"],
                    "winkler-80": p["winkler_80"],
                }
            )
    if rows:
        st.dataframe(pl.DataFrame(rows).round(5), hide_index=True, width="stretch")

    frames = []
    for name, pred in methods.items():
        frames.append(
            pred.select(
                "date", (pl.col("p_0.9") - pl.col("p_0.1")).alias("width")
            ).with_columns(pl.lit(name).alias("method"))
        )
    widths = pl.concat(frames).with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(strict=False)
    )
    st.altair_chart(
        alt.Chart(widths.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("width:Q", title="central-80 width", scale=alt.Scale(zero=False)),
            color=alt.Color("method:N", title=None),
        )
        .properties(height=320, title=f"Interval width over time — {cls}, h={h}"),
        width="stretch",
    )
    st.caption(
        "The Sugeno width system contracts the fan when volatility says the base intervals are too wide; "
        "CQR and ACI apply a single scalar adjustment independent of market state."
    )


def render() -> None:
    st.title("Playground")
    t1, t2, t3 = st.tabs(
        ["Live neural inference", "Manual Sugeno calculator", "Per-class comparison"]
    )
    with t1:
        _live_tab()
    with t2:
        _calculator_tab()
    with t3:
        _compare_tab()
