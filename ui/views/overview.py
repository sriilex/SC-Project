"""Overview page: abstract, method, and the headline result."""

from __future__ import annotations

import polars as pl
import streamlit as st
from lib import loaders


def _test_value(df: pl.DataFrame, h: int, model: str, col: str):
    if df.is_empty():
        return None
    sub = df.filter(
        (pl.col("split") == "test")
        & (pl.col("h") == h)
        & (pl.col("model") == model)
        & (pl.col("scope") == "POOLED")
    )
    return sub[col][0] if sub.height else None


def _pct(a, b):
    if a is None or b in (None, 0):
        return "—"
    return f"{100 * (a - b) / b:+.1f}%"


def render() -> None:
    st.title("Sugeno Fuzzy Calibration of Quantile Forecasts")
    st.caption(
        "Dry bulk vessel-class freight indices · Naman Hans, Srisurya Saket Karuturi, "
        "Ark Malhotra, Srilex Pooria, and Dr. Poonam Rani"
    )

    m = loaders.load_metrics_common()
    base90 = _test_value(m, 90, "lstm", "pinball_mean")
    fz90 = _test_value(m, 90, "lstm_fuzzy", "pinball_mean")
    cqr90 = _test_value(m, 90, "cqr", "pinball_mean")
    aci90 = _test_value(m, 90, "aci", "pinball_mean")
    cov90 = _test_value(m, 90, "lstm_fuzzy", "cover_80")
    fz1 = _test_value(m, 1, "lstm_fuzzy", "pinball_mean")
    base1 = _test_value(m, 1, "lstm", "pinball_mean")

    st.markdown(
        """
Deep quantile forecasters are usually trained once and used as they are, even though
interval miscalibration and median bias change with market conditions. This work adds a
**small Sugeno fuzzy inference layer** on top of a frozen CNN-BiLSTM-attention quantile
model for four dry bulk vessel-class freight indices. The layer reads only signals
available at forecast time — recent interval coverage, recent volatility, and recent
median bias — and rescales the width and shifts the centre of the forecast fan.
Quantile ordering is preserved by construction and the learned rules stay readable.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "h = 90 pinball vs base",
        _pct(fz90, base90),
        help="Relative change in mean pinball loss on the frozen test split.",
    )
    c2.metric(
        "h = 90 vs CQR", _pct(fz90, cqr90), help="Borderline significant (p = 0.047)."
    )
    c3.metric(
        "h = 90 vs ACI", _pct(fz90, aci90), help="Statistically indistinguishable."
    )
    c4.metric(
        "h = 90 coverage",
        f"{cov90:.3f}" if cov90 is not None else "—",
        delta=f"{cov90 - 0.80:+.3f} vs 0.80" if cov90 is not None else None,
        help="Closest to nominal 0.80 among the four methods.",
    )

    st.info(
        f"The benefit is horizon-dependent. At **h = 1** the layer is worse than the base "
        f"model ({_pct(fz1, base1)}) and worse than both conformal calibrators. At **h = 30** "
        "all differences are non-significant. The failures are reported alongside the gains."
    )

    st.divider()
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Method in one picture")
        st.markdown(
            """
1. **Frozen base model** — monotone quantile head emits `P10, P50, P90` of the future log level.
2. **Matured-residual signals** — at issue date `t` only forecasts resolved by `t` are used:
   `cov_gap` (coverage error), `vol` (trailing volatility, z-scored), `bias_z` (median residual).
3. **Two small Sugeno systems** — Gaussian membership functions, product t-norm, weighted-average
   defuzzification, 3 rules (one input) or 9 rules (two inputs).
4. **Apply** — `P50' = P50 + s`, and both half-widths are scaled by `exp(m)`, so the fan keeps its order.
"""
        )
        st.latex(
            r"\hat q_{0.5} = q_{0.5} + s,\qquad "
            r"\hat q_{0.1} = \hat q_{0.5} - (q_{0.5}-q_{0.1})\,e^{m},\qquad "
            r"\hat q_{0.9} = \hat q_{0.5} + (q_{0.9}-q_{0.5})\,e^{m}"
        )
        st.latex(
            r"m = 0.7\tanh\!\Big(\frac{\sum_i w_i r_i}{\sum_i w_i}\Big),\qquad "
            r"s = 0.15\tanh\!\Big(\frac{\sum_i w_i r_i}{\sum_i w_i}\Big)"
        )
        st.caption(
            "Consequents start at zero, so the untrained layer is the identity. "
            "`exp(m)` is bounded to roughly [0.50, 2.01] and the location shift to ±0.15 log units."
        )

    with right:
        st.subheader("Contributions")
        st.markdown(
            """
- A calibration layer of two small Sugeno systems that adjusts a **frozen** quantile fan,
  preserves ordering, and exposes its rules.
- An input design based on **matured** forecast residuals — no outcome unknown at forecast
  time is used.
- A horizon-by-horizon comparison against the uncalibrated model and two conformal
  calibrators (CQR, ACI) with Diebold–Mariano tests and block-bootstrap intervals,
  including the horizons where the layer does **not** help.
"""
        )
        st.subheader("Compared methods")
        st.markdown(
            """
| Method | Adjustment |
|---|---|
| **Base** | none (5-seed ensemble) |
| **CQR** | split-conformal interval shift |
| **ACI** | online, bounded coverage tracking |
| **Sugeno** | learned rule gate on coverage, vol, bias |
"""
        )

    st.divider()
    with st.expander("Pipeline health check", expanded=False):
        health = loaders.data_health()
        cols = st.columns(4)
        cols[0].metric("Train rows", health.get("train_rows", "—"))
        cols[1].metric("Valid rows", health.get("valid_rows", "—"))
        cols[2].metric("Test rows (on disk)", health.get("test_rows", "—"))
        cols[3].metric("LSTM cache key", health.get("pred_key") or "—")
        st.write(
            {
                "result tables present": {
                    k: v for k, v in health.items() if k.endswith(".csv")
                },
                "learned parameter files": health.get("param_files"),
                "BASELINES_INCLUDE_TEST": health.get("env_test_flag") or "(unset)",
            }
        )
        st.caption(
            "The UI only displays precomputed frozen-test rows on the Results page. "
            "Interactive re-scoring defaults to the validation split."
        )
