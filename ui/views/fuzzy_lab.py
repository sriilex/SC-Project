"""Fuzzy Rule Lab.

Inspect, edit and run the Sugeno calibration layer:

* membership curves, rule consequents and defuzzified surfaces per system;
* a rule/firing scanner for a single matured forecast;
* live re-scoring of a whole split (validation by default; test only behind an
  explicit, non-tuning confirmation).
"""

from __future__ import annotations

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import streamlit as st
from lib import fuzzy, loaders, metrics, viz
from lib.paths import CLASSES, HORIZONS, PARAM_DIR

from ml.conformal import aci_adjust, apply_cqr, fit_cqr
from ml.fuzzy_calibration import calibrate as fuzzy_calibrate
from ml.fuzzy_calibration import compute_fuzzy_inputs

SYSTEMS = ("width", "loc")


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------


def _state_keys(h: int) -> tuple[str, str]:
    return f"fuzzy_params_h{h}", f"fuzzy_frozen_h{h}"


def get_params(h: int) -> tuple[dict, dict]:
    key, frozen_key = _state_keys(h)
    if key not in st.session_state:
        loaded = loaders.load_fuzzy_params(h)
        st.session_state[key] = copy.deepcopy(loaded)
        st.session_state[frozen_key] = copy.deepcopy(loaded)
    return st.session_state[key], st.session_state[frozen_key]


def clear_widget_state(h: int) -> None:
    """Drop editor widget state so a reset is not overwritten by stale widgets."""
    prefixes = (f"cons_{h}_", f"c_{h}_", f"w_{h}_")
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(prefixes):
            del st.session_state[key]


def is_dirty(h: int) -> bool:
    key, frozen_key = _state_keys(h)
    cur = st.session_state.get(key)
    frozen = st.session_state.get(frozen_key)
    if cur is None or frozen is None:
        return False
    return json.dumps(cur, sort_keys=True) != json.dumps(frozen, sort_keys=True)


# ---------------------------------------------------------------------------
# cached scoring
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _scored(h: int, split: str, params_json: str, include_test: bool):
    params = json.loads(params_json)
    base = loaders.load_base_pred(split, h)
    history = loaders.build_history(h, include_test=include_test)
    if base.is_empty() or history.is_empty():
        return base, base, pl.DataFrame()
    cal = fuzzy_calibrate(base, history, h, params)
    inputs = compute_fuzzy_inputs(history, h)
    joined = base.join(
        inputs.select(
            "date", "target_class", "vol_z", "cov_gap", "bias_z", "n_matured"
        ),
        on=["date", "target_class"],
        how="left",
    )
    return base, cal, joined


@st.cache_data(show_spinner=False)
def _valid_inputs(h: int) -> pl.DataFrame:
    history = loaders.build_history(h, include_test=False)
    return (
        compute_fuzzy_inputs(history, h) if not history.is_empty() else pl.DataFrame()
    )


@st.cache_data(show_spinner=False)
def _cqr_aci(h: int, split: str, include_test: bool):
    from ml.baselines import _context_frame

    base = loaders.load_base_pred(split, h)
    if base.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    preds = {
        "train": {h: loaders.load_base_pred("train", h)},
        "valid": {h: loaders.load_base_pred("valid", h)},
    }
    splits = {s: loaders.load_split_cached(s) for s in ("train", "valid")}
    calib = _context_frame(preds, splits, ["valid"], h)
    cqr = apply_cqr(base, fit_cqr(calib))
    aci, _ = aci_adjust(base, loaders.load_split_cached(split), h)
    return cqr, aci


# ---------------------------------------------------------------------------
# editor
# ---------------------------------------------------------------------------


def _render_editor(h: int) -> None:
    params, _ = get_params(h)
    for system in SYSTEMS:
        spec = params.get(system, {})
        if not spec:
            continue
        label = "Width system f_w" if system == "width" else "Location system f_l"

        with st.expander(f"Edit {label} — consequents", expanded=False):
            df = fuzzy.describe_rules(spec)
            edited = st.data_editor(
                df,
                key=f"cons_{h}_{system}",
                hide_index=True,
                width="stretch",
                disabled=["rule", "if"],
                column_config={
                    "consequent": st.column_config.NumberColumn(
                        "consequent",
                        min_value=-10.0,
                        max_value=10.0,
                        step=0.01,
                        format="%.4f",
                    )
                },
            )
            if isinstance(edited, pl.DataFrame) and "consequent" in edited.columns:
                values = [
                    v if v is not None else 0.0 for v in edited["consequent"].to_list()
                ]
                params[system] = fuzzy.set_consequents(spec, values)

        with st.expander(f"Edit {label} — membership functions", expanded=False):
            spec = params.get(system, {})
            inputs = list(spec.get("inputs") or [])
            centers = [list(map(float, row)) for row in spec.get("centers", [])]
            widths = [list(map(float, row)) for row in spec.get("widths", [])]
            for i, iname in enumerate(inputs):
                cols = st.columns(int(spec.get("n_mfs", 3)))
                for k, col in enumerate(cols):
                    with col:
                        centers[i][k] = st.slider(
                            f"{fuzzy.INPUT_LABELS.get(iname, iname)} · {fuzzy.LINGUISTIC[k]} centre",
                            -3.0,
                            3.0,
                            float(centers[i][k]),
                            0.01,
                            key=f"c_{h}_{system}_{i}_{k}",
                        )
                        widths[i][k] = st.slider(
                            f"{fuzzy.INPUT_LABELS.get(iname, iname)} · {fuzzy.LINGUISTIC[k]} width",
                            0.05,
                            2.0,
                            float(widths[i][k]),
                            0.01,
                            key=f"w_{h}_{system}_{i}_{k}",
                        )
            if inputs:
                new_spec = copy.deepcopy(spec)
                new_spec["centers"] = centers
                new_spec["widths"] = widths
                params[system] = new_spec

    st.session_state[_state_keys(h)[0]] = params


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------


def _tab_systems(h: int, params: dict) -> None:
    st.dataframe(fuzzy.summarise(params), hide_index=True, width="stretch")

    st.markdown("**Membership functions** (x-axis is the standardised input)")
    c1, c2 = st.columns(2, gap="large")
    for system, col in (("width", c1), ("loc", c2)):
        fig = viz.membership_figure(params.get(system, {}), system)
        col.pyplot(fig)
        plt.close(fig)

    st.markdown("**Defuzzified surfaces**")
    c3, c4 = st.columns(2, gap="large")
    for system, col in (("width", c3), ("loc", c4)):
        fig = viz.surface_figure(params.get(system, {}), system)
        col.pyplot(fig)
        plt.close(fig)

    st.markdown("**Rule consequents**")
    c5, c6 = st.columns(2, gap="large")
    for system, col in (("width", c5), ("loc", c6)):
        fig = viz.consequence_heatmap(params.get(system, {}), system)
        if fig is not None:
            col.pyplot(fig)
            plt.close(fig)


def _tab_scanner(h: int) -> None:
    inputs = _valid_inputs(h)
    if inputs.is_empty():
        st.info("Fuzzy inputs unavailable.")
        return
    usable = inputs.filter(pl.col("n_matured") >= 5)
    if usable.is_empty():
        st.warning(
            "No rows have enough matured forecasts to compute causal inputs at this horizon."
        )
        return
    c1, c2 = st.columns(2)
    cls = c1.selectbox("Vessel class", CLASSES, index=0, key="scan_cls")
    dates = usable.filter(pl.col("target_class") == cls)["date"].sort().to_list()
    if not dates:
        st.warning("No usable dates for this class.")
        return
    picked = c2.selectbox(
        "Forecast issue date", dates, index=len(dates) - 1, key="scan_date"
    )

    row = usable.filter((pl.col("target_class") == cls) & (pl.col("date") == picked))
    if row.is_empty():
        st.warning("No input row.")
        return
    x_raw = {
        "vol_z": float(row["vol_z"][0]),
        "cov_gap": float(row["cov_gap"][0]),
        "bias_z": float(row["bias_z"][0]),
    }
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("volatility (z)", f"{x_raw['vol_z']:+.3f}")
    m2.metric("coverage gap", f"{x_raw['cov_gap']:+.3f}")
    m3.metric("median bias (z)", f"{x_raw['bias_z']:+.3f}")
    m4.metric("matured rows", int(row["n_matured"][0]))

    params, _ = get_params(h)
    left, right = st.columns(2, gap="large")
    for system, col in (("width", left), ("loc", right)):
        spec = params.get(system, {})
        col.markdown(f"**{fuzzy.SYSTEM_LABELS.get(system, system)}**")
        cols = list(spec.get("inputs") or [])
        if not cols:
            col.caption("inactive (no inputs)")
            continue
        x = np.array([[x_raw[c] for c in cols]])
        mu = fuzzy.memberships(spec, x)[0]
        table = []
        for i, iname in enumerate(cols):
            table.append(
                {
                    "input": fuzzy.INPUT_LABELS.get(iname, iname),
                    **{
                        fuzzy.LINGUISTIC[k]: round(float(mu[i, k]), 4)
                        for k in range(mu.shape[1])
                    },
                }
            )
        col.dataframe(pl.DataFrame(table), hide_index=True, width="stretch")
        weights = fuzzy.rule_weights(spec, x)[0]
        col.altair_chart(viz.firing_chart(weights, spec), width="stretch")
        out = float(fuzzy.evaluate(spec, x)[0])
        scale = "m" if system == "width" else "s"
        col.metric(f"defuzzified output {scale}", f"{out:+.4f}")

    params_json = json.dumps(params, sort_keys=True)
    vbase, cal, _ = _scored(h, "valid", params_json, False)
    vtruth = loaders.truth_frame("valid", h)
    b = vbase.filter(pl.col("target_class") == cls).sort("date")
    t = vtruth.filter(pl.col("target_class") == cls).sort("date")
    if not cal.is_empty():
        c = cal.filter(pl.col("target_class") == cls).sort("date")
        st.altair_chart(
            viz.fan_chart(
                b,
                c,
                t.select("date", "y"),
                title=f"Base vs Sugeno — {cls}, h={h} (validation)",
            ),
            width="stretch",
        )

    single = vbase.filter((pl.col("target_class") == cls) & (pl.col("date") == picked))
    if not single.is_empty() and not cal.is_empty():
        cs = cal.filter((pl.col("target_class") == cls) & (pl.col("date") == picked))
        ts = vtruth.filter((pl.col("target_class") == cls) & (pl.col("date") == picked))
        rows = [
            {
                "quantile": "P10",
                "base": float(single["p_0.1"][0]),
                "calibrated": float(cs["p_0.1"][0]),
            },
            {
                "quantile": "P50",
                "base": float(single["p_0.5"][0]),
                "calibrated": float(cs["p_0.5"][0]),
            },
            {
                "quantile": "P90",
                "base": float(single["p_0.9"][0]),
                "calibrated": float(cs["p_0.9"][0]),
            },
        ]
        st.markdown("**Selected forecast**")
        st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")
        if not ts.is_empty():
            y = float(ts["y"][0])
            st.write(
                {
                    "realised y": round(y, 4),
                    "base covered": float(single["p_0.1"][0])
                    <= y
                    <= float(single["p_0.9"][0]),
                    "Sugeno covered": float(cs["p_0.1"][0])
                    <= y
                    <= float(cs["p_0.9"][0]),
                }
            )


def _tab_scoring(h: int, params: dict, include_test: bool) -> None:
    split = "test" if include_test else "valid"
    params_json = json.dumps(params, sort_keys=True)
    base, cal, _ = _scored(h, split, params_json, include_test)
    truth = loaders.truth_frame(split, h)
    if base.is_empty() or truth.is_empty() or cal.is_empty():
        st.info("Prediction cache missing for this split/horizon.")
        return

    rows = []
    for model_name, pred in (("lstm", base), ("lstm_fuzzy", cal)):
        p = metrics.pooled(pred, truth, model_name, split)
        if p:
            rows.append(
                {
                    "model": model_name,
                    "n": p["n"],
                    "pinball": p["pinball_mean"],
                    "cover-80": p["cover_80"],
                    "width": p["width"],
                    "winkler-80": p["winkler_80"],
                }
            )
    if st.toggle("Also compute CQR and ACI on the base fan", value=(split == "valid")):
        cqr, aci = _cqr_aci(h, split, include_test)
        for name, pred in (("cqr", cqr), ("aci", aci)):
            if not pred.is_empty():
                p = metrics.pooled(pred, truth, name, split)
                if p:
                    rows.append(
                        {
                            "model": name,
                            "n": p["n"],
                            "pinball": p["pinball_mean"],
                            "cover-80": p["cover_80"],
                            "width": p["width"],
                            "winkler-80": p["winkler_80"],
                        }
                    )
    if not rows:
        st.info("No metrics.")
        return
    table = pl.DataFrame(rows)
    base_pin = table.filter(pl.col("model") == "lstm")["pinball"][0]
    table = table.with_columns(
        ((pl.col("pinball") - base_pin) / base_pin).alias("Δ pinball vs base")
    )
    st.dataframe(
        table.with_columns(
            pl.col("pinball").round(5), pl.col("Δ pinball vs base").round(4)
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Negative Δ pinball is better. "
        + (
            "Test is the frozen split."
            if split == "test"
            else "Validation is in-sample for layers fit on valid."
        )
    )

    pc_rows = []
    for model_name, pred in (("lstm", base), ("lstm_fuzzy", cal)):
        pc = metrics.per_class(pred, truth, model_name, split)
        if not pc.is_empty():
            pc_rows.append(
                pc.select(
                    "scope", "n", "pinball_mean", "cover_80", "winkler_80"
                ).with_columns(pl.lit(model_name).alias("model"))
            )
    if pc_rows:
        pcdf = pl.concat(pc_rows)
        pivot = pcdf.pivot(on="model", index="scope", values="pinball_mean").rename(
            {"lstm": "base pinball", "lstm_fuzzy": "Sugeno pinball"}
        )
        if {"base pinball", "Sugeno pinball"}.issubset(pivot.columns):
            pivot = pivot.with_columns(
                (
                    (pl.col("Sugeno pinball") - pl.col("base pinball"))
                    / pl.col("base pinball")
                ).alias("Δ")
            )
        st.dataframe(pivot.sort("scope"), hide_index=True, width="stretch")

    from ml.evaluation import aligned_losses, block_bootstrap_ci, dm_test

    source = truth.with_columns(pl.col("y").alias(f"y_step_h{h}"))
    la, lb, _ = aligned_losses(cal, base, source, h, q=None)
    if la.size:
        dm = dm_test(la, lb, h)
        boot = block_bootstrap_ci(la - lb, block=max(h, 5), n_boot=1000)
        st.markdown("**Diebold–Mariano: Sugeno − base (mean pinball)**")
        st.write(
            {
                "mean diff": round(dm["dm_mean_diff"], 6),
                "DM stat": round(dm["dm_stat"], 3),
                "p-value": round(dm["dm_p"], 4),
                "bootstrap 95%": [round(boot["boot_lo"], 6), round(boot["boot_hi"], 6)],
                "n": dm["dm_n"],
            }
        )

    st.divider()
    st.markdown("**Export this configuration**")
    tag = st.text_input("File tag", value=f"ui_h{h}")
    c1, c2 = st.columns(2)
    if c1.button("Save parameters", key=f"save_{h}"):
        path = PARAM_DIR / f"{tag}.json"
        path.write_text(json.dumps(params, indent=2))
        st.success(f"Wrote {path.name} to src/data/fuzzy_params/")
    c2.download_button(
        "Download JSON",
        data=json.dumps(params, indent=2),
        file_name=f"{tag}.json",
        mime="application/json",
        key=f"dl_{h}",
    )


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def render() -> None:
    st.title("Fuzzy Rule Lab")
    st.caption(
        "The calibration layer is a pair of tiny Sugeno systems. Edit the rules and watch the "
        "surface, the firing pattern and the split metrics move."
    )

    h = st.selectbox("Horizon (days)", HORIZONS, index=3)
    if not loaders.load_fuzzy_params(h):
        st.error(f"No learned parameters for h={h}. Run the baselines step first.")
        return
    get_params(h)

    _render_editor(h)
    params, _ = get_params(h)
    dirty = is_dirty(h)

    b1, b2, b3 = st.columns([2, 2, 3])
    if dirty:
        b1.warning("Edited (not the frozen configuration)")
        if b2.button("Reset to frozen", key=f"reset_{h}"):
            clear_widget_state(h)
            st.session_state[_state_keys(h)[0]] = copy.deepcopy(
                st.session_state[_state_keys(h)[1]]
            )
            st.rerun()
    else:
        b1.success("Frozen configuration")

    with b3:
        allow_test = st.toggle(
            "Score frozen test split (advanced)",
            value=False,
            key=f"test_toggle_{h}",
            disabled=dirty,
            help="Disabled while you have unsaved edits, so the UI cannot be used to tune on test.",
        )
        confirmed = False
        if allow_test:
            st.caption(
                "The frozen test split is read once, at the end. Confirm to proceed."
            )
            ok = st.checkbox(
                "I will not use these numbers to tune anything", key=f"test_ack_{h}"
            )
            phrase = st.text_input("Type SCORE TEST to confirm", key=f"test_phrase_{h}")
            confirmed = (
                bool(ok and phrase.strip().upper() == "SCORE TEST") and not dirty
            )
            if not confirmed:
                st.info("Confirmation incomplete — scoring validation instead.")

    tabs = st.tabs(["Learned systems", "What-if scanner", "Live scoring"])
    with tabs[0]:
        _tab_systems(h, params)
    with tabs[1]:
        _tab_scanner(h)
    with tabs[2]:
        _tab_scoring(h, params, confirmed)
