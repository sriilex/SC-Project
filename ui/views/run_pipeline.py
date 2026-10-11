"""Run Pipeline: stream the project's CLI steps and refit the fuzzy layer in-process."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st
from lib import fuzzy, loaders, metrics, runner, viz
from lib.paths import HORIZONS, PARAM_DIR, REPO_ROOT

from ml.fuzzy_calibration import calibrate as fuzzy_calibrate
from ml.fuzzy_calibration import fit_fuzzy


def _run_step_ui() -> None:
    st.subheader("Pipeline commands")
    st.caption(
        "These shell out to the same modules used on the command line. Logs stream live; "
        "after a run the UI cache is cleared so new CSVs are picked up."
    )
    labels = {k: v.label for k, v in runner.STEPS.items()}
    key = st.selectbox("Step", list(runner.STEPS), format_func=lambda k: labels[k])
    step = runner.STEPS[key]

    st.markdown(f"**What it does.** {step.description}")
    meta = st.columns(2)
    meta[0].write({"module": " ".join(step.args), "heavy": step.heavy})
    meta[1].write(
        {
            "env overrides": step.env or "(none)",
            "outputs": [p.name for p in step.outputs],
        }
    )

    confirm = True
    if step.heavy:
        st.warning(
            "This is the slow step (trains/loads the LSTM ensemble and the tree models it selects). "
            "Expect minutes, not seconds."
        )
        confirm = st.checkbox("I want to run this heavy step", key=f"heavy_{key}")

    if st.button("Run step", type="primary", disabled=not confirm, key=f"run_{key}"):
        placeholder = st.empty()
        lines: list[str] = []
        rc: int | None = None
        for line, code in runner.stream(step):
            if code is not None:
                rc = code
                break
            lines.append(line)
            placeholder.code("\n".join(lines[-300:]), language="text")
        st.cache_data.clear()
        if rc == 0:
            st.success("Finished with exit code 0.")
        else:
            st.error(f"Finished with exit code {rc}. See log above.")

    st.divider()
    st.markdown("**Latest outputs on disk**")
    tail_cols = st.columns(len(step.outputs) if step.outputs else 1)
    for col, path in zip(tail_cols, step.outputs):
        with col:
            st.caption(path.name)
            st.code(runner.tail(path, 25), language="text")


def _refit_ui() -> None:
    st.subheader("Refit the Sugeno layer on validation (in-process)")
    st.caption(
        "Trains only the FIS parameters with the base model frozen, exactly as the pipeline does. "
        "Fit and scoring use the validation split, so this is safe to experiment with."
    )

    with st.form("refit_form"):
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox("Horizon", HORIZONS, index=3)
        epochs = c2.slider("Epochs", 50, 800, 300, 50)
        lr = c3.select_slider(
            "Learning rate", options=[0.01, 0.02, 0.05, 0.1, 0.2], value=0.05
        )
        c4, c5, c6, c7 = st.columns(4)
        use_width = c4.checkbox("Width system", value=True)
        use_loc = c5.checkbox("Location system", value=True)
        with_vol = c6.checkbox("Use volatility input", value=True)
        linear = c7.checkbox("Linear consequents", value=False)
        seed = st.number_input("Seed", 0, 999, 0, 1)
        submitted = st.form_submit_button("Refit", type="primary")

    if submitted:
        if not (use_width or use_loc):
            st.error("Enable at least one system.")
            return
        if h >= 30 and use_width and with_vol:
            st.info(
                "At h ≥ 30 the paper's width system is volatility-only; coverage-gap input is dropped automatically."
            )
        from ml.baselines import _context_frame

        preds = {
            "train": {h: loaders.load_base_pred("train", h)},
            "valid": {h: loaders.load_base_pred("valid", h)},
        }
        splits = {s: loaders.load_split_cached(s) for s in ("train", "valid")}
        context = _context_frame(preds, splits, ["train", "valid"], h)
        train_frame = _context_frame(preds, splits, ["valid"], h)
        with st.spinner(
            f"Training {epochs} epochs on {train_frame.height} validation rows…"
        ):
            params, stats = fit_fuzzy(
                context,
                train_frame,
                h,
                use_width=use_width,
                use_loc=use_loc,
                with_vol=with_vol,
                linear=linear,
                epochs=int(epochs),
                lr=float(lr),
                seed=int(seed),
            )
        st.session_state[f"refit_{h}"] = {"params": params, "stats": stats}
        st.success("Refit complete.")

    store = st.session_state.get("refit")
    if not store:
        # Show the most recent refit for any horizon if present.
        candidates = [
            k for k in st.session_state if isinstance(k, str) and k.startswith("refit_")
        ]
        store = st.session_state.get(candidates[-1]) if candidates else None
    if not store:
        st.info("No refit yet in this session.")
        return

    params = store["params"]
    stats = store["stats"]
    h = int(params.get("h", 90))
    st.markdown(f"**Refit statistics (h = {h})**")
    st.write(stats)

    c1, c2 = st.columns(2, gap="large")
    for system, col in (("width", c1), ("loc", c2)):
        fig = viz.membership_figure(params.get(system, {}), system)
        col.pyplot(fig)
        plt.close(fig)

    st.dataframe(fuzzy.summarise(params), hide_index=True, width="stretch")

    # Score the refit against the frozen layer and the base on valid.
    base = loaders.load_base_pred("valid", h)
    context = loaders.build_history(h, include_test=False)
    res = []
    if not base.is_empty() and not context.is_empty():
        truth = loaders.truth_frame("valid", h)
        frozen = loaders.load_fuzzy_params(h)
        for name, prm in (
            ("lstm (base)", None),
            ("fuzzy (frozen)", frozen),
            ("fuzzy (refit)", params),
        ):
            pred = base if prm is None else fuzzy_calibrate(base, context, h, prm)
            p = metrics.pooled(pred, truth, name)
            if p:
                res.append(
                    {
                        "variant": name,
                        "n": p["n"],
                        "pinball": p["pinball_mean"],
                        "cover-80": p["cover_80"],
                        "width": p["width"],
                        "winkler-80": p["winkler_80"],
                    }
                )
    if res:
        table = pl.DataFrame(res)
        b = table.filter(pl.col("variant") == "lstm (base)")["pinball"][0]
        st.dataframe(
            table.with_columns(((pl.col("pinball") - b) / b).alias("Δ vs base")).round(
                5
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("Validation only; the frozen-test split is never read here.")

    tag = st.text_input("Save refit as", value=f"ui_refit_h{h}")
    c1, c2 = st.columns(2)
    if c1.button("Save refit JSON"):
        (PARAM_DIR / f"{tag}.json").write_text(json.dumps(params, indent=2))
        st.success(f"Wrote {tag}.json to src/data/fuzzy_params/")
    c2.download_button(
        "Download refit JSON",
        data=json.dumps(params, indent=2),
        file_name=f"{tag}.json",
        mime="application/json",
    )


def render() -> None:
    st.title("Run Pipeline")
    st.caption(f"Working directory: `{REPO_ROOT}`")
    t1, t2 = st.tabs(["CLI steps", "Refit fuzzy layer"])
    with t1:
        _run_step_ui()
    with t2:
        _refit_ui()
