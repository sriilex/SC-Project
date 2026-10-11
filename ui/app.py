"""Streamlit entry point for the Sugeno fuzzy calibration console.

Run from the repository root::

    uv run streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

import streamlit as st

st.set_page_config(
    page_title="Sugeno Fuzzy Calibration",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib.paths import REPO_ROOT
from views import (
    base_model,
    data_explorer,
    fuzzy_lab,
    overview,
    playground,
    results,
    run_pipeline,
)

PAGES = {
    "Overview": overview.render,
    "Data Explorer": data_explorer.render,
    "Base Model": base_model.render,
    "Results & Significance": results.render,
    "Fuzzy Rule Lab": fuzzy_lab.render,
    "Run Pipeline": run_pipeline.render,
    "Playground": playground.render,
}

ICONS = {
    "Overview": "📄",
    "Data Explorer": "📈",
    "Base Model": "🧠",
    "Results & Significance": "🏁",
    "Fuzzy Rule Lab": "🧩",
    "Run Pipeline": "⚙️",
    "Playground": "🎛️",
}


def main() -> None:
    with st.sidebar:
        st.markdown("## Sugeno Fuzzy Calibration")
        st.caption("Quantile forecasts for dry bulk vessel-class indices")
        page = st.radio(
            "Navigate",
            list(PAGES),
            format_func=lambda name: f"{ICONS[name]}  {name}",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            "Reads the pipeline's own artifacts and imports `src/ml` directly, so "
            "metrics and fuzzy inference match the paper."
        )
        st.caption(f"Repo root: `{REPO_ROOT}`")
        st.caption(
            "Frozen test split is displayed read-only; interactive scoring defaults to validation."
        )

    try:
        PAGES[page]()
    except Exception as exc:  # noqa: BLE001 - keep the app alive and show context
        st.error(f"`{page}` failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc(), language="text")


if __name__ == "__main__":
    main()
