"""Repository paths and project-wide constants for the Streamlit UI.

Importing this module puts ``<repo>/src`` on ``sys.path`` so the UI can reuse
the project's own ``ml`` package (metric definitions, fuzzy calibration,
conformal baselines, inference) instead of re-implementing them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

UI_DIR: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = UI_DIR.parent
SRC: Final[Path] = REPO_ROOT / "src"
DATA: Final[Path] = SRC / "data"
PRED_DIR: Final[Path] = DATA / "preds"
PARAM_DIR: Final[Path] = DATA / "fuzzy_params"
FIG_DIR: Final[Path] = REPO_ROOT / "docs" / "figures"

# Make ``import ml`` work regardless of the process working directory.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CLASSES: Final[tuple[str, ...]] = ("Capesize", "Panamax", "Supramax", "Handysize")
HORIZONS: Final[tuple[int, ...]] = (1, 7, 30, 90)
QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)

#: Raw index series in ``master_long.parquet`` for each vessel class.
CLASS_SERIES: Final[dict[str, str]] = {
    "Capesize": "BC_INDEX",
    "Panamax": "BPI_INDEX",
    "Supramax": "BSI_INDEX",
    "Handysize": "BHSI_INDEX",
}
BDI_SERIES: Final[str] = "BD_INDEX"

#: Model set used in the paper's headline common-support table.
PAPER_MODELS: Final[tuple[str, ...]] = (
    "lstm",
    "lstm_fuzzy",
    "cqr",
    "aci",
    "ar1",
    "xgb",
    "rw_emp",
)

#: Human-readable labels.
MODEL_LABELS: Final[dict[str, str]] = {
    "lstm": "Base LSTM",
    "lstm_fuzzy": "Sugeno fuzzy",
    "lstm_fuzzy_old": "Fuzzy (old)",
    "lstm_fuzzy_ui": "Fuzzy (UI refit)",
    "cqr": "CQR",
    "aci": "ACI",
    "const_shift": "Const. shift",
    "ar1": "AR(1)",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "rw": "Random walk",
    "rw_emp": "RW (empirical)",
}

CALIBRATED_MODELS: Final[tuple[str, ...]] = (
    "lstm_fuzzy",
    "cqr",
    "aci",
    "const_shift",
    "lstm_fuzzy_old",
)

#: Rolling-window constants mirrored from ``ml.fuzzy_calibration``.
N_MATURED: Final[int] = 60
VOL_WINDOW: Final[int] = 14
