import os

# 1. Update src/ml/models/__init__.py
INIT_CODE = """from typing import Protocol
import polars as pl

class ForecastingModel(Protocol):
    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        ...
    def predict(self, eval_frames: dict[str, pl.DataFrame]) -> dict[str, dict[int, pl.DataFrame]]:
        ...

from .lstm import LSTMModel
from .xgb import XGBModel
from .timesfm_model import TimesFMModel
from .hybrid_tfm_xgb import XGBPlusTFMModel
from .hybrid_linear_xgb import XGBPlusLinearModel

# Add new models here to have them automatically picked up by baselines
MODELS = {
    "lstm": LSTMModel,
    "xgb": XGBModel,
    "timesfm": TimesFMModel,
    "xgb+tfm": XGBPlusTFMModel,
    "xgb+linear": XGBPlusLinearModel,
}
"""
with open("src/ml/models/__init__.py", "w") as f:
    f.write(INIT_CODE)

# 2. Update baselines.py
with open("src/ml/baselines.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "for model_name, predict_fn in MODELS.items():" in line:
        new_lines.extend([
            "        for model_name, ModelClass in MODELS.items():\n",
            "            model = ModelClass()\n",
            "            model.fit(splits[\"train\"], HORIZONS)\n",
            "            preds = model.predict({\"valid\": splits[\"valid\"], \"test\": splits[\"test\"]})\n",
            "            for split_name, h_preds in preds.items():\n",
            "                for curr_h, pred_df in h_preds.items():\n",
            "                    if curr_h == h:\n",
            "                        candidates.setdefault(model_name, {})[split_name] = pred_df\n"
        ])
    elif "candidates[model_name] = predict_fn(" in line or "splits[\"train\"]," in line or "{\"valid\": splits[\"valid\"], \"test\": splits[\"test\"]}," in line or "h," in line:
        if "h," in line and "def predict(" not in line:
            pass # skip these lines
    else:
        new_lines.append(line)

with open("src/ml/baselines.py", "w") as f:
    f.writelines(new_lines)
