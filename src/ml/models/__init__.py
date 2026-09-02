from .lstm import predict as predict_lstm
from .xgb import predict as predict_xgb
from .timesfm_model import predict as predict_timesfm

# Add new models here to have them automatically picked up by baselines
MODELS = {
    "lstm": predict_lstm,
    "xgb": predict_xgb,
    "timesfm": predict_timesfm,
}
