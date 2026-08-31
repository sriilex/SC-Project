from .lstm import predict as predict_lstm
from .xgb import predict as predict_xgb

# Add new models here to have them automatically picked up by baselines
MODELS = {
    "lstm": predict_lstm,
    "xgb": predict_xgb,
}
