"""
Next Activity Prediction Module with One-Hot Encoding

LSTM-based next activity prediction for process simulation using one-hot encoding.
Filters event logs to start/complete lifecycles and predicts next activity with END detection.
"""

from .predictor import LSTMNextActivityPredictorOneHot
from .config import NextActivityConfigOneHot
from .trainer import train_model_onehot

__all__ = [
    'LSTMNextActivityPredictorOneHot',
    'NextActivityConfigOneHot',
    'train_model_onehot',
]

