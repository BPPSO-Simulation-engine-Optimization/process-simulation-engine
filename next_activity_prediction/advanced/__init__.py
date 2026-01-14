"""
Next Activity Prediction Module

LSTM-based next activity prediction for process simulation.
Filters event logs to start/complete lifecycles and predicts next activity with END detection.

Also includes suffix prediction model that predicts entire remaining sequences.
"""

from .predictor import LSTMNextActivityPredictor
from .config import NextActivityConfig
from .trainer import train_model
from .suffix_predictor import LSTMSuffixPredictor
from .suffix_trainer import train_suffix_model

__all__ = [
    'LSTMNextActivityPredictor',
    'NextActivityConfig',
    'train_model',
    'LSTMSuffixPredictor',
    'train_suffix_model',
]

