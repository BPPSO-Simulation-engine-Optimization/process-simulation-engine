"""
processing_time_prediction
==========================
Package for predicting inter-event (processing) times in process-mining event logs.

Supported prediction methods
-----------------------------
"distribution"     - Fit log-normal distributions per activity-transition pair.
"ml"               - Random-Forest point-prediction with hand-crafted features.
"probabilistic_ml" - LSTM with heteroscedastic Gaussian output.
"xgboost"          - Activity-specific XGBoost / quantile-regression models.

Public API
----------
Core classes
    ProcessingTimeTrainer          - Train / fit any of the above methods.
    ProcessingTimePredictionClass  - Load a saved model and make predictions.

XGBoost building blocks (usable independently)
    DataLoader              - Load XES / CSV event logs into DataFrames.
    FeatureEngineering      - Calculate processing times and build feature matrices.
    ModelTrainer            - Wrap XGBRegressor in a sklearn Pipeline.
    QuantileModelTrainer    - Train multiple quantile-regression models.
    Evaluator               - Compute MAE / RMSE / R2 metrics.
    ActivitySpecificModel   - Orchestrate per-activity XGBoost model training.

Scripts
    evaluate_all.main()     - Full evaluation workflow (data -> train -> evaluate -> save).
"""

from .ProcessingTimeTrainer import ProcessingTimeTrainer
from .ProcessingTimePredictionClass import ProcessingTimePredictionClass

# XGBoost sub-modules (imported individually so the package still works
# even if xgboost / lightgbm are not installed).
try:
    from .data_loader import DataLoader
    from .feature_engineering import FeatureEngineering
    from .model_trainer import ModelTrainer
    from .quantile_model import QuantileModelTrainer
    from .evaluator import Evaluator
    from .activity_specific_model import ActivitySpecificModel
    _XGBOOST_MODULES_AVAILABLE = True
except ImportError:
    _XGBOOST_MODULES_AVAILABLE = False

__all__ = [
    # Core
    "ProcessingTimeTrainer",
    "ProcessingTimePredictionClass",
    # XGBoost building blocks
    "DataLoader",
    "FeatureEngineering",
    "ModelTrainer",
    "QuantileModelTrainer",
    "Evaluator",
    "ActivitySpecificModel",
]
