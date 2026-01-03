"""
Advanced Next-Activity Prediction Package

A class-based framework for training and evaluating LSTM models
for predicting the next activity at BPMN decision points.
"""

# Core classes
from .preprocessing import DecisionPointExtractor, TrainingDataGenerator
from .models import SequenceEncoder, ContextEncoder, LSTMPredictor
from .evaluation import ModelEvaluator, FeatureImportanceAnalyzer
from .parsers import BPMNParser
from .storage import ModelPersistence
from .api import load_models, predict_next_activity
from .simulation import (
    load_simulation_assets,
    decision_function_advanced,
    simulate_cases_advanced,
    events_to_dataframe,
)

# Backward-compatible function aliases
from .preprocessing.decision_points import extract_bpmn_decision_point_map
from .preprocessing.training_data import generate_enriched_training_sets_simple
from .models.encoders import prepare_sequences_and_labels, prepare_context_attributes
from .models.lstm import build_lstm_model, train_model, expand_and_cast
from .evaluation.metrics import evaluate_model, evaluate_baseline, compare_f1_for_trained_model
from .evaluation.importance import (
    add_unknown_label,
    model_score,
    permutation_importance_context,
    permutation_importance_all_features,
    plot_feature_importance,
)

# Legacy parser alias
from .parsers.bpmn import AdvancedBPMNParser

__all__ = [
    # Classes
    "DecisionPointExtractor",
    "TrainingDataGenerator",
    "SequenceEncoder",
    "ContextEncoder",
    "LSTMPredictor",
    "ModelEvaluator",
    "FeatureImportanceAnalyzer",
    "BPMNParser",
    "AdvancedBPMNParser",
    "ModelPersistence",
    "load_models",
    "predict_next_activity",
    "load_simulation_assets",
    "decision_function_advanced",
    "simulate_cases_advanced",
    "events_to_dataframe",
    # Legacy functions - preprocessing
    "extract_bpmn_decision_point_map",
    "generate_enriched_training_sets_simple",
    # Legacy functions - training
    "prepare_sequences_and_labels",
    "prepare_context_attributes",
    "build_lstm_model",
    "train_model",
    "expand_and_cast",
    # Legacy functions - evaluation
    "evaluate_model",
    "evaluate_baseline",
    "compare_f1_for_trained_model",
    "add_unknown_label",
    "model_score",
    "permutation_importance_context",
    "permutation_importance_all_features",
    "plot_feature_importance",
]

