"""
Unified Activity-Lifecycle Prediction Module.

A single model that predicts both next activity AND lifecycle transition,
avoiding the loop issues of per-decision-point models.
"""

from .data_generator import UnifiedDataGenerator
from .model import build_unified_model, UnifiedPredictor
from .predictor import UnifiedNextActivityPredictor
from .persistence import UnifiedModelPersistence

__all__ = [
    "UnifiedDataGenerator",
    "build_unified_model",
    "UnifiedPredictor",
    "UnifiedNextActivityPredictor",
    "UnifiedModelPersistence",
]

