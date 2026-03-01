"""
Joint next activity + lifecycle prediction.
"""

from .config import DualPredictionConfig
from .predictor import DualLifecycleNextActivityPredictor

# Keep runtime imports lightweight so simulation can load predictor
# without requiring full training dependencies (e.g., scikit-learn).
try:
    from .trainer import run_full_experiment, train_variant
except Exception:  # pragma: no cover - optional training dependency path
    run_full_experiment = None
    train_variant = None

__all__ = ["DualPredictionConfig", "DualLifecycleNextActivityPredictor", "train_variant", "run_full_experiment"]
