"""
Persistence utilities for BPIC17 simplified model.
"""

import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Dict, Any
import numpy as np

try:
    import keras
    _register = keras.saving.register_keras_serializable
except AttributeError:
    try:
        import keras
        _register = keras.utils.register_keras_serializable
    except AttributeError:
        def _register(package=None):
            return lambda fn: fn


class BPIC17SimplifiedPersistence:
    """Saves and loads BPIC17 simplified model bundles."""

    MODEL_FILE = "model.keras"
    ENCODER_FILE = "encoder.pkl"
    METADATA_FILE = "metadata.pkl"

    @classmethod
    def save(cls, predictor, directory: str):
        """
        Save model to directory.

        Args:
            predictor: BPIC17SimplifiedModel instance with trained model.
            directory: Output directory path.
        """
        os.makedirs(directory, exist_ok=True)

        predictor.model.save(os.path.join(directory, cls.MODEL_FILE))
        joblib.dump(predictor.encoder, os.path.join(directory, cls.ENCODER_FILE))

        metadata = {
            "context_keys": predictor.context_keys,
            "max_seq_len": predictor.max_seq_len,
            "lstm_units": predictor.lstm_units,
            "hidden_units": predictor.hidden_units,
        }
        joblib.dump(metadata, os.path.join(directory, cls.METADATA_FILE))
        
        print(f"Saved model to {directory}")
        print(f"  - {cls.MODEL_FILE}")
        print(f"  - {cls.ENCODER_FILE}")
        print(f"  - {cls.METADATA_FILE}")

    @classmethod
    def load(cls, directory: str) -> Dict[str, Any]:
        """
        Load model bundle from directory.

        Args:
            directory: Directory containing saved model files.

        Returns:
            Dictionary with model and encoder.
        """
        import sys
        from pathlib import Path
        
        current_file = Path(__file__).resolve()
        simplified_dir = current_file.parent
        na_root = simplified_dir.parent.parent
        
        paths_to_add = [str(na_root)]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        model_path = os.path.join(directory, cls.MODEL_FILE)
        encoder_path = os.path.join(directory, cls.ENCODER_FILE)
        metadata_path = os.path.join(directory, cls.METADATA_FILE)
        
        if not os.path.exists(model_path):
            checkpoint_path = os.path.join(directory, "checkpoints", "best_model.keras")
            if os.path.exists(checkpoint_path):
                model_path = checkpoint_path
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or {checkpoint_path}")
        
        model = load_model(
            model_path,
            custom_objects={},
            safe_mode=False,
        )

        if os.path.exists(encoder_path):
            try:
                from bpic17_simplified.model import BPIC17SimplifiedEncoder
                import bpic17_simplified.model as simplified_model_module
            except ImportError:
                raise ImportError(
                    "Could not import BPIC17SimplifiedEncoder. Make sure Next-Activity-Prediction "
                    "is in Python path."
                )
            
            import sys
            if simplified_model_module and 'bpic17_simplified.model' not in sys.modules:
                sys.modules['bpic17_simplified.model'] = simplified_model_module
            
            encoder = joblib.load(encoder_path)
        else:
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")

        metadata = {}
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)

        class ModelWrapper:
            def __init__(self, keras_model, enc):
                self._model = keras_model
                self._encoder = enc

            def predict(self, activities, lifecycles, resources, context, top_k=5):
                X = self._encoder.transform_single(
                    activities, lifecycles, resources, context
                )
                preds = self._model.predict(X, verbose=0)

                activity_probs = preds[0][0]
                lifecycle_probs = preds[1][0]

                act_indices = np.argsort(activity_probs)[::-1][:top_k]
                activity_results = [
                    (self._encoder.target_activity_encoder.classes_[i], float(activity_probs[i]))
                    for i in act_indices
                    if i < len(self._encoder.target_activity_encoder.classes_)
                ]

                lc_indices = np.argsort(lifecycle_probs)[::-1][:top_k]
                lifecycle_results = [
                    (self._encoder.target_lifecycle_encoder.classes_[i], float(lifecycle_probs[i]))
                    for i in lc_indices
                    if i < len(self._encoder.target_lifecycle_encoder.classes_)
                ]

                return activity_results, lifecycle_results

        return {
            "model": ModelWrapper(model, encoder),
            "encoder": encoder,
            "metadata": metadata,
        }


