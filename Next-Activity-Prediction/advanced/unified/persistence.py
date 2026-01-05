"""
Persistence utilities for unified model.
"""

import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Dict, Any

# Handle different Keras versions for serialization
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


@_register(package="UnifiedLayers")
def expand_and_cast(x):
    """Expand dimensions and cast to float32 for duration input."""
    return tf.cast(tf.expand_dims(x, axis=-1), tf.float32)


class UnifiedModelPersistence:
    """Saves and loads unified model bundles."""

    MODEL_FILE = "model.keras"
    ENCODER_FILE = "encoder.pkl"
    METADATA_FILE = "metadata.pkl"

    @classmethod
    def save(cls, predictor, directory: str):
        """
        Save unified predictor to directory.

        Args:
            predictor: UnifiedPredictor instance with trained model.
            directory: Output directory path.
        """
        os.makedirs(directory, exist_ok=True)

        # Save Keras model
        predictor.model.save(os.path.join(directory, cls.MODEL_FILE))

        # Save encoder
        joblib.dump(predictor.encoder, os.path.join(directory, cls.ENCODER_FILE))

        # Save metadata
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
        Load unified model bundle from directory.

        Args:
            directory: Directory containing saved model files.

        Returns:
            Dictionary with model and encoder.
        """
        import numpy as np
        import sys
        from pathlib import Path
        
        # Ensure unified module is importable before unpickling
        # The encoder pickle references unified.model.UnifiedEncoder
        current_file = Path(__file__).resolve()
        unified_dir = current_file.parent
        advanced_dir = unified_dir.parent
        na_root = advanced_dir.parent
        
        # Add paths to sys.path if not already there
        paths_to_add = [
            str(na_root),  # For "advanced.unified"
            str(advanced_dir),  # For "unified" (if saved that way)
        ]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        model_path = os.path.join(directory, cls.MODEL_FILE)
        encoder_path = os.path.join(directory, cls.ENCODER_FILE)
        metadata_path = os.path.join(directory, cls.METADATA_FILE)
        
        # Check if files exist
        if not os.path.exists(model_path):
            # Try checkpoint path
            checkpoint_path = os.path.join(directory, "checkpoints", "best_model.keras")
            if os.path.exists(checkpoint_path):
                model_path = checkpoint_path
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or {checkpoint_path}")
        
        # Load Keras model with custom objects
        model = load_model(
            model_path,
            custom_objects={"expand_and_cast": expand_and_cast},
            safe_mode=False,
        )

        # Load encoder - ensure module is importable
        if os.path.exists(encoder_path):
            # Import the encoder class to make it available for unpickling
            # Try both import paths
            encoder_imported = False
            unified_module = None
            unified_model_module = None
            
            try:
                from advanced.unified.model import UnifiedEncoder
                import advanced.unified as unified_module
                import advanced.unified.model as unified_model_module
                encoder_imported = True
            except ImportError:
                try:
                    from unified.model import UnifiedEncoder
                    import unified as unified_module
                    import unified.model as unified_model_module
                    encoder_imported = True
                except ImportError:
                    pass
            
            if not encoder_imported:
                raise ImportError(
                    "Could not import UnifiedEncoder. Make sure Next-Activity-Prediction/advanced "
                    "is in Python path."
                )
            
            # Create module aliases in sys.modules to handle different import paths
            # This allows pickle to find the classes regardless of how they were saved
            import sys
            if unified_module and 'unified' not in sys.modules:
                sys.modules['unified'] = unified_module
            if unified_model_module and 'unified.model' not in sys.modules:
                sys.modules['unified.model'] = unified_model_module
            
            # Load encoder - joblib will use the module aliases
            encoder = joblib.load(encoder_path)
        else:
            raise FileNotFoundError(f"Encoder not found at {encoder_path}. Run save() first.")

        # Load metadata (optional)
        metadata = {}
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)

        # Create a wrapper for prediction that matches the expected interface
        class ModelWrapper:
            def __init__(self, keras_model, enc):
                self._model = keras_model
                self._encoder = enc

            def predict(self, activities, lifecycles, resources, durations, context, top_k=5):
                X = self._encoder.transform_single(
                    activities, lifecycles, resources, durations, context
                )
                preds = self._model.predict(X, verbose=0)

                activity_probs = preds[0][0]
                lifecycle_probs = preds[1][0]

                # Get top-k activities
                act_indices = np.argsort(activity_probs)[::-1][:top_k]
                activity_results = [
                    (self._encoder.target_activity_encoder.classes_[i], float(activity_probs[i]))
                    for i in act_indices
                    if i < len(self._encoder.target_activity_encoder.classes_)
                ]

                # Get top-k lifecycles
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

