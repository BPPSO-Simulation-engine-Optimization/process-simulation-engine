"""
Processing Time Prediction - Prefix-based Cumulative Time

Predicts cumulative elapsed time given a prefix of events.
"""

from typing import List, Optional
import numpy as np
import joblib
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ProcessingTimePrediction:
    """
    Predicts cumulative time from case start given an event prefix.
    
    Usage:
        predictor = ProcessingTimePrediction()
        predictor.load("models/processing_time")
        
        # Predict cumulative time for a prefix of activities
        elapsed = predictor.predict(["Activity A", "Activity B", "Activity C"])
    """
    
    def __init__(self):
        self.model = None
        self.method: str = "lstm"
        self.activity_encoder = None
        self.num_activities: int = 0
        self.max_prefix_length: int = 50
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.fallback_mean: float = 0.0
        self.pad_idx: int = 0
        self._loaded = False

    def load(self, filepath: str) -> bool:
        """Load model from saved files."""
        config_path = f"{filepath}_config.joblib"
        
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}")
            return False
        
        config = joblib.load(config_path)
        
        self.method = config['method']
        self.activity_encoder = config['activity_encoder']
        self.num_activities = config['num_activities']
        self.max_prefix_length = config['max_prefix_length']
        self.y_mean = config['y_mean']
        self.y_std = config['y_std']
        self.fallback_mean = config['fallback_mean']
        
        # Get pad index
        try:
            self.pad_idx = int(self.activity_encoder.transform(["<PAD>"])[0])
        except:
            self.pad_idx = 0
        
        # Load model
        if self.method == "lstm":
            if not TF_AVAILABLE:
                print("TensorFlow required for LSTM model")
                return False
            model_path = f"{filepath}_model.keras"
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
            else:
                print(f"Model not found: {model_path}")
                return False
        else:
            model_path = f"{filepath}_model.joblib"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                print(f"Model not found: {model_path}")
                return False
        
        self._loaded = True
        print(f"Loaded {self.method.upper()} model from {filepath}")
        return True

    def _prepare_prefix(self, activities: List[str]) -> np.ndarray:
        """Encode and pad activity prefix."""
        encoded = []
        
        for act in activities:
            act_str = str(act)
            try:
                idx = int(self.activity_encoder.transform([act_str])[0])
            except:
                idx = self.pad_idx
            encoded.append(idx)
        
        # Truncate if too long
        if len(encoded) > self.max_prefix_length:
            encoded = encoded[-self.max_prefix_length:]
        
        prefix_len = len(encoded)
        
        # Pad at the beginning
        while len(encoded) < self.max_prefix_length:
            encoded = [self.pad_idx] + encoded
        
        return np.array(encoded, dtype=np.int32), prefix_len

    def predict(self, activities: List[str]) -> float:
        """
        Predict cumulative elapsed time for a prefix.
        
        Args:
            activities: List of activity names from case start
            
        Returns:
            Cumulative time in seconds from case start
        """
        if not self._loaded:
            return self.fallback_mean
        
        if len(activities) < 1:
            return 0.0
        
        prefix, prefix_len = self._prepare_prefix(activities)
        
        if self.method == "lstm":
            X = prefix.reshape(1, -1)
            pred_norm = self.model.predict(X, verbose=0)[0, 0]
            pred = np.exp(pred_norm * self.y_std + self.y_mean) - 1.0
        else:
            X = np.hstack([prefix, prefix_len]).reshape(1, -1)
            pred = self.model.predict(X)[0]
        
        return max(0.0, float(pred))

    def predict_batch(self, prefixes: List[List[str]]) -> List[float]:
        """Predict cumulative times for multiple prefixes."""
        if not self._loaded:
            return [self.fallback_mean] * len(prefixes)
        
        if self.method == "lstm":
            X = []
            for activities in prefixes:
                prefix, _ = self._prepare_prefix(activities)
                X.append(prefix)
            
            X = np.array(X)
            preds_norm = self.model.predict(X, verbose=0).flatten()
            preds = np.exp(preds_norm * self.y_std + self.y_mean) - 1.0
            return [max(0.0, float(p)) for p in preds]
        else:
            results = []
            for activities in prefixes:
                results.append(self.predict(activities))
            return results
