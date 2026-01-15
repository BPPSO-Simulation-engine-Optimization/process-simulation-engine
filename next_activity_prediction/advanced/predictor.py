"""
Predictor class implementing NextActivityPredictor protocol for simulation integration.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False

from .model import load_model
from .data_preprocessing import pad_sequence

logger = logging.getLogger(__name__)


class LSTMNextActivityPredictor:
    """
    LSTM-based next activity predictor.
    
    Implements NextActivityPredictor protocol for integration with simulation engine.
    Maintains case history and predicts next activity with END detection.
    """
    
    def __init__(
        self,
        model_path: str = "models/next_activity_lstm",
        seed: Optional[int] = None
    ):
        """
        Initialize predictor by loading trained model.
        
        Args:
            model_path: Path to model directory
            seed: Random seed for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
        
        self.model_path = Path(model_path)
        
        if seed is not None:
            np.random.seed(seed)
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
            except:
                pass
        
        self.model, metadata = load_model(str(self.model_path))
        self.sequence_length = metadata.get('sequence_length', 50)
        self.activity_to_idx = {k: int(v) for k, v in metadata.get('activity_to_idx', {}).items()}
        self.idx_to_activity = {int(k): v for k, v in metadata.get('idx_to_activity', {}).items()}
        self.end_token_idx = metadata.get('end_token_idx', self.activity_to_idx.get('END', -1))
        
        if not self.activity_to_idx or not self.idx_to_activity:
            raise ValueError("Model metadata missing vocabulary mappings")
        
        if 'END' not in self.activity_to_idx:
            logger.warning("END token not found in vocabulary - case ending detection may not work correctly")
        
        self.case_histories: Dict[str, list] = {}
        
        logger.info(f"Loaded LSTMNextActivityPredictor from {model_path}")
        logger.info(f"Vocabulary size: {len(self.activity_to_idx)} (including END)")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"END token index: {self.end_token_idx}")
    
    def predict(self, case_state) -> tuple[str, bool]:
        """
        Predict next activity for a case.
        
        Args:
            case_state: CaseState object with activity_history and case_id
            
        Returns:
            Tuple of (next_activity_name, is_case_ended)
        """
        case_id = case_state.case_id
        activity_history = case_state.activity_history or []
        
        # Use the activity history directly from case_state (it's always up-to-date)
        # We keep case_histories for reference but use the current state
        if case_id not in self.case_histories:
            self.case_histories[case_id] = []
        
        # Sync internal history with case_state for tracking
        history = self.case_histories[case_id]
        if len(activity_history) > len(history):
            history.extend(activity_history[len(history):])
        
        # Use activity_history directly for prediction (it's the source of truth)
        history_for_prediction = activity_history if activity_history else history
        
        if not history_for_prediction:
            # Always start with A_Create Application for new cases
            if "A_Create Application" in self.activity_to_idx:
                return "A_Create Application", False
            # Fallback if not in vocabulary
            if self.idx_to_activity:
                activities = [v for k, v in sorted(self.idx_to_activity.items()) if k != 0]
                if activities:
                    first_activity = activities[0]
                    logger.warning(f"A_Create Application not in vocabulary, using {first_activity}")
                    return first_activity, False
            return "A_Create Application", False
        
        try:
            sequence_padded = pad_sequence(history_for_prediction, self.activity_to_idx, self.sequence_length)
            X = np.array([sequence_padded], dtype=np.int32)
            
            activity_probs = self.model.predict(X, verbose=0)
            probs = activity_probs[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probs)[-3:][::-1]
            top_3_probs = probs[top_3_indices]
            
            # Normalize probabilities for sampling
            top_3_probs_normalized = top_3_probs / np.sum(top_3_probs)
            
            # Sample from top 3 based on probabilities
            next_activity_idx = np.random.choice(top_3_indices, p=top_3_probs_normalized)
            next_activity = self.idx_to_activity.get(next_activity_idx, history_for_prediction[-1] if history_for_prediction else "A_Complete")
            
            # Check if predicted activity is END token
            is_end = (next_activity == "END" or next_activity_idx == self.end_token_idx)
            
            if is_end:
                # Case ends - return the last logged activity (not END token or A_Complete)
                # Clear history
                if case_id in self.case_histories:
                    del self.case_histories[case_id]
                # Return the last activity from history as the final activity
                last_activity = history_for_prediction[-1] if history_for_prediction else "END"
                return last_activity, True
            
            return next_activity, False
            
        except Exception as e:
            logger.warning(f"Prediction failed for case {case_id}: {e}")
            # On error, end the case with the last logged activity
            if history_for_prediction:
                last_activity = history_for_prediction[-1]
                return last_activity, True
            # If no history at all, use A_Complete as fallback
            return "FAILED", True
    
    def reset_case(self, case_id: str):
        """Reset history for a case."""
        if case_id in self.case_histories:
            del self.case_histories[case_id]
    
    def clear(self):
        """Clear all case histories."""
        self.case_histories.clear()

