"""
Suffix prediction predictor class implementing NextActivityPredictor protocol.

This predictor predicts the entire remaining suffix (sequence of activities) for a case,
then returns activities one by one as needed by the simulation engine.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, List
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

from .suffix_model import load_suffix_model

logger = logging.getLogger(__name__)


class LSTMSuffixPredictor:
    """
    LSTM-based suffix predictor.
    
    Predicts the entire remaining sequence (suffix) for a case, then returns
    activities one by one as needed by the simulation engine.
    
    Implements NextActivityPredictor protocol for integration with simulation engine.
    """
    
    def __init__(
        self,
        model_path: str = "models/suffix_prediction_lstm",
        seed: Optional[int] = None
    ):
        """
        Initialize predictor by loading trained model.
        
        Args:
            model_path: Path to model directory or checkpoint file
            seed: Random seed for reproducibility
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
        
        self.model_path = Path(model_path)
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Load model and metadata
        logger.info(f"Loading suffix prediction model from {model_path}...")
        self.model, metadata = load_suffix_model(str(model_path))
        
        # Extract metadata
        self.activity_to_idx = {k: int(v) for k, v in metadata.get('activity_to_idx', {}).items()}
        self.idx_to_activity = {int(k): v for k, v in metadata.get('idx_to_activity', {}).items()}
        self.end_token_idx = metadata.get('end_token_idx', -1)
        self.prefix_length = metadata.get('prefix_length', 50)
        self.suffix_length = metadata.get('suffix_length', 30)
        
        # Cache predicted suffixes for each case
        # Format: {case_id: [activity1, activity2, ..., END]}
        self.predicted_suffixes: Dict[str, List[str]] = {}
        
        # Track which activity in the suffix we're currently on
        # Format: {case_id: current_index_in_suffix}
        self.suffix_positions: Dict[str, int] = {}
        
        logger.info(f"Loaded suffix prediction model")
        logger.info(f"Prefix length: {self.prefix_length}, Suffix length: {self.suffix_length}")
        logger.info(f"Vocabulary size: {len(self.activity_to_idx)}")
        logger.info(f"END token index: {self.end_token_idx}")
    
    def _pad_prefix(self, sequence: List[str]) -> List[int]:
        """Pad or truncate prefix sequence to model's prefix_length."""
        indices = [self.activity_to_idx.get(act, 0) for act in sequence]
        
        if len(indices) < self.prefix_length:
            indices = [0] * (self.prefix_length - len(indices)) + indices
        else:
            indices = indices[-self.prefix_length:]
        
        return indices
    
    def _predict_suffix(self, prefix: List[str]) -> List[str]:
        """
        Predict the suffix (remaining activities) given a prefix.
        
        Args:
            prefix: List of activity names representing the case history
            
        Returns:
            List of predicted activity names (suffix, may include END)
        """
        # Pad prefix
        prefix_padded = self._pad_prefix(prefix)
        X = np.array([prefix_padded], dtype=np.int32)
        
        # Predict suffix sequence
        suffix_probs = self.model.predict(X, verbose=0)
        # Shape: (1, suffix_length, vocab_size)
        
        # Convert probabilities to activity indices (greedy decoding)
        suffix_indices = np.argmax(suffix_probs[0], axis=1)  # Shape: (suffix_length,)
        
        # Convert indices to activity names
        suffix = []
        for idx in suffix_indices:
            activity = self.idx_to_activity.get(int(idx), "UNKNOWN")
            suffix.append(activity)
            
            # Stop if we hit END token (but include it in the suffix)
            if activity == "END" or idx == self.end_token_idx:
                break
        
        return suffix
    
    def predict(self, case_state) -> tuple[str, bool]:
        """
        Predict next activity for a case.
        
        This method maintains a predicted suffix for each case and returns
        activities from that suffix one by one. When the suffix is exhausted
        or contains END, the case is marked as ended.
        
        Args:
            case_state: CaseState object with activity_history and case_id
            
        Returns:
            Tuple of (next_activity_name, is_case_ended)
        """
        case_id = case_state.case_id
        activity_history = case_state.activity_history or []
        
        # Always start with A_Create Application for new cases
        if not activity_history:
            if "A_Create Application" in self.activity_to_idx:
                return "A_Create Application", False
            # Fallback if not in vocabulary
            if self.idx_to_activity:
                activities = [v for k, v in sorted(self.idx_to_activity.items()) if k != 0 and v != "END"]
                if activities:
                    first_activity = activities[0]
                    logger.warning(f"A_Create Application not in vocabulary, using {first_activity}")
                    return first_activity, False
            return "A_Create Application", False
        
        # Check if we need to predict a new suffix
        # This happens when:
        # 1. We don't have a suffix for this case yet
        # 2. We've exhausted the current suffix
        # 3. The case history has changed (new activities added)
        
        needs_new_prediction = False
        
        if case_id not in self.predicted_suffixes:
            needs_new_prediction = True
        elif case_id not in self.suffix_positions:
            needs_new_prediction = True
        elif self.suffix_positions[case_id] >= len(self.predicted_suffixes[case_id]):
            needs_new_prediction = True
        else:
            # Check if history has changed (new activities were added)
            # We should re-predict if the history length changed
            # For simplicity, we'll predict a new suffix when called
            # (The engine might call predict multiple times, so we'll cache)
            pass
        
        # Predict new suffix if needed
        if needs_new_prediction:
            try:
                suffix = self._predict_suffix(activity_history)
                self.predicted_suffixes[case_id] = suffix
                self.suffix_positions[case_id] = 0
                logger.debug(f"Predicted suffix for case {case_id}: {suffix[:5]}..." if len(suffix) > 5 else f"Predicted suffix for case {case_id}: {suffix}")
            except Exception as e:
                logger.warning(f"Suffix prediction failed for case {case_id}: {e}")
                # Fallback: end the case with last activity
                if activity_history:
                    return activity_history[-1], True
                return "FAILED", True
        
        # Get next activity from cached suffix
        if case_id not in self.predicted_suffixes or case_id not in self.suffix_positions:
            # Should not happen, but handle gracefully
            if activity_history:
                return activity_history[-1], True
            return "FAILED", True
        
        suffix = self.predicted_suffixes[case_id]
        pos = self.suffix_positions[case_id]
        
        # Check if we've reached the end of the suffix
        if pos >= len(suffix):
            # Suffix exhausted, end the case with last activity
            if case_id in self.predicted_suffixes:
                del self.predicted_suffixes[case_id]
            if case_id in self.suffix_positions:
                del self.suffix_positions[case_id]
            
            last_activity = activity_history[-1] if activity_history else "FAILED"
            return last_activity, True
        
        next_activity = suffix[pos]
        
        # Check if next activity is END token
        if next_activity == "END" or (isinstance(next_activity, (int, np.integer)) and int(next_activity) == self.end_token_idx):
            # Case ends - clear caches
            if case_id in self.predicted_suffixes:
                del self.predicted_suffixes[case_id]
            if case_id in self.suffix_positions:
                del self.suffix_positions[case_id]
            
            # Return last logged activity, not END
            last_activity = activity_history[-1] if activity_history else "FAILED"
            return last_activity, True
        
        # Advance position for next call
        self.suffix_positions[case_id] = pos + 1
        
        return next_activity, False
    
    def reset_case(self, case_id: str):
        """Reset suffix cache for a case."""
        if case_id in self.predicted_suffixes:
            del self.predicted_suffixes[case_id]
        if case_id in self.suffix_positions:
            del self.suffix_positions[case_id]
    
    def clear(self):
        """Clear all cached suffixes."""
        self.predicted_suffixes.clear()
        self.suffix_positions.clear()

