"""
ProcessTransformer predictor for simulation engine integration.

Implements NextActivityPredictor protocol for DESEngine.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default HuggingFace repository for auto-download
DEFAULT_HF_REPO = "lgk03/bpic17-process-transformer_v1"

# Counter for logging progress
_PREDICTION_COUNT = 0


class ProcessTransformerPredictor:
    """
    Next activity predictor using ProcessTransformer model.

    Uses a trained Transformer model to predict the next activity in a process trace.
    Provides probabilistic predictions for stochastic simulation.

    If model files are not found locally, automatically downloads from HuggingFace Hub.
    """

    END_ACTIVITIES = {"A_Cancelled", "A_Complete", "END", "<END>"}
    START_ACTIVITY = "A_Create Application"

    def __init__(
        self,
        model_path: str = "models/process_transformer",
        hf_repo_id: Optional[str] = None,
        temperature: float = 2.0,
        repetition_penalty: float = 1.0,
        repetition_window: int = 3,
        seed: int = 42,
        auto_download: bool = True,
        end_token_penalty: float = 1.0,
    ):
        """
        Initialize the ProcessTransformer predictor.

        Args:
            model_path: Path to model directory containing model.keras, vocab.json, config.json.
            hf_repo_id: HuggingFace repository ID for auto-download (default: lgk03/bpic17-process-transformer_v1).
            temperature: Sampling temperature (1.0 = neutral, <1 = more deterministic).
            repetition_penalty: Multiplicative penalty for recently seen activities.
            repetition_window: How many recent activities to apply penalty to.
            seed: Random seed for reproducibility.
            auto_download: If True, automatically download model from HuggingFace if not found locally.
            end_token_penalty: Divisor for end activity probabilities (1.0 = no change, >1.0 = suppress ending).
        """
        self.model_path = Path(model_path)
        self.hf_repo_id = hf_repo_id or DEFAULT_HF_REPO
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window
        self.end_token_penalty = end_token_penalty
        self.rng = random.Random(seed)
        self.auto_download = auto_download

        self._load_model()

        logger.info(
            f"ProcessTransformerPredictor initialized: "
            f"vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len}"
        )

    def _ensure_model_available(self):
        """Check if model exists locally, download from HuggingFace if not."""
        model_file = self.model_path / "model.keras"
        vocab_file = self.model_path / "vocab.json"
        config_file = self.model_path / "config.json"

        required_files = [model_file, vocab_file, config_file]
        missing_files = [f for f in required_files if not f.exists()]

        if not missing_files:
            return  # All files present

        if not self.auto_download:
            raise FileNotFoundError(
                f"Model files not found: {missing_files}. "
                f"Set auto_download=True to download from HuggingFace, "
                f"or manually place files in {self.model_path}"
            )

        logger.info(f"Model not found locally, downloading from HuggingFace: {self.hf_repo_id}")
        try:
            from .downloader import download_model
            download_model(
                repo_id=self.hf_repo_id,
                cache_dir=str(self.model_path),
            )
            logger.info(f"Model downloaded to {self.model_path}")
        except ImportError:
            raise FileNotFoundError(
                f"Model files not found and could not import downloader. "
                f"Install huggingface_hub: pip install huggingface_hub"
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download model from HuggingFace ({self.hf_repo_id}): {e}"
            )

    def _load_model(self):
        """Load model, vocabulary, and config from disk."""
        # Ensure model is available (download if needed)
        self._ensure_model_available()

        model_file = self.model_path / "model.keras"
        vocab_file = self.model_path / "vocab.json"
        config_file = self.model_path / "config.json"

        # Lazy import TensorFlow / Keras
        try:
            import tf_keras as keras
            import os
            # Set internal flag to ensure tf_keras uses legacy format if needed
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
        except ImportError:
            import tensorflow as tf
            keras = tf.keras

        self.model = keras.models.load_model(str(model_file))

        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)

        self.activity_to_idx = vocab_data['activity_to_idx']
        self.idx_to_activity = {int(k): v for k, v in vocab_data['idx_to_activity'].items()}
        self.vocab_size = vocab_data['vocab_size']

        with open(config_file, 'r') as f:
            config = json.load(f)

        self.max_seq_len = config['max_seq_len']
        self.pad_idx = self.activity_to_idx.get('<PAD>', 0)
        self.start_idx = self.activity_to_idx.get('<START>', 1)
        self.end_idx = self.activity_to_idx.get('<END>', 2)

    def predict(self, case_state: Any) -> Tuple[str, bool]:
        """
        Predict the next activity for a case.

        Implements the NextActivityPredictor protocol.

        Args:
            case_state: CaseState object with activity_history attribute.

        Returns:
            Tuple of (next_activity_name, is_case_ended).
        """
        import numpy as np

        if not case_state.activity_history:
            return self.START_ACTIVITY, False

        current = case_state.activity_history[-1]
        if current in self.END_ACTIVITIES:
            return current, True

        prefix_indices = self._encode_prefix(case_state.activity_history)
        probs = self._get_probabilities(prefix_indices)
        probs = self._apply_repetition_penalty(probs, case_state.activity_history)
        probs = self._apply_end_penalty(probs)

        if self.temperature != 1.0:
            probs = np.power(probs, 1.0 / self.temperature)
            probs = probs / probs.sum()

        next_idx = self.rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0]
        next_activity = self.idx_to_activity.get(next_idx, self.START_ACTIVITY)

        is_end = next_activity in self.END_ACTIVITIES
        return next_activity, is_end

    def _encode_prefix(self, history: List[str]) -> List[int]:
        """Convert activity history to token indices."""
        indices = [self.start_idx]
        for activity in history:
            if activity in self.activity_to_idx:
                indices.append(self.activity_to_idx[activity])
        return indices

    def _get_probabilities(self, prefix_indices: List[int]):
        """Get probability distribution from model."""
        import numpy as np

        if len(prefix_indices) < self.max_seq_len:
            padded = [self.pad_idx] * (self.max_seq_len - len(prefix_indices)) + prefix_indices
        else:
            padded = prefix_indices[-self.max_seq_len:]

        input_array = np.array([padded])
        
        # Optimize: Use __call__ instead of predict() for single-sample inference
        # predict() has significant overhead for small batches
        # training=False is important for dropout/batchnorm layers
        probs = self.model(input_array, training=False).numpy()[0]
        
        # Log progress every 100 predictions to monitor hang
        global _PREDICTION_COUNT
        _PREDICTION_COUNT += 1
        if _PREDICTION_COUNT % 100 == 0:
            logger.info(f"ProcessTransformer: Generated {_PREDICTION_COUNT} predictions")

        return probs

    def _apply_repetition_penalty(self, probs, history: List[str]):
        """Reduce probability of recently seen activities."""
        import numpy as np

        probs = probs.copy()
        recent = history[-self.repetition_window:] if len(history) >= self.repetition_window else history

        for activity in recent:
            if activity in self.activity_to_idx:
                idx = self.activity_to_idx[activity]
                probs[idx] *= self.repetition_penalty

        probs[self.pad_idx] = 0
        probs[self.start_idx] = 0

        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones_like(probs)
            probs[self.pad_idx] = 0
            probs[self.start_idx] = 0
            probs = probs / probs.sum()

        return probs

    def get_distribution(self, case_state: Any) -> Dict[str, float]:
        """Get full probability distribution for next activity (for debugging)."""
        if not case_state.activity_history:
            prefix_indices = [self.start_idx]
        else:
            prefix_indices = self._encode_prefix(case_state.activity_history)

        probs = self._get_probabilities(prefix_indices)

        return {
            self.idx_to_activity[i]: float(p)
            for i, p in enumerate(probs)
            if p > 0.001
        }

    def _apply_end_penalty(self, probs):
        """Apply penalty to end activities to encourage longer traces."""
        import numpy as np
        
        if self.end_token_penalty == 1.0:
            return probs
            
        probs = probs.copy()
        for end_act in self.END_ACTIVITIES:
            if end_act in self.activity_to_idx:
                idx = self.activity_to_idx[end_act]
                probs[idx] /= self.end_token_penalty
        
        # Normalize
        total = np.sum(probs)
        if total > 0:
            probs /= total
            
        return probs
