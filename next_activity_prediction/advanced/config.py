"""
Configuration for next activity prediction model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class NextActivityConfig:
    """Configuration for LSTM next activity prediction model."""
    
    # Model architecture
    sequence_length: int = 50
    embedding_dim: int = 128
    lstm_units: int = 256
    lstm_layers: int = 2
    dropout_rate: float = 0.3
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Paths
    model_dir: str = "models/next_activity_lstm"
    event_log_path: Optional[str] = None
    
    # Data preprocessing
    min_case_length: int = 2
    max_case_length: int = 200
    
    # Class weighting (for imbalanced classes like END token)
    use_class_weights: bool = True  # Whether to use class weights
    end_token_weight: Optional[float] = None  # Manual weight for END token (None = auto-calculate)
    class_weight_method: str = "balanced"  # "balanced" (sklearn style) or "inverse_freq" or "custom"
    
    # Position-based sample weighting (later positions more likely to be END)
    use_position_weights: bool = False  # Whether to weight samples by position in case
    position_weight_power: float = 1.5  # Power for position weighting (higher = more emphasis on later positions)
    
    def __post_init__(self):
        """Ensure model_dir is a Path object."""
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

