"""
Configuration for next activity prediction model with one-hot encoding.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class NextActivityConfigOneHot:
    """Configuration for LSTM next activity prediction model with one-hot encoding."""
    
    sequence_length: int = 50
    lstm_units: int = 256
    lstm_layers: int = 2
    dropout_rate: float = 0.3
    
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    model_dir: str = "models/next_activity_lstm_onehot"
    event_log_path: Optional[str] = None
    
    min_case_length: int = 2
    max_case_length: int = 200
    
    use_class_weights: bool = True
    end_token_weight: Optional[float] = None
    class_weight_method: str = "balanced"
    
    use_position_weights: bool = False
    position_weight_power: float = 1.5
    
    def __post_init__(self):
        """Ensure model_dir is a Path object."""
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

