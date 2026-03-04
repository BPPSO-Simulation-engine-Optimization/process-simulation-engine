from dataclasses import dataclass
from pathlib import Path


@dataclass
class DualPredictionConfig:
    sequence_length: int = 50
    embedding_dim: int = 96
    lstm_units: int = 192
    lstm_layers: int = 2
    dropout_rate: float = 0.25

    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 40
    validation_split: float = 0.2
    early_stopping_patience: int = 8

    min_case_length: int = 2
    max_case_length: int = 200

    random_seed: int = 42
    model_root: str = "next_activity_prediction_lifecycle_dual/models"

    def __post_init__(self) -> None:
        if isinstance(self.model_root, str):
            self.model_root = Path(self.model_root)
