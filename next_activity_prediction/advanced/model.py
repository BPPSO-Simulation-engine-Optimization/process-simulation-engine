"""
LSTM model architecture for next activity prediction.
"""

import logging
from typing import Tuple
import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
        from keras import layers, Model
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


def build_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = 128,
    lstm_units: int = 256,
    lstm_layers: int = 2,
    dropout_rate: float = 0.3
) -> Model:
    """
    Build LSTM model for next activity prediction.
    
    Model architecture:
    - Embedding layer for activity tokens
    - Stacked LSTM layers
    - Single output: Activity prediction (softmax over vocabulary, including END token)
    
    Args:
        vocab_size: Size of activity vocabulary (including PAD token and END token)
        sequence_length: Input sequence length
        embedding_dim: Embedding dimension
        lstm_units: Number of LSTM units per layer
        lstm_layers: Number of LSTM layers
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
    
    inputs = layers.Input(shape=(sequence_length,), name='sequence_input')
    
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=sequence_length,
        mask_zero=True,
        name='activity_embedding'
    )(inputs)
    
    x = embedding
    
    for i in range(lstm_layers):
        return_sequences = (i < lstm_layers - 1)
        x = layers.LSTM(
            lstm_units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f'lstm_{i+1}'
        )(x)
    
    x = layers.Dropout(dropout_rate, name='final_dropout')(x)
    
    activity_output = layers.Dense(
        vocab_size,
        activation='softmax',
        name='activity_prediction'
    )(x)
    
    model = Model(inputs=inputs, outputs=activity_output, name='next_activity_lstm')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    return model


def load_model(model_path: str) -> Tuple[Model, dict]:
    """
    Load trained model and metadata.
    
    Args:
        model_path: Path to model directory containing model.keras and metadata,
                    or direct path to a checkpoint file (e.g., checkpoints/best_model.keras)
        
    Returns:
        Tuple of (model, metadata_dict)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
    
    from pathlib import Path
    import json
    
    model_path_obj = Path(model_path)
    
    # Check if it's a direct path to a model file
    if model_path_obj.is_file() and model_path_obj.suffix == '.keras':
        model_file = model_path_obj
        # Metadata should be in the parent directory (model_dir)
        model_dir = model_path_obj.parent.parent
        metadata_file = model_dir / "metadata.json"
    else:
        # It's a directory path
        model_dir = model_path_obj
        model_file = model_dir / "model.keras"
        metadata_file = model_dir / "metadata.json"
        
        # If model.keras doesn't exist, try checkpoints/best_model.keras
        if not model_file.exists():
            checkpoint_file = model_dir / "checkpoints" / "best_model.keras"
            if checkpoint_file.exists():
                model_file = checkpoint_file
            else:
                raise FileNotFoundError(
                    f"Model file not found. Tried:\n"
                    f"  - {model_file}\n"
                    f"  - {checkpoint_file}"
                )
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = keras.models.load_model(str(model_file))
    
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        logger.warning(f"Metadata file not found: {metadata_file}")
    
    return model, metadata

