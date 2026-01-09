"""
Sequence-to-sequence LSTM model architecture for suffix prediction.
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


def build_suffix_model(
    vocab_size: int,
    prefix_length: int,
    suffix_length: int,
    embedding_dim: int = 128,
    encoder_lstm_units: int = 256,
    decoder_lstm_units: int = 256,
    encoder_lstm_layers: int = 2,
    decoder_lstm_layers: int = 2,
    dropout_rate: float = 0.3
) -> Model:
    """
    Build sequence-to-sequence LSTM model for suffix prediction.
    
    Model architecture:
    - Encoder: Embedding + Stacked LSTM layers (processes prefix)
    - Decoder: Stacked LSTM layers + Dense output (generates suffix)
    - Output: Sequence of activities (including END token)
    
    Args:
        vocab_size: Size of activity vocabulary (including PAD token and END token)
        prefix_length: Input prefix sequence length
        suffix_length: Output suffix sequence length
        embedding_dim: Embedding dimension
        encoder_lstm_units: Number of LSTM units in encoder layers
        decoder_lstm_units: Number of LSTM units in decoder layers
        encoder_lstm_layers: Number of encoder LSTM layers
        decoder_lstm_layers: Number of decoder LSTM layers
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
    
    # Encoder: Process prefix
    prefix_input = layers.Input(shape=(prefix_length,), name='prefix_input')
    
    # Embedding layer
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=prefix_length,
        mask_zero=True,
        name='activity_embedding'
    )(prefix_input)
    
    # Encoder LSTM layers
    x = embedding
    encoder_outputs = []
    encoder_states = []
    
    for i in range(encoder_lstm_layers):
        return_sequences = (i < encoder_lstm_layers - 1)
        lstm = layers.LSTM(
            encoder_lstm_units,
            return_sequences=return_sequences,
            return_state=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f'encoder_lstm_{i+1}'
        )
        
        if i == 0:
            x, *states = lstm(x)
        else:
            x, *states = lstm(x, initial_state=encoder_states[-2:] if i > 0 else None)
        
        encoder_states.extend(states)
    
    # Decoder: Generate suffix
    # We use a simple approach: repeat the encoder's final output for each decoder timestep
    # This works well for training and is simpler than full sequence-to-sequence with attention
    
    # Repeat encoder output for decoder steps
    decoder_input = layers.RepeatVector(suffix_length, name='repeat_vector')(x)
    
    # Decoder LSTM layers
    decoder_lstm_output = decoder_input
    
    for i in range(decoder_lstm_layers):
        return_sequences = True  # Always return sequences in decoder
        lstm = layers.LSTM(
            decoder_lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f'decoder_lstm_{i+1}'
        )
        decoder_lstm_output = lstm(decoder_lstm_output)
    
    # Output layer: Dense for each timestep
    decoder_output = layers.TimeDistributed(
        layers.Dense(
            vocab_size,
            activation='softmax',
            name='suffix_prediction'
        ),
        name='time_distributed_output'
    )(decoder_lstm_output)
    
    model = Model(
        inputs=prefix_input, 
        outputs=decoder_output, 
        name='suffix_prediction_lstm'
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    return model


def load_suffix_model(model_path: str) -> Tuple[Model, dict]:
    """
    Load trained suffix prediction model and metadata.
    
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

