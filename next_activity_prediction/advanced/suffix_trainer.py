"""
Training pipeline for suffix prediction model.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict
import json
import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .suffix_model import build_suffix_model
from .suffix_data_preprocessing import preprocess_event_log_for_suffix

logger = logging.getLogger(__name__)


def train_suffix_model(
    event_log_path: str,
    model_dir: str = "models/suffix_prediction_lstm",
    prefix_length: int = 50,
    suffix_length: int = 30,
    embedding_dim: int = 128,
    encoder_lstm_units: int = 256,
    decoder_lstm_units: int = 256,
    encoder_lstm_layers: int = 2,
    decoder_lstm_layers: int = 2,
    dropout_rate: float = 0.3,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 50,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    min_case_length: int = 2,
    max_case_length: int = 200,
    min_prefix_length: int = 1,
    random_seed: int = 42
) -> Tuple[any, Dict]:
    """
    Train suffix prediction model.
    
    Args:
        event_log_path: Path to event log file
        model_dir: Directory to save model
        prefix_length: Input prefix sequence length
        suffix_length: Output suffix sequence length
        embedding_dim: Embedding dimension
        encoder_lstm_units: Number of LSTM units in encoder
        decoder_lstm_units: Number of LSTM units in decoder
        encoder_lstm_layers: Number of encoder LSTM layers
        decoder_lstm_layers: Number of decoder LSTM layers
        dropout_rate: Dropout rate
        batch_size: Training batch size
        learning_rate: Learning rate
        epochs: Maximum number of epochs
        validation_split: Fraction of data for validation
        early_stopping_patience: Early stopping patience
        min_case_length: Minimum case length
        max_case_length: Maximum case length
        min_prefix_length: Minimum prefix length
        random_seed: Random seed
        
    Returns:
        Tuple of (trained_model, metadata_dict)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
    
    # Setup paths
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess data
    logger.info("Preprocessing event log...")
    X_train, y_train, X_val, y_val, activity_to_idx, idx_to_activity = preprocess_event_log_for_suffix(
        event_log_path,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        min_case_length=min_case_length,
        max_case_length=max_case_length,
        min_prefix_length=min_prefix_length,
        validation_split=validation_split,
        random_seed=random_seed
    )
    
    vocab_size = len(activity_to_idx)
    end_token_idx = activity_to_idx.get("END", -1)
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"END token index: {end_token_idx}")
    
    # Build model
    logger.info("Building model...")
    model = build_suffix_model(
        vocab_size=vocab_size,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        embedding_dim=embedding_dim,
        encoder_lstm_units=encoder_lstm_units,
        decoder_lstm_units=decoder_lstm_units,
        encoder_lstm_layers=encoder_lstm_layers,
        decoder_lstm_layers=decoder_lstm_layers,
        dropout_rate=dropout_rate
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    logger.info("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = model_dir / "model.keras"
    model.save(str(final_model_path))
    logger.info(f"Saved model to {final_model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "suffix_prediction_lstm",
        "vocab_size": vocab_size,
        "prefix_length": prefix_length,
        "suffix_length": suffix_length,
        "embedding_dim": embedding_dim,
        "encoder_lstm_units": encoder_lstm_units,
        "decoder_lstm_units": decoder_lstm_units,
        "encoder_lstm_layers": encoder_lstm_layers,
        "decoder_lstm_layers": decoder_lstm_layers,
        "dropout_rate": dropout_rate,
        "activity_to_idx": activity_to_idx,
        "idx_to_activity": {int(k): v for k, v in idx_to_activity.items()},  # Ensure keys are int
        "end_token_idx": end_token_idx,
        "training_history": {
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_train_acc": float(history.history['sparse_categorical_accuracy'][-1]),
            "final_val_acc": float(history.history['val_sparse_categorical_accuracy'][-1]),
            "epochs_trained": len(history.history['loss'])
        }
    }
    
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    return model, metadata

