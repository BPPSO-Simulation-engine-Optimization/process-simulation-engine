"""
Training script for next activity prediction model.
"""

import logging
import json
from pathlib import Path
from typing import Optional
import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False

from .config import NextActivityConfig
from .data_preprocessing import (
    prepare_training_data, 
    extract_case_sequences, 
    filter_lifecycles, 
    load_event_log, 
    create_vocabulary
)
from .model import build_model
from .utils import calculate_class_weights, calculate_position_weights, combine_sample_weights

logger = logging.getLogger(__name__)


def train_model(
    config: NextActivityConfig,
    log_path: Optional[str] = None
) -> dict:
    """
    Train next activity prediction model.
    
    Args:
        config: Configuration object
        log_path: Path to event log (overrides config.event_log_path)
        
    Returns:
        Training history dictionary
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
    
    log_path = log_path or config.event_log_path
    if not log_path:
        raise ValueError("Event log path must be provided in config or as argument")
    
    logger.info("Starting model training...")
    logger.info(f"Event log: {log_path}")
    logger.info(f"Model directory: {config.model_dir}")
    
    # Load and preprocess data
    df = load_event_log(log_path)
    df = filter_lifecycles(df)
    sequences = extract_case_sequences(df, config.min_case_length, config.max_case_length)
    activity_to_idx, idx_to_activity = create_vocabulary(sequences)
    
    vocab_size = len(activity_to_idx)
    end_token_idx = activity_to_idx.get("END", -1)
    logger.info(f"Vocabulary size: {vocab_size} (including END token)")
    
    # Prepare training data with position info if needed
    need_position_info = config.use_position_weights
    X, y_activity, positions = prepare_training_data(
        sequences, 
        activity_to_idx, 
        config.sequence_length,
        return_position_info=need_position_info
    )
    
    # Split into train/val
    np.random.seed(42)  # Use fixed seed for reproducibility
    split_idx = int(len(X) * (1 - config.validation_split))
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    y_activity_train = y_activity[train_indices]
    X_val = X[val_indices]
    y_activity_val = y_activity[val_indices]
    
    # Get position info for training set if needed
    positions_train = positions[train_indices] if positions is not None else None
    positions_val = positions[val_indices] if positions is not None else None
    
    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    model = build_model(
        vocab_size=vocab_size,
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        lstm_units=config.lstm_units,
        lstm_layers=config.lstm_layers,
        dropout_rate=config.dropout_rate
    )
    
    logger.info("Model architecture:")
    model.summary()
    
    config.model_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(config.model_dir / "checkpoints" / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    (config.model_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Calculate class weights if enabled
    class_weight_dict = None
    if config.use_class_weights:
        class_weight_dict = calculate_class_weights(
            y=y_activity_train,
            method=config.class_weight_method,
            end_token_idx=end_token_idx,
            end_token_weight=config.end_token_weight,
            vocab_size=vocab_size
        )
    
    # Calculate sample weights if position weighting is enabled
    sample_weight_train = None
    if config.use_position_weights and positions_train is not None:
        position_weights = calculate_position_weights(
            positions_train,
            power=config.position_weight_power
        )
        
        # Combine with class weights if both are enabled
        if class_weight_dict is not None:
            sample_weight_train = combine_sample_weights(
                class_weight_dict,
                position_weights,
                y_activity_train
            )
        else:
            sample_weight_train = position_weights
        
        # Note: Keras fit() uses class_weight OR sample_weight, not both
        # If we have sample_weight, we should not use class_weight
        if sample_weight_train is not None:
            class_weight_dict = None
            logger.info("Using sample weights (combining class and position weights)")
    
    history = model.fit(
        X_train,
        y_activity_train,
        validation_data=(X_val, y_activity_val),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        sample_weight=sample_weight_train,
        verbose=1
    )
    
    final_model_path = config.model_dir / "model.keras"
    model.save(str(final_model_path))
    logger.info(f"Saved final model to {final_model_path}")
    
    end_token_idx = activity_to_idx.get("END", -1)
    
    metadata = {
        'vocab_size': vocab_size,
        'sequence_length': config.sequence_length,
        'embedding_dim': config.embedding_dim,
        'lstm_units': config.lstm_units,
        'lstm_layers': config.lstm_layers,
        'dropout_rate': config.dropout_rate,
        'activity_to_idx': activity_to_idx,
        'idx_to_activity': {int(k): v for k, v in idx_to_activity.items()},
        'end_token_idx': end_token_idx
    }
    
    metadata_path = config.model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    train_acc = history.history['sparse_categorical_accuracy'][-1]
    val_acc = history.history['val_sparse_categorical_accuracy'][-1]
    
    logger.info("Training completed!")
    logger.info(f"Final train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
    logger.info(f"Final train accuracy: {train_acc:.4f}, val accuracy: {val_acc:.4f}")
    
    return history.history


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python -m next_activity_prediction.trainer <event_log_path>")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    config = NextActivityConfig(
        event_log_path=log_path,
        epochs=50,
        batch_size=64
    )
    
    train_model(config)

