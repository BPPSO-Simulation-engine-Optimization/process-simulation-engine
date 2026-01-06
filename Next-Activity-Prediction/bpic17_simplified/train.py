"""
Training script for BPIC17 simplified next activity prediction model.
"""

import os
import argparse
import logging
from pathlib import Path

from .data_preprocessing import load_and_filter_bpic17, add_end_tokens
from .data_generator import BPIC17SimplifiedDataGenerator
from .model import BPIC17SimplifiedModel
from .persistence import BPIC17SimplifiedPersistence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train BPIC17 simplified next activity predictor")
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to BPIC 2017 XES file (default: Dataset/BPI Challenge 2017.xes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/bpic17_simplified",
        help="Output directory for saved model",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=15,
        help="Maximum history length for sequences",
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        default=256,
        help="Number of LSTM units",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=256,
        help="Number of hidden units",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per activity-lifecycle pair",
    )

    args = parser.parse_args()

    logger.info("Loading and filtering BPIC 2017 event log...")
    df_log = load_and_filter_bpic17(log_path=args.log_path)

    logger.info("Adding END tokens to traces...")
    df_log = add_end_tokens(df_log)

    logger.info("Generating training data...")
    generator = BPIC17SimplifiedDataGenerator(
        df_log,
        max_history=args.max_history,
        min_samples=args.min_samples,
    )
    df_train = generator.generate()

    logger.info(f"Generated {len(df_train):,} training sequences")

    logger.info("Initializing model...")
    model = BPIC17SimplifiedModel(
        max_seq_len=args.max_history,
        lstm_units=args.lstm_units,
        hidden_units=args.hidden_units,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoints" / "best_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Training model...")
    history = model.fit(
        df_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        checkpoint_path=str(checkpoint_path),
    )

    logger.info("Saving model...")
    BPIC17SimplifiedPersistence.save(model, str(output_dir))

    logger.info(f"Training complete! Model saved to {output_dir}")
    logger.info(f"Final validation accuracy: {max(history.history.get('val_activity_accuracy', [0])):.4f}")


if __name__ == "__main__":
    main()


