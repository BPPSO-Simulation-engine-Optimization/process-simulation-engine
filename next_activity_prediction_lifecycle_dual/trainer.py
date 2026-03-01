import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

try:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except ImportError:
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .config import DualPredictionConfig
from .data import END_ACTIVITY, END_LIFECYCLE, PreparedData, prepare_data
from .model import build_dual_head_model, predict_labels

logger = logging.getLogger(__name__)


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    labels, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(labels)
    return {int(label): float(n_samples / (n_classes * count)) for label, count in zip(labels, counts)}


def _build_sample_weights(y_activity: np.ndarray, y_lifecycle: np.ndarray) -> np.ndarray:
    activity_weights = _compute_class_weights(y_activity)
    lifecycle_weights = _compute_class_weights(y_lifecycle)
    weights = np.ones(len(y_activity), dtype=np.float32)
    for idx in range(len(weights)):
        wa = activity_weights.get(int(y_activity[idx]), 1.0)
        wl = lifecycle_weights.get(int(y_lifecycle[idx]), 1.0)
        weights[idx] = np.sqrt(wa * wl)
    return weights / np.mean(weights)


def _evaluate(data: PreparedData, activity_pred: np.ndarray, lifecycle_pred: np.ndarray) -> Dict[str, float]:
    activity_true = data.y_activity_val
    lifecycle_true = data.y_lifecycle_val

    joint_true = np.stack([activity_true, lifecycle_true], axis=1)
    joint_pred = np.stack([activity_pred, lifecycle_pred], axis=1)
    joint_accuracy = float(np.mean(np.all(joint_true == joint_pred, axis=1)))

    metrics = {
        "activity_accuracy": float(accuracy_score(activity_true, activity_pred)),
        "activity_macro_f1": float(f1_score(activity_true, activity_pred, average="macro", zero_division=0)),
        "lifecycle_accuracy": float(accuracy_score(lifecycle_true, lifecycle_pred)),
        "lifecycle_macro_f1": float(f1_score(lifecycle_true, lifecycle_pred, average="macro", zero_division=0)),
        "joint_accuracy": joint_accuracy,
    }
    metrics["balanced_score"] = float(
        np.mean(
            [
                metrics["activity_macro_f1"],
                metrics["lifecycle_macro_f1"],
                metrics["joint_accuracy"],
            ]
        )
    )
    return metrics


def _metadata_dict(data: PreparedData, config: DualPredictionConfig, mode: str, methodology: str) -> Dict:
    return {
        "mode": mode,
        "methodology": methodology,
        "sequence_length": config.sequence_length,
        "activity_to_idx": data.activity_to_idx,
        "lifecycle_to_idx": data.lifecycle_to_idx,
        "idx_to_activity": {int(k): v for k, v in data.idx_to_activity.items()},
        "idx_to_lifecycle": {int(k): v for k, v in data.idx_to_lifecycle.items()},
        "end_activity_token": END_ACTIVITY,
        "end_lifecycle_token": END_LIFECYCLE,
    }


def train_variant(config: DualPredictionConfig, log_path: str, mode: str) -> List[Dict]:
    data = prepare_data(
        log_path=log_path,
        mode=mode,
        sequence_length=config.sequence_length,
        min_case_length=config.min_case_length,
        max_case_length=config.max_case_length,
        validation_split=config.validation_split,
        random_seed=config.random_seed,
    )

    results: List[Dict] = []
    for methodology in ["baseline", "balanced"]:
        out_dir = Path(config.model_root) / mode / methodology
        checkpoints_dir = out_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        model = build_dual_head_model(
            activity_vocab_size=len(data.activity_to_idx),
            lifecycle_vocab_size=len(data.lifecycle_to_idx),
            sequence_length=config.sequence_length,
            embedding_dim=config.embedding_dim,
            lstm_units=config.lstm_units,
            lstm_layers=config.lstm_layers,
            dropout_rate=config.dropout_rate,
            learning_rate=config.learning_rate,
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=str(checkpoints_dir / "best_model.keras"), monitor="val_loss", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ]

        sample_weight = None
        if methodology == "balanced":
            sample_weight = _build_sample_weights(data.y_activity_train, data.y_lifecycle_train)

        history = model.fit(
            [data.X_activity_train, data.X_lifecycle_train],
            [data.y_activity_train, data.y_lifecycle_train],
            validation_data=(
                [data.X_activity_val, data.X_lifecycle_val],
                [data.y_activity_val, data.y_lifecycle_val],
            ),
            sample_weight=sample_weight,
            batch_size=config.batch_size,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(str(out_dir / "model.keras"))
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(_metadata_dict(data, config, mode, methodology), f, indent=2)

        preds = predict_labels(model, data.X_activity_val, data.X_lifecycle_val)
        metrics = _evaluate(
            data=data,
            activity_pred=np.array(preds["activity_pred"], dtype=np.int32),
            lifecycle_pred=np.array(preds["lifecycle_pred"], dtype=np.int32),
        )

        result = {
            "mode": mode,
            "methodology": methodology,
            "metrics": metrics,
            "model_dir": str(out_dir),
        }
        results.append(result)

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info("%s | %s => balanced_score=%.4f", mode, methodology, metrics["balanced_score"])

    return results


def run_full_experiment(log_path: str, config: DualPredictionConfig | None = None) -> Dict:
    config = config or DualPredictionConfig()
    Path(config.model_root).mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for mode in ["start_complete", "full_lifecycle"]:
        all_results.extend(train_variant(config=config, log_path=log_path, mode=mode))

    all_results_sorted = sorted(all_results, key=lambda x: x["metrics"]["balanced_score"], reverse=True)
    summary = {
        "best_model": all_results_sorted[0],
        "all_results": all_results_sorted,
    }

    summary_path = Path(config.model_root) / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote experiment summary to %s", summary_path)
    return summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train and compare dual-head next-event predictors.")
    parser.add_argument("--log-path", required=True, help="Path to CSV/XES event log")
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--sequence-length", type=int, default=50, help="Input sequence length")
    parser.add_argument("--model-root", default="next_activity_prediction_lifecycle_dual/models", help="Output folder")
    args = parser.parse_args()

    cfg = DualPredictionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        model_root=args.model_root,
    )
    run_full_experiment(log_path=args.log_path, config=cfg)
