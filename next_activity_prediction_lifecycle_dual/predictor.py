import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from tensorflow import keras
except ImportError:
    import keras

logger = logging.getLogger(__name__)


class DualLifecycleNextActivityPredictor:
    """
    Runtime predictor for the dual-head lifecycle model.

    It predicts both activity and lifecycle internally, while exposing only
    next-activity behavior expected by the simulation engine.
    """

    def __init__(self, model_path: str, seed: Optional[int] = None):
        self.model_path = Path(model_path)
        self.model, metadata = self._load_model_and_metadata(self.model_path)

        self.sequence_length = int(metadata["sequence_length"])
        self.activity_to_idx = {k: int(v) for k, v in metadata["activity_to_idx"].items()}
        self.lifecycle_to_idx = {k: int(v) for k, v in metadata["lifecycle_to_idx"].items()}
        self.idx_to_activity = {int(k): v for k, v in metadata["idx_to_activity"].items()}
        self.idx_to_lifecycle = {int(k): v for k, v in metadata["idx_to_lifecycle"].items()}

        self.end_activity_idx = self.activity_to_idx.get(metadata.get("end_activity_token", "END_ACTIVITY"), -1)
        self.complete_lifecycle_idx = self.lifecycle_to_idx.get("complete", 0)

        self.case_activity_histories: Dict[str, list] = {}
        self.case_lifecycle_histories: Dict[str, list] = {}

        if seed is not None:
            np.random.seed(seed)

        logger.info("Loaded DualLifecycleNextActivityPredictor from %s", model_path)

    @staticmethod
    def _load_model_and_metadata(model_path: Path):
        if model_path.is_file() and model_path.suffix == ".keras":
            model_file = model_path
            model_dir = model_path.parent if model_path.name == "model.keras" else model_path.parent.parent
        else:
            model_dir = model_path
            model_file = model_dir / "model.keras"
            if not model_file.exists():
                checkpoint = model_dir / "checkpoints" / "best_model.keras"
                if checkpoint.exists():
                    model_file = checkpoint
                else:
                    raise FileNotFoundError(f"No model found in {model_dir}")

        metadata_file = model_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")

        model = keras.models.load_model(str(model_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, metadata

    def _pad(self, indices: list[int]) -> list[int]:
        if len(indices) < self.sequence_length:
            return [0] * (self.sequence_length - len(indices)) + indices
        return indices[-self.sequence_length :]

    def _sync_histories(self, case_id: str, observed_activities: list[str], observed_lifecycles: list[str]) -> tuple[list[str], list[str]]:
        if case_id not in self.case_activity_histories:
            self.case_activity_histories[case_id] = []
            self.case_lifecycle_histories[case_id] = []

        act_hist = self.case_activity_histories[case_id]
        life_hist = self.case_lifecycle_histories[case_id]

        if len(observed_activities) > len(act_hist):
            new_count = len(observed_activities) - len(act_hist)
            act_hist.extend(observed_activities[len(act_hist) :])
            if observed_lifecycles and len(observed_lifecycles) >= len(observed_activities):
                life_hist.extend(observed_lifecycles[len(life_hist) : len(life_hist) + new_count])
            else:
                # Fallback for legacy case state without lifecycle history.
                life_hist.extend(["complete"] * new_count)

        return act_hist, life_hist

    def predict(self, case_state) -> tuple[str, str, bool]:
        case_id = case_state.case_id
        observed_activities = case_state.activity_history or []
        observed_lifecycles = getattr(case_state, "lifecycle_history", []) or []
        act_hist, life_hist = self._sync_histories(case_id, observed_activities, observed_lifecycles)

        if not act_hist:
            if "A_Create Application" in self.activity_to_idx:
                return "A_Create Application", "complete", False
            fallback = next((v for k, v in sorted(self.idx_to_activity.items()) if k > 0 and v != "END_ACTIVITY"), "A_Create Application")
            return fallback, "complete", False

        act_idx = [self.activity_to_idx.get(a, 0) for a in act_hist]
        life_idx = [self.lifecycle_to_idx.get(l, self.complete_lifecycle_idx) for l in life_hist]

        X_activity = np.array([self._pad(act_idx)], dtype=np.int32)
        X_lifecycle = np.array([self._pad(life_idx)], dtype=np.int32)

        pred_activity_probs, pred_lifecycle_probs = self.model.predict_on_batch([X_activity, X_lifecycle])
        probs = pred_activity_probs[0]
        lifecycle_probs = pred_lifecycle_probs[0]

        top_k = min(3, len(probs))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)
        next_idx = int(np.random.choice(top_indices, p=top_probs))
        next_lifecycle_idx = int(np.argmax(lifecycle_probs))

        predicted_activity = self.idx_to_activity.get(next_idx, act_hist[-1])
        predicted_lifecycle = self.idx_to_lifecycle.get(next_lifecycle_idx, "complete")
        is_end = next_idx == self.end_activity_idx or predicted_activity == "END_ACTIVITY"

        if is_end:
            self.case_activity_histories.pop(case_id, None)
            self.case_lifecycle_histories.pop(case_id, None)
            return act_hist[-1], predicted_lifecycle, True

        return predicted_activity, predicted_lifecycle, False

    def reset_case(self, case_id: str) -> None:
        self.case_activity_histories.pop(case_id, None)
        self.case_lifecycle_histories.pop(case_id, None)

    def clear(self) -> None:
        self.case_activity_histories.clear()
        self.case_lifecycle_histories.clear()
