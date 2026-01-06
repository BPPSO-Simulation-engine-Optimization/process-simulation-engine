"""
BPIC17 Simplified Next Activity Predictor for simulation integration.

Implements NextActivityPredictor protocol with END token support.
"""

import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CaseTracker:
    """Tracks state for a single case during simulation."""

    activities: List[str] = field(default_factory=list)
    lifecycles: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    seen_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    last_timestamp: Optional[datetime] = None

    def add_event(
        self,
        activity: str,
        lifecycle: str,
        resource: str = "Unknown",
        timestamp: Optional[datetime] = None,
    ):
        """Record an event."""
        self.activities.append(activity)
        self.lifecycles.append(lifecycle)
        self.resources.append(resource)
        self.seen_pairs.add((activity, lifecycle))
        self.last_timestamp = timestamp

    def get_history(self, max_len: int = 15) -> Tuple[List, List, List]:
        """Get recent history for prediction."""
        return (
            self.activities[-max_len:],
            self.lifecycles[-max_len:],
            self.resources[-max_len:],
        )


class BPIC17SimplifiedPredictor:
    """
    Next activity predictor using simplified BPIC17 model with END token support.

    Implements the NextActivityPredictor protocol for DESEngine integration.
    Filters to start/complete lifecycle transitions only.
    """

    END_TOKEN = "END"
    START_ACTIVITY = "A_Create Application"
    DEFAULT_LIFECYCLE = "complete"

    def __init__(
        self,
        model_path: str = "models/bpic17_simplified",
        max_history: int = 15,
        repetition_penalty: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model directory.
            max_history: Maximum history length for predictions.
            repetition_penalty: Penalty factor for seen (activity, lifecycle) pairs.
            seed: Random seed for sampling.
        """
        self.model_path = model_path
        self.max_history = max_history
        self.repetition_penalty = repetition_penalty
        self.rng = random.Random(seed)

        self._load_model()

        self._case_trackers: Dict[str, CaseTracker] = {}

    def _load_model(self):
        """Load the model and encoder."""
        from .persistence import BPIC17SimplifiedPersistence

        bundle = BPIC17SimplifiedPersistence.load(self.model_path)
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]

        logger.info(
            f"Loaded BPIC17SimplifiedPredictor: "
            f"{self.encoder.num_target_activities} activities, "
            f"{self.encoder.num_target_lifecycles} lifecycles"
        )

    def _get_tracker(self, case_id: str) -> CaseTracker:
        """Get or create tracker for a case."""
        if case_id not in self._case_trackers:
            self._case_trackers[case_id] = CaseTracker()
        return self._case_trackers[case_id]

    def _apply_repetition_penalty(
        self,
        activity_probs: List[Tuple[str, float]],
        lifecycle_probs: List[Tuple[str, float]],
        seen_pairs: Set[Tuple[str, str]],
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Apply penalty to seen (activity, lifecycle) combinations."""
        activity_counts = {}
        for act, lc in seen_pairs:
            activity_counts[act] = activity_counts.get(act, 0) + 1

        penalized_activities = []
        for act, prob in activity_probs:
            count = activity_counts.get(act, 0)
            penalty = self.repetition_penalty ** count
            penalized_activities.append((act, prob * penalty))

        total = sum(p for _, p in penalized_activities)
        if total > 0:
            penalized_activities = [(a, p / total) for a, p in penalized_activities]

        return penalized_activities, lifecycle_probs

    def _sample_next(
        self,
        activity_probs: List[Tuple[str, float]],
        lifecycle_probs: List[Tuple[str, float]],
    ) -> Tuple[str, str]:
        """Sample next activity and lifecycle from probability distributions."""
        activities = [a for a, _ in activity_probs]
        act_weights = [p for _, p in activity_probs]
        next_activity = self.rng.choices(activities, weights=act_weights, k=1)[0]

        lifecycles = [l for l, _ in lifecycle_probs]
        lc_weights = [p for _, p in lifecycle_probs]
        next_lifecycle = self.rng.choices(lifecycles, weights=lc_weights, k=1)[0]

        return next_activity, next_lifecycle

    def predict(self, case_state: Any) -> Tuple[str, bool]:
        """
        Predict next activity for a case.

        Args:
            case_state: CaseState object with case_id and activity_history.

        Returns:
            Tuple of (next_activity, is_case_ended).
        """
        case_id = case_state.case_id
        tracker = self._get_tracker(case_id)

        if not case_state.activity_history:
            tracker.add_event(
                self.START_ACTIVITY,
                self.DEFAULT_LIFECYCLE,
                case_state.current_resource or "Unknown",
            )
            return self.START_ACTIVITY, False

        current_activity = case_state.activity_history[-1]

        if len(tracker.activities) < len(case_state.activity_history):
            for act in case_state.activity_history[len(tracker.activities):]:
                tracker.add_event(
                    act,
                    self.DEFAULT_LIFECYCLE,
                    case_state.current_resource or "Unknown",
                )

        if current_activity == self.END_TOKEN:
            return current_activity, True

        context = {
            "case:LoanGoal": case_state.case_type,
            "case:ApplicationType": case_state.application_type,
            "case:RequestedAmount": case_state.requested_amount,
        }

        activities, lifecycles, resources = tracker.get_history(self.max_history)

        try:
            activity_probs, lifecycle_probs = self.model.predict(
                activities, lifecycles, resources, context, top_k=10
            )

            activity_probs, lifecycle_probs = self._apply_repetition_penalty(
                activity_probs, lifecycle_probs, tracker.seen_pairs
            )

            next_activity, next_lifecycle = self._sample_next(activity_probs, lifecycle_probs)

        except Exception as e:
            logger.warning(f"Prediction failed for case {case_id}: {e}")
            return self.END_TOKEN, True

        tracker.add_event(next_activity, next_lifecycle, case_state.current_resource or "Unknown")

        is_end = next_activity == self.END_TOKEN
        return next_activity, is_end

    def reset_case(self, case_id: str):
        """Reset tracker for a case (when case ends)."""
        if case_id in self._case_trackers:
            del self._case_trackers[case_id]

    def clear(self):
        """Clear all case trackers."""
        self._case_trackers.clear()


