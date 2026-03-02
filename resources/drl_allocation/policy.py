"""
DRL allocation policies: inference adapter and training bridge.

DRLAllocationPolicy — wraps a trained MaskablePPO for use during simulation.
InteractiveBatchPolicy — queue bridge between Gym env (main thread) and
                         DES engine (background thread) during training.
"""

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from resources.batch_policies import BatchAllocationPolicy, BatchDecision, TaskInfo, WorkerInfo
from resources.drl_allocation.state import DRLStateBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Objects for the training bridge
# ---------------------------------------------------------------------------

@dataclass
class DRLDecisionRequest:
    """Sent from engine thread -> env (main thread)."""
    freed_resource: str
    current_time_s: float
    current_time_dt: datetime
    tasks: List[TaskInfo]
    eligible_map: Dict[str, Set[str]]
    waiting_activities: Set[str]
    pool_snapshot: dict
    active_case_count: int


@dataclass
class DRLDecisionResponse:
    """Sent from env (main thread) -> engine thread."""
    decision: Optional[BatchDecision]


# ---------------------------------------------------------------------------
# Inference Policy
# ---------------------------------------------------------------------------

class DRLAllocationPolicy(BatchAllocationPolicy):
    """
    Inference adapter: wraps a trained MaskablePPO model.

    Called by DESEngine._process_waiting_queue_drl().  Before each decide(),
    the engine calls set_engine_state() so the policy can build observations
    from the live resource pool.
    """

    def __init__(
        self,
        model_path: str,
        state_builder: DRLStateBuilder,
        deterministic: bool = True,
    ):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)
        self.state_builder = state_builder
        self.deterministic = deterministic
        self._resource_pool = None
        self._case_manager = None

    def set_engine_state(self, resource_pool, case_manager) -> None:
        """Called by engine before each decide() so policy can read live state."""
        self._resource_pool = resource_pool
        self._case_manager = case_manager

    def decide(
        self,
        freed_resource: str,
        current_time_s: float,
        tasks: List[TaskInfo],
        workers: List[WorkerInfo],
        eligible_map: Dict[str, Set[str]],
        processing_time_fn: Callable[[str, str], float],
    ) -> Optional[BatchDecision]:
        """Select which activity queue to serve using the trained PPO agent."""
        if self._resource_pool is None:
            logger.warning("DRLAllocationPolicy.decide() called without set_engine_state()")
            return None

        current_time_dt = datetime.fromtimestamp(current_time_s)
        active_count = self._case_manager.active_count() if self._case_manager else 0

        # Build observation
        obs = self.state_builder.build(
            freed_resource=freed_resource,
            current_time=current_time_dt,
            resource_pool=self._resource_pool,
            active_case_count=active_count,
        )

        # Build action mask
        waiting_activities = {t.allocation_activity for t in tasks}
        mask = self.state_builder.build_action_mask(
            freed_resource=freed_resource,
            eligible_map=eligible_map,
            waiting_activities=waiting_activities,
        )

        # Predict action
        action, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        action = int(action)

        # Interpret action
        if action >= self.state_builder.num_activities:
            # Postpone
            return None

        chosen_activity = self.state_builder.activity_list[action]

        # Find the oldest waiting task for this activity
        for task in sorted(tasks, key=lambda t: -t.hours_waited):
            if task.allocation_activity == chosen_activity:
                return BatchDecision(task_id=task.task_id, worker_id=freed_resource)

        # Should not reach here if mask was correct
        logger.warning(
            "DRL chose activity %s but no matching task found", chosen_activity
        )
        return None

    def reset(self) -> None:
        self._resource_pool = None
        self._case_manager = None


# ---------------------------------------------------------------------------
# Training Bridge
# ---------------------------------------------------------------------------

class InteractiveBatchPolicy(BatchAllocationPolicy):
    """
    Queue bridge for training: DES engine (background thread) communicates
    with the Gym env (main thread) via two queues.

    Engine calls decide() -> puts DRLDecisionRequest -> blocks on response.
    Env step() reads request -> builds obs/mask -> gets action -> puts response.
    """

    def __init__(self):
        self.request_queue: queue.Queue = queue.Queue(maxsize=1)
        self.response_queue: queue.Queue = queue.Queue(maxsize=1)
        self._abort = threading.Event()

    def decide(
        self,
        freed_resource: str,
        current_time_s: float,
        tasks: List[TaskInfo],
        workers: List[WorkerInfo],
        eligible_map: Dict[str, Set[str]],
        processing_time_fn: Callable[[str, str], float],
        *,
        current_time_dt: datetime = None,
        waiting_activities: Set[str] = None,
        pool_snapshot: dict = None,
        active_case_count: int = 0,
    ) -> Optional[BatchDecision]:
        """
        Put a decision request and block until the env responds.

        Extra kwargs (current_time_dt, waiting_activities, pool_snapshot,
        active_case_count) are passed by _process_waiting_queue_drl() but
        are NOT part of the base BatchAllocationPolicy interface.
        """
        if self._abort.is_set():
            return None

        if current_time_dt is None:
            current_time_dt = datetime.fromtimestamp(current_time_s)
        if waiting_activities is None:
            waiting_activities = {t.allocation_activity for t in tasks}
        if pool_snapshot is None:
            pool_snapshot = {}

        request = DRLDecisionRequest(
            freed_resource=freed_resource,
            current_time_s=current_time_s,
            current_time_dt=current_time_dt,
            tasks=tasks,
            eligible_map=eligible_map,
            waiting_activities=waiting_activities,
            pool_snapshot=pool_snapshot,
            active_case_count=active_case_count,
        )

        self.request_queue.put(request)

        try:
            response = self.response_queue.get(timeout=30.0)
        except queue.Empty:
            logger.warning("InteractiveBatchPolicy timed out waiting for response")
            return None

        if self._abort.is_set():
            return None

        return response.decision

    def signal_episode_end(self) -> None:
        """Put None into request_queue to signal env that the episode is over."""
        try:
            self.request_queue.put_nowait(None)
        except queue.Full:
            pass

    def abort(self) -> None:
        """Abort the bridge: unblock any stuck get() calls."""
        self._abort.set()
        # Unblock response_queue.get() in decide()
        try:
            self.response_queue.put_nowait(DRLDecisionResponse(decision=None))
        except queue.Full:
            pass
        # Unblock request_queue.get() in env step()
        try:
            self.request_queue.put_nowait(None)
        except queue.Full:
            pass

    def reset(self) -> None:
        """Reset for a new episode."""
        self._abort.clear()
        # Drain queues
        for q in (self.request_queue, self.response_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
