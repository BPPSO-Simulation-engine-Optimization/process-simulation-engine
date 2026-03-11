"""
Gymnasium environment for DRL resource allocation training.

Wraps the DES engine in a Gym env. The engine runs in a background thread,
communicating with the env via an InteractiveBatchPolicy queue bridge.
"""

import logging
import threading
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from resources.drl_allocation.policy import (
    DRLDecisionResponse,
    InteractiveBatchPolicy,
)
from resources.drl_allocation.state import DRLStateBuilder
from resources.batch_policies import BatchDecision

logger = logging.getLogger(__name__)


class ResourceAllocationEnv(gym.Env):
    """
    Gymnasium env for training a PPO agent on resource allocation.

    The DES engine runs in a background thread. At each decision point
    (resource freed + waiting work), the engine sends a DRLDecisionRequest
    via the InteractiveBatchPolicy bridge. The env's step() receives this,
    builds obs/mask, forwards the action, and computes the reward from
    completed case cycle times.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        engine_factory: Callable[[InteractiveBatchPolicy], "TrainingDESEngine"],
        state_builder: DRLStateBuilder,
        num_cases: int = 500,
        max_steps: int = 500_000,
        reward_tau: float = 100.0,
    ):
        """
        Args:
            engine_factory: Callable that takes an InteractiveBatchPolicy and returns
                a configured TrainingDESEngine (with all predictors, allocator, etc.).
            state_builder: DRLStateBuilder for observation/mask construction.
            num_cases: Number of cases per episode.
            max_steps: Safety limit on decision steps per episode.
            reward_tau: Reference time for reward scaling: r = 1/(1 + CT/tau).
        """
        super().__init__()

        self._engine_factory = engine_factory
        self._state_builder = state_builder
        self._num_cases = num_cases
        self._max_steps = max_steps
        self._reward_tau = reward_tau

        # Gym spaces
        self.action_space = spaces.Discrete(state_builder.num_activities + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(state_builder.observation_size,),
            dtype=np.float32,
        )

        # Episode state
        self._bridge: Optional[InteractiveBatchPolicy] = None
        self._engine = None
        self._engine_thread: Optional[threading.Thread] = None
        self._current_request = None
        self._steps = 0
        self._episode_done = False

    def reset(self, *, seed=None, options=None):
        """Start a new episode: create fresh engine + bridge, launch engine thread."""
        super().reset(seed=seed)

        # Clean up previous episode
        self._cleanup()

        # Create fresh bridge and engine
        self._bridge = InteractiveBatchPolicy()
        self._engine = self._engine_factory(self._bridge)
        self._steps = 0
        self._episode_done = False

        # Launch engine in background thread
        self._engine_thread = threading.Thread(
            target=self._run_engine, daemon=True
        )
        self._engine_thread.start()

        # Wait for first decision request
        self._current_request = self._bridge.request_queue.get()

        if self._current_request is None:
            # Engine finished immediately (no decisions needed)
            self._episode_done = True
            obs = np.zeros(self._state_builder.observation_size, dtype=np.float32)
            return obs, {}

        obs = self._build_obs()
        return obs, {}

    def step(self, action):
        """Send action to engine, wait for next request, compute reward."""
        if self._episode_done:
            obs = np.zeros(self._state_builder.observation_size, dtype=np.float32)
            return obs, 0.0, True, False, {}

        # Map action to BatchDecision
        decision = self._action_to_decision(action)

        # Send response to engine (unblocks engine thread)
        self._bridge.response_queue.put(DRLDecisionResponse(decision=decision))

        # Wait for next decision request (engine runs, processes events, blocks again)
        self._current_request = self._bridge.request_queue.get()

        # NOW safe to read completed cases (engine is blocked in next decide())
        cycle_times = self._engine.pop_completed_cases()

        # Compute reward from completed cases
        reward = 0.0
        for ct_hours in cycle_times:
            reward += 1.0 / (1.0 + ct_hours / self._reward_tau)

        self._steps += 1

        if self._current_request is None:
            # Episode is over (engine finished)
            self._episode_done = True
            # Collect any remaining completed cases
            remaining = self._engine.pop_completed_cases()
            for ct_hours in remaining:
                reward += 1.0 / (1.0 + ct_hours / self._reward_tau)
            obs = np.zeros(self._state_builder.observation_size, dtype=np.float32)
            return obs, reward, True, False, {}

        if self._steps >= self._max_steps:
            self._episode_done = True
            obs = self._build_obs()
            return obs, reward, False, True, {}  # truncated

        obs = self._build_obs()
        return obs, reward, False, False, {}

    def action_masks(self) -> np.ndarray:
        """Return action mask for MaskablePPO (required by sb3-contrib)."""
        if self._current_request is None or self._episode_done:
            # All masked except postpone
            mask = np.zeros(self._state_builder.num_activities + 1, dtype=np.bool_)
            mask[-1] = True
            return mask

        return self._state_builder.build_action_mask(
            freed_resource=self._current_request.freed_resource,
            eligible_map=self._current_request.eligible_map,
            waiting_activities=self._current_request.waiting_activities,
        )

    def close(self):
        """Clean up engine thread."""
        self._cleanup()

    def _build_obs(self) -> np.ndarray:
        """Build observation from current request."""
        req = self._current_request
        return self._state_builder.build_from_snapshot(
            freed_resource=req.freed_resource,
            current_time_dt=req.current_time_dt,
            pool_snapshot=req.pool_snapshot,
            active_case_count=req.active_case_count,
        )

    def _action_to_decision(self, action: int) -> Optional[BatchDecision]:
        """Map discrete action to BatchDecision."""
        if action >= self._state_builder.num_activities:
            return None  # Postpone

        chosen_activity = self._state_builder.activity_list[action]
        req = self._current_request

        # Find oldest waiting task for this activity
        matching = [
            t for t in req.tasks
            if t.allocation_activity == chosen_activity
        ]
        if not matching:
            return None

        # Pick the one that waited longest
        best = max(matching, key=lambda t: t.hours_waited)
        return BatchDecision(task_id=best.task_id, worker_id=req.freed_resource)

    def _run_engine(self):
        """Run the DES engine in a background thread."""
        try:
            self._engine.run(num_cases=self._num_cases)
        except Exception as e:
            logger.error("Engine thread exception: %s", e, exc_info=True)
        finally:
            # Ensure env unblocks even on error
            if self._bridge is not None:
                self._bridge.signal_episode_end()

    def _cleanup(self):
        """Abort bridge and join engine thread."""
        if self._bridge is not None:
            self._bridge.abort()
        if self._engine_thread is not None and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=5.0)
            if self._engine_thread.is_alive():
                logger.warning("Engine thread did not exit within timeout")
        self._bridge = None
        self._engine = None
        self._engine_thread = None
        self._current_request = None
