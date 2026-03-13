"""
Unit tests for DRL resource allocation components.

Tests: DRLStateBuilder, action masks, InteractiveBatchPolicy, ResourceAllocationEnv.

Usage:
    python resources/drl_allocation/tests/test_policy.py
"""

import math
import queue
import threading
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from resources.drl_allocation.state import DRLStateBuilder
from resources.drl_allocation.policy import (
    DRLDecisionRequest,
    DRLDecisionResponse,
    InteractiveBatchPolicy,
)
from resources.batch_policies import BatchDecision, TaskInfo


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_state_builder(num_roles=3, num_activities=5):
    """Create a small DRLStateBuilder for testing."""
    activities = [f"Act_{i}" for i in range(num_activities)]
    groups = [
        {f"R_{i*3+j}" for j in range(3)}
        for i in range(num_roles)
    ]
    resource_to_role = {}
    for idx, g in enumerate(groups):
        for r in g:
            resource_to_role[r] = idx

    activity_to_roles = {
        activities[0]: {0},
        activities[1]: {0, 1},
        activities[2]: {1},
        activities[3]: {2},
        activities[4]: {0, 2},
    }

    return DRLStateBuilder(
        activity_list=activities,
        role_groups=groups,
        resource_to_role=resource_to_role,
        activity_to_roles=activity_to_roles,
        max_queue_length=100.0,
        max_wait_hours=24.0,
        max_remaining_busy_hours=12.0,
        max_active_cases=1000.0,
    )


def make_mock_resource_pool(busy=None, waiting=None):
    """Create a mock resource pool."""
    pool = MagicMock()
    pool._busy_resources = busy or {}
    pool._waiting_queues = waiting or {}
    pool.has_waiting_work = lambda act=None: bool(waiting) if act is None else bool(waiting.get(act))
    return pool


# ---------------------------------------------------------------------------
# TestDRLStateBuilder
# ---------------------------------------------------------------------------

class TestDRLStateBuilder(unittest.TestCase):

    def setUp(self):
        self.sb = make_state_builder(num_roles=3, num_activities=5)

    def test_observation_size(self):
        """3K + 3A + 5 = 3*3 + 3*5 + 5 = 29."""
        self.assertEqual(self.sb.observation_size, 29)

    def test_build_zeros_when_idle(self):
        """All zeros (except temporal) when pool is empty and no active cases."""
        pool = make_mock_resource_pool()
        now = datetime(2024, 1, 8, 12, 0)  # Monday noon
        obs = self.sb.build("R_0", now, pool, active_case_count=0)

        self.assertEqual(obs.shape, (29,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

        # Role idle fractions should be 1.0 (all idle)
        K = 3
        np.testing.assert_array_equal(obs[0:K], [1.0, 1.0, 1.0])

        # Queue lengths should be 0
        A = 5
        np.testing.assert_array_equal(obs[2*K:2*K+A], [0.0] * A)

    def test_build_values_in_range(self):
        """All observation values should be in [0, 1]."""
        busy_until = datetime(2024, 1, 8, 14, 0)
        pool = make_mock_resource_pool(
            busy={"R_0": (busy_until, "case1", "Act_0")}
        )
        now = datetime(2024, 1, 8, 12, 0)
        obs = self.sb.build("R_0", now, pool, active_case_count=500)

        self.assertTrue(np.all(obs >= 0.0), f"Min value: {obs.min()}")
        self.assertTrue(np.all(obs <= 1.0), f"Max value: {obs.max()}")

    def test_freed_role_onehot(self):
        """Freed resource's role should have exactly one 1.0 in the one-hot slice."""
        pool = make_mock_resource_pool()
        now = datetime(2024, 1, 8, 12, 0)
        K, A = 3, 5

        # R_0 is in role 0
        obs = self.sb.build("R_0", now, pool, active_case_count=0)
        onehot_slice = obs[2*K + 2*A : 2*K + 2*A + K]
        np.testing.assert_array_equal(onehot_slice, [1.0, 0.0, 0.0])

        # R_3 is in role 1
        obs = self.sb.build("R_3", now, pool, active_case_count=0)
        onehot_slice = obs[2*K + 2*A : 2*K + 2*A + K]
        np.testing.assert_array_equal(onehot_slice, [0.0, 1.0, 0.0])

    def test_freed_eligible_activities(self):
        """Freed resource's eligible activity bits should match activity_to_roles."""
        pool = make_mock_resource_pool()
        now = datetime(2024, 1, 8, 12, 0)
        K, A = 3, 5

        # R_0 is in role 0, eligible for Act_0 ({0}), Act_1 ({0,1}), Act_4 ({0,2})
        obs = self.sb.build("R_0", now, pool, active_case_count=0)
        eligible_slice = obs[3*K + 2*A : 3*K + 3*A]
        np.testing.assert_array_equal(eligible_slice, [1.0, 1.0, 0.0, 0.0, 1.0])

    def test_cyclical_encoding(self):
        """Temporal features should be cyclical sin/cos in [0, 1]."""
        pool = make_mock_resource_pool()
        K, A = 3, 5
        base = 3*K + 3*A

        # Midnight: hour_frac=0
        midnight = datetime(2024, 1, 8, 0, 0)  # Monday
        obs = self.sb.build("R_0", midnight, pool, active_case_count=0)
        self.assertAlmostEqual(obs[base], 0.5, places=3)      # sin(0)=0 -> 0.5
        self.assertAlmostEqual(obs[base + 1], 1.0, places=3)  # cos(0)=1 -> 1.0

    def test_idle_fraction_with_busy_resources(self):
        """Idle fraction should decrease when resources are busy."""
        busy_until = datetime(2024, 1, 8, 14, 0)
        pool = make_mock_resource_pool(
            busy={
                "R_0": (busy_until, "case1", "Act_0"),
                "R_1": (busy_until, "case2", "Act_1"),
            }
        )
        now = datetime(2024, 1, 8, 12, 0)
        obs = self.sb.build("R_0", now, pool, active_case_count=0)

        # Role 0 has 3 resources, 2 busy -> idle_frac = 1/3
        self.assertAlmostEqual(obs[0], 1.0 / 3.0, places=3)

    def test_build_from_snapshot(self):
        """Snapshot-based build should produce same shape and range."""
        snapshot = {
            "busy_resources": {
                "R_0": {"busy_until": datetime(2024, 1, 8, 14, 0), "case_id": "c1", "activity": "Act_0"},
            },
            "waiting_queues": {
                "Act_0": [
                    {"case_id": "c2", "activity": "Act_0", "allocation_activity": "Act_0",
                     "arrival_time": datetime(2024, 1, 8, 11, 0)},
                ],
            },
        }
        now = datetime(2024, 1, 8, 12, 0)
        obs = self.sb.build_from_snapshot("R_0", now, snapshot, active_case_count=100)

        self.assertEqual(obs.shape, (29,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

        # Queue length for Act_0 should be > 0
        K, A = 3, 5
        self.assertGreater(obs[2*K + 0], 0.0)


# ---------------------------------------------------------------------------
# TestDRLActionMask
# ---------------------------------------------------------------------------

class TestDRLActionMask(unittest.TestCase):

    def setUp(self):
        self.sb = make_state_builder(num_roles=3, num_activities=5)

    def test_mask_shape(self):
        """Mask should have num_activities + 1 entries."""
        mask = self.sb.build_action_mask("R_0", {}, set())
        self.assertEqual(mask.shape, (6,))

    def test_postpone_always_feasible(self):
        """Postpone action (last) should always be True."""
        mask = self.sb.build_action_mask("R_0", {}, set())
        self.assertTrue(mask[-1])

    def test_empty_queue_masked(self):
        """Activities with no waiting work should be masked."""
        eligible_map = {"Act_0": {"R_0"}}
        waiting = set()  # No waiting activities
        mask = self.sb.build_action_mask("R_0", eligible_map, waiting)
        self.assertFalse(mask[0])  # Act_0 has eligible but no waiting

    def test_ineligible_masked(self):
        """Activities the freed resource can't do should be masked."""
        eligible_map = {"Act_0": {"R_3"}}  # R_3 eligible, not R_0
        waiting = {"Act_0"}
        mask = self.sb.build_action_mask("R_0", eligible_map, waiting)
        self.assertFalse(mask[0])

    def test_feasible_action(self):
        """Action should be True when resource is eligible AND work is waiting."""
        eligible_map = {"Act_0": {"R_0", "R_1"}}
        waiting = {"Act_0"}
        mask = self.sb.build_action_mask("R_0", eligible_map, waiting)
        self.assertTrue(mask[0])
        self.assertTrue(mask[-1])  # postpone always on


# ---------------------------------------------------------------------------
# TestInteractiveBatchPolicy
# ---------------------------------------------------------------------------

class TestInteractiveBatchPolicy(unittest.TestCase):

    def test_request_response_roundtrip(self):
        """Bridge should pass request and receive response correctly."""
        bridge = InteractiveBatchPolicy()

        decision = BatchDecision(task_id="c1::Act_0", worker_id="R_0")

        def responder():
            req = bridge.request_queue.get(timeout=2.0)
            self.assertIsNotNone(req)
            bridge.response_queue.put(DRLDecisionResponse(decision=decision))

        t = threading.Thread(target=responder)
        t.start()

        result = bridge.decide(
            freed_resource="R_0",
            current_time_s=1000.0,
            tasks=[TaskInfo("c1::Act_0", "c1", "Act_0", "Act_0", 1.0)],
            workers=[],
            eligible_map={"Act_0": {"R_0"}},
            processing_time_fn=lambda w, a: 0.0,
        )

        t.join(timeout=2.0)
        self.assertEqual(result, decision)

    def test_episode_end_signal(self):
        """signal_episode_end() should put None into request queue."""
        bridge = InteractiveBatchPolicy()
        bridge.signal_episode_end()
        msg = bridge.request_queue.get(timeout=1.0)
        self.assertIsNone(msg)

    def test_abort_unblocks(self):
        """abort() should unblock a stuck decide() call."""
        bridge = InteractiveBatchPolicy()

        result_holder = [None]

        def caller():
            result_holder[0] = bridge.decide(
                freed_resource="R_0",
                current_time_s=1000.0,
                tasks=[],
                workers=[],
                eligible_map={},
                processing_time_fn=lambda w, a: 0.0,
            )

        t = threading.Thread(target=caller)
        t.start()

        # Give the thread time to block
        import time
        time.sleep(0.1)

        bridge.abort()
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertIsNone(result_holder[0])

    def test_abort_flag_fast_exit(self):
        """After abort, decide() should return None immediately."""
        bridge = InteractiveBatchPolicy()
        bridge.abort()

        result = bridge.decide(
            freed_resource="R_0",
            current_time_s=1000.0,
            tasks=[],
            workers=[],
            eligible_map={},
            processing_time_fn=lambda w, a: 0.0,
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TestResourceAllocationEnv
# ---------------------------------------------------------------------------

class TestResourceAllocationEnv(unittest.TestCase):
    """
    Integration test for the Gym environment.

    Uses a mock engine factory that simulates a minimal decision loop.
    """

    def _make_env(self):
        """Create a minimal env with a mock engine."""
        from resources.drl_allocation.env import ResourceAllocationEnv

        sb = make_state_builder(num_roles=3, num_activities=5)

        def engine_factory(bridge):
            """Mock engine that produces 2 decision points then finishes."""
            engine = MagicMock()
            engine.pop_completed_cases = MagicMock(side_effect=[
                [10.0],    # First step: one case completed with CT=10h
                [20.0],    # Second step: one case completed with CT=20h
                [],        # Final cleanup
            ])

            def mock_run(num_cases=100):
                # Simulate 2 decision points
                tasks = [TaskInfo("c1::Act_0", "c1", "Act_0", "Act_0", 1.0)]
                eligible = {"Act_0": {"R_0"}}
                waiting = {"Act_0"}
                snapshot = {"busy_resources": {}, "waiting_queues": {}}
                now = datetime(2024, 1, 8, 12, 0)

                for _ in range(2):
                    bridge.decide(
                        freed_resource="R_0",
                        current_time_s=now.timestamp(),
                        tasks=tasks,
                        workers=[],
                        eligible_map=eligible,
                        processing_time_fn=lambda w, a: 0.0,
                        current_time_dt=now,
                        waiting_activities=waiting,
                        pool_snapshot=snapshot,
                        active_case_count=5,
                    )

                bridge.signal_episode_end()

            engine.run = mock_run
            return engine

        env = ResourceAllocationEnv(
            engine_factory=engine_factory,
            state_builder=sb,
            num_cases=10,
            reward_tau=100.0,
        )
        return env

    def test_obs_action_space_shapes(self):
        """Observation and action spaces should match state builder dims."""
        env = self._make_env()
        self.assertEqual(env.observation_space.shape, (29,))
        self.assertEqual(env.action_space.n, 6)
        env.close()

    def test_step_returns_valid_tuple(self):
        """step() should return (obs, reward, terminated, truncated, info)."""
        env = self._make_env()
        obs, info = env.reset()

        self.assertEqual(obs.shape, (29,))

        # Take action: postpone (action 5)
        obs, reward, terminated, truncated, info = env.step(5)
        self.assertEqual(obs.shape, (29,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

        env.close()

    def test_episode_terminates(self):
        """Episode should terminate after engine finishes."""
        env = self._make_env()
        obs, info = env.reset()

        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(5)  # postpone
            done = terminated or truncated
            steps += 1
            if steps > 10:
                self.fail("Episode did not terminate within 10 steps")

        env.close()

    def test_action_mask(self):
        """action_masks() should return correct shape."""
        env = self._make_env()
        obs, info = env.reset()
        mask = env.action_masks()

        self.assertEqual(mask.shape, (6,))
        self.assertTrue(mask[-1])  # Postpone always feasible

        env.close()


if __name__ == "__main__":
    unittest.main()
