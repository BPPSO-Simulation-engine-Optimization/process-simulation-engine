"""
Resource selection strategies for the simulation engine.

Implements three push-pattern heuristics for selecting a resource from
the pool of candidates that passed the 3-tier filtering pipeline
(permissions -> availability -> busy state):

- R-RMA (Random Allocation, Pattern 15): uniform random selection
- R-RRA (Round Robin Allocation, Pattern 16): cyclic per-activity selection
- R-SHQ (Shortest Queue Allocation, Pattern 17): least-assigned resource,
  with R-RMA tiebreaker
"""

import random
from abc import ABC, abstractmethod
from typing import List, Dict
from collections import defaultdict


class ResourceSelectionStrategy(ABC):
    """Abstract base class for resource selection strategies."""

    @abstractmethod
    def select(self, available_resources: List[str], activity: str) -> str:
        """
        Select one resource from the available candidates.

        Args:
            available_resources: Non-empty list of resources that passed
                all 3 tiers of filtering (eligible, on-duty, not busy).
            activity: The activity being assigned (used as allocation group key).

        Returns:
            The selected resource identifier.
        """
        ...

    def notify_assignment(self, resource: str, activity: str) -> None:
        """Called after a resource is assigned. Override in stateful strategies."""
        pass

    def notify_release(self, resource: str) -> None:
        """Called when a resource is released. Override in stateful strategies."""
        pass

    def reset(self) -> None:
        """Reset internal state for a new simulation run."""
        pass


class RandomStrategy(ResourceSelectionStrategy):
    """
    R-RMA: Random Allocation (Pattern 15).

    Selects a resource uniformly at random from the available pool.
    """

    def select(self, available_resources: List[str], activity: str) -> str:
        return random.choice(available_resources)


class RoundRobinStrategy(ResourceSelectionStrategy):
    """
    R-RRA: Round Robin Allocation (Pattern 16).

    Cycles through resources on a per-activity basis. Resources are sorted
    alphabetically to ensure stable ordering across calls where the available
    set may vary due to availability/busy fluctuations.

    State: maintains LastAssignedIndex per activity (allocation group).
    """

    def __init__(self):
        # activity -> last assigned index into sorted available list
        self._last_index: Dict[str, int] = defaultdict(lambda: -1)

    def select(self, available_resources: List[str], activity: str) -> str:
        sorted_resources = sorted(available_resources)
        last = self._last_index[activity]
        next_idx = (last + 1) % len(sorted_resources)
        self._last_index[activity] = next_idx
        return sorted_resources[next_idx]

    def reset(self) -> None:
        self._last_index.clear()


class ShortestQueueStrategy(ResourceSelectionStrategy):
    """
    R-SHQ: Shortest Queue Allocation (Pattern 17).

    Per Russell et al. (2004), R-SHQ selects the resource with the fewest
    work items **currently allocated** (instantaneous queue depth).  The
    Tier 3 busy-state filter guarantees all candidates are idle, so their
    instantaneous depth is trivially zero — producing a universal tie that
    is broken by R-RMA (random selection).

    Using cumulative assignment counts as a proxy (whether global or
    per-activity) creates pathological load concentration at scale: it
    steers work toward resources whose natural availability is low for a
    given activity, increasing queue build-up and cascading wait times.
    """

    def __init__(self):
        # Tracks current (not cumulative) assignments per resource.
        # After Tier 3 filtering all candidates have 0 current items,
        # so select() always degenerates to random tiebreaker.
        self._current_assignments: Dict[str, int] = defaultdict(int)

    def select(self, available_resources: List[str], activity: str) -> str:
        # All candidates passed Tier 3 (not busy) → current depth is 0.
        # Fall through to random tiebreaker, matching the paper's intent.
        min_count = min(self._current_assignments.get(r, 0) for r in available_resources)
        candidates = [r for r in available_resources if self._current_assignments.get(r, 0) == min_count]
        return random.choice(candidates)

    def notify_assignment(self, resource: str, activity: str) -> None:
        self._current_assignments[resource] += 1

    def notify_release(self, resource: str) -> None:
        """Called when a resource completes work and is released."""
        if self._current_assignments[resource] > 0:
            self._current_assignments[resource] -= 1

    def reset(self) -> None:
        self._current_assignments.clear()


# ---------------------------------------------------------------------------
# Registry & Factory
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, type] = {
    "random": RandomStrategy,
    "round_robin": RoundRobinStrategy,
    "shortest_queue": ShortestQueueStrategy,
}


def create_strategy(name: str) -> ResourceSelectionStrategy:
    """
    Create a selection strategy by name.

    Args:
        name: One of "random", "round_robin", "shortest_queue".

    Raises:
        ValueError: If the name is not in the registry.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        valid = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown resource selection strategy: '{name}'. "
            f"Valid options: {valid}"
        )
    return cls()
