"""
Event Queue - Priority queue for simulation events.

Uses heapq for O(log n) insert and O(log n) pop.
Events are ordered by timestamp.
"""

import heapq
from typing import Optional, List

from .events import SimulationEvent


class EventQueue:
    """Priority queue for simulation events, ordered by timestamp."""
    
    def __init__(self):
        self._queue: List[SimulationEvent] = []
    
    def schedule(self, event: SimulationEvent) -> None:
        """Add an event to the queue."""
        heapq.heappush(self._queue, event)
    
    def pop(self) -> Optional[SimulationEvent]:
        """Remove and return the earliest event, or None if empty."""
        if self._queue:
            return heapq.heappop(self._queue)
        return None
    
    def peek(self) -> Optional[SimulationEvent]:
        """Return the earliest event without removing it."""
        if self._queue:
            return self._queue[0]
        return None
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._queue) == 0
    
    def __len__(self) -> int:
        """Return the number of events in the queue."""
        return len(self._queue)
    
    def clear(self) -> None:
        """Remove all events from the queue."""
        self._queue.clear()
