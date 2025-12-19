"""
Unit tests for EventQueue.
"""

import pytest
from datetime import datetime, timedelta

from simulation.events import SimulationEvent, EventType
from simulation.event_queue import EventQueue


class TestEventQueue:
    """Tests for the EventQueue class."""
    
    def test_empty_queue(self):
        """Empty queue should return None on pop."""
        queue = EventQueue()
        assert queue.is_empty()
        assert len(queue) == 0
        assert queue.pop() is None
        assert queue.peek() is None
    
    def test_single_event(self):
        """Single event should be retrievable."""
        queue = EventQueue()
        event = SimulationEvent(
            timestamp=datetime(2024, 1, 1, 12, 0),
            event_type=EventType.CASE_ARRIVAL,
            case_id="test_case_1",
        )
        
        queue.schedule(event)
        
        assert not queue.is_empty()
        assert len(queue) == 1
        assert queue.peek() == event
        assert len(queue) == 1  # peek doesn't remove
        
        popped = queue.pop()
        assert popped == event
        assert queue.is_empty()
    
    def test_timestamp_ordering(self):
        """Events should be popped in timestamp order."""
        queue = EventQueue()
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Add events out of order
        event3 = SimulationEvent(
            timestamp=base_time + timedelta(hours=3),
            event_type=EventType.CASE_END,
            case_id="case_3",
        )
        event1 = SimulationEvent(
            timestamp=base_time + timedelta(hours=1),
            event_type=EventType.CASE_ARRIVAL,
            case_id="case_1",
        )
        event2 = SimulationEvent(
            timestamp=base_time + timedelta(hours=2),
            event_type=EventType.ACTIVITY_COMPLETE,
            case_id="case_2",
        )
        
        queue.schedule(event3)
        queue.schedule(event1)
        queue.schedule(event2)
        
        assert len(queue) == 3
        
        # Should come out in timestamp order
        assert queue.pop().case_id == "case_1"
        assert queue.pop().case_id == "case_2"
        assert queue.pop().case_id == "case_3"
        assert queue.is_empty()
    
    def test_same_timestamp(self):
        """Events with same timestamp should all be retrievable."""
        queue = EventQueue()
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        event1 = SimulationEvent(
            timestamp=timestamp,
            event_type=EventType.CASE_ARRIVAL,
            case_id="case_1",
        )
        event2 = SimulationEvent(
            timestamp=timestamp,
            event_type=EventType.CASE_ARRIVAL,
            case_id="case_2",
        )
        
        queue.schedule(event1)
        queue.schedule(event2)
        
        assert len(queue) == 2
        
        # Both should be retrievable
        results = {queue.pop().case_id, queue.pop().case_id}
        assert results == {"case_1", "case_2"}
    
    def test_clear(self):
        """Clear should remove all events."""
        queue = EventQueue()
        
        for i in range(5):
            queue.schedule(SimulationEvent(
                timestamp=datetime.now(),
                event_type=EventType.CASE_ARRIVAL,
                case_id=f"case_{i}",
            ))
        
        assert len(queue) == 5
        queue.clear()
        assert queue.is_empty()
        assert len(queue) == 0
