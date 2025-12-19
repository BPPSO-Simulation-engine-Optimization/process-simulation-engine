"""
Unit tests for DESEngine.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from simulation.engine import DESEngine
from simulation.events import EventType
from simulation.case_manager import CaseState


class MockResourceAllocator:
    """Mock allocator that returns a fixed resource."""
    
    def __init__(self, resource: str = "User_42"):
        self.resource = resource
        self.calls = []
    
    def allocate(self, activity: str, timestamp: datetime, case_type: str = None):
        self.calls.append((activity, timestamp, case_type))
        return self.resource


class TestDESEngine:
    """Tests for the DESEngine class."""
    
    def test_single_case_simulation(self):
        """Single case should complete with expected events."""
        allocator = MockResourceAllocator()
        engine = DESEngine(
            resource_allocator=allocator,
            start_time=datetime(2024, 1, 1, 9, 0),
        )
        
        events = engine.run(num_cases=1)
        
        # Should have generated events
        assert len(events) > 0
        
        # All events should have required XES columns
        for event in events:
            assert 'case:concept:name' in event
            assert 'concept:name' in event
            assert 'time:timestamp' in event
            assert 'org:resource' in event
        
        # Stats should reflect completion
        assert engine.stats['cases_started'] == 1
        assert engine.stats['cases_completed'] == 1
    
    def test_multiple_cases(self):
        """Multiple cases should run concurrently."""
        allocator = MockResourceAllocator()
        engine = DESEngine(
            resource_allocator=allocator,
            start_time=datetime(2024, 1, 1, 9, 0),
        )
        
        events = engine.run(num_cases=5)
        
        # Should have events from all cases
        case_ids = set(e['case:concept:name'] for e in events)
        assert len(case_ids) == 5
        
        assert engine.stats['cases_started'] == 5
        assert engine.stats['cases_completed'] == 5
    
    def test_max_time_limit(self):
        """Simulation should stop at max_time."""
        allocator = MockResourceAllocator()
        start = datetime(2024, 1, 1, 9, 0)
        engine = DESEngine(
            resource_allocator=allocator,
            start_time=start,
        )
        
        # Stop after 1 hour
        max_time = start + timedelta(hours=1)
        events = engine.run(num_cases=100, max_time=max_time)
        
        # Should have some events but not complete all 100 cases
        assert len(events) > 0
        for event in events:
            assert event['time:timestamp'] <= max_time
    
    def test_resource_allocation_called(self):
        """Allocator should be called for each activity."""
        allocator = MockResourceAllocator()
        engine = DESEngine(
            resource_allocator=allocator,
            start_time=datetime(2024, 1, 1, 9, 0),
        )
        
        engine.run(num_cases=1)
        
        # Allocator should have been called at least once per activity
        assert len(allocator.calls) > 0
        
        # Each call should have activity and timestamp
        for activity, timestamp, case_type in allocator.calls:
            assert activity is not None
            assert timestamp is not None
    
    def test_allocation_failure_uses_fallback(self):
        """Allocation failure should use fallback resource."""
        # Allocator that returns None (failure)
        allocator = Mock()
        allocator.allocate = Mock(return_value=None)
        
        engine = DESEngine(
            resource_allocator=allocator,
            start_time=datetime(2024, 1, 1, 9, 0),
        )
        
        events = engine.run(num_cases=1)
        
        # Should still complete with fallback resource
        assert len(events) > 0
        assert engine.stats['allocation_failures'] > 0
        
        # Events should have fallback resource
        for event in events:
            assert event['org:resource'] == "User_1"
    
    def test_event_timestamps_monotonic(self):
        """Event timestamps should be monotonically increasing per case."""
        allocator = MockResourceAllocator()
        engine = DESEngine(
            resource_allocator=allocator,
            start_time=datetime(2024, 1, 1, 9, 0),
        )
        
        events = engine.run(num_cases=3)
        
        # Group by case and check ordering
        from collections import defaultdict
        by_case = defaultdict(list)
        for e in events:
            by_case[e['case:concept:name']].append(e['time:timestamp'])
        
        for case_id, timestamps in by_case.items():
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i-1], \
                    f"Timestamps not monotonic for {case_id}"
    
    def test_case_attributes_in_events(self):
        """Events should include case attributes."""
        allocator = MockResourceAllocator()
        engine = DESEngine(
            resource_allocator=allocator,
            start_time=datetime(2024, 1, 1, 9, 0),
        )
        
        events = engine.run(num_cases=1)
        
        # Each event should have case attributes
        for event in events:
            assert 'case:LoanGoal' in event
            assert 'case:ApplicationType' in event
            assert 'case:RequestedAmount' in event
