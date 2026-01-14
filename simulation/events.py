"""
Simulation Events - Core event types for the DES engine.

MVP: Using only complete lifecycle for simplicity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(Enum):
    """Internal simulation event types."""
    CASE_ARRIVAL = "case_arrival"
    ACTIVITY_COMPLETE = "activity_complete"
    CASE_END = "case_end"


@dataclass(order=True)
class SimulationEvent:
    """
    Priority queue-compatible simulation event.
    
    Events are ordered by timestamp for the priority queue.
    Other fields are excluded from comparison.
    """
    timestamp: datetime
    event_type: EventType = field(compare=False)
    case_id: str = field(compare=False)
    activity: Optional[str] = field(default=None, compare=False)
    resource: Optional[str] = field(default=None, compare=False)
    payload: dict = field(default_factory=dict, compare=False)
    
    def to_log_record(self) -> dict:
        """Convert event to a log record for export."""
        record = {
            'case:concept:name': self.case_id,
            'concept:name': self.activity,
            'org:resource': self.resource,
            'time:timestamp': self.timestamp,
            'lifecycle:transition': 'complete',  # MVP: always complete
        }
        # Merge payload (case attributes)
        record.update(self.payload)
        return record
