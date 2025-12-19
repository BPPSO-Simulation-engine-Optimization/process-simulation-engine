"""
Case Manager - Tracks active case state during simulation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class CaseState:
    """
    State of an active case during simulation.
    
    Tracks case attributes and activity history.
    """
    case_id: str
    case_type: str  # LoanGoal: "Home improvement", "Car", etc.
    application_type: str  # "New credit", "Limit raise"
    requested_amount: float
    # hier auch noch die offer lvl attributes? wie gehen wir damit um, dass es pro case pot. mehrere offers gibt?
    
    # Runtime state
    activity_history: List[str] = field(default_factory=list)
    current_activity: Optional[str] = None
    current_resource: Optional[str] = None
    start_time: Optional[datetime] = None
    
    def add_activity(self, activity: str, resource: str = None) -> None:
        """Record a completed activity."""
        self.activity_history.append(activity)
        self.current_activity = activity
        if resource:
            self.current_resource = resource
    
    def get_payload(self) -> Dict:
        """Get case attributes for event payload."""
        return {
            'case:LoanGoal': self.case_type,
            'case:ApplicationType': self.application_type,
            'case:RequestedAmount': self.requested_amount,
        }


class CaseManager:
    """Manages active cases during simulation."""
    
    def __init__(self):
        self._cases: Dict[str, CaseState] = {}
    
    def create_case(self, case_id: str, case_type: str, 
                    application_type: str, requested_amount: float,
                    start_time: datetime = None) -> CaseState:
        """Create and register a new case."""
        case = CaseState(
            case_id=case_id,
            case_type=case_type,
            application_type=application_type,
            requested_amount=requested_amount,
            start_time=start_time,
        )
        self._cases[case_id] = case
        return case
    
    def get_case(self, case_id: str) -> Optional[CaseState]:
        """Get a case by ID."""
        return self._cases.get(case_id)
    
    def remove_case(self, case_id: str) -> Optional[CaseState]:
        """Remove and return a case."""
        return self._cases.pop(case_id, None)
    
    def active_count(self) -> int:
        """Number of active cases."""
        return len(self._cases)
    
    def clear(self) -> None:
        """Remove all cases."""
        self._cases.clear()
