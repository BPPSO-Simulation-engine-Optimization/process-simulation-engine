"""
Case Manager - Tracks active case state during simulation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


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

    # Offer-level attributes (populated when O_Create Offer occurs)
    credit_score: Optional[float] = None
    offered_amount: Optional[float] = None
    first_withdrawal_amount: Optional[float] = None
    number_of_terms: Optional[int] = None
    monthly_cost: Optional[float] = None
    selected: Optional[bool] = None
    accepted: Optional[bool] = None

    # Reference to attribute engine's case state for lazy evaluation
    _attr_engine_case: Any = field(default=None, repr=False)

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
        payload = {
            'case:LoanGoal': self.case_type,
            'case:ApplicationType': self.application_type,
            'case:RequestedAmount': self.requested_amount,
        }
        # Add offer-level attributes if they've been generated
        if self.credit_score is not None:
            payload['CreditScore'] = self.credit_score
        if self.offered_amount is not None:
            payload['OfferedAmount'] = self.offered_amount
        if self.first_withdrawal_amount is not None:
            payload['FirstWithdrawalAmount'] = self.first_withdrawal_amount
        if self.number_of_terms is not None:
            payload['NumberOfTerms'] = self.number_of_terms
        if self.monthly_cost is not None:
            payload['MonthlyCost'] = self.monthly_cost
        if self.selected is not None:
            payload['Selected'] = self.selected
        if self.accepted is not None:
            payload['Accepted'] = self.accepted
        return payload


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
