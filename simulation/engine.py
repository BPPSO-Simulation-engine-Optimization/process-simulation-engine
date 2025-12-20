"""
DES Engine - Discrete Event Simulation Engine for BPIC17.

The main simulation loop that orchestrates:
1. Event Queue (time-ordered processing)
2. Predictors (next activity, processing time, case arrivals)
3. Resource Allocator (who performs the activity)
4. Event Logging (for CSV/XES export)
"""

import uuid
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Protocol

from .events import SimulationEvent, EventType
from .event_queue import EventQueue
from .clock import SimulationClock
from .case_manager import CaseState, CaseManager

logger = logging.getLogger(__name__)


# Protocol definitions for pluggable predictors
class NextActivityPredictor(Protocol):
    """Interface for next activity prediction."""
    def predict(self, case_state: CaseState) -> tuple[str, bool]:
        """
        Predict the next activity for a case.
        
        Args:
            case_state: Current case state.
            
        Returns:
            Tuple of (activity_name, is_case_ended).
        """
        ...


class ProcessingTimePredictor(Protocol):
    """Interface for activity processing time prediction."""
    def predict(self, activity: str, case_state: CaseState) -> timedelta:
        """
        Predict processing time for an activity.
        
        Args:
            activity: Activity name.
            case_state: Current case state.
            
        Returns:
            Predicted processing time.
        """
        ...


class CaseArrivalPredictor(Protocol):
    """Interface for case inter-arrival time prediction."""
    def predict(self) -> timedelta:
        """
        Predict time until next case arrival.
        
        Returns:
            Time delta until next case.
        """
        ...


class CaseAttributePredictor(Protocol):
    """Interface for case attribute prediction."""
    def predict(self) -> tuple[str, str, float]:
        """
        Predict case attributes.
        
        Returns:
            Tuple of (loan_goal, application_type, requested_amount).
        """
        ...


class ResourceAllocator(Protocol):
    """Interface for resource allocation."""
    def allocate(self, activity: str, timestamp: datetime, 
                 case_type: str = None) -> Optional[str]:
        """
        Allocate a resource for an activity.
        
        Returns:
            Resource name or None if unavailable.
        """
        ...


class DESEngine:
    """
    Discrete Event Simulation Engine for BPIC17.
    
    MVP: Uses complete events only.
    
    Flow:
    1. CASE_ARRIVAL -> create case, predict first activity, schedule ACTIVITY_COMPLETE
    2. ACTIVITY_COMPLETE -> log event, predict next -> schedule next or CASE_END
    3. CASE_END -> cleanup
    """
    
    def __init__(
        self,
        resource_allocator: ResourceAllocator,
        next_activity_predictor: NextActivityPredictor = None,
        processing_time_predictor: ProcessingTimePredictor = None,
        case_arrival_predictor: CaseArrivalPredictor = None,
        case_attribute_predictor: CaseAttributePredictor = None,
        start_time: datetime = None,
    ):
        """
        Initialize the DES Engine.
        
        Args:
            resource_allocator: Resource allocation component.
            next_activity_predictor: Predicts next activity (uses stub if None).
            processing_time_predictor: Predicts processing time (uses stub if None).
            case_arrival_predictor: Predicts inter-arrival time (uses stub if None).
            case_attribute_predictor: Predicts case attributes (uses stub if None).
            start_time: Simulation start time.
        """
        self.queue = EventQueue()
        self.clock = SimulationClock(start_time)
        self.case_manager = CaseManager()
        self.allocator = resource_allocator
        
        # Predictors (use stubs if not provided)
        self._next_activity = next_activity_predictor or _StubNextActivityPredictor()
        self._processing_time = processing_time_predictor or _StubProcessingTimePredictor()
        self._case_arrival = case_arrival_predictor or _StubCaseArrivalPredictor()
        self._case_attribute = case_attribute_predictor or _StubCaseAttributePredictor()
        
        # Output: collected events for export
        self.completed_events: List[Dict] = []
        
        # Statistics
        self.stats = {
            'cases_started': 0,
            'cases_completed': 0,
            'events_processed': 0,
            'allocation_failures': 0,
        }
    
    def run(self, num_cases: int = 100, max_time: datetime = None) -> List[Dict]:
        """
        Run the simulation.
        
        Args:
            num_cases: Number of cases to simulate.
            max_time: Optional end time for simulation.
            
        Returns:
            List of event dictionaries for export.
        """
        logger.info(f"Starting simulation: {num_cases} cases")
        
        # Reset state
        self.completed_events.clear()
        self.queue.clear()
        self.case_manager.clear()
        
        # Schedule initial case arrivals
        self._schedule_case_arrivals(num_cases)
        
        # Main simulation loop
        while not self.queue.is_empty():
            event = self.queue.pop()
            
            if max_time and event.timestamp > max_time:
                logger.info(f"Reached max_time: {max_time}")
                break
            
            self.clock.advance_to(event.timestamp)
            self._handle_event(event)
        
        logger.info(
            f"Simulation complete: {self.stats['cases_completed']} cases, "
            f"{len(self.completed_events)} events"
        )
        
        return self.completed_events
    
    def _schedule_case_arrivals(self, num_cases: int) -> None:
        """Schedule initial case arrival events."""
        current_time = self.clock.now
        
        for _ in range(num_cases):
            # Predict inter-arrival time
            inter_arrival = self._case_arrival.predict()
            current_time = current_time + inter_arrival
            
            # Generate case ID
            case_id = f"Application_{random.randint(10000000, 1999999999)}"
            
            # Schedule arrival
            event = SimulationEvent(
                timestamp=current_time,
                event_type=EventType.CASE_ARRIVAL,
                case_id=case_id,
            )
            self.queue.schedule(event)
    
    def _handle_event(self, event: SimulationEvent) -> None:
        """Route event to appropriate handler."""
        self.stats['events_processed'] += 1
        
        handlers = {
            EventType.CASE_ARRIVAL: self._on_case_arrival,
            EventType.ACTIVITY_COMPLETE: self._on_activity_complete,
            EventType.CASE_END: self._on_case_end,
        }
        
        handler = handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            logger.warning(f"Unknown event type: {event.event_type}")
    
    def _on_case_arrival(self, event: SimulationEvent) -> None:
        """Handle case arrival: create case state, schedule first activity."""
        self.stats['cases_started'] += 1
        
        # Predict case attributes
        loan_goal, app_type, amount = self._case_attribute.predict()
        
        # Create case state
        case = self.case_manager.create_case(
            case_id=event.case_id,
            case_type=loan_goal,
            application_type=app_type,
            requested_amount=amount,
            start_time=event.timestamp,
        )
        
        # Predict first activity
        activity, is_end = self._next_activity.predict(case)
        
        if is_end:
            # Edge case: case ends immediately
            self._schedule_case_end(event.case_id, event.timestamp)
            return
        
        # Allocate resource and schedule activity
        self._schedule_activity(event.case_id, activity, event.timestamp, case)
    
    def _on_activity_complete(self, event: SimulationEvent) -> None:
        """Handle activity completion: log event, predict next activity."""
        case = self.case_manager.get_case(event.case_id)
        if not case:
            logger.warning(f"Case not found: {event.case_id}")
            return
        
        # Record activity in case history
        case.add_activity(event.activity, event.resource)
        
        # Log the event for export
        log_record = event.to_log_record()
        log_record.update(case.get_payload())
        self.completed_events.append(log_record)
        
        # Predict next activity
        next_activity, is_end = self._next_activity.predict(case)
        
        if is_end:
            self._schedule_case_end(event.case_id, event.timestamp)
        else:
            self._schedule_activity(event.case_id, next_activity, event.timestamp, case)
    
    def _on_case_end(self, event: SimulationEvent) -> None:
        """Handle case end: cleanup."""
        self.stats['cases_completed'] += 1
        self.case_manager.remove_case(event.case_id)
    
    def _schedule_activity(self, case_id: str, activity: str, 
                           current_time: datetime, case: CaseState) -> None:
        """Allocate resource and schedule activity completion."""
        # Allocate resource
        resource = self.allocator.allocate(
            activity=activity,
            timestamp=current_time,
            case_type=case.case_type,
        )
        
        if resource is None:
            self.stats['allocation_failures'] += 1
            # Fallback: use generic resource
            resource = "User_1"
            logger.debug(f"Allocation failed for {activity}, using fallback")
        
        # Predict processing time
        processing_time = self._processing_time.predict(activity, case)
        completion_time = current_time + processing_time
        
        # Schedule completion event
        event = SimulationEvent(
            timestamp=completion_time,
            event_type=EventType.ACTIVITY_COMPLETE,
            case_id=case_id,
            activity=activity,
            resource=resource,
            payload=case.get_payload(),
        )
        self.queue.schedule(event)
    
    def _schedule_case_end(self, case_id: str, timestamp: datetime) -> None:
        """Schedule case end event."""
        event = SimulationEvent(
            timestamp=timestamp,
            event_type=EventType.CASE_END,
            case_id=case_id,
        )
        self.queue.schedule(event)


# Stub predictors for testing/fallback
class _StubNextActivityPredictor:
    """Stub that returns a simple activity sequence."""
    
    ACTIVITIES = [
        "A_Create Application",
        "A_Submitted",
        "W_Complete application",
        "A_Concept",
        "A_Accepted",
        "O_Create Offer",
        "A_Complete",
    ]
    
    def predict(self, case_state: CaseState) -> tuple[str, bool]:
        history_len = len(case_state.activity_history)
        if history_len >= len(self.ACTIVITIES):
            return self.ACTIVITIES[-1], True  # End after sequence
        return self.ACTIVITIES[history_len], False


class _StubProcessingTimePredictor:
    """Stub that returns random processing times."""
    
    def predict(self, activity: str, case_state: CaseState) -> timedelta:
        # Random 5-60 minutes
        minutes = random.randint(5, 60)
        return timedelta(minutes=minutes)


class _StubCaseArrivalPredictor:
    """Stub that returns random inter-arrival times."""
    
    def predict(self) -> timedelta:
        # Random 1-30 minutes between cases
        minutes = random.randint(1, 30)
        return timedelta(minutes=minutes)


class _StubCaseAttributePredictor:
    """Stub that returns random case attributes."""
    
    LOAN_GOALS = ["Home improvement", "Car", "Existing loan takeover", "Other"]
    APP_TYPES = ["New credit", "Limit raise"]
    
    def predict(self) -> tuple[str, str, float]:
        return (
            random.choice(self.LOAN_GOALS),
            random.choice(self.APP_TYPES),
            random.uniform(5000, 50000),
        )


# Main block for testing
if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime

    # Add project root to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from resources import ResourceAllocator
    from simulation.log_exporter import LogExporter

    # Enable logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("Starting simulation...")

    # Initialize resource allocator
    log_path = "eventlog/eventlog.xes.gz"
    print(f"Loading ResourceAllocator from {log_path}...")
    allocator = ResourceAllocator(log_path=log_path)

    # Run simulation (use 2016 start time to match availability model training data: Jan 2016 - Feb 2017)
    start_time = datetime(2016, 1, 4, 8, 0)  # Monday 8am, Jan 2016
    engine = DESEngine(
        resource_allocator=allocator,
        start_time=start_time,
    )

    events = engine.run(num_cases=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(events)} events for 2 cases")
    print(f"Stats: {engine.stats}")
    print(f"{'='*60}\n")

    # Export to output folder
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "simulated_log.csv")
    xes_path = os.path.join(output_dir, "simulated_log.xes")

    LogExporter.to_csv(events, csv_path)
    print(f"Exported CSV to: {csv_path}")

    LogExporter.to_xes(events, xes_path)
    print(f"Exported XES to: {xes_path}")

    # Show events grouped by case
    from collections import defaultdict
    by_case = defaultdict(list)
    for e in events:
        by_case[e['case:concept:name']].append(e)

    for case_id, case_events in by_case.items():
        print(f"\nCase: {case_id}")
        print(f"  LoanGoal: {case_events[0].get('case:LoanGoal')}")
        print(f"  Activities:")
        for e in case_events:
            ts = e['time:timestamp'].strftime('%H:%M')
            print(f"    [{ts}] {e['concept:name']} (by {e['org:resource']})")
        print()
