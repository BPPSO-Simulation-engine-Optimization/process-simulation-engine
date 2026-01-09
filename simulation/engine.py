"""
DES Engine - Discrete Event Simulation Engine for BPIC17.

The main simulation loop that orchestrates:
1. Event Queue (time-ordered processing)
2. Predictors (next activity, processing time, case arrivals)
3. Resource Allocator (who performs the activity)
4. Resource Pool (dynamic busy tracking + waiting queues)
5. Event Logging (for CSV/XES export)

Resource Allocation Model:
- When an activity needs a resource, we check:
  1. Eligibility (permission model - who CAN do this activity)
  2. Availability (working hours - who is ON DUTY at this time)
  3. Busy state (dynamic - who is NOT currently working on another activity)

- If no resource is available:
  - Work is added to a per-activity waiting queue (FIFO)
  - NO fallback to User_1 or other default resource

- When an activity completes:
  - The resource is released
  - The waiting queue is checked for work this resource can handle
  - Waiting work is dispatched to the freed resource

This creates realistic resource contention and waiting times.
"""

import uuid
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Protocol, Set
from collections import defaultdict
import heapq

import pandas as pd

from .events import SimulationEvent, EventType
from .event_queue import EventQueue
from .clock import SimulationClock
from .case_manager import CaseState, CaseManager

logger = logging.getLogger(__name__)


@dataclass
class WaitingWork:
    """Represents work waiting for a resource."""
    case_id: str
    # Original activity label used for logging/simulation semantics
    activity: str
    # Normalized label used for permission/availability checks and queue routing
    allocation_activity: str
    arrival_time: datetime  # When the work arrived (for FIFO ordering)
    case_state: CaseState

    def __lt__(self, other):
        """For heap ordering - earlier arrival time = higher priority."""
        return self.arrival_time < other.arrival_time


class ResourcePool:
    """
    Tracks resource busy state during simulation.

    Manages:
    - Which resources are currently busy
    - When resources will become free
    - Queue of work waiting for resources
    """

    def __init__(self, availability_model=None):
        """
        Initialize the resource pool.

        Args:
            availability_model: The availability model for checking working hours.
        """
        # resource_id -> (busy_until, case_id, activity)
        self._busy_resources: Dict[str, tuple] = {}

        # allocation_activity -> list of WaitingWork (heap ordered by arrival time)
        self._waiting_queues: Dict[str, List[WaitingWork]] = defaultdict(list)

        # Reference to availability model for working hours checks
        self._availability = availability_model

        # Stats
        self.stats = {
            'total_waits': 0,
            'max_queue_length': 0,
            'total_wait_time_seconds': 0,
        }

    def is_busy(self, resource_id: str, current_time: datetime) -> bool:
        """Check if a resource is currently busy."""
        if resource_id not in self._busy_resources:
            return False
        busy_until, _, _ = self._busy_resources[resource_id]
        if current_time >= busy_until:
            # Resource has finished, clean up
            del self._busy_resources[resource_id]
            return False
        return True

    def mark_busy(self, resource_id: str, until: datetime,
                  case_id: str, activity: str) -> None:
        """Mark a resource as busy until a given time."""
        self._busy_resources[resource_id] = (until, case_id, activity)

    def release(self, resource_id: str) -> None:
        """Release a resource (mark as free)."""
        if resource_id in self._busy_resources:
            del self._busy_resources[resource_id]

    def get_busy_until(self, resource_id: str) -> Optional[datetime]:
        """Get the time when a resource will become free."""
        if resource_id in self._busy_resources:
            return self._busy_resources[resource_id][0]
        return None

    def add_to_waiting_queue(self, work: WaitingWork) -> None:
        """Add work to the waiting queue for its activity."""
        heapq.heappush(self._waiting_queues[work.allocation_activity], work)
        self.stats['total_waits'] += 1
        queue_len = len(self._waiting_queues[work.allocation_activity])
        if queue_len > self.stats['max_queue_length']:
            self.stats['max_queue_length'] = queue_len

    def get_waiting_work(self, activity: str) -> Optional[WaitingWork]:
        """Get the next waiting work for an activity (FIFO)."""
        if activity in self._waiting_queues and self._waiting_queues[activity]:
            return heapq.heappop(self._waiting_queues[activity])
        return None

    def has_waiting_work(self, activity: str = None) -> bool:
        """Check if there's waiting work (optionally for a specific activity)."""
        if activity:
            return bool(self._waiting_queues.get(activity))
        return any(q for q in self._waiting_queues.values())

    def get_all_waiting_activities(self) -> Set[str]:
        """Get all activities that have waiting work."""
        return {act for act, q in self._waiting_queues.items() if q}

    def peek_waiting_work(self, activity: str) -> Optional[WaitingWork]:
        """Peek at the next waiting work without removing it."""
        if activity in self._waiting_queues and self._waiting_queues[activity]:
            return self._waiting_queues[activity][0]
        return None

    def get_available_resources(self, resources: List[str],
                                 current_time: datetime) -> List[str]:
        """Filter resources to only those not currently busy."""
        return [r for r in resources if not self.is_busy(r, current_time)]

    def get_total_waiting_count(self) -> int:
        """Get total number of cases waiting across all activities."""
        return sum(len(q) for q in self._waiting_queues.values())

    def get_waiting_summary(self) -> Dict[str, int]:
        """Get summary of waiting work per activity."""
        return {act: len(q) for act, q in self._waiting_queues.items() if q}


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
    """Interface for activity processing time prediction (ProcessingTimePredictionClass)."""
    def predict(
        self,
        prev_activity: str,
        prev_lifecycle: str,
        curr_activity: str,
        curr_lifecycle: str,
        context: dict = None,
    ) -> float:
        """
        Predict processing time for a transition.

        Args:
            prev_activity: Previous activity name.
            prev_lifecycle: Previous lifecycle transition.
            curr_activity: Current/next activity name.
            curr_lifecycle: Current/next lifecycle transition.
            context: Optional context dictionary.

        Returns:
            Predicted processing time in seconds.
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
    """Interface for case attribute prediction (AttributeSimulationEngine)."""
    def start_new_case(self):
        """
        Start a new case and return a CaseState with attributes.

        Returns:
            CaseState with loan_goal, application_type, requested_amount.
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
        arrival_timestamps: List[datetime] = None,
        next_activity_predictor: NextActivityPredictor = None,
        processing_time_predictor: ProcessingTimePredictor = None,
        case_arrival_predictor: CaseArrivalPredictor = None,
        case_attribute_predictor: CaseAttributePredictor = None,
        start_time: datetime = None,
        max_activities_per_case: int = 500,
    ):
        """
        Initialize the DES Engine.
        
        Args:
            resource_allocator: Resource allocation component.
            arrival_timestamps: Pre-generated list of case arrival timestamps.
                If provided, overrides case_arrival_predictor.
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
        
        # Pre-generated arrival timestamps (optional)
        self._arrival_timestamps = arrival_timestamps
        
        # Predictors
        self._next_activity = next_activity_predictor or self._create_next_activity_predictor()
        self._case_arrival = case_arrival_predictor or _StubCaseArrivalPredictor()

        # Processing time predictor is required (must be ProcessingTimePredictionClass)
        if processing_time_predictor is None:
            raise ValueError(
                "processing_time_predictor is required. "
                "Use ProcessingTimePredictionClass from processing_time_prediction"
            )
        self._processing_time = processing_time_predictor

        # Case attribute predictor is required (must be AttributeSimulationEngine)
        if case_attribute_predictor is None:
            raise ValueError(
                "case_attribute_predictor is required. "
                "Use AttributeSimulationEngine from case_attribute_prediction.simulator"
            )
        self._case_attribute = case_attribute_predictor

        # Safety: prevent infinite loops in process graph simulation
        self._max_activities_per_case = max_activities_per_case

        # Resource pool for dynamic busy tracking and waiting queue
        self.resource_pool = ResourcePool(
            availability_model=resource_allocator.availability if resource_allocator else None
        )

        # Output: collected events for export
        self.completed_events: List[Dict] = []

        # Statistics
        self.stats = {
            'cases_started': 0,
            'cases_completed': 0,
            'events_processed': 0,
            'no_eligible_failures': 0,  # Permission model gaps (actual problem)
            'outside_hours_count': 0,   # Expected - resources not working at this time
            'waiting_events': 0,  # Cases that had to wait for resources
            'wait_time_total_seconds': 0,  # Total time spent waiting
        }
    
    def _create_next_activity_predictor(self):
        """
        Auto-load next activity predictor based on available models.
        
        Priority:
        1. LSTM next activity model (models/next_activity_lstm/)
        2. ProcessSequencePrediction model (ProcessSequencePrediction-master/code/output_files/models/)
        3. BranchPredictor model (models/branch_predictor.joblib)
        4. Stub predictor (fallback)
        """
        from pathlib import Path
        
        # Try LSTM next activity model
        # Check both project root and next_activity_prediction module locations
        possible_paths = [
            Path("models/next_activity_lstm"),
            Path("next_activity_prediction/models/next_activity_lstm"),
        ]
        
        for lstm_model_dir in possible_paths:
            model_file = lstm_model_dir / "model.keras"
            checkpoint_file = lstm_model_dir / "checkpoints" / "best_model.keras"
            
            if lstm_model_dir.exists() and (model_file.exists() or checkpoint_file.exists()):
                try:
                    logger.info(f"Loading LSTMNextActivityPredictor from {lstm_model_dir}...")
                    from next_activity_prediction import LSTMNextActivityPredictor
                    # Use checkpoint if available, otherwise use model.keras
                    if checkpoint_file.exists():
                        logger.info(f"Using checkpoint: {checkpoint_file}")
                        return LSTMNextActivityPredictor(str(checkpoint_file))
                    elif model_file.exists():
                        logger.info(f"Using model: {model_file}")
                        return LSTMNextActivityPredictor(str(lstm_model_dir))
                except Exception as e:
                    logger.warning(f"Could not load LSTM predictor from {lstm_model_dir}: {e}")
                    continue
        
        # Try ProcessSequencePrediction model
        psp_model_dir = Path(ProcessSequencePredictor.DEFAULT_MODEL_PATH)
        if psp_model_dir.exists():
            model_files = list(psp_model_dir.glob("*.h5"))
            if model_files:
                try:
                    logger.info("Loading ProcessSequencePredictor...")
                    return ProcessSequencePredictor(str(psp_model_dir))
                except Exception as e:
                    logger.warning(f"Could not load ProcessSequencePredictor: {e}")
        
        # Fallback to BranchPredictor
        model_path = Path(BranchNextActivityPredictor.DEFAULT_MODEL_PATH)
        if model_path.exists():
            try:
                return BranchNextActivityPredictor(str(model_path))
            except Exception as e:
                logger.warning(f"Failed to load next activity model: {e}")
        
        # Last resort: stub
        logger.info("Using stub next activity predictor")
        return _StubNextActivityPredictor()
    
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
        self.resource_pool = ResourcePool(
            availability_model=self.allocator.availability if self.allocator else None
        )
        self.stats = {
            'cases_started': 0,
            'cases_completed': 0,
            'events_processed': 0,
            'no_eligible_failures': 0,
            'outside_hours_count': 0,
            'waiting_events': 0,
            'wait_time_total_seconds': 0,
        }

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

        # Drain phase: process remaining waiting work by advancing time
        if self.resource_pool.has_waiting_work():
            logger.info("Starting drain phase for waiting work...")
            self._drain_waiting_queues(max_time=max_time)

        # Check for stuck cases
        pending_count = self.resource_pool.get_total_waiting_count()
        if pending_count > 0:
            pending_summary = self.resource_pool.get_waiting_summary()
            logger.warning(
                f"Simulation ended with {pending_count} cases still waiting for resources! "
                f"Breakdown: {pending_summary}"
            )
            self.stats['stuck_cases'] = pending_count
            self.stats['stuck_cases_by_activity'] = pending_summary

        logger.info(
            f"Simulation complete: {self.stats['cases_completed']} cases, "
            f"{len(self.completed_events)} events, "
            f"{self.stats['waiting_events']} waits, "
            f"{self.stats['outside_hours_count']} outside hours, "
            f"{self.stats['no_eligible_failures']} no eligible"
            + (f", {pending_count} stuck" if pending_count > 0 else "")
        )

        return self.completed_events
    
    def _schedule_case_arrivals(self, num_cases: int) -> None:
        """Schedule initial case arrival events."""
        current_time = self.clock.now
        
        # 1. Use pre-generated timestamps if provided
        if self._arrival_timestamps:
            # Sort to ensure chronological order
            timestamps = sorted(self._arrival_timestamps)
            # Use all timestamps unless num_cases was explicitly set to limit (if logic dictates)
            # engine.run(num_cases=X) implies we want X cases total. 
            # If timestamps provided, we strictly use them up to num_cases or all of them.
            # But run() calls this with num_cases as arg, so let's respect that limit if timestamps > num_cases
            # However, usually timestamps are generated for specific count.
            
            # Note: engine.run() logic checks num_cases argument. 
            # If arrival_timestamps is set, num_cases might be ignored or used as limit.
            # Let's use up to min(len, num_cases) but usually list determines count.
            count = min(len(timestamps), num_cases) if num_cases else len(timestamps)
            
            for i in range(count):
                ts = timestamps[i]
                case_id = f"Application_{random.randint(10000000, 1999999999)}"
                event = SimulationEvent(
                    timestamp=ts,
                    event_type=EventType.CASE_ARRIVAL,
                    case_id=case_id,
                )
                self.queue.schedule(event)
            return

        # 2. Use predictor
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

        # Get case attributes from AttributeSimulationEngine
        attr_case = self._case_attribute.start_new_case()
        loan_goal = attr_case.loan_goal
        app_type = attr_case.application_type
        amount = attr_case.requested_amount

        # Create case state
        case = self.case_manager.create_case(
            case_id=event.case_id,
            case_type=loan_goal,
            application_type=app_type,
            requested_amount=amount,
            start_time=event.timestamp,
        )
        # Store reference to attr engine case for later offer attribute generation
        case._attr_engine_case = attr_case

        # Predict first activity
        activity, is_end = self._next_activity.predict(case)

        if is_end:
            # Edge case: case ends immediately
            self._schedule_case_end(event.case_id, event.timestamp)
            return

        # Allocate resource and schedule activity
        self._schedule_activity(event.case_id, activity, event.timestamp, case)
    
    def _on_activity_complete(self, event: SimulationEvent) -> None:
        """Handle activity completion: log event, release resource, process waiting queue."""
        case = self.case_manager.get_case(event.case_id)
        if not case:
            logger.warning(f"Case not found: {event.case_id}")
            return

        # Release the resource that completed this activity
        if event.resource:
            self.resource_pool.release(event.resource)
            # Try to dispatch waiting work now that this resource is free
            self._process_waiting_queue(event.resource, event.timestamp)

        # Generate offer-dependent attributes when O_Create Offer completes
        if event.activity == "O_Create Offer" and case._attr_engine_case is not None:
            # Populate offer attributes directly on the stored case reference
            # (uses explicit CaseState, not internal _active_case pointer)
            self._case_attribute.populate_offer_attributes(case._attr_engine_case)
            attr = case._attr_engine_case
            # Use pd.notna() for proper NaN handling (np.nan is NOT None)
            case.credit_score = float(attr.credit_score) if pd.notna(attr.credit_score) else None
            case.offered_amount = float(attr.offered_amount) if pd.notna(attr.offered_amount) else None
            case.first_withdrawal_amount = float(attr.first_withdrawal_amount) if pd.notna(attr.first_withdrawal_amount) else None
            case.number_of_terms = int(attr.number_of_terms) if pd.notna(attr.number_of_terms) else None
            case.monthly_cost = float(attr.monthly_cost) if pd.notna(attr.monthly_cost) else None
            case.selected = attr.selected
            case.accepted = attr.accepted

        # Record activity in case history
        case.add_activity(event.activity, event.resource)

        # Safety guard: if a case keeps looping, stop it instead of hanging the run
        if self._max_activities_per_case and len(case.activity_history) >= self._max_activities_per_case:
            logger.warning(
                f"Case {event.case_id} hit max_activities_per_case={self._max_activities_per_case}; "
                f"ending to avoid infinite loop (last={case.activity_history[-1]})."
            )
            log_record = event.to_log_record()
            log_record.update(case.get_payload())
            self.completed_events.append(log_record)
            self._schedule_case_end(event.case_id, event.timestamp)
            return

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

    def _process_waiting_queue(self, freed_resource: str, current_time: datetime) -> None:
        """
        Process waiting queue when a resource becomes free.

        Tries to dispatch waiting work to the freed resource if it's eligible.
        """
        # Check which activities have waiting work
        waiting_activities = self.resource_pool.get_all_waiting_activities()
        if not waiting_activities:
            return

        # Check which activities this resource is eligible for
        for allocation_activity in waiting_activities:
            # Check if freed resource is eligible for this activity
            try:
                eligible = self.allocator.permissions.get_eligible_resources(
                    allocation_activity, timestamp=current_time
                )
            except TypeError:
                eligible = self.allocator.permissions.get_eligible_resources(allocation_activity)

            if freed_resource not in eligible:
                continue

            # Check if resource is available (working hours)
            if not self.allocator.availability.is_available(freed_resource, current_time):
                continue

            # Found matching work - dispatch it
            waiting_work = self.resource_pool.get_waiting_work(allocation_activity)
            if waiting_work:
                # Calculate wait time for stats
                wait_seconds = (current_time - waiting_work.arrival_time).total_seconds()
                self.stats['wait_time_total_seconds'] += wait_seconds

                logger.debug(
                    f"Dispatching waiting {waiting_work.activity} for case {waiting_work.case_id} "
                    f"to {freed_resource} (waited {wait_seconds:.0f}s)"
                )

                # Schedule the activity with the freed resource
                self._schedule_activity_with_resource(
                    waiting_work.case_id,
                    waiting_work.activity,
                    current_time,
                    waiting_work.case_state,
                    freed_resource,
                )
                # Resource is now busy again, stop looking
                return
    
    def _on_case_end(self, event: SimulationEvent) -> None:
        """Handle case end: cleanup."""
        self.stats['cases_completed'] += 1
        self.case_manager.remove_case(event.case_id)
    
    def _schedule_activity(self, case_id: str, activity: str,
                           current_time: datetime, case: CaseState) -> None:
        """Allocate resource and schedule activity completion, or queue if unavailable."""
        # Some activities are control-flow artifacts (e.g., decision points) and must not
        # require an organizational resource.
        if not self._activity_requires_resource(activity):
            self._schedule_activity_without_resource(case_id, activity, current_time, case)
            return

        allocation_activity = self._normalize_activity_for_resources(activity)

        # Try to allocate a resource (with dynamic busy checking)
        resource, failure_reason = self._try_allocate_resource_with_reason(allocation_activity, current_time, case)

        if resource is None:
            # No resource available - add to waiting queue
            self.stats['waiting_events'] += 1
            waiting_work = WaitingWork(
                case_id=case_id,
                activity=activity,
                allocation_activity=allocation_activity,
                arrival_time=current_time,
                case_state=case,
            )
            self.resource_pool.add_to_waiting_queue(waiting_work)

            # Log the reason for waiting
            if failure_reason == 'no_eligible':
                logger.warning(
                    f"No eligible resources for activity '{activity}' - case {case_id} may be stuck. "
                    f"Check permission model configuration."
                )
            else:
                logger.debug(
                    f"No resource for {activity} at {current_time} ({failure_reason}), "
                    f"queued case {case_id}"
                )
            return

        # Resource allocated - schedule the activity
        self._schedule_activity_with_resource(
            case_id, activity, current_time, case, resource
        )

    def _activity_requires_resource(self, activity: Optional[str]) -> bool:
        """Return True if this activity needs an org resource assignment."""
        if not activity:
            return True
        # Decision points / process gateways are not executed by a human resource.
        return not (activity.startswith('DP ') or activity.startswith('PG '))

    def _normalize_activity_for_resources(self, activity: str) -> str:
        """Normalize activity labels to match the resource permission model."""
        # Do not normalize control-flow artifacts (already treated as resource-free)
        if not activity or activity.startswith('DP ') or activity.startswith('PG '):
            return activity

        # Many parts of this project generate duplicate labels to avoid loops
        # (e.g., "W_Complete application 1", "A_Cancelled 2").
        # The permission model is typically trained on the base label.
        parts = activity.rsplit(' ', 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0].strip()
            return base or activity
        return activity

    def _schedule_activity_without_resource(
        self,
        case_id: str,
        activity: str,
        current_time: datetime,
        case: CaseState,
    ) -> None:
        """Schedule an activity completion without allocating any resource."""
        prev_activity = case.activity_history[-1] if case.activity_history else "START"

        # Many learned processing-time models won't know DP/PG labels.
        # For control-flow artifacts, treat duration as instantaneous.
        processing_seconds = 0.0
        try:
            processing_seconds = float(self._processing_time.predict(
                prev_activity=prev_activity,
                prev_lifecycle="complete",
                curr_activity=activity,
                curr_lifecycle="complete",
                context={
                    'hour': current_time.hour,
                    'weekday': current_time.weekday(),
                    'month': current_time.month,
                    'day_of_year': current_time.timetuple().tm_yday,
                    'case:LoanGoal': case.case_type,
                    'case:ApplicationType': case.application_type,
                    'event_position_in_case': len(case.activity_history) + 1,
                    'case_duration_so_far': (current_time - case.start_time).total_seconds() if case.start_time else 0.0,
                    'resource_1': case.current_resource or 'unknown',
                    'resource_2': 'none',
                    'Accepted': case.accepted,
                    'Selected': case.selected,
                },
            ))
        except Exception:
            processing_seconds = 0.0

        completion_time = current_time + timedelta(seconds=processing_seconds)

        event = SimulationEvent(
            timestamp=completion_time,
            event_type=EventType.ACTIVITY_COMPLETE,
            case_id=case_id,
            activity=activity,
            resource=None,
            payload=case.get_payload(),
        )
        self.queue.schedule(event)

    def _try_allocate_resource_with_reason(self, activity: str, timestamp: datetime,
                                            case: CaseState) -> tuple:
        """
        Try to allocate a resource, returning (resource, failure_reason).

        failure_reason is one of: None (success), 'no_eligible', 'outside_hours', 'all_busy'
        """
        # Get eligible resources from permission model
        try:
            eligible_resources = self.allocator.permissions.get_eligible_resources(
                activity, timestamp=timestamp, case_type=case.case_type
            )
        except TypeError:
            eligible_resources = self.allocator.permissions.get_eligible_resources(activity)

        if not eligible_resources:
            self.stats['no_eligible_failures'] += 1
            return None, 'no_eligible'

        # Filter by availability model (working hours, holidays, etc.)
        available_by_hours = [
            res for res in eligible_resources
            if self.allocator.availability.is_available(res, timestamp)
        ]

        if not available_by_hours:
            # No one working at this time (expected behavior)
            self.stats['outside_hours_count'] += 1
            return None, 'outside_hours'

        # Filter by dynamic busy state
        truly_available = self.resource_pool.get_available_resources(
            available_by_hours, timestamp
        )

        if not truly_available:
            # Everyone qualified is busy
            return None, 'all_busy'

        # Select randomly from truly available resources
        import random
        return random.choice(truly_available), None

    def _schedule_activity_with_resource(self, case_id: str, activity: str,
                                          current_time: datetime, case: CaseState,
                                          resource: str) -> None:
        """Schedule activity completion with an allocated resource."""
        prev_activity = case.activity_history[-1] if case.activity_history else "START"

        # Build context for processing time prediction
        context = {
            # Temporal features (from simulation clock, not wall clock)
            'hour': current_time.hour,
            'weekday': current_time.weekday(),
            'month': current_time.month,
            'day_of_year': current_time.timetuple().tm_yday,

            # Case attributes
            'case:LoanGoal': case.case_type,
            'case:ApplicationType': case.application_type,

            # Event position tracking
            'event_position_in_case': len(case.activity_history) + 1,
            'case_duration_so_far': (current_time - case.start_time).total_seconds() if case.start_time else 0.0,

            # Resource info (current allocation)
            'resource_1': case.current_resource or 'unknown',
            'resource_2': resource,

            # Offer-level attributes (available after O_Create Offer)
            'Accepted': case.accepted,
            'Selected': case.selected,
        }

        processing_seconds = self._processing_time.predict(
            prev_activity=prev_activity,
            prev_lifecycle="complete",
            curr_activity=activity,
            curr_lifecycle="complete",
            context=context,
        )
        processing_time = timedelta(seconds=processing_seconds)
        completion_time = current_time + processing_time

        # Mark resource as busy until completion
        self.resource_pool.mark_busy(resource, completion_time, case_id, activity)

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

    def _get_next_business_hour(self, current_time: datetime) -> datetime:
        """
        Find the next business hour from the given time.

        Uses the availability model's working hours configuration.
        Handles weekends (can advance up to 72h from Friday evening to Monday morning).

        Returns:
            Next datetime when business hours start.
        """
        # Get working hours from availability model (defaults: 8-17)
        avail = self.allocator.availability
        start_hour = getattr(avail, 'workday_start_hour', 8)
        end_hour = getattr(avail, 'workday_end_hour', 17)

        # Check if we're already within business hours on a weekday
        weekday = current_time.weekday()
        hour = current_time.hour

        # Working days (Mon-Fri = 0-4)
        if weekday < 5 and start_hour <= hour < end_hour:
            # Already in business hours
            return current_time

        # Need to find next business hour
        next_time = current_time

        # If after end_hour or not a working day, move to next day's start
        if weekday >= 5 or hour >= end_hour:
            # Move to next day at start_hour
            next_time = (current_time + timedelta(days=1)).replace(
                hour=start_hour, minute=0, second=0, microsecond=0
            )
        elif hour < start_hour:
            # Before start_hour on a weekday - just advance to start_hour
            next_time = current_time.replace(
                hour=start_hour, minute=0, second=0, microsecond=0
            )

        # Skip weekends
        while next_time.weekday() >= 5:
            next_time += timedelta(days=1)

        return next_time

    def _drain_waiting_queues(self, max_time: datetime = None) -> None:
        """
        Drain phase: process remaining waiting work by advancing time.

        Iteratively advances simulation time to next business hour and
        attempts to dispatch waiting work. Continues until queues are
        empty or no progress is made (truly stuck cases).

        Args:
            max_time: Optional maximum time to advance to.
        """
        max_iterations = 100  # Safety limit (100 * potential 72h = weeks of simulation time)
        iterations_without_progress = 0
        max_no_progress = 3  # Stop if 3 time advances produce no dispatches

        initial_waiting = self.resource_pool.get_total_waiting_count()
        logger.info(f"Drain phase starting with {initial_waiting} waiting cases")

        while self.resource_pool.has_waiting_work() and iterations_without_progress < max_no_progress:
            current_time = self.clock.now
            dispatched_this_round = 0

            # Get all waiting activities
            waiting_activities = self.resource_pool.get_all_waiting_activities()

            # Try to dispatch each waiting activity
            for activity in list(waiting_activities):
                while self.resource_pool.has_waiting_work(activity):
                    # Try to allocate a resource
                    waiting_work = self.resource_pool.peek_waiting_work(activity)
                    if not waiting_work:
                        break

                    resource, failure_reason = self._try_allocate_resource_with_reason(
                        activity, current_time, waiting_work.case_state
                    )

                    if resource:
                        # Got a resource - dispatch the work
                        work = self.resource_pool.get_waiting_work(activity)
                        wait_seconds = (current_time - work.arrival_time).total_seconds()
                        self.stats['wait_time_total_seconds'] += wait_seconds

                        logger.debug(
                            f"[Drain] Dispatching {activity} for case {work.case_id} "
                            f"to {resource} (waited {wait_seconds:.0f}s)"
                        )

                        self._schedule_activity_with_resource(
                            work.case_id, work.activity, current_time,
                            work.case_state, resource
                        )
                        dispatched_this_round += 1
                    else:
                        # No resource available for this activity right now
                        break

            # Process any completion events that are now schedulable
            while not self.queue.is_empty():
                event = self.queue.pop()
                if max_time and event.timestamp > max_time:
                    logger.info(f"Drain phase reached max_time: {max_time}")
                    return
                self.clock.advance_to(event.timestamp)
                self._handle_event(event)

            if dispatched_this_round > 0:
                iterations_without_progress = 0
                logger.debug(f"[Drain] Dispatched {dispatched_this_round} cases this round")
            else:
                # No progress - advance time to next business hour
                current_time = self.clock.now
                next_business_hour = self._get_next_business_hour(
                    current_time + timedelta(minutes=1)  # Advance at least 1 minute
                )

                if max_time and next_business_hour > max_time:
                    logger.info(f"Drain phase would exceed max_time, stopping")
                    return

                time_jump = next_business_hour - current_time
                logger.info(
                    f"[Drain] No resources available, advancing {time_jump} to {next_business_hour}"
                )
                self.clock.advance_to(next_business_hour)
                iterations_without_progress += 1

            max_iterations -= 1
            if max_iterations <= 0:
                logger.warning("Drain phase hit iteration limit, stopping")
                break

        final_waiting = self.resource_pool.get_total_waiting_count()
        drained = initial_waiting - final_waiting
        logger.info(f"Drain phase complete: {drained} cases dispatched, {final_waiting} remaining")


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


class _StubCaseArrivalPredictor:
    """Stub that returns random inter-arrival times."""
    
    def predict(self) -> timedelta:
        # Random 1-30 minutes between cases
        minutes = random.randint(1, 30)
        return timedelta(minutes=minutes)


class BranchNextActivityPredictor:
    """
    Model-based next activity predictor using trained BranchPredictor.
    
    Loads a pre-trained model (models/branch_predictor.joblib) that contains XOR gateway
    branch probabilities learned from the BPMN model and event log.
    
    IMPORTANT: The model only covers activities that precede XOR gateways in the BPMN.
    See TODOs below for activities that need explicit transition handling.
    """
    
    END_ACTIVITIES = {"A_Cancelled", "A_Complete"}
    START_ACTIVITY = "A_Create Application"
    DEFAULT_MODEL_PATH = "models/branch_predictor.joblib"
    
    # Activities NOT covered by BranchPredictor (they don't precede XOR gateways).
    # These use empirical "most frequent next activity" from BPIC17 event log.
    #
    # Gateway model covers (13 activities):
    #   A_Complete, A_Concept, A_Incomplete, A_Pending, O_Cancelled, O_Created,
    #   O_Refused, O_Returned, O_Sent (mail and online), O_Sent (online only),
    #   W_Call after offers, W_Complete application, W_Validate application
    #
    # Fallback transitions below cover all remaining activities with outgoing edges.
    # End activities (A_Cancelled, A_Complete) are handled by END_ACTIVITIES set.
    FALLBACK_TRANSITIONS = {
        # Original activities without gateway coverage
        "A_Create Application": "A_Submitted",
        "A_Submitted": "W_Handle leads",
        "W_Handle leads": "W_Complete application",
        "A_Accepted": "O_Create Offer",
        "O_Create Offer": "O_Created",
        # TODO(@next-activity-team): See message thread about missing activity transitions.
        # The following transitions were causing infinite loops and have been removed:
        #   A_Validating → O_Returned → W_Validate application → A_Validating (cycle!)
        # Need to coordinate proper handling of: A_Denied, A_Validating, O_Accepted,
        # W_Assess potential fraud, W_Call incomplete files, W_Personal Loan collection,
        # W_Shortened completion
    }
    
    def __init__(self, model_path: str = None, seed: int = 42):
        """
        Initialize the predictor by loading a trained model.
        
        Args:
            model_path: Path to the trained model file (.joblib).
            seed: Random seed for reproducibility.
        """
        import joblib
        
        self.rng = random.Random(seed)
        model_path = model_path or self.DEFAULT_MODEL_PATH
        
        data = joblib.load(model_path)
        self.probabilities = data['probabilities']
        self.gateway_branches = data['gateway_branches']
        self.gateway_connections = data['gateway_connections']
        
        # Build activity -> gateway mapping for fast lookup
        self.activity_to_gateways = {}
        for gw_id, conn in self.gateway_connections.items():
            for act in conn['preceding']:
                if act not in self.activity_to_gateways:
                    self.activity_to_gateways[act] = []
                self.activity_to_gateways[act].append({
                    'gateway': gw_id,
                    'branches': conn['branches']
                })
        
        logger.info(f"Loaded BranchNextActivityPredictor: "
                   f"{len(self.probabilities)} decision points, "
                   f"{len(self.gateway_branches)} gateways, "
                   f"{len(self.activity_to_gateways)} activities covered")
    
    def predict(self, case_state: CaseState) -> tuple[str, bool]:
        """
        Predict the next activity for a case.
        
        Args:
            case_state: Current case state with activity history.
            
        Returns:
            Tuple of (next_activity_name, is_case_ended).
        """
        # First activity
        if not case_state.activity_history:
            return self.START_ACTIVITY, False
        
        current = case_state.activity_history[-1]
        
        # Already ended
        if current in self.END_ACTIVITIES:
            return current, True
        
        # 1. Try gateway-based prediction (uses trained model)
        next_act = self._predict_via_gateway(current)
        if next_act:
            is_end = next_act in self.END_ACTIVITIES
            return next_act, is_end
        
        # 2. Fallback for non-gateway activities
        if current in self.FALLBACK_TRANSITIONS:
            next_act = self.FALLBACK_TRANSITIONS[current]
            return next_act, next_act in self.END_ACTIVITIES
        
        # 3. Last resort: end the case
        logger.warning(f"No transition found for activity '{current}', ending case")
        return "A_Complete", True
    
    def _predict_via_gateway(self, current_activity: str) -> Optional[str]:
        """Use trained gateway probabilities to predict next activity."""
        if current_activity not in self.activity_to_gateways:
            return None
        
        for gw_info in self.activity_to_gateways[current_activity]:
            gw_id = gw_info['gateway']
            branches = gw_info['branches']
            
            # Use learned probabilities if available
            key = (gw_id, current_activity)
            probs = self.probabilities.get(key)
            if probs:
                branch_options = list(probs.keys())
                weights = list(probs.values())
                return self.rng.choices(branch_options, weights=weights)[0]
            
            # Fallback: random from branches
            if branches:
                return self.rng.choice(branches)
        
        return None


class ProcessSequencePredictor:
    """
    Next activity predictor using ProcessSequencePrediction LSTM model.
    
    This predictor uses the LSTM-based approach from ProcessSequencePrediction
    to predict the next activity based on activity sequences and timing information.
    """
    
    END_ACTIVITIES = {"A_Cancelled", "A_Complete", "End"}
    START_ACTIVITY = "A_Create Application"
    DEFAULT_MODEL_PATH = "ProcessSequencePrediction-master/code/output_files/models"
    
    def __init__(
        self,
        model_path: str = None,
        model_file: str = None,
        activity_mapping_path: str = None,
        max_history: int = 50,
        seed: int = 42,
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Directory containing the model file.
            model_file: Name of the model file (e.g., "model_89-1.50.h5").
            activity_mapping_path: Path to activity mapping file (JSON or pickle).
            max_history: Maximum history length for predictions.
            seed: Random seed for sampling.
        """
        import numpy as np
        from pathlib import Path
        import json
        import pickle
        
        self.rng = random.Random(seed)
        self.max_history = max_history
        self.maxlen = 50
        
        model_path = model_path or self.DEFAULT_MODEL_PATH
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        if model_file:
            model_file_path = model_path / model_file
        else:
            model_files = list(model_path.glob("*.h5"))
            if not model_files:
                raise FileNotFoundError(f"No .h5 model files found in {model_path}")
            model_file_path = model_files[0]
            logger.info(f"Using model file: {model_file_path.name}")
        
        if not model_file_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        try:
            from keras.models import load_model
            self.model = load_model(str(model_file_path))
            logger.info(f"Loaded ProcessSequencePrediction model from {model_file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        activity_mapping_path = activity_mapping_path or (model_path.parent / "activity_mapping.json")
        
        if activity_mapping_path.exists():
            try:
                with open(activity_mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.char_indices = {k: int(v) for k, v in mapping_data.get('char_indices', {}).items()}
                    self.indices_char = {int(k): v for k, v in mapping_data.get('indices_char', {}).items()}
                    self.target_char_indices = {k: int(v) for k, v in mapping_data.get('target_char_indices', {}).items()}
                    self.target_indices_char = {int(k): v for k, v in mapping_data.get('target_indices_char', {}).items()}
                    self.activity_to_char = mapping_data.get('activity_to_char', {})
                    self.char_to_activity = mapping_data.get('char_to_activity', {})
                    self.divisor = mapping_data.get('divisor', 3600.0)
                    self.divisor2 = mapping_data.get('divisor2', 86400.0)
                    self.ascii_offset = mapping_data.get('ascii_offset', 161)
                    logger.info("Loaded activity mapping from file")
            except Exception as e:
                logger.warning(f"Failed to load activity mapping: {e}. Will attempt to infer from model.")
                self._infer_mappings()
        else:
            logger.warning(f"Activity mapping file not found at {activity_mapping_path}. Attempting to infer mappings.")
            self._infer_mappings()
        
        self.case_histories: Dict[str, Dict] = {}
    
    def _infer_mappings(self):
        """Infer activity mappings from common BPIC17 activities."""
        self.ascii_offset = 161
        self.divisor = 3600.0
        self.divisor2 = 86400.0
        
        common_activities = [
            "A_Create Application", "A_Submitted", "A_Accepted", "A_Validating",
            "A_Complete", "A_Cancelled", "A_Incomplete", "A_Concept", "A_Pending",
            "O_Create Offer", "O_Created", "O_Sent (mail and online)", "O_Sent (online only)",
            "O_Returned", "O_Refused", "O_Cancelled",
            "W_Validate application", "W_Complete application", "W_Handle leads",
            "W_Call after offers", "W_Call incomplete files", "W_Assess potential fraud",
            "W_Personal Loan collection", "W_Shortened completion"
        ]
        
        self.activity_to_char = {}
        self.char_to_activity = {}
        self.char_indices = {}
        self.indices_char = {}
        self.target_char_indices = {}
        self.target_indices_char = {}
        
        for idx, activity in enumerate(common_activities):
            char = chr(idx + self.ascii_offset)
            self.activity_to_char[activity] = char
            self.char_to_activity[char] = activity
            self.char_indices[char] = idx
            self.indices_char[idx] = char
            self.target_char_indices[char] = idx
            self.target_indices_char[idx] = char
        
        logger.info(f"Inferred mappings for {len(common_activities)} activities")
    
    def _get_case_history(self, case_id: str) -> Dict:
        """Get or create history tracker for a case."""
        if case_id not in self.case_histories:
            self.case_histories[case_id] = {
                'activities': [],
                'times': [],
                'times2': [],
                'times3': [],
                'start_time': None,
            }
        return self.case_histories[case_id]
    
    def _encode_sequence(self, activities: List[str], times: List[float], times2: List[float], 
                        times3: List[datetime], maxlen: int = None):
        """Encode activity sequence and times into model input format."""
        import numpy as np
        from collections import Counter
        
        maxlen = maxlen or self.maxlen
        num_features = len(self.char_indices) + 5
        
        X = np.zeros((1, maxlen, num_features), dtype=np.float32)
        
        sentence = ''.join([self.activity_to_char.get(act, chr(self.ascii_offset)) for act in activities])
        leftpad = maxlen - len(sentence)
        
        for t, char in enumerate(sentence):
            if char not in self.char_indices:
                continue
            char_idx = self.char_indices[char]
            X[0, t + leftpad, char_idx] = 1
            X[0, t + leftpad, len(self.char_indices)] = t + 1
            if t < len(times):
                X[0, t + leftpad, len(self.char_indices) + 1] = times[t] / self.divisor
            if t < len(times2):
                X[0, t + leftpad, len(self.char_indices) + 2] = times2[t] / self.divisor2
            if t < len(times3):
                midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (times3[t] - midnight).total_seconds()
                X[0, t + leftpad, len(self.char_indices) + 3] = timesincemidnight / 86400
                X[0, t + leftpad, len(self.char_indices) + 4] = times3[t].weekday() / 7
        
        return X
    
    def _decode_prediction(self, predictions) -> str:
        """Decode model prediction to activity name."""
        import numpy as np
        activity_probs = predictions[0][0]
        max_idx = np.argmax(activity_probs)
        predicted_char = self.target_indices_char.get(max_idx, chr(self.ascii_offset))
        predicted_activity = self.char_to_activity.get(predicted_char, self.START_ACTIVITY)
        return predicted_activity
    
    def predict(self, case_state: CaseState) -> tuple[str, bool]:
        """
        Predict next activity using ProcessSequencePrediction model.
        
        Args:
            case_state: Current case state with activity history.
            
        Returns:
            Tuple of (next_activity_name, is_case_ended).
        """
        case_id = case_state.case_id
        history = self._get_case_history(case_id)
        
        if not case_state.activity_history:
            history['start_time'] = case_state.start_time or datetime.now()
            return self.START_ACTIVITY, False
        
        current_activity = case_state.activity_history[-1]
        
        if current_activity in self.END_ACTIVITIES:
            if case_id in self.case_histories:
                del self.case_histories[case_id]
            return current_activity, True
        
        activities = case_state.activity_history[-self.max_history:]
        history['activities'] = activities
        
        if not history['start_time']:
            history['start_time'] = case_state.start_time or datetime.now()
        
        if len(history['times']) < len(activities):
            new_times = []
            new_times2 = []
            new_times3 = []
            base_time = history['start_time']
            
            for i, act in enumerate(activities):
                if i < len(history['times']):
                    new_times.append(history['times'][i])
                    new_times2.append(history['times2'][i])
                    new_times3.append(history['times3'][i])
                else:
                    if i == 0:
                        time_diff = 0.0
                        event_time = base_time
                    else:
                        time_diff = 3600.0
                        event_time = new_times3[-1] + timedelta(seconds=time_diff)
                    
                    new_times.append(time_diff)
                    new_times2.append((event_time - base_time).total_seconds())
                    new_times3.append(event_time)
            
            history['times'] = new_times
            history['times2'] = new_times2
            history['times3'] = new_times3
        
        try:
            X = self._encode_sequence(
                history['activities'],
                history['times'],
                history['times2'],
                history['times3']
            )
            
            y = self.model.predict(X, verbose=0)
            next_activity = self._decode_prediction(y)
            
            if next_activity == '!' or next_activity not in self.char_to_activity.values():
                next_activity = "A_Complete"
            
            is_end = next_activity in self.END_ACTIVITIES
            
            return next_activity, is_end
            
        except Exception as e:
            logger.warning(f"ProcessSequencePrediction failed for case {case_id}: {e}")
            return "A_Complete", True
    
    def reset_case(self, case_id: str):
        """Reset tracker for a case (when case ends)."""
        if case_id in self.case_histories:
            del self.case_histories[case_id]
    
    def clear(self):
        """Clear all case trackers."""
        self.case_histories.clear()


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
