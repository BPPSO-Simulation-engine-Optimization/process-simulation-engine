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

import pandas as pd

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
        
        # Output: collected events for export
        self.completed_events: List[Dict] = []
        
        # Statistics
        self.stats = {
            'cases_started': 0,
            'cases_completed': 0,
            'events_processed': 0,
            'allocation_failures': 0,
        }
    
    def _create_next_activity_predictor(self):
        """
        Auto-load BranchNextActivityPredictor if trained model exists, else use stub.
        
        The model is trained via: python Next-Activity-Prediction/train.py
        """
        from pathlib import Path
        model_path = Path(BranchNextActivityPredictor.DEFAULT_MODEL_PATH)
        if model_path.exists():
            try:
                return BranchNextActivityPredictor(str(model_path))
            except Exception as e:
                logger.warning(f"Failed to load next activity model: {e}")
        logger.info("Using stub next activity predictor "
                   "(train model with: python Next-Activity-Prediction/train.py)")
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
        """Handle activity completion: log event, predict next activity."""
        case = self.case_manager.get_case(event.case_id)
        if not case:
            logger.warning(f"Case not found: {event.case_id}")
            return

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
        # ProcessingTimePredictionClass uses (prev_activity, prev_lifecycle, curr_activity, curr_lifecycle)
        # TODO: Currently all lifecycles are hardcoded to "complete" (MVP decision).
        #       Extend to support start/schedule lifecycles for more accurate timing.
        prev_activity = case.activity_history[-1] if case.activity_history else "START"

        # Build context from simulation state (P0-1 fix: pass context to processing time predictor)
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
            'resource_2': resource,  # The resource being allocated for this activity

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


class _StubCaseArrivalPredictor:
    """Stub that returns random inter-arrival times."""
    
    def predict(self) -> timedelta:
        # Random 1-30 minutes between cases
        minutes = random.randint(1, 30)
        return timedelta(minutes=minutes)


class BranchNextActivityPredictor:
    """
    Model-based next activity predictor using trained BranchPredictor from Next-Activity-Prediction.
    
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
        # TODO(@next-activity-prediction-team): Replace this with model-based prediction
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
