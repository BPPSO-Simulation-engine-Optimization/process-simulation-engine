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

import os
import time
import uuid
import random
import logging
import threading
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Protocol, Set, Tuple
from collections import defaultdict
import heapq

import numpy as np
import pandas as pd
import scipy.stats

from .events import SimulationEvent, EventType
from .event_queue import EventQueue
from .clock import SimulationClock
from .case_manager import CaseState, CaseManager
from resources.selection_strategies import ResourceSelectionStrategy, RandomStrategy

logger = logging.getLogger(__name__)


class NextActivityPredictorType(Enum):
    """
    Available next activity predictor types.

    Use with DESEngine's next_activity_predictor_type parameter to specify
    which predictor to load automatically.
    """
    UNIFIED = "unified"
    LSTM = "lstm"
    BRANCH = "branch"
    STUB = "stub"
    PROCESS_TRANSFORMER = "process_transformer"


@dataclass
class WaitingWork:
    """Represents work waiting for a resource."""
    case_id: str
    # Original activity label used for logging/simulation semantics
    activity: str
    lifecycle: str
    # Normalized label used for permission/availability checks and queue routing
    allocation_activity: str
    arrival_time: datetime  # When the work arrived (for FIFO ordering)
    case_state: CaseState
    # Estimated processing time (seconds) for SPT ordering. Set when task is added
    # to a resource worklist. None if not yet estimated.
    estimated_pt_seconds: Optional[float] = None

    def __lt__(self, other):
        """For heap ordering - earlier arrival time = higher priority."""
        return self.arrival_time < other.arrival_time


class SimulationProfiler:
    """Lightweight profiler using time.perf_counter() for wall-clock measurement."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._totals: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._wall_start: float = 0.0

    def start_wall_clock(self):
        if self.enabled:
            self._wall_start = time.perf_counter()

    @contextmanager
    def measure(self, component: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self._totals[component] += elapsed
        self._counts[component] += 1

    def print_report(self):
        if not self.enabled:
            return
        wall_total = time.perf_counter() - self._wall_start

        # Separate event-level from component-level measurements
        event_items = {k: v for k, v in self._totals.items() if k.startswith("event.")}
        component_items = {k: v for k, v in self._totals.items() if not k.startswith("event.")}

        print(f"\n{'='*80}")
        print("PERFORMANCE PROFILE")
        print(f"{'='*80}")
        print(f"Total wall-clock time: {wall_total:.2f}s\n")

        # Event-level breakdown
        print("Event type breakdown:")
        print(f"{'Event type':<45} {'Total (s)':>10} {'Calls':>7} {'Avg (ms)':>10} {'% Wall':>8}")
        print("-" * 80)
        for comp in sorted(event_items, key=event_items.get, reverse=True):
            total = self._totals[comp]
            count = self._counts[comp]
            avg_ms = (total / count * 1000) if count else 0
            pct = (total / wall_total * 100) if wall_total else 0
            print(f"{comp:<45} {total:>10.3f} {count:>7} {avg_ms:>10.3f} {pct:>7.1f}%")
        event_accounted = sum(event_items.values())
        event_unaccounted = wall_total - event_accounted
        print(f"{'(unaccounted / overhead)':<45} {event_unaccounted:>10.3f} {'':>7} {'':>10} {(event_unaccounted/wall_total*100) if wall_total else 0:>7.1f}%")

        # Component-level breakdown
        print(f"\nComponent breakdown (within event handlers):")
        print(f"{'Component':<45} {'Total (s)':>10} {'Calls':>7} {'Avg (ms)':>10} {'% Wall':>8}")
        print("-" * 80)
        for comp in sorted(component_items, key=component_items.get, reverse=True):
            total = self._totals[comp]
            count = self._counts[comp]
            avg_ms = (total / count * 1000) if count else 0
            pct = (total / wall_total * 100) if wall_total else 0
            print(f"{comp:<45} {total:>10.3f} {count:>7} {avg_ms:>10.3f} {pct:>7.1f}%")
        print(f"{'='*80}\n")


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
        # INFINITE CAPACITY FIX: User_1 is the 'Applicant' and has infinite capacity.
        # They are never busy in the sense of being blocked.
        if resource_id == 'User_1':
            return False

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
        # INFINITE CAPACITY FIX: User_1 never gets marked as busy
        if resource_id == 'User_1':
            return
            
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

    def get_all_waiting_tasks(self) -> List[WaitingWork]:
        """
        Return a flat list of all waiting work across all queues.

        Non-destructive: items remain in their queues.
        """
        result = []
        for q in self._waiting_queues.values():
            result.extend(q)
        return result

    def remove_task_by_id(
        self, allocation_activity: str, case_id: str
    ) -> Optional[WaitingWork]:
        """
        Remove a specific task from its waiting queue by case_id.

        O(n) scan + heapify.  Only called on worker-idle events so
        performance is fine for queue sizes in the hundreds.

        Returns:
            The removed WaitingWork, or None if not found.
        """
        q = self._waiting_queues.get(allocation_activity)
        if not q:
            return None

        for idx, work in enumerate(q):
            if work.case_id == case_id:
                q.pop(idx)
                heapq.heapify(q)
                return work
        return None


# Protocol definitions for pluggable predictors
class NextActivityPredictor(Protocol):
    """Interface for next activity prediction."""
    def predict(self, case_state: CaseState):
        """
        Predict the next activity for a case.
        
        Args:
            case_state: Current case state.
            
        Returns:
            Either (activity_name, is_case_ended) or
            (activity_name, lifecycle_transition, is_case_ended).
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

    PT_LIFECYCLE_MODES = {"native", "gt_activity_gated"}
    PT_GT_START_ACTIVITIES = {
        "W_Assess potential fraud",
        "W_Call after offers",
        "W_Call incomplete files",
        "W_Complete application",
        "W_Handle leads",
        "W_Validate application",
    }
    
    def __init__(
        self,
        resource_allocator: ResourceAllocator,
        arrival_timestamps: List[datetime] = None,
        next_activity_predictor: NextActivityPredictor = None,
        next_activity_predictor_type: NextActivityPredictorType = None,
        next_activity_config: Dict = None,
        processing_time_predictor: ProcessingTimePredictor = None,
        case_arrival_predictor: CaseArrivalPredictor = None,
        case_attribute_predictor: CaseAttributePredictor = None,
        start_time: datetime = None,
        max_activities_per_case: int = 100,
        resource_selection_strategy: ResourceSelectionStrategy = None,
        batch_allocation_policy=None,
        processing_time_estimator=None,
        drl_policy=None,
        drl_max_postpone_wait_hours: float = 4.0,
        pmsp_config=None,
        pt_lifecycle_mode: str = "native",
        enable_profiling: bool = False,
        pmsp_log_file: Optional[str] = None,
        incremental_csv_path: Optional[str] = None,
    ):
        """
        Initialize the DES Engine.

        Args:
            resource_allocator: Resource allocation component.
            arrival_timestamps: Pre-generated list of case arrival timestamps.
                If provided, overrides case_arrival_predictor.
            next_activity_predictor: Predicts next activity. If provided, takes precedence.
            next_activity_predictor_type: Type of predictor to auto-load if next_activity_predictor
                is not provided. See NextActivityPredictorType enum.
            processing_time_predictor: Predicts processing time (required).
            case_arrival_predictor: Predicts inter-arrival time (uses stub if None).
            case_attribute_predictor: Predicts case attributes (required).
            start_time: Simulation start time.
            max_activities_per_case: Safety limit to prevent infinite loops.
            resource_selection_strategy: Heuristic for selecting among available resources.
                Defaults to RandomStrategy (R-RMA).
            batch_allocation_policy: Optional BatchAllocationPolicy (e.g. OneBatchOnePolicy).
                When set, _process_waiting_queue uses holistic MILP instead of greedy.
            processing_time_estimator: Optional ProcessingTimeEstimator for batch policy
                p_{ij} lookups.  Required when batch_allocation_policy is set.
            drl_policy: Optional DRL allocation policy (DRLAllocationPolicy or
                InteractiveBatchPolicy).  When set, overrides both batch and greedy.
            pmsp_config: Optional SelectionConfig for PMSP-based resource optimization.
                When set, _process_waiting_queue uses PMSP solver instead of greedy.
            pt_lifecycle_mode: PT-only lifecycle logging mode.
                "native": keep predictor lifecycle output.
                "gt_activity_gated": emit synthetic "start" for GT start-capable
                activities and force completion logs to "complete".
            enable_profiling: If True, measure wall-clock time per component.
        """
        self.queue = EventQueue()
        self.clock = SimulationClock(start_time)
        self.case_manager = CaseManager()
        self.allocator = resource_allocator

        # Pre-generated arrival timestamps (optional)
        self._arrival_timestamps = arrival_timestamps

        # Processing time predictor is required (must be ProcessingTimePredictionClass)
        # NOTE: This MUST be assigned before _create_next_activity_predictor, because
        # the Process Transformer path overrides self._processing_time with its own
        # PTTimeAdapter. If we assign after, the override gets clobbered.
        if processing_time_predictor is None:
            raise ValueError(
                "processing_time_predictor is required. "
                "Use ProcessingTimePredictionClass from processing_time_prediction"
            )
        self._processing_time = processing_time_predictor

        # Next activity predictor: use provided instance, or create from type
        if next_activity_predictor is not None:
            self._next_activity = next_activity_predictor
        elif next_activity_predictor_type is not None:
            self._next_activity = self._create_next_activity_predictor(
                next_activity_predictor_type,
                next_activity_config or {}
            )
        else:
            raise ValueError(
                "Either next_activity_predictor or next_activity_predictor_type is required. "
                "Use a valid NextActivityPredictorType or pass a predictor instance."
            )

        self._is_process_transformer_predictor = self._detect_process_transformer_predictor(
            predictor=self._next_activity,
            predictor_type=next_activity_predictor_type,
        )
        self._pt_lifecycle_mode = str(pt_lifecycle_mode or "native")
        if self._pt_lifecycle_mode not in self.PT_LIFECYCLE_MODES:
            raise ValueError(
                f"Unknown pt_lifecycle_mode: {self._pt_lifecycle_mode}. "
                f"Expected one of: {sorted(self.PT_LIFECYCLE_MODES)}"
            )
        if self._pt_lifecycle_mode == "gt_activity_gated" and not self._is_process_transformer_predictor:
            raise ValueError(
                "pt_lifecycle_mode='gt_activity_gated' is only valid with the "
                "Process Transformer next activity predictor."
            )
        self._case_arrival = case_arrival_predictor or _StubCaseArrivalPredictor()

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

        # Resource selection heuristic (R-RMA, R-RRA, or R-SHQ)
        self._resource_strategy = resource_selection_strategy or RandomStrategy()

        # Batch allocation policy (optional, overrides greedy waiting-queue logic)
        self._batch_policy = batch_allocation_policy
        self._pt_estimator = processing_time_estimator

        # DRL allocation policy (optional, overrides both batch and greedy)
        self._drl_policy = drl_policy
        self._drl_max_postpone_wait_hours = drl_max_postpone_wait_hours

        # PMSP resource optimization config (optional)
        self._pmsp_config = pmsp_config

        # Per-resource worklists for PMSP mode (resource_id -> list of WaitingWork)
        self._resource_worklists: Dict[str, List[WaitingWork]] = defaultdict(list)

        # PT prediction cache shared across PMSP cycles.
        # Key: (case_id, allocation_activity, resource) → predicted seconds.
        # Entries are invalidated when the owning case ends (_on_case_end).
        self._pmsp_pt_cache: Dict = {}


        # Output: collected events for export
        self.completed_events: List[Dict] = []
        
        # Incremental CSV export (write every 100 cases)
        self._incremental_csv_path: Optional[str] = incremental_csv_path
        self._last_csv_exported_events_count: int = 0
        self._last_csv_exported_cases: int = 0

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

        # Profiler
        self.profiler = SimulationProfiler(enabled=enable_profiling)

        # Per-transition P99 duration caps and activity whitelist
        self._transition_p99_caps, self._valid_activities = self._load_transition_caps()
        # Hard override: cap A_Concept (complete) → A_Accepted (complete) to 72h if not present
        try:
            key_override = ("A_Concept", "complete", "A_Accepted", "complete")
            if key_override not in self._transition_p99_caps:
                self._transition_p99_caps[key_override] = 72 * 3600.0
        except Exception:
            # Be defensive: if caps not loaded, ensure structure exists and set
            self._transition_p99_caps = {("A_Concept", "complete", "A_Accepted", "complete"): 72 * 3600.0}

        # Per-activity repetition limit (GT max is ~10-12 for W_ activities)
        self._max_activity_repeats = 15

    @staticmethod
    def _load_transition_caps() -> Tuple[Dict[tuple, float], Set[str]]:
        """Load per-transition P99 caps from the distribution model.

        Returns (transition_key → P99_seconds, set_of_valid_activities).
        If the model file is missing, returns empty structures (no caps applied).
        """
        model_path = os.path.join("models", "processing_time_model_complete_only_distributions.joblib")
        if not os.path.exists(model_path):
            logger.warning("Distribution model not found at %s — P99 caps disabled", model_path)
            return {}, set()

        import joblib
        dist_params = joblib.load(model_path)

        caps: Dict[tuple, float] = {}
        activities: Set[str] = set()
        for key, params in dist_params.items():
            mu, sigma = float(params["mu"]), float(params["sigma"])
            p99 = float(scipy.stats.lognorm(s=sigma, scale=np.exp(mu)).ppf(0.99))
            caps[key] = p99
            activities.add(key[0])
            activities.add(key[2])

        logger.info(
            "Loaded P99 caps for %d transitions, %d valid activities",
            len(caps), len(activities),
        )
        return caps, activities

    @staticmethod
    def _detect_process_transformer_predictor(
        predictor,
        predictor_type: Optional[NextActivityPredictorType],
    ) -> bool:
        """Best-effort detection of Process Transformer predictor wiring."""
        if predictor_type == NextActivityPredictorType.PROCESS_TRANSFORMER:
            return True
        if predictor is None:
            return False

        cls = predictor.__class__
        module_name = getattr(cls, "__module__", "")
        class_name = getattr(cls, "__name__", "")
        if "process_transformer_v2" in module_name:
            return True
        return class_name in {"PTActivityAdapter", "ProcessTransformerV2Predictor"}
    
    def _create_next_activity_predictor(
        self, 
        predictor_type: NextActivityPredictorType,
        config: Dict = None
    ):
        """
        Create a next activity predictor based on the specified type.

        Args:
            predictor_type: The type of predictor to create.
            config: Configuration dict (temperature, end_token_penalty for Process Transformer).

        Returns:
            An instance of the requested predictor.

        Raises:
            ValueError: If the predictor type is unknown or cannot be loaded.
        """
        config = config or {}
        import sys
        from pathlib import Path

        # Add next_activity_prediction to path for imports
        project_root = Path(__file__).parent.parent
        na_root = project_root / "next_activity_prediction"
        if str(na_root) not in sys.path:
            sys.path.insert(0, str(na_root))

        if predictor_type == NextActivityPredictorType.UNIFIED:
            logger.info("Loading UnifiedNextActivityPredictor...")
            return UnifiedNextActivityPredictor(model_path="models/unified_next_activity")

        elif predictor_type == NextActivityPredictorType.LSTM:
            logger.info("Loading LSTMNextActivityPredictor...")
            return LSTMNextActivityPredictor(models_dir="next_activity_prediction/advanced/models_lstm_new")

        elif predictor_type == NextActivityPredictorType.BRANCH:
            logger.info("Loading BranchNextActivityPredictor...")
            return BranchNextActivityPredictor(model_path="models/branch_predictor.joblib")

        elif predictor_type == NextActivityPredictorType.STUB:
            logger.info("Using StubNextActivityPredictor...")
            return _StubNextActivityPredictor()

        elif predictor_type == NextActivityPredictorType.PROCESS_TRANSFORMER:
            logger.info("Loading ProcessTransformerV2Predictor (Unified) as main Process Transformer...")
            from process_transformer_v2.inference import ProcessTransformerV2Predictor, PTActivityAdapter, PTTimeAdapter
            
            # Unified wrapper
            # We assume the models are in the default location relative to the inference file
            # or we can pass a specific path if config has it.
            temperature = config.get('temperature', 1.5)
            logger.info(f"ProcessTransformerV2: Setting temperature to {temperature}")
            unified = ProcessTransformerV2Predictor(temperature=temperature) 
            
            # Register BOTH parts
            # 1. Activity Predictor (returned here)
            activity_adapter = PTActivityAdapter(unified)
            
            # 2. Time Predictor (injected into self)
            # This is a bit of a hack: The engine calls this method to get the *Activity* predictor.
            # But we also need to set the *Time* predictor.
            # Since we have reference to 'self', we can override it!
            max_dur = config.get('pt_max_duration_seconds')
            time_adapter = PTTimeAdapter(unified, max_duration_seconds=max_dur)
            self._processing_time = time_adapter
            if max_dur:
                logger.info(f"ProcessTransformerV2: Duration cap set to {max_dur/3600:.0f}h ({max_dur/86400:.0f} days)")
            logger.info("ProcessTransformerV2: Also registered as ProcessingTimePredictor.")
            
            return activity_adapter

        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
    
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

        # Start profiler wall clock
        self.profiler.start_wall_clock()
        
        # Initialize incremental CSV file (delete if exists to start fresh)
        if self._incremental_csv_path:
            import os
            if os.path.exists(self._incremental_csv_path):
                os.remove(self._incremental_csv_path)
            self._last_csv_exported_events_count = 0
            logger.info(f"Incremental CSV export enabled: will write to {self._incremental_csv_path} every 100 cases")

        # Main simulation loop
        print(f"\n{'='*60}", flush=True)
        print(f"Starting simulation loop with {len(self.queue)} scheduled events...", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        event_count = 0
        progress_every_n_events = 100

        try:
            while not self.queue.is_empty():
                event = self.queue.pop()

                if max_time and event.timestamp > max_time:
                    break

                self.clock.advance_to(event.timestamp)
                self._handle_event(event)

                event_count += 1

                # Progress: print every N processed events (only for non-PMSP runs).
                if (
                    progress_every_n_events
                    and event_count % progress_every_n_events == 0
                ):
                    current_time = self.clock.now

                    # Free resources: known to availability model, not busy, currently on-duty
                    all_resources = getattr(self.allocator.availability, 'resources', [])
                    free_resources = [
                        r for r in all_resources
                        if not self.resource_pool.is_busy(r, current_time)
                        and self.allocator.availability.is_available(r, current_time)
                    ]

                    # Worklist summary: resource -> task count
                    worklist_summary = {
                        r: len(tasks)
                        for r, tasks in self._resource_worklists.items()
                        if tasks
                    }

                    # Waiting summary: allocation_activity -> task count (helps debug "many waits, few free resources")
                    waiting_summary = self.resource_pool.get_waiting_summary()
                    waiting_summary_top = sorted(waiting_summary.items(), key=lambda x: -x[1])[:10]
                    waiting_summary_str = ", ".join(f"{act}:{n}" for act, n in waiting_summary_top)
                    waiting_summary_ellipsis = "..." if len(waiting_summary) > 10 else ""

                    print(
                        f"Progress: {self.stats['cases_started']} cases started, "
                        f"{self.stats['cases_completed']} completed, "
                        f"{len(self.completed_events)} events logged, "
                        f"{self.resource_pool.get_total_waiting_count()} waiting, "
                        f"simulation time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"  Free resources ({len(free_resources)}): "
                        f"{', '.join(free_resources[:10])}"
                        f"{'...' if len(free_resources) > 10 else ''}\n"
                        f"  Worklists ({len(worklist_summary)} resources): "
                        f"{', '.join(f'{r}:{n}' for r, n in sorted(worklist_summary.items())[:10])}"
                        f"{'...' if len(worklist_summary) > 10 else ''}\n"
                        f"  Waiting by activity ({len(waiting_summary)} activities): "
                        f"{waiting_summary_str}{waiting_summary_ellipsis}",
                        flush=True,
                    )
            
                # Incremental CSV export: write every 100 cases
                if self._incremental_csv_path and self.stats['cases_started'] % 100 == 0 and self.stats['cases_started'] > 0:
                    new_events = self.completed_events[self._last_csv_exported_events_count:]
                    if new_events:
                        from simulation.log_exporter import LogExporter
                        write_header = (self._last_csv_exported_events_count == 0)
                        LogExporter.append_to_csv(new_events, self._incremental_csv_path, write_header=write_header)
                        self._last_csv_exported_events_count = len(self.completed_events)
        finally:
            pass

        # Drain phase: process remaining waiting work by advancing time
        if self.resource_pool.has_waiting_work():
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
        
        # Write remaining events to incremental CSV if enabled
        if self._incremental_csv_path:
            remaining_events = self.completed_events[self._last_csv_exported_events_count:]
            if remaining_events:
                from simulation.log_exporter import LogExporter
                write_header = (self._last_csv_exported_events_count == 0)
                LogExporter.append_to_csv(remaining_events, self._incremental_csv_path, write_header=write_header)
                logger.info(f"Final incremental CSV export: wrote {len(remaining_events)} remaining events to {self._incremental_csv_path}")

        self.profiler.print_report()

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
            with self.profiler.measure(f"event.{event.event_type.name}"):
                handler(event)
        else:
            logger.warning(f"Unknown event type: {event.event_type}")

    def _normalize_next_prediction(self, prediction_result) -> tuple[str, str, bool]:
        """Normalize predictor output to (activity, lifecycle, is_end)."""
        if not isinstance(prediction_result, tuple):
            raise ValueError("Next activity predictor must return a tuple")

        if len(prediction_result) == 2:
            activity, is_end = prediction_result
            return activity, "complete", is_end

        if len(prediction_result) == 3:
            activity, lifecycle, is_end = prediction_result
            return activity, (lifecycle or "complete"), is_end

        raise ValueError("Next activity predictor must return 2 or 3 values")
    
    def _on_case_arrival(self, event: SimulationEvent) -> None:
        """Handle case arrival: create case state, schedule first activity."""
        self.stats['cases_started'] += 1

        # Delay arrival to next business hour if outside working hours
        arrival_time = event.timestamp
        if not self._is_business_hours(arrival_time):
            arrival_time = self._get_next_business_hour(arrival_time)

        # Print arrival info for visibility
        if self.stats['cases_started'] <= 100 or self.stats['cases_started'] % 100 == 0:
            print(f"[ARRIVAL] Case {self.stats['cases_started']}: {event.case_id} at {arrival_time.strftime('%Y-%m-%d %H:%M')}", flush=True)

        # Get case attributes from AttributeSimulationEngine
        with self.profiler.measure("case_attribute.start_new_case"):
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
            start_time=arrival_time,
        )
        # Store reference to attr engine case for later offer attribute generation
        case._attr_engine_case = attr_case

        # Predict first activity
        with self.profiler.measure("next_activity.predict"):
            activity, lifecycle, is_end = self._normalize_next_prediction(self._next_activity.predict(case))

        if is_end:
            # Edge case: case ends immediately
            self._schedule_case_end(event.case_id, arrival_time)
            return

        # Allocate resource and schedule activity
        with self.profiler.measure("schedule_activity"):
            logger.info(f"Scheduling activity after case arrival: {activity} {lifecycle}")
            self._schedule_activity(event.case_id, activity, lifecycle, arrival_time, case)

    def _on_activity_complete(self, event: SimulationEvent) -> None:
        """Handle activity completion: log event, release resource, process waiting queue."""
        case = self.case_manager.get_case(event.case_id)
        if not case:
            logger.warning(f"Case not found: {event.case_id}")
            return

        # Release the resource that completed this activity
        if event.resource:
            self.resource_pool.release(event.resource)
            self._resource_strategy.notify_release(event.resource)
            # Try to dispatch waiting work now that this resource is free
            with self.profiler.measure("process_waiting_queue"):
                logger.debug("Processing waiting queue after activity completion. Resource freed: %s", event.resource)
                # In PMSP mode: first drain the pre-planned worklist for this resource.
                if self._pmsp_config is not None:
                    logger.info("Processing resource worklist of ressource %s", event.resource)
                    self._process_resource_worklist(event.resource, event.timestamp)
                    # In PMSP mode, do NOT optimize on every resource-free event.
                    # Only run PMSP when the waiting batch is "full" according to
                    # `pmsp_optimization_batch_size` (0 = always).
                    waiting_count = self.resource_pool.get_total_waiting_count()
                    batch_size = getattr(self._pmsp_config, "optimization_batch_size", 0) or 0

                    if waiting_count > 0 and (batch_size == 0 or waiting_count >= batch_size):
                        logger.info(
                            "PMSP [TRIGGER]: Resource %s freed -> running PMSP optimization "
                            "(waiting_count=%d, batch_size=%d)",
                            event.resource,
                            waiting_count,
                            batch_size,
                        )
                        self._process_waiting_queue_pmsp(event.resource, event.timestamp)
                    else:
                        logger.debug(
                            "PMSP [SKIP TRIGGER]: Resource %s freed, but waiting_count=%d "
                            "does not satisfy batch_size=%d",
                            event.resource,
                            waiting_count,
                            batch_size,
                        )
                else:
                    self._process_waiting_queue(event.resource, event.timestamp)

        # Generate offer-dependent attributes when O_Create Offer completes
        if event.activity == "O_Create Offer" and case._attr_engine_case is not None:
            # Populate offer attributes directly on the stored case reference
            # (uses explicit CaseState, not internal _active_case pointer)
            with self.profiler.measure("case_attribute.populate_offer"):
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

        completion_lifecycle = event.lifecycle
        if self._pt_lifecycle_mode == "gt_activity_gated" and self._is_process_transformer_predictor:
            completion_lifecycle = "complete"

        # Record activity in case history
        case.add_activity(event.activity, event.resource, completion_lifecycle)

        # Safety guard: if a case keeps looping, stop it instead of hanging the run
        if self._max_activities_per_case and len(case.activity_history) >= self._max_activities_per_case:
            logger.warning(
                f"Case {event.case_id} hit max_activities_per_case={self._max_activities_per_case}; "
                f"ending to avoid infinite loop (last={case.activity_history[-1]})."
            )
            log_record = event.to_log_record()
            log_record['lifecycle:transition'] = completion_lifecycle
            log_record.update(case.get_payload())
            self.completed_events.append(log_record)
            self._schedule_case_end(event.case_id, event.timestamp)
            return



        # Log the event for export
        log_record = event.to_log_record()
        log_record['lifecycle:transition'] = completion_lifecycle
        log_record.update(case.get_payload())
        self.completed_events.append(log_record)

        # Predict next activity
        with self.profiler.measure("next_activity.predict"):
            next_activity, next_lifecycle, is_end = self._normalize_next_prediction(self._next_activity.predict(case))

        # Guard: cap per-activity repetitions to prevent runaway loops
        if not is_end:
            activity_count = case.activity_history.count(next_activity)
            if activity_count >= self._max_activity_repeats:
                logger.warning(
                    "Case %s: %s repeated %d times, forcing end",
                    event.case_id, next_activity, activity_count,
                )
                is_end = True

        if is_end:
            self._schedule_case_end(event.case_id, event.timestamp)
        else:
            with self.profiler.measure("schedule_activity"):
                logger.info("Schedule activity after activity completion: {next_activity} {next_lifecycle}")
                self._schedule_activity(event.case_id, next_activity, next_lifecycle, event.timestamp, case)

    def _process_waiting_queue(self, freed_resource: str, current_time: datetime) -> None:
        """
        Process waiting queue when a resource becomes free.

        Tries to dispatch waiting work to the freed resource if it's eligible.
        """
        logger.debug("Processing waiting queue.")
        # DRL policy overrides both batch and greedy
        if self._drl_policy is not None:
            self._process_waiting_queue_drl(freed_resource, current_time)
            return

        # Batch allocation policy overrides greedy logic
        if self._batch_policy is not None:
            self._process_waiting_queue_batch(freed_resource, current_time)
            return

        # PMSP optimization overrides greedy logic
        if self._pmsp_config is not None:
            self._process_waiting_queue_pmsp(freed_resource, current_time)
            return

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

                # Track assignment for strategy (keeps SHQ counts accurate)
                self._resource_strategy.notify_assignment(freed_resource, allocation_activity)

                # Schedule the activity with the freed resource
                self._schedule_activity_with_resource(
                    waiting_work.case_id,
                    waiting_work.activity,
                    waiting_work.lifecycle,
                    current_time,
                    waiting_work.case_state,
                    freed_resource,
                )
                # Resource is now busy again, stop looking
                return

    def _transfer_unavailable_resource_worklists(self, current_time: datetime) -> int:
        """
        K-Batching: Transfer tasks from unavailable resources' worklists back to waiting queue.
        
        When a resource becomes unavailable (e.g., outside working hours), all tasks
        on its worklist are transferred back to the set of unassigned tasks.
        
        Returns:
            Number of tasks transferred back to waiting queue.
        """
        if not self._pmsp_config:
            return 0
        
        total_transferred = 0
        
        # Check all resources with worklists
        resources_to_check = list(self._resource_worklists.keys())
        
        for resource in resources_to_check:
            # Check if resource is unavailable (not available due to working hours, etc.)
            if not self.allocator.availability.is_available(resource, current_time):
                worklist = self._resource_worklists.get(resource, [])
                if worklist:
                    logger.debug(
                        "K-Batching: Resource %s became unavailable, transferring %d tasks back to waiting queue",
                        resource, len(worklist)
                    )
                    
                    # Transfer all tasks from worklist back to waiting queue
                    for work in worklist:
                        self.resource_pool.add_to_waiting_queue(work)
                        total_transferred += 1
                    
                    # Clear the worklist
                    del self._resource_worklists[resource]
        
        return total_transferred

    def _estimate_pt_seconds(self, work: "WaitingWork", resource: str, current_time: datetime) -> float:
        """Estimate processing time (seconds) for a waiting task on a given resource.

        Used for SPT ordering of worklists.  Falls back to 0.0 on any error so
        the sort remains stable even if the predictor is unavailable.
        """
        try:
            from resources.resource_optimization.resource_optimization import predict_processing_seconds
            prev_activity = getattr(work, 'prev_activity', 'START') or 'START'
            prev_lifecycle = getattr(work, 'prev_lifecycle', 'complete') or 'complete'
            result = predict_processing_seconds(
                predictor=self._processing_time,
                prev_activity=prev_activity,
                prev_lifecycle=prev_lifecycle,
                curr_activity=work.activity,
                curr_lifecycle='complete',
                context={'resource_2': resource},
            )
            return result if result is not None else 0.0
        except Exception:
            return 0.0

    def _process_resource_worklist(self, resource: str, current_time: datetime) -> bool:
        """
        Process tasks from a resource's worklist in PMSP mode.

        When a resource is freed, first check if there are tasks in its worklist
        and execute those before processing the general waiting queue.
        
        K-Batching: If the resource is unavailable, transfer tasks back to waiting queue.

        Returns True if at least one worklist task was dispatched (resource is now
        busy again), False if the worklist was empty.
        """
        worklist = self._resource_worklists.get(resource, [])
        if not worklist:
            return False
        logger.info("Processing resource worklist of ressource %s", resource)

        # K-Batching: Check if resource is unavailable - if so, transfer tasks back
        if not self.allocator.availability.is_available(resource, current_time):
            logger.debug(
                "K-Batching: Resource %s is unavailable, transferring %d tasks from worklist back to waiting queue",
                resource, len(worklist)
            )
            for work in worklist:
                self.resource_pool.add_to_waiting_queue(work)
            del self._resource_worklists[resource]
            return False

        logger.info("PMSP: Processing worklist for resource %s (%d tasks)", resource, len(worklist))

        # SPT ordering: sort worklist by estimated processing time (shortest first)
        # Paper: "when multiple tasks are assigned to a resource, use shortest processing time first"
        if len(worklist) > 1:
            # Use pre-estimated PT if available, otherwise estimate on-the-fly
            def get_pt(w):
                if w.estimated_pt_seconds is not None:
                    return w.estimated_pt_seconds
                # Fallback: estimate now (shouldn't happen if tasks were added via PMSP dispatch)
                return self._estimate_pt_seconds(w, resource, current_time)
            worklist.sort(key=get_pt)

        dispatched = False
        # Dispatch exactly ONE task from the worklist (PMSP 1:1 no-preemption).
        # We intentionally do NOT use a while-loop here: resources with infinite
        # capacity (e.g. User_1, is_busy always False) would otherwise drain the
        # entire worklist in a single call, producing parallel execution.
        # The next task will be dispatched when this resource is freed again
        # and _process_resource_worklist is called for the next completion event.
        if worklist and not self.resource_pool.is_busy(resource, current_time):
            work = worklist.pop(0)  # SPT order (sorted above)

            # Calculate wait time for stats (clamp to 0 as safeguard)
            wait_seconds = max(0.0, (current_time - work.arrival_time).total_seconds())
            self.stats['wait_time_total_seconds'] += wait_seconds

            # Track assignment for strategy
            self._resource_strategy.notify_assignment(resource, work.allocation_activity)

            # Schedule the activity – this also marks the resource as busy
            self._schedule_activity_with_resource(
                work.case_id,
                work.activity,
                work.lifecycle,
                current_time,
                work.case_state,
                resource,
            )
            dispatched = True

        # Clean up the entry if the worklist is now empty
        if not worklist and resource in self._resource_worklists:
            del self._resource_worklists[resource]

        return dispatched

    def _log_resource_worklists(self) -> None:
        """Log the current worklists for all resources."""
        logger.info("PMSP [Step 8 - Worklists]: Current worklists for all resources:")
        if self._resource_worklists:
            for resource, worklist in sorted(self._resource_worklists.items()):
                if worklist:
                    logger.info("  Resource %s: %d tasks in worklist:", resource, len(worklist))
                    for idx, work in enumerate(worklist, 1):
                        logger.info(
                            "    [%d] Task '%s' (case %s, activity '%s', arrived: %s)",
                            idx,
                            work.allocation_activity,
                            work.case_id,
                            work.activity,
                            work.arrival_time,
                        )
                else:
                    logger.info("  Resource %s: worklist empty", resource)
        else:
            logger.info("  No resources have worklists")

    def _process_waiting_queue_pmsp(
        self, freed_resource: Optional[str], current_time: datetime
    ) -> None:
        """
        PMSP-mode waiting-queue processing (K-Batching adaptation).

        K-Batching behavior:
        1. Transfer tasks from unavailable resources' worklists back to waiting queue
        2. Collect all waiting tasks
        3. If k tasks have arrived (or batch_size == 0), solve PMSP for k tasks and all available resources
        4. Assign tasks to resources (add to worklists if resource is busy)
        5. When a resource becomes unavailable, tasks on its worklist are transferred back

        Only triggers optimization if the number of waiting tasks reaches
        the configured batch size (or always if batch_size == 0).
        """
        logger.info("=" * 80)
        logger.info(
            "PMSP [PROCESS START]: Processing waiting queue in PMSP mode (K-Batching) at time %s",
            current_time,
        )
        logger.info("=" * 80)
        from resources.resource_optimization.resource_optimization import (
            handle_batch_scheduling_optimization,
        )

        # K-Batching: First, transfer tasks from unavailable resources back to waiting queue
        transferred = self._transfer_unavailable_resource_worklists(current_time)
        if transferred > 0:
            logger.info(
                "PMSP [Step 1 - Transfer]: Transferred %d tasks from unavailable resources back to waiting queue",
                transferred,
            )

        # Before every optimization cycle, release all worklists back to the
        # waiting queue so PMSP always optimizes over the complete open workload.
        # We keep per-resource backlog (sum of estimated PTs) for cost modeling.
        resource_worklist_backlog: Dict[str, float] = {}  # resource -> remaining_seconds
        if self._resource_worklists:
            resources_with_worklists = sorted(
                r for r, wl in self._resource_worklists.items() if wl
            )
            for resource in resources_with_worklists:
                worklist = self._resource_worklists.get(resource, [])
                backlog_seconds = sum(
                    (w.estimated_pt_seconds or 0.0)
                    for w in worklist
                    if w.estimated_pt_seconds is not None
                )
                resource_worklist_backlog[resource] = backlog_seconds
                logger.info(
                    "PMSP [Step 1 - Transfer]: Releasing full worklist of resource %s "
                    "(%d task(s), backlog: %.1fs) back to waiting queue before optimization",
                    resource,
                    len(worklist),
                    backlog_seconds,
                )
                for work in worklist:
                    self.resource_pool.add_to_waiting_queue(work)
                del self._resource_worklists[resource]

        all_waiting_tasks = self.resource_pool.get_all_waiting_tasks()
        if not all_waiting_tasks:
            logger.info("PMSP [PROCESS END]: No waiting tasks, exiting")
            # Log worklists even if no waiting tasks
            self._log_resource_worklists()
            return

        # Filter out tasks whose arrival_time is in the future relative to current_time.
        # These are cases that arrived outside business hours and were snapped to a
        # future business-hour slot. They are not "ready" yet and must not be dispatched
        # now — doing so would produce negative wait times and wrong timestamps.
        waiting_tasks = [wt for wt in all_waiting_tasks if wt.arrival_time <= current_time]
        future_tasks  = [wt for wt in all_waiting_tasks if wt.arrival_time >  current_time]
        if future_tasks:
            logger.info(
                "PMSP [Step 2 - Queue Analysis]: %d task(s) deferred (arrival_time > current_time, "
                "business-hour snap): %s",
                len(future_tasks),
                ", ".join(f"{wt.case_id}/{wt.activity}" for wt in future_tasks),
            )
        if not waiting_tasks:
            logger.info("PMSP [PROCESS END]: No ready tasks (all deferred), exiting")
            self._log_resource_worklists()
            return

        # Log waiting tasks details
        waiting_cases = set(wt.case_id for wt in waiting_tasks)
        activity_counts = {}
        for wt in waiting_tasks:
            activity_counts[wt.allocation_activity] = activity_counts.get(wt.allocation_activity, 0) + 1
        logger.info("Activity counts are: %s", activity_counts)
        
        logger.info(
            "PMSP [Step 2 - Queue Analysis]: Found %d waiting tasks across %d cases",
            len(waiting_tasks),
            len(waiting_cases),
        )
        logger.info(
            "PMSP [Step 2 - Queue Analysis]: Activities in queue: %s",
            ", ".join(
                f"{act}({count})" for act, count in sorted(activity_counts.items())
            ),
        )
        logger.info(
            "PMSP [Step 2 - Queue Analysis]: Cases with waiting tasks: %s",
            ", ".join(sorted(waiting_cases))
            if len(waiting_cases) <= 20
            else f"{len(waiting_cases)} cases",
        )

        logger.info(
            "PMSP [Step 3 - Task Set]: Total tasks for optimization: %d",
            len(waiting_tasks),
        )

        # --- Single-resource short-circuit ---
        # If only User_1 is available (all other resources busy / off-duty),
        # running CP-SAT is pointless: the solution is always "assign eligible
        # tasks to User_1 in SPT order, leave the rest".  Do this directly and
        # skip the expensive optimizer call.
        all_known_resources = list(getattr(self.allocator.availability, 'resources', []))
        available_resources = [
            r for r in all_known_resources
            if not self.resource_pool.is_busy(r, current_time)
            and self.allocator.availability.is_available(r, current_time)
        ]
        if set(available_resources) <= {'User_1'}:
            logger.info(
                "PMSP [Step 4 - Short-circuit]: Only User_1 available (%d free) – "
                "skipping optimizer, assigning eligible tasks directly to worklist",
                len(available_resources),
            )
            user1_eligible = [
                wt for wt in waiting_tasks
                if 'User_1' in (
                    self.allocator.permissions.get_eligible_resources(wt.allocation_activity)
                    if hasattr(self.allocator.permissions, 'get_eligible_resources')
                    else []
                )
            ]
            # Estimate PT and sort ascending (SPT)
            for wt in user1_eligible:
                if wt.estimated_pt_seconds is None:
                    wt.estimated_pt_seconds = self._estimate_pt_seconds(
                        wt, 'User_1', current_time
                    )
            user1_eligible.sort(key=lambda wt: wt.estimated_pt_seconds or 0.0)
            # Move eligible tasks from waiting queue to User_1's worklist
            for wt in user1_eligible:
                removed = self.resource_pool.remove_task_by_id(
                    wt.allocation_activity, wt.case_id
                )
                if removed is not None:
                    self._resource_worklists.setdefault('User_1', []).append(removed)
            if user1_eligible:
                logger.info(
                    "PMSP [Step 4 - Short-circuit]: Added %d task(s) to User_1 worklist (SPT order)",
                    len(user1_eligible),
                )
            # Trigger worklist processing for User_1 immediately
            self._process_resource_worklist('User_1', current_time)
            self._log_resource_worklists()
            return

        logger.info("PMSP [Step 5 - Optimization]: Starting optimization...")
        with self.profiler.measure("pmsp.optimize"):
            # The PMSP step is the most compute-intensive part; print a compact timing line
            # so it shows up in stdout-based logs (sim_run_log.txt).
            opt_t0 = time.perf_counter()
            assignment, debug = handle_batch_scheduling_optimization(
                cfg=self._pmsp_config,
                timestamp=current_time,
                waiting_tasks=waiting_tasks,
                processing_time_predictor=self._processing_time,
                allocator=self.allocator,
                resource_pool=self.resource_pool,
                pt_cache=self._pmsp_pt_cache,
                resource_worklist_backlog=resource_worklist_backlog,
            )
            opt_elapsed = time.perf_counter() - opt_t0

        if assignment is not None:
            nT = (debug or {}).get("nT", len(waiting_tasks))
            nR_P = (debug or {}).get("nR_P", None)
            solver = (debug or {}).get("solver", None)
            # print(
            #     f"PMSP optimize finished: waiting_tasks={len(waiting_tasks)} nT={nT} "
            #     f"nR_P={nR_P} solver={solver} elapsed={opt_elapsed:.2f}s"
            # )
        else:
            print(f"PMSP optimize returned None (elapsed={opt_elapsed:.2f}s)")

        if assignment is None:
            logger.info("PMSP [PROCESS END]: Optimization returned no assignment, exiting")
            # Log worklists even if no assignment
            self._log_resource_worklists()
            return

        # Log assignment results
        assigned_count = sum(1 for v in assignment.values() if v is not None)
        unassigned_count = len(assignment) - assigned_count
        logger.info(
            "PMSP [Step 6 - Assignment Results]: Optimization completed - %d assigned, %d unassigned (dummy)",
            assigned_count,
            unassigned_count,
        )
        
        # Group assignments by resource for logging
        assignments_by_resource = {}
        for task_id, resource in assignment.items():
            if resource is not None:
                if resource not in assignments_by_resource:
                    assignments_by_resource[resource] = []
                assignments_by_resource[resource].append(task_id)
        
        # Log in a single, compact table-style block (resource assignments + dummy).
        unassigned_tasks = [task_id for task_id, res in assignment.items() if res is None]
        logger.info("PMSP [Step 6 - Assignment Results]: Assignments table (incl. DUMMY):")
        logger.info("  %-12s | %-5s | %s", "Resource", "Tasks", "Task IDs (truncated)")
        logger.info("  %s", "-" * 72)
        for resource, task_list in sorted(assignments_by_resource.items()):
            preview = ", ".join(task_list[:5])
            suffix = "" if len(task_list) <= 5 else f" (+{len(task_list) - 5} more)"
            logger.info("  %-12s | %5d | %s%s", resource, len(task_list), preview, suffix)
        if unassigned_tasks:
            preview = ", ".join(unassigned_tasks[:10])
            suffix = "" if len(unassigned_tasks) <= 10 else f" (+{len(unassigned_tasks) - 10} more)"
            logger.info("  %-12s | %5d | %s%s", "DUMMY", len(unassigned_tasks), preview, suffix)

        # Extract raw processing times from optimization (for SPT ordering)
        raw_processing_times = debug.get("raw_processing_times", {})  # task_id -> resource -> seconds

        # Apply assignments: dispatch tasks to assigned resources or add to worklists
        logger.info("PMSP [Step 7 - Dispatch]: Applying assignments...")
        dispatched_count = 0
        worklist_count = 0
        # Track resources that already received a dispatch in this cycle.
        # Needed for resources with infinite capacity (e.g. User_1) that are never
        # marked busy, so is_busy() can't catch a second dispatch in the same loop.
        dispatched_this_cycle: set = set()
        for task_id, assigned_resource in assignment.items():
            if assigned_resource is None:
                # Dummy assignment — task stays in the queue
                continue

            # Find the matching WaitingWork object
            matched_work = None
            for wt in waiting_tasks:
                if f"{wt.case_id}_{wt.allocation_activity}" == task_id:
                    matched_work = wt
                    break
            if matched_work is None:
                continue

            # Remove from the waiting queue
            removed = self.resource_pool.remove_task_by_id(
                matched_work.allocation_activity, matched_work.case_id
            )
            if removed is None:
                continue

            # K-Batching: Check if resource is unavailable - if so, keep task in waiting queue
            if not self.allocator.availability.is_available(assigned_resource, current_time):
                # Resource is unavailable - task stays in waiting queue (will be reassigned later)
                logger.debug(
                    "K-Batching: Resource %s is unavailable, keeping task %s for case %s in waiting queue",
                    assigned_resource, removed.activity, removed.case_id
                )
                # Re-add to waiting queue since resource is unavailable
                self.resource_pool.add_to_waiting_queue(removed)
                continue

            # Check if the assigned resource is actually free right now.
            # Also check dispatched_this_cycle: resources with infinite capacity
            # (e.g. User_1) are never marked busy globally, but within a single
            # PMSP dispatch cycle the 1:1 no-preemption rule must still hold.
            resource_occupied = (
                self.resource_pool.is_busy(assigned_resource, current_time)
                or assigned_resource in dispatched_this_cycle
            )
            if resource_occupied:
                # Resource is busy - add to its worklist instead of dispatching
                # Use pre-estimated PT from optimization if available (avoids re-estimation)
                if removed.estimated_pt_seconds is None:
                    if task_id in raw_processing_times and assigned_resource in raw_processing_times[task_id]:
                        removed.estimated_pt_seconds = raw_processing_times[task_id][assigned_resource]
                    else:
                        # Fallback: estimate now (shouldn't happen if optimization ran)
                        removed.estimated_pt_seconds = self._estimate_pt_seconds(removed, assigned_resource, current_time)
                self._resource_worklists[assigned_resource].append(removed)
                worklist_count += 1
                logger.info(
                    "PMSP [Step 7 - Dispatch]: Resource %s is busy, adding task '%s' (case %s) to worklist (estimated PT: %.1fs)",
                    assigned_resource,
                    removed.activity,
                    removed.case_id,
                    removed.estimated_pt_seconds,
                )
                continue

            # Calculate wait time for stats (clamp to 0 as safeguard)
            wait_seconds = max(0.0, (current_time - removed.arrival_time).total_seconds())
            self.stats['wait_time_total_seconds'] += wait_seconds

            dispatched_count += 1
            dispatched_this_cycle.add(assigned_resource)
            logger.info(
                "PMSP [Step 7 - Dispatch]: DISPATCHING task '%s' (case %s) -> resource %s (waited %.1fs)",
                removed.activity,
                removed.case_id,
                assigned_resource,
                wait_seconds,
            )

            # Track assignment for strategy
            self._resource_strategy.notify_assignment(assigned_resource, removed.allocation_activity)

            # Schedule the activity with the assigned resource
            self._schedule_activity_with_resource(
                removed.case_id,
                removed.activity,
                removed.lifecycle,
                current_time,
                removed.case_state,
                assigned_resource,
            )
        
        logger.info(
            "PMSP [Step 7 - Dispatch]: Dispatch summary - %d dispatched, %d to worklists",
            dispatched_count,
            worklist_count,
        )
        
        # Log worklists for all resources
        self._log_resource_worklists()

        # Post-dispatch progress dump (after the assignments have been applied).
        # This is useful for debugging why tasks remain in "waiting" even when
        # eligible + free resources exist before optimization.
        if len(self.completed_events) % 10 == 0:
            all_resources_post = getattr(self.allocator.availability, 'resources', [])
            free_resources_post = [
                r for r in all_resources_post
                if not self.resource_pool.is_busy(r, current_time)
                and self.allocator.availability.is_available(r, current_time)
            ]
            worklist_summary_post = {
                r: len(tasks) for r, tasks in self._resource_worklists.items() if tasks
            }
            waiting_summary_post = self.resource_pool.get_waiting_summary()
            waiting_summary_top_post = sorted(waiting_summary_post.items(), key=lambda x: -x[1])[:10]
            waiting_summary_str_post = ", ".join(f"{act}:{n}" for act, n in waiting_summary_top_post)
            waiting_summary_ellipsis_post = "..." if len(waiting_summary_post) > 10 else ""

            eligible_summary_post = "n/a"
            if hasattr(self.allocator, "permissions") and hasattr(self.allocator.permissions, "get_eligible_resources"):
                top_for_eligibles_post = waiting_summary_top_post[:5]
                parts_post: list[str] = []
                for act, _cnt in top_for_eligibles_post:
                    eligible = self.allocator.permissions.get_eligible_resources(act) or []
                    eligible_sorted = sorted(set(eligible))
                    max_show = 40
                    if len(eligible_sorted) <= max_show:
                        elig_str = ", ".join(eligible_sorted)
                        extra = ""
                    else:
                        elig_str = ", ".join(eligible_sorted[:max_show])
                        extra = f" ... (+{len(eligible_sorted) - max_show} more)"
                    parts_post.append(f"{act} eligibles({len(eligible_sorted)}): {elig_str}{extra}")
                if parts_post:
                    eligible_summary_post = "; ".join(parts_post)

            print(
                f"Post-assign progress: {self.stats['cases_started']} cases started, "
                f"{self.stats['cases_completed']} completed, "
                f"{len(self.completed_events)} events logged, "
                f"{self.resource_pool.get_total_waiting_count()} waiting, "
                f"simulation time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Free resources ({len(free_resources_post)}): "
                f"{', '.join(free_resources_post[:10])}"
                f"{'...' if len(free_resources_post) > 10 else ''}\n"
                f"  Worklists ({len(worklist_summary_post)} resources): "
                f"{', '.join(f'{r}:{n}' for r, n in sorted(worklist_summary_post.items())[:10])}"
                f"{'...' if len(worklist_summary_post) > 10 else ''}\n"
                f"  Waiting by activity ({len(waiting_summary_post)} activities): "
                f"{waiting_summary_str_post}{waiting_summary_ellipsis_post}\n"
                f"  Eligible resources (top waiting activities): {eligible_summary_post}",
                flush=True,
            )
        
        logger.info("=" * 80)
        logger.info("PMSP [PROCESS END]: Completed PMSP processing cycle")
        logger.info("=" * 80)

    def _process_waiting_queue_batch(
        self, freed_resource: str, current_time: datetime
    ) -> None:
        """
        Batch-mode waiting-queue processing (1-Batch-1 / MSA).

        Collects all waiting tasks, builds the eligible worker set, and
        delegates to the configured BatchAllocationPolicy.  Only the
        assignment for *freed_resource* is committed.
        """
        from resources.batch_policies import TaskInfo, WorkerInfo

        # 1. Collect all waiting tasks
        all_waiting = self.resource_pool.get_all_waiting_tasks()
        if not all_waiting:
            return

        # 2. Build TaskInfo list
        tasks: list[TaskInfo] = []
        for w in all_waiting:
            hours_waited = (current_time - w.arrival_time).total_seconds() / 3600.0
            tasks.append(TaskInfo(
                task_id=f"{w.case_id}::{w.allocation_activity}",
                case_id=w.case_id,
                activity=w.activity,
                allocation_activity=w.allocation_activity,
                hours_waited=max(0.0, hours_waited),
            ))

        # 3. Build eligible_map: allocation_activity -> set of eligible worker IDs
        #    Cache per activity since many tasks share the same activity
        eligible_map: dict[str, set[str]] = {}
        unique_activities = {t.allocation_activity for t in tasks}

        for act in unique_activities:
            # Tier 1: permissions
            try:
                eligible = self.allocator.permissions.get_eligible_resources(
                    act, timestamp=current_time
                )
            except TypeError:
                eligible = self.allocator.permissions.get_eligible_resources(act)

            # Tier 2: availability (working hours)
            available = {
                r for r in eligible
                if self.allocator.availability.is_available(r, current_time)
            }
            eligible_map[act] = available

        # 3b. Scope to freed worker's competitive neighborhood
        neighbor_tasks = {t.allocation_activity for t in tasks
                         if freed_resource in eligible_map.get(t.allocation_activity, set())}
        neighbor_workers: set[str] = set()
        for act in neighbor_tasks:
            neighbor_workers.update(eligible_map[act])
        for act, workers_set in eligible_map.items():
            if neighbor_workers & workers_set:
                neighbor_tasks.add(act)
        tasks = [t for t in tasks if t.allocation_activity in neighbor_tasks]
        eligible_map = {act: ws for act, ws in eligible_map.items() if act in neighbor_tasks}

        # 3c. Neighborhood too large — skip MILP entirely, dispatch greedily
        if len(tasks) > 50:
            sorted_tasks = sorted(tasks, key=lambda t: t.hours_waited, reverse=True)
            for t in sorted_tasks:
                if freed_resource in eligible_map.get(t.allocation_activity, set()):
                    parts = t.task_id.split("::", 1)
                    if len(parts) != 2:
                        break
                    case_id, allocation_activity = parts
                    removed = self.resource_pool.remove_task_by_id(allocation_activity, case_id)
                    if removed is None:
                        break
                    wait_seconds = (current_time - removed.arrival_time).total_seconds()
                    self.stats['wait_time_total_seconds'] += wait_seconds
                    logger.debug(
                        "Neighborhood too large (%d tasks), greedy dispatch %s for case %s to %s (waited %.0fs)",
                        len(tasks), removed.activity, removed.case_id, freed_resource, wait_seconds,
                    )
                    self._resource_strategy.notify_assignment(freed_resource, allocation_activity)
                    self._batch_policy._diag_greedy_neighborhood_too_large += 1
                    self._schedule_activity_with_resource(
                        removed.case_id, removed.activity, removed.lifecycle, current_time,
                        removed.case_state, freed_resource,
                    )
                    return
            # No eligible task found for freed worker in this large neighborhood
            return

        # 4. Build WorkerInfo list: union of all eligible workers (including busy)
        all_eligible_ids: set[str] = set()
        for s in eligible_map.values():
            all_eligible_ids.update(s)

        workers: list[WorkerInfo] = []
        for wid in all_eligible_ids:
            busy_until = self.resource_pool.get_busy_until(wid)
            if busy_until is not None:
                remaining = max(0.0, (busy_until - current_time).total_seconds())
            else:
                remaining = 0.0
            workers.append(WorkerInfo(worker_id=wid, remaining_busy_seconds=remaining))

        # 5. Call batch policy
        decision = self._batch_policy.decide(
            freed_resource=freed_resource,
            current_time_s=current_time.timestamp(),
            tasks=tasks,
            workers=workers,
            eligible_map=eligible_map,
            processing_time_fn=self._pt_estimator.estimate,
        )

        if decision is None:
            logger.debug(
                "Batch policy returned no assignment for %s at %s",
                freed_resource, current_time,
            )
            return

        # 6. Parse task_id -> (case_id, allocation_activity)
        parts = decision.task_id.split("::", 1)
        if len(parts) != 2:
            logger.warning("Invalid task_id from batch policy: %s", decision.task_id)
            return
        case_id, allocation_activity = parts

        # Remove the task from its waiting queue
        removed = self.resource_pool.remove_task_by_id(allocation_activity, case_id)
        if removed is None:
            logger.warning(
                "Batch policy assigned task %s but it was not found in queue",
                decision.task_id,
            )
            return

        # Calculate wait time for stats
        wait_seconds = (current_time - removed.arrival_time).total_seconds()
        self.stats['wait_time_total_seconds'] += wait_seconds

        logger.debug(
            "Batch dispatching %s for case %s to %s (waited %.0fs)",
            removed.activity, removed.case_id, freed_resource, wait_seconds,
        )

        # Track assignment for strategy (keeps SHQ counts accurate)
        self._resource_strategy.notify_assignment(freed_resource, allocation_activity)

        # Schedule the activity with the freed resource
        self._schedule_activity_with_resource(
            removed.case_id,
            removed.activity,
            removed.lifecycle,
            current_time,
            removed.case_state,
            freed_resource,
        )

    def _process_waiting_queue_drl(
        self, freed_resource: str, current_time: datetime
    ) -> None:
        """
        DRL-mode waiting-queue processing.

        Builds TaskInfo list and eligible_map (no neighborhood scoping),
        applies auto-postpone filter, then delegates to the DRL policy.
        """
        from resources.batch_policies import TaskInfo

        # 1. Collect all waiting tasks
        all_waiting = self.resource_pool.get_all_waiting_tasks()
        if not all_waiting:
            return

        # 2. Build TaskInfo list
        tasks: list[TaskInfo] = []
        for w in all_waiting:
            hours_waited = (current_time - w.arrival_time).total_seconds() / 3600.0
            tasks.append(TaskInfo(
                task_id=f"{w.case_id}::{w.allocation_activity}",
                case_id=w.case_id,
                activity=w.activity,
                allocation_activity=w.allocation_activity,
                hours_waited=max(0.0, hours_waited),
            ))

        # 3. Build eligible_map (no neighborhood scoping for DRL)
        eligible_map: dict[str, set[str]] = {}
        unique_activities = {t.allocation_activity for t in tasks}

        for act in unique_activities:
            try:
                eligible = self.allocator.permissions.get_eligible_resources(
                    act, timestamp=current_time
                )
            except TypeError:
                eligible = self.allocator.permissions.get_eligible_resources(act)

            available = {
                r for r in eligible
                if self.allocator.availability.is_available(r, current_time)
            }
            eligible_map[act] = available

        # 4. Auto-postpone: skip decide() if freed resource can't serve any waiting activity
        any_feasible = any(
            freed_resource in eligible_map.get(t.allocation_activity, set())
            for t in tasks
        )
        if not any_feasible:
            return

        # 5. Compute per-activity max wait hours (for postpone starvation prevention)
        task_max_wait: dict[str, float] = {}
        for t in tasks:
            current_max = task_max_wait.get(t.allocation_activity, 0.0)
            if t.hours_waited > current_max:
                task_max_wait[t.allocation_activity] = t.hours_waited

        # 6. Provide engine state to policy (for observation building)
        if hasattr(self._drl_policy, 'set_engine_state'):
            self._drl_policy.set_engine_state(self.resource_pool, self.case_manager)

        # 7. Build pool snapshot for training bridge (InteractiveBatchPolicy)
        pool_snapshot = self._build_pool_snapshot()
        waiting_activities = {t.allocation_activity for t in tasks if self.resource_pool.has_waiting_work(t.allocation_activity)}

        # 8. Call policy
        decision = self._drl_policy.decide(
            freed_resource=freed_resource,
            current_time_s=current_time.timestamp(),
            tasks=tasks,
            workers=[],  # Not used by DRL policy
            eligible_map=eligible_map,
            processing_time_fn=lambda w, a: 0.0,  # Not used by DRL policy
            current_time_dt=current_time,
            waiting_activities=waiting_activities,
            pool_snapshot=pool_snapshot,
            active_case_count=self.case_manager.active_count(),
            task_max_wait_hours=task_max_wait,
            max_postpone_wait_hours=self._drl_max_postpone_wait_hours,
        )

        if decision is None:
            return

        # 9. Parse task_id and dispatch
        parts = decision.task_id.split("::", 1)
        if len(parts) != 2:
            logger.warning("Invalid task_id from DRL policy: %s", decision.task_id)
            return
        case_id, allocation_activity = parts

        removed = self.resource_pool.remove_task_by_id(allocation_activity, case_id)
        if removed is None:
            logger.warning(
                "DRL policy assigned task %s but it was not found in queue",
                decision.task_id,
            )
            return

        wait_seconds = (current_time - removed.arrival_time).total_seconds()
        self.stats['wait_time_total_seconds'] += wait_seconds

        logger.debug(
            "DRL dispatching %s for case %s to %s (waited %.0fs)",
            removed.activity, removed.case_id, freed_resource, wait_seconds,
        )

        self._resource_strategy.notify_assignment(freed_resource, allocation_activity)

        self._schedule_activity_with_resource(
            removed.case_id,
            removed.activity,
            removed.lifecycle,
            current_time,
            removed.case_state,
            freed_resource,
        )

    def _build_pool_snapshot(self) -> dict:
        """Serialize resource pool state into plain dicts for thread-safe access."""
        busy = {}
        for res_id, (busy_until, case_id, activity) in self.resource_pool._busy_resources.items():
            busy[res_id] = {
                "busy_until": busy_until,
                "case_id": case_id,
                "activity": activity,
            }

        waiting = {}
        for act, queue_items in self.resource_pool._waiting_queues.items():
            if queue_items:
                waiting[act] = [
                    {
                        "case_id": w.case_id,
                        "activity": w.activity,
                        "allocation_activity": w.allocation_activity,
                        "arrival_time": w.arrival_time,
                    }
                    for w in queue_items
                ]

        return {"busy_resources": busy, "waiting_queues": waiting}

    def _on_case_end(self, event: SimulationEvent) -> None:
        """Handle case end: cleanup."""
        self.stats['cases_completed'] += 1
        self.case_manager.remove_case(event.case_id)
        # Clean up predictor state for this case (prevents memory leak)
        if hasattr(self._next_activity, 'reset_case'):
            self._next_activity.reset_case(event.case_id)
        # Invalidate all PT cache entries belonging to this case
        if self._pmsp_pt_cache:
            keys_to_remove = [k for k in self._pmsp_pt_cache if k[0] == event.case_id]
            for k in keys_to_remove:
                del self._pmsp_pt_cache[k]
    
    # Wird aufgerufen bei acticity completion oder case arrival
    def _schedule_activity(self, case_id: str, activity: str, lifecycle: str,
                           current_time: datetime, case: CaseState) -> None:
        """Allocate resource and schedule activity completion, or queue if unavailable."""
        # Some activities are control-flow artifacts (e.g., decision points) and must not
        # require an organizational resource.
        if not self._activity_requires_resource(activity):
            self._schedule_activity_without_resource(case_id, activity, lifecycle, current_time, case)
            return

        allocation_activity = self._normalize_activity_for_resources(activity)

        if self._pmsp_config is None:
            # Try to allocate a resource (with dynamic busy checking)
            # This must be handled differently in pmsp. Activities always get queued in waiting queue for optimization. Resource unavailability gets handled differentlty
            with self.profiler.measure("resource_allocation"):
                resource, failure_reason = self._try_allocate_resource_with_reason(allocation_activity, current_time, case)

            if resource is None:
                # No resource available - add to waiting queue
                self.stats['waiting_events'] += 1
                waiting_work = WaitingWork(
                    case_id=case_id,
                    activity=activity,
                    lifecycle=lifecycle,
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
                case_id, activity, lifecycle, current_time, case, resource
            )

        else: 
            # Im optimization mode werden die tasks erst in batches gesammelt und nicht sofort zugeorndet. 
            # Hier ist eine Resource freigeworden 
            # NOTE: This log line used to say "Activity completed...", but at this point
            # we are *not* completing anything. In PMSP mode we enqueue work for batch
            # optimization and only later dispatch via _process_waiting_queue_pmsp().
            logger.info(
                "PMSP: Queuing activity for batch optimization (case=%s, activity=%s %s). Checking if optimization already applies ...",
                case_id,
                activity,
                lifecycle,
            )
            self.stats['waiting_events'] += 1
            waiting_work = WaitingWork(
                case_id=case_id,
                activity=activity,
                lifecycle=lifecycle,
                allocation_activity=allocation_activity,
                arrival_time=current_time,
                case_state=case,
            )
            self.resource_pool.add_to_waiting_queue(waiting_work)

            waiting_tasks = self.resource_pool.get_all_waiting_tasks()
            # Very chatty (runs on every enqueue). Keep at DEBUG.
            logger.info("PMSP: Currently waiting tasks: %d", len(waiting_tasks))
            batch_size = self._pmsp_config.optimization_batch_size
            if batch_size == 0 or len(waiting_tasks) >= batch_size:
                logger.info("PMSP: Threshold of batch size reached! Triggering optimization")
                self._process_waiting_queue_pmsp(None, current_time)

    def _activity_requires_resource(self, activity: Optional[str]) -> bool:
        """Return True if this activity needs an org resource assignment."""
        if not activity:
            return True
        # Decision points / process gateways are not executed by a human resource.
        # Additionally, certain system-generated application events should not require
        # human resources (they occur instantly in GT and are not performed by users).
        RESOURCE_FREE_ACTIVITIES = {
            "A_Create Application",
            "A_Submitted",
            "A_Concept",
        }
        normalized = self._normalize_activity_label(activity)
        return not (
            activity.startswith('DP ')
            or activity.startswith('PG ')
            or (normalized in RESOURCE_FREE_ACTIVITIES)
        )

    def _normalize_activity_for_resources(self, activity: str) -> str:
        """Normalize activity labels to match the resource permission model."""
        return self._normalize_activity_label(activity)

    def _normalize_activity_label(self, activity: Optional[str]) -> Optional[str]:
        """Normalize labels by removing loop suffixes (e.g. 'X 2' -> 'X')."""
        if not activity:
            return activity
        if activity.startswith('DP ') or activity.startswith('PG '):
            return activity

        normalized = re.sub(r"\s+\d+$", "", activity).strip()
        return normalized or activity

    def _should_emit_pt_gt_start(self, activity: Optional[str]) -> bool:
        """Return True if this activity gets a synthetic start in PT GT-gated mode."""
        if self._pt_lifecycle_mode != "gt_activity_gated":
            return False
        if not self._is_process_transformer_predictor:
            return False
        normalized = self._normalize_activity_label(activity)
        return normalized in self.PT_GT_START_ACTIVITIES

    def _append_synthetic_start_record(
        self,
        case_id: str,
        activity: str,
        resource: Optional[str],
        timestamp: datetime,
        case: CaseState,
    ) -> None:
        """Append synthetic 'start' event directly to export records."""
        if not self._should_emit_pt_gt_start(activity):
            return

        log_record = {
            'case:concept:name': case_id,
            'concept:name': activity,
            'org:resource': resource,
            'time:timestamp': timestamp,
            'lifecycle:transition': 'start',
        }
        log_record.update(case.get_payload())
        self.completed_events.append(log_record)

    def _schedule_activity_without_resource(
        self,
        case_id: str,
        activity: str,
        lifecycle: str,
        current_time: datetime,
        case: CaseState,
    ) -> None:
        """Schedule an activity completion without allocating any resource."""
        prev_activity = case.activity_history[-1] if case.activity_history else "START"
        prev_lifecycle = case.lifecycle_history[-1] if case.lifecycle_history else "complete"

        # Many learned processing-time models won't know DP/PG labels.
        # For control-flow artifacts, treat duration as instantaneous.
        processing_seconds = 0.0

        # Special case: START is a synthetic sentinel – use 1 second (see _schedule_activity_with_resource).
        if prev_activity == "START":
            completion_time = current_time + timedelta(seconds=1.0)
            event = SimulationEvent(
                timestamp=completion_time,
                event_type=EventType.ACTIVITY_COMPLETE,
                case_id=case_id,
                activity=activity,
                lifecycle=lifecycle,
                resource=None,
                payload=case.get_payload(),
            )
            self.queue.schedule(event)
            return

        try:
            with self.profiler.measure("processing_time.predict"):
                processing_seconds = float(self._processing_time.predict(
                    prev_activity=prev_activity,
                    prev_lifecycle=prev_lifecycle,
                    curr_activity=activity,
                    curr_lifecycle=lifecycle,
                    context={
                        'case_id': case_id,
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

        self._append_synthetic_start_record(case_id, activity, None, current_time, case)

        event = SimulationEvent(
            timestamp=completion_time,
            event_type=EventType.ACTIVITY_COMPLETE,
            case_id=case_id,
            activity=activity,
            lifecycle=lifecycle,
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

        # Apply resource selection heuristic (R-RMA / R-RRA / R-SHQ)
        selected = self._resource_strategy.select(truly_available, activity)
        self._resource_strategy.notify_assignment(selected, activity)
        return selected, None

    def _schedule_activity_with_resource(self, case_id: str, activity: str, lifecycle: str,
                                          current_time: datetime, case: CaseState,
                                          resource: str) -> None:
        """Schedule activity completion with an allocated resource."""
        prev_activity = case.activity_history[-1] if case.activity_history else "START"
        prev_lifecycle = case.lifecycle_history[-1] if case.lifecycle_history else "complete"

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
            'case_id': case_id,
        }

        # Special case: START is a synthetic sentinel, not a real activity.
        # The arrival timestamps are derived from the first real event in the log,
        # so predicting START→first_activity would just shift all events forward.
        # Use 1 second as a neutral placeholder to avoid this double-counting.
        if prev_activity == "START":
            processing_seconds = 1.0
        else:
            with self.profiler.measure("processing_time.predict"):
                processing_seconds = self._processing_time.predict(
                    # Request inter-event time (complete → complete) to use trained distributions
                    # This allows the model to predict realistic processing times based on training data
                    prev_activity=prev_activity,
                    prev_lifecycle=prev_lifecycle,
                    curr_activity=activity,
                    curr_lifecycle=lifecycle,
                    context=context,
                )

        # Apply per-transition P99 cap from GT distributions
        if self._transition_p99_caps:
            transition_key = (prev_activity, prev_lifecycle, activity, lifecycle)
            cap = self._transition_p99_caps.get(transition_key)
            if cap is None:
                # Fallback: match by activity pair, ignoring lifecycle
                for key, val in self._transition_p99_caps.items():
                    if key[0] == prev_activity and key[2] == activity:
                        cap = val
                        break
            if cap is not None and processing_seconds > cap:
                logger.debug(
                    "P99 cap: %s→%s capped %.1fs → %.1fs",
                    prev_activity, activity, processing_seconds, cap,
                )
                processing_seconds = cap

        processing_time = timedelta(seconds=processing_seconds)
        completion_time = current_time + processing_time

        # NOTE: Completion clamping to resource working hours was tried and reverted.
        # Per-resource clamping created a ~3500-event spike at 7 AM (first available hour)
        # and inflated TPT by ~100h (548→648h vs GT 526h). The processing time predictor
        # already embeds overnight gaps in its inter-event predictions, so clamping
        # double-counts the delay. See commit history on fix/event_distribution branch.
        # Future work: "pause clock" approach that subtracts off-hours from predicted
        # duration rather than shifting completion forward.

        # Resource is held until the task actually completes (completion_time).
        # Previously a separate predict_resource_hold_time() was used to free the
        # resource earlier than completion (modelling "work burst vs. inter-event gap").
        # This violated the PMSP 1:1 no-preemption assumption: the resource was
        # reassigned while still officially mid-task, causing parallel execution.
        resource_release_time = completion_time

        # Helpful visibility in PMSP mode: when we dispatch, completion might be far
        # in the future (e.g., long predicted inter-event times), which can look like
        # "nothing is being processed".
        if self._pmsp_config is not None:
            logger.info(
                "PMSP: Scheduled %s %s for case %s on %s (pt=%.1fs) -> completes at %s (resource hold until %s)",
                activity,
                lifecycle,
                case_id,
                resource,
                float(processing_seconds),
                completion_time,
                resource_release_time,
            )
        self.resource_pool.mark_busy(resource, resource_release_time, case_id, activity)
        self._append_synthetic_start_record(case_id, activity, resource, current_time, case)

        # Schedule completion event
        event = SimulationEvent(
            timestamp=completion_time,
            event_type=EventType.ACTIVITY_COMPLETE,
            case_id=case_id,
            activity=activity,
            lifecycle=lifecycle,
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

    def _is_business_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within business hours (Mon-Fri, 8-17, no holidays)."""
        avail = self.allocator.availability
        start_hour = getattr(avail, 'workday_start_hour', 8)
        end_hour = getattr(avail, 'workday_end_hour', 17)
        weekday = timestamp.weekday()
        hour = timestamp.hour
        is_holiday = timestamp.date() in avail.nl_holidays
        return weekday < 5 and start_hour <= hour < end_hour and not is_holiday

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
        # BUG FIX: Also check holidays!
        is_holiday = current_time.date() in self.allocator.availability.nl_holidays
        if weekday < 5 and start_hour <= hour < end_hour and not is_holiday:
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

        # Skip weekends AND holidays
        while next_time.weekday() >= 5 or next_time.date() in self.allocator.availability.nl_holidays:
            next_time += timedelta(days=1)
            # Reset to start of day when skipping
            next_time = next_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)

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
        max_iterations = 50000  # Safety limit increased for large simulations (was 100)
        iterations_without_progress = 0
        max_no_progress = 1000  # Stop if 1000 time advances produce no dispatches (was 3)

        initial_waiting = self.resource_pool.get_total_waiting_count()
        logger.info(f"Drain phase starting with {initial_waiting} waiting cases")

        failure_reason = "No waiting work"  # Default reason

        while self.resource_pool.has_waiting_work() and iterations_without_progress < max_no_progress:
            current_time = self.clock.now
            dispatched_this_round = 0

            # If PMSP is enabled, run a single PMSP optimization cycle instead of
            # greedy per-activity allocation. This ensures the final leftover work
            # respects the configured optimization batch behavior.
            if self._pmsp_config is not None:
                waiting_before = self.resource_pool.get_total_waiting_count()
                self._process_waiting_queue_pmsp(None, current_time)
                waiting_after = self.resource_pool.get_total_waiting_count()

                if waiting_after < waiting_before:
                    dispatched_this_round = waiting_before - waiting_after
                elif not self.queue.is_empty():
                    # PMSP scheduled completion events even if tasks remain in queues
                    # (e.g., transferred to resource worklists).
                    dispatched_this_round = 1
                else:
                    failure_reason = "PMSP no progress"
            else:
                # Get all waiting activities
                waiting_activities = self.resource_pool.get_all_waiting_activities()

                # Try to dispatch each waiting activity
                for activity in list(waiting_activities):
                    while self.resource_pool.has_waiting_work(activity):
                        # Try to allocate a resource
                        waiting_work = self.resource_pool.peek_waiting_work(activity)
                        if not waiting_work:
                            break

                        resource, reason = self._try_allocate_resource_with_reason(
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
                                work.case_id, work.activity, work.lifecycle, current_time,
                                work.case_state, resource
                            )
                            dispatched_this_round += 1
                        else:
                            # No resource available for this activity right now
                            failure_reason = reason
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
                logger.debug(
                    f"[Drain] No resources available (last reason: {failure_reason}), "
                    f"advancing {time_jump} to {next_business_hour}"
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


# ---------------------------------------------------------------------------
# Training DES Engine — subclass for DRL training
# ---------------------------------------------------------------------------

class TrainingDESEngine(DESEngine):
    """
    DESEngine subclass that captures per-case cycle times for reward computation.

    Used during DRL training: the Gym env reads completed case cycle times
    via pop_completed_cases() after each decision step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completed_cycle_times: list[float] = []
        self._lock = threading.Lock()

    def _on_case_end(self, event: SimulationEvent) -> None:
        """Override: capture cycle time before cleanup."""
        case = self.case_manager.get_case(event.case_id)
        if case and case.start_time:
            ct_hours = (event.timestamp - case.start_time).total_seconds() / 3600.0
            with self._lock:
                self._completed_cycle_times.append(ct_hours)
        super()._on_case_end(event)

    def pop_completed_cases(self) -> list[float]:
        """Return and clear accumulated cycle times (thread-safe)."""
        with self._lock:
            times = self._completed_cycle_times[:]
            self._completed_cycle_times.clear()
        return times

    def run(self, num_cases: int = 100, max_time: datetime = None) -> List[Dict]:
        """Override: signal episode end to DRL bridge after run completes."""
        try:
            result = super().run(num_cases=num_cases, max_time=max_time)
        finally:
            if self._drl_policy is not None and hasattr(self._drl_policy, 'signal_episode_end'):
                self._drl_policy.signal_episode_end()
        return result


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


class LSTMNextActivityPredictor:
    """
    Advanced next activity predictor using LSTM models per decision point.
    
    Wraps the existing decision_function_advanced() from Next-Activity-Prediction/advanced.
    """
    
    END_ACTIVITIES = {"A_Cancelled", "A_Complete"}
    START_ACTIVITY = "A_Create Application"
    
    def __init__(
        self,
        models_dir: str = "Next-Activity-Prediction/advanced/models_lstm_new",
        max_history: int = 15,
        seed: int = 42,
    ):
        """Load LSTM models, process graph, and decision point map."""
        import sys
        import importlib
        from pathlib import Path
        
        # Add Next-Activity-Prediction parent to sys.path.
        # IMPORTANT: avoid importing a top-level module named "simulation" here, because
        # this project already has a package named "simulation".
        na_root = Path(__file__).parent.parent / "Next-Activity-Prediction"
        if str(na_root) not in sys.path:
            sys.path.insert(0, str(na_root))

        advanced_api = importlib.import_module("advanced.api")
        advanced_sim = importlib.import_module("advanced.simulation")

        load_models = advanced_api.load_models
        load_simulation_assets = advanced_sim.load_simulation_assets
        decision_function_advanced = advanced_sim.decision_function_advanced
        
        self.models = load_models(models_dir)
        self.process_graph, self.decision_point_map = load_simulation_assets()
        self.decision_function = decision_function_advanced
        self.max_history = max_history
        self.rng = random.Random(seed)
        
        # Build activity -> DP mapping
        self._activity_to_dp = {}
        for dp, info in self.decision_point_map.items():
            if "incoming" in info:
                for act in info["incoming"]:
                    self._activity_to_dp[act] = dp
        
        logger.info(
            f"Loaded LSTMNextActivityPredictor: {len(self.models)} models, "
            f"{len(self.process_graph)} nodes, {len(self._activity_to_dp)} activity-to-DP mappings"
        )
    
    def predict(self, case_state: CaseState) -> tuple[str, bool]:
        """Predict next activity using LSTM models at decision points."""
        if not case_state.activity_history:
            return self.START_ACTIVITY, False
        
        current = case_state.activity_history[-1]
        
        if current in self.END_ACTIVITIES:
            return current, True
        
        next_options = self.process_graph.get(current, [])
        
        if not next_options:
            logger.debug(f"No outgoing edges for {current}, ending case")
            return current, True
        
        if len(next_options) == 1:
            next_act = next_options[0]
            return next_act, next_act in self.END_ACTIVITIES
        
        # Multiple options - use LSTM at decision point
        decision_point = self._activity_to_dp.get(current)
        
        if not decision_point:
            # No DP mapping - random choice
            logger.debug(f"No DP for {current}, random choice from {next_options}")
            next_act = self.rng.choice(next_options)
            return next_act, next_act in self.END_ACTIVITIES

        # Use outgoing activities from decision point map (matches advanced/simulation.py)
        dp_info = self.decision_point_map.get(decision_point, {})
        decision_options_raw = dp_info.get('outgoing', next_options)

        # Filter out DPs/PGs; engine will still handle them, but feeding real activities
        # to the LSTM decision improves stability and avoids DP->DP loops.
        decision_options = [
            act for act in decision_options_raw
            if not (act.startswith('DP ') or act.startswith('PG '))
        ]
        if not decision_options:
            decision_options = list(decision_options_raw)
        
        # Convert CaseState to history format
        history = self._case_state_to_history(case_state)
        
        try:
            next_act, prob = self.decision_function(
                current_dp=decision_point,
                history=history,
                options=decision_options,
                models=self.models,
                decision_point_map=self.decision_point_map,
                process_graph=self.process_graph,
                max_history=self.max_history,
            )
            logger.debug(f"LSTM predicted {next_act} at {decision_point} (p={prob:.3f})")
            return next_act, next_act in self.END_ACTIVITIES
        except Exception as e:
            logger.warning(f"LSTM prediction failed at {decision_point}: {e}, using random")
            next_act = self.rng.choice(next_options)
            return next_act, next_act in self.END_ACTIVITIES
    
    def _case_state_to_history(self, case_state: CaseState) -> List[Dict]:
        """Convert CaseState to history format for decision_function_advanced."""
        from datetime import timedelta
        
        base_time = case_state.start_time or datetime.now()
        history = []
        
        for i, activity in enumerate(case_state.activity_history):
            history.append({
                'task': activity,
                'resource': case_state.current_resource or 'User_1',
                'timestamp': base_time + timedelta(seconds=i),
            })
        
        return history


class UnifiedNextActivityPredictor:
    """
    Unified next activity predictor using dual-output LSTM model.
    
    Predicts both activity AND lifecycle to avoid simulation loops.
    Uses repetition penalty on seen (activity, lifecycle) pairs.
    """
    
    END_ACTIVITIES = {"A_Cancelled", "A_Complete", "End"}
    START_ACTIVITY = "A_Create Application"
    DEFAULT_MODEL_PATH = "models/unified_next_activity"
    
    def __init__(self, model_path: str = None, max_history: int = 15, seed: int = 42):
        """Initialize the unified predictor."""
        self.rng = random.Random(seed)
        self.max_history = max_history
        model_path = model_path or self.DEFAULT_MODEL_PATH
        
        # Import and load the unified predictor
        from pathlib import Path
        import sys
        import importlib
        
        # Add Next-Activity-Prediction to path (so we can import advanced.unified)
        project_root = Path(__file__).parent.parent
        na_root = project_root / "Next-Activity-Prediction"
        if str(na_root) not in sys.path:
            sys.path.insert(0, str(na_root))
        
        # Import the unified predictor module
        # Import parent packages first to ensure package structure is recognized
        # This ensures relative imports in predictor.py (like "from .persistence") work correctly
        try:
            # Import in order to establish package hierarchy
            import advanced
            import advanced.unified
            # Now import the predictor module - relative imports should work
            predictor_module = importlib.import_module("advanced.unified.predictor")
            _Impl = predictor_module.UnifiedNextActivityPredictor
        except ImportError as e:
            logger.error(f"Failed to import unified predictor: {e}")
            logger.error(f"Python path: {sys.path[:3]}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        self._impl = _Impl(model_path=model_path, max_history=max_history, seed=seed)
        
        logger.info(f"Loaded UnifiedNextActivityPredictor from {model_path}")
    
    def predict(self, case_state: CaseState) -> tuple[str, bool]:
        """Predict next activity using unified model."""
        return self._impl.predict(case_state)


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
