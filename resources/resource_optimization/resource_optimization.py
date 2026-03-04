"""
resources.optimized_allocation

Utility functions for *in-simulation* resource selection based on predicted processing times.

Goal
----
Replace the current random choice among available resources with a more
"optimized" choice that is compatible with the paper's idea of a *dummy*
assignment (postponing work even if a resource is available).

This module is deliberately *pure* (no PM4Py / no log loading):
- It only needs a `ProcessingTimePredictor`-like object with `.predict(...)`
- and the current simulation state (`CaseState`, timestamp, etc.).

Selection modes
---------------
- "random": caller chooses random; this module not used
- "min_time": choose available resource with smallest predicted processing time
- "min_time_with_dummy": choose smallest predicted processing time, but allow returning
  None ("dummy") if the best available resource is still "too slow" compared to the
  average across *all eligible* resources.

Dummy rule (paper-style, simplified for online dispatch)
--------------------------------------------------------
Let:
- R_auth(t) = eligible resources for task t (from permission model)
- c(t,r)    = predicted processing time for task t on resource r
- c̄(t)      = average_{r in R_auth(t)} c(t,r)
- c_d(t)    = δ * c̄(t)   (dummy cost)

If min_{r in R_avail(t)} c(t,r) > c_d(t), we return None (postpone the task).
Otherwise, we return the argmin resource among the currently available resources.

Notes
-----
- If |R_auth(t)| <= 1, the dummy rule is usually undesirable for δ < 1
  (it would postpone forever). We therefore disable the dummy in that case.
- If the predictor fails or returns invalid values, we fall back to a
  deterministic (sorted) choice to keep the simulation robust.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math
import random

# LSTM predictor for resource optimization (fallback when ProcessTransformer is used)
_lstm_predictor_cache: Optional[Any] = None

import numpy as np
from scipy.optimize import linear_sum_assignment

from cvxopt import matrix
from cvxopt.glpk import ilp, options as glpk_options


@dataclass(frozen=True)
class SelectionConfig:
    mode: str = "random"  # random | min_time | min_time_with_dummy | pmsp | k_batching
    allow_dummy: bool = False
    dummy_delta: float = 1.0  # δ
    avg_sample_size: int = 25  # for large eligible sets, sample for avg-cost estimate
    rng_seed: Optional[int] = None
    pmsp_solver_time_limit_seconds: Optional[float] = None
    pmsp_max_solves_per_run: Optional[int] = 1000
    k_batch_size: int = 10
    prediction_batch_size: int = 0  # Max total predictions per optimization run (0 = unlimited)


def solve_assignment_jv(
    *,
    tasks: List[str],
    resources: List[str],
    costs: Dict[Tuple[str, str], int],
    dummy_cost: Dict[str, int],
    remaining: Dict[str, int],
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """
    Solve assignment adaptation with Jonker-Volgenant (scipy linear_sum_assignment).
    Worst-case O(N^3), where N = max(n_tasks, n_resources + n_dummy_cols).
    """
    T = list(tasks)
    R = list(resources)
    nT = len(T)
    nR = len(R)
    assignment: Dict[str, Optional[str]] = {t: None for t in T}
    if nT == 0:
        return assignment, {"solver": "scipy_jv", "nT": 0, "nR": nR, "objective": 0}

    # One dummy column per task so multiple tasks can be postponed independently.
    n_cols = nR + nT
    big_m = float(10**9)
    C = np.full((nT, n_cols), big_m, dtype=np.float64)

    for i, t in enumerate(T):
        # Real resources (one task per resource at this decision point)
        for j, r in enumerate(R):
            if (t, r) in costs:
                C[i, j] = float(remaining.get(r, 0) + costs[(t, r)])
        # Task-specific dummy option
        C[i, nR + i] = float(dummy_cost.get(t, 10**9))

    row_ind, col_ind = linear_sum_assignment(C)
    objective = 0.0
    for i, j in zip(row_ind, col_ind):
        t = T[i]
        objective += float(C[i, j])
        if j < nR and C[i, j] < big_m:
            assignment[t] = R[j]
        else:
            assignment[t] = None

    return assignment, {
        "solver": "scipy_jv",
        "nT": nT,
        "nR": nR,
        "objective": int(round(objective)),
    }


def build_processing_time_context(
    case: Any,
    current_time: datetime,
    candidate_resource: str,
    *,
    case_id: Optional[str] = None,
    prev_resource: Optional[str] = None,
    event_position_in_case: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a context dict compatible with ProcessingTimePredictionClass."""
    cid = case_id or getattr(case, "case_id", None)
    prev_res = prev_resource or getattr(case, "current_resource", None) or "unknown"
    pos = event_position_in_case
    if pos is None:
        hist = getattr(case, "activity_history", []) or []
        pos = len(hist) + 1

    start_time = getattr(case, "start_time", None)
    dur_so_far = (current_time - start_time).total_seconds() if start_time else 0.0

    return {
        "case_id": cid,
        "hour": current_time.hour,
        "weekday": current_time.weekday(),
        "month": current_time.month,
        "day_of_year": current_time.timetuple().tm_yday,
        "case:LoanGoal": getattr(case, "case_type", None),
        "case:ApplicationType": getattr(case, "application_type", None),
        "event_position_in_case": pos,
        "case_duration_so_far": dur_so_far,
        "resource_1": prev_res,
        "resource_2": candidate_resource,
        "Accepted": getattr(case, "accepted", None),
        "Selected": getattr(case, "selected", None),
    }


_MAX_SECONDS = 7 * 24 * 3600  # 1 Woche als absolute Obergrenze

def _safe_seconds(x: Any) -> Optional[float]:
    """Normalize predictor output to a sane positive float (seconds)."""
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v) or v < 0:
        return None
    # Cap at maximum to prevent overflow in CP-SAT solver (int64 limits)
    return min(v, _MAX_SECONDS)


def _get_lstm_predictor() -> Any:
    """Get or create LSTM predictor for resource optimization."""
    global _lstm_predictor_cache
    if _lstm_predictor_cache is None:
        try:
            from processing_time_prediction.ProcessingTimePredictionClass import (
                ProcessingTimePredictionClass
            )
            # Try LSTM first, fallback to standard ML model if not available
            try:
                _lstm_predictor_cache = ProcessingTimePredictionClass(
                    method="probabilistic_ml",
                    model_path="models/processing_time_model_lstm"
                )
                print("  Initialized LSTM predictor for resource optimization")
            except (FileNotFoundError, Exception) as e:
                print(f"  WARNING: LSTM model not found, using standard ML model: {e}")
                # Fallback to standard ML model
                _lstm_predictor_cache = ProcessingTimePredictionClass(
                    method="ml",
                    model_path="models/processing_time_model"
                )
                print("  Initialized ML predictor (fallback) for resource optimization")
        except Exception as e:
            print(f"  WARNING: Failed to initialize predictor: {e}")
            return None
    return _lstm_predictor_cache


def predict_processing_seconds(
    predictor: Any,
    *,
    prev_activity: str,
    curr_activity: str,
    current_time: datetime,
    case: Any,
    candidate_resource: str,
) -> Optional[float]:
    """Predict processing time in seconds for assigning `curr_activity` to `candidate_resource`."""
    if predictor is None:
        print("  WARNING: predictor is None!")
        return None

    ctx = build_processing_time_context(case, current_time, candidate_resource)

    # Always use LSTM for resource optimization
    lstm_predictor = _get_lstm_predictor()
    if lstm_predictor is None:
        print("  WARNING: LSTM predictor not available!")
        return None

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            seconds = lstm_predictor.predict(
                prev_activity=prev_activity,
                prev_lifecycle="complete",
                curr_activity=curr_activity,
                curr_lifecycle="complete",
                context=ctx,
            )
        safe_sec = _safe_seconds(seconds)
        return safe_sec
    except Exception as e:
        print(f"  ERROR in LSTM predictor.predict: {e}")
        import traceback
        traceback.print_exc()
        return None

def handle_batch_scheduling_optimization(
    *,
    cfg: SelectionConfig,
    activity: str,
    timestamp: datetime,
    case: Any,
    waiting_tasks: List[Any],
    processing_time_predictor: Any,
    allocator: Any,
    resource_pool: Any = None,
) -> Tuple[Optional[Dict[str, Optional[str]]], Dict[str, Any]]:
    """Handle batch scheduling optimization using PMSP approach."""
    print("In handle_batch_scheduling_optimization")
    
    if not waiting_tasks:
        return None, {"decision": "no_waiting_tasks"}

    # Dictionary to store authorized resources for each waiting task
    authorized_resources_by_waiting_task: Dict[str, List[str]] = {}
    task_ids: List[str] = []

    # Collect all authorized resources for each waiting task
    for waiting_work in waiting_tasks:
        task_id = f"{waiting_work.case_id}_{waiting_work.allocation_activity}"
        task_ids.append(task_id)
        allocation_activity = waiting_work.allocation_activity
        
        try:
            eligible_resources = allocator.permissions.get_eligible_resources(
                allocation_activity, timestamp=timestamp, case_type=waiting_work.case_state.case_type
            )
        except TypeError:
            eligible_resources = allocator.permissions.get_eligible_resources(allocation_activity)

        if not eligible_resources:
            continue
        authorized_resources_by_waiting_task[task_id] = eligible_resources

    # Pretty-print authorized resources ## DELETE ME AFTER DEBUGGING
    # print(f"\n=== Authorized Resources per Task ({len(authorized_resources_by_waiting_task)} tasks) ===")
    # for tid, resources in authorized_resources_by_waiting_task.items():
    #     print(f"  {tid:<40s} → [{', '.join(resources)}]")
    # print("=" * 55)

    # Für jede autorisierte Ressource: busy_until vom resource_pool holen
    resource_busy_until: Dict[str, Optional[datetime]] = {}
    all_authorized_resources = set()
    for resources in authorized_resources_by_waiting_task.values():
        all_authorized_resources.update(resources)
    
    for r in all_authorized_resources:
        if resource_pool is not None:
            resource_busy_until[r] = resource_pool.get_busy_until(r)
        else:
            resource_busy_until[r] = None

    if not authorized_resources_by_waiting_task:
        return None, {"decision": "no_eligible_resources"}

    # Pretty-print busy-until times ## DELETE ME AFTER DEBUGGING
    print(f"\n=== Resource Busy-Until ({len(resource_busy_until)} resources) ===")
    for r, until in sorted(resource_busy_until.items()):
        status = f"until {until:%H:%M:%S}" if until else "free"
        print(f"  {r:<30s} | {status}")
    print("=" * 45)

    # Build R_P candidates first: authorized resources that are available in this time slot.
    # If R_P is empty, skip PMSP processing-time prediction entirely.
    authorized_and_timeslotoperating_resources_per_task: Dict[str, List[str]] = {}
    for task_id in task_ids:
        # Falls keine autorisierte Ressource für diesen task existiert skip
        if task_id not in authorized_resources_by_waiting_task:
            continue

        # Nimm alle autorisierten Ressourcen für diesen task
        authorized_resources = authorized_resources_by_waiting_task[task_id]
        # Filter to only resources that are available in this time slot
        available_resources = [
            res for res in authorized_resources
            if allocator.availability.is_available(res, timestamp)
        ]
        authorized_and_timeslotoperating_resources_per_task[task_id] = available_resources

    rp_candidates = {
        r for resources in authorized_and_timeslotoperating_resources_per_task.values() for r in resources
    }
    if not rp_candidates:
        print("R_P is empty for current timeslot -> skipping PMSP processing-time prediction.")
        assignment_all_dummy = {task_id: None for task_id in task_ids}
        debug = {
            "status": 0,
            "solver": "empty_timeslot",
            "decision": "no_timeslot_available_resources",
            "nT": len(task_ids),
            "nR_P": 0,
            "costs_per_task_resource": {},
        }
        return assignment_all_dummy, debug

    # Calculate PMSP parameters
    dummy_costs, predicted_remaining_times, costs_authorized_resource_task_assignment = calculate_pmsp_parameters(
        delta=cfg.dummy_delta,
        authorized_resources_per_task=authorized_resources_by_waiting_task,
        waiting_tasks=waiting_tasks,
        timestamp=timestamp,
        processing_time_predictor=processing_time_predictor,
        resource_busy_until=resource_busy_until,
        allocator=allocator,
        prediction_batch_size=cfg.prediction_batch_size,
    )

    costs_authorized_and_timeslotoperating_resources_per_task: Dict[str, Dict[str, int]] = {}
    for task_id in task_ids:
        available_resources = authorized_and_timeslotoperating_resources_per_task.get(task_id, [])
        if task_id in costs_authorized_resource_task_assignment:
            costs_authorized_and_timeslotoperating_resources_per_task[task_id] = {
                res: costs_authorized_resource_task_assignment[task_id][res]
                for res in available_resources
                if res in costs_authorized_resource_task_assignment[task_id]
            }

    # Solve PMSP
    assignment, debug = solve_pmsp_ilp(
        delta=cfg.dummy_delta,
        tasks=task_ids,
        authorized_resources_by_waiting_task=authorized_resources_by_waiting_task,
        authorized_and_timeSlotOperating_resources_by_task=authorized_and_timeslotoperating_resources_per_task,
        costs_authorized_and_timeslotoperating_resources_per_task=costs_authorized_and_timeslotoperating_resources_per_task,
        dummy_costs=dummy_costs,
        predicted_remaining_times=predicted_remaining_times,
        costs_authorized_resource_task_assignment=costs_authorized_resource_task_assignment,
        solver_time_limit_seconds=cfg.pmsp_solver_time_limit_seconds,
    )

    # Kosten mitgeben, damit die Worklist nach SPT sortiert werden kann
    debug["costs_per_task_resource"] = costs_authorized_and_timeslotoperating_resources_per_task
    return assignment, debug

def calculate_pmsp_parameters(
    *,
    delta: float,
    authorized_resources_per_task: Dict[str, List[str]],
    waiting_tasks: List[Any],
    timestamp: datetime,
    processing_time_predictor: Any,
    resource_busy_until: Optional[Dict[str, Optional[datetime]]] = None,
    allocator: Any = None,
    prediction_batch_size: int = 0,
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    Calculate PMSP parameters according to Eq. 4b-4d.

    Prediction budget (per task)
    ----------------------------
    Instead of predicting c(t,r) for *every* authorized resource, we first
    predict for authorized resources that are **available** right now
    (``allocator.availability.is_available(res, timestamp)``).  Then, if the
    per-task prediction budget (``prediction_batch_size``) has not been
    exhausted, we predict for additional *unavailable* authorized resources
    until the budget is reached.  ``prediction_batch_size == 0`` means
    unlimited (old behaviour).  Dummy costs are always calculated for every
    task based on whatever predictions were made.

    Returns:
        (dummy_costs, predicted_remaining_times, costs_authorized_resource_task_assignment)
    """
    #DELETE ME AFTER DEBUGGING
    print("\n╔══════════════════════════════════════════╗")
    print("║   Calculating PMSP Parameters (Eq 4b-4d)  ║")
    print("Timestamp: ", timestamp)
    print("╚══════════════════════════════════════════╝")
    dummy_costs: Dict[str, int] = {}
    predicted_remaining_times: Dict[str, float] = {}
    costs_authorized_resource_task_assignment: Dict[str, Dict[str, int]] = {}
    
    # Create a mapping from task_id to waiting_work object
    task_id_to_waiting_work: Dict[str, Any] = {}
    for waiting_work in waiting_tasks:
        task_id = f"{waiting_work.case_id}_{waiting_work.allocation_activity}"
        task_id_to_waiting_work[task_id] = waiting_work

    # Per-task batch limit: max predictions per task_id
    batch_limit_per_task = prediction_batch_size if prediction_batch_size > 0 else float('inf')
    
    # Calculate processing times for each task-resource combination (c(t,r))
    for task_id, authorized_resources in authorized_resources_per_task.items():
        if task_id not in task_id_to_waiting_work:
            continue

        print("For task ", task_id, " we have the following authorized resources: ", authorized_resources)
        
        waiting_work = task_id_to_waiting_work[task_id]
        allocation_activity = waiting_work.allocation_activity
        case_state = waiting_work.case_state
        
        # Get previous activity from case history
        hist = getattr(case_state, "activity_history", []) or []
        prev_activity = hist[-1] if hist else "START"
        
        costs_authorized_resource_task_assignment[task_id] = {}
        task_costs = []
        task_prediction_count = 0

        # ── Split authorized resources into available / unavailable ──
        if allocator is not None:
            timeslotoperating_authorized_resources = [
                r for r in authorized_resources
                if allocator.availability.is_available(r, timestamp)
            ]
            nottimeslotoperating_authorized_resources = [
                r for r in authorized_resources
                if not allocator.availability.is_available(r, timestamp)
            ]
        else:
            # Fallback: treat all as available (old behaviour)
            timeslotoperating_authorized_resources = list(authorized_resources)
            nottimeslotoperating_authorized_resources = []

        # Predict available (timeslotoperating) resources first, then fill with unavailable
        ordered_resources = timeslotoperating_authorized_resources + nottimeslotoperating_authorized_resources
        
        # Calculate c(t,r) for each resource (respecting per-task batch budget)
        for resource in ordered_resources:
            if task_prediction_count >= batch_limit_per_task:
                break

            sec = predict_processing_seconds(
                processing_time_predictor,
                prev_activity=prev_activity,
                curr_activity=allocation_activity,
                current_time=timestamp,
                case=case_state,
                candidate_resource=resource,
            )
            task_prediction_count += 1

            if sec is not None:
                # Convert to integer (milliseconds for precision)
                print("Processing time for task ", task_id, " and resource ", resource, " is ", sec)
                cost_ms = int(sec * 1000)
                costs_authorized_resource_task_assignment[task_id][resource] = cost_ms
                task_costs.append(sec)  # Keep in seconds for average calculation
        
        # Dummy costs müssen für jeden Task berechnet werden
        # c_d(t) = δ * (1/|R_predicted(t)|) * sum(c(t,r) for r in R_predicted(t))
        n_predicted = len(task_costs)
        if task_costs and n_predicted > 0:
            avg_cost = sum(task_costs) / n_predicted
            dummy_costs[task_id] = int(delta * avg_cost * 1000)  # Convert to milliseconds
        else:
            # Large penalty if no predictions available
            dummy_costs[task_id] = 10**9

    print(f"  [Batch] {len(authorized_resources_per_task)} tasks "
          f"(prediction limit per task: {prediction_batch_size or 'unlimited'})")
    
    # Nicht alle Ressourcen sind authorisiert für einen Task zu arbeiten. D.H. 
    # Calculate remaining processing times for working resources (c_r(r))
    # Collect all unique resources from authorized resources
    all_resources = set()
    for resources in authorized_resources_per_task.values():
        all_resources.update(resources)
    
    for r in all_resources:
        if resource_busy_until and r in resource_busy_until:
            busy_until = resource_busy_until[r]
            if busy_until is not None and busy_until > timestamp:
                # Resource is busy, calculate remaining time in seconds
                remaining_seconds = (busy_until - timestamp).total_seconds()
                predicted_remaining_times[r] = float(remaining_seconds)
            else:
                # Resource is not busy
                predicted_remaining_times[r] = 0.0
        else:
            # No busy information available, assume not busy
            predicted_remaining_times[r] = 0.0
    
    # Pretty-print predicted costs for timeslotoperating resources per task
    # We need to identify which resources are timeslotoperating
    if allocator is not None:
        print(f"\n--- Predicted Costs c(t,r) for timeslot-available resources ---")
        for task_id, res_costs in costs_authorized_resource_task_assignment.items():
            avail_costs = {
                r: c for r, c in res_costs.items()
                if allocator.availability.is_available(r, timestamp)
            }
            if avail_costs:
                print(f"  Task {task_id}:")
                for r, cost in avail_costs.items():
                    print(f"    {r:<28s} → {cost:>10,} ms  ({cost/1000:.1f}s)")
            else:
                print(f"  Task {task_id}:  (no timeslot-available resource)")
        print("─" * 60)

    # Pretty-print PMSP parameters ## DELETE ME AFTER DEBUGGING
    print(f"\n--- Dummy Costs c_d(t)  ({len(dummy_costs)} tasks) ---")
    for tid, cost in dummy_costs.items():
        print(f"  {tid:<40s} | c_d = {cost:>10,} ms  ({cost/1000:.1f}s)")

    print(f"\n--- Remaining Times c_r(r)  ({len(predicted_remaining_times)} resources) ---")
    for r, rem in sorted(predicted_remaining_times.items()):
        status = f"{rem:>8.1f}s remaining" if rem > 0 else "free"
        print(f"  {r:<30s} | {status}")

    # print(f"\n--- Processing Costs c(t,r)  ---")
    # for tid, res_costs in costs_authorized_resource_task_assignment.items():
    #     print(f"  Task {tid}:")
    #     for r, cost in res_costs.items():
    #         print(f"    {r:<28s} → {cost:>10,} ms  ({cost/1000:.1f}s)")
    # print("─" * 50)
    return dummy_costs, predicted_remaining_times, costs_authorized_resource_task_assignment


# Passt, soweit ich das nachvollziehen kann.
def solve_pmsp_ilp(
    *,
    delta: float,
    tasks: List[str],
    authorized_resources_by_waiting_task: Dict[str, List[str]],
    authorized_and_timeSlotOperating_resources_by_task: Dict[str, List[str]],
    costs_authorized_and_timeslotoperating_resources_per_task: Dict[str, Dict[str, int]],
    dummy_costs: Dict[str, int],
    predicted_remaining_times: Dict[str, float],
    costs_authorized_resource_task_assignment: Dict[str, Dict[str, int]],
    solver_time_limit_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """
    Solve PMSP adaptation according to Eq. 4.
    
    Objective: min C = k_m + k_f + sum(y_t * c_d(t))
    where:
    - k_m = max(c_r(r) + sum(x_tr * c(t,r)) for r in R_P)
    - k_f = (1/|R_P|) * sum(k_m - (c_r(r) + sum(x_tr * c(t,r))) for r in R_P)
    
    Primary solver: OR-Tools CP-SAT (time-limited, default 2s).
    Backup solver: SciPy Jonker-Volgenant assignment adaptation.
    """
    print("In solve_pmsp_ilp")
    T = list(tasks)
    # R_P is the set of all resources that are available and authorized for at least one task
    R_P_set = set()
    for resources in authorized_and_timeSlotOperating_resources_by_task.values():
        R_P_set.update(resources)
    R_P = list(R_P_set)
    nT = len(T)
    nR_P = len(R_P)


    print("T: ", T)
    print("R_P: ", R_P)
    print("nT: ", nT)
    print("nR_P: ", nR_P)
    
    if nT == 0 or nR_P == 0:
        return {t: None for t in T}, {"status": 0, "nT": nT, "nR_P": nR_P, "solver": "empty"}

    # Prepare backup assignment solver (for fallback)
    # Convert costs to format needed for assignment solver
    costs_for_jv: Dict[Tuple[str, str], int] = {}
    dummy_cost_for_jv: Dict[str, int] = {}
    remaining_for_jv: Dict[str, int] = {}
    
    for task_id in T:
        if task_id in costs_authorized_and_timeslotoperating_resources_per_task:
            for r, cost in costs_authorized_and_timeslotoperating_resources_per_task[task_id].items():
                costs_for_jv[(task_id, r)] = cost
        if task_id in dummy_costs:
            dummy_cost_for_jv[task_id] = dummy_costs[task_id]
        else:
            dummy_cost_for_jv[task_id] = 10**9
    
    for r in R_P:
        remaining_for_jv[r] = int(predicted_remaining_times.get(r, 0.0) * 1000)  # Convert to milliseconds
    
    # TODO: Backup assignment solution
    assignment_jv, debug_jv = solve_assignment_jv(
        tasks=T,
        resources=R_P,
        costs=costs_for_jv,
        dummy_cost=dummy_cost_for_jv,
        remaining=remaining_for_jv,
    )

    effective_time_limit = 2.0 if solver_time_limit_seconds is None else float(solver_time_limit_seconds)
    effective_time_limit = max(0.01, effective_time_limit)

    try:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        x: Dict[Tuple[str, str], Any] = {}
        y: Dict[str, Any] = {}

        # Create x only for task-resource pairs with an available c(t,r) prediction.
        # This prevents zero-cost "free edges" when prediction budget is limited
        # or predictor outputs are missing.
        for task_id in T:
            task_costs = costs_authorized_and_timeslotoperating_resources_per_task.get(task_id, {})
            if task_id in authorized_and_timeSlotOperating_resources_by_task:
                for r in authorized_and_timeSlotOperating_resources_by_task[task_id]:
                    if r in R_P and r in task_costs:
                        x[(task_id, r)] = model.NewBoolVar(f"x__{task_id}__{r}")
        
        # Create decision variables y_t for dummy allocation
        for t in T:
            y[t] = model.NewBoolVar(f"y__{t}")

        # Calculate upper bound for k_m (makespan)
        # 
        # Warum brauchen wir eine obere Schranke?
        # Der CP-SAT Solver benötigt für jede Integer-Variable einen Wertebereich.
        # k_m ist definiert als: k_m = max(c_r(r) + sum(x_tr * c(t,r)) for r in R_P)
        # 
        # Im schlimmsten Fall könnte eine einzelne Ressource r alle Tasks bearbeiten:
        # - Die Ressource hat bereits c_r(r) verbleibende Zeit (wenn sie gerade beschäftigt ist)
        # - Plus die Summe aller Task-Kosten, wenn alle Tasks dieser Ressource zugewiesen werden
        # 
        # Daher: ub_km = max_remaining + max_task_cost * nT
        # 
        # max_remaining: Die maximale verbleibende Zeit einer Ressource (c_r(r))
        max_remaining = max((int(predicted_remaining_times.get(r, 0.0) * 1000) for r in R_P), default=0)
        
        # max_task_cost: Die maximale Bearbeitungszeit für eine Task-Ressource-Kombination
        # (durchsuche alle Tasks und deren Ressourcen-Kosten)
        max_task_cost = max(
            (cost for task_costs in costs_authorized_and_timeslotoperating_resources_per_task.values() 
             for cost in task_costs.values()),
            default=0
        )
        
        # Obere Schranke: Im Worst-Case bearbeitet eine Ressource alle Tasks
        ub_km = max_remaining + max_task_cost * nT
        
        # Erstelle k_m als Integer-Variable mit dieser oberen Schranke
        k_m = model.NewIntVar(0, ub_km, "k_m")

        # Constraint (4e): Each task is allocated to exactly one resource or dummy
        for t in T:
            x_t = [x[(t, r)] for r in R_P if (t, r) in x]
            model.Add(sum(x_t) + y[t] == 1)

        # Constraint (4f): x_tr = 0 for r not in R_auth(t) (already enforced by only creating x for authorized resources)
        # Additional constraint: x_tr = 0 for r not in R_P (already enforced)
        
        # k_m wird in der Zeilfunktion minimiert, muss aber gleichzeitig immer größer sein als c_r + sum(x_tr * c(t,r)) für alle r in R_P, daher Maximalwert von c_r + sum(x_tr * c(t,r)) für alle r in R_P
        # Makespan constraints: k_m >= c_r(r) + sum(x_tr * c(t,r)) for all r in R_P
        for r in R_P:
            c_r = int(predicted_remaining_times.get(r, 0.0) * 1000)  # Convert to milliseconds
            terms = []
            for t in T:
                if (t, r) in x:
                    # Get cost from the appropriate dict
                    if t in costs_authorized_and_timeslotoperating_resources_per_task:
                        if r in costs_authorized_and_timeslotoperating_resources_per_task[t]:
                            cost = costs_authorized_and_timeslotoperating_resources_per_task[t][r]
                            terms.append(cost * x[(t, r)])
            # Enforce k_m >= c_r(r) even if no task can be assigned to r.
            model.Add(c_r + sum(terms) <= k_m)

        # Calculate k_f = (1/|R_P|) * sum(k_m - (c_r(r) + sum(x_tr * c(t,r))) for r in R_P)
        # We need to model this in the objective. Since k_f = k_m - (1/|R_P|) * sum(c_r(r) + sum(x_tr * c(t,r))),
        # the objective becomes: k_m + k_f + sum(y_t * c_d(t))
        # = k_m + k_m - (1/|R_P|) * sum(c_r(r) + sum(x_tr * c(t,r))) + sum(y_t * c_d(t))
        # = 2*k_m - (1/|R_P|) * sum(c_r(r) + sum(x_tr * c(t,r))) + sum(y_t * c_d(t))
        
        # For CP-SAT, we'll use integer arithmetic. Scale by |R_P| to avoid fractions:
        # Objective = |R_P| * (2*k_m) - sum(c_r(r) + sum(x_tr * c(t,r))) + |R_P| * sum(y_t * c_d(t))
        
        # Calculate sum of c_r(r) + sum(x_tr * c(t,r)) for all r in R_P
        # This is used to calculate k_f
        sum_workload_terms = []
        sum_c_r_constant = 0  # Sum of c_r(r) for all r in R_P (constant part)
        
        for r in R_P:
            c_r = int(predicted_remaining_times.get(r, 0.0) * 1000)
            sum_c_r_constant += c_r  # Add constant c_r(r) for each resource
            
            # Add variable part: sum(x_tr * c(t,r)) for this resource
            for t in T:
                if (t, r) in x:
                    if t in costs_authorized_and_timeslotoperating_resources_per_task:
                        if r in costs_authorized_and_timeslotoperating_resources_per_task[t]:
                            cost = costs_authorized_and_timeslotoperating_resources_per_task[t][r]
                            sum_workload_terms.append(cost * x[(t, r)])
        
        # Create a variable for the sum (constant + variable parts)
        max_sum_workload = max_remaining * nR_P + max_task_cost * nT * nR_P
        sum_workload_var = model.NewIntVar(0, max_sum_workload, "sum_workload")
        model.Add(sum_workload_var == sum_c_r_constant + sum(sum_workload_terms))
        
        # Objective: |R_P| * 2 * k_m - sum_workload_var + |R_P| * sum(y_t * c_d(t))
        dummy_sum_terms = []
        for t in T:
            if t in dummy_costs:
                dummy_sum_terms.append(dummy_costs[t] * y[t])
        
        if nR_P > 0:
            # Scale objective by nR_P to match the formulation
            objective = nR_P * 2 * k_m - sum_workload_var + nR_P * sum(dummy_sum_terms)
        else:
            objective = sum(dummy_sum_terms)
        
        model.Minimize(objective)

        # --- Debug: Modellgröße & Schranken ausgeben ---
        print(f"\n  [CP-SAT] Variables: {model.Proto().variables.__len__()}, "
              f"Constraints: {len(model.Proto().constraints)}")
        print(f"  [CP-SAT] ub_km={ub_km}, max_remaining={max_remaining}, "
              f"max_task_cost={max_task_cost}, max_sum_workload={max_sum_workload}")

        # Validierung: Prüfe ob das Modell konsistent ist
        validation = model.Validate()
        if validation:
            print(f"  [CP-SAT] Modell-Validierung fehlgeschlagen: {validation}")
            raise ValueError(f"CP-SAT model validation failed: {validation}")

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = effective_time_limit

        # --- KEINE Parallelisierung im Solver: exakt 1 Worker ---
        solver.parameters.num_search_workers = 1

        print("  [CP-SAT] Starting solver...")
        import sys; sys.stdout.flush()
        status = solver.Solve(model)
        print(f"  [CP-SAT] Solver finished.")
        status_name = solver.StatusName(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignment_cp: Dict[str, Optional[str]] = {t: None for t in T}
            for t in T:
                if solver.Value(y[t]) >= 1:
                    assignment_cp[t] = None
                    continue
                chosen = None
                for r in R_P:
                    if (t, r) in x and solver.Value(x[(t, r)]) >= 1:
                        chosen = r
                        break
                assignment_cp[t] = chosen

            # Print assignment results
            print(f"\n=== PMSP Assignment ({status_name}) ===")
            for t, r in assignment_cp.items():
                if r is not None:
                    cost = costs_authorized_and_timeslotoperating_resources_per_task.get(t, {}).get(r, "?")
                    print(f"  Task {t}  →  Resource {r}  (cost: {cost} ms)")
                else:
                    dc = dummy_costs.get(t, "?")
                    print(f"  Task {t}  →  DUMMY (postponed, dummy_cost: {dc} ms)")
            print(f"{'=' * 40}\n")

            # Calculate actual objective value
            k_m_val = solver.Value(k_m)
            sum_workload_val = solver.Value(sum_workload_var)
            
            # Calculate k_f = (1/|R_P|) * sum(k_m - (c_r(r) + sum(x_tr * c(t,r))) for r in R_P)
            # = k_m - (1/|R_P|) * sum(c_r(r) + sum(x_tr * c(t,r)))
            # = k_m - (1/|R_P|) * sum_workload_val
            if nR_P > 0:
                k_f_val = k_m_val - (sum_workload_val / nR_P)
            else:
                k_f_val = 0.0
            
            dummy_sum = sum(int(dummy_costs.get(t, 0)) * solver.Value(y[t]) for t in T)
            cp_obj = int(round(k_m_val + k_f_val + dummy_sum))
            jv_obj = int(debug_jv.get("objective", 10**18))

            if status != cp_model.OPTIMAL and jv_obj <= cp_obj:
                dbg = dict(debug_jv)
                dbg.update({"status": int(status), "status_name": status_name, "fallback_from": "cp_sat"})
                return assignment_jv, dbg

            # Das Assignment sorgt für eine Oprimisierung der Ressourcen 
            # Allerdings wird die Optimierung aufgerufen wenn eine neue Aktivität eine Ressource benögt. 
            # TODO: Zuteilung der Assignemnt zu Worklist
            # TODO: Scheduling der assignten Ressourcen
            return assignment_cp, {
                "status": int(status),
                "status_name": status_name,
                "nT": nT,
                "nR_P": nR_P,
                "solver": "cp_sat",
                "objective": cp_obj,
                "k_m": int(k_m_val),
                "k_f": int(round(k_f_val)),
            }

        dbg = dict(debug_jv)
        dbg.update({"status": int(status), "status_name": status_name, "fallback_from": "cp_sat"})
        return assignment_jv, dbg

    except Exception as e:
        import traceback
        traceback.print_exc()
        dbg = dict(debug_jv)
        dbg.update({"fallback_from": "cp_sat_import_or_runtime_error", "error": str(e)})
        return assignment_jv, dbg
