"""
PMSP-based resource optimization for in-simulation resource selection.

Solves a Parallel Machine Scheduling Problem (PMSP) adaptation to assign
waiting tasks to available resources, minimizing makespan + fairness + dummy costs.

Selection modes:
- "pmsp": batch optimization via CP-SAT (with JV fallback)

Dummy rule:
  c_d(t) = delta * avg(c(t,r) for r in R_predicted(t))
  If the best available resource is worse than the dummy cost, postpone the task.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# LSTM predictor cache for resource optimization
_lstm_predictor_cache: Optional[Any] = None

_MAX_SECONDS = 7 * 24 * 3600  # 1 week upper bound

# Tie-breaking constant (ms) added to the dummy/unassigned cost so that a real
# resource is always preferred when its cost exactly equals the dummy threshold.
# With delta=1 and a single eligible resource the dummy cost equals that
# resource's cost exactly, which previously caused systematic "postpone" decisions.
_DUMMY_TIEBREAK_MS = 1


@dataclass(frozen=True)
class SelectionConfig:
    mode: str = "pmsp"
    dummy_delta: float = 1.0
    avg_sample_size: int = 25
    pmsp_solver_time_limit_seconds: Optional[float] = None
    prediction_batch_size: int = 0  # Max predictions per task (0 = unlimited)
    optimization_batch_size: int = 0  # Min waiting tasks to trigger optimization (0 = always optimize)


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

    n_cols = nR + nT
    big_m = float(10**9)
    C = np.full((nT, n_cols), big_m, dtype=np.float64)

    for i, t in enumerate(T):
        for j, r in enumerate(R):
            if (t, r) in costs:
                C[i, j] = float(remaining.get(r, 0) + costs[(t, r)])
        # +_DUMMY_TIEBREAK_MS: real resources win ties against the dummy column
        C[i, nR + i] = float(dummy_cost.get(t, 10**9)) + _DUMMY_TIEBREAK_MS

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


def _safe_seconds(x: Any) -> Optional[float]:
    """Normalize predictor output to a sane positive float (seconds)."""
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v) or v < 0:
        return None
    return min(v, _MAX_SECONDS)


def _get_lstm_predictor() -> Any:
    """Get or create LSTM predictor for resource optimization."""
    global _lstm_predictor_cache
    if _lstm_predictor_cache is None:
        try:
            from processing_time_prediction.ProcessingTimePredictionClass import (
                ProcessingTimePredictionClass
            )
            try:
                _lstm_predictor_cache = ProcessingTimePredictionClass(
                    method="probabilistic_ml",
                    model_path="models/processing_time_model_lstm"
                )
                logger.info("Initialized LSTM predictor for resource optimization")
            except (FileNotFoundError, Exception) as e:
                logger.warning("LSTM model not found, using standard ML model: %s", e)
                _lstm_predictor_cache = ProcessingTimePredictionClass(
                    method="ml",
                    model_path="models/processing_time_model"
                )
                logger.info("Initialized ML predictor (fallback) for resource optimization")
        except Exception as e:
            logger.warning("Failed to initialize predictor: %s", e)
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
        return None

    ctx = build_processing_time_context(case, current_time, candidate_resource)

    lstm_predictor = _get_lstm_predictor()
    if lstm_predictor is None:
        return None

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
        safe_seconds = _safe_seconds(seconds)
        if safe_seconds == 0.0:
            logger.debug("LSTM predictor returned 0.00s for %s -> %s (raw: %s)", 
                        curr_activity, candidate_resource, seconds)
        return safe_seconds
    except Exception as e:
        logger.error("LSTM predictor.predict failed: %s", e, exc_info=True)
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
    if not waiting_tasks:
        return None, {"decision": "no_waiting_tasks"}

    # logger.info("PMSP [Optimization]: Starting optimization for %d waiting tasks", len(waiting_tasks))

    authorized_resources_by_waiting_task: Dict[str, List[str]] = {}
    task_ids: List[str] = []
    task_details: Dict[str, Dict[str, Any]] = {}  # For logging

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
            # logger.debug("PMSP [Optimization]: Task %s (activity: %s, case: %s) has no eligible resources", 
            #             task_id, allocation_activity, waiting_work.case_id)
            continue
        authorized_resources_by_waiting_task[task_id] = eligible_resources
        task_details[task_id] = {
            "activity": allocation_activity,
            "case_id": waiting_work.case_id,
            "eligible_resources": len(eligible_resources)
        }

    # Get busy_until for each authorized resource from the resource pool
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
        # logger.warning("PMSP [Optimization]: No tasks with eligible resources, aborting")
        return None, {"decision": "no_eligible_resources"}

    # logger.info("PMSP [Optimization]: Found %d tasks with eligible resources", len(authorized_resources_by_waiting_task))
    # for task_id, details in task_details.items():
    #     logger.info("  Task %s: Activity '%s' (case %s) - %d eligible resources", 
    #                task_id, details["activity"], details["case_id"], details["eligible_resources"])

    # Build R_P candidates: authorized resources available in this time slot
    authorized_and_timeslotoperating_resources_per_task: Dict[str, List[str]] = {}
    for task_id in task_ids:
        if task_id not in authorized_resources_by_waiting_task:
            continue
        authorized_resources = authorized_resources_by_waiting_task[task_id]
        available_resources = [
            res for res in authorized_resources
            if allocator.availability.is_available(res, timestamp)
        ]
        authorized_and_timeslotoperating_resources_per_task[task_id] = available_resources

    rp_candidates = {
        r for resources in authorized_and_timeslotoperating_resources_per_task.values() for r in resources
    }
    if not rp_candidates:
        # logger.warning("PMSP [Optimization]: R_P is empty for current timeslot -> skipping PMSP")
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

    # logger.info("PMSP [Optimization]: Calculating parameters for %d tasks with %d available resources (R_P)", 
    #             len(authorized_resources_by_waiting_task), len(rp_candidates))
    # logger.info("PMSP [Optimization]: Available resources: %s", ", ".join(sorted(rp_candidates)))
    # Calculate PMSP parameters
    dummy_costs, predicted_remaining_times, costs_authorized_resource_task_assignment, raw_processing_times = calculate_pmsp_parameters(
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

    # logger.info("PMSP [Optimization]: Solving optimization problem for %d tasks and %d resources", 
    #             len(task_ids), len(rp_candidates))
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

    debug["costs_per_task_resource"] = costs_authorized_and_timeslotoperating_resources_per_task
    debug["raw_processing_times"] = raw_processing_times  # task_id -> resource -> seconds (for SPT ordering)
    if assignment:
        assigned_count = sum(1 for v in assignment.values() if v is not None)
        # logger.info("PMSP [Optimization]: Optimization finished - %d/%d tasks assigned to resources", 
        #            assigned_count, len(assignment))
        
        # Log detailed assignment results
        # logger.info("PMSP [Optimization]: Detailed assignment results:")
        # for task_id, resource in sorted(assignment.items()):
        #     task_info = task_details.get(task_id, {})
        #     activity = task_info.get("activity", "unknown")
        #     case_id = task_info.get("case_id", "unknown")
        #     if resource is not None:
        #         logger.info("  Task '%s' (case %s, activity '%s') -> Resource %s", 
        #                    task_id, case_id, activity, resource)
        #     else:
        #         logger.info("  Task '%s' (case %s, activity '%s') -> DUMMY (unassigned)", 
        #                    task_id, case_id, activity)
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

    Predicts c(t,r) for available resources first, then fills with unavailable
    ones up to ``prediction_batch_size`` per task (0 = unlimited).

    Returns:
        (dummy_costs, predicted_remaining_times, costs_authorized_resource_task_assignment, raw_processing_times)
    """
    dummy_costs: Dict[str, int] = {}
    predicted_remaining_times: Dict[str, float] = {}
    costs_authorized_resource_task_assignment: Dict[str, Dict[str, int]] = {}
    raw_processing_times: Dict[str, Dict[str, float]] = {}  # task_id -> resource -> seconds (for SPT ordering)

    task_id_to_waiting_work: Dict[str, Any] = {}
    for waiting_work in waiting_tasks:
        task_id = f"{waiting_work.case_id}_{waiting_work.allocation_activity}"
        task_id_to_waiting_work[task_id] = waiting_work

    batch_limit_per_task = prediction_batch_size if prediction_batch_size > 0 else float('inf')

    total_tasks = len(authorized_resources_per_task)
    task_index = 0
    for task_id, authorized_resources in authorized_resources_per_task.items():
        task_index += 1
        # Log progress every 5 tasks or at milestones (10%, 25%, 50%, 75%, 100%)
        # if total_tasks > 10 and (task_index % 5 == 0 or 
        #                           task_index == 1 or 
        #                           task_index == max(1, int(total_tasks * 0.1)) or
        #                           task_index == max(1, int(total_tasks * 0.25)) or
        #                           task_index == max(1, int(total_tasks * 0.5)) or
        #                           task_index == max(1, int(total_tasks * 0.75)) or
        #                           task_index == total_tasks):
        #     logger.info("PMSP: Calculating parameters for task %d/%d (%.1f%%)", 
        #                task_index, total_tasks, 100.0 * task_index / total_tasks)
        if task_id not in task_id_to_waiting_work:
            continue

        waiting_work = task_id_to_waiting_work[task_id]
        allocation_activity = waiting_work.allocation_activity
        case_state = waiting_work.case_state
        case_id = waiting_work.case_id

        hist = getattr(case_state, "activity_history", []) or []
        prev_activity = hist[-1] if hist else "START"

        costs_authorized_resource_task_assignment[task_id] = {}
        raw_processing_times[task_id] = {}  # Store raw PTs for SPT ordering
        task_costs = []
        task_prediction_count = 0
        
        # Log task start
        # logger.info("PMSP [Parameters]: Calculating costs for task %s (activity: '%s', case: %s)", 
        #            task_id, allocation_activity, case_id)

        # Split authorized resources into available / unavailable
        if allocator is not None:
            timeslot_available = [
                r for r in authorized_resources
                if allocator.availability.is_available(r, timestamp)
            ]
            timeslot_unavailable = [
                r for r in authorized_resources
                if not allocator.availability.is_available(r, timestamp)
            ]
        else:
            timeslot_available = list(authorized_resources)
            timeslot_unavailable = []

        ordered_resources = timeslot_available + timeslot_unavailable

        # logger.info("PMSP [Parameters]:   Predicting processing times for %d resources:", len(ordered_resources))
        for resource in ordered_resources:
            if task_prediction_count >= batch_limit_per_task:
                # logger.info("PMSP [Parameters]:   (Stopped at batch limit: %d predictions)", batch_limit_per_task)
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
                # Store raw processing time (for SPT ordering in worklists)
                raw_processing_times[task_id][resource] = sec
                # Cost = processing_time
                cost_ms = int(sec * 1000)
                costs_authorized_resource_task_assignment[task_id][resource] = cost_ms
                task_costs.append(sec)  # store raw processing time for dummy-cost averaging
                
                # Log each prediction
                # logger.info("PMSP [Parameters]:     Resource %s: predicted_processing_time=%.2fs, cost=%d ms", 
                #            resource, sec, cost_ms)
            else:
                # logger.warning("PMSP [Parameters]:     Resource %s: prediction failed (returned None)", resource)
                pass

        # c_d(t) = delta * avg(c(t,r))
        if task_costs:
            avg_cost = sum(task_costs) / len(task_costs)
            dummy_cost_seconds = delta * avg_cost
            dummy_costs[task_id] = int(dummy_cost_seconds * 1000)
            
            # Log dummy cost calculation
            # logger.info("PMSP [Parameters]:   Dummy cost calculation for task %s:", task_id)
            # logger.info("PMSP [Parameters]:     - Number of predictions: %d", len(task_costs))
            # logger.info("PMSP [Parameters]:     - Average processing time: %.2f seconds", avg_cost)
            # logger.info("PMSP [Parameters]:     - Delta (multiplier): %.2f", delta)
            # logger.info("PMSP [Parameters]:     - Final dummy cost: %.2f seconds = %d ms", 
            #            dummy_cost_seconds, dummy_costs[task_id])
        else:
            dummy_costs[task_id] = 10**9
            # logger.warning("PMSP [Parameters]:   Dummy cost for task %s: %d ms (no valid predictions)", 
            #               task_id, dummy_costs[task_id])
            pass

    # logger.info(
    #     "PMSP [Parameters]: Calculated parameters for %d tasks (prediction limit per task: %s)",
    #     len(authorized_resources_per_task),
    #     prediction_batch_size or "unlimited",
    # )
    
    # Log cost matrix summary
    # logger.info("PMSP [Parameters]: Cost matrix summary:")
    # for task_id in sorted(authorized_resources_per_task.keys()):
    #     if task_id in costs_authorized_resource_task_assignment:
    #         costs = costs_authorized_resource_task_assignment[task_id]
    #         dummy_cost = dummy_costs.get(task_id, 0)
    #         logger.info("PMSP [Parameters]:   Task %s: %d resources with costs, dummy_cost=%d ms", 
    #                    task_id, len(costs), dummy_cost)
    #         # Show min/max costs
    #         if costs:
    #             min_cost = min(costs.values())
    #             max_cost = max(costs.values())
    #             logger.info("PMSP [Parameters]:     Cost range: %d ms - %d ms (min: %s, max: %s)", 
    #                        min_cost, max_cost,
    #                        [r for r, c in costs.items() if c == min_cost][0],
    #                        [r for r, c in costs.items() if c == max_cost][0])

    # Calculate remaining processing times for working resources (c_r(r))
    all_resources = set()
    for resources in authorized_resources_per_task.values():
        all_resources.update(resources)

    for r in all_resources:
        if resource_busy_until and r in resource_busy_until:
            busy_until = resource_busy_until[r]
            if busy_until is not None and busy_until > timestamp:
                predicted_remaining_times[r] = float((busy_until - timestamp).total_seconds())
            else:
                predicted_remaining_times[r] = 0.0
        else:
            predicted_remaining_times[r] = 0.0

    return dummy_costs, predicted_remaining_times, costs_authorized_resource_task_assignment, raw_processing_times


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
    T = list(tasks)
    R_P_set = set()
    for resources in authorized_and_timeSlotOperating_resources_by_task.values():
        R_P_set.update(resources)
    R_P = list(R_P_set)
    nT = len(T)
    nR_P = len(R_P)

    if nT == 0 or nR_P == 0:
        # logger.debug("PMSP: Empty problem (nT=%d, nR_P=%d), skipping", nT, nR_P)
        return {t: None for t in T}, {"status": 0, "nT": nT, "nR_P": nR_P, "solver": "empty"}

    # Fast-path: single-task case can be solved greedily without CP-SAT/JV.
    if nT == 1:
        t = T[0]
        task_costs = costs_authorized_and_timeslotoperating_resources_per_task.get(t, {})
        # No usable resource costs in this timeslot -> dummy.
        if not task_costs:
            # logger.info("PMSP [Solver]: Single-task fast-path: no available resources for task %s -> DUMMY", t)
            return {t: None}, {
                "status": 0,
                "nT": 1,
                "nR_P": nR_P,
                "solver": "greedy_single",
                "decision": "no_available_resource",
            }

        # Choose resource that minimizes remaining[r] + c(t,r).
        best_r = None
        best_total_ms = None
        for r, cost_ms in task_costs.items():
            rem_ms = int(predicted_remaining_times.get(r, 0.0) * 1000)
            total_ms = rem_ms + cost_ms
            if best_total_ms is None or total_ms < best_total_ms:
                best_total_ms = total_ms
                best_r = r

        dummy_ms = int(dummy_costs.get(t, 10**9))
        # Effective dummy objective includes the tie-break; we emulate the same
        # behavior by letting the real resource win on exact equality.
        if best_r is not None and best_total_ms is not None and best_total_ms <= dummy_ms:
            # logger.info(
            #     "PMSP [Solver]: Single-task fast-path: assigning task %s -> %s "
            #     "(best_total=%d ms, dummy=%d ms)",
            #     t,
            #     best_r,
            #     best_total_ms,
            #     dummy_ms,
            # )
            return {t: best_r}, {
                "status": 1,
                "nT": 1,
                "nR_P": nR_P,
                "solver": "greedy_single",
                "decision": "assign_real",
                "objective": best_total_ms,
            }

        # logger.info(
        #     "PMSP [Solver]: Single-task fast-path: assigning task %s -> DUMMY "
        #     "(best_total=%s ms, dummy=%d ms)",
        #     t,
        #     best_total_ms if best_total_ms is not None else -1,
        #     dummy_ms,
        # )
        return {t: None}, {
            "status": 1,
            "nT": 1,
            "nR_P": nR_P,
            "solver": "greedy_single",
            "decision": "assign_dummy",
            "objective": dummy_ms + _DUMMY_TIEBREAK_MS,
        }

    def _run_jv_fallback() -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
        """
        Lazily run JV fallback only when CP-SAT is unavailable or clearly worse.
        Avoids always paying O(N^3) JV cost when CP-SAT is fast and optimal.
        """
        costs_for_jv: Dict[Tuple[str, str], int] = {}
        dummy_cost_for_jv: Dict[str, int] = {}
        remaining_for_jv: Dict[str, int] = {}

        for task_id in T:
            if task_id in costs_authorized_and_timeslotoperating_resources_per_task:
                for r, cost in costs_authorized_and_timeslotoperating_resources_per_task[task_id].items():
                    costs_for_jv[(task_id, r)] = cost
            dummy_cost_for_jv[task_id] = dummy_costs.get(task_id, 10**9)

        for r in R_P:
            remaining_for_jv[r] = int(predicted_remaining_times.get(r, 0.0) * 1000)

        return solve_assignment_jv(
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

        for task_id in T:
            task_costs = costs_authorized_and_timeslotoperating_resources_per_task.get(task_id, {})
            if task_id in authorized_and_timeSlotOperating_resources_by_task:
                for r in authorized_and_timeSlotOperating_resources_by_task[task_id]:
                    if r in R_P and r in task_costs:
                        x[(task_id, r)] = model.NewBoolVar(f"x__{task_id}__{r}")

        for t in T:
            y[t] = model.NewBoolVar(f"y__{t}")

        # Upper bound for makespan
        max_remaining = max((int(predicted_remaining_times.get(r, 0.0) * 1000) for r in R_P), default=0)
        max_task_cost = max(
            (cost for task_costs in costs_authorized_and_timeslotoperating_resources_per_task.values()
             for cost in task_costs.values()),
            default=0
        )
        ub_km = max_remaining + max_task_cost * nT
        k_m = model.NewIntVar(0, ub_km, "k_m")

        # Constraint (4e): Each task -> exactly one resource or dummy
        for t in T:
            x_t = [x[(t, r)] for r in R_P if (t, r) in x]
            model.Add(sum(x_t) + y[t] == 1)

        # Makespan constraints: k_m >= c_r(r) + sum(x_tr * c(t,r)) for all r in R_P
        for r in R_P:
            c_r = int(predicted_remaining_times.get(r, 0.0) * 1000)
            terms = []
            for t in T:
                if (t, r) in x:
                    if t in costs_authorized_and_timeslotoperating_resources_per_task:
                        if r in costs_authorized_and_timeslotoperating_resources_per_task[t]:
                            cost = costs_authorized_and_timeslotoperating_resources_per_task[t][r]
                            terms.append(cost * x[(t, r)])
            model.Add(c_r + sum(terms) <= k_m)

        # Objective: scaled by |R_P| to avoid fractions
        # = |R_P| * 2 * k_m - sum_workload + |R_P| * sum(y_t * c_d(t))
        sum_workload_terms = []
        sum_c_r_constant = 0

        for r in R_P:
            c_r = int(predicted_remaining_times.get(r, 0.0) * 1000)
            sum_c_r_constant += c_r
            for t in T:
                if (t, r) in x:
                    if t in costs_authorized_and_timeslotoperating_resources_per_task:
                        if r in costs_authorized_and_timeslotoperating_resources_per_task[t]:
                            cost = costs_authorized_and_timeslotoperating_resources_per_task[t][r]
                            sum_workload_terms.append(cost * x[(t, r)])

        max_sum_workload = max_remaining * nR_P + max_task_cost * nT * nR_P
        sum_workload_var = model.NewIntVar(0, max_sum_workload, "sum_workload")
        model.Add(sum_workload_var == sum_c_r_constant + sum(sum_workload_terms))

        dummy_sum_terms = []
        for t in T:
            if t in dummy_costs:
                # +_DUMMY_TIEBREAK_MS per dummy assignment so that a real resource
                # is strictly preferred when its cost equals the dummy threshold.
                dummy_sum_terms.append((dummy_costs[t] + _DUMMY_TIEBREAK_MS) * y[t])

        if nR_P > 0:
            objective = nR_P * 2 * k_m - sum_workload_var + nR_P * sum(dummy_sum_terms)
        else:
            objective = sum(dummy_sum_terms)

        model.Minimize(objective)

        validation = model.Validate()
        if validation:
            raise ValueError(f"CP-SAT model validation failed: {validation}")

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = effective_time_limit
        solver.parameters.num_search_workers = 1

        # logger.info("PMSP [Solver]: Starting CP-SAT solver for %d tasks, %d resources (time limit: %.2fs)", 
        #            nT, nR_P, effective_time_limit)
        # logger.info("PMSP [Solver]: Problem size - Tasks: %d, Resources: %d, Variables: %d", 
        #            nT, nR_P, len(x) + len(y))
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        # logger.info("PMSP [Solver]: CP-SAT solver finished with status: %s (wall_time: %.3fs)", 
        #            status_name, solver.WallTime())

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

            k_m_val = solver.Value(k_m)
            sum_workload_val = solver.Value(sum_workload_var)
            k_f_val = k_m_val - (sum_workload_val / nR_P) if nR_P > 0 else 0.0
            dummy_sum = sum(int(dummy_costs.get(t, 0)) * solver.Value(y[t]) for t in T)
            dummy_count = sum(1 for t in T if solver.Value(y[t]) >= 1)
            cp_obj = int(round(k_m_val + k_f_val + dummy_sum))
            # Only run JV if CP-SAT is non-optimal and we want to compare objectives.
            if status != cp_model.OPTIMAL:
                assignment_jv, debug_jv = _run_jv_fallback()
                jv_obj = int(debug_jv.get("objective", 10**18))
                if jv_obj <= cp_obj:
                    dbg = dict(debug_jv)
                    dbg.update({"status": int(status), "status_name": status_name, "fallback_from": "cp_sat"})
                    return assignment_jv, dbg

            assigned_count = sum(1 for v in assignment_cp.values() if v is not None)
            # logger.info("PMSP [Solver]: Optimization complete - %d/%d tasks assigned (objective: %d, k_m: %d, k_f: %d)", 
            #            assigned_count, nT, cp_obj, int(k_m_val), int(round(k_f_val)))
            # logger.info("PMSP [Solver]: Makespan (k_m): %d ms, Fairness (k_f): %d ms, Dummy assignments: %d/%d tasks (cost: %d ms)", 
            #            int(k_m_val), int(round(k_f_val)), dummy_count, nT, dummy_sum)
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

        # CP-SAT finished but without a feasible/optimal solution: use JV fallback.
        assignment_jv, debug_jv = _run_jv_fallback()
        dbg = dict(debug_jv)
        dbg.update({"status": int(status), "status_name": status_name, "fallback_from": "cp_sat"})
        return assignment_jv, dbg

    except Exception as e:
        # logger.warning("PMSP [Solver]: CP-SAT solver failed, using JV fallback: %s", e)
        assignment_jv, debug_jv = _run_jv_fallback()
        assigned_count = sum(1 for v in assignment_jv.values() if v is not None)
        # logger.info("PMSP [Solver]: Using JV fallback - %d/%d tasks assigned", assigned_count, nT)
        dbg = dict(debug_jv)
        dbg.update({"fallback_from": "cp_sat_import_or_runtime_error", "error": str(e)})
        return assignment_jv, dbg
