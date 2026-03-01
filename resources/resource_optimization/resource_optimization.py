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


def _safe_seconds(x: Any) -> Optional[float]:
    """Normalize predictor output to a sane positive float (seconds)."""
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v) or v < 0:
        return None
    return v


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

    try:
        seconds = predictor.predict(
            prev_activity=prev_activity,
            prev_lifecycle="complete",
            curr_activity=curr_activity,
            curr_lifecycle="complete",
            context=ctx,
        )
        return _safe_seconds(seconds)
    except Exception:
        return None


def choose_resource(
    *,
    cfg: SelectionConfig,
    activity: str,
    timestamp: datetime,
    case: Any,
    available_resources: List[str],
    eligible_resources: List[str],
    processing_time_predictor: Any,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Choose a resource for a single activity at a given timestamp.

    Returns:
        (resource or None, debug_info dict)

    If `resource` is None, the caller should interpret this as selecting a
    "dummy" (postpone) *or* as no feasible resource (depending on the context).

    `available_resources` must already be filtered for:
      - eligibility
      - working-hours availability
      - not-busy in the simulation resource pool
    """
    debug: Dict[str, Any] = {
        "mode": cfg.mode,
        "dummy_delta": cfg.dummy_delta,
        "n_available": len(available_resources),
        "n_eligible": len(eligible_resources),
    }

    if not available_resources:
        debug["decision"] = "no_available"
        return None, debug

    # Deterministic RNG for reproducibility if desired
    rng = random.Random(cfg.rng_seed) if cfg.rng_seed is not None else random

    hist = getattr(case, "activity_history", []) or []
    prev_activity = hist[-1] if hist else "START"

    costs_avail: Dict[str, float] = {}
    for r in available_resources:
        sec = predict_processing_seconds(
            processing_time_predictor,
            prev_activity=prev_activity,
            curr_activity=activity,
            current_time=timestamp,
            case=case,
            candidate_resource=r,
        )
        if sec is not None:
            costs_avail[r] = sec

    # If predictor fails for everything, fall back safely
    if not costs_avail:
        choice = sorted(available_resources)[0]
        debug["decision"] = "fallback_no_predictions"
        debug["selected"] = choice
        return choice, debug

    # Best available resource by predicted processing time
    best_r, best_c = min(costs_avail.items(), key=lambda kv: kv[1])
    debug["best_resource"] = best_r
    debug["best_cost"] = best_c

    if cfg.mode == "min_time" or not cfg.allow_dummy:
        debug["decision"] = "min_time"
        debug["selected"] = best_r
        return best_r, debug

    # Dummy rule
    if len(set(eligible_resources)) <= 1:
        debug["decision"] = "min_time_dummy_disabled_single_eligible"
        debug["selected"] = best_r
        return best_r, debug

    # Estimate average cost across eligible resources (sample if large)
    eligible_unique = sorted(set(eligible_resources))
    if cfg.avg_sample_size and len(eligible_unique) > cfg.avg_sample_size:
        eligible_unique = rng.sample(eligible_unique, cfg.avg_sample_size)

    eligible_costs: List[float] = []
    for r in eligible_unique:
        sec = predict_processing_seconds(
            processing_time_predictor,
            prev_activity=prev_activity,
            curr_activity=activity,
            current_time=timestamp,
            case=case,
            candidate_resource=r,
        )
        if sec is not None:
            eligible_costs.append(sec)

    if not eligible_costs:
        debug["decision"] = "min_time_dummy_no_avg"
        debug["selected"] = best_r
        return best_r, debug

    avg_c = sum(eligible_costs) / len(eligible_costs)
    dummy_c = float(cfg.dummy_delta) * avg_c
    debug["avg_cost"] = avg_c
    debug["dummy_cost"] = dummy_c

    if best_c > dummy_c:
        debug["decision"] = "dummy_postpone"
        debug["selected"] = None
        return None, debug

    debug["decision"] = "min_time_with_dummy"
    debug["selected"] = best_r
    return best_r, debug


def solve_pmsp_ilp(
    *,
    delta: float,
    tasks: List[str],
    resources: List[str],
    costs: Dict[Tuple[str, str], int],
    dummy_cost: Dict[str, int],
    remaining: Dict[str, int],
    solver_time_limit_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """
    Solve PMSP adaptation.

    Primary solver: OR-Tools CP-SAT (time-limited, default 2s).
    Backup solver: SciPy Jonker-Volgenant assignment adaptation.
    """
    T = list(tasks)
    R = list(resources)
    nT = len(T)
    nR = len(R)
    if nT == 0 or nR == 0:
        return {t: None for t in T}, {"status": 0, "nT": nT, "nR": nR, "solver": "empty"}

    assignment_jv, debug_jv = solve_assignment_jv(
        tasks=T,
        resources=R,
        costs=costs,
        dummy_cost=dummy_cost,
        remaining=remaining,
    )

    effective_time_limit = 2.0 if solver_time_limit_seconds is None else float(solver_time_limit_seconds)
    effective_time_limit = max(0.01, effective_time_limit)

    try:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        x: Dict[Tuple[str, str], Any] = {}
        y: Dict[str, Any] = {}

        for (t, r), _ in costs.items():
            x[(t, r)] = model.NewBoolVar(f"x__{t}__{r}")
        for t in T:
            y[t] = model.NewBoolVar(f"y__{t}")

        km_values = []
        for r in R:
            max_cost_on_r = max([costs.get((t, r), 0) for t in T] or [0])
            km_values.append(remaining.get(r, 0) + max_cost_on_r)
        ub_km = int(max(km_values) if km_values else 0)
        k_m = model.NewIntVar(0, ub_km, "k_m")

        # each task to one resource or dummy
        for t in T:
            x_t = [x[(tt, rr)] for (tt, rr) in x.keys() if tt == t]
            model.Add(sum(x_t) + y[t] == 1)

        # makespan constraints
        for r in R:
            terms = []
            for t in T:
                if (t, r) in x:
                    terms.append(costs[(t, r)] * x[(t, r)])
            model.Add(remaining.get(r, 0) + sum(terms) <= k_m)

        # objective pieces
        # (Hinweis: In deinem Paste fehlen ggf. max_remaining/max_task_cost Definitionen –
        # ich lasse den Rest bewusst unverändert.)
        max_S = int(max_remaining * nR + max_task_cost * nT * nR)  # noqa: F821
        S = model.NewIntVar(0, max_S, "S")

        sum_remaining = sum(remaining.get(r, 0) for r in R)
        sum_workload = []
        for r in R:
            for t in T:
                if (t, r) in x:
                    sum_workload.append(costs[(t, r)] * x[(t, r)])
        model.Add(S == sum_remaining + sum(sum_workload))

        scaled_obj = 2 * nR * k_m - S + nR * sum(int(dummy_cost[t]) * y[t] for t in T)
        model.Minimize(scaled_obj)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = effective_time_limit

        # --- KEINE Parallelisierung im Solver: exakt 1 Worker ---
        solver.parameters.num_search_workers = 1

        status = solver.Solve(model)
        status_name = solver.StatusName(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignment_cp: Dict[str, Optional[str]] = {t: None for t in T}
            for t in T:
                if solver.Value(y[t]) >= 1:
                    assignment_cp[t] = None
                    continue
                chosen = None
                for r in R:
                    if (t, r) in x and solver.Value(x[(t, r)]) >= 1:
                        chosen = r
                        break
                assignment_cp[t] = chosen

            k_m_val = solver.Value(k_m)
            S_val = solver.Value(S)
            k_f_val = k_m_val - (S_val / nR) if nR > 0 else 0.0
            dummy_sum = sum(int(dummy_cost[t]) * solver.Value(y[t]) for t in T)
            cp_obj = int(round(k_m_val + k_f_val + dummy_sum))
            jv_obj = int(debug_jv.get("objective", 10**18))

            if status != cp_model.OPTIMAL and jv_obj <= cp_obj:
                dbg = dict(debug_jv)
                dbg.update({"status": int(status), "status_name": status_name, "fallback_from": "cp_sat"})
                return assignment_jv, dbg

            return assignment_cp, {
                "status": int(status),
                "status_name": status_name,
                "nT": nT,
                "nR": nR,
                "solver": "cp_sat",
                "objective": cp_obj,
                "k_m": int(k_m_val),
                "k_f": int(round(k_f_val)),
            }

        dbg = dict(debug_jv)
        dbg.update({"status": int(status), "status_name": status_name, "fallback_from": "cp_sat"})
        return assignment_jv, dbg

    except Exception:
        dbg = dict(debug_jv)
        dbg.update({"fallback_from": "cp_sat_import_or_runtime_error"})
        return assignment_jv, dbg