"""
Targeted PMSP bottleneck profiler.
Measures calculate_pmsp_parameters (inference) vs solve_pmsp_ilp (optimization) separately.
Run from project root: python _pmsp_bottleneck_profiler.py
"""
import sys
import time
import os
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np

# ── 1. Load real processing-time predictor ────────────────────────────────────
print("Loading processing-time predictor...")
t0 = time.perf_counter()
from processing_time_prediction.ProcessingTimePredictionClass import ProcessingTimePredictionClass
pt_predictor = ProcessingTimePredictionClass(method="ml", model_path="models/processing_time_model")
print(f"  Loaded in {time.perf_counter()-t0:.2f}s  (method={pt_predictor.method})\n")

# ── 2. Load real resource allocator ──────────────────────────────────────────
print("Loading ResourceAllocator...")
t0 = time.perf_counter()
from resources import ResourceAllocator
allocator = ResourceAllocator(log_path="eventlog/eventlog.xes.gz")
print(f"  Loaded in {time.perf_counter()-t0:.2f}s\n")

# ── 3. Build synthetic waiting tasks that mimic real PMSP calls ───────────────
from simulation.engine import WaitingWork
from simulation.case_manager import CaseState

ALL_ACTIVITIES = [
    "W_Complete application",
    "W_Call after offers",
    "W_Validate application",
    "W_Call incomplete files",
    "W_Handle leads",
    "W_Assess potential fraud",
]

RESOURCES = allocator.availability.resources  # real resource list

def make_case_state(case_id: str, activity: str) -> CaseState:
    cs = CaseState(
        case_id=case_id,
        case_type="Personal Loan",
        application_type="New credit",
        requested_amount=5000.0,
        start_time=datetime(2016, 1, 4, 8, 0),
    )
    cs.activity_history = [activity]
    cs.lifecycle_history = ["complete"]
    return cs

def make_waiting_tasks(n: int, timestamp: datetime) -> list:
    tasks = []
    for i in range(n):
        act = ALL_ACTIVITIES[i % len(ALL_ACTIVITIES)]
        case_id = f"case_{i:04d}"
        cs = make_case_state(case_id, act)
        ww = WaitingWork(
            case_id=case_id,
            activity=act,
            lifecycle="complete",
            allocation_activity=act,
            arrival_time=timestamp,
            case_state=cs,
        )
        tasks.append(ww)
    return tasks

TIMESTAMP = datetime(2016, 1, 5, 10, 0)

# ── 4. Import PMSP functions ──────────────────────────────────────────────────
from resources.resource_optimization.resource_optimization import (
    SelectionConfig,
    calculate_pmsp_parameters,
    solve_pmsp_ilp,
    handle_batch_scheduling_optimization,
)

# ── 5. Micro-benchmark ─────────────────────────────────────────────────────────
BATCH_SIZES = [5, 10, 20]
REPS = 3

print("=" * 70)
print("PMSP BOTTLENECK PROFILER")
print("=" * 70)

for n_tasks in BATCH_SIZES:
    print(f"\n{'─'*70}")
    print(f"  Batch size: {n_tasks} tasks")
    print(f"{'─'*70}")

    waiting_tasks = make_waiting_tasks(n_tasks, TIMESTAMP)

    # Build authorized_resources_per_task for each task
    authorized_resources_per_task = {}
    for wt in waiting_tasks:
        task_id = f"{wt.case_id}_{wt.allocation_activity}"
        try:
            eligible = allocator.permissions.get_eligible_resources(
                wt.allocation_activity, timestamp=TIMESTAMP, case_type=wt.case_state.case_type
            )
        except TypeError:
            eligible = allocator.permissions.get_eligible_resources(wt.allocation_activity)
        authorized_resources_per_task[task_id] = eligible or []

    n_eligible = sum(len(v) for v in authorized_resources_per_task.values())
    print(f"  Total (task × resource) pairs: {n_eligible}")

    # ── Phase A: Inference (calculate_pmsp_parameters) ──────────────────────
    inference_times = []
    for rep in range(REPS):
        pt_cache = {}
        t0 = time.perf_counter()
        dummy_costs, pred_remaining, costs_dict, raw_pt = calculate_pmsp_parameters(
            delta=1.5,
            authorized_resources_per_task=authorized_resources_per_task,
            waiting_tasks=waiting_tasks,
            timestamp=TIMESTAMP,
            processing_time_predictor=pt_predictor,
            resource_busy_until={},
            allocator=allocator,
            prediction_batch_size=0,
            pt_cache=pt_cache,
        )
        elapsed = time.perf_counter() - t0
        inference_times.append(elapsed)
        print(f"  [Inference] rep {rep+1}: {elapsed*1000:.1f} ms  ({len(pt_cache)} cache entries)")

    mean_infer = np.mean(inference_times[1:]) if len(inference_times) > 1 else inference_times[0]
    print(f"  [Inference] MEAN (excl. warm-up): {mean_infer*1000:.1f} ms")

    # Also check cache hit scenario (2nd run with same cache)
    pt_cache_warm = {}
    calculate_pmsp_parameters(
        delta=100.0,
        authorized_resources_per_task=authorized_resources_per_task,
        waiting_tasks=waiting_tasks,
        timestamp=TIMESTAMP,
        processing_time_predictor=pt_predictor,
        resource_busy_until={},
        allocator=allocator,
        prediction_batch_size=0,
        pt_cache=pt_cache_warm,
    )
    t0 = time.perf_counter()
    calculate_pmsp_parameters(
        delta=100.0,
        authorized_resources_per_task=authorized_resources_per_task,
        waiting_tasks=waiting_tasks,
        timestamp=TIMESTAMP,
        processing_time_predictor=pt_predictor,
        resource_busy_until={},
        allocator=allocator,
        prediction_batch_size=0,
        pt_cache=pt_cache_warm,
    )
    cache_hit_time = time.perf_counter() - t0
    print(f"  [Inference] 100%% cache-hit run: {cache_hit_time*1000:.1f} ms")

    # ── Phase B: CP-SAT Optimization (solve_pmsp_ilp) ───────────────────────
    # Build inputs for the solver using real cost data
    task_ids = list(authorized_resources_per_task.keys())
    authorized_and_ts_resources_per_task = {}
    for task_id in task_ids:
        authorized_and_ts_resources_per_task[task_id] = [
            r for r in authorized_resources_per_task[task_id]
            if allocator.availability.is_available(r, TIMESTAMP)
        ]

    costs_ts = {}
    for task_id in task_ids:
        costs_ts[task_id] = {
            r: costs_dict.get(task_id, {}).get(r, 60000)
            for r in authorized_and_ts_resources_per_task.get(task_id, [])
            if r in costs_dict.get(task_id, {})
        }

    predicted_remaining_times = {r: 0.0 for r in RESOURCES}

    solver_times = []
    solver_statuses = []
    for rep in range(REPS):
        t0 = time.perf_counter()
        assignment, debug = solve_pmsp_ilp(
            delta=1.5,
            tasks=task_ids,
            authorized_resources_by_waiting_task=authorized_resources_per_task,
            authorized_and_timeSlotOperating_resources_by_task=authorized_and_ts_resources_per_task,
            costs_authorized_and_timeslotoperating_resources_per_task=costs_ts,
            dummy_costs=dummy_costs,
            predicted_remaining_times=predicted_remaining_times,
            costs_authorized_resource_task_assignment=costs_dict,
            solver_time_limit_seconds=2.0,
        )
        elapsed = time.perf_counter() - t0
        solver_times.append(elapsed)
        solver_statuses.append(debug.get("status_name", debug.get("solver", "?")))
        print(f"  [CP-SAT]    rep {rep+1}: {elapsed*1000:.1f} ms  status={solver_statuses[-1]}")

    mean_solver = np.mean(solver_times[1:]) if len(solver_times) > 1 else solver_times[0]
    print(f"  [CP-SAT]    MEAN (excl. warm-up): {mean_solver*1000:.1f} ms")

    # ── Phase C: Full end-to-end (handle_batch_scheduling_optimization) ──────
    cfg = SelectionConfig(
        mode="pmsp",
        dummy_delta=1.5,
        pmsp_solver_time_limit_seconds=2.0,
        prediction_batch_size=0,
        optimization_batch_size=0,
    )
    e2e_times = []
    for rep in range(REPS):
        pt_cache_e2e = {}
        t0 = time.perf_counter()
        handle_batch_scheduling_optimization(
            cfg=cfg,
            timestamp=TIMESTAMP,
            waiting_tasks=waiting_tasks,
            processing_time_predictor=pt_predictor,
            allocator=allocator,
            resource_pool=None,
            pt_cache=pt_cache_e2e,
            resource_worklist_backlog={},
        )
        elapsed = time.perf_counter() - t0
        e2e_times.append(elapsed)
        print(f"  [E2E]       rep {rep+1}: {elapsed*1000:.1f} ms")

    mean_e2e = np.mean(e2e_times[1:]) if len(e2e_times) > 1 else e2e_times[0]
    mean_solver_2 = np.mean(solver_times[1:]) if len(solver_times) > 1 else solver_times[0]
    pct_inference = 100.0 * mean_infer / mean_e2e if mean_e2e > 0 else 0
    pct_solver    = 100.0 * mean_solver_2 / mean_e2e if mean_e2e > 0 else 0

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │ n_tasks={n_tasks:2d}  total (task×resource)={n_eligible:4d}      │")
    print(f"  │ Inference (avg): {mean_infer*1000:8.1f} ms  ({pct_inference:5.1f}%)   │")
    print(f"  │ CP-SAT    (avg): {mean_solver_2*1000:8.1f} ms  ({pct_solver:5.1f}%)   │")
    print(f"  │ E2E       (avg): {mean_e2e*1000:8.1f} ms                    │")
    print(f"  └─────────────────────────────────────────────┘")

print(f"\n{'='*70}")
print("SUMMARY: The dominant bottleneck is whichever phase has the highest ms above.")
print("="*70)
