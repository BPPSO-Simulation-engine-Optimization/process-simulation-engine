"""
Run N paired simulations (with vs without terminated resources) and save a comparison CSV.

Each pair uses the same random seed so arrivals are identical; only resource exclusion
differs. Produces a distribution of metric changes for use in log_comparison notebooks.

Usage:
    python -m integration.run_termination_comparison --n-runs 10 --exclude-resources User_128,User_129
    python -m integration.run_termination_comparison --n-runs 50 --jobs 8 --exclude-resources User_128,User_129
    python -m integration.run_termination_comparison --n-runs 5 --num-cases 200 --output-csv evaluation/termination_runs.csv

Progress is written after each run pair. Re-run with the same --output-csv and --n-runs to resume
(only missing run_ids are executed). Use --no-resume to start from scratch.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from integration.config import SimulationConfig
from integration.setup import setup_simulation
from integration.test_integration import (
    load_event_log,
    create_resource_allocator,
    run_simulation,
)
from simulation.engine import DESEngine, NextActivityPredictorType
from simulation.log_exporter import LogExporter

from evaluation.log_metrics import load_log, compute_all_metrics

COST_PER_FTE_HOUR = 50.0
REPO_ROOT = Path(__file__).parent.parent


def _pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or a == 0:
        return float("nan")
    return (b - a) / a * 100


def _available_memory_bytes_windows() -> int | None:
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) == 0:
            return None
        return int(stat.ullAvailPhys)
    except Exception:
        return None


def _choose_worker_count(requested: int | None, backend: str) -> int:
    cpu = os.cpu_count() or 1
    if requested is not None and requested > 0:
        return min(requested, cpu)

    if backend == "processes":
        return cpu

    avail = _available_memory_bytes_windows() if os.name == "nt" else None
    if avail is None:
        return cpu

    per_worker_bytes = int(2.5 * 1024**3)
    return max(1, min(cpu, avail // per_worker_bytes))


def _append_row_to_csv(row: dict, path: Path, write_header: bool, columns: list[str] | None) -> list[str]:
    if columns is None:
        columns = ["run_id"] + sorted(k for k in row if k != "run_id")
    ordered = [row.get(c) for c in columns]
    df_one = pd.DataFrame([ordered], columns=columns)
    mode = "w" if write_header else "a"
    path.parent.mkdir(parents=True, exist_ok=True)
    df_one.to_csv(path, mode=mode, header=write_header, index=False)
    return columns


_WORKER_DF: pd.DataFrame | None = None
_WORKER_LOG_PATH: str | None = None


def _init_worker(log_path: str) -> None:
    global _WORKER_DF, _WORKER_LOG_PATH
    _WORKER_LOG_PATH = log_path
    _WORKER_DF = load_event_log(log_path)


def run_one_pair(
    run_id: int,
    config: SimulationConfig,
    df: pd.DataFrame | None,
    log_path: str,
    base_output_dir: Path,
    exclude_resources: list[str],
    cost_per_fte: float,
) -> dict:
    """Run with and without termination; return metrics and differences for this pair."""
    import copy

    if df is None:
        df = _WORKER_DF
    if df is None:
        df = load_event_log(log_path)

    config = copy.deepcopy(config)
    config.random_seed = (config.random_seed or 42) + run_id
    out_base = base_output_dir / f"run_{run_id}"
    out_with = out_base / "with_termination"
    out_without = out_base / "without_termination"

    config.exclude_resources = exclude_resources
    allocator_with = create_resource_allocator(log_path, config)
    run_simulation(config, df, allocator_with, str(out_with))

    config.exclude_resources = []
    allocator_without = create_resource_allocator(log_path, config)
    run_simulation(config, df, allocator_without, str(out_without))

    xes_with = out_with / "simulated_log.xes"
    xes_without = out_without / "simulated_log.xes"
    if not xes_with.exists():
        xes_with = out_with / "simulated_log.csv"
    if not xes_without.exists():
        xes_without = out_without / "simulated_log.csv"

    df_with = load_log(xes_with)
    df_without = load_log(xes_without)
    m_with = compute_all_metrics(df_with, cost_per_fte)
    m_without = compute_all_metrics(df_without, cost_per_fte)

    row = {"run_id": run_id}
    for k in m_with:
        a, b = m_with[k], m_without[k]
        row[f"{k}_with"] = a
        row[f"{k}_without"] = b
        row[f"{k}_diff"] = b - a if not (pd.isna(a) or pd.isna(b)) else float("nan")
        row[f"{k}_pct"] = _pct_change(a, b)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Run N paired simulations (with/without terminated resources) and save comparison CSV"
    )
    parser.add_argument("--n-runs", type=int, default=10, help="Number of run pairs")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel workers (default: all CPU cores)",
    )
    parser.add_argument(
        "--backend",
        choices=["threads", "processes"],
        default="threads",
        help="Parallel backend. Use 'threads' to share memory (default); use 'processes' for true CPU parallelism.",
    )
    parser.add_argument("--num-cases", type=int, default=1000, help="Cases per run (default: full log)")
    parser.add_argument(
        "--exclude-resources",
        type=str,
        default=None,
        help="Comma-separated resources to exclude (terminated), e.g. User_128,User_129",
    )
    parser.add_argument(
        "--event-log",
        default="Dataset/BPI Challenge 2017.xes",
        help="Path to event log",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory for run subdirs (default: integration/output/termination_runs)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for comparison CSV (default: evaluation/termination_comparison_runs.csv)",
    )
    parser.add_argument("--mode", choices=["basic", "advanced", "mixed"], default="basic")
    parser.add_argument(
        "--next-activity",
        choices=[
            "lstm", "process_transformer",
            "lifecycle_dual_full_baseline", "lifecycle_dual_full_balanced",
            "lifecycle_dual_start_complete_baseline",
        ],
        default="lifecycle_dual_full_balanced",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed; run i uses seed + i")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output CSV and run all n-runs from scratch",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "integration" / "output" / "termination_runs"
    else:
        args.output_dir = Path(args.output_dir)
    if args.output_csv is None:
        args.output_csv = REPO_ROOT / "evaluation" / "termination_comparison_runs.csv"
    else:
        args.output_csv = Path(args.output_csv)

    exclude_list = []
    if args.exclude_resources:
        exclude_list = [r.strip() for r in args.exclude_resources.split(",") if r.strip()]
    if not exclude_list:
        parser.error("--exclude-resources is required (e.g. --exclude-resources User_128,User_129)")

    done_ids: set[int] = set()
    csv_columns: list[str] | None = None
    if args.output_csv.exists() and not args.no_resume:
        try:
            existing = pd.read_csv(args.output_csv)
            if "run_id" in existing.columns and len(existing) > 0:
                done_ids = set(existing["run_id"].astype(int))
                csv_columns = list(existing.columns)
                if done_ids:
                    print(f"Resuming: {len(done_ids)} runs already in {args.output_csv}")
        except Exception as e:
            print(f"Could not read existing CSV: {e}")

    run_ids_to_do = [i for i in range(args.n_runs) if i not in done_ids]
    if not run_ids_to_do:
        print("All runs already done.")
        result = pd.read_csv(args.output_csv)
        diff_cols = [c for c in result.columns if c.endswith("_diff")]
        if diff_cols:
            summary = result[diff_cols].agg(["mean", "std", "min", "max"])
            summary.columns = [c.replace("_diff", "") for c in summary.columns]
            print(summary.round(4).to_string())
        return

    log_path = args.event_log
    if not Path(log_path).is_absolute():
        log_path = str(REPO_ROOT / log_path)
    print(f"Loading event log: {log_path}")
    df = load_event_log(log_path)

    num_cases = args.num_cases
    if num_cases is None:
        num_cases = df["case:concept:name"].nunique()
    print(f"Cases per run: {num_cases}")

    if args.mode == "basic":
        config = SimulationConfig.all_basic()
    elif args.mode == "advanced":
        config = SimulationConfig.all_advanced(
            event_log_path=log_path,
            num_cases=num_cases,
        )
    else:
        config = SimulationConfig(num_cases=num_cases, event_log_path=log_path)
    config.num_cases = num_cases
    config.random_seed = args.seed
    config.next_activity_class = args.next_activity
    config.exclude_resources = exclude_list

    if args.next_activity == "lifecycle_dual_full_balanced":
        candidate = [
            "next_activity_prediction_lifecycle_dual/models/full_lifecycle/balanced",
            "next_activity_prediction_lifecycle_dual/next_activity_prediction_lifecycle_dual/models/full_lifecycle/balanced",
        ]
        for p in candidate:
            if (REPO_ROOT / p).exists():
                config.next_activity_model_path = p
                break
        config.next_activity_class = "lstm"
        config.next_activity_mode = "advanced"
        config.next_activity_model_type = "lifecycle_dual"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_workers = _choose_worker_count(args.jobs, args.backend)
    total_to_do = len(run_ids_to_do)
    print(f"Running {total_to_do} pairs (of {args.n_runs}) with {max_workers} workers ({args.backend})")

    from concurrent.futures import as_completed

    file_has_content = args.output_csv.exists() and args.output_csv.stat().st_size > 0
    if args.backend == "threads":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    run_one_pair,
                    run_id,
                    config,
                    df,
                    log_path,
                    args.output_dir,
                    exclude_list,
                    COST_PER_FTE_HOUR,
                ): run_id
                for run_id in run_ids_to_do
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                write_header = not file_has_content
                csv_columns = _append_row_to_csv(row, args.output_csv, write_header, csv_columns)
                file_has_content = True
                print(f"Completed {i}/{total_to_do} (run_id={row['run_id']}) -> {args.output_csv}")
    else:
        from concurrent.futures import ProcessPoolExecutor

        print(
            "Note: process backend duplicates the event log per worker; "
            "if you hit memory errors, lower --jobs or use --backend threads."
        )
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(log_path,),
        ) as ex:
            futures = {
                ex.submit(
                    run_one_pair,
                    run_id,
                    config,
                    None,
                    log_path,
                    args.output_dir,
                    exclude_list,
                    COST_PER_FTE_HOUR,
                ): run_id
                for run_id in run_ids_to_do
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                write_header = not file_has_content
                csv_columns = _append_row_to_csv(row, args.output_csv, write_header, csv_columns)
                file_has_content = True
                print(f"Completed {i}/{total_to_do} (run_id={row['run_id']}) -> {args.output_csv}")

    result = pd.read_csv(args.output_csv).sort_values("run_id").reset_index(drop=True)
    print(f"\nSaved comparison CSV: {args.output_csv} ({len(result)} runs)")

    diff_cols = [c for c in result.columns if c.endswith("_diff")]
    print("\nDistribution summary (difference: without - with):")
    summary = result[diff_cols].agg(["mean", "std", "min", "max"])
    summary.columns = [c.replace("_diff", "") for c in summary.columns]
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
