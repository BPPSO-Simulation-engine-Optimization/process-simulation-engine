"""
Integration test for the simulation engine with all prediction components.

This script runs a full simulation using:
- Case arrivals: basic (stub) or advanced (CaseInterarrivalPipeline)
- Processing times: basic (stub) or advanced (ProcessingTimePredictionClass)
- Case attributes: basic (stub) or advanced (AttributeSimulationEngine)

+ It saves the respective subset of the GT EL in case num-cases is specified

Usage:
    python -m integration.test_integration --num-cases 1000 --event-log eventlog/eventlog.xes.gz
    python -m integration.test_integration --arrivals advanced --processing advanced --num-cases 100
    python -m integration.test_integration --processing advanced --processing-model-path models/processing_time_model
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from integration.config import SimulationConfig
from integration.setup import setup_simulation
from simulation.engine import DESEngine, NextActivityPredictorType
from simulation.log_exporter import LogExporter
from resources.selection_strategies import create_strategy


def load_event_log(path: str) -> pd.DataFrame:
    """Load event log from XES or CSV file."""
    if path.endswith('.xes') or path.endswith('.xes.gz'):
        import pm4py
        log = pm4py.read_xes(path)
        df = pm4py.convert_to_dataframe(log)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    print(f"Loaded event log: {len(df)} events, {df['case:concept:name'].nunique()} cases")
    return df


def create_resource_allocator(log_path: str):
    """Create resource allocator from event log."""
    try:
        from resources import ResourceAllocator
        allocator = ResourceAllocator(log_path=log_path)
        print("Loaded ResourceAllocator from event log")
        return allocator
    except Exception as e:
        raise Exception(f"Could not load ResourceAllocator: {e}")


def save_ground_truth_subset(df: pd.DataFrame, num_cases: int, output_dir: str):
    """
    Save a subset of the original event log with the first N cases (by arrival time).

    Args:
        df: Original event log DataFrame
        num_cases: Number of cases to keep
        output_dir: Output directory for the reduced log
    """
    # Get case arrival times (first event per case)
    case_arrivals = df.groupby('case:concept:name')['time:timestamp'].min().sort_values()

    # Select first N cases by arrival time
    selected_cases = case_arrivals.head(num_cases).index.tolist()

    # Filter the event log
    reduced_df = df[df['case:concept:name'].isin(selected_cases)].copy()

    print(f"\nGround truth subset: {len(selected_cases)} cases, {len(reduced_df)} events")

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "ground_truth_log.csv")
    reduced_df.to_csv(csv_path, index=False)
    print(f"Exported ground truth CSV to: {csv_path}")

    try:
        import pm4py
        xes_path = os.path.join(output_dir, "ground_truth_log.xes")
        pm4py.write_xes(reduced_df, xes_path)
        print(f"Exported ground truth XES to: {xes_path}")
    except Exception as e:
        print(f"Could not export ground truth XES: {e}")

    return reduced_df


def run_simulation(config: SimulationConfig, df: pd.DataFrame, allocator, output_dir: str, enable_profiling: bool = False):
    """Run the simulation with given configuration."""
    print("\n" + "=" * 60)
    print("SIMULATION CONFIGURATION")
    print("=" * 60)
    print(f"  Processing time mode: {config.processing_time_mode}")
    print(f"  Case arrival mode: {config.case_arrival_mode}")
    print(f"  Resource selection: {config.resource_selection_strategy}")
    print(f"  Resource allocation mode: {config.resource_allocation_mode}")
    if config.next_activity_class == "process_transformer":
        print(f"  PT lifecycle mode: {config.pt_lifecycle_mode}")
        print(f"  PT max duration: {config.pt_max_duration_seconds / 3600:.0f}h ({config.pt_max_duration_seconds / 86400:.0f} days)")
    if config.resource_allocation_mode == "batch":
        print(f"  Batch policy: {config.batch_policy}")
    elif config.resource_allocation_mode == "drl":
        print(f"  DRL model: {config.drl_model_path}")
    elif config.resource_allocation_mode == "pmsp":
        print(f"  PMSP delta: {config.pmsp_dummy_delta}")
        print(f"  PMSP solver time limit: {config.pmsp_solver_time_limit_seconds}s")
        print(f"  PMSP prediction batch size: {config.pmsp_prediction_batch_size}")
        print(f"  PMSP optimization batch size: {config.pmsp_optimization_batch_size}")
    print(f"  Number of cases: {config.num_cases}")
    print("=" * 60 + "\n")

    # Get start date - use Monday 8am Jan 4, 2016 to avoid weekend/holiday issues
    # (The event log starts on 2016-01-01 which is a holiday, causing many arrivals
    #  to be pushed to 2016-01-04 08:00:00, making PMSP timestamps stuck)
    start_date = datetime(2016, 1, 4, 8, 0)
    print(f"Simulation start date: {start_date}")

    # Setup predictors
    print("\nSetting up predictors...")
    # Pass df if arrival mode is advanced (needed for training/fallback)
    needs_df = config.case_arrival_mode == "advanced"
    arrivals, next_act_pred, proc_pred, attr_pred = setup_simulation(
        config,
        df=df if needs_df else None,
        start_date=start_date,
    )
    print(f"Generated {len(arrivals)} arrival timestamps")

    # Create engine
    print("\nInitializing DESEngine...")

    # Adjust start_time to be the earliest of simulation start date or first arrival
    # This prevents "Cannot go back in time" errors if the arrival generator
    # produces timestamps earlier in the day than the log's start time (due to normalization).
    engine_start_time = start_date
    if arrivals and len(arrivals) > 0:
         if arrivals[0] < start_date:
             engine_start_time = arrivals[0]
             print(f"Adjusting simulation start time to first arrival: {engine_start_time}")

    # Determine appropriate predictor type argument.
    # DESEngine requires either an explicit predictor instance OR a predictor type
    # to auto-load. For "lstm" we delegate loading to the engine, so we must pass
    # NextActivityPredictorType.LSTM here.
    pred_type = None
    if config.next_activity_class == "process_transformer":
        pred_type = NextActivityPredictorType.PROCESS_TRANSFORMER
    elif config.next_activity_class == "lstm":
        pred_type = NextActivityPredictorType.LSTM

    # Create resource selection strategy
    resource_strategy = create_strategy(config.resource_selection_strategy)

    # Create batch allocation policy (if requested)
    batch_policy = None
    pt_estimator = None
    if config.resource_allocation_mode == "batch":
        from resources.batch_policies import create_batch_policy
        from resources.processing_time_estimator import ProcessingTimeEstimator

        batch_policy = create_batch_policy(config.batch_policy)
        pt_estimator = ProcessingTimeEstimator(df=df)
        print(f"Created batch policy: {config.batch_policy}")

    # Create DRL allocation policy (if requested)
    drl_policy = None
    if config.resource_allocation_mode == "drl":
        import pickle
        from resources.drl_allocation.state import DRLStateBuilder
        from resources.drl_allocation.policy import DRLAllocationPolicy

        model_path = config.drl_model_path
        config_path = os.path.join(os.path.dirname(model_path), "state_builder_config.pkl")

        with open(config_path, "rb") as f:
            sb_config = pickle.load(f)

        state_builder = DRLStateBuilder(
            activity_list=sb_config["activity_list"],
            role_groups=sb_config["role_groups"],
            resource_to_role=sb_config["resource_to_role"],
            activity_to_roles=sb_config["activity_to_roles"],
        )

        drl_policy = DRLAllocationPolicy(
            model_path=model_path,
            state_builder=state_builder,
            deterministic=config.drl_deterministic,
        )
        print(f"Created DRL policy from: {model_path}")

    # Create PMSP config (if requested)
    pmsp_config = None
    if config.resource_allocation_mode == "pmsp":
        from resources.resource_optimization.resource_optimization import SelectionConfig
        pmsp_config = SelectionConfig(
            mode="pmsp",
            dummy_delta=config.pmsp_dummy_delta,
            pmsp_solver_time_limit_seconds=config.pmsp_solver_time_limit_seconds,
            prediction_batch_size=config.pmsp_prediction_batch_size,
            optimization_batch_size=config.pmsp_optimization_batch_size,
        )
        print(f"Created PMSP config (delta={config.pmsp_dummy_delta})")

    # Prepare output directory and CSV path for incremental writing
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "simulated_log.csv")

    engine = DESEngine(
        resource_allocator=allocator,
        arrival_timestamps=arrivals,
        next_activity_predictor=next_act_pred,  # May be None for auto-load or if delegated
        next_activity_predictor_type=pred_type,  # Explicit type trigger if predictor is None
        next_activity_config={
            'temperature': config.next_activity_temperature,
            'pt_max_duration_seconds': config.pt_max_duration_seconds,
        },
        pt_lifecycle_mode=config.pt_lifecycle_mode,
        processing_time_predictor=proc_pred,
        case_attribute_predictor=attr_pred,
        start_time=engine_start_time,
        resource_selection_strategy=resource_strategy,
        batch_allocation_policy=batch_policy,
        processing_time_estimator=pt_estimator,
        drl_policy=drl_policy,
        drl_max_postpone_wait_hours=config.drl_max_postpone_wait_hours,
        pmsp_config=pmsp_config,
        enable_profiling=enable_profiling,
        incremental_csv_path=csv_path,  # Enable incremental CSV writing every 100 cases
    )

    # Run simulation
    print("\nRunning simulation...")
    events = engine.run(num_cases=len(arrivals))

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Cases started: {engine.stats['cases_started']}")
    print(f"  Cases completed: {engine.stats['cases_completed']}")
    print(f"  Events generated: {len(events)}")
    print(f"  Outside hours: {engine.stats['outside_hours_count']}")
    print(f"  No eligible: {engine.stats['no_eligible_failures']}")
    print("=" * 60)

    if batch_policy is not None and hasattr(batch_policy, 'print_diagnostics_summary'):
        batch_policy.print_diagnostics_summary()

    # Export results (CSV path already defined above for incremental writing)
    xes_path = os.path.join(output_dir, "simulated_log.xes")
    
    # Note: If incremental_csv_path was set, events are already written incrementally.
    # This final export ensures all events are in the file (including any remaining ones).
    if not hasattr(engine, '_incremental_csv_path') or engine._incremental_csv_path != csv_path:
        LogExporter.to_csv(events, csv_path)
        print(f"\nExported CSV to: {csv_path}")
    else:
        print(f"\nCSV already written incrementally to: {csv_path}")

    try:
        LogExporter.to_xes(events, xes_path)
        print(f"Exported XES to: {xes_path}")
    except Exception as e:
        print(f"Could not export XES: {e}")

    # Show sample events
    print("\nSample events (first 5):")
    for e in events[:5]:
        ts = e['time:timestamp'].strftime('%Y-%m-%d %H:%M')
        print(f"  [{ts}] {e['case:concept:name']}: {e['concept:name']} (by {e['org:resource']})")

    return events


def main():
    parser = argparse.ArgumentParser(description="Run integration test for simulation engine")
    parser.add_argument(
        "--arrivals",
        choices=["basic", "advanced"],
        default="advanced",
        help="Case arrival mode (default: advanced)"
    )
    parser.add_argument(
        "--processing",
        choices=["basic", "advanced"],
        default="basic",
        help="Processing time mode (default: basic)"
    )
    parser.add_argument(
        "--processing-model-path",
        default=None,
        help="Path to processing time model (base path without suffixes)"
    )
    parser.add_argument(
        "--processing-time-method",
        choices=["distribution", "ml", "probabilistic_ml"],
        default="probabilistic_ml",
        help="Processing time prediction method (default: probabilistic_ml)"
    )
    parser.add_argument(
        "--next-activity",
        choices=["lstm", "process_transformer", "lifecycle_dual_full_baseline", "lifecycle_dual_start_complete_baseline"],
        default="lstm",
        help="Next activity predictor implementation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Sampling temperature for next activity prediction (process_transformer only)"
    )
    parser.add_argument(
        "--pt-lifecycle-mode",
        choices=["native", "gt_activity_gated"],
        default="native",
        help="PT-only lifecycle logging mode: native predictor output, or GT activity-gated synthetic starts"
    )
    parser.add_argument(
        "--pt-max-duration-days",
        type=float,
        default=30.0,
        help="Max PT duration cap in days (prevents outlier durations from cascading queue buildup, default: 30)"
    )
    parser.add_argument(
        "--event-log",
        default="Dataset/BPI Challenge 2017.xes",
        help="Path to event log file"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=None,
        help="Number of cases to simulate (default: same as original log)"
    )
    parser.add_argument(
        "--output-dir",
        default="integration/output",
        help="Output directory for simulated log"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--resource-strategy",
        choices=["random", "round_robin", "shortest_queue"],
        default="random",
        help="Resource selection heuristic (R-RMA=random, R-RRA=round_robin, R-SHQ=shortest_queue)"
    )
    parser.add_argument(
        "--resource-allocation-mode",
        choices=["greedy", "batch", "drl", "pmsp"],
        default="greedy",
        help="Resource allocation mode (greedy=per-task heuristic, batch=MILP-based 1-Batch-1, drl=trained PPO, pmsp=PMSP optimizer)"
    )
    parser.add_argument(
        "--pmsp-dummy-delta",
        type=float,
        default=1.5,
        help="PMSP dummy cost multiplier delta (default: 1.5)"
    )
    parser.add_argument(
        "--pmsp-solver-time-limit",
        type=float,
        default=2.0,
        help="PMSP CP-SAT solver time limit in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--pmsp-prediction-batch-size",
        type=int,
        default=25,
        help="PMSP max predictions per task (0=unlimited, default: 25)"
    )
    parser.add_argument(
        "--pmsp-optimization-batch-size",
        type=int,
        default=0,
        help="PMSP min waiting tasks to trigger optimization (0=always optimize, default: 0)"
    )
    parser.add_argument(
        "--drl-model-path",
        default="models/drl_allocation/drl_allocation_model",
        help="Path to trained DRL model (for --resource-allocation-mode drl)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling of simulation components"
    )

    args = parser.parse_args()

    if args.pt_lifecycle_mode == "gt_activity_gated" and args.next_activity != "process_transformer":
        raise ValueError(
            "--pt-lifecycle-mode=gt_activity_gated is only valid with "
            "--next-activity process_transformer. "
            "Use --pt-lifecycle-mode native for non-PT predictors."
        )

    # Setup logging
    # Note: many callers redirect stdout to a file. Python logging defaults to stderr,
    # so we explicitly log to stdout to make PMSP/optimizer messages visible.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    # Load event log
    print(f"Loading event log from: {args.event_log}")
    df = load_event_log(args.event_log)

    # Determine number of cases
    num_cases = args.num_cases
    if num_cases is None:
        num_cases = df['case:concept:name'].nunique() #
        print(f"Simulating {num_cases} cases (same as original log)")

    # FILTERING FOR PROCESS TRANSFORMER V2
    # The PT v2 model was trained ONLY on start and complete events.
    # To ensure fair comparison and correct input, we filter the log.
    if args.next_activity == "process_transformer":
        if 'lifecycle:transition' in df.columns:
            # Case-insensitive check for start/complete
            mask = df['lifecycle:transition'].astype(str).str.lower().isin(['start', 'complete'])
            df = df[mask].copy()

    # Create configuration from individual flags
    config = SimulationConfig(
        processing_time_mode=args.processing,
        case_arrival_mode=args.arrivals,
        event_log_path=args.event_log,
        num_cases=num_cases,
        verbose=args.verbose,
    )

    if args.processing_model_path:
        config.processing_time_model_path = args.processing_model_path

    config.processing_time_method = args.processing_time_method

    # Map CLI next-activity choice to config fields
    config.next_activity_temperature = args.temperature
    config.pt_lifecycle_mode = args.pt_lifecycle_mode
    config.pt_max_duration_seconds = args.pt_max_duration_days * 24 * 3600
    if args.next_activity == "lifecycle_dual_start_complete_baseline":
        config.next_activity_class = "lifecycle_dual"
        config.next_activity_lifecycle_variant = "start_complete"
        config.next_activity_model_path = "next_activity_prediction_lifecycle_dual/models/start_complete/baseline"
    elif args.next_activity == "lifecycle_dual_full_baseline":
        config.next_activity_class = "lifecycle_dual"
        config.next_activity_lifecycle_variant = "full_lifecycle"
        config.next_activity_model_path = "next_activity_prediction_lifecycle_dual/models/full_lifecycle/baseline"
    else:
        config.next_activity_class = args.next_activity  # "lstm" or "process_transformer"

    config.num_cases = num_cases
    config.resource_selection_strategy = args.resource_strategy
    config.resource_allocation_mode = args.resource_allocation_mode
    if hasattr(args, 'drl_model_path') and args.drl_model_path:
        config.drl_model_path = args.drl_model_path
    if args.resource_allocation_mode == "pmsp":
        config.pmsp_dummy_delta = args.pmsp_dummy_delta
        config.pmsp_solver_time_limit_seconds = args.pmsp_solver_time_limit
        config.pmsp_prediction_batch_size = args.pmsp_prediction_batch_size
        config.pmsp_optimization_batch_size = args.pmsp_optimization_batch_size

    # Create resource allocator
    allocator = create_resource_allocator(args.event_log)

    # Save ground truth subset for comparison
    print(f"\nSaving ground truth subset ({num_cases} cases) for comparison...")
    save_ground_truth_subset(df, num_cases, args.output_dir)

    # Run simulation
    events = run_simulation(config, df, allocator, args.output_dir, enable_profiling=args.profile)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
