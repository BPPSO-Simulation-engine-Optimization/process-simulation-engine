"""
Integration test for the simulation engine with all prediction components.

This script runs a full simulation using:
- Case arrivals: basic (stub) or advanced (CaseInterarrivalPipeline)
- Processing times: basic (stub) or advanced (ProcessingTimePredictionClass)
- Case attributes: basic (stub) or advanced (AttributeSimulationEngine)

+ It saves the respective subset of the GT EL in case num-cases is specified

Usage:
    python -m integration.test_integration --mode basic
    python -m integration.test_integration --mode advanced --num-cases 31000
    python -m integration.test_integration --mode mixed --arrivals advanced --attributes basic
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
from simulation.engine import DESEngine
from simulation.log_exporter import LogExporter


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


def run_simulation(config: SimulationConfig, df: pd.DataFrame, allocator, output_dir: str):
    """Run the simulation with given configuration."""
    print("\n" + "=" * 60)
    print("SIMULATION CONFIGURATION")
    print("=" * 60)
    print(f"  Processing time mode: {config.processing_time_mode}")
    print(f"  Case arrival mode: {config.case_arrival_mode}")
    print(f"  Case attribute mode: {config.case_attribute_mode}")
    print(f"  Number of cases: {config.num_cases}")
    print("=" * 60 + "\n")

    # Get start date from event log
    if 'time:timestamp' in df.columns:
        start_date = pd.to_datetime(df['time:timestamp']).min().to_pydatetime()
    else:
        start_date = datetime(2016, 1, 4, 8, 0)
    print(f"Simulation start date: {start_date}")

    # Setup predictors
    print("\nSetting up predictors...")
    arrivals, proc_pred, attr_pred = setup_simulation(
        config,
        df=df if config.case_arrival_mode == "advanced" or config.case_attribute_mode == "advanced" else None,
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

    engine = DESEngine(
        resource_allocator=allocator,
        arrival_timestamps=arrivals,
        processing_time_predictor=proc_pred,
        case_attribute_predictor=attr_pred,
        start_time=engine_start_time,
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

    # Export results
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "simulated_log.csv")
    xes_path = os.path.join(output_dir, "simulated_log.xes")

    LogExporter.to_csv(events, csv_path)
    print(f"\nExported CSV to: {csv_path}")

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
        "--mode",
        choices=["basic", "advanced", "mixed"],
        default="basic",
        help="Simulation mode (basic=all stubs, advanced=all ML, mixed=custom)"
    )
    parser.add_argument(
        "--arrivals",
        choices=["basic", "advanced"],
        default=None,
        help="Case arrival mode (for mixed mode)"
    )
    parser.add_argument(
        "--processing",
        choices=["basic", "advanced"],
        default=None,
        help="Processing time mode (for mixed mode)"
    )
    parser.add_argument(
        "--attributes",
        choices=["basic", "advanced"],
        default=None,
        help="Case attribute mode (for mixed mode)"
    )
    parser.add_argument(
        "--event-log",
        default="eventlog/eventlog.xes.gz",
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

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(message)s')

    # Load event log
    print(f"Loading event log from: {args.event_log}")
    df = load_event_log(args.event_log)

    # Determine number of cases
    num_cases = args.num_cases
    if num_cases is None:
        num_cases = df['case:concept:name'].nunique() #
        print(f"Simulating {num_cases} cases (same as original log)")

    # Create configuration
    if args.mode == "basic":
        config = SimulationConfig.all_basic()
    elif args.mode == "advanced":
        config = SimulationConfig.all_advanced(
            event_log_path=args.event_log,
            num_cases=num_cases,
        )
    else:  # mixed
        config = SimulationConfig(
            processing_time_mode=args.processing or "basic",
            case_arrival_mode=args.arrivals or "basic",
            case_attribute_mode=args.attributes or "basic",
            event_log_path=args.event_log,
            num_cases=num_cases,
            verbose=args.verbose,
        )

    config.num_cases = num_cases

    # Create resource allocator
    allocator = create_resource_allocator(args.event_log)

    # Save ground truth subset for comparison
    print(f"\nSaving ground truth subset ({num_cases} cases) for comparison...")
    save_ground_truth_subset(df, num_cases, args.output_dir)

    # Run simulation
    events = run_simulation(config, df, allocator, args.output_dir)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
