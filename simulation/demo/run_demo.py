#!/usr/bin/env python3
"""
Live Demo Script for Process Simulation Engine

A clean presentation demo showing:
1. Modular component loading
2. Simulation execution
3. Process model discovery and visualization

Usage:
    python -m simulation.demo.run_demo [--num-cases 50]
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# =============================================================================
# LOGGING & WARNING SUPPRESSION
# =============================================================================

# Suppress TensorFlow and other warnings for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'

import warnings
warnings.filterwarnings('ignore')

# Suppress all logging below WARNING level
import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('simulation').setLevel(logging.WARNING)
logging.getLogger('process_transformer').setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# VISUAL OUTPUT HELPERS
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def print_banner():
    """Print demo banner."""
    banner = f"""
{Colors.BOLD}{Colors.CYAN}
  ____  ____  ____   ___  ____   __      ____  _____ ____
 | __ )|  _ \\/ ___| / _ \\|___ \\ / /_    |  _ \\| ____/ ___|
 |  _ \\| |_) \\___ \\| | | | __) | '_ \\   | | | |  _| \\___ \\
 | |_) |  __/ ___) | |_| |/ __/| (_) |  | |_| | |___ ___) |
 |____/|_|   |____/ \\___/|_____|\\___/   |____/|_____|____/
{Colors.RESET}
{Colors.DIM}   Process Simulation Live Environment{Colors.RESET}
"""
    print(banner)


def print_section(title: str, icon: str = ""):
    """Print a section header."""
    width = 60
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {icon}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * width}{Colors.RESET}")


def print_component(name: str, status: str = "loading"):
    """Print component status."""
    icons = {
        "loading": f"{Colors.YELLOW}[...]",
        "done": f"{Colors.GREEN}[OK]",
        "skip": f"{Colors.DIM}[--]",
    }
    icon = icons.get(status, icons["loading"])
    print(f"  {icon}{Colors.RESET} {name}")


def print_stat(label: str, value, unit: str = ""):
    """Print a statistic."""
    print(f"  {Colors.CYAN}{label}:{Colors.RESET} {Colors.BOLD}{value}{Colors.RESET} {unit}")


def print_progress(current: int, total: int, label: str = "Progress"):
    """Print inline progress bar."""
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    pct = 100 * current / total
    print(f"\r  {Colors.CYAN}{label}:{Colors.RESET} [{bar}] {pct:5.1f}%", end='', flush=True)
    if current >= total:
        print()  # newline when done


# =============================================================================
# DEMO COMPONENTS
# =============================================================================

def suppress_library_output():
    """Suppress noisy output from libraries."""
    import logging
    # Silence all loggers
    for name in ['simulation', 'simulation.engine', 'process_transformer',
                 'resources', 'tensorflow', 'absl', 'pm4py']:
        logging.getLogger(name).setLevel(logging.ERROR)


def load_components(num_cases: int):
    """Load all simulation components with visual feedback."""
    suppress_library_output()
    print_section("LOADING MODULAR COMPONENTS", "")

    # 1. Resource Allocator (suppress its verbose output)
    print_component("Resource Allocator (Permissions + Availability)", "loading")
    import io
    import contextlib
    from resources import ResourceAllocator
    with contextlib.redirect_stdout(io.StringIO()):
        allocator = ResourceAllocator(log_path="Dataset/BPI Challenge 2017.xes")
    # Count resources from availability model if available
    n_resources = len(allocator.availability._patterns) if hasattr(allocator.availability, '_patterns') else "cached"
    print_component(f"Resource Allocator: {n_resources} resource patterns", "done")

    # 2. Processing Time Predictor
    print_component("Processing Time Predictor (ML-based)", "loading")
    from processing_time_prediction.ProcessingTimePredictionClass import ProcessingTimePredictionClass
    with contextlib.redirect_stdout(io.StringIO()):
        proc_time = ProcessingTimePredictionClass(method="ml", model_path="models/processing_time_model")
    print_component("Processing Time Predictor: ML ensemble model", "done")

    # 3. Case Attribute Generator
    print_component("Case Attribute Generator (Statistical)", "loading")
    # Add path for case attribute module
    advanced_path = project_root / "Instance Spawn Rate" / "Advanced"
    if str(advanced_path) not in sys.path:
        sys.path.insert(0, str(advanced_path))
    from case_attribute_prediction.simulator import AttributeSimulationEngine
    attr_gen = AttributeSimulationEngine(df=None, seed=42)  # Uses cached artifacts
    print_component("Case Attribute Generator: LoanGoal, ApplicationType, Amount", "done")

    # 4. Case Arrival Generator (basic for demo speed)
    print_component("Case Arrival Generator (Basic)", "loading")
    from datetime import timedelta
    import random
    rng = random.Random(42)
    start_date = datetime(2016, 1, 4, 8, 0)  # Monday 8am
    arrivals = []
    current_time = start_date
    for _ in range(num_cases):
        current_time += timedelta(minutes=rng.randint(5, 45))
        arrivals.append(current_time)
    print_component(f"Case Arrival Generator: {num_cases} arrival timestamps", "done")

    # 5. Next Activity Predictor
    print_component("Next Activity Predictor (Process Transformer)", "loading")
    # Add Next-Activity-Prediction paths
    na_root = project_root / "Next-Activity-Prediction"
    if str(na_root) not in sys.path:
        sys.path.insert(0, str(na_root))
    from simulation.engine import NextActivityPredictorType
    pred_type = NextActivityPredictorType.PROCESS_TRANSFORMER
    print_component("Next Activity Predictor: Process Transformer (neural)", "done")

    return {
        "allocator": allocator,
        "proc_time": proc_time,
        "attr_gen": attr_gen,
        "arrivals": arrivals,
        "pred_type": pred_type,
        "start_date": start_date,
    }


def run_simulation(components: dict, num_cases: int):
    """Run the simulation with progress display."""
    print_section("RUNNING SIMULATION", "")

    from simulation.engine import DESEngine

    # Create engine
    print(f"  {Colors.DIM}Initializing DES Engine...{Colors.RESET}")
    engine = DESEngine(
        resource_allocator=components["allocator"],
        arrival_timestamps=components["arrivals"],
        next_activity_predictor_type=components["pred_type"],
        processing_time_predictor=components["proc_time"],
        case_attribute_predictor=components["attr_gen"],
        start_time=components["start_date"],
    )

    # Run simulation
    print(f"  {Colors.DIM}Simulating {num_cases} cases...{Colors.RESET}\n")
    start_time = time.time()
    events = engine.run(num_cases=num_cases)
    elapsed = time.time() - start_time

    # Print results
    print()
    print_stat("Cases started", engine.stats['cases_started'])
    print_stat("Cases completed", engine.stats['cases_completed'])
    print_stat("Events generated", len(events))
    print_stat("Simulation time", f"{elapsed:.2f}", "seconds")

    # Sample trace - find a case with a good number of events
    print(f"\n  {Colors.DIM}Sample trace:{Colors.RESET}")
    from collections import Counter
    case_counts = Counter(e['case:concept:name'] for e in events)
    # Pick a case with a decent number of events (not too few, not too many)
    best_case = None
    for case_id, count in case_counts.most_common():
        if 5 <= count <= 15:
            best_case = case_id
            break
    if not best_case:
        best_case = case_counts.most_common(1)[0][0] if case_counts else None

    if best_case:
        all_case_events = [e for e in events if e['case:concept:name'] == best_case]
        case_events = all_case_events[:7]
        for e in case_events:
            ts = e['time:timestamp'].strftime('%m-%d %H:%M')
            act = e['concept:name'][:30].ljust(30)
            res = (e['org:resource'][:10] if e['org:resource'] else 'system').ljust(10)
            print(f"    {Colors.DIM}[{ts}]{Colors.RESET} {act} {Colors.DIM}by{Colors.RESET} {res}")
        if len(case_events) < len(all_case_events):
            print(f"    {Colors.DIM}... +{len(all_case_events) - len(case_events)} more{Colors.RESET}")

    return events


def mine_and_visualize(events: list, output_dir: str):
    """Mine process model and create visualization."""
    print_section("PROCESS DISCOVERY & VISUALIZATION", "")

    import pandas as pd
    import pm4py
    from simulation.log_exporter import LogExporter

    # Convert to DataFrame
    print(f"  {Colors.DIM}Converting to event log...{Colors.RESET}")
    df = LogExporter.to_dataframe(events)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save event log
    csv_path = os.path.join(output_dir, "simulated_log.csv")
    df.to_csv(csv_path, index=False)
    print_component(f"Event log saved: {csv_path}", "done")

    # Mine DFG
    print(f"  {Colors.DIM}Mining Directly-Follows Graph...{Colors.RESET}")
    dfg, start_activities, end_activities = pm4py.discover_dfg(df)

    # Calculate statistics
    num_activities = len(set(df['concept:name']))
    num_transitions = len(dfg)
    print_stat("Unique activities", num_activities)
    print_stat("DFG transitions", num_transitions)

    # Visualize and save
    print(f"\n  {Colors.DIM}Generating visualization...{Colors.RESET}")

    # Save DFG as image
    dfg_path = os.path.join(output_dir, "process_dfg.png")
    pm4py.save_vis_dfg(dfg, start_activities, end_activities, dfg_path)
    print_component(f"DFG image saved: {dfg_path}", "done")

    # Display the DFG
    print(f"\n  {Colors.BOLD}Opening visualization...{Colors.RESET}")
    try:
        pm4py.view_dfg(dfg, start_activities, end_activities)
        print_component("Process model displayed", "done")
    except Exception as e:
        print(f"  {Colors.YELLOW}Note: Could not auto-display. Open {dfg_path} manually.{Colors.RESET}")

    return dfg_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process Simulation Engine Demo")
    parser.add_argument("--num-cases", type=int, default=50, help="Number of cases to simulate")
    parser.add_argument("--output-dir", default="simulation/demo/output", help="Output directory")
    args = parser.parse_args()

    # Banner
    print_banner()

    print(f"  {Colors.DIM}Demo configuration:{Colors.RESET}")
    print_stat("Cases to simulate", args.num_cases)
    print_stat("Output directory", args.output_dir)

    # Load components
    components = load_components(args.num_cases)

    # Run simulation
    events = run_simulation(components, args.num_cases)

    # Mine and visualize
    if events:
        mine_and_visualize(events, args.output_dir)
    else:
        print(f"  {Colors.RED}No events generated!{Colors.RESET}")


if __name__ == "__main__":
    main()
