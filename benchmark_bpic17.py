"""
Benchmark simulated log against 100 sample cases from BPIC 2017 event log.

Filters the original log to only include start/complete lifecycles,
samples 100 cases, and compares against the simulated log.

Usage:
    1. First, run a simulation to generate a simulated log:
       python integration/test_integration.py --mode advanced --num-cases 100
    
    2. Then run this benchmark:
       python benchmark_bpic17.py

The benchmark compares:
    - Original: 100 randomly sampled cases from BPIC 2017 (start/complete lifecycles only)
    - Simulated: Your simulated event log

Metrics computed:
    - Basic statistics (events, cases, activities, resources)
    - Events per case distribution
    - Case throughput times (cycle times)
    - Case arrival and completion patterns
    - Control flow (directly-follows graph)
    - Trace variant distributions
    - Overall activity frequency distribution
    - Start and end activity distributions
    - Activity durations
    - Simple similarity metrics (Jaccard similarity, overlap percentages)
    - Resource statistics (if available)

Results are exported to: integration/output/bpic17_benchmark_results.xlsx
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import pm4py
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("ERROR: pm4py is required. Install with: pip install pm4py")
    sys.exit(1)

from integration.SimulationBenchmark import SimulationBenchmark
from next_activity_prediction.data_preprocessing import filter_lifecycles


def load_bpic17_sample(log_path: str, num_cases: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Load BPIC 2017 event log, filter to start/complete lifecycles, and sample cases.
    
    Args:
        log_path: Path to BPIC 2017 event log file
        num_cases: Number of cases to sample
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with filtered and sampled event log
    """
    if not Path(log_path).exists():
        raise FileNotFoundError(f"Event log not found: {log_path}")
    
    print(f"Loading BPIC 2017 event log from {log_path}...")
    
    # Load event log
    if log_path.endswith('.xes') or log_path.endswith('.xes.gz'):
        log = pm4py.read_xes(log_path)
        df = pm4py.convert_to_dataframe(log)
    elif log_path.endswith('.csv') or log_path.endswith('.csv.gz'):
        df = pd.read_csv(log_path)
    else:
        raise ValueError(f"Unsupported file format: {log_path}")
    
    print(f"Loaded {len(df):,} events, {df['case:concept:name'].nunique():,} cases")
    
    # Filter to start/complete lifecycles
    print("\nFiltering to start/complete lifecycles...")
    df = filter_lifecycles(df)
    
    # Sample cases
    all_cases = df['case:concept:name'].unique()
    np.random.seed(seed)
    sampled_cases = np.random.choice(all_cases, size=min(num_cases, len(all_cases)), replace=False)
    
    df_sampled = df[df['case:concept:name'].isin(sampled_cases)].copy()
    
    print(f"Sampled {len(sampled_cases)} cases")
    print(f"Result: {len(df_sampled):,} events from {df_sampled['case:concept:name'].nunique():,} cases")
    
    return df_sampled


def main():
    """Main benchmarking function."""
    # Paths
    bpic17_log = project_root / "eventlog" / "eventlog.xes.gz"
    simulated_log = project_root / "integration" / "output" / "simulated_log.xes"
    
    # Check if files exist
    if not bpic17_log.exists():
        print(f"ERROR: BPIC 2017 event log not found: {bpic17_log}")
        print("Please ensure the event log is available at this path.")
        return
    
    if not simulated_log.exists():
        print(f"ERROR: Simulated log not found: {simulated_log}")
        print("Run simulation first: python integration/test_integration.py --mode advanced --num-cases 100")
        return
    
    print("=" * 80)
    print("BPIC 2017 BENCHMARK")
    print("=" * 80)
    print(f"Original log: {bpic17_log}")
    print(f"Simulated log: {simulated_log}")
    print("=" * 80)
    print()
    
    # Load and prepare BPIC 2017 sample
    print("Preparing BPIC 2017 sample (100 cases, start/complete only)...")
    bpic17_sample = load_bpic17_sample(str(bpic17_log), num_cases=100, seed=42)
    
    # Create benchmark
    print("\nInitializing benchmark...")
    benchmark = SimulationBenchmark(
        bpic17_sample,  # Original: BPIC 2017 sample (DataFrame)
        str(simulated_log)  # Simulated: from file
    )
    
    # Compute all metrics
    print("\nComputing benchmark metrics...")
    print("This may take a few minutes...")
    results = benchmark.compute_all_metrics()
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    benchmark.print_summary()
    
    # Export results
    output_path = project_root / "integration" / "output" / "bpic17_benchmark_results.xlsx"
    print(f"\nExporting results to: {output_path}")
    benchmark.export_results(str(output_path))
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print("\nThe benchmark compares:")
    print("  - Original: 100 sampled cases from BPIC 2017 (start/complete lifecycles only)")
    print("  - Simulated: Your simulated event log")
    print("\nKey metrics include:")
    print("  - Basic statistics (events, cases, activities)")
    print("  - Events per case distribution")
    print("  - Case throughput times")
    print("  - Control flow (directly-follows graph)")
    print("  - Trace variant distributions")
    print("  - Overall activity frequency distribution")
    print("  - Simple similarity metrics (Jaccard, overlap)")
    print("  - Activity frequencies and durations")
    print("  - Resource statistics (if available)")


if __name__ == "__main__":
    main()

