"""
Run simulation benchmark comparing ground truth vs simulated logs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from integration.SimulationBenchmark import SimulationBenchmark

def main():
    # Paths to logs
    ground_truth = project_root / "integration" / "output" / "ground_truth_log.xes"
    simulated = project_root / "integration" / "output" / "simulated_log.xes"
    
    # Check if files exist
    if not ground_truth.exists():
        print(f"ERROR: Ground truth log not found: {ground_truth}")
        print("Run simulation first: python -m integration.test_integration --mode advanced")
        return
    
    if not simulated.exists():
        print(f"ERROR: Simulated log not found: {simulated}")
        print("Run simulation first: python -m integration.test_integration --mode advanced")
        return
    
    print("=" * 60)
    print("SIMULATION BENCHMARK")
    print("=" * 60)
    print(f"Ground truth: {ground_truth}")
    print(f"Simulated: {simulated}")
    print("=" * 60)
    print()
    
    # Create benchmark
    benchmark = SimulationBenchmark(
        str(ground_truth),
        str(simulated)
    )
    
    # Compute all metrics
    print("Computing metrics...")
    results = benchmark.compute_all_metrics()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    benchmark.print_summary()
    
    # Export results
    output_path = project_root / "integration" / "output" / "benchmark_results.xlsx"
    print(f"\nExporting results to: {output_path}")
    benchmark.export_results(str(output_path))
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

