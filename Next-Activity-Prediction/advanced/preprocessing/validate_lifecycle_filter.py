"""
Validation script for lifecycle filtering.

Run this script to validate the LifecycleFilter implementation:
    python -m preprocessing.validate_lifecycle_filter

Or from the notebooks directory:
    python ../preprocessing/validate_lifecycle_filter.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter

from preprocessing import LifecycleFilter


def load_event_log(xes_path: str) -> pd.DataFrame:
    """Load event log from XES file."""
    print(f"Loading event log: {xes_path}")
    event_log = xes_importer.apply(xes_path)
    df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    print(f"Loaded {len(df)} events from {df['case:concept:name'].nunique()} cases")
    return df


def validate_lifecycle_distribution(lf: LifecycleFilter) -> bool:
    """Validate that lifecycle transitions match expected distribution."""
    print("\n" + "="*60)
    print("1. LIFECYCLE DISTRIBUTION VALIDATION")
    print("="*60)

    dist = lf.get_lifecycle_distribution()
    print("\nLifecycle transition counts:")
    print(dist.to_string())

    # Check expected transitions exist
    expected = {"complete", "start"}
    found = set(dist.index)

    if expected.issubset(found):
        print("\n✓ Found expected 'start' and 'complete' transitions")
        return True
    else:
        missing = expected - found
        print(f"\n✗ Missing expected transitions: {missing}")
        return False


def validate_activity_classification(lf: LifecycleFilter) -> bool:
    """Validate activity classification (which have start events)."""
    print("\n" + "="*60)
    print("2. ACTIVITY CLASSIFICATION VALIDATION")
    print("="*60)

    summary = lf.get_activity_lifecycle_summary()
    print("\nActivity lifecycle summary:")
    print(summary.to_string())

    # Check O_* activities don't have start events
    o_activities = summary[summary["activity"].str.startswith("O_")]
    w_a_activities = summary[~summary["activity"].str.startswith("O_")]

    print(f"\nO_* activities (expected: no start events):")
    o_with_start = o_activities[o_activities["has_start"]]
    if len(o_with_start) == 0:
        print("✓ All O_* activities have NO start events (as expected)")
    else:
        print(f"✗ Found O_* activities WITH start events:")
        print(o_with_start.to_string())
        return False

    print(f"\nW_*/A_* activities (expected: have start events):")
    w_a_with_start = w_a_activities[w_a_activities["has_start"]]
    w_a_without_start = w_a_activities[~w_a_activities["has_start"]]

    print(f"  With start events: {len(w_a_with_start)}")
    print(f"  Without start events: {len(w_a_without_start)}")

    if len(w_a_without_start) > 0:
        print("  Activities without start (will be treated as instant):")
        print(w_a_without_start[["activity", "complete_count"]].to_string())

    return True


def validate_event_pairing(lf: LifecycleFilter, df_filtered: pd.DataFrame) -> bool:
    """Validate that start/complete events are correctly paired."""
    print("\n" + "="*60)
    print("3. EVENT PAIRING VALIDATION")
    print("="*60)

    # Check processing times are non-negative
    negative_times = df_filtered[df_filtered["processing_time"] < 0]
    if len(negative_times) > 0:
        print(f"✗ Found {len(negative_times)} events with negative processing time!")
        print(negative_times.head().to_string())
        return False
    else:
        print("✓ All processing times are non-negative")

    # Check instant activities have 0 duration
    o_events = df_filtered[df_filtered["concept:name"].str.startswith("O_")]
    o_non_zero = o_events[o_events["processing_time"] > 0]
    if len(o_non_zero) > 0:
        print(f"✗ Found {len(o_non_zero)} O_* events with non-zero processing time!")
        return False
    else:
        print(f"✓ All {len(o_events)} O_* events have 0 processing time (instant)")

    # Check work activities have reasonable durations
    work_events = df_filtered[~df_filtered["concept:name"].str.startswith("O_")]
    work_with_time = work_events[work_events["processing_time"] > 0]
    print(f"✓ {len(work_with_time)} work activities have positive processing time")
    print(f"  (out of {len(work_events)} total work activities)")

    return True


def validate_processing_time_stats(lf: LifecycleFilter, df_filtered: pd.DataFrame) -> bool:
    """Validate processing time statistics are reasonable."""
    print("\n" + "="*60)
    print("4. PROCESSING TIME STATISTICS")
    print("="*60)

    stats = lf.get_processing_time_stats(df_filtered)
    print("\nProcessing time statistics per activity (seconds):")
    print(stats.to_string())

    # Check for unreasonably long durations (> 30 days)
    max_reasonable = 30 * 24 * 60 * 60  # 30 days in seconds
    unreasonable = stats[stats["max"] > max_reasonable]
    if len(unreasonable) > 0:
        print(f"\n⚠ Warning: {len(unreasonable)} activities have max duration > 30 days")
        print("  This may indicate pairing issues or genuine long-running activities")

    return True


def validate_sequence_reduction(df_original: pd.DataFrame, df_filtered: pd.DataFrame) -> bool:
    """Validate sequence length reduction after filtering."""
    print("\n" + "="*60)
    print("5. SEQUENCE REDUCTION VALIDATION")
    print("="*60)

    original_count = len(df_original)
    filtered_count = len(df_filtered)
    reduction = (1 - filtered_count / original_count) * 100

    print(f"\nOriginal events: {original_count:,}")
    print(f"Filtered events: {filtered_count:,}")
    print(f"Reduction: {reduction:.1f}%")

    # Check case preservation
    original_cases = df_original["case:concept:name"].nunique()
    filtered_cases = df_filtered["case:concept:name"].nunique()

    if original_cases == filtered_cases:
        print(f"✓ All {original_cases:,} cases preserved")
    else:
        print(f"✗ Cases changed: {original_cases:,} → {filtered_cases:,}")
        return False

    # Check reasonable reduction (expect 40-60% reduction)
    if 30 <= reduction <= 70:
        print(f"✓ Reduction {reduction:.1f}% is in expected range (30-70%)")
    else:
        print(f"⚠ Warning: Reduction {reduction:.1f}% outside expected range")

    return True


def run_validation(xes_path: str) -> dict:
    """Run all validation checks."""
    results = {}

    # Load data
    df_log = load_event_log(xes_path)

    # Initialize filter
    lf = LifecycleFilter(df_log)

    # Run validations
    results["lifecycle_distribution"] = validate_lifecycle_distribution(lf)
    results["activity_classification"] = validate_activity_classification(lf)

    # Filter and collapse
    print("\n" + "="*60)
    print("APPLYING LIFECYCLE FILTER")
    print("="*60)
    df_filtered = lf.transform()

    results["event_pairing"] = validate_event_pairing(lf, df_filtered)
    results["processing_time_stats"] = validate_processing_time_stats(lf, df_filtered)
    results["sequence_reduction"] = validate_sequence_reduction(df_log, df_filtered)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = all(results.values())
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")

    print("\n" + ("✓ ALL VALIDATIONS PASSED" if all_passed else "✗ SOME VALIDATIONS FAILED"))

    return results


if __name__ == "__main__":
    # Default path - adjust if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_xes = os.path.join(script_dir, "..", "..", "..", "Dataset", "BPI Challenge 2017.xes")

    # Allow override via command line
    xes_path = sys.argv[1] if len(sys.argv) > 1 else default_xes

    if not os.path.exists(xes_path):
        print(f"Error: Event log not found at {xes_path}")
        print("Usage: python validate_lifecycle_filter.py [path_to_xes_file]")
        sys.exit(1)

    results = run_validation(xes_path)
    sys.exit(0 if all(results.values()) else 1)
