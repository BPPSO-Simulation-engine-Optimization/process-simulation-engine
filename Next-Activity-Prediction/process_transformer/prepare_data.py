"""
Prepare BPIC17 data for ProcessTransformer training.

Usage:
    python prepare_data.py --input eventlog/eventlog.xes.gz --output data/bpic17_pt.csv
"""

import pandas as pd
import pm4py
from pathlib import Path


def prepare_bpic17_for_process_transformer(
    input_path: str,
    output_path: str,
    lifecycle_filter: str = "complete",
    min_case_length: int = 10,#based on GT log
    max_case_length: int = 180, #based on GT log
) -> pd.DataFrame:
    """
    Transform BPIC17 to ProcessTransformer format.

    Args:
        input_path: Path to XES/XES.GZ file
        output_path: Path for output CSV
        lifecycle_filter: Only include events with this lifecycle (None = all)
        min_case_length: Exclude cases shorter than this
        max_case_length: Truncate cases longer than this

    Returns:
        Prepared DataFrame
    """
    # Load event log
    log = pm4py.read_xes(input_path)
    df = pm4py.convert_to_dataframe(log)

    print(f"Loaded: {len(df)} events, {df['case:concept:name'].nunique()} cases")

    # Filter by lifecycle if specified
    if lifecycle_filter:
        df = df[df['lifecycle:transition'] == lifecycle_filter]
        print(f"After lifecycle filter: {len(df)} events")

    # Sort by case and timestamp
    df = df.sort_values(['case:concept:name', 'time:timestamp'])

    # Calculate case lengths
    case_lengths = df.groupby('case:concept:name').size()

    # Filter by case length
    valid_cases = case_lengths[
        (case_lengths >= min_case_length) &
        (case_lengths <= max_case_length)
    ].index
    df = df[df['case:concept:name'].isin(valid_cases)]
    print(f"After length filter: {len(df)} events, {len(valid_cases)} cases")

    # Truncate long cases (keep first max_case_length events)
    df = df.groupby('case:concept:name').head(max_case_length)

    # Select and rename columns
    result = df[['case:concept:name', 'concept:name', 'time:timestamp']].copy()
    result.columns = ['Case ID', 'Activity', 'Complete Timestamp']

    # Ensure timestamp format
    result['Complete Timestamp'] = pd.to_datetime(result['Complete Timestamp'])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"Final: {len(result)} events, {result['Case ID'].nunique()} cases")
    print(f"Activities: {result['Activity'].nunique()} unique")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eventlog/eventlog.xes.gz")
    parser.add_argument("--output", default="data/bpic17_pt.csv")
    parser.add_argument("--lifecycle", default="complete")
    args = parser.parse_args()

    prepare_bpic17_for_process_transformer(
        args.input,
        args.output,
        lifecycle_filter=args.lifecycle,
    )

