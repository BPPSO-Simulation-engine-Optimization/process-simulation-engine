"""
Test script for ProcessingTimePredictionClass distribution fitting.
Loads an XES file via pm4py and tests the distribution fitting functionality.
"""

import pm4py
import pandas as pd
from processing_time_prediction.ProcessingTimePredictionClass import ProcessingTimePredictionClass
import sys
import os


def load_xes_file(file_path: str) -> pd.DataFrame:
    """
    Load an XES file and convert it to a pandas DataFrame.
    
    Args:
        file_path: Path to the XES file
        
    Returns:
        DataFrame with event log data
    """
    print(f"Loading XES file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XES file not found: {file_path}")
    
    # Read XES file using pm4py
    log = pm4py.read_xes(file_path)
    
    # Convert to DataFrame
    df = pm4py.convert_to_dataframe(log)
    
    print(f"Loaded {len(df)} events from {df['case:concept:name'].nunique()} cases")
    print(f"Columns: {list(df.columns)}")
    
    return df


def test_distribution_fitting(df: pd.DataFrame):
    """
    Test the ProcessingTimePredictionClass distribution fitting.
    
    Args:
        df: DataFrame with event log data
    """
    print("\n" + "="*80)
    print("Testing ProcessingTimePredictionClass distribution fitting")
    print("="*80)
    
    # Initialize the predictor (this will automatically fit distributions)
    print("\nInitializing ProcessingTimePredictionClass...")
    predictor = ProcessingTimePredictionClass(df, method="distribution", min_observations=2)
    
    # Get distribution information
    print("\n" + "-"*80)
    print("Distribution Statistics")
    print("-"*80)
    dist_info = predictor.get_distribution_info()
    
    print(f"\nTotal number of fitted distributions: {dist_info['num_distributions']}")
    print(f"Fallback mean: {dist_info['fallback_mean']:.2f} seconds ({dist_info['fallback_mean']/3600:.2f} hours)")
    print(f"Fallback std: {dist_info['fallback_std']:.2f} seconds ({dist_info['fallback_std']/3600:.2f} hours)")
    
    # Show some example distributions
    print("\n" + "-"*80)
    print("Example Distributions (first 10)")
    print("-"*80)
    
    example_distributions = list(dist_info['distributions'].items())[:10]
    for i, (transition, stats) in enumerate(example_distributions, 1):
        print(f"\n{i}. Transition: {transition}")
        print(f"   Count: {stats['count']} observations")
        print(f"   Mean: {stats['mean']:.2f}s ({stats['mean']/3600:.2f}h)")
        print(f"   Median: {stats['median']:.2f}s ({stats['median']/3600:.2f}h)")
        print(f"   Std: {stats['std']:.2f}s ({stats['std']/3600:.2f}h)")
        print(f"   Log-normal params: mu={stats['mu']:.4f}, sigma={stats['sigma']:.4f}")
    
    # Test prediction on some transitions
    print("\n" + "-"*80)
    print("Testing Predictions")
    print("-"*80)
    
    # Get some example transitions from the data (use the actual tuple keys from predictor)
    test_transitions = list(predictor.distributions.keys())[:5]
    
    # Debug: print transition types and lengths
    if test_transitions:
        print(f"\nDebug: First transition type: {type(test_transitions[0])}, length: {len(test_transitions[0]) if isinstance(test_transitions[0], tuple) else 'N/A'}, value: {test_transitions[0]}")
    
    for transition in test_transitions:
        # Handle transition - ensure it's a tuple and extract first 4 elements
        if isinstance(transition, tuple):
            if len(transition) >= 4:
                prev_activity, prev_lifecycle, curr_activity, curr_lifecycle = transition[:4]
            else:
                print(f"Warning: Skipping transition with insufficient elements: {transition}")
                continue
        elif isinstance(transition, str):
            # If it's a string representation, try to evaluate it
            import ast
            try:
                transition_tuple = ast.literal_eval(transition)
                if isinstance(transition_tuple, tuple) and len(transition_tuple) >= 4:
                    prev_activity, prev_lifecycle, curr_activity, curr_lifecycle = transition_tuple[:4]
                else:
                    print(f"Warning: Skipping invalid transition format: {transition}")
                    continue
            except:
                print(f"Warning: Skipping invalid transition format: {transition}")
                continue
        else:
            print(f"Warning: Skipping invalid transition format: {transition} (type: {type(transition)})")
            continue
        
        # Make 5 predictions
        predictions = []
        for _ in range(5):
            pred = predictor.predict(
                prev_activity, prev_lifecycle,
                curr_activity, curr_lifecycle
            )
            predictions.append(pred)
        
        # Use the 4-tuple key to look up stats
        transition_key = (prev_activity, prev_lifecycle, curr_activity, curr_lifecycle)
        stats_info = predictor.distributions.get(transition_key)
        if stats_info is None:
            # Try with the original transition if it's a valid key
            if transition in predictor.distributions:
                stats_info = predictor.distributions[transition]
            else:
                print(f"Warning: Could not find stats for transition: {transition_key}")
                continue
        
        print(f"\nTransition: {prev_activity} ({prev_lifecycle}) -> {curr_activity} ({curr_lifecycle})")
        print(f"  True mean: {stats_info['mean']:.2f}s")
        print(f"  Predictions: {[f'{p:.2f}s' for p in predictions]}")
        print(f"  Prediction avg: {sum(predictions)/len(predictions):.2f}s")
    
    # Test fallback prediction (for unseen transition)
    print("\n" + "-"*80)
    print("Testing Fallback Prediction (unseen transition)")
    print("-"*80)
    
    fallback_pred = predictor.predict(
        "Unknown_Activity", "complete",
        "Another_Unknown", "start"
    )
    print(f"Prediction for unseen transition: {fallback_pred:.2f}s ({fallback_pred/3600:.2f}h)")
    print(f"Used fallback mean: {predictor.fallback_mean:.2f}s")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


def main():
    
    try:
        df = load_xes_file("Dataset/BPI Challenge 2017.xes")
        test_distribution_fitting(df)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
