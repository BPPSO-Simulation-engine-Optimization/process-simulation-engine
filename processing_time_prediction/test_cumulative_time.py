"""
Test script for prefix-based cumulative time prediction.

Usage:
    python test_cumulative_time.py --train              # Train new model
    python test_cumulative_time.py --evaluate           # Evaluate on test traces
    python test_cumulative_time.py --train --evaluate   # Both
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_event_log(path: str) -> pd.DataFrame:
    """Load XES file into DataFrame."""
    print(f"Loading: {path}")
    
    import pm4py
    df = pm4py.read_xes(path)
    
    # Ensure required columns exist
    if "case:concept:name" not in df.columns and "case:@@index" in df.columns:
        df["case:concept:name"] = df["case:@@index"]
    
    print(f"Loaded {len(df)} events, {df['case:concept:name'].nunique()} cases")
    return df


def train_model(df: pd.DataFrame, method: str = "lstm"):
    """Train the model."""
    from ProcessingTimeTrainer import ProcessingTimeTrainer
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "..", "models", f"cumulative_time_{method}")
    
    trainer = ProcessingTimeTrainer(
        df=df,
        method=method,
        max_prefix_length=50,
        epochs=50,
        batch_size=128
    )
    
    trainer.train(save_path=save_path)


def evaluate_model(df: pd.DataFrame, method: str = "lstm", n_traces: int = 100):
    """Evaluate model on random traces."""
    from ProcessingTimePredictionClass import ProcessingTimePrediction
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "models", f"cumulative_time_{method}")
    
    predictor = ProcessingTimePrediction()
    if not predictor.load(model_path):
        print("Model not found. Run with --train first.")
        return
    
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    
    # Sample traces
    case_ids = df["case:concept:name"].unique()
    np.random.seed(42)
    sample_ids = np.random.choice(case_ids, min(n_traces, len(case_ids)), replace=False)
    
    all_predictions = []
    all_actuals = []
    
    print(f"\nEvaluating on {len(sample_ids)} traces...")
    
    for case_id in sample_ids:
        case = df[df["case:concept:name"] == case_id].reset_index(drop=True)
        
        if len(case) < 3:
            continue
        
        case_start = case["time:timestamp"].iloc[0]
        
        # Predict at each position
        for i in range(1, len(case)):
            prefix = list(case["concept:name"].iloc[:i+1])
            actual = (case["time:timestamp"].iloc[i] - case_start).total_seconds()
            
            if actual < 0 or actual > 365 * 24 * 3600:
                continue
            
            predicted = predictor.predict(prefix)
            all_predictions.append(predicted)
            all_actuals.append(actual)
    
    y_true = np.array(all_actuals)
    y_pred = np.array(all_predictions)
    
    # Metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R² with bounds
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10
    r2 = 1 - (ss_res / ss_tot)
    
    rel_errors = np.abs(y_true - y_pred) / (y_true + 1e-6)
    within_25 = np.mean(rel_errors <= 0.25) * 100
    within_50 = np.mean(rel_errors <= 0.50) * 100
    
    print("\n" + "=" * 60)
    print(f"RESULTS - {method.upper()} Model")
    print("=" * 60)
    print(f"Predictions: {len(y_pred)}")
    print(f"Actual mean: {np.mean(y_true)/3600:.1f} hours")
    print(f"Predicted mean: {np.mean(y_pred)/3600:.1f} hours")
    print("-" * 60)
    print(f"MAE:        {mae/3600:.2f} hours ({mae:.0f}s)")
    print(f"RMSE:       {rmse/3600:.2f} hours ({rmse:.0f}s)")
    print(f"R²:         {r2:.4f}")
    print("-" * 60)
    print(f"Within 25%: {within_25:.1f}%")
    print(f"Within 50%: {within_50:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test cumulative time prediction")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--method", default="lstm", choices=["lstm", "ml"], help="Model type")
    parser.add_argument("--traces", type=int, default=100, help="Number of traces for evaluation")
    # Determine default log path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_log = os.path.join(script_dir, "..", "Dataset", "BPI Challenge 2017.xes")
    parser.add_argument("--log", default=default_log, help="Event log path")
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True
    
    df = load_event_log(args.log)
    
    if args.train:
        train_model(df, args.method)
    
    if args.evaluate:
        evaluate_model(df, args.method, args.traces)


if __name__ == "__main__":
    main()

