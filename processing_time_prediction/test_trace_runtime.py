"""
Test script to compare actual vs predicted total runtime of sampled traces.

This tests how well each prediction method estimates the total duration of 
complete process traces (cases), not just individual event transitions.

Usage:
    python test_trace_runtime.py                    # Test all methods
    python test_trace_runtime.py --methods dist ml  # Test specific methods
    python test_trace_runtime.py --traces 100       # Sample 100 traces
    python test_trace_runtime.py --train            # Force re-training
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_event_log(dataset_path: str) -> pd.DataFrame:
    import pm4py
    print(f"Loading event log from {dataset_path}...")
    log = pm4py.read_xes(dataset_path)
    df = pm4py.convert_to_dataframe(log)
    print(f"Loaded {len(df)} events from {df['case:concept:name'].nunique()} cases")
    return df


def extract_traces(df: pd.DataFrame, n_traces: int = 100, seed: int = 42) -> List[Dict]:
    """Extract complete traces with all transitions for runtime prediction."""
    print(f"\nExtracting traces for testing...")
    
    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df_sorted["time:timestamp"] = pd.to_datetime(df_sorted["time:timestamp"], errors="coerce")
    df_sorted = df_sorted.dropna(subset=["time:timestamp"])
    
    traces = []
    
    for case_id, case_data in df_sorted.groupby("case:concept:name"):
        case_data = case_data.reset_index(drop=True)
        if len(case_data) < 3:  # Need at least 3 events for meaningful trace
            continue
        
        case_start = case_data["time:timestamp"].min()
        case_end = case_data["time:timestamp"].max()
        actual_duration = (case_end - case_start).total_seconds()
        
        if actual_duration <= 0 or actual_duration > 365 * 24 * 3600:
            continue
        
        case_attrs = {}
        for col in ["case:LoanGoal", "case:ApplicationType"]:
            if col in case_data.columns:
                val = case_data[col].iloc[0]
                case_attrs[col] = val if not pd.isna(val) else None
        
        transitions = []
        for i in range(len(case_data) - 1):
            prev = case_data.iloc[i]
            curr = case_data.iloc[i + 1]
            
            if pd.isna(prev["time:timestamp"]) or pd.isna(curr["time:timestamp"]):
                continue
            
            time_diff = (curr["time:timestamp"] - prev["time:timestamp"]).total_seconds()
            if time_diff < 0:
                continue
            
            context = {
                'resource_1': str(prev.get("org:resource", "unknown")) if not pd.isna(prev.get("org:resource")) else "unknown",
                'resource_2': str(curr.get("org:resource", "unknown")) if not pd.isna(curr.get("org:resource")) else "unknown",
                'hour': curr["time:timestamp"].hour,
                'weekday': curr["time:timestamp"].weekday(),
                'month': curr["time:timestamp"].month,
                'day_of_year': curr["time:timestamp"].timetuple().tm_yday,
                'event_position_in_case': i + 1,
                'case_duration_so_far': (prev["time:timestamp"] - case_start).total_seconds(),
                **case_attrs
            }
            
            for col in ["Accepted", "Selected"]:
                if col in curr.index:
                    context[col] = curr[col] if not pd.isna(curr[col]) else None
            
            transitions.append({
                'prev_activity': str(prev["concept:name"]) if not pd.isna(prev["concept:name"]) else "unknown",
                'prev_lifecycle': str(prev.get("lifecycle:transition", "complete")) if not pd.isna(prev.get("lifecycle:transition")) else "complete",
                'curr_activity': str(curr["concept:name"]) if not pd.isna(curr["concept:name"]) else "unknown",
                'curr_lifecycle': str(curr.get("lifecycle:transition", "complete")) if not pd.isna(curr.get("lifecycle:transition")) else "complete",
                'context': context,
                'actual_time': time_diff
            })
        
        if len(transitions) >= 2:
            traces.append({
                'case_id': case_id,
                'num_events': len(case_data),
                'num_transitions': len(transitions),
                'actual_duration': actual_duration,
                'transitions': transitions
            })
    
    np.random.seed(seed)
    np.random.shuffle(traces)
    traces = traces[:n_traces]
    
    durations = [t['actual_duration'] for t in traces]
    lengths = [t['num_transitions'] for t in traces]
    
    print(f"Extracted {len(traces)} traces")
    print(f"Trace lengths: mean={np.mean(lengths):.1f}, min={min(lengths)}, max={max(lengths)} transitions")
    print(f"Actual durations: mean={np.mean(durations)/3600:.1f}h, median={np.median(durations)/3600:.1f}h")
    
    return traces


def get_or_train_model(df: pd.DataFrame, method: str, model_dir: str, force_train: bool = False):
    from ProcessingTimeTrainer import ProcessingTimeTrainer
    from ProcessingTimePredictionClass import ProcessingTimePredictionClass
    
    method_map = {
        'dist': 'distribution',
        'distribution': 'distribution',
        'ml': 'ml',
        'rf': 'ml',
        'lstm': 'probabilistic_ml',
        'probabilistic_ml': 'probabilistic_ml'
    }
    method_name = method_map.get(method.lower(), method)
    
    model_path = os.path.join(model_dir, f"processing_time_{method_name}")
    metadata_path = f"{model_path}_metadata.joblib"
    
    if not force_train and os.path.exists(metadata_path):
        print(f"Loading existing {method_name} model from {model_path}")
        try:
            predictor = ProcessingTimePredictionClass(method=method_name, model_path=model_path)
            return predictor
        except Exception as e:
            print(f"Failed to load model: {e}. Will retrain.")
    
    print(f"\nTraining {method_name} model...")
    trainer = ProcessingTimeTrainer(df, method=method_name)
    
    if method_name == "probabilistic_ml":
        trainer.train(cache_path=model_path, force_recompute=force_train)
    else:
        trainer.train()
    
    trainer.save_model(model_path)
    predictor = ProcessingTimePredictionClass(method=method_name, model_path=model_path)
    
    return predictor


def predict_trace_runtime(predictor, trace: Dict) -> float:
    """Predict total runtime by summing predicted times for all transitions."""
    total_predicted = 0.0
    
    for trans in trace['transitions']:
        try:
            pred = predictor.predict(
                prev_activity=trans['prev_activity'],
                prev_lifecycle=trans['prev_lifecycle'],
                curr_activity=trans['curr_activity'],
                curr_lifecycle=trans['curr_lifecycle'],
                context=trans['context']
            )
            total_predicted += max(0.0, pred)
        except:
            total_predicted += predictor.fallback_mean or 3600.0
    
    return total_predicted


def evaluate_trace_predictions(traces: List[Dict], predictions: Dict[str, List[float]]) -> Dict:
    """Evaluate prediction accuracy for trace runtimes."""
    actual = np.array([t['actual_duration'] for t in traces])
    
    results = {}
    for method, preds in predictions.items():
        preds = np.array(preds)
        
        mae = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        r2 = r2_score(actual, preds)
        
        errors = actual - preds
        rel_errors = errors / (actual + 1e-6)
        mape = np.mean(np.abs(rel_errors)) * 100
        
        median_ae = np.median(np.abs(errors))
        
        within_1h = np.mean(np.abs(errors) <= 3600) * 100
        within_6h = np.mean(np.abs(errors) <= 21600) * 100
        within_24h = np.mean(np.abs(errors) <= 86400) * 100
        within_10pct = np.mean(np.abs(rel_errors) <= 0.10) * 100
        within_25pct = np.mean(np.abs(rel_errors) <= 0.25) * 100
        within_50pct = np.mean(np.abs(rel_errors) <= 0.50) * 100
        
        over_predictions = np.mean(preds > actual) * 100
        under_predictions = np.mean(preds < actual) * 100
        
        results[method] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'MedianAE': median_ae,
            'Within1h': within_1h,
            'Within6h': within_6h,
            'Within24h': within_24h,
            'Within10pct': within_10pct,
            'Within25pct': within_25pct,
            'Within50pct': within_50pct,
            'OverPredicted': over_predictions,
            'UnderPredicted': under_predictions,
            'MeanPredicted': np.mean(preds),
            'MeanActual': np.mean(actual)
        }
    
    return results


def print_results(results: Dict, traces: List[Dict]):
    actual_durations = [t['actual_duration'] for t in traces]
    
    print("\n" + "=" * 95)
    print("TRACE RUNTIME PREDICTION - COMPARISON RESULTS")
    print("=" * 95)
    print(f"\nActual trace durations: mean={np.mean(actual_durations)/3600:.2f}h, median={np.median(actual_durations)/3600:.2f}h")
    
    for method, m in results.items():
        print(f"\n{'-'*50}")
        print(f"{method}")
        print(f"{'-'*50}")
        print(f"  MAE:          {m['MAE']/3600:>8.2f} hours ({m['MAE']:>10.0f}s)")
        print(f"  RMSE:         {m['RMSE']/3600:>8.2f} hours ({m['RMSE']:>10.0f}s)")
        print(f"  Median AE:    {m['MedianAE']/3600:>8.2f} hours ({m['MedianAE']:>10.0f}s)")
        print(f"  R²:           {m['R²']:>8.4f}")
        print(f"  MAPE:         {m['MAPE']:>8.1f}%")
        print(f"  Mean Pred:    {m['MeanPredicted']/3600:>8.2f} hours")
        print(f"  Mean Actual:  {m['MeanActual']/3600:>8.2f} hours")
        print(f"  Over/Under:   {m['OverPredicted']:.1f}% / {m['UnderPredicted']:.1f}%")
        print(f"  Within 1h:    {m['Within1h']:>8.1f}%")
        print(f"  Within 24h:   {m['Within24h']:>8.1f}%")
        print(f"  Within 10%:   {m['Within10pct']:>8.1f}%")
        print(f"  Within 25%:   {m['Within25pct']:>8.1f}%")
        print(f"  Within 50%:   {m['Within50pct']:>8.1f}%")
    
    print("\n" + "=" * 95)
    print("SUMMARY TABLE - ABSOLUTE ERRORS")
    print("=" * 95)
    header = f"{'Method':<25} {'MAE (h)':<10} {'RMSE (h)':<10} {'R²':<10} {'Within 1h':<12} {'Within 24h':<12}"
    print(header)
    print("-" * 95)
    for method, m in results.items():
        row = f"{method:<25} {m['MAE']/3600:<10.2f} {m['RMSE']/3600:<10.2f} {m['R²']:<10.4f} {m['Within1h']:<12.1f}% {m['Within24h']:<12.1f}%"
        print(row)
    
    print("\n" + "=" * 95)
    print("SUMMARY TABLE - RELATIVE ERRORS")
    print("=" * 95)
    header = f"{'Method':<25} {'MAPE':<10} {'Within 10%':<12} {'Within 25%':<12} {'Within 50%':<12}"
    print(header)
    print("-" * 95)
    for method, m in results.items():
        row = f"{method:<25} {m['MAPE']:<10.1f}% {m['Within10pct']:<12.1f}% {m['Within25pct']:<12.1f}% {m['Within50pct']:<12.1f}%"
        print(row)
    
    print("\n" + "-" * 95)
    print("BEST PERFORMERS")
    print("-" * 95)
    best_mae = min(results.items(), key=lambda x: x[1]['MAE'])
    best_rmse = min(results.items(), key=lambda x: x[1]['RMSE'])
    best_r2 = max(results.items(), key=lambda x: x[1]['R²'])
    best_mape = min(results.items(), key=lambda x: x[1]['MAPE'])
    best_within_25pct = max(results.items(), key=lambda x: x[1]['Within25pct'])
    
    print(f"Best MAE:          {best_mae[0]} ({best_mae[1]['MAE']/3600:.2f}h)")
    print(f"Best RMSE:         {best_rmse[0]} ({best_rmse[1]['RMSE']/3600:.2f}h)")
    print(f"Best R²:           {best_r2[0]} ({best_r2[1]['R²']:.4f})")
    print(f"Best MAPE:         {best_mape[0]} ({best_mape[1]['MAPE']:.1f}%)")
    print(f"Best Within 25%:   {best_within_25pct[0]} ({best_within_25pct[1]['Within25pct']:.1f}%)")


def print_sample_predictions(traces: List[Dict], predictions: Dict[str, List[float]], n_samples: int = 10):
    """Print sample predictions for inspection."""
    print("\n" + "=" * 95)
    print(f"SAMPLE TRACE PREDICTIONS (first {n_samples} traces)")
    print("=" * 95)
    
    methods = list(predictions.keys())
    header = f"{'Trace':<8} {'Events':<8} {'Actual (h)':<12}"
    for m in methods:
        header += f" {m[:12]:<14}"
    header += " {'Best':>10}"
    print(header)
    print("-" * 95)
    
    for i in range(min(n_samples, len(traces))):
        trace = traces[i]
        actual_h = trace['actual_duration'] / 3600
        
        row = f"{i+1:<8} {trace['num_events']:<8} {actual_h:<12.2f}"
        
        errors = {}
        for method in methods:
            pred_h = predictions[method][i] / 3600
            row += f" {pred_h:<14.2f}"
            errors[method] = abs(actual_h - pred_h)
        
        best = min(errors.items(), key=lambda x: x[1])
        row += f" {best[0][:10]}"
        
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Test trace runtime predictions")
    parser.add_argument("--methods", nargs="+", default=["dist", "ml", "lstm"],
                       choices=["dist", "ml", "lstm", "distribution", "probabilistic_ml"],
                       help="Methods to test (default: all)")
    parser.add_argument("--traces", type=int, default=100,
                       help="Number of traces to sample (default: 100)")
    parser.add_argument("--train", action="store_true",
                       help="Force re-training of models")
    parser.add_argument("--dataset", type=str, default="Dataset/BPI Challenge 2017.xes",
                       help="Path to XES event log")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory for model files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for trace sampling")
    parser.add_argument("--show-samples", type=int, default=10,
                       help="Number of sample predictions to display")
    
    args = parser.parse_args()
    
    os.chdir(Path(__file__).parent.parent)
    os.makedirs(args.model_dir, exist_ok=True)
    
    print("=" * 95)
    print("TRACE RUNTIME PREDICTION - METHOD COMPARISON")
    print("=" * 95)
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Traces to sample: {args.traces}")
    print(f"Force training: {args.train}")
    
    df = load_event_log(args.dataset)
    traces = extract_traces(df, n_traces=args.traces, seed=args.seed)
    
    method_names = {
        'dist': 'Distribution',
        'distribution': 'Distribution',
        'ml': 'Random Forest',
        'rf': 'Random Forest',
        'lstm': 'LSTM (Probabilistic)',
        'probabilistic_ml': 'LSTM (Probabilistic)'
    }
    
    predictions = {}
    
    for method in args.methods:
        method_display = method_names.get(method, method)
        print(f"\n{'='*50}")
        print(f"Testing: {method_display}")
        print('='*50)
        
        predictor = get_or_train_model(df, method, args.model_dir, args.train)
        
        print(f"\nPredicting runtimes for {len(traces)} traces...")
        start_time = time.time()
        
        trace_predictions = []
        for i, trace in enumerate(traces):
            if (i + 1) % 25 == 0:
                print(f"  {method_display}: {i+1}/{len(traces)} traces...")
            
            pred = predict_trace_runtime(predictor, trace)
            trace_predictions.append(pred)
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f}s ({len(traces)/elapsed:.1f} traces/s)")
        
        predictions[method_display] = trace_predictions
    
    results = evaluate_trace_predictions(traces, predictions)
    
    print_results(results, traces)
    
    if args.show_samples > 0:
        print_sample_predictions(traces, predictions, args.show_samples)
    
    print("\n" + "=" * 95)
    print("Testing complete!")
    print("=" * 95)


if __name__ == "__main__":
    main()

