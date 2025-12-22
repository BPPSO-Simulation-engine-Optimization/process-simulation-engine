"""
Testing script to compare different processing time prediction methods.

Methods:
- distribution: Log-normal distributions fitted per activity transition
- ml: Random Forest Regressor with contextual features  
- probabilistic_ml: LSTM with uncertainty estimation

Usage:
    python test_prediction_methods.py                    # Test all methods
    python test_prediction_methods.py --methods dist ml  # Test specific methods
    python test_prediction_methods.py --train            # Force re-training
    python test_prediction_methods.py --samples 500      # Use 500 test samples
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_event_log(dataset_path: str) -> pd.DataFrame:
    import pm4py
    print(f"Loading event log from {dataset_path}...")
    log = pm4py.read_xes(dataset_path)
    df = pm4py.convert_to_dataframe(log)
    print(f"Loaded {len(df)} events from {df['case:concept:name'].nunique()} cases")
    return df


def prepare_test_samples(df: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> list:
    print(f"\nPreparing test samples...")
    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df_sorted["time:timestamp"] = pd.to_datetime(df_sorted["time:timestamp"], errors="coerce")
    df_sorted = df_sorted.dropna(subset=["time:timestamp"])
    
    samples = []
    for case_id, case_data in df_sorted.groupby("case:concept:name"):
        case_data = case_data.reset_index(drop=True)
        if len(case_data) < 2:
            continue
        
        case_start = case_data["time:timestamp"].min()
        case_attrs = {}
        for col in ["case:LoanGoal", "case:ApplicationType"]:
            if col in case_data.columns:
                val = case_data[col].iloc[0]
                case_attrs[col] = val if not pd.isna(val) else None
        
        for i in range(len(case_data) - 1):
            prev = case_data.iloc[i]
            curr = case_data.iloc[i + 1]
            
            if pd.isna(prev["time:timestamp"]) or pd.isna(curr["time:timestamp"]):
                continue
            
            time_diff = (curr["time:timestamp"] - prev["time:timestamp"]).total_seconds()
            if time_diff <= 0 or time_diff > 31536000:
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
            
            samples.append({
                'prev_activity': str(prev["concept:name"]) if not pd.isna(prev["concept:name"]) else "unknown",
                'prev_lifecycle': str(prev.get("lifecycle:transition", "complete")) if not pd.isna(prev.get("lifecycle:transition")) else "complete",
                'curr_activity': str(curr["concept:name"]) if not pd.isna(curr["concept:name"]) else "unknown",
                'curr_lifecycle': str(curr.get("lifecycle:transition", "complete")) if not pd.isna(curr.get("lifecycle:transition")) else "complete",
                'context': context,
                'actual_time': time_diff
            })
    
    np.random.seed(seed)
    np.random.shuffle(samples)
    samples = samples[:n_samples]
    
    actual_times = [s['actual_time'] for s in samples]
    print(f"Prepared {len(samples)} test samples")
    print(f"Actual times: mean={np.mean(actual_times):.1f}s, median={np.median(actual_times):.1f}s, std={np.std(actual_times):.1f}s")
    
    return samples


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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    errors = y_true - y_pred
    mape = np.mean(np.abs(errors / (y_true + 1e-6))) * 100
    median_ae = np.median(np.abs(errors))
    
    within_1h = np.mean(np.abs(errors) <= 3600) * 100
    within_6h = np.mean(np.abs(errors) <= 21600) * 100
    within_24h = np.mean(np.abs(errors) <= 86400) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'MedianAE': median_ae,
        'Within1h': within_1h,
        'Within6h': within_6h,
        'Within24h': within_24h
    }


def run_predictions(predictor, samples: list, method_name: str) -> np.ndarray:
    predictions = []
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f"  {method_name}: {i+1}/{len(samples)} predictions...")
        
        try:
            pred = predictor.predict(
                prev_activity=sample['prev_activity'],
                prev_lifecycle=sample['prev_lifecycle'],
                curr_activity=sample['curr_activity'],
                curr_lifecycle=sample['curr_lifecycle'],
                context=sample['context']
            )
            predictions.append(max(0.0, pred))
        except Exception as e:
            predictions.append(predictor.fallback_mean or 3600.0)
    
    elapsed = time.time() - start_time
    print(f"  {method_name}: {len(samples)} predictions in {elapsed:.2f}s ({len(samples)/elapsed:.1f} pred/s)")
    
    return np.array(predictions)


def print_results(results: dict):
    print("\n" + "=" * 90)
    print("COMPARISON RESULTS")
    print("=" * 90)
    
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  MAE:       {metrics['MAE']:>10.1f}s  ({metrics['MAE']/3600:.2f}h)")
        print(f"  RMSE:      {metrics['RMSE']:>10.1f}s  ({metrics['RMSE']/3600:.2f}h)")
        print(f"  MedianAE:  {metrics['MedianAE']:>10.1f}s  ({metrics['MedianAE']/3600:.2f}h)")
        print(f"  R²:        {metrics['R²']:>10.4f}")
        print(f"  MAPE:      {metrics['MAPE']:>10.1f}%")
        print(f"  Within 1h: {metrics['Within1h']:>10.1f}%")
        print(f"  Within 6h: {metrics['Within6h']:>10.1f}%")
        print(f"  Within 24h:{metrics['Within24h']:>10.1f}%")
    
    print("\n" + "-" * 90)
    print("SUMMARY TABLE")
    print("-" * 90)
    header = f"{'Method':<25} {'MAE (s)':<12} {'RMSE (s)':<12} {'R²':<10} {'Within 1h':<10} {'Within 24h':<10}"
    print(header)
    print("-" * 90)
    
    for method, m in results.items():
        row = f"{method:<25} {m['MAE']:<12.1f} {m['RMSE']:<12.1f} {m['R²']:<10.4f} {m['Within1h']:<10.1f}% {m['Within24h']:<10.1f}%"
        print(row)
    
    print("\n" + "-" * 90)
    print("BEST PERFORMERS")
    print("-" * 90)
    
    best_mae = min(results.items(), key=lambda x: x[1]['MAE'])
    best_rmse = min(results.items(), key=lambda x: x[1]['RMSE'])
    best_r2 = max(results.items(), key=lambda x: x[1]['R²'])
    best_within_1h = max(results.items(), key=lambda x: x[1]['Within1h'])
    
    print(f"Best MAE:       {best_mae[0]} ({best_mae[1]['MAE']:.1f}s)")
    print(f"Best RMSE:      {best_rmse[0]} ({best_rmse[1]['RMSE']:.1f}s)")
    print(f"Best R²:        {best_r2[0]} ({best_r2[1]['R²']:.4f})")
    print(f"Best Within 1h: {best_within_1h[0]} ({best_within_1h[1]['Within1h']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test processing time prediction methods")
    parser.add_argument("--methods", nargs="+", default=["dist", "ml", "lstm"],
                       choices=["dist", "ml", "lstm", "distribution", "probabilistic_ml"],
                       help="Methods to test (default: all)")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of test samples (default: 1000)")
    parser.add_argument("--train", action="store_true",
                       help="Force re-training of models")
    parser.add_argument("--dataset", type=str, default="Dataset/BPI Challenge 2017.xes",
                       help="Path to XES event log")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory for model files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for test sample selection")
    
    args = parser.parse_args()
    
    os.chdir(Path(__file__).parent.parent)
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    print("=" * 90)
    print("PROCESSING TIME PREDICTION - METHOD COMPARISON")
    print("=" * 90)
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Test samples: {args.samples}")
    print(f"Force training: {args.train}")
    
    df = load_event_log(args.dataset)
    test_samples = prepare_test_samples(df, n_samples=args.samples, seed=args.seed)
    y_true = np.array([s['actual_time'] for s in test_samples])
    
    method_names = {
        'dist': 'Distribution',
        'distribution': 'Distribution',
        'ml': 'Random Forest',
        'rf': 'Random Forest',
        'lstm': 'LSTM (Probabilistic)',
        'probabilistic_ml': 'LSTM (Probabilistic)'
    }
    
    results = {}
    
    for method in args.methods:
        print(f"\n{'='*40}")
        print(f"Testing: {method_names.get(method, method)}")
        print('='*40)
        
        predictor = get_or_train_model(df, method, args.model_dir, args.train)
        predictions = run_predictions(predictor, test_samples, method_names.get(method, method))
        metrics = evaluate_predictions(y_true, predictions)
        results[method_names.get(method, method)] = metrics
    
    print_results(results)
    
    print("\n" + "=" * 90)
    print("Testing complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()

