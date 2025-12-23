import pandas as pd
import pm4py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ProcessingTimeTrainer import ProcessingTimeTrainer
from ProcessingTimePredictionClass import ProcessingTimePredictionClass

print("="*80)
print("Processing Time Prediction Methods Comparison")
print("="*80)

print("\n[1/5] Loading event log...")
log = pm4py.read_xes("Dataset/BPI Challenge 2017.xes")
df = pm4py.convert_to_dataframe(log)
print(f"Loaded {len(df)} events")

print("\n[2/5] Preparing test set...")
df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
df_sorted["time:timestamp"] = pd.to_datetime(df_sorted["time:timestamp"], errors="coerce")
df_sorted = df_sorted.dropna(subset=["time:timestamp"])

test_samples = []
for case_id, case_data in df_sorted.groupby("case:concept:name"):
    case_data = case_data.reset_index(drop=True)
    if len(case_data) < 2:
        continue
    
    for i in range(len(case_data) - 1):
        prev_event = case_data.iloc[i]
        curr_event = case_data.iloc[i + 1]
        
        if pd.isna(prev_event["time:timestamp"]) or pd.isna(curr_event["time:timestamp"]):
            continue
        
        time_diff = (curr_event["time:timestamp"] - prev_event["time:timestamp"]).total_seconds()
        if time_diff <= 0 or time_diff > 31536000:
            continue
        
        prev_activity = str(prev_event["concept:name"]) if not pd.isna(prev_event["concept:name"]) else "unknown"
        prev_lifecycle = "complete" if pd.isna(prev_event.get("lifecycle:transition")) else str(prev_event["lifecycle:transition"])
        curr_activity = str(curr_event["concept:name"]) if not pd.isna(curr_event["concept:name"]) else "unknown"
        curr_lifecycle = "complete" if pd.isna(curr_event.get("lifecycle:transition")) else str(curr_event["lifecycle:transition"])
        
        timestamp = curr_event["time:timestamp"]
        case_start_time = case_data["time:timestamp"].min()
        time_since_start = (prev_event["time:timestamp"] - case_start_time).total_seconds()
        
        context = {
            'resource_1': str(prev_event.get("org:resource", "unknown")) if not pd.isna(prev_event.get("org:resource")) else "unknown",
            'resource_2': str(curr_event.get("org:resource", "unknown")) if not pd.isna(curr_event.get("org:resource")) else "unknown",
            'hour': timestamp.hour,
            'weekday': timestamp.weekday(),
            'month': timestamp.month,
            'day_of_year': timestamp.timetuple().tm_yday,
            'event_position_in_case': i + 1,
            'case_duration_so_far': time_since_start
        }
        
        case_attrs = {}
        for col in ["case:LoanGoal", "case:ApplicationType"]:
            if col in case_data.columns:
                val = case_data[col].iloc[0] if len(case_data) > 0 else None
                case_attrs[col] = val if not pd.isna(val) else None
            else:
                case_attrs[col] = None
        
        context['case:LoanGoal'] = case_attrs.get("case:LoanGoal")
        context['case:ApplicationType'] = case_attrs.get("case:ApplicationType")
        
        event_attrs = ["Accepted", "Selected"]
        for col in event_attrs:
            if col in curr_event.index:
                context[col] = curr_event[col] if not pd.isna(curr_event[col]) else None
            else:
                context[col] = None
        
        test_samples.append({
            'prev_activity': prev_activity,
            'prev_lifecycle': prev_lifecycle,
            'curr_activity': curr_activity,
            'curr_lifecycle': curr_lifecycle,
            'context': context,
            'actual_time': time_diff
        })

np.random.seed(42)
np.random.shuffle(test_samples)
test_size = min(1000, len(test_samples))
test_set = test_samples[:test_size]

print(f"Created test set with {len(test_set)} samples")
print(f"Test set processing times: mean={np.mean([s['actual_time'] for s in test_set]):.2f}s, median={np.median([s['actual_time'] for s in test_set]):.2f}s")

print("\n[3/5] Training models...")

print("\n" + "-"*80)
print("Training Distribution Method")
print("-"*80)
trainer_dist = ProcessingTimeTrainer(df, method="distribution")
trainer_dist.train()
trainer_dist.save_model("models/processing_time_model_dist")
predictor_dist = ProcessingTimePredictionClass(method="distribution", model_path="models/processing_time_model_dist")

print("\n" + "-"*80)
print("Training ML Method (Random Forest)")
print("-"*80)
trainer_ml = ProcessingTimeTrainer(df, method="ml")
trainer_ml.train()
trainer_ml.save_model("models/processing_time_model_ml")
predictor_ml = ProcessingTimePredictionClass(method="ml", model_path="models/processing_time_model_ml")

print("\n" + "-"*80)
print("Training Probabilistic ML Method (LSTM)")
print("-"*80)
trainer_lstm = ProcessingTimeTrainer(df, method="probabilistic_ml")
trainer_lstm.train(cache_path="models/processing_time_model_lstm", force_recompute=False)
trainer_lstm.save_model("models/processing_time_model_lstm")
predictor_lstm = ProcessingTimePredictionClass(method="probabilistic_ml", model_path="models/processing_time_model_lstm")

print("\n[4/5] Evaluating on test set...")

y_true = np.array([s['actual_time'] for s in test_set])

predictions_dist = []
predictions_ml = []
predictions_lstm = []

for i, sample in enumerate(test_set):
    if (i + 1) % 100 == 0:
        print(f"  Processing sample {i+1}/{len(test_set)}...")
    
    try:
        pred_dist = predictor_dist.predict(
            prev_activity=sample['prev_activity'],
            prev_lifecycle=sample['prev_lifecycle'],
            curr_activity=sample['curr_activity'],
            curr_lifecycle=sample['curr_lifecycle'],
            context=sample['context']
        )
        predictions_dist.append(pred_dist)
    except:
        predictions_dist.append(trainer_dist.fallback_mean if trainer_dist.fallback_mean else 3600.0)
    
    try:
        pred_ml = predictor_ml.predict(
            prev_activity=sample['prev_activity'],
            prev_lifecycle=sample['prev_lifecycle'],
            curr_activity=sample['curr_activity'],
            curr_lifecycle=sample['curr_lifecycle'],
            context=sample['context']
        )
        predictions_ml.append(pred_ml)
    except:
        predictions_ml.append(trainer_ml.fallback_mean if trainer_ml.fallback_mean else 3600.0)
    
    try:
        pred_lstm = predictor_lstm.predict(
            prev_activity=sample['prev_activity'],
            prev_lifecycle=sample['prev_lifecycle'],
            curr_activity=sample['curr_activity'],
            curr_lifecycle=sample['curr_lifecycle'],
            context=sample['context']
        )
        predictions_lstm.append(pred_lstm)
    except:
        predictions_lstm.append(trainer_lstm.fallback_mean if trainer_lstm.fallback_mean else 3600.0)

predictions_dist = np.array(predictions_dist)
predictions_ml = np.array(predictions_ml)
predictions_lstm = np.array(predictions_lstm)

print("\n[5/5] Computing metrics...")

def compute_metrics(y_true, y_pred, method_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    
    return {
        'method': method_name,
        'MAE (seconds)': mae,
        'MAE (hours)': mae / 3600,
        'RMSE (seconds)': rmse,
        'RMSE (hours)': rmse / 3600,
        'R²': r2,
        'MAPE (%)': mape
    }

results = [
    compute_metrics(y_true, predictions_dist, "Distribution"),
    compute_metrics(y_true, predictions_ml, "ML (Random Forest)"),
    compute_metrics(y_true, predictions_lstm, "Probabilistic ML (LSTM)")
]

print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

for result in results:
    print(f"\n{result['method']}:")
    print(f"  MAE:  {result['MAE (seconds)']:.2f} seconds ({result['MAE (hours)']:.2f} hours)")
    print(f"  RMSE: {result['RMSE (seconds)']:.2f} seconds ({result['RMSE (hours)']:.2f} hours)")
    print(f"  R²:   {result['R²']:.4f}")
    print(f"  MAPE: {result['MAPE (%)']:.2f}%")

print("\n" + "-"*80)
print("Summary Table")
print("-"*80)
print(f"{'Method':<30} {'MAE (s)':<12} {'RMSE (s)':<12} {'R²':<8} {'MAPE (%)':<10}")
print("-"*80)
for result in results:
    print(f"{result['method']:<30} {result['MAE (seconds)']:<12.2f} {result['RMSE (seconds)']:<12.2f} {result['R²']:<8.4f} {result['MAPE (%)']:<10.2f}")

best_mae = min(results, key=lambda x: x['MAE (seconds)'])
best_rmse = min(results, key=lambda x: x['RMSE (seconds)'])
best_r2 = max(results, key=lambda x: x['R²'])

print("\n" + "-"*80)
print("Best Performers")
print("-"*80)
print(f"Best MAE:  {best_mae['method']} ({best_mae['MAE (seconds)']:.2f}s)")
print(f"Best RMSE: {best_rmse['method']} ({best_rmse['RMSE (seconds)']:.2f}s)")
print(f"Best R²:   {best_r2['method']} ({best_r2['R²']:.4f})")

print("\n" + "="*80)
print("Comparison completed!")
print("="*80)

