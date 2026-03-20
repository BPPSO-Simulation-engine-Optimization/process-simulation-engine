"""
Benchmark processing time prediction models on held-out test data.

Splits the event log by case (train/test) so test cases are never seen during
training. Evaluates distribution, ML (Random Forest), and probabilistic_ml (LSTM)
on MAE, RMSE, R², MAPE, Median AE, and MdAPE.

Usage:
  python benchmark_models.py --log Dataset/BPI Challenge 2017.xes
  python benchmark_models.py --log eventlog.xes --test_samples 2000 --output results.csv
  python benchmark_models.py --log eventlog.xes --skip_train --model_dir models
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import pm4py
except ImportError:
    pm4py = None

from ProcessingTimeTrainer import ProcessingTimeTrainer
from ProcessingTimePredictionClass import ProcessingTimePredictionClass


def load_log(path: str) -> pd.DataFrame:
    if pm4py is None:
        raise ImportError("pm4py is required. Install with: pip install pm4py")
    if path.endswith(".xes") or path.endswith(".xes.gz"):
        log = pm4py.read_xes(path)
    else:
        log = pd.read_csv(path)
    return pm4py.convert_to_dataframe(log) if hasattr(log, "__iter__") and not isinstance(log, pd.DataFrame) else log


def split_cases(df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42):
    case_col = "case:concept:name"
    if case_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{case_col}'")
    case_ids = df[case_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(case_ids)
    n_train = max(1, int(len(case_ids) * train_ratio))
    train_cases = set(case_ids[:n_train])
    test_cases = set(case_ids[n_train:])
    train_df = df[df[case_col].isin(train_cases)].copy()
    test_df = df[df[case_col].isin(test_cases)].copy()
    return train_df, test_df


def extract_test_samples(df: pd.DataFrame, max_samples: Optional[int] = None, seed: int = 42):
    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df_sorted["time:timestamp"] = pd.to_datetime(df_sorted["time:timestamp"], errors="coerce")
    df_sorted = df_sorted.dropna(subset=["time:timestamp"])

    samples = []
    for case_id, case_data in df_sorted.groupby("case:concept:name"):
        case_data = case_data.reset_index(drop=True)
        if len(case_data) < 2:
            continue
        case_start_time = case_data["time:timestamp"].min()
        case_attrs = {}
        for col in ["case:LoanGoal", "case:ApplicationType"]:
            if col in case_data.columns:
                val = case_data[col].iloc[0]
                case_attrs[col] = val if not pd.isna(val) else None
            else:
                case_attrs[col] = None

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
            time_since_start = (prev_event["time:timestamp"] - case_start_time).total_seconds()

            context = {
                "resource_1": str(prev_event.get("org:resource", "unknown")) if not pd.isna(prev_event.get("org:resource")) else "unknown",
                "resource_2": str(curr_event.get("org:resource", "unknown")) if not pd.isna(curr_event.get("org:resource")) else "unknown",
                "hour": timestamp.hour,
                "weekday": timestamp.weekday(),
                "month": timestamp.month,
                "day_of_year": timestamp.timetuple().tm_yday,
                "event_position_in_case": i + 1,
                "case_duration_so_far": time_since_start,
                "case:LoanGoal": case_attrs.get("case:LoanGoal"),
                "case:ApplicationType": case_attrs.get("case:ApplicationType"),
            }
            for col in ["Accepted", "Selected"]:
                context[col] = curr_event[col] if col in curr_event.index and not pd.isna(curr_event[col]) else None

            samples.append({
                "prev_activity": prev_activity,
                "prev_lifecycle": prev_lifecycle,
                "curr_activity": curr_activity,
                "curr_lifecycle": curr_lifecycle,
                "context": context,
                "actual_time": time_diff,
            })

    if max_samples and len(samples) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[j] for j in idx]
    return samples


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {}
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    median_ae = np.median(np.abs(y_true - y_pred))
    mdape = np.median(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {
        "MAE_s": mae,
        "MAE_h": mae / 3600,
        "RMSE_s": rmse,
        "RMSE_h": rmse / 3600,
        "R2": r2,
        "MAPE_pct": mape,
        "MedianAE_s": median_ae,
        "MdAPE_pct": mdape,
    }


def run_predictions(predictor, test_samples: list, fallback: float, name: str) -> np.ndarray:
    preds = []
    for sample in test_samples:
        try:
            p = predictor.predict(
                prev_activity=sample["prev_activity"],
                prev_lifecycle=sample["prev_lifecycle"],
                curr_activity=sample["curr_activity"],
                curr_lifecycle=sample["curr_lifecycle"],
                context=sample["context"],
            )
            preds.append(float(p))
        except Exception:
            preds.append(fallback)
    return np.array(preds)


def main():
    p = argparse.ArgumentParser(description="Benchmark processing time prediction models")
    p.add_argument("--log", default="Dataset/BPI Challenge 2017.xes", help="Path to event log (.xes or .csv)")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Fraction of cases used for training")
    p.add_argument("--test_samples", type=int, default=None, help="Max test samples (default: all)")
    p.add_argument("--model_dir", default="models", help="Directory for saving/loading models")
    p.add_argument("--output", default=None, help="Save results to this CSV path")
    p.add_argument("--skip_train", action="store_true", help="Only evaluate; load existing models from model_dir")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    print("Loading event log...")
    df = load_log(args.log)
    print(f"  Events: {len(df)}")

    train_df, test_df = split_cases(df, train_ratio=args.train_ratio, seed=args.seed)
    print(f"  Train cases: {train_df['case:concept:name'].nunique()}, test cases: {test_df['case:concept:name'].nunique()}")

    test_samples = extract_test_samples(test_df, max_samples=args.test_samples, seed=args.seed)
    if not test_samples:
        print("No test samples. Check log format and columns.")
        sys.exit(1)
    y_true = np.array([s["actual_time"] for s in test_samples])
    print(f"  Test samples: {len(test_samples)} (mean={y_true.mean():.0f}s, median={np.median(y_true):.0f}s)")

    base = args.model_dir
    os.makedirs(base, exist_ok=True)
    predictors = []
    fallbacks = []

    if not args.skip_train:
        print("\nTraining models...")
        for method, label in [
            ("distribution", "Distribution"),
            ("ml", "ML (Random Forest)"),
            ("probabilistic_ml", "Probabilistic ML (LSTM)"),
        ]:
            print(f"  {label}...")
            trainer = ProcessingTimeTrainer(train_df, method=method)
            if method == "probabilistic_ml":
                trainer.train(cache_path=os.path.join(base, "processing_time_model_lstm"), force_recompute=False)
            else:
                trainer.train()
            path = os.path.join(base, f"processing_time_model_{'lstm' if method == 'probabilistic_ml' else method}")
            trainer.save_model(path)
            pred = ProcessingTimePredictionClass(method=method, model_path=path)
            predictors.append((label, pred))
            fallbacks.append(trainer.fallback_mean if trainer.fallback_mean else 3600.0)
    else:
        print("\nLoading models...")
        for method, label in [
            ("distribution", "Distribution"),
            ("ml", "ML (Random Forest)"),
            ("probabilistic_ml", "Probabilistic ML (LSTM)"),
        ]:
            path = os.path.join(base, f"processing_time_model_{'lstm' if method == 'probabilistic_ml' else method}")
            pred = ProcessingTimePredictionClass(method=method, model_path=path)
            predictors.append((label, pred))
            fallbacks.append(pred.fallback_mean if pred.fallback_mean else 3600.0)

    print("\nEvaluating...")
    results = []
    for (label, predictor), fallback in zip(predictors, fallbacks):
        y_pred = run_predictions(predictor, test_samples, fallback, label)
        m = compute_metrics(y_true, y_pred)
        m["model"] = label
        results.append(m)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    cols = ["model", "MAE_s", "RMSE_s", "R2", "MAPE_pct", "MedianAE_s", "MdAPE_pct"]
    table = pd.DataFrame(results)[cols]
    table = table.rename(columns={
        "MAE_s": "MAE (s)",
        "RMSE_s": "RMSE (s)",
        "R2": "R²",
        "MAPE_pct": "MAPE (%)",
        "MedianAE_s": "Median AE (s)",
        "MdAPE_pct": "MdAPE (%)",
    })
    print(table.to_string(index=False))

    print("\nBest per metric:")
    df_res = pd.DataFrame(results)
    print(f"  MAE:     {df_res.loc[df_res['MAE_s'].idxmin(), 'model']} ({df_res['MAE_s'].min():.2f}s)")
    print(f"  RMSE:    {df_res.loc[df_res['RMSE_s'].idxmin(), 'model']} ({df_res['RMSE_s'].min():.2f}s)")
    print(f"  R²:      {df_res.loc[df_res['R2'].idxmax(), 'model']} ({df_res['R2'].max():.4f})")
    print(f"  MAPE:    {df_res.loc[df_res['MAPE_pct'].idxmin(), 'model']} ({df_res['MAPE_pct'].min():.2f}%)")
    print(f"  Median AE: {df_res.loc[df_res['MedianAE_s'].idxmin(), 'model']} ({df_res['MedianAE_s'].min():.2f}s)")

    if args.output:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
