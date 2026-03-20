#!/usr/bin/env python3
"""
Full evaluation of all activity-specific XGBoost models with tabular output.

Run this script directly:
    python evaluate_all.py

Or call the main() function from another script after adjusting the paths.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

# Support both package imports and direct script execution
try:
    from .activity_specific_model import ActivitySpecificModel
    from .data_loader import DataLoader
except ImportError:
    from activity_specific_model import ActivitySpecificModel
    from data_loader import DataLoader


def main():
    print("=" * 80)
    print("ACTIVITY-SPECIFIC MODEL EVALUATION")
    print("=" * 80)
    print()
    overall_start = datetime.now()
    print(f"[{overall_start:%H:%M:%S}] Pipeline-Start")

    # ─── Configuration ────────────────────────────────────────────────────────
    # Set USE_BUSINESS_TIME = False to disable business-hour filtering (5:00-22:00)
    USE_BUSINESS_TIME = True

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "XGBoost_evaluation_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CSV_PATH = os.path.join(OUTPUT_DIR, "bpic2017.csv")
    XES_PATH = os.path.join(PARENT_DIR, "BPI Challenge 2017.xes.gz")

    # ─── Data loading ─────────────────────────────────────────────────────────
    print("Loading data...")
    data_loader = DataLoader()

    load_start = datetime.now()
    if os.path.exists(CSV_PATH):
        print(f"[{load_start:%H:%M:%S}] CSV file found: {CSV_PATH}")
        print("→ Lade Events aus CSV (schnell)...")
        df = data_loader.load_csv_to_dataframe(CSV_PATH)
    else:
        print(f"[{load_start:%H:%M:%S}] CSV file not found: {CSV_PATH}")
        print(f"→ Lade Events aus XES-Datei (kann einige Minuten dauern): {XES_PATH}")
        df = data_loader.load_xes_to_dataframe(XES_PATH)
        print("→ Speichere als CSV für zukünftige, schnellere Läufe ...")
        data_loader.save_dataframe_to_csv(df, CSV_PATH)
        print(f"CSV saved to: {CSV_PATH}")
    load_end = datetime.now()
    print(f"[{load_end:%H:%M:%S}] Datenladen abgeschlossen "
          f"({(load_end - load_start).total_seconds()/60:.1f} Minuten)")

    print(f"Loaded {len(df)} events from {len(df['case_id'].unique())} cases\n")

    # ─── Feature configuration ────────────────────────────────────────────────
    custom_base_features = [
        "event",                   # event name (categorical) – corresponds to concept:name
        "org:resource",            # resource / person (categorical)
        "EventOrigin",             # event origin (categorical)
        "event_index",             # position in case (numerical)
        "hour",                    # hour of day (numerical)
        "minute",                  # minute (numerical)
        "second",                  # second (numerical)
        "microsecond",             # microsecond (numerical)
        "weekday",                 # weekday (numerical, 0=Monday)
        "day_of_month",            # day of month (numerical)
        "month",                   # month (numerical)
        "day_of_year",             # day of year (numerical)
        # ── Advanced context features (new) ──────────────────────────────────
        "time_since_case_start",   # hours elapsed since case first event (numerical)
        "time_since_last_event",   # hours since previous event in case (numerical)
        "prev_activity",           # previous event name (categorical)
        "is_weekend",              # 1 = Saturday/Sunday, else 0 (numerical)
        "hour_sin",                # cyclical hour encoding – sine (numerical)
        "hour_cos",                # cyclical hour encoding – cosine (numerical)
        "weekday_sin",             # cyclical weekday encoding – sine (numerical)
        "weekday_cos",             # cyclical weekday encoding – cosine (numerical)
    ]

    custom_case_features = [
        "case:LoanGoal",           # categorical
        "case:ApplicationType",    # categorical
        "case:RequestedAmount",    # numerical
        "FirstWithdrawalAmount",   # numerical (O_Create Offer only)
        "NumberOfTerms",           # numerical (O_Create Offer only)
        "Accepted",                # boolean (O_Create Offer only)
        "MonthlyCost",             # numerical (O_Create Offer only)
        "Selected",                # boolean (O_Create Offer only)
        "CreditScore",             # numerical (O_Create Offer only)
        "OfferedAmount",           # numerical (O_Create Offer only)
        "OfferID",                 # categorical (O_Create Offer only)
    ]

    # ─── Model initialisation ─────────────────────────────────────────────────
    models_dir = os.path.join(OUTPUT_DIR, "activity_models")
    os.makedirs(models_dir, exist_ok=True)

    print("Initializing Activity-Specific Model...")
    activity_model = ActivitySpecificModel(
        base_features=custom_base_features,
        case_features=custom_case_features,
        random_state=42,
        models_dir=models_dir,
    )
    print(f"Will train models for {len(activity_model.activities)} activities\n")

    # ─── Data preparation ─────────────────────────────────────────────────────
    if USE_BUSINESS_TIME:
        print("Preparing activity-specific data (with business-time filtering 5:00-22:00)...")
    else:
        print("Preparing activity-specific data (NO business-time filtering – using all time)...")
    activity_model.prepare_activity_data(
        df,
        test_size=0.2,
        use_business_time=USE_BUSINESS_TIME,
        output_dir=OUTPUT_DIR,
    )
    print()

    # ─── Training ─────────────────────────────────────────────────────────────
    # Optionally customise quantiles per activity, e.g.:
    # quantile_config = {
    #     "A_Concept": [0.8, 0.9, 0.95],
    #     "A_Complete": [0.5],
    # }
    print("Training models...")
    activity_model.train_activity_models()
    print()

    # ─── Evaluation ───────────────────────────────────────────────────────────
    print("Evaluating models...")
    results = activity_model.evaluate_activity_models()
    print()

    overall_metrics = activity_model.get_overall_metrics(results)
    weighted_metrics = activity_model.get_weighted_overall_metrics(results)

    # ─── Display table ────────────────────────────────────────────────────────
    print("=" * 80)
    print("DETAILED METRICS PER ACTIVITY")
    print("=" * 80)
    print()

    df_display = overall_metrics[
        ["activity", "mae_hours", "rmse_hours", "mae_log", "rmse_log", "test_samples"]
    ].copy()

    df_display["is_fixed"] = df_display["activity"].isin(activity_model.fixed_activities)
    df_display["Activity"] = df_display.apply(
        lambda row: f"{row['activity']} [FIXED]" if row["is_fixed"] else row["activity"],
        axis=1,
    )
    df_display["sort_key"] = df_display.apply(
        lambda row: (row["test_samples"] == 0, row["mae_hours"]),
        axis=1,
    )
    df_display = df_display.sort_values("sort_key").drop("sort_key", axis=1)
    df_display = df_display[
        ["Activity", "mae_hours", "rmse_hours", "mae_log", "rmse_log", "test_samples"]
    ]
    df_display.columns = [
        "Activity", "MAE (hours)", "RMSE (hours)", "MAE (log)", "RMSE (log)", "Test Samples"
    ]

    df_display["MAE (hours)"] = df_display["MAE (hours)"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    df_display["RMSE (hours)"] = df_display["RMSE (hours)"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    df_display["MAE (log)"] = df_display["MAE (log)"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )
    df_display["RMSE (log)"] = df_display["RMSE (log)"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )
    df_display["Test Samples"] = df_display["Test Samples"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "0 (no data)"
    )

    print(df_display.to_string(index=False))
    print()

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY METRICS (Original Scale & Log Scale)")
    print("=" * 80)
    print(f"Weighted Mean MAE  (hours): {weighted_metrics['weighted_mean_mae_hours']:.2f}")
    print(f"Weighted Mean RMSE (hours): {weighted_metrics['weighted_mean_rmse_hours']:.2f}")
    print(f"Weighted Mean MAE  (log):   {weighted_metrics['weighted_mean_mae_log']:.4f}")
    print(f"Weighted Mean RMSE (log):   {weighted_metrics['weighted_mean_rmse_log']:.4f}")
    print()
    print(f"Unweighted Mean MAE  (hours): {overall_metrics['mae_hours'].mean():.2f}")
    print(f"Unweighted Mean RMSE (hours): {overall_metrics['rmse_hours'].mean():.2f}")
    print(f"Unweighted Mean MAE  (log):   {overall_metrics['mae_log'].mean():.4f}")
    print(f"Unweighted Mean RMSE (log):   {overall_metrics['rmse_log'].mean():.4f}")
    print(f"Total Test Samples: {int(weighted_metrics['total_test_samples']):,}")
    print()

    # ─── Save results ─────────────────────────────────────────────────────────
    print("Saving results...")
    results_dir = os.path.join(OUTPUT_DIR, "activity_results")
    activity_model.save_evaluation_results(results, results_dir)
    if activity_model.models_dir:
        activity_model.save_models(activity_model.models_dir)
    print()
    pipeline_end = datetime.now()
    total_min = (pipeline_end - overall_start).total_seconds() / 60
    print("=" * 80)
    print("EVALUATION COMPLETED!")
    print("=" * 80)
    print(f"[{pipeline_end:%H:%M:%S}] Gesamtlaufzeit: {total_min:.1f} Minuten")
    print(f"Results saved to: {results_dir}/")
    print(f"Models  saved to: {models_dir}/")
    print(f"All outputs:      {OUTPUT_DIR}/")

    return {
        "results": results,
        "overall_metrics": overall_metrics,
        "weighted_metrics": weighted_metrics,
        "activity_model": activity_model,
    }


if __name__ == "__main__":
    main()
