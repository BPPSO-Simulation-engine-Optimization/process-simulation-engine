#!/usr/bin/env python3
"""
Vollständige Evaluation aller Activity-Specific Modelle mit tabellarischer Ausgabe.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from activity_specific_model import ActivitySpecificModel
from data_loader import DataLoader

def main():
    print("="*80)
    print("ACTIVITY-SPECIFIC MODEL EVALUATION")
    print("="*80)
    print()
    
    # ⚠️ BUSINESS-ZEITEN TEMPORÄR AUSSCHALTEN:
    # Setze USE_BUSINESS_TIME auf False, um Business-Zeiten (9:00-17:00) auszuschalten
    USE_BUSINESS_TIME = True  # ← Ändere hier auf False, um Business-Zeiten auszuschalten
    
    # Create output directory for all evaluation outputs (in the same directory as this script)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data - check if CSV exists, otherwise convert from XES
    CSV_PATH = os.path.join(OUTPUT_DIR, "bpic2017.csv")
    XES_PATH = os.path.join(PARENT_DIR, "BPI Challenge 2017.xes.gz")
    
    print("Loading data...")
    data_loader = DataLoader()
    
    if os.path.exists(CSV_PATH):
        print(f"CSV file found: {CSV_PATH}")
        print("Loading from CSV (fast)...")
        df = data_loader.load_csv_to_dataframe(CSV_PATH)
    else:
        print(f"CSV file not found: {CSV_PATH}")
        print(f"Loading from XES file: {XES_PATH} (this may take a while)...")
        df = data_loader.load_xes_to_dataframe(XES_PATH)
        print(f"Saving to CSV for faster future loading...")
        data_loader.save_dataframe_to_csv(df, CSV_PATH)
        print(f"CSV saved to: {CSV_PATH}")
    
    print(f"Loaded {len(df)} events from {len(df['case_id'].unique())} cases\n")
    
    # Initialize model
    print("Initializing Activity-Specific Model...")
    
    # ⚠️ ÄNDERE HIER DIE FEATURES DYNAMISCH:
    # Du kannst die Liste der Features hier anpassen:
    custom_base_features = [
        "event",                    # Event-Name (kategorisch) - entspricht "concept:name"
        # "lifecycle:transition" removed - not used for prediction
        "org:resource",             # Ressource/Person (kategorisch)
        "EventOrigin",              # Event-Origin (kategorisch)
        "event_index",              # Position im Case (numerisch)
        "hour",                     # Stunde des Tages (numerisch)
        "minute",                   # Minute (numerisch)
        "second",                   # Sekunde (numerisch)
        "microsecond",              # Mikrosekunde (numerisch)
        "weekday",                  # Wochentag (numerisch, 0=Montag)
        "day_of_month",             # Tag des Monats (numerisch)
        "month",                    # Monat (numerisch)
        "day_of_year"               # Tag des Jahres (numerisch)
    ]
    
    custom_case_features = [
        "case:LoanGoal",            # Kategorisch
        "case:ApplicationType",     # Kategorisch
        "case:RequestedAmount",     # Numerisch
        # Features die nur bei O_Create Offer verfügbar sind:
        "FirstWithdrawalAmount",    # Numerisch (nur bei O_Create Offer)
        "NumberOfTerms",            # Numerisch (nur bei O_Create Offer)
        "Accepted",                 # Kategorisch/Boolean (nur bei O_Create Offer)
        "MonthlyCost",              # Numerisch (nur bei O_Create Offer)
        "Selected",                 # Kategorisch/Boolean (nur bei O_Create Offer)
        "CreditScore",              # Numerisch (nur bei O_Create Offer)
        "OfferedAmount",            # Numerisch (nur bei O_Create Offer)
        "OfferID"                   # Kategorisch (nur bei O_Create Offer)
    ]
    
    # Create models directory
    models_dir = os.path.join(OUTPUT_DIR, "activity_models")
    os.makedirs(models_dir, exist_ok=True)
    
    activity_model = ActivitySpecificModel(
        base_features=custom_base_features,      # ← Hier werden die Base-Features übergeben
        case_features=custom_case_features,      # ← Hier werden die Case-Features übergeben
        random_state=42,
        models_dir=models_dir                   # ← Modelle werden hier gespeichert/geladen
    )
    print(f"Will train models for {len(activity_model.activities)} activities\n")
    
    # Prepare data with business-time filtering
    if USE_BUSINESS_TIME:
        print("Preparing activity-specific data (with business-time filtering 9:00-17:00)...")
    else:
        print("Preparing activity-specific data (NO business-time filtering - using all time)...")
    activity_model.prepare_activity_data(df, test_size=0.2, use_business_time=USE_BUSINESS_TIME, output_dir=OUTPUT_DIR)
    print()
    
    # Train models with custom quantiles
    # You can customize quantiles per activity - any number of quantiles is allowed:
    # quantile_config = {
    #     "A_Concept": [0.8, 0.9, 0.95],  # 3 quantiles
    #     "A_Complete": [0.5],  # Only median (1 quantile)
    #     "Other_Activity": [0.5, 0.75, 0.9, 0.95, 0.99]  # 5 quantiles
    # }
    # activity_model.train_activity_models(quantile_config=quantile_config)
    
    print("Training models...")
    # Default: A_Concept uses [0.8, 0.9, 0.95], other activities use standard training
    activity_model.train_activity_models()
    print()
    
    # Evaluate
    print("Evaluating models...")
    results = activity_model.evaluate_activity_models()
    print()
    
    # Get overall metrics
    overall_metrics = activity_model.get_overall_metrics(results)
    weighted_metrics = activity_model.get_weighted_overall_metrics(results)
    
    # Display table
    print("="*80)
    print("DETAILED METRICS PER ACTIVITY")
    print("="*80)
    print()
    
    # Create formatted table with log metrics
    df_display = overall_metrics[['activity', 'mae_hours', 'rmse_hours', 'mae_log', 'rmse_log', 'test_samples']].copy()
    
    # Mark fixed activities
    df_display['is_fixed'] = df_display['activity'].isin(activity_model.fixed_activities)
    df_display['Activity'] = df_display.apply(
        lambda row: f"{row['activity']} [FIXED]" if row['is_fixed'] else row['activity'],
        axis=1
    )
    
    # Sort by MAE, but put fixed activities with no data at the end
    df_display['sort_key'] = df_display.apply(
        lambda row: (row['test_samples'] == 0, row['mae_hours']),
        axis=1
    )
    df_display = df_display.sort_values('sort_key').drop('sort_key', axis=1)
    
    # Rename columns after sorting
    df_display = df_display[['Activity', 'mae_hours', 'rmse_hours', 'mae_log', 'rmse_log', 'test_samples']]
    df_display.columns = ['Activity', 'MAE (hours)', 'RMSE (hours)', 'MAE (log)', 'RMSE (log)', 'Test Samples']
    
    # Format numbers
    df_display['MAE (hours)'] = df_display['MAE (hours)'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
    df_display['RMSE (hours)'] = df_display['RMSE (hours)'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
    df_display['MAE (log)'] = df_display['MAE (log)'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    df_display['RMSE (log)'] = df_display['RMSE (log)'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    df_display['Test Samples'] = df_display['Test Samples'].apply(lambda x: f'{int(x):,}' if pd.notna(x) and x > 0 else '0 (no data)')
    
    print(df_display.to_string(index=False))
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY METRICS (Original Scale & Log Scale)")
    print("="*80)
    print(f"Weighted Mean MAE (hours):  {weighted_metrics['weighted_mean_mae_hours']:.2f}")
    print(f"Weighted Mean RMSE (hours): {weighted_metrics['weighted_mean_rmse_hours']:.2f}")
    print(f"Weighted Mean MAE (log):    {weighted_metrics['weighted_mean_mae_log']:.4f}")
    print(f"Weighted Mean RMSE (log):  {weighted_metrics['weighted_mean_rmse_log']:.4f}")
    print()
    print(f"Unweighted Mean MAE (hours):  {overall_metrics['mae_hours'].mean():.2f}")
    print(f"Unweighted Mean RMSE (hours): {overall_metrics['rmse_hours'].mean():.2f}")
    print(f"Unweighted Mean MAE (log):    {overall_metrics['mae_log'].mean():.4f}")
    print(f"Unweighted Mean RMSE (log):  {overall_metrics['rmse_log'].mean():.4f}")
    print(f"Total Test Samples: {int(weighted_metrics['total_test_samples']):,}")
    print()
    print("Comparison with Unified Model:")
    print(f"  Unified Model MAE:        5.50 hours")
    print(f"  Activity-Specific (weighted): {weighted_metrics['weighted_mean_mae_hours']:.2f} hours")
    diff = weighted_metrics['weighted_mean_mae_hours'] - 5.50
    print(f"  Difference:                {diff:+.2f} hours ({diff/5.50*100:+.1f}%)")
    print()
    
    # Save results (models are already saved during training, but save again to ensure consistency)
    print("Saving results...")
    results_dir = os.path.join(OUTPUT_DIR, "activity_results")
    activity_model.save_evaluation_results(results, results_dir)
    # Models are already saved during training, but save again to ensure consistency
    if activity_model.models_dir:
        activity_model.save_models(activity_model.models_dir)
    print()
    print("="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"Results saved to: {results_dir}/")
    print(f"Models saved to: {models_dir}/")
    print(f"All outputs saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
