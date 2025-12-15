"""
Training script for Processing Time Prediction ML Model

This script trains a Random Forest Regressor model for predicting processing times
between consecutive events in a process simulation. It includes a training schedule
for optimal performance.

Usage:
    python train_processing_time_model.py [--xes_path PATH] [--model_path PATH] [--tune_hyperparameters]
"""

import argparse
import os
import sys
import time
import pm4py
import pandas as pd
from processing_time_prediction.ProcessingTimePredictionClass import ProcessingTimePredictionClass
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def load_data(xes_path: str = "Dataset\BPI Challenge 2017.xes") -> pd.DataFrame:
    """
    Load XES file and convert to DataFrame.
    
    Args:
        xes_path: Path to XES file
        
    Returns:
        DataFrame with event log data
    """
    print(f"\n{'='*80}")
    print("Loading Event Log Data")
    print(f"{'='*80}")
    
    if not os.path.exists(xes_path):
        raise FileNotFoundError(f"XES file not found: {xes_path}")
    
    print(f"Loading XES file: {xes_path}")
    start_time = time.time()

    log = pm4py.read_xes(xes_path)
    df = pm4py.convert_to_dataframe(log)

    
    load_time = time.time() - start_time
    print(f"\nLoaded {len(df)} events from {df['case:concept:name'].nunique()} cases")
    print(f"Load time: {load_time:.2f} seconds")
    print(f"Columns: {list(df.columns)}")
    
    return df


def train_baseline_model(df: pd.DataFrame, model_save_path: str = "models/processing_time_model"):
    """
    Train baseline model with default hyperparameters.
    
    Args:
        df: Event log DataFrame
        model_save_path: Path to save the trained model
    """
    print(f"\n{'='*80}")
    print("Training Baseline Model")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else ".", exist_ok=True)
    
    # Initialize and train model
    predictor = ProcessingTimePredictionClass(df, method="ml")
    
    # Save model
    predictor.save_model(model_save_path)
    
    train_time = time.time() - start_time
    print(f"\nTotal training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    return predictor


def train_tuned_model(df: pd.DataFrame, model_save_path: str = "models/processing_time_model_tuned"):
    """
    Train model with hyperparameter tuning for better performance.
    
    Args:
        df: Event log DataFrame
        model_save_path: Path to save the trained model
    """
    print(f"\n{'='*80}")
    print("Training Model with Hyperparameter Tuning")
    print(f"{'='*80}")
    
    # First, extract training data
    print("\n[Step 1/4] Extracting training data...")
    temp_predictor = ProcessingTimePredictionClass(df, method="ml")
    X_raw, y = temp_predictor._extract_training_data()
    X = temp_predictor._prepare_features(X_raw, is_training=True)
    
    # Remove outliers
    mean_y = y.mean()
    std_y = y.std()
    outlier_threshold = mean_y + 3 * std_y
    valid_mask = y <= outlier_threshold
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Training samples: {len(X)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Define hyperparameter grid
    print("\n[Step 2/4] Setting up hyperparameter search space...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [20, 30, 40, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    
    # Use RandomizedSearchCV for faster search (or GridSearchCV for exhaustive search)
    print("\n[Step 3/4] Performing hyperparameter search (this may take a while)...")
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0)
    
    # Use RandomizedSearchCV for faster results (50 iterations)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search_start = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - search_start
    
    print(f"\nHyperparameter search completed in {search_time:.2f} seconds ({search_time/60:.2f} minutes)")
    print(f"\nBest parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Evaluate best model
    print("\n[Step 4/4] Evaluating best model...")
    best_model = search.best_estimator_
    
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\nBest Model Performance:")
    print(f"  Training MAE: {train_mae:.2f}s ({train_mae/3600:.2f}h)")
    print(f"  Validation MAE: {val_mae:.2f}s ({val_mae/3600:.2f}h)")
    print(f"  Validation RMSE: {val_rmse:.2f}s ({val_rmse/3600:.2f}h)")
    print(f"  Validation R²: {val_r2:.4f}")
    
    # Create predictor with tuned model
    temp_predictor.ml_model = best_model
    temp_predictor.fallback_mean = float(y.median())
    temp_predictor.fallback_std = float(y.std())
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else ".", exist_ok=True)
    temp_predictor.save_model(model_save_path)
    
    return temp_predictor


def print_training_schedule():
    """
    Print recommended training schedule for optimal performance.
    """
    print("\n" + "="*80)
    print("RECOMMENDED TRAINING SCHEDULE FOR OPTIMAL PERFORMANCE")
    print("="*80)
    print("""
Phase 1: Initial Training (Baseline Model)
-------------------------------------------
1. Load XES dataset
2. Train baseline model with default hyperparameters
   - n_estimators=200, max_depth=30, min_samples_split=10
   - Time: ~5-15 minutes (depending on dataset size)
3. Evaluate baseline performance
4. Save baseline model

Phase 2: Hyperparameter Tuning (Optional, for better performance)
------------------------------------------------------------------
1. Use RandomizedSearchCV with 50 iterations
2. Search space:
   - n_estimators: [100, 200, 300]
   - max_depth: [20, 30, 40, None]
   - min_samples_split: [5, 10, 20]
   - min_samples_leaf: [2, 5, 10]
   - max_features: ['sqrt', 'log2']
3. Time: ~30-60 minutes (depending on dataset size)
4. Save tuned model

Phase 3: Model Evaluation and Selection
---------------------------------------
1. Compare baseline vs tuned model on validation set
2. Check for overfitting (train vs validation performance)
3. Select best model based on validation MAE/RMSE
4. Test on holdout test set (if available)

Phase 4: Production Deployment
-------------------------------
1. Load saved model using load_model()
2. Use in SimulationEngineClass with method="ml"
3. Monitor prediction performance in production
4. Retrain periodically as new data becomes available

PERFORMANCE TIPS:
----------------
1. Use CSV cache (event_log.csv) for faster subsequent loads
2. For large datasets (>1M events), consider sampling for hyperparameter tuning
3. Monitor feature importance to identify most predictive features
4. Consider ensemble methods (stacking) for even better performance
5. Regular retraining (monthly/quarterly) helps adapt to concept drift

EXPECTED PERFORMANCE:
--------------------
- Baseline model: MAE ~2-5 hours, R² ~0.3-0.6 (depending on dataset)
- Tuned model: MAE ~1.5-4 hours, R² ~0.4-0.7 (improvement of 10-30%)
- Processing time prediction is inherently difficult due to high variance
""")


def main():
    parser = argparse.ArgumentParser(
        description="Train ML model for processing time prediction"
    )
    parser.add_argument(
        "--xes_path",
        type=str,
        default="Dataset/BPI Challenge 2017.xes",
        help="Path to XES event log file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/processing_time_model",
        help="Base path for saving the trained model"
    )
    parser.add_argument(
        "--tune_hyperparameters",
        action="store_true",
        help="Perform hyperparameter tuning (slower but better performance)"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Print training schedule and exit"
    )
    
    args = parser.parse_args()
    
    if args.schedule:
        print_training_schedule()
        return
    
    try:
        # Load data
        df = load_data()
        
        if args.tune_hyperparameters:
            # Train with hyperparameter tuning
            model_path = f"{args.model_path}_tuned"
            predictor = train_tuned_model(df, model_path)
        else:
            # Train baseline model
            predictor = train_baseline_model(df, args.model_path)
        
        print("\n" + "="*80)
        print("Training completed successfully!")
        print("="*80)
        print(f"\nModel saved to: {args.model_path}_*.joblib")
        print("\nTo use the model in simulation:")
        print("  engine = SimulationEngineClass(processing_time_method='ml')")
        print("  engine.load_or_prepare_log('path/to/xes')")
        print("  engine.simulate(100)")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
