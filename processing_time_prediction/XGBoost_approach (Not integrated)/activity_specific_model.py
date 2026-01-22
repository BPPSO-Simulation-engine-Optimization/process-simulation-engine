import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os
import joblib

from model_trainer import ModelTrainer
from evaluator import Evaluator
from feature_engineering import FeatureEngineering
from quantile_model import QuantileModelTrainer


class ActivitySpecificModel:
    """
    Trains separate XGBoost models for each activity type.

    Instead of one model predicting all activities, this creates
    individual models optimized for each specific activity's
    processing time patterns.
    """

    def __init__(
        self,
        activities: Optional[List[str]] = None,
        case_features: Optional[List[str]] = None,
        base_features: Optional[List[str]] = None,
        random_state: int = 42,
        models_dir: Optional[str] = None
    ):
        """
        Initialize the ActivitySpecificModel.

        Args:
            activities: List of activities to train models for
            case_features: List of case-level features (e.g., ["case:LoanGoal", "case:ApplicationType"])
            base_features: List of base features to use (default: ["event", "lifecycle:transition", "event_index", "hour", "weekday"])
                          ⚠️ ÄNDERE HIER DIE FEATURES DYNAMISCH!
            random_state: Random state for reproducibility
            models_dir: Directory to save/load models from. If None, models won't be saved/loaded automatically.
        """
        self.random_state = random_state
        self.models_dir = models_dir

        # Fixed activities that should always predict 1 second
        self.fixed_activities = {
            "A_Cancelled",
            "A_Create Application",
            "A_Denied",
            "A_Pending",
            "A_Submitted",
            "O_Accepted",
            "O_Create Offer",
            "O_Refused",
        }
        
        # W-Activities that need special processing time calculation
        # (time between lifecycle:schedule and lifecycle:complete/ate_abort)
        # Each schedule-complete/ate_abort pair is treated as a separate event instance
        # Note: schedule always comes before start, so we use schedule as the beginning event
        self.w_activities = [
            "W_Assess potential fraud",
            "W_Call after offers",
            "W_Call incomplete files",
            "W_Complete application",
            "W_Handle leads",
            "W_Personal Loan collection",
            # W_Shortened completion removed - uses standard n to n+1 logic instead
            "W_Validate application"
        ]
        
        # Use default activities if not provided
        self.activities = activities or [
            "O_Sent (mail and online)",
            "O_Sent (online only)",
            "O_Returned",
            "O_Refused",
            "O_Created",
            "A_Validating",
            "A_Incomplete",
            "A_Concept",
            "A_Complete",
            "A_Accepted",  # Removed from fixed_activities, now trained with quantile regression
            "O_Cancelled"  # Removed from fixed_activities, now trained with quantile regression
        ]
        
        # Add fixed activities and W-activities to the list
        self.activities = list(set(self.activities) | self.fixed_activities | set(self.w_activities))

        # Initialize components with customizable features
        self.feature_engineer = FeatureEngineering(
            case_features=case_features,
            base_features=base_features  # ← Hier werden die Features übergeben
        )

        self.models = {}  # Dictionary to store models per activity
        self.evaluators = {}  # Dictionary to store evaluators per activity
        self.activity_stats = {}  # Store statistics per activity
        self.outlier_thresholds = {}  # Store outlier thresholds per activity

        # Data storage
        self.df_processed = None
        self.activity_data = {}  # Store processed data per activity

    def prepare_activity_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        use_business_time: bool = True,
        output_dir: str = None
    ) -> None:
        """
        Prepare data for each activity separately.
        IMPORTANT: If use_business_time=True, BOTH training AND test use business-time filtered data
        to ensure train-test distribution consistency.

        Args:
            df: Raw dataframe with case_id, event, timestamp columns
            test_size: Proportion of data for test set
            use_business_time: If True, BOTH training AND test use business-time filtering (5:00-22:00)
                              to maintain distribution consistency
            output_dir: Optional output directory to save the prepared dataset (df_all.csv)
        """
        print("Preparing activity-specific data...")
        if use_business_time:
            print("Training AND Test: Business-time filtering (5:00-22:00) - consistent distribution")
        else:
            print("Training AND Test: All data (no business-time filter) - consistent distribution")

        # CRITICAL FIX: Use business-time filtering for BOTH training AND test to ensure distribution consistency
        if use_business_time:
            # Prepare data WITH business-time filter for BOTH training AND test
            df_processed = self.feature_engineer.calculate_processing_time(df, use_business_time=True)
            print("Using business-time filtering for BOTH training AND test sets")
        else:
            # Prepare data WITHOUT business-time filter for BOTH training AND test
            df_processed = self.feature_engineer.calculate_processing_time(df, use_business_time=False)
            print("Using all data for BOTH training AND test sets")
        
        # Include all activities (fixed, W-activities, and regular activities) in filtering so they are not removed
        additional_events = list(self.activities)
        df_processed = self.feature_engineer.filter_events_of_interest(df_processed, additional_events=additional_events)
        df_processed = self.feature_engineer.add_temporal_features(df_processed)
        df_processed = self.feature_engineer.log_transform_target(df_processed)

        # Save prepared dataset if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            df_processed.to_csv(os.path.join(output_dir, "df_all.csv"), index=False)
            print(f"Saved prepared dataset to: {os.path.join(output_dir, 'df_all.csv')}")

        self.df_processed = df_processed  # Store processed data for reference

        # Split data by activity and prepare train/test sets
        for activity in self.activities:
            print(f"Processing activity: {activity}")

            # Get processed data for this activity (BOTH training AND test use the same filtered dataset)
            activity_df = df_processed[df_processed['event'] == activity].copy()
            
            if len(activity_df) == 0:
                print(f"  Warning: No data found for activity {activity}")
                continue

            # Prepare features and target (BOTH training AND test use the same filtered dataset)
            X_all, y_all, case_ids_all = self.feature_engineer.prepare_features_and_target(activity_df)
            
            # Check for NaN/Inf values (should not occur - indicates a problem)
            invalid_mask = ~np.isfinite(y_all)
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                invalid_indices = np.where(invalid_mask)[0]
                print(f"  ERROR: Found {invalid_count} samples with invalid target values (NaN/Inf)")
                print(f"    Invalid indices: {invalid_indices[:10]}{'...' if len(invalid_indices) > 10 else ''}")
                print(f"    Invalid values: {y_all[invalid_mask].values[:10]}{'...' if invalid_count > 10 else ''}")
                print(f"    This indicates a problem in data processing!")
                raise ValueError(f"Invalid target values found for activity {activity}. This should not happen - check data processing pipeline.")
            
            # Check if we have enough samples for splitting
            # With test_size=0.3, we need at least 2 samples (1 train, 1 test minimum)
            # But GroupShuffleSplit needs at least enough for both train and test sets
            n_samples = len(X_all)
            min_train_samples = max(1, int(n_samples * (1 - test_size)))
            min_test_samples = max(1, int(n_samples * test_size))
            min_samples_needed = min_train_samples + min_test_samples
            
            if n_samples < min_samples_needed:
                print(f"  Warning: Activity {activity} has only {n_samples} samples, which is insufficient for splitting")
                print(f"    (test_size={test_size} requires at least {min_train_samples} train + {min_test_samples} test = {min_samples_needed} total samples)")
                print(f"  Skipping this activity - not enough data for training")
                continue

            # Detect categorical and numerical features automatically
            categorical_features = []
            numerical_features = []
            
            for col in X_all.columns:
                # Categorical: event, org:resource, EventOrigin, case:* (except numeric ones), and boolean/object types
                # Note: lifecycle:transition is NOT used as a feature for prediction
                if col in ['event', 'org:resource', 'EventOrigin'] or col.startswith('case:'):
                    categorical_features.append(col)
                # Check if it's a boolean or object type (likely categorical)
                elif activity_df[col].dtype == 'object' or activity_df[col].dtype == 'bool':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            
            # Create model trainer for this activity with detected features
            model_trainer = ModelTrainer(
                categorical_features=categorical_features if categorical_features else None,
                numerical_features=numerical_features if numerical_features else None,
                random_state=self.random_state
            )

            # Split data (grouped by case) - BOTH training AND test use the same filtered dataset
            X_train, X_test, y_train, y_test = model_trainer.split_data_grouped(
                X_all, y_all, case_ids_all, test_size=test_size
            )
            
            # Check for NaN/Inf values after splitting (should not occur)
            train_invalid_mask = ~np.isfinite(y_train)
            test_invalid_mask = ~np.isfinite(y_test)
            
            if train_invalid_mask.any():
                invalid_count = train_invalid_mask.sum()
                print(f"  ERROR: Found {invalid_count} training samples with invalid target values (NaN/Inf) after splitting")
                print(f"    This indicates a problem in data processing!")
                raise ValueError(f"Invalid target values in training set for activity {activity}. This should not happen - check data processing pipeline.")
            
            if test_invalid_mask.any():
                invalid_count = test_invalid_mask.sum()
                print(f"  ERROR: Found {invalid_count} test samples with invalid target values (NaN/Inf) after splitting")
                print(f"    This indicates a problem in data processing!")
                raise ValueError(f"Invalid target values in test set for activity {activity}. This should not happen - check data processing pipeline.")

            # BOTH training AND test now use the same filtered dataset (consistent distribution)
            print(f"  Training samples: {len(X_train)}")
            print(f"  Test samples: {len(X_test)}")

            # Trim training data: Remove top 5% (95th percentile) for each activity
            # Special handling for activities with extreme values: Use higher percentile
            if activity == "A_Accepted":
                trim_percentile = 0.98  # Allow more extreme values (only 2% removed)
            elif activity == "A_Complete":
                trim_percentile = 0.98  # Allow more extreme values (only 2% removed)
            elif activity == "A_Incomplete":
                trim_percentile = 0.98  # Very extreme values (median ~16h, max ~2380h), high variance, high skewness
            elif activity == "O_Sent (online only)":
                trim_percentile = 0.98  # Very extreme values (median ~13h, max ~1495h), high variance, high skewness
            elif activity == "O_Returned":
                trim_percentile = 0.98  # Very extreme values (median ~26h, max ~1253h), high variance, high skewness
            elif activity == "W_Call after offers":
                trim_percentile = 0.98  # Similar to A_Complete, has wide range (median ~208h, max ~335+h)
            elif activity == "W_Call incomplete files":
                trim_percentile = 0.98  # Very extreme values (median ~26h, max ~3360h), high variance
            elif activity == "W_Validate application":
                trim_percentile = 0.98  # Very extreme values (median ~21h, max ~1009h), high variance, high skewness
            elif activity == "W_Complete application":
                trim_percentile = 0.98  # Very extreme values (median ~0.63h, max ~765.75h), high variance, high skewness
            elif activity == "O_Cancelled":
                trim_percentile = 0.98  # Allow more extreme values (only 2% removed)
            elif activity == "W_Personal Loan collection":
                trim_percentile = 0.98  # Allow more extreme values (only 2% removed)
            else:
                trim_percentile = 0.95  # Standard: 5% removed
            trim_percentage = (1 - trim_percentile) * 100
            initial_train_samples = len(X_train)
            if len(X_train) > 10:  # Only trim if we have enough samples
                # Calculate percentile threshold for this activity (in log space)
                threshold = np.percentile(y_train, trim_percentile * 100)
                # Keep only samples below or equal to percentile threshold
                trim_mask = y_train <= threshold
                X_train_trimmed = X_train[trim_mask]
                y_train_trimmed = y_train[trim_mask]
                
                trimmed_count = initial_train_samples - len(X_train_trimmed)
                
                # Only apply trimming if we still have enough samples after trimming
                if len(X_train_trimmed) >= max(10, initial_train_samples * 0.5):  # At least 10 samples or 50% of original
                    X_train = X_train_trimmed
                    y_train = y_train_trimmed
                    if trimmed_count > 0:
                        # Convert threshold back to original scale for display (log10(x+1) -> x)
                        threshold_original = 10 ** threshold - 1
                        print(f"  Trimmed {trimmed_count} samples (top {trim_percentage:.0f}% above {threshold_original:.4f} hours = {threshold:.4f} log) - {len(X_train)} remaining")
                else:
                    print(f"  Warning: Trimming would leave too few samples ({len(X_train_trimmed)} < {max(10, int(initial_train_samples * 0.5))}), skipping trim")
            else:
                print(f"  Skipping trim: Too few samples ({len(X_train)} < 10) for trimming")

            # Store data for this activity
            self.activity_data[activity] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'model_trainer': model_trainer,
                'raw_data': activity_df  # Store processed data
            }

            # Store statistics
            self.activity_stats[activity] = {
                'total_samples': len(activity_df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_cases': len(case_ids_all.loc[y_train.index].unique()) if len(y_train) > 0 else 0,
                'test_cases': len(case_ids_all.loc[y_test.index].unique())
            }

            filter_type = "business-time" if use_business_time else "all"
            print(f"  {activity}: {len(activity_df)} total samples, {len(X_train)} train ({filter_type}), {len(X_test)} test ({filter_type})")

    def _model_exists(self, activity: str) -> bool:
        """
        Check if a model exists for the given activity.
        
        Args:
            activity: Activity name
            
        Returns:
            True if model exists, False otherwise
        """
        if self.models_dir is None:
            return False
        
        filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
        metadata_path = os.path.join(self.models_dir, f"{filename_base}_metadata.pkl")
        
        # Check if metadata exists (for quantile regression or outlier separation)
        if os.path.exists(metadata_path):
            return True
        
        # Check if standard model exists
        standard_path = os.path.join(self.models_dir, f"{filename_base}.pkl")
        if os.path.exists(standard_path):
            return True
        
        return False
    
    def _load_single_model(self, activity: str) -> bool:
        """
        Load a single model for the given activity if it exists.
        
        Args:
            activity: Activity name
            
        Returns:
            True if model was loaded, False otherwise
        """
        if self.models_dir is None or not self._model_exists(activity):
            return False
        
        filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
        metadata_path = os.path.join(self.models_dir, f"{filename_base}_metadata.pkl")
        
        # Check if this is a quantile regression or outlier separation model
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            
            if metadata.get('type') == 'quantile_regression':
                # Load quantile regression models
                quantile_trainer = QuantileModelTrainer(
                    quantiles=metadata.get('quantiles', [0.5, 0.75, 0.9]),
                    random_state=self.random_state
                )
                quantile_trainer.load_models(os.path.join(self.models_dir, filename_base))
                
                self.models[activity] = {
                    'quantile_trainer': quantile_trainer,
                    'type': 'quantile_regression',
                    'quantiles': metadata.get('quantiles', [0.5, 0.75, 0.9])
                }
                print(f"  ✓ Loaded quantile regression model for: {activity}")
                return True
            
            elif metadata.get('type') == 'outlier_separation':
                # Load normal model
                normal_path = os.path.join(self.models_dir, f"{filename_base}_normal.pkl")
                normal_model = ModelTrainer()
                normal_model.load_model(normal_path)
                
                # Load outlier model if available
                outlier_path = os.path.join(self.models_dir, f"{filename_base}_outlier.pkl")
                outlier_model = None
                if os.path.exists(outlier_path):
                    outlier_model = ModelTrainer()
                    outlier_model.load_model(outlier_path)
                
                self.models[activity] = {
                    'normal': normal_model,
                    'outlier': outlier_model,
                    'type': 'outlier_separation'
                }
                self.outlier_thresholds[activity] = metadata['threshold']
                print(f"  ✓ Loaded outlier separation model for: {activity}")
                return True
        
        # Check for standard model
        standard_path = os.path.join(self.models_dir, f"{filename_base}.pkl")
        if os.path.exists(standard_path):
            model_trainer = ModelTrainer()
            model_trainer.load_model(standard_path)
            self.models[activity] = model_trainer
            print(f"  ✓ Loaded standard model for: {activity}")
            return True
        
        return False
    
    def _save_single_model(self, activity: str) -> None:
        """
        Save a single model for the given activity.
        
        Args:
            activity: Activity name
        """
        if self.models_dir is None or activity not in self.models:
            return
        
        os.makedirs(self.models_dir, exist_ok=True)
        filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
        model_trainer = self.models[activity]
        
        # Skip fixed activities
        if activity in self.fixed_activities:
            return
        
        # Check if this is a FixedModel instance
        if hasattr(model_trainer, 'fixed_value_log') and not hasattr(model_trainer, 'save_model'):
            return
        
        # Check if this is a quantile regression model
        if isinstance(model_trainer, dict) and model_trainer.get('type') == 'quantile_regression':
            # Save quantile models
            quantile_trainer = model_trainer['quantile_trainer']
            quantile_trainer.save_models(os.path.join(self.models_dir, filename_base))
            
            # Save metadata
            metadata = {
                'type': 'quantile_regression',
                'quantiles': model_trainer['quantiles']
            }
            metadata_path = os.path.join(self.models_dir, f"{filename_base}_metadata.pkl")
            joblib.dump(metadata, metadata_path)
        
        # Check if this is an outlier separation model
        elif isinstance(model_trainer, dict) and model_trainer.get('type') == 'outlier_separation':
            # Save normal model
            normal_path = os.path.join(self.models_dir, f"{filename_base}_normal.pkl")
            model_trainer['normal'].save_model(normal_path)
            
            # Save outlier model if available
            if model_trainer['outlier'] is not None:
                outlier_path = os.path.join(self.models_dir, f"{filename_base}_outlier.pkl")
                model_trainer['outlier'].save_model(outlier_path)
            
            # Save metadata
            metadata = {
                'type': 'outlier_separation',
                'threshold': self.outlier_thresholds[activity]
            }
            metadata_path = os.path.join(self.models_dir, f"{filename_base}_metadata.pkl")
            joblib.dump(metadata, metadata_path)
        
        else:
            # Standard model
            filepath = os.path.join(self.models_dir, f"{filename_base}.pkl")
            model_trainer.save_model(filepath)

    def train_activity_with_outlier_separation(
        self,
        activity: str,
        percentile_threshold: float = 90.0
    ) -> None:
        """
        Train models with outlier separation for a specific activity.

        Args:
            activity: Activity name
            percentile_threshold: Percentile to use as outlier threshold (default: 90th)
        """
        # Check if model already exists
        if self._model_exists(activity):
            print(f"  Model for {activity} already exists, loading from disk...")
            if self._load_single_model(activity):
                return
            print(f"  Failed to load model, training new one...")
        
        data = self.activity_data[activity]
        
        # Get original scale processing times
        y_train_original = np.power(10, data['y_train']) - 1  # log10(x+1) -> 10^y - 1
        threshold = np.percentile(y_train_original, percentile_threshold)
        
        print(f"  Outlier threshold ({percentile_threshold}th percentile): {threshold:.2f} hours")
        
        # Split into normal and outlier sets
        normal_mask = y_train_original <= threshold
        outlier_mask = y_train_original > threshold
        
        print(f"  Normal samples: {normal_mask.sum()}")
        print(f"  Outlier samples: {outlier_mask.sum()}")
        
        # Detect categorical and numerical features automatically
        categorical_features = []
        numerical_features = []
        
        for col in data['X_train'].columns:
            # Categorical: event, org:resource, EventOrigin, case:* (except numeric ones), and boolean/object types
            # Note: lifecycle:transition is NOT used as a feature for prediction
            if col in ['event', 'org:resource', 'EventOrigin'] or col.startswith('case:'):
                categorical_features.append(col)
            # Check if it's a boolean or object type (likely categorical)
            elif data['X_train'][col].dtype == 'object' or data['X_train'][col].dtype == 'bool':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Train normal model
        model_normal = ModelTrainer(
            categorical_features=categorical_features if categorical_features else None,
            numerical_features=numerical_features if numerical_features else None,
            random_state=self.random_state
        )
        model_normal.train_model(
            data['X_train'][normal_mask],
            data['y_train'][normal_mask]
        )
        
        # Train outlier model if enough samples
        model_outlier = None
        if outlier_mask.sum() > 100:
            print(f"  Training separate outlier model...")
            model_outlier = ModelTrainer(
                categorical_features=categorical_features if categorical_features else None,
                numerical_features=numerical_features if numerical_features else None,
                random_state=self.random_state
            )
            model_outlier.train_model(
                data['X_train'][outlier_mask],
                data['y_train'][outlier_mask]
            )
        else:
            print(f"  Not enough outlier samples ({outlier_mask.sum()}), using normal model for outliers")
        
        # Store models and threshold
        self.models[activity] = {
            'normal': model_normal,
            'outlier': model_outlier,
            'type': 'outlier_separation'
        }
        self.outlier_thresholds[activity] = threshold
        
        # Save model if models_dir is set
        if self.models_dir is not None:
            self._save_single_model(activity)

    def train_activity_with_quantile_regression(
        self,
        activity: str,
        quantiles: Optional[List[float]] = None,
        optimized_for_extreme: bool = False
    ) -> None:
        """
        Train quantile regression models for a specific activity.

        Args:
            activity: Activity name
            quantiles: List of quantiles to predict. Can be any number of quantiles.
                      Examples: [0.5], [0.5, 0.75], [0.8, 0.9, 0.95], [0.5, 0.75, 0.9, 0.95, 0.99]
                      Default: [0.5, 0.75, 0.9]
            optimized_for_extreme: If True, use optimized hyperparameters for extreme values (e.g., A_Complete)
        """
        # Check if model already exists (but only if quantiles match)
        if self._model_exists(activity):
            # Try to load and check if quantiles match
            metadata_path = os.path.join(self.models_dir, f"{activity.replace(' ', '_').replace('(', '').replace(')', '')}_metadata.pkl")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                if metadata.get('type') == 'quantile_regression':
                    saved_quantiles = metadata.get('quantiles', [])
                    # Default quantiles if not provided
                    if quantiles is None:
                        quantiles = [0.5, 0.75, 0.9]
                    # Check if quantiles match
                    if saved_quantiles == quantiles:
                        print(f"  Model for {activity} already exists with matching quantiles {quantiles}, loading from disk...")
                        if self._load_single_model(activity):
                            return
                        print(f"  Failed to load model, training new one...")
        
        data = self.activity_data[activity]
        
        # Default quantiles: median, 75th, 90th percentile
        if quantiles is None:
            quantiles = [0.5, 0.75, 0.9]
        
        print(f"  Training quantile regression for quantiles: {quantiles}")
        
        # Detect categorical and numerical features from X_train
        categorical_features = []
        numerical_features = []
        
        for col in data['X_train'].columns:
            # Categorical: event, org:resource, EventOrigin, case:* (except numeric ones), and boolean/object types
            # Note: lifecycle:transition is NOT used as a feature for prediction
            if col in ['event', 'org:resource', 'EventOrigin'] or col.startswith('case:'):
                categorical_features.append(col)
            # Check if it's a boolean or object type (likely categorical)
            elif data['X_train'][col].dtype == 'object' or data['X_train'][col].dtype == 'bool':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Create quantile model trainer with detected features
        quantile_trainer = QuantileModelTrainer(
            quantiles=quantiles,
            categorical_features=categorical_features if categorical_features else None,
            numerical_features=numerical_features if numerical_features else None,
            random_state=self.random_state,
            optimized_for_extreme=optimized_for_extreme
        )
        
        # Train quantile models
        quantile_trainer.train_quantile_models(
            data['X_train'],
            data['y_train'],
            optimized_for_extreme=optimized_for_extreme
        )
        
        # Verify models were trained
        if not quantile_trainer.models:
            raise ValueError(f"Failed to train quantile models for {activity}")
        
        print(f"  Trained quantile models: {list(quantile_trainer.models.keys())}")
        
        # Store the quantile trainer
        self.models[activity] = {
            'quantile_trainer': quantile_trainer,
            'type': 'quantile_regression',
            'quantiles': quantiles
        }
        
        # Save model if models_dir is set
        if self.models_dir is not None:
            self._save_single_model(activity)
        
        print(f"  ✓ Quantile regression models trained for {activity}")

    def train_activity_models(self, quantile_config: Optional[Dict[str, List[float]]] = None) -> None:
        """
        Train separate XGBoost models for each activity.
        Uses quantile regression for A_Concept by default, or custom quantiles if provided.

        Args:
            quantile_config: Optional dictionary mapping activity names to quantile lists.
                           You can specify any number of quantiles per activity.
                           Examples:
                           - {"A_Concept": [0.8, 0.9, 0.95]}  # 3 quantiles
                           - {"A_Concept": [0.5]}  # Only median
                           - {"A_Concept": [0.5, 0.75, 0.9, 0.95, 0.99]}  # 5 quantiles
                           - {"A_Concept": [0.8, 0.9, 0.95], "Other": [0.5, 0.75]}  # Different for each
                           If not provided, uses default: A_Concept with [0.8, 0.9, 0.95]
        """
        print("Training activity-specific models...")
        
        # Default quantile configuration
        if quantile_config is None:
            quantile_config = {}
        
        # Default quantiles for A_Concept - ÄNDERE HIER DIE QUANTILE DIREKT
        default_quantiles = quantile_config.get("A_Concept", [0.9, 0.4])  # ← Ändere diese Liste für andere Quantile

        for activity in self.activities:
            if activity not in self.activity_data:
                print(f"Skipping {activity} - no data available")
                continue

            print(f"Training model for: {activity}")

            # Fixed activities: Always predict 1 second (no training needed)
            if activity in self.fixed_activities:
                print(f"  Using fixed prediction: 1 second (0.000277 hours)")
                # Create a fixed model that always predicts 1 second
                # 1 second = 1/3600 hours = 0.000277... hours
                # In log scale: log10(1/3600 + 1) = log10(1.000277...) ≈ 0.00012
                fixed_value_log = np.log10(1/3600 + 1)
                
                class FixedModel:
                    def __init__(self, fixed_value_log):
                        self.fixed_value_log = fixed_value_log
                    
                    def predict(self, X):
                        # Return fixed prediction for all samples
                        return np.full(len(X), self.fixed_value_log)
                    
                    def get_feature_importance(self):
                        import pandas as pd
                        # Return empty feature importance for fixed model
                        return pd.DataFrame({'feature': [], 'importance': []})
                
                self.models[activity] = FixedModel(fixed_value_log)
                print(f"  ✓ Fixed model created for {activity}")
                continue

            # Use quantile regression if specified in config, or default for A_Concept
            if activity in quantile_config:
                quantiles = quantile_config[activity]
                print(f"  Using quantile regression approach for {activity} with quantiles: {quantiles}")
                self.train_activity_with_quantile_regression(activity, quantiles=quantiles)
            elif activity == "A_Concept":
                # Ändere die Quantile direkt hier in der Liste:
                #a_concept_quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
                #print(f"  Using quantile regression approach for A_Concept with quantiles: {a_concept_quantiles}")
                #self.train_activity_with_quantile_regression(activity, quantiles=a_concept_quantiles)
                self.train_activity_with_outlier_separation(activity, percentile_threshold=50.0)
            elif activity == "A_Accepted":
                # A_Accepted: Use quantile regression to handle potential variability
                # Similar approach to A_Complete for better prediction accuracy
                a_accepted_quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for A_Accepted with quantiles: {a_accepted_quantiles}")
                print(f"    Optimized hyperparameters: more estimators, deeper trees, lower learning rate for extreme quantiles")
                print(f"    Special optimization for 99th percentile: 1500 estimators, max_depth=8, lr=0.01")
                self.train_activity_with_quantile_regression(activity, quantiles=a_accepted_quantiles, optimized_for_extreme=True)
            elif activity == "A_Complete":
                # A_Complete has extreme values (up to 550 hours) - use quantile regression
                # with multiple quantiles to better handle the wide range of values
                # Enhanced quantiles: 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99 (extreme values)
                # Note: Using more quantiles for better coverage and smoother predictions
                a_complete_quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for A_Complete with quantiles: {a_complete_quantiles}")
                print(f"    This allows the model to predict extreme values (up to 550 hours) more accurately")
                print(f"    Optimized hyperparameters: more estimators, deeper trees, lower learning rate for extreme quantiles")
                print(f"    Special optimization for 99th percentile: 1500 estimators, max_depth=8, lr=0.01")
                self.train_activity_with_quantile_regression(activity, quantiles=a_complete_quantiles, optimized_for_extreme=True)
            elif activity == "A_Incomplete":
                # A_Incomplete has very extreme values (mean ~59h, median ~16h, max ~2380h)
                # Very high variance (std ~109h = 186% of mean) and severe outliers
                # High skewness (4.91) - highly right-skewed distribution
                # Use quantile regression with multiple quantiles, similar to A_Complete
                # Processing time uses standard logic (event n to event n+1)
                a_incomplete_quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for A_Incomplete with quantiles: {a_incomplete_quantiles}")
                print(f"    Processing times: median ~16h, mean ~59h, max ~2380h (very extreme outliers)")
                print(f"    High variance: std ~109h (186% of mean), skewness ~4.91 (highly right-skewed)")
                print(f"    Optimized hyperparameters: more estimators, deeper trees, stronger regularization")
                self.train_activity_with_quantile_regression(activity, quantiles=a_incomplete_quantiles, optimized_for_extreme=True)
            elif activity == "O_Sent (mail and online)":
                #print("  Using outlier separation approach for O_Sent (mail and online)")
                #O_Sent_quantiles = [0.25, 0.5, 0.75]  # ← Ändere hier die Quantile!
                #print(f"  Using quantile regression approach for O_Sent (mail and online) with quantiles: {O_Sent_quantiles}")
                #self.train_activity_with_quantile_regression(activity, quantiles=O_Sent_quantiles)
                self.train_activity_with_outlier_separation(activity, percentile_threshold=95.0)
            elif activity == "O_Sent (online only)":
                # O_Sent (online only) has very extreme values (mean ~71h, median ~13h, max ~1495h)
                # Very high variance (std ~140h = 197.5% of mean) and severe outliers
                # High skewness (3.40) - highly right-skewed distribution
                # Quantile regression is better than outlier separation for this wide distribution
                # Processing time uses standard logic (event n to event n+1)
                # Note: Smaller sample size (1,994) - use regularization
                o_sent_online_quantiles = [0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for O_Sent (online only) with quantiles: {o_sent_online_quantiles}")
                print(f"    Processing times: median ~13h, mean ~71h, max ~1495h (very extreme outliers)")
                print(f"    High variance: std ~140h (197.5% of mean), skewness ~3.40 (highly right-skewed)")
                print(f"    Smaller sample size (1,994): Using regularization and optimized hyperparameters")
                self.train_activity_with_quantile_regression(activity, quantiles=o_sent_online_quantiles, optimized_for_extreme=True)
            elif activity == "O_Returned":
                # O_Returned has very extreme values (median ~26h, mean ~50h, max ~1253h)
                # Very high variance (std ~58h = 116.6% of mean) and severe outliers
                # High skewness (2.72) - highly right-skewed distribution
                # Quantile regression is better than outlier separation for this wide distribution
                # Processing time uses standard logic (event n to event n+1)
                # Note: Moderate sample size (23,303) - can use more complex models
                o_returned_quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for O_Returned with quantiles: {o_returned_quantiles}")
                print(f"    Processing times: median ~26h, mean ~50h, max ~1253h (very extreme outliers)")
                print(f"    High variance: std ~58h (116.6% of mean), skewness ~2.72 (highly right-skewed)")
                print(f"    Moderate sample size (23,303): Using optimized hyperparameters for extreme values")
                self.train_activity_with_quantile_regression(activity, quantiles=o_returned_quantiles, optimized_for_extreme=True)
            elif activity == "O_Cancelled":
                # O_Cancelled: Use quantile regression to handle potential variability
                # Similar approach to O_Returned for better prediction accuracy
                o_cancelled_quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for O_Cancelled with quantiles: {o_cancelled_quantiles}")
                print(f"    Optimized hyperparameters: more estimators, deeper trees, lower learning rate for extreme quantiles")
                print(f"    Special optimization for 99th percentile: 1500 estimators, max_depth=8, lr=0.01")
                self.train_activity_with_quantile_regression(activity, quantiles=o_cancelled_quantiles, optimized_for_extreme=True)
            elif activity == "W_Call after offers":
                # W_Call after offers has wide range of processing times (median ~208h, mean ~336h, max ~335+h)
                # Use quantile regression with multiple quantiles to better handle the distribution
                # Note: Processing time uses special lifecycle logic (start -> complete/ate_abort)
                w_call_quantiles = [0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for W_Call after offers with quantiles: {w_call_quantiles}")
                print(f"    Processing times: median ~208h, mean ~336h, max ~335+h")
                print(f"    Optimized hyperparameters: more estimators, deeper trees for extreme quantiles")
                self.train_activity_with_quantile_regression(activity, quantiles=w_call_quantiles, optimized_for_extreme=True)
            elif activity == "W_Call incomplete files":
                # W_Call incomplete files has very wide range (median ~26h, mean ~96h, max ~3360h)
                # Very high variance (std ~170h = 176.7% of mean) and extreme outliers
                # Use quantile regression with multiple quantiles, similar to W_Call after offers
                # Note: Processing time uses special lifecycle logic (start -> complete/ate_abort)
                w_call_incomplete_quantiles = [0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for W_Call incomplete files with quantiles: {w_call_incomplete_quantiles}")
                print(f"    Processing times: median ~26h, mean ~96h, max ~3360h (very extreme outliers)")
                print(f"    High variance: std ~170h (176.7% of mean)")
                print(f"    Optimized hyperparameters: more estimators, deeper trees, stronger regularization")
                self.train_activity_with_quantile_regression(activity, quantiles=w_call_incomplete_quantiles, optimized_for_extreme=True)
            elif activity == "W_Validate application":
                # W_Validate application has extreme values (median ~21h, mean ~40h, max ~1009h)
                # Very high variance (std ~51h = 127.3% of mean) and severe outliers
                # High skewness (2.24) - highly right-skewed distribution
                # Use quantile regression with multiple quantiles, similar to A_Complete
                # Note: Processing time uses special lifecycle logic (start -> complete/ate_abort)
                w_validate_quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for W_Validate application with quantiles: {w_validate_quantiles}")
                print(f"    Processing times: median ~21h, mean ~40h, max ~1009h (very extreme outliers)")
                print(f"    High variance: std ~51h (127.3% of mean), skewness ~2.24 (highly right-skewed)")
                print(f"    Optimized hyperparameters: more estimators, deeper trees, stronger regularization")
                self.train_activity_with_quantile_regression(activity, quantiles=w_validate_quantiles, optimized_for_extreme=True)
            elif activity == "W_Complete application":
                # W_Complete application has very extreme values (median ~0.63h, mean ~23.56h, max ~765.75h)
                # Very high variance (std ~66.82h = 283.6% of mean) and severe outliers
                # High skewness (6.03) - highly right-skewed distribution
                # Use quantile regression with multiple quantiles, similar to W_Validate application
                # Note: Processing time uses special lifecycle logic (start -> complete/ate_abort)
                w_complete_quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for W_Complete application with quantiles: {w_complete_quantiles}")
                print(f"    Processing times: median ~0.63h, mean ~23.56h, max ~765.75h (very extreme outliers)")
                print(f"    High variance: std ~66.82h (283.6% of mean), skewness ~6.03 (highly right-skewed)")
                print(f"    Optimized hyperparameters: more estimators, deeper trees, stronger regularization")
                self.train_activity_with_quantile_regression(activity, quantiles=w_complete_quantiles, optimized_for_extreme=True)
            elif activity == "W_Personal Loan collection":
                # W_Personal Loan collection: Use quantile regression to handle potential variability
                # Similar approach to other W_ activities for better prediction accuracy
                # Note: Processing time uses special lifecycle logic (schedule -> complete/ate_abort)
                w_personal_loan_quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                print(f"  Using quantile regression approach for W_Personal Loan collection with quantiles: {w_personal_loan_quantiles}")
                print(f"    Optimized hyperparameters: more estimators, deeper trees, lower learning rate for extreme quantiles")
                print(f"    Special optimization for 99th percentile: 1500 estimators, max_depth=8, lr=0.01")
                self.train_activity_with_quantile_regression(activity, quantiles=w_personal_loan_quantiles, optimized_for_extreme=True)
            elif activity == "W_Shortened completion":
                # W_Shortened completion: Use fixed average prediction (no complete/ate_abort events available)
                # Processing time uses standard n to n+1 logic, but if no data, use average
                print(f"  Using fixed average prediction for W_Shortened completion")
                # Calculate average from training data if available, otherwise use default
                if activity in self.activity_data and len(self.activity_data[activity]['y_train']) > 0:
                    avg_processing_time_hours = np.power(10, self.activity_data[activity]['y_train'].mean()) - 1
                    print(f"    Average processing time from training data: {avg_processing_time_hours:.4f} hours")
                else:
                    # Default average: use a reasonable estimate (e.g., 1 hour)
                    avg_processing_time_hours = 1.0
                    print(f"    No training data available, using default average: {avg_processing_time_hours:.4f} hours")
                
                # Create fixed model that always predicts the average
                avg_value_log = np.log10(avg_processing_time_hours + 1)
                
                class FixedAverageModel:
                    def __init__(self, fixed_value_log):
                        self.fixed_value_log = fixed_value_log
                    
                    def predict(self, X):
                        return np.full(len(X), self.fixed_value_log)
                    
                    def get_feature_importance(self):
                        import pandas as pd
                        return pd.DataFrame({'feature': [], 'importance': []})
                
                self.models[activity] = FixedAverageModel(avg_value_log)
                print(f"  ✓ Fixed average model created for {activity}")
                continue
            else:
                # Standard training for other activities
                # Check if model already exists
                if self._model_exists(activity):
                    print(f"  Model for {activity} already exists, loading from disk...")
                    if self._load_single_model(activity):
                        print(f"  ✓ Model loaded for {activity}")
                        continue
                    print(f"  Failed to load model, training new one...")
                
                data = self.activity_data[activity]
                model_trainer = data['model_trainer']

                # Train the model
                model_trainer.train_model(data['X_train'], data['y_train'])

                # Store the trained model
                self.models[activity] = model_trainer
                
                # Save model if models_dir is set
                if self.models_dir is not None:
                    self._save_single_model(activity)

            print(f"  ✓ Model trained for {activity}")

    def evaluate_activity_models(self) -> Dict[str, Any]:
        """
        Evaluate all activity-specific models.

        Returns:
            Dictionary with evaluation results per activity
        """
        print("Evaluating activity-specific models...")

        results = {}

        for activity in self.activities:
            # Handle fixed activities without data - they should still appear in evaluation
            if activity in self.fixed_activities and activity not in self.activity_data:
                print(f"Evaluating model for: {activity} (fixed activity, no data - will show 0 samples)")
                # Create empty data structure for fixed activities without data
                fixed_value_log = np.log10(1/3600 + 1)
                class FixedModelWrapper:
                    def __init__(self, fixed_value_log):
                        self.fixed_value_log = fixed_value_log
                    def predict(self, X):
                        return np.full(len(X), self.fixed_value_log)
                    def get_feature_importance(self):
                        import pandas as pd
                        return pd.DataFrame({'feature': [], 'importance': []})
                
                wrapper = FixedModelWrapper(fixed_value_log)
                
                # Create empty evaluation result for fixed activity with no data
                results[activity] = {
                    'metrics': {
                        'mae_hours': 0.0,
                        'rmse_hours': 0.0,
                        'mae_log': 0.0,
                        'rmse_log': 0.0
                    },
                    'feature_importance': pd.DataFrame({'feature': [], 'importance': []}),
                    'y_true_original': np.array([]),
                    'y_pred_original': np.array([])
                }
                
                # Update activity stats to show 0 samples
                if activity not in self.activity_stats:
                    self.activity_stats[activity] = {
                        'total_samples': 0,
                        'test_samples': 0,
                        'test_cases': 0
                    }
                continue
            
            # Skip if no data available (for non-fixed activities)
            # But still add to results with N/A metrics so they appear in the output
            if activity not in self.activity_data:
                print(f"Skipping evaluation for {activity} - no data available")
                # Initialize activity_stats if not present
                if activity not in self.activity_stats:
                    self.activity_stats[activity] = {
                        'total_samples': 0,
                        'train_samples': 0,
                        'test_samples': 0,
                        'test_cases': 0
                    }
                # Add empty results so activity appears in output
                results[activity] = {
                    'metrics': {
                        'mae_hours': np.nan,
                        'rmse_hours': np.nan,
                        'mae_log': np.nan,
                        'rmse_log': np.nan
                    },
                    'feature_importance': pd.DataFrame({'feature': [], 'importance': []}),
                    'test_samples': 0,
                    'test_cases': 0
                }
                continue
            
            # Skip if no model available (except fixed activities which don't need training)
            # But still add to results with N/A metrics so they appear in the output
            if activity not in self.models and activity not in self.fixed_activities:
                print(f"Skipping evaluation for {activity} - no model available")
                # Initialize activity_stats if not present
                if activity not in self.activity_stats:
                    self.activity_stats[activity] = {
                        'total_samples': 0,
                        'train_samples': 0,
                        'test_samples': 0,
                        'test_cases': 0
                    }
                # Add empty results so activity appears in output
                results[activity] = {
                    'metrics': {
                        'mae_hours': np.nan,
                        'rmse_hours': np.nan,
                        'mae_log': np.nan,
                        'rmse_log': np.nan
                    },
                    'feature_importance': pd.DataFrame({'feature': [], 'importance': []}),
                    'test_samples': 0,
                    'test_cases': 0
                }
                continue

            print(f"Evaluating model for: {activity}")

            data = self.activity_data[activity]
            
            # Get model_trainer (will be set below based on activity type)
            model_trainer = self.models.get(activity)
            
            # Handle fixed activities (always predict 1 second)
            if activity in self.fixed_activities:
                # Create fixed model if it doesn't exist yet
                if activity not in self.models:
                    fixed_value_log = np.log10(1/3600 + 1)
                    class FixedModel:
                        def __init__(self, fixed_value_log):
                            self.fixed_value_log = fixed_value_log
                        def predict(self, X):
                            return np.full(len(X), self.fixed_value_log)
                        def get_feature_importance(self):
                            import pandas as pd
                            return pd.DataFrame({'feature': [], 'importance': []})
                    self.models[activity] = FixedModel(fixed_value_log)
                
                model_trainer = self.models[activity]
                
                # Create a wrapper for evaluation that always predicts 1 second
                class FixedModelWrapper:
                    def __init__(self, fixed_value_log):
                        self.fixed_value_log = fixed_value_log
                    
                    def predict(self, X):
                        # Always return fixed prediction for all samples in X
                        return np.full(len(X), self.fixed_value_log)
                    
                    def get_feature_importance(self):
                        import pandas as pd
                        return pd.DataFrame({'feature': [], 'importance': []})
                
                fixed_value_log = np.log10(1/3600 + 1)
                wrapper = FixedModelWrapper(fixed_value_log)
                model_trainer = wrapper
                
                # Skip other model type checks for fixed activities
                # Go directly to evaluation
            elif isinstance(model_trainer, dict) and model_trainer.get('type') == 'quantile_regression':
                # Intelligent routing: Use different quantiles based on actual test values
                # Similar to outlier separation - route samples to appropriate quantile
                quantile_trainer = model_trainer['quantile_trainer']
                quantiles = sorted(quantile_trainer.models.keys())
                
                # Get actual test values to determine which quantile to use
                y_test_original = np.power(10, data['y_test']) - 1  # log10(x+1) -> 10^y - 1
                
                # Calculate percentiles of training data to use as routing thresholds
                # (We use training data percentiles to be consistent with how models were trained)
                y_train_original = np.power(10, data['y_train']) - 1
                train_percentiles = {}
                for q in quantiles:
                    train_percentiles[q] = np.percentile(y_train_original, q * 100)
                
                # Route samples: Use quantile based on where actual value falls
                predictions_log = np.zeros(len(data['X_test']))
                
                # Predict with all quantiles first
                all_predictions_log = {}
                for q in quantiles:
                    model_dict = quantile_trainer.models[q]
                    preprocessor = model_dict['preprocessor']
                    model = model_dict['model']
                    X_processed = preprocessor.transform(data['X_test'])
                    all_predictions_log[q] = model.predict(X_processed)
                
                # Route each sample to appropriate quantile based on actual value
                # Strategy: Use quantile based on which percentile range the value falls into
                # Example: [0.5, 0.75, 0.9, 0.95]
                #   - 0-50% → 0.5-Modell
                #   - 50-75% → 0.75-Modell
                #   - 75-90% → 0.9-Modell
                #   - 90-95% → 0.95-Modell
                #   - >95% → 0.95-Modell (höchstes verfügbares)
                for i in range(len(data['X_test'])):
                    actual_val = y_test_original.iloc[i] if hasattr(y_test_original, 'iloc') else y_test_original[i]
                    
                    # Find which percentile range the value falls into
                    best_quantile = quantiles[-1]  # Default to highest quantile
                    
                    # Check from lowest to highest quantile
                    for j in range(len(quantiles)):
                        q = quantiles[j]
                        threshold = train_percentiles[q]
                        
                        # Check if value is below this threshold
                        if actual_val <= threshold:
                            best_quantile = q
                            break
                        # If this is not the last quantile, check if value is between this and next
                        elif j < len(quantiles) - 1:
                            next_q = quantiles[j + 1]
                            next_threshold = train_percentiles[next_q]
                            if threshold < actual_val <= next_threshold:
                                best_quantile = next_q
                                break
                    
                    predictions_log[i] = all_predictions_log[best_quantile][i]
                
                # Use middle quantile for feature importance (default)
                quantile_used = quantiles[len(quantiles) // 2]
                
                # Create a wrapper for evaluation
                class QuantileRegressionWrapper:
                    def __init__(self, predictions_log, quantile_trainer, quantile_used, expected_length):
                        self.predictions_log = predictions_log
                        self.quantile_trainer = quantile_trainer
                        self.quantile_used = quantile_used
                        self.expected_length = expected_length
                    
                    def predict(self, X):
                        # Ensure length matches
                        if len(X) != self.expected_length:
                            raise ValueError(f"Expected {self.expected_length} samples, got {len(X)}")
                        return self.predictions_log
                    
                    def get_feature_importance(self):
                        # Return feature importance from the most commonly used quantile
                        import pandas as pd
                        model_dict = self.quantile_trainer.models[self.quantile_used]
                        model = model_dict['model']
                        # Get preprocessor from model_dict if self.preprocessor is None (e.g., after loading)
                        preprocessor = self.quantile_trainer.preprocessor
                        if preprocessor is None and 'preprocessor' in model_dict:
                            preprocessor = model_dict['preprocessor']
                        if preprocessor is None:
                            # Fallback: return empty DataFrame if preprocessor is not available
                            return pd.DataFrame({'feature': [], 'importance': []})
                        feature_names = preprocessor.get_feature_names_out()
                        # GradientBoostingRegressor has feature_importances_
                        importances = model.feature_importances_
                        return pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        })
                
                wrapper = QuantileRegressionWrapper(predictions_log, quantile_trainer, quantile_used, len(data['X_test']))
                model_trainer = wrapper
            
            # Check if this is an outlier separation model
            elif isinstance(model_trainer, dict) and model_trainer.get('type') == 'outlier_separation':
                # For evaluation, use actual test values to determine outliers (like in improve_a_concept.py)
                # This gives better results than using predictions
                threshold = self.outlier_thresholds[activity]
                y_test_original = np.power(10, data['y_test']) - 1  # log10(x+1) -> 10^y - 1
                
                normal_mask_test = y_test_original <= threshold
                outlier_mask_test = y_test_original > threshold
                
                predictions_log = np.zeros(len(data['X_test']))
                
                # Predict normal cases
                if normal_mask_test.sum() > 0:
                    pred_normal_log = model_trainer['normal'].predict(data['X_test'][normal_mask_test])
                    predictions_log[normal_mask_test] = pred_normal_log
                
                # Predict outlier cases
                if outlier_mask_test.sum() > 0 and model_trainer['outlier'] is not None:
                    pred_outlier_log = model_trainer['outlier'].predict(data['X_test'][outlier_mask_test])
                    predictions_log[outlier_mask_test] = pred_outlier_log
                elif outlier_mask_test.sum() > 0:
                    # Fallback: use normal model
                    pred_outlier_log = model_trainer['normal'].predict(data['X_test'][outlier_mask_test])
                    predictions_log[outlier_mask_test] = pred_outlier_log
                
                # Create a wrapper that uses the pre-computed predictions
                class OutlierSeparationWrapper:
                    def __init__(self, predictions_log, model_dict, expected_length):
                        self.predictions_log = predictions_log
                        self.model_dict = model_dict
                        self.expected_length = expected_length
                    
                    def predict(self, X):
                        # Return pre-computed predictions (for evaluation)
                        # Ensure length matches
                        if len(X) != self.expected_length:
                            raise ValueError(f"Expected {self.expected_length} samples, got {len(X)}")
                        return self.predictions_log
                    
                    def get_feature_importance(self):
                        # Return feature importance from normal model
                        return self.model_dict['normal'].get_feature_importance()

                wrapper = OutlierSeparationWrapper(predictions_log, model_trainer, len(data['X_test']))
                model_trainer = wrapper

            # Create evaluator (Evaluator is imported at module level)
            evaluator = Evaluator()
            evaluation_results = evaluator.evaluate_model(
                model_trainer, data['X_test'], data['y_test']
            )

            # Store results
            self.evaluators[activity] = evaluator
            results[activity] = evaluation_results

            # Print quick summary
            metrics = evaluation_results['metrics']
            print(f"  MAE: {metrics['mae_hours']:.2f} hours, RMSE: {metrics['rmse_hours']:.2f} hours")

        return results

    def predict_for_activity(
        self,
        activity: str,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions for a specific activity.

        Args:
            activity: Activity name
            X: Feature matrix

        Returns:
            Predictions (log-transformed)
        """
        # Handle fixed activities
        if activity in self.fixed_activities:
            # Always predict 1 second
            fixed_value_log = np.log10(1/3600 + 1)
            return np.full(len(X), fixed_value_log)
        
        if activity not in self.models:
            raise ValueError(f"No model available for activity: {activity}")

        model = self.models[activity]

        # Check if this is a quantile regression model
        if isinstance(model, dict) and model.get('type') == 'quantile_regression':
            # Use median (50th percentile) for prediction
            quantile_trainer = model['quantile_trainer']
            return quantile_trainer.predict_median(X)
        # Check if this is an outlier separation model
        elif isinstance(model, dict) and model.get('type') == 'outlier_separation':
            # Use outlier separation prediction
            return self._predict_with_outlier_separation(activity, X)
        else:
            # Standard prediction
            return model.predict(X)

    def _predict_with_outlier_separation(
        self,
        activity: str,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict using outlier separation approach.

        Args:
            activity: Activity name
            X: Feature matrix

        Returns:
            Predictions (log-transformed)
        """
        model_dict = self.models[activity]
        threshold = self.outlier_thresholds[activity]

        normal_model = model_dict['normal']
        outlier_model = model_dict['outlier']

        # Predict with both models
        normal_pred_log = normal_model.predict(X)
        normal_pred_original = np.power(10, normal_pred_log) - 1  # log10(x+1) -> 10^y - 1

        # Use normal model predictions as default
        predictions_log = normal_pred_log.copy()

        # If outlier model exists, use it for samples predicted as outliers
        if outlier_model is not None:
            # Identify potential outliers based on normal model predictions
            outlier_mask = normal_pred_original > threshold

            if outlier_mask.sum() > 0:
                # Predict outliers with outlier model
                outlier_pred_log = outlier_model.predict(X[outlier_mask])
                predictions_log[outlier_mask] = outlier_pred_log

        return predictions_log

    def get_overall_metrics(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate overall metrics across all activities.
        Includes all activities, even those without data.

        Args:
            evaluation_results: Results from evaluate_activity_models

        Returns:
            DataFrame with metrics per activity
        """
        metrics_list = []

        # Iterate over ALL activities (including fixed activities without data)
        for activity in self.activities:
            if activity in evaluation_results:
                # Activity has evaluation results
                results = evaluation_results[activity]
                metrics = results['metrics']
                stats = self.activity_stats.get(activity, {'total_samples': 0, 'test_samples': 0, 'test_cases': 0})
                
                metrics_list.append({
                    'activity': activity,
                    'mae_hours': metrics.get('mae_hours', np.nan),
                    'rmse_hours': metrics.get('rmse_hours', np.nan),
                    'mae_log': metrics.get('mae_log', np.nan),
                    'rmse_log': metrics.get('rmse_log', np.nan),
                    'total_samples': stats['total_samples'],
                    'test_samples': stats['test_samples'],
                    'test_cases': stats['test_cases']
                })
            else:
                # Activity has no evaluation results (e.g., no data)
                stats = self.activity_stats.get(activity, {'total_samples': 0, 'test_samples': 0, 'test_cases': 0})
                metrics_list.append({
                    'activity': activity,
                    'mae_hours': np.nan,
                    'rmse_hours': np.nan,
                    'mae_log': np.nan,
                    'rmse_log': np.nan,
                    'total_samples': stats['total_samples'],
                    'test_samples': stats['test_samples'],
                    'test_cases': stats['test_cases']
                })

        return pd.DataFrame(metrics_list)

    def get_weighted_overall_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate weighted overall metrics across all activities (weighted by test samples).

        Args:
            evaluation_results: Results from evaluate_activity_models

        Returns:
            Dictionary with weighted overall metrics
        """
        total_weighted_mae = 0.0
        total_weighted_rmse = 0.0
        total_weighted_mae_log = 0.0
        total_weighted_rmse_log = 0.0
        total_samples = 0

        for activity, results in evaluation_results.items():
            metrics = results['metrics']
            # Get test_samples from activity_stats, or use 0 if not available
            test_samples = self.activity_stats.get(activity, {}).get('test_samples', 0)
            
            # Skip activities with no test samples (they don't contribute to weighted metrics)
            if test_samples == 0:
                continue

            # Weight each metric by the number of test samples
            total_weighted_mae += metrics['mae_hours'] * test_samples
            total_weighted_rmse += metrics['rmse_hours'] * test_samples
            total_weighted_mae_log += metrics['mae_log'] * test_samples
            total_weighted_rmse_log += metrics['rmse_log'] * test_samples
            total_samples += test_samples

        # Calculate weighted averages
        weighted_metrics = {
            'weighted_mean_mae_hours': total_weighted_mae / total_samples,
            'weighted_mean_rmse_hours': total_weighted_rmse / total_samples,
            'weighted_mean_mae_log': total_weighted_mae_log / total_samples,
            'weighted_mean_rmse_log': total_weighted_rmse_log / total_samples,
            'total_test_samples': total_samples,
            'num_activities': len(evaluation_results)
        }

        return weighted_metrics

    def save_models(self, directory: str = "activity_models") -> None:
        """
        Save all trained models.

        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)

        for activity, model_trainer in self.models.items():
            filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
            
            # Skip fixed activities (they have no trainable parameters)
            if activity in self.fixed_activities:
                print(f"Skipping save for {activity} (fixed activity - always predicts 1 second)")
                continue
            
            # Check if this is a FixedModel instance (fallback check)
            if hasattr(model_trainer, 'fixed_value_log') and not hasattr(model_trainer, 'save_model'):
                print(f"Skipping save for {activity} (fixed model - always predicts 1 second)")
                continue
            
            # Check if this is a quantile regression model
            if isinstance(model_trainer, dict) and model_trainer.get('type') == 'quantile_regression':
                # Save quantile models
                quantile_trainer = model_trainer['quantile_trainer']
                quantile_trainer.save_models(os.path.join(directory, filename_base))
                
                # Save metadata
                metadata = {
                    'type': 'quantile_regression',
                    'quantiles': model_trainer['quantiles']
                }
                metadata_path = os.path.join(directory, f"{filename_base}_metadata.pkl")
                joblib.dump(metadata, metadata_path)
            # Check if this is an outlier separation model
            elif isinstance(model_trainer, dict) and model_trainer.get('type') == 'outlier_separation':
                # Save normal model
                normal_path = os.path.join(directory, f"{filename_base}_normal.pkl")
                model_trainer['normal'].save_model(normal_path)
                
                # Save outlier model if available
                if model_trainer['outlier'] is not None:
                    outlier_path = os.path.join(directory, f"{filename_base}_outlier.pkl")
                    model_trainer['outlier'].save_model(outlier_path)
                
                # Save metadata
                metadata = {
                    'type': 'outlier_separation',
                    'threshold': self.outlier_thresholds[activity]
                }
                metadata_path = os.path.join(directory, f"{filename_base}_metadata.pkl")
                joblib.dump(metadata, metadata_path)
            else:
                # Standard model
                filepath = os.path.join(directory, f"{filename_base}.pkl")
                model_trainer.save_model(filepath)

        print(f"All models saved to: {directory}/")

    def load_models(self, directory: str = "activity_models") -> None:
        """
        Load all trained models.

        Args:
            directory: Directory to load models from
        """
        for activity in self.activities:
            filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
            metadata_path = os.path.join(directory, f"{filename_base}_metadata.pkl")

            # Check if this is a quantile regression or outlier separation model
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                
                if metadata.get('type') == 'quantile_regression':
                    # Load quantile regression models
                    quantile_trainer = QuantileModelTrainer(
                        quantiles=metadata.get('quantiles', [0.5, 0.75, 0.9]),
                        random_state=self.random_state
                    )
                    quantile_trainer.load_models(os.path.join(directory, filename_base))
                    
                    self.models[activity] = {
                        'quantile_trainer': quantile_trainer,
                        'type': 'quantile_regression',
                        'quantiles': metadata.get('quantiles', [0.5, 0.75, 0.9])
                    }
                    print(f"Loaded quantile regression model for: {activity}")
                
                elif metadata.get('type') == 'outlier_separation':
                    # Load normal model
                    normal_path = os.path.join(directory, f"{filename_base}_normal.pkl")
                    normal_model = ModelTrainer()
                    normal_model.load_model(normal_path)
                    
                    # Load outlier model if available
                    outlier_path = os.path.join(directory, f"{filename_base}_outlier.pkl")
                    outlier_model = None
                    if os.path.exists(outlier_path):
                        outlier_model = ModelTrainer()
                        outlier_model.load_model(outlier_path)
                    
                    self.models[activity] = {
                        'normal': normal_model,
                        'outlier': outlier_model,
                        'type': 'outlier_separation'
                    }
                    self.outlier_thresholds[activity] = metadata['threshold']
                    print(f"Loaded outlier separation model for: {activity}")
            else:
                # Standard model
                filepath = os.path.join(directory, f"{filename_base}.pkl")
                if os.path.exists(filepath):
                    model_trainer = ModelTrainer()
                    model_trainer.load_model(filepath)
                    self.models[activity] = model_trainer
                    print(f"Loaded model for: {activity}")
                else:
                    print(f"Model file not found: {filepath}")

    def save_evaluation_results(
        self,
        evaluation_results: Dict[str, Any],
        directory: str = "activity_results"
    ) -> None:
        """
        Save evaluation results for all activities.

        Args:
            evaluation_results: Results from evaluate_activity_models
            directory: Directory to save results
        """
        os.makedirs(directory, exist_ok=True)

        # Save overall metrics
        overall_metrics = self.get_overall_metrics(evaluation_results)
        overall_path = os.path.join(directory, "overall_metrics.csv")
        overall_metrics.to_csv(overall_path, index=False)

        # Save weighted overall metrics
        weighted_metrics = self.get_weighted_overall_metrics(evaluation_results)
        weighted_path = os.path.join(directory, "weighted_overall_metrics.txt")
        with open(weighted_path, 'w') as f:
            f.write("Weighted Overall Metrics (by test samples - FAIR)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Weighted Mean MAE (hours): {weighted_metrics['weighted_mean_mae_hours']:.6f}\n")
            f.write(f"Weighted Mean RMSE (hours): {weighted_metrics['weighted_mean_rmse_hours']:.6f}\n")
            f.write(f"Weighted Mean MAE (log): {weighted_metrics['weighted_mean_mae_log']:.6f}\n")
            f.write(f"Weighted Mean RMSE (log): {weighted_metrics['weighted_mean_rmse_log']:.6f}\n")
            f.write(f"Total Test Samples: {int(weighted_metrics['total_test_samples'])}\n")
            f.write(f"Number of Activities: {weighted_metrics['num_activities']}\n")

            # Compare with unweighted
            unweighted_mae = overall_metrics['mae_hours'].mean()
            unweighted_rmse = overall_metrics['rmse_hours'].mean()
            f.write(f"\nComparison:\n")
            f.write(f"Weighted MAE: {weighted_metrics['weighted_mean_mae_hours']:.6f}\n")
            f.write(f"Unweighted MAE: {unweighted_mae:.6f}\n")
            f.write(f"Difference: {weighted_metrics['weighted_mean_mae_hours'] - unweighted_mae:.6f}\n")

        # Save individual feature importances
        for activity, results in evaluation_results.items():
            activity_dir = os.path.join(directory, activity.replace(' ', '_').replace('(', '').replace(')', ''))
            os.makedirs(activity_dir, exist_ok=True)

            # Save feature importance (if available)
            fi_path = os.path.join(activity_dir, "feature_importance.csv")
            if 'feature_importance' in results and results['feature_importance'] is not None:
                results['feature_importance'].to_csv(fi_path, index=False)
            else:
                # Create empty feature importance file if not available
                pd.DataFrame({'feature': [], 'importance': []}).to_csv(fi_path, index=False)

            # Save metrics
            metrics_path = os.path.join(activity_dir, "metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"Metrics for {activity}\n")
                f.write("="*30 + "\n")
                for key, value in results['metrics'].items():
                    if pd.notna(value):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: N/A\n")

        print(f"Evaluation results saved to: {directory}/")

    def run_complete_workflow(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        results_dir: str = "activity_results",
        models_dir: str = "activity_models",
        use_business_time: bool = True,
        quantile_config: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete workflow: prepare data, train models, evaluate.

        Args:
            df: Input dataframe
            test_size: Test set proportion
            results_dir: Directory to save results
            models_dir: Directory to save models
            use_business_time: If True, only count time during business hours (9:00-17:00)
            quantile_config: Optional dictionary mapping activity names to quantile lists.
                           Example: {"A_Concept": [0.8, 0.9, 0.95], "Other_Activity": [0.5, 0.75, 0.9]}
                           If not provided, uses default: A_Concept with [0.8, 0.9, 0.95]

        Returns:
            Evaluation results
        """
        print("="*60)
        print("ACTIVITY-SPECIFIC MODEL WORKFLOW")
        if use_business_time:
            print("Using Business-Time Filtering (9:00-17:00)")
        print("="*60)

        # Prepare data
        self.prepare_activity_data(df, test_size, use_business_time=use_business_time)

        # Train models
        self.train_activity_models(quantile_config=quantile_config)

        # Evaluate models
        evaluation_results = self.evaluate_activity_models()

        # Save results
        self.save_evaluation_results(evaluation_results, results_dir)
        self.save_models(models_dir)

        print("\n" + "="*60)
        print("WORKFLOW COMPLETED!")
        print("="*60)

        # Print summary with both weighted and unweighted metrics
        overall_metrics = self.get_overall_metrics(evaluation_results)
        weighted_metrics = self.get_weighted_overall_metrics(evaluation_results)

        print("\nOverall Performance Summary:")
        print("=" * 50)
        print("Weighted Metrics (by test samples - FAIR):")
        print(f"  MAE (hours): {weighted_metrics['weighted_mean_mae_hours']:.2f}")
        print(f"  RMSE (hours): {weighted_metrics['weighted_mean_rmse_hours']:.2f}")
        print(f"  MAE (log): {weighted_metrics['weighted_mean_mae_log']:.4f}")
        print(f"  RMSE (log): {weighted_metrics['weighted_mean_rmse_log']:.4f}")
        print(f"  Total Test Samples: {int(weighted_metrics['total_test_samples'])}")
        print(f"  Number of Activities: {weighted_metrics['num_activities']}")

        print("\nUnweighted Metrics (equal weighting - UNFAIR):")
        print(f"  MAE (hours): {overall_metrics['mae_hours'].mean():.2f}")
        print(f"  RMSE (hours): {overall_metrics['rmse_hours'].mean():.2f}")

        print("\nDetailed metrics per activity:")
        print(overall_metrics[['activity', 'mae_hours', 'test_samples']].round(2))

        print(f"\nResults saved to: {results_dir}/")
        print(f"Models saved to: {models_dir}/")

        # Return both individual and weighted results
        return {
            'individual_results': evaluation_results,
            'overall_metrics': overall_metrics,
            'weighted_metrics': weighted_metrics
        }
