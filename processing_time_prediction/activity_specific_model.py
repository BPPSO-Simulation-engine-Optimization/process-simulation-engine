import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os
import joblib
from datetime import datetime

# Support both package imports (from .x import X) and direct script execution (from x import X)
try:
    from .model_trainer import ModelTrainer
    from .evaluator import Evaluator
    from .feature_engineering import FeatureEngineering
    from .quantile_model import QuantileModelTrainer
except ImportError:
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
            case_features: List of case-level features
            base_features: List of base features to use
            random_state: Random state for reproducibility
            models_dir: Directory to save/load models from. If None, models won't be
                        saved/loaded automatically.
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
        self.w_activities = [
            "W_Assess potential fraud",
            "W_Call after offers",
            "W_Call incomplete files",
            "W_Complete application",
            "W_Handle leads",
            "W_Personal Loan collection",
            "W_Validate application"
        ]

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
            "A_Accepted",
            "O_Cancelled"
        ]

        # Merge with fixed and W-activities
        self.activities = list(set(self.activities) | self.fixed_activities | set(self.w_activities))

        # Initialize feature engineer with customizable features
        self.feature_engineer = FeatureEngineering(
            case_features=case_features,
            base_features=base_features
        )

        self.models = {}            # Dict: activity -> model (or model dict for complex types)
        self.evaluators = {}        # Dict: activity -> Evaluator
        self.activity_stats = {}    # Dict: activity -> statistics
        self.outlier_thresholds = {}  # Dict: activity -> outlier threshold

        self.df_processed = None
        self.activity_data = {}     # Dict: activity -> {'X_train', 'X_test', 'y_train', 'y_test', ...}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_activity_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        use_business_time: bool = True,
        output_dir: str = None
    ) -> None:
        """
        Prepare data for each activity separately.

        Args:
            df: Raw dataframe with case_id, event, timestamp columns
            test_size: Proportion of data for test set
            use_business_time: If True, BOTH training AND test use business-time filtering (5:00-22:00)
            output_dir: Optional directory to save the prepared dataset (df_all.csv)
        """
        print("Preparing activity-specific data...")
        start_time = datetime.now()
        print(f"[{start_time:%H:%M:%S}] Step 1/4: Feature-Engineering & Target-Berechnung startet ...")
        if use_business_time:
            print("Training AND Test: Business-time filtering (5:00-22:00) - consistent distribution")
        else:
            print("Training AND Test: All data (no business-time filter) - consistent distribution")

        df_processed = self.feature_engineer.calculate_processing_time(df, use_business_time=use_business_time)

        additional_events = list(self.activities)
        df_processed = self.feature_engineer.filter_events_of_interest(df_processed, additional_events=additional_events)
        df_processed = self.feature_engineer.add_temporal_features(df_processed)
        df_processed = self.feature_engineer.add_advanced_features(df_processed)
        df_processed = self.feature_engineer.log_transform_target(df_processed)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            df_processed.to_csv(os.path.join(output_dir, "df_all.csv"), index=False)
            print(f"Saved prepared dataset to: {os.path.join(output_dir, 'df_all.csv')}")

        self.df_processed = df_processed

        print(f"[{datetime.now():%H:%M:%S}] Step 2/4: Split nach Aktivitäten & Train/Test-Splits ...")

        num_activities = len(self.activities)
        processed_activities = 0

        for activity in self.activities:
            processed_activities += 1
            print("-" * 80)
            print(f"[{datetime.now():%H:%M:%S}] Activity {processed_activities}/{num_activities}: {activity}")

            activity_df = df_processed[df_processed['event'] == activity].copy()

            if len(activity_df) == 0:
                print(f"  Warning: No data found for activity {activity}")
                continue

            X_all, y_all, case_ids_all = self.feature_engineer.prepare_features_and_target(activity_df)

            invalid_mask = ~np.isfinite(y_all)
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                print(f"  ERROR: Found {invalid_count} samples with invalid target values (NaN/Inf)")
                raise ValueError(f"Invalid target values for activity {activity}. Check data processing pipeline.")

            n_samples = len(X_all)
            min_train_samples = max(1, int(n_samples * (1 - test_size)))
            min_test_samples = max(1, int(n_samples * test_size))
            min_samples_needed = min_train_samples + min_test_samples

            if n_samples < min_samples_needed:
                print(f"  Warning: Activity {activity} has only {n_samples} samples – insufficient for splitting. Skipping.")
                continue

            # Detect categorical and numerical features automatically
            _ALWAYS_CATEGORICAL = {'event', 'org:resource', 'EventOrigin', 'prev_activity'}
            categorical_features = []
            numerical_features = []

            for col in X_all.columns:
                if col in _ALWAYS_CATEGORICAL or col.startswith('case:'):
                    categorical_features.append(col)
                elif activity_df[col].dtype == 'object' or activity_df[col].dtype == 'bool':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)

            model_trainer = ModelTrainer(
                categorical_features=categorical_features if categorical_features else None,
                numerical_features=numerical_features if numerical_features else None,
                random_state=self.random_state
            )

            X_train, X_test, y_train, y_test = model_trainer.split_data_grouped(
                X_all, y_all, case_ids_all, test_size=test_size
            )

            # Validate splits
            for split_name, y_split in [("training", y_train), ("test", y_test)]:
                invalid = ~np.isfinite(y_split)
                if invalid.any():
                    raise ValueError(
                        f"Invalid target values in {split_name} set for {activity}. "
                        "Check data processing pipeline."
                    )

            # Trim top outliers from training set
            trim_percentile = self._get_trim_percentile(activity)
            trim_percentage = (1 - trim_percentile) * 100
            initial_train_samples = len(X_train)

            if len(X_train) > 10:
                threshold = np.percentile(y_train, trim_percentile * 100)
                trim_mask = y_train <= threshold
                X_train_trimmed = X_train[trim_mask]
                y_train_trimmed = y_train[trim_mask]

                trimmed_count = initial_train_samples - len(X_train_trimmed)

                if len(X_train_trimmed) >= max(10, initial_train_samples * 0.5):
                    X_train = X_train_trimmed
                    y_train = y_train_trimmed
                    if trimmed_count > 0:
                        threshold_original = 10 ** threshold - 1
                        print(f"  Trimmed {trimmed_count} samples (top {trim_percentage:.0f}% "
                              f"above {threshold_original:.4f}h) – {len(X_train)} remaining")
                else:
                    print(f"  Warning: Trimming would leave too few samples, skipping trim")
            else:
                print(f"  Skipping trim: Too few samples ({len(X_train)} < 10)")

            self.activity_data[activity] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'model_trainer': model_trainer,
                'raw_data': activity_df
            }

            self.activity_stats[activity] = {
                'total_samples': len(activity_df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_cases': len(case_ids_all.loc[y_train.index].unique()) if len(y_train) > 0 else 0,
                'test_cases': len(case_ids_all.loc[y_test.index].unique())
            }

            filter_type = "business-time" if use_business_time else "all"
            print(f"  {activity}: {len(activity_df)} total samples, "
                  f"{len(X_train)} train ({filter_type}), {len(X_test)} test ({filter_type})")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print(f"[{end_time:%H:%M:%S}] Step 2/4 abgeschlossen – Datenaufbereitung für Aktivitäten fertig "
              f"({duration:.1f} Minuten seit Start von Step 1).")

    def _get_trim_percentile(self, activity: str) -> float:
        """Return the upper-trimming percentile for the given activity."""
        activities_98 = {
            "A_Accepted", "A_Complete", "A_Incomplete",
            "O_Sent (online only)", "O_Returned", "O_Cancelled",
            "W_Call after offers", "W_Call incomplete files",
            "W_Validate application", "W_Complete application",
            "W_Personal Loan collection"
        }
        return 0.98 if activity in activities_98 else 0.95

    # ------------------------------------------------------------------
    # Model existence / save / load helpers
    # ------------------------------------------------------------------

    def _model_exists(self, activity: str) -> bool:
        if self.models_dir is None:
            return False
        filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
        metadata_path = os.path.join(self.models_dir, f"{filename_base}_metadata.pkl")
        if os.path.exists(metadata_path):
            return True
        standard_path = os.path.join(self.models_dir, f"{filename_base}.pkl")
        return os.path.exists(standard_path)

    def _load_single_model(self, activity: str) -> bool:
        if self.models_dir is None or not self._model_exists(activity):
            return False

        filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
        metadata_path = os.path.join(self.models_dir, f"{filename_base}_metadata.pkl")

        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)

            if metadata.get('type') == 'quantile_regression':
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
                normal_path = os.path.join(self.models_dir, f"{filename_base}_normal.pkl")
                normal_model = ModelTrainer()
                normal_model.load_model(normal_path)

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

        standard_path = os.path.join(self.models_dir, f"{filename_base}.pkl")
        if os.path.exists(standard_path):
            model_trainer = ModelTrainer()
            model_trainer.load_model(standard_path)
            self.models[activity] = model_trainer
            print(f"  ✓ Loaded standard model for: {activity}")
            return True

        return False

    def _save_single_model(self, activity: str) -> None:
        if self.models_dir is None or activity not in self.models:
            return

        os.makedirs(self.models_dir, exist_ok=True)
        filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
        model_trainer = self.models[activity]

        if activity in self.fixed_activities:
            return
        if hasattr(model_trainer, 'fixed_value_log') and not hasattr(model_trainer, 'save_model'):
            return

        if isinstance(model_trainer, dict) and model_trainer.get('type') == 'quantile_regression':
            quantile_trainer = model_trainer['quantile_trainer']
            quantile_trainer.save_models(os.path.join(self.models_dir, filename_base))
            metadata = {'type': 'quantile_regression', 'quantiles': model_trainer['quantiles']}
            joblib.dump(metadata, os.path.join(self.models_dir, f"{filename_base}_metadata.pkl"))

        elif isinstance(model_trainer, dict) and model_trainer.get('type') == 'outlier_separation':
            model_trainer['normal'].save_model(os.path.join(self.models_dir, f"{filename_base}_normal.pkl"))
            if model_trainer['outlier'] is not None:
                model_trainer['outlier'].save_model(os.path.join(self.models_dir, f"{filename_base}_outlier.pkl"))
            metadata = {'type': 'outlier_separation', 'threshold': self.outlier_thresholds[activity]}
            joblib.dump(metadata, os.path.join(self.models_dir, f"{filename_base}_metadata.pkl"))

        else:
            model_trainer.save_model(os.path.join(self.models_dir, f"{filename_base}.pkl"))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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
        if self._model_exists(activity):
            print(f"  Model for {activity} already exists, loading from disk...")
            if self._load_single_model(activity):
                return
            print(f"  Failed to load model, training new one...")

        data = self.activity_data[activity]
        y_train_original = np.power(10, data['y_train']) - 1
        threshold = np.percentile(y_train_original, percentile_threshold)

        print(f"  Outlier threshold ({percentile_threshold}th percentile): {threshold:.2f} hours")

        normal_mask = y_train_original <= threshold
        outlier_mask = y_train_original > threshold

        print(f"  Normal samples: {normal_mask.sum()}")
        print(f"  Outlier samples: {outlier_mask.sum()}")

        categorical_features, numerical_features = self._detect_features(data['X_train'])

        model_normal = ModelTrainer(
            categorical_features=categorical_features or None,
            numerical_features=numerical_features or None,
            random_state=self.random_state
        )
        model_normal.train_model(data['X_train'][normal_mask], data['y_train'][normal_mask])

        model_outlier = None
        if outlier_mask.sum() > 100:
            print(f"  Training separate outlier model...")
            model_outlier = ModelTrainer(
                categorical_features=categorical_features or None,
                numerical_features=numerical_features or None,
                random_state=self.random_state
            )
            model_outlier.train_model(data['X_train'][outlier_mask], data['y_train'][outlier_mask])
        else:
            print(f"  Not enough outlier samples ({outlier_mask.sum()}), using normal model for outliers")

        self.models[activity] = {
            'normal': model_normal,
            'outlier': model_outlier,
            'type': 'outlier_separation'
        }
        self.outlier_thresholds[activity] = threshold

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
            quantiles: List of quantiles to predict. Default: [0.5, 0.75, 0.9]
            optimized_for_extreme: If True, use optimized hyperparameters for extreme values
        """
        if self._model_exists(activity):
            filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')
            metadata_path = os.path.join(self.models_dir, f"{filename_base}_metadata.pkl") \
                if self.models_dir else None
            if metadata_path and os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                if metadata.get('type') == 'quantile_regression':
                    saved_quantiles = metadata.get('quantiles', [])
                    if quantiles is None:
                        quantiles = [0.5, 0.75, 0.9]
                    if saved_quantiles == quantiles:
                        print(f"  Model for {activity} already exists with matching quantiles {quantiles}, loading...")
                        if self._load_single_model(activity):
                            return
                        print(f"  Failed to load model, training new one...")

        data = self.activity_data[activity]

        if quantiles is None:
            quantiles = [0.5, 0.75, 0.9]

        print(f"  Training quantile regression for quantiles: {quantiles}")

        categorical_features, numerical_features = self._detect_features(data['X_train'])

        quantile_trainer = QuantileModelTrainer(
            quantiles=quantiles,
            categorical_features=categorical_features or None,
            numerical_features=numerical_features or None,
            random_state=self.random_state,
            optimized_for_extreme=optimized_for_extreme
        )

        quantile_trainer.train_quantile_models(
            data['X_train'],
            data['y_train'],
            optimized_for_extreme=optimized_for_extreme
        )

        if not quantile_trainer.models:
            raise ValueError(f"Failed to train quantile models for {activity}")

        print(f"  Trained quantile models: {list(quantile_trainer.models.keys())}")

        self.models[activity] = {
            'quantile_trainer': quantile_trainer,
            'type': 'quantile_regression',
            'quantiles': quantiles
        }

        if self.models_dir is not None:
            self._save_single_model(activity)

        print(f"  ✓ Quantile regression models trained for {activity}")

    def _detect_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Automatically detect categorical and numerical features."""
        # Columns that are always treated as categorical regardless of dtype
        _ALWAYS_CATEGORICAL = {'event', 'org:resource', 'EventOrigin', 'prev_activity'}

        categorical_features = []
        numerical_features = []
        for col in X.columns:
            if col in _ALWAYS_CATEGORICAL or col.startswith('case:'):
                categorical_features.append(col)
            elif X[col].dtype == 'object' or X[col].dtype == 'bool':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        return categorical_features, numerical_features

    def train_activity_models(self, quantile_config: Optional[Dict[str, List[float]]] = None) -> None:
        """
        Train separate XGBoost models for each activity.

        Args:
            quantile_config: Optional dict mapping activity names to quantile lists.
                             Example: {"A_Concept": [0.8, 0.9, 0.95]}
        """
        print("Training activity-specific models...")
        start_time = datetime.now()
        print(f"[{start_time:%H:%M:%S}] Step 3/4: Training der XGBoost-Modelle startet ...")

        if quantile_config is None:
            quantile_config = {}

        num_acts = len(self.activities)
        trained = 0

        for activity in self.activities:
            if activity not in self.activity_data:
                print(f"Skipping {activity} – no data available")
                continue

            trained += 1
            print("-" * 80)
            print(f"[{datetime.now():%H:%M:%S}] Training {trained}/{num_acts}: {activity}")

            # Fixed activities: always predict 1 second
            if activity in self.fixed_activities:
                print(f"  Using fixed prediction: 1 second (0.000277 hours)")
                fixed_value_log = np.log10(1 / 3600 + 1)
                self.models[activity] = _FixedModel(fixed_value_log)
                print(f"  ✓ Fixed model created for {activity}")
                continue

            # Custom quantile config
            if activity in quantile_config:
                quantiles = quantile_config[activity]
                print(f"  Using quantile regression for {activity} with quantiles: {quantiles}")
                self.train_activity_with_quantile_regression(activity, quantiles=quantiles)

            # Per-activity defaults
            elif activity == "A_Concept":
                self.train_activity_with_outlier_separation(activity, percentile_threshold=50.0)

            elif activity == "A_Accepted":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "A_Complete":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "A_Incomplete":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "O_Sent (mail and online)":
                self.train_activity_with_outlier_separation(activity, percentile_threshold=95.0)

            elif activity == "O_Sent (online only)":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "O_Returned":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "O_Cancelled":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "W_Call after offers":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "W_Call incomplete files":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "W_Validate application":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "W_Complete application":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "W_Personal Loan collection":
                self.train_activity_with_quantile_regression(
                    activity, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
                    optimized_for_extreme=True
                )
            elif activity == "W_Shortened completion":
                # Fixed average prediction
                print(f"  Using fixed average prediction for W_Shortened completion")
                data = self.activity_data.get(activity)
                if data and len(data['y_train']) > 0:
                    avg_hours = np.power(10, data['y_train'].mean()) - 1
                else:
                    avg_hours = 1.0
                avg_value_log = np.log10(avg_hours + 1)
                self.models[activity] = _FixedModel(avg_value_log)
                print(f"  ✓ Fixed average model created for {activity}")
                continue

            else:
                # Standard training
                if self._model_exists(activity):
                    print(f"  Model for {activity} already exists, loading from disk...")
                    if self._load_single_model(activity):
                        print(f"  ✓ Model loaded for {activity}")
                        continue
                    print(f"  Failed to load model, training new one...")

                data = self.activity_data[activity]
                model_trainer = data['model_trainer']
                model_trainer.train_model(data['X_train'], data['y_train'])
                self.models[activity] = model_trainer

                if self.models_dir is not None:
                    self._save_single_model(activity)

            print(f"  ✓ Model trained for {activity}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print(f"[{end_time:%H:%M:%S}] Step 3/4 abgeschlossen – alle verfügbaren Aktivitäten trainiert "
              f"({duration:.1f} Minuten für Training).")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_activity_models(self) -> Dict[str, Any]:
        """
        Evaluate all activity-specific models.

        Returns:
            Dictionary with evaluation results per activity
        """
        print("Evaluating activity-specific models...")
        start_time = datetime.now()
        print(f"[{start_time:%H:%M:%S}] Step 4/4: Evaluation der Modelle startet ...")

        results = {}

        num_acts = len(self.activities)
        done = 0

        for activity in self.activities:
            # Fixed activities without data
            if activity in self.fixed_activities and activity not in self.activity_data:
                print(f"Evaluating model for: {activity} (fixed activity, no data)")
                results[activity] = {
                    'metrics': {'mae_hours': 0.0, 'rmse_hours': 0.0, 'mae_log': 0.0, 'rmse_log': 0.0},
                    'feature_importance': pd.DataFrame({'feature': [], 'importance': []}),
                    'y_true_original': np.array([]),
                    'y_pred_original': np.array([])
                }
                if activity not in self.activity_stats:
                    self.activity_stats[activity] = {'total_samples': 0, 'test_samples': 0, 'test_cases': 0}
                continue

            if activity not in self.activity_data:
                print(f"Skipping evaluation for {activity} – no data available")
                if activity not in self.activity_stats:
                    self.activity_stats[activity] = {
                        'total_samples': 0, 'train_samples': 0, 'test_samples': 0, 'test_cases': 0
                    }
                results[activity] = {
                    'metrics': {'mae_hours': np.nan, 'rmse_hours': np.nan,
                                'mae_log': np.nan, 'rmse_log': np.nan},
                    'feature_importance': pd.DataFrame({'feature': [], 'importance': []}),
                    'test_samples': 0, 'test_cases': 0
                }
                continue

            if activity not in self.models and activity not in self.fixed_activities:
                print(f"Skipping evaluation for {activity} – no model available")
                if activity not in self.activity_stats:
                    self.activity_stats[activity] = {
                        'total_samples': 0, 'train_samples': 0, 'test_samples': 0, 'test_cases': 0
                    }
                results[activity] = {
                    'metrics': {'mae_hours': np.nan, 'rmse_hours': np.nan,
                                'mae_log': np.nan, 'rmse_log': np.nan},
                    'feature_importance': pd.DataFrame({'feature': [], 'importance': []}),
                    'test_samples': 0, 'test_cases': 0
                }
                continue

            done += 1
            print("-" * 80)
            print(f"[{datetime.now():%H:%M:%S}] Evaluating {done}/{num_acts}: {activity}")

            data = self.activity_data[activity]
            model_trainer = self.models.get(activity)

            # --- Fixed activities ---
            if activity in self.fixed_activities:
                if activity not in self.models:
                    fixed_value_log = np.log10(1 / 3600 + 1)
                    self.models[activity] = _FixedModel(fixed_value_log)
                model_trainer = _FixedModelWrapper(np.log10(1 / 3600 + 1))

            # --- Quantile regression (evaluate with median for honest, production-
            #     representative metrics; this matches what predict_for_activity returns)
            elif isinstance(model_trainer, dict) and model_trainer.get('type') == 'quantile_regression':
                quantile_trainer = model_trainer['quantile_trainer']
                # Use the median (0.5) quantile – identical to the prediction strategy
                # used at inference time. This gives fair, reproducible metrics.
                predictions_log = quantile_trainer.predict_median(data['X_test'])

                median_quantile = (
                    0.5 if 0.5 in quantile_trainer.models
                    else min(quantile_trainer.models.keys())
                )
                model_trainer = _QuantileRegressionWrapper(
                    predictions_log, quantile_trainer, median_quantile, len(data['X_test'])
                )

            # --- Outlier separation ---
            elif isinstance(model_trainer, dict) and model_trainer.get('type') == 'outlier_separation':
                threshold = self.outlier_thresholds[activity]
                y_test_original = np.power(10, data['y_test']) - 1

                normal_mask_test = y_test_original <= threshold
                outlier_mask_test = y_test_original > threshold

                predictions_log = np.zeros(len(data['X_test']))

                if normal_mask_test.sum() > 0:
                    predictions_log[normal_mask_test] = model_trainer['normal'].predict(
                        data['X_test'][normal_mask_test]
                    )
                if outlier_mask_test.sum() > 0:
                    outlier_model = model_trainer['outlier'] or model_trainer['normal']
                    predictions_log[outlier_mask_test] = outlier_model.predict(
                        data['X_test'][outlier_mask_test]
                    )

                model_trainer = _OutlierSeparationWrapper(predictions_log, model_trainer, len(data['X_test']))

            evaluator = Evaluator()
            evaluation_results = evaluator.evaluate_model(model_trainer, data['X_test'], data['y_test'])

            self.evaluators[activity] = evaluator
            results[activity] = evaluation_results

            metrics = evaluation_results['metrics']
            print(f"  MAE: {metrics['mae_hours']:.2f} hours, RMSE: {metrics['rmse_hours']:.2f} hours")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print(f"[{end_time:%H:%M:%S}] Step 4/4 abgeschlossen – Evaluation beendet "
              f"({duration:.1f} Minuten für Evaluation).")

        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

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
            Predictions (log-transformed: log10(hours+1))
        """
        if activity in self.fixed_activities:
            fixed_value_log = np.log10(1 / 3600 + 1)
            return np.full(len(X), fixed_value_log)

        if activity not in self.models:
            raise ValueError(f"No model available for activity: {activity}")

        model = self.models[activity]

        if isinstance(model, dict) and model.get('type') == 'quantile_regression':
            return model['quantile_trainer'].predict_median(X)
        elif isinstance(model, dict) and model.get('type') == 'outlier_separation':
            return self._predict_with_outlier_separation(activity, X)
        else:
            return model.predict(X)

    def _predict_with_outlier_separation(self, activity: str, X: pd.DataFrame) -> np.ndarray:
        """Predict using outlier separation approach."""
        model_dict = self.models[activity]
        threshold = self.outlier_thresholds[activity]

        normal_model = model_dict['normal']
        outlier_model = model_dict['outlier']

        normal_pred_log = normal_model.predict(X)
        normal_pred_original = np.power(10, normal_pred_log) - 1

        predictions_log = normal_pred_log.copy()

        if outlier_model is not None:
            outlier_mask = normal_pred_original > threshold
            if outlier_mask.sum() > 0:
                outlier_pred_log = outlier_model.predict(X[outlier_mask])
                predictions_log[outlier_mask] = outlier_pred_log

        return predictions_log

    # ------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------

    def get_overall_metrics(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate overall metrics across all activities.

        Args:
            evaluation_results: Results from evaluate_activity_models

        Returns:
            DataFrame with metrics per activity
        """
        metrics_list = []

        for activity in self.activities:
            stats = self.activity_stats.get(activity, {'total_samples': 0, 'test_samples': 0, 'test_cases': 0})
            if activity in evaluation_results:
                metrics = evaluation_results[activity]['metrics']
            else:
                metrics = {'mae_hours': np.nan, 'rmse_hours': np.nan,
                           'mae_log': np.nan, 'rmse_log': np.nan}

            metrics_list.append({
                'activity': activity,
                'mae_hours': metrics.get('mae_hours', np.nan),
                'rmse_hours': metrics.get('rmse_hours', np.nan),
                'mae_log': metrics.get('mae_log', np.nan),
                'rmse_log': metrics.get('rmse_log', np.nan),
                'total_samples': stats.get('total_samples', 0),
                'test_samples': stats.get('test_samples', 0),
                'test_cases': stats.get('test_cases', 0)
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
            test_samples = self.activity_stats.get(activity, {}).get('test_samples', 0)
            if test_samples == 0:
                continue
            total_weighted_mae += metrics['mae_hours'] * test_samples
            total_weighted_rmse += metrics['rmse_hours'] * test_samples
            total_weighted_mae_log += metrics['mae_log'] * test_samples
            total_weighted_rmse_log += metrics['rmse_log'] * test_samples
            total_samples += test_samples

        if total_samples == 0:
            return {
                'weighted_mean_mae_hours': np.nan,
                'weighted_mean_rmse_hours': np.nan,
                'weighted_mean_mae_log': np.nan,
                'weighted_mean_rmse_log': np.nan,
                'total_test_samples': 0,
                'num_activities': len(evaluation_results)
            }

        return {
            'weighted_mean_mae_hours': total_weighted_mae / total_samples,
            'weighted_mean_rmse_hours': total_weighted_rmse / total_samples,
            'weighted_mean_mae_log': total_weighted_mae_log / total_samples,
            'weighted_mean_rmse_log': total_weighted_rmse_log / total_samples,
            'total_test_samples': total_samples,
            'num_activities': len(evaluation_results)
        }

    # ------------------------------------------------------------------
    # Save / load all models
    # ------------------------------------------------------------------

    def save_models(self, directory: str = "activity_models") -> None:
        """
        Save all trained models.

        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)

        for activity, model_trainer in self.models.items():
            filename_base = activity.replace(' ', '_').replace('(', '').replace(')', '')

            if activity in self.fixed_activities:
                continue
            if hasattr(model_trainer, 'fixed_value_log') and not hasattr(model_trainer, 'save_model'):
                continue

            if isinstance(model_trainer, dict) and model_trainer.get('type') == 'quantile_regression':
                quantile_trainer = model_trainer['quantile_trainer']
                quantile_trainer.save_models(os.path.join(directory, filename_base))
                metadata = {'type': 'quantile_regression', 'quantiles': model_trainer['quantiles']}
                joblib.dump(metadata, os.path.join(directory, f"{filename_base}_metadata.pkl"))

            elif isinstance(model_trainer, dict) and model_trainer.get('type') == 'outlier_separation':
                model_trainer['normal'].save_model(os.path.join(directory, f"{filename_base}_normal.pkl"))
                if model_trainer['outlier'] is not None:
                    model_trainer['outlier'].save_model(os.path.join(directory, f"{filename_base}_outlier.pkl"))
                metadata = {'type': 'outlier_separation', 'threshold': self.outlier_thresholds[activity]}
                joblib.dump(metadata, os.path.join(directory, f"{filename_base}_metadata.pkl"))

            else:
                model_trainer.save_model(os.path.join(directory, f"{filename_base}.pkl"))

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

            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)

                if metadata.get('type') == 'quantile_regression':
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
                    normal_path = os.path.join(directory, f"{filename_base}_normal.pkl")
                    normal_model = ModelTrainer()
                    normal_model.load_model(normal_path)

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

        overall_metrics = self.get_overall_metrics(evaluation_results)
        overall_metrics.to_csv(os.path.join(directory, "overall_metrics.csv"), index=False)

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

        for activity, results in evaluation_results.items():
            activity_dir = os.path.join(directory, activity.replace(' ', '_').replace('(', '').replace(')', ''))
            os.makedirs(activity_dir, exist_ok=True)

            fi_path = os.path.join(activity_dir, "feature_importance.csv")
            if 'feature_importance' in results and results['feature_importance'] is not None:
                results['feature_importance'].to_csv(fi_path, index=False)
            else:
                pd.DataFrame({'feature': [], 'importance': []}).to_csv(fi_path, index=False)

            metrics_path = os.path.join(activity_dir, "metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"Metrics for {activity}\n")
                f.write("=" * 30 + "\n")
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
            df: Input dataframe (with case_id, event, timestamp columns)
            test_size: Test set proportion
            results_dir: Directory to save results
            models_dir: Directory to save models
            use_business_time: If True, only count time during business hours (5:00-22:00)
            quantile_config: Optional dict mapping activity names to quantile lists

        Returns:
            Dict with 'individual_results', 'overall_metrics', 'weighted_metrics'
        """
        print("=" * 60)
        print("ACTIVITY-SPECIFIC MODEL WORKFLOW")
        if use_business_time:
            print("Using Business-Time Filtering (5:00-22:00)")
        print("=" * 60)

        self.prepare_activity_data(df, test_size, use_business_time=use_business_time)
        self.train_activity_models(quantile_config=quantile_config)
        evaluation_results = self.evaluate_activity_models()
        self.save_evaluation_results(evaluation_results, results_dir)
        self.save_models(models_dir)

        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETED!")
        print("=" * 60)

        overall_metrics = self.get_overall_metrics(evaluation_results)
        weighted_metrics = self.get_weighted_overall_metrics(evaluation_results)

        print("\nOverall Performance Summary:")
        print("=" * 50)
        print("Weighted Metrics (by test samples):")
        print(f"  MAE (hours): {weighted_metrics['weighted_mean_mae_hours']:.2f}")
        print(f"  RMSE (hours): {weighted_metrics['weighted_mean_rmse_hours']:.2f}")
        print(f"  MAE (log): {weighted_metrics['weighted_mean_mae_log']:.4f}")
        print(f"  RMSE (log): {weighted_metrics['weighted_mean_rmse_log']:.4f}")
        print(f"  Total Test Samples: {int(weighted_metrics['total_test_samples'])}")
        print(f"  Number of Activities: {weighted_metrics['num_activities']}")

        print(f"\nResults saved to: {results_dir}/")
        print(f"Models saved to: {models_dir}/")

        return {
            'individual_results': evaluation_results,
            'overall_metrics': overall_metrics,
            'weighted_metrics': weighted_metrics
        }


# ---------------------------------------------------------------------------
# Internal wrapper helpers (module-private)
# ---------------------------------------------------------------------------

class _FixedModel:
    """Always predicts a fixed log-scale value."""
    def __init__(self, fixed_value_log: float):
        self.fixed_value_log = fixed_value_log

    def predict(self, X):
        return np.full(len(X), self.fixed_value_log)

    def get_feature_importance(self):
        return pd.DataFrame({'feature': [], 'importance': []})


class _FixedModelWrapper(_FixedModel):
    """Alias used during evaluation."""
    pass


class _QuantileRegressionWrapper:
    """Wrapper for quantile regression during evaluation."""
    def __init__(self, predictions_log, quantile_trainer, quantile_used, expected_length):
        self.predictions_log = predictions_log
        self.quantile_trainer = quantile_trainer
        self.quantile_used = quantile_used
        self.expected_length = expected_length

    def predict(self, X):
        if len(X) != self.expected_length:
            raise ValueError(f"Expected {self.expected_length} samples, got {len(X)}")
        return self.predictions_log

    def get_feature_importance(self):
        model_dict = self.quantile_trainer.models.get(self.quantile_used)
        if model_dict is None:
            return pd.DataFrame({'feature': [], 'importance': []})
        preprocessor = self.quantile_trainer.preprocessor
        if preprocessor is None and 'preprocessor' in model_dict:
            preprocessor = model_dict['preprocessor']
        if preprocessor is None:
            return pd.DataFrame({'feature': [], 'importance': []})
        model = model_dict['model']
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
        return pd.DataFrame({'feature': feature_names, 'importance': importances})


class _OutlierSeparationWrapper:
    """Wrapper for outlier separation during evaluation."""
    def __init__(self, predictions_log, model_dict, expected_length):
        self.predictions_log = predictions_log
        self.model_dict = model_dict
        self.expected_length = expected_length

    def predict(self, X):
        if len(X) != self.expected_length:
            raise ValueError(f"Expected {self.expected_length} samples, got {len(X)}")
        return self.predictions_log

    def get_feature_importance(self):
        return self.model_dict['normal'].get_feature_importance()
