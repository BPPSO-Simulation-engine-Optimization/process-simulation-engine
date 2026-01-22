import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
import joblib


class QuantileModelTrainer:
    """
    Trains quantile regression models for handling multimodal distributions.
    
    Instead of predicting the mean, this predicts multiple quantiles (e.g., 
    median, 75th percentile, 90th percentile) which is better for skewed
    and multimodal distributions like A_Concept.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        random_state: int = 42,
        optimized_for_extreme: bool = False
    ):
        """
        Initialize the QuantileModelTrainer.

        Args:
            quantiles: List of quantiles to predict. Can be any number of quantiles.
                      Examples: [0.5], [0.5, 0.75], [0.5, 0.75, 0.9], [0.8, 0.9, 0.95, 0.99]
                      Default: [0.5, 0.75, 0.9]
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            random_state: Random state for reproducibility
            optimized_for_extreme: If True, use optimized hyperparameters for extreme values (e.g., 0.99 quantile)
        """
        self.random_state = random_state
        self.optimized_for_extreme = optimized_for_extreme

        # Default quantiles: median, 75th, 90th percentile
        self.quantiles = quantiles or [0.5, 0.75, 0.9]

        # Default feature lists
        self.categorical_features = categorical_features or ["event", "lifecycle:transition"]
        self.numerical_features = numerical_features or [
            "event_index", "hour", "weekday"
        ]

        self.models = {}  # Dictionary to store models per quantile
        self.preprocessor = None

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline with NaN handling.

        Returns:
            Configured preprocessing pipeline
        """
        # Handle NaN values in numerical features with median imputation
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("num", SimpleImputer(strategy="median"), self.numerical_features),
            ]
        )

        self.preprocessor = preprocessor
        return preprocessor

    def train_quantile_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        additional_numerical_features: Optional[List[str]] = None,
        optimized_for_extreme: Optional[bool] = None
    ) -> Dict[float, Pipeline]:
        """
        Train separate models for each quantile.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector (log-transformed)
            additional_numerical_features: Additional numerical features
            optimized_for_extreme: If True, use optimized hyperparameters for extreme quantiles (0.9+).
                                  If None, uses the value from __init__

        Returns:
            Dictionary with trained models per quantile
        """
        # Use instance attribute if parameter not explicitly provided
        if optimized_for_extreme is None:
            optimized_for_extreme = getattr(self, 'optimized_for_extreme', False)
        
        print(f"Training quantile models for quantiles: {self.quantiles}")

        # Update numerical features if provided
        if additional_numerical_features:
            self.numerical_features.extend(additional_numerical_features)

        # Check for NaN values in features before preprocessing
        nan_counts = X_train.isna().sum()
        if nan_counts.any():
            print(f"  Warning: Found NaN values in features:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"    {col}: {count} NaN values ({count/len(X_train)*100:.1f}%)")
            
            # Remove features that are >95% NaN (likely activity-specific features)
            features_to_remove = nan_counts[nan_counts / len(X_train) > 0.95].index.tolist()
            if features_to_remove:
                print(f"  Removing {len(features_to_remove)} features with >95% NaN values: {features_to_remove}")
                X_train = X_train.drop(columns=features_to_remove)
                # Update feature lists to exclude removed features
                self.categorical_features = [f for f in self.categorical_features if f not in features_to_remove]
                self.numerical_features = [f for f in self.numerical_features if f not in features_to_remove]
                # Recreate preprocessor with updated features
                self.preprocessor = None
        
        # Create preprocessing pipeline
        if self.preprocessor is None:
            self.create_preprocessing_pipeline()

        # Preprocess features (NaN values will be imputed by SimpleImputer)
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Check if there are still NaN values after preprocessing (should not occur - indicates a problem)
        if isinstance(X_train_processed, np.ndarray):
            nan_count_after = np.isnan(X_train_processed).sum()
            if nan_count_after > 0:
                print(f"ERROR: {nan_count_after} NaN values still present after preprocessing")
                print(f"  This indicates a problem in feature preprocessing!")
                raise ValueError(f"NaN values found in features after preprocessing. This should not happen - check feature preprocessing pipeline (SimpleImputer should have filled NaN values).")

        # Train one model per quantile
        for quantile in self.quantiles:
            print(f"  Training model for {quantile*100:.0f}th percentile...")

            # Use XGBoost with custom quantile loss function
            # This is better than GradientBoostingRegressor
            from xgboost import XGBRegressor
            
            # Custom quantile loss function for XGBoost
            def quantile_loss(y_pred, y_true):
                """Quantile loss function for XGBoost"""
                import numpy as np
                residual = y_true - y_pred
                return np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)
            
            # Use XGBoost with squared error objective (we'll use custom eval metric)
            # For quantile regression, we need to use a custom objective or eval_metric
            # XGBoost doesn't have built-in quantile regression, so we use a workaround:
            # Train with reg:squarederror but adjust predictions post-hoc
            # OR use sklearn's approach but with better hyperparameters
            
            # Option 1: Use sklearn GradientBoostingRegressor (current, but slower)
            # Option 2: Use LightGBM (faster and often better for quantile regression)
            # Option 3: Use XGBoost with custom quantile objective (requires custom implementation)
            # Try LightGBM first if available, otherwise fall back to sklearn
            
            use_lightgbm = False
            try:
                import lightgbm as lgb
                use_lightgbm = True
            except ImportError:
                use_lightgbm = False
            
            if not use_lightgbm:
                from sklearn.ensemble import GradientBoostingRegressor
            
            # Optimize hyperparameters based on quantile and whether extreme values are expected
            # For extreme quantiles (0.9+) or if optimized_for_extreme, use more estimators and deeper trees
            is_extreme_quantile = quantile >= 0.9
            use_optimized = optimized_for_extreme or is_extreme_quantile
            
            if use_optimized:
                # Optimized hyperparameters for extreme values:
                # - More estimators for better coverage of rare events
                # - Deeper trees to capture complex patterns
                # - Lower learning rate for better generalization
                # - More patience for extreme quantiles to avoid early stopping
                if quantile >= 0.99:
                    # Extreme quantile (99th): Maximum effort for rare events
                    n_estimators = 1500
                    max_depth = 8
                    learning_rate = 0.01  # Very low learning rate for stability
                    min_samples_split = 50  # Very conservative to prevent overfitting
                    min_samples_leaf = 25
                    n_iter_no_change = 25  # Much more patience
                elif quantile >= 0.95:
                    # Very high quantile (95th)
                    n_estimators = 1000
                    max_depth = 7
                    learning_rate = 0.015
                    min_samples_split = 30
                    min_samples_leaf = 15
                    n_iter_no_change = 20
                else:
                    # High quantile (90th)
                    n_estimators = 800
                    max_depth = 6
                    learning_rate = 0.02
                    min_samples_split = 20
                    min_samples_leaf = 10
                    n_iter_no_change = 15
                
                subsample = 0.85  # Slightly more data per tree
                
                if quantile >= 0.99:
                    print(f"    Using extreme-optimized hyperparameters (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}, min_samples_split={min_samples_split})")
            else:
                # Standard hyperparameters for normal quantiles
                n_estimators = 500
                max_depth = 6
                learning_rate = 0.03
                subsample = 0.8
                n_iter_no_change = 10
                min_samples_split = 2  # Default
                min_samples_leaf = 1  # Default
            
            # Use LightGBM if available and for extreme quantiles (better performance)
            # Note: LightGBM needs to be installed separately (pip install lightgbm)
            if use_lightgbm and (quantile >= 0.95 or use_optimized):
                try:
                    import lightgbm as lgb
                    
                    # LightGBM parameters optimized for quantile regression
                    params = {
                        'objective': 'quantile',
                        'alpha': quantile,  # Target quantile
                        'metric': 'quantile',
                        'boosting_type': 'gbdt',
                        'num_leaves': min(2 ** max_depth, 255),  # More leaves for deeper trees, but capped
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators,
                        'subsample': subsample,
                        'subsample_freq': 1,
                        'colsample_bytree': 0.8,
                        'min_child_samples': min_samples_leaf,
                        'min_split_gain': 0.0,
                        'reg_alpha': 0.1,  # L1 regularization
                        'reg_lambda': 0.1,  # L2 regularization
                        'random_state': self.random_state,
                        'n_jobs': -1,
                        'verbose': -1,
                        'force_row_wise': True  # For better performance
                    }
                    
                    # Create LightGBM dataset
                    train_data = lgb.Dataset(X_train_processed, label=y_train)
                    
                    # Train with early stopping
                    valid_sets = [train_data]
                    valid_names = ['train']
                    callbacks = [
                        lgb.early_stopping(stopping_rounds=n_iter_no_change, verbose=False),
                        lgb.log_evaluation(period=100, show_stdv=False)
                    ]
                    
                    xgb_model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=n_estimators,
                        valid_sets=valid_sets,
                        valid_names=valid_names,
                        callbacks=callbacks
                    )
                    
                    if quantile >= 0.99:
                        print(f"    Using LightGBM for extreme quantile (faster and often better)")
                    use_lightgbm_actual = True
                except Exception as e:
                    # Fall back to sklearn if LightGBM fails
                    print(f"    Warning: LightGBM failed ({e}), falling back to GradientBoostingRegressor")
                    use_lightgbm_actual = False
            else:
                use_lightgbm_actual = False
            
            if not use_lightgbm_actual:
                # Use sklearn GradientBoostingRegressor (always available)
                xgb_model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    loss='quantile',
                    alpha=quantile,  # Target quantile (0.5 = median, 0.75 = 75th percentile, etc.)
                    random_state=self.random_state,
                    validation_fraction=0.1,  # Early stopping
                    n_iter_no_change=n_iter_no_change,
                    tol=1e-4
                )

            # Train the model
            if not (use_lightgbm_actual and hasattr(xgb_model, 'predict')):
                # Train sklearn model (LightGBM models are already trained)
                xgb_model.fit(X_train_processed, y_train)

            # Store the model (with indicator for LightGBM)
            self.models[quantile] = {
                'model': xgb_model,
                'preprocessor': self.preprocessor,
                'is_lightgbm': use_lightgbm_actual
            }

            print(f"    ✓ Model trained for {quantile*100:.0f}th percentile")

        return self.models

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all quantiles.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with predictions for each quantile
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_quantile_models() first.")

        predictions = {}

        for quantile, model_dict in self.models.items():
            preprocessor = model_dict['preprocessor']
            model = model_dict['model']
            is_lightgbm = model_dict.get('is_lightgbm', False)

            # Preprocess features
            X_processed = preprocessor.transform(X)

            # Make predictions (in log space)
            # Both LightGBM and sklearn models use .predict(), but LightGBM has optional num_iteration parameter
            try:
                if is_lightgbm and hasattr(model, 'best_iteration'):
                    pred_log = model.predict(X_processed, num_iteration=model.best_iteration)
                else:
                    pred_log = model.predict(X_processed)
            except Exception:
                # Fallback for any prediction issues
                pred_log = model.predict(X_processed)

            # Transform back to original scale: log10(x+1) -> 10^y - 1
            pred_original = np.power(10, pred_log) - 1
            
            # Ensure non-negative predictions (negative processing times are impossible)
            pred_original = np.maximum(pred_original, 0)

            predictions[f'quantile_{quantile*100:.0f}'] = pred_original

        return pd.DataFrame(predictions)

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict median (50th percentile) in log-transformed space.
        If 0.5 is not available, uses the lowest available quantile.

        Args:
            X: Feature matrix

        Returns:
            Median predictions (log-transformed)
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_quantile_models() first.")
        
        # Use 0.5 if available, otherwise use the lowest available quantile
        if 0.5 in self.models:
            quantile = 0.5
        else:
            quantile = min(self.models.keys())
            print(f"  Note: Using quantile {quantile} instead of 0.5 (median)")

        model_dict = self.models[quantile]
        preprocessor = model_dict['preprocessor']
        model = model_dict['model']
        is_lightgbm = model_dict.get('is_lightgbm', False)

        X_processed = preprocessor.transform(X)
        
        # Make predictions (in log space)
        if is_lightgbm:
            pred_log = model.predict(X_processed, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
        else:
            pred_log = model.predict(X_processed)
        
        return pred_log  # Return log-transformed values for consistency
    
    def predict_lowest_quantile(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the lowest available quantile in log-transformed space.

        Args:
            X: Feature matrix

        Returns:
            Predictions (log-transformed)
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_quantile_models() first.")
        
        quantile = min(self.models.keys())
        model_dict = self.models[quantile]
        preprocessor = model_dict['preprocessor']
        model = model_dict['model']
        is_lightgbm = model_dict.get('is_lightgbm', False)

        X_processed = preprocessor.transform(X)
        
        # Make predictions (in log space)
        if is_lightgbm:
            pred_log = model.predict(X_processed, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
        else:
            pred_log = model.predict(X_processed)
        
        return pred_log

    def save_models(self, filepath_prefix: str) -> None:
        """
        Save all quantile models.

        Args:
            filepath_prefix: Prefix for model files (e.g., "a_concept_quantile")
        """
        if not self.models:
            raise ValueError("No models to save.")

        for quantile, model_dict in self.models.items():
            filepath = f"{filepath_prefix}_q{int(quantile*100)}.pkl"
            joblib.dump(model_dict, filepath)
            print(f"Saved quantile {quantile*100:.0f} model to: {filepath}")

    def load_models(self, filepath_prefix: str) -> None:
        """
        Load all quantile models.

        Args:
            filepath_prefix: Prefix for model files
        """
        for quantile in self.quantiles:
            filepath = f"{filepath_prefix}_q{int(quantile*100)}.pkl"
            try:
                model_dict = joblib.load(filepath)
                self.models[quantile] = model_dict
                # Set preprocessor from the first loaded model (all models use the same preprocessor)
                if self.preprocessor is None and 'preprocessor' in model_dict:
                    self.preprocessor = model_dict['preprocessor']
                print(f"Loaded quantile {quantile*100:.0f} model from: {filepath}")
            except FileNotFoundError:
                print(f"Warning: Model file not found: {filepath}")


class ClassificationRegressionModel:
    """
    Two-stage approach: First classify into speed categories, then regress.
    
    This is better for multimodal distributions where different patterns
    exist for fast vs. slow cases.
    """

    def __init__(
        self,
        speed_thresholds: Optional[List[float]] = None,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize the ClassificationRegressionModel.

        Args:
            speed_thresholds: Thresholds for speed categories (e.g., [5, 25, 45])
                             Creates: fast (<5h), medium (5-25h), slow (25-45h), very_slow (>45h)
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            random_state: Random state for reproducibility
        """
        self.random_state = random_state

        # Default thresholds based on A_Concept distribution
        self.speed_thresholds = speed_thresholds or [5.0, 25.0, 45.0]

        # Default feature lists
        self.categorical_features = categorical_features or ["event", "lifecycle:transition"]
        self.numerical_features = numerical_features or [
            "event_index", "hour", "weekday"
        ]

        self.classifier = None
        self.regressors = {}  # One regressor per category
        self.preprocessor = None

    def create_speed_categories(self, y: pd.Series) -> pd.Series:
        """
        Create speed categories based on thresholds.

        Args:
            y: Target values (in original scale, not log)

        Returns:
            Series with category labels
        """
        categories = pd.Series(index=y.index, dtype=str)

        # Convert from log scale if needed
        if y.max() < 10:  # Likely log scale
            y_original = np.expm1(y)
        else:
            y_original = y

        # Create categories
        categories[y_original < self.speed_thresholds[0]] = "fast"
        
        if len(self.speed_thresholds) == 1:
            categories[y_original >= self.speed_thresholds[0]] = "slow"
        elif len(self.speed_thresholds) == 2:
            categories[(y_original >= self.speed_thresholds[0]) & 
                      (y_original < self.speed_thresholds[1])] = "medium"
            categories[y_original >= self.speed_thresholds[1]] = "slow"
        else:
            categories[(y_original >= self.speed_thresholds[0]) & 
                      (y_original < self.speed_thresholds[1])] = "medium"
            categories[(y_original >= self.speed_thresholds[1]) & 
                      (y_original < self.speed_thresholds[2])] = "slow"
            categories[y_original >= self.speed_thresholds[2]] = "very_slow"

        return categories

    def train_classifier_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_train_original: pd.Series
    ) -> None:
        """
        Train classifier and separate regressors for each category.

        Args:
            X_train: Training feature matrix
            y_train: Training target (log-transformed)
            y_train_original: Training target (original scale)
        """
        from sklearn.ensemble import RandomForestClassifier

        print("Training classification-regression model...")

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("num", "passthrough", self.numerical_features),
            ]
        )

        self.preprocessor = preprocessor
        X_train_processed = preprocessor.fit_transform(X_train)

        # Create speed categories
        categories = self.create_speed_categories(y_train_original)
        print(f"  Category distribution:")
        print(categories.value_counts())

        # Train classifier
        print("  Training classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.classifier.fit(X_train_processed, categories)

        # Train separate regressor for each category
        print("  Training regressors per category...")
        for category in categories.unique():
            category_mask = categories == category
            if category_mask.sum() < 10:
                print(f"    Skipping {category} - too few samples ({category_mask.sum()})")
                continue

            print(f"    Training regressor for {category} ({category_mask.sum()} samples)...")

            X_cat = X_train_processed[category_mask]
            y_cat = y_train[category_mask]

            regressor = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.random_state,
                n_jobs=-1
            )

            regressor.fit(X_cat, y_cat)
            self.regressors[category] = regressor

            print(f"      ✓ Regressor trained for {category}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using two-stage approach.

        Args:
            X: Feature matrix

        Returns:
            Predictions (original scale)
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet.")

        X_processed = self.preprocessor.transform(X)

        # Stage 1: Classify
        categories = self.classifier.predict(X_processed)

        # Stage 2: Regress per category
        predictions_log = np.zeros(len(X))

        for category in self.regressors.keys():
            category_mask = categories == category
            if category_mask.sum() > 0:
                X_cat = X_processed[category_mask]
                pred_log = self.regressors[category].predict(X_cat)
                predictions_log[category_mask] = pred_log

        # Transform back to original scale
        return np.expm1(predictions_log)
