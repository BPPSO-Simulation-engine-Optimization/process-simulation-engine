import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
import joblib


class ModelTrainer:
    """Handles machine learning pipeline for process mining prediction."""

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize the ModelTrainer.

        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            random_state: Random state for reproducibility
        """
        self.random_state = random_state

        # Default feature lists if not provided
        self.categorical_features = categorical_features or ["event", "lifecycle:transition"]
        self.numerical_features = numerical_features or [
            "event_index", "hour", "weekday"
        ]

        self.pipeline = None
        self.feature_names = None

    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Create sklearn preprocessing pipeline.

        Returns:
            Configured preprocessing pipeline
        """
        print("Creating preprocessing pipeline...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("num", "passthrough", self.numerical_features),
            ]
        )

        # XGBoost model with good defaults for this task
        xgb_model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1
        )

        # Create full pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("xgb", xgb_model)
        ])

        self.pipeline = pipeline
        return pipeline

    def split_data_grouped(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data using GroupShuffleSplit to maintain case integrity.

        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (case IDs)
            test_size: Proportion of data for test set

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"Splitting data with test_size={test_size}...")

        # Check if we have enough samples for splitting
        n_samples = len(X)
        min_train_size = max(1, int(n_samples * (1 - test_size)))
        min_test_size = max(1, int(n_samples * test_size))
        
        if n_samples < (min_train_size + min_test_size):
            raise ValueError(
                f"Insufficient samples for splitting: {n_samples} samples with test_size={test_size} "
                f"requires at least {min_train_size + min_test_size} samples. "
                f"Minimum train size: {min_train_size}, minimum test size: {min_test_size}"
            )

        gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=self.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train cases: {len(groups.iloc[train_idx].unique())}")
        print(f"Test cases: {len(groups.iloc[test_idx].unique())}")

        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        additional_numerical_features: Optional[List[str]] = None
    ) -> Pipeline:
        """
        Train the XGBoost model.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            additional_numerical_features: Additional numerical features to include

        Returns:
            Trained pipeline
        """
        print("Training XGBoost model...")

        # Update numerical features if additional ones provided
        if additional_numerical_features:
            self.numerical_features.extend(additional_numerical_features)
            # Recreate pipeline with updated features
            self.create_preprocessing_pipeline()

        # Create pipeline if not exists
        if self.pipeline is None:
            self.create_preprocessing_pipeline()

        # Train the model
        self.pipeline.fit(X_train, y_train)

        # Store feature names for later analysis
        self.feature_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()

        print("Model training completed")

        return self.pipeline

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Feature matrix for prediction

        Returns:
            Predictions array
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        return self.pipeline.predict(X)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("No trained model to save.")

        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> Pipeline:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            Loaded pipeline
        """
        self.pipeline = joblib.load(filepath)

        # Try to restore feature names
        try:
            self.feature_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        except:
            print("Warning: Could not restore feature names")

        print(f"Model loaded from: {filepath}")
        return self.pipeline

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet.")

        if self.feature_names is None:
            raise ValueError("Feature names not available.")

        importances = self.pipeline.named_steps["xgb"].feature_importances_

        fi_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        return fi_df
