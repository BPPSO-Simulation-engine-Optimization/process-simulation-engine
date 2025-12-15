from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from scipy import stats
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


class ProcessingTimePredictionClass:
    """
    Fits log-normal probability distributions for processing times between consecutive events.
    Processing time is the time until the next activity, grouped by activity pairs and lifecycle transitions.
    """

    def __init__(self, data_log_df: pd.DataFrame, method: str = "distribution", min_observations: int = 2):
        """
        Args:
            data_log_df: DataFrame with event log data (must have columns: 
                        case:concept:name, concept:name, lifecycle:transition, time:timestamp)
            method: Method to use ("distribution" for probability distributions, "ml" for machine learning)
            min_observations: Minimum number of observations required to fit a distribution
        """
        self.data_log_df = data_log_df.copy()
        self.method = method
        self.min_observations = min_observations
        
        # Key: (prev_activity, prev_lifecycle, curr_activity, curr_lifecycle)
        # Value: dict with 'distribution', 'mu', 'sigma', 'count', 'mean', 'std'
        self.distributions: Dict[Tuple[str, str, str, str], Dict] = {}
        
        # Fallback distribution (overall mean and std)
        self.fallback_mean: Optional[float] = None
        self.fallback_std: Optional[float] = None
        
        # ML model components
        self.ml_model: Optional[RandomForestRegressor] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.feature_defaults: Dict[str, any] = {}
        
        # Extract and fit distributions if using distribution method
        if method == "distribution":
            self._extract_and_fit_distributions()
        elif method == "ml":
            self._train_ml_model()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'distribution' or 'ml'.")

    def _extract_and_fit_distributions(self):
        
        # Ensure timestamp column is datetime
        if "time:timestamp" in self.data_log_df.columns:
            self.data_log_df["time:timestamp"] = pd.to_datetime(
                self.data_log_df["time:timestamp"], errors="coerce"
            )
        
        # Sort by case and timestamp
        df_sorted = self.data_log_df.sort_values(
            ["case:concept:name", "time:timestamp"]
        ).copy()
        
        required_cols = ["case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"]
        missing_cols = [col for col in required_cols if col not in df_sorted.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing timestamps
        df_sorted = df_sorted.dropna(subset=["time:timestamp"])
        
        processing_times_by_transition: Dict[Tuple[str, str, str, str], list] = {}
        
        for case_id, case_data in df_sorted.groupby("case:concept:name"):
            case_data = case_data.reset_index(drop=True)
            
            # Skip cases with only one event (no processing time)
            if len(case_data) < 2:
                continue
            
            for i in range(len(case_data) - 1):
                prev_event = case_data.iloc[i]
                curr_event = case_data.iloc[i + 1]
                
                # Skip if missing critical information
                if pd.isna(prev_event["time:timestamp"]) or pd.isna(curr_event["time:timestamp"]):
                    continue
                
                # Get activity and lifecycle information (handle NaN values)
                prev_activity = str(prev_event["concept:name"])
                prev_lifecycle = "complete" if pd.isna(prev_event.get("lifecycle:transition")) else str(prev_event["lifecycle:transition"])
                curr_activity = str(curr_event["concept:name"])
                curr_lifecycle = "complete" if pd.isna(curr_event.get("lifecycle:transition")) else str(curr_event["lifecycle:transition"])
                
                # Calculate processing time in seconds
                time_diff = (curr_event["time:timestamp"] - prev_event["time:timestamp"]).total_seconds()
                
                # Skip negative or zero times (data quality issues)
                if time_diff <= 0:
                    continue
                
                transition_key = (prev_activity, prev_lifecycle, curr_activity, curr_lifecycle)
                
                if transition_key not in processing_times_by_transition:
                    processing_times_by_transition[transition_key] = []
                
                processing_times_by_transition[transition_key].append(time_diff)
        
        print(f"Found {len(processing_times_by_transition)} unique transition patterns")
        
        # Fit log-normal distributions for each transition
        all_processing_times = []
        
        for transition_key, times in processing_times_by_transition.items():
            # Need at least min_observations to fit a distribution
            if len(times) < self.min_observations:
                continue
            
            all_processing_times.extend(times)
            
            # Log-normal is defined as: X ~ lognormal(mu, sigma) where log(X) ~ N(mu, sigma)
            log_times = np.log(times)
            mu = np.mean(log_times)
            sigma = np.std(log_times, ddof=1)  # Use sample std (ddof=1)
            
            # Ensure sigma is positive and not too small
            if sigma < 1e-6:
                sigma = 1e-6
            
            # Create scipy log-normal distribution
            # scipy.stats.lognorm uses shape parameter s=sigma and scale=exp(mu)
            dist = stats.lognorm(s=sigma, scale=np.exp(mu))
            
            self.distributions[transition_key] = {
                'distribution': dist,
                'mu': mu,
                'sigma': sigma,
                'count': len(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'median': np.median(times)
            }
        
        # Calculate fallback statistics (overall mean and std of all processing times)
        if all_processing_times:
            self.fallback_mean = np.mean(all_processing_times)
            self.fallback_std = np.std(all_processing_times)
            print(f"Fitted {len(self.distributions)} distributions")
            print(f"Fallback statistics: mean={self.fallback_mean:.2f}s, std={self.fallback_std:.2f}s")
        else:
            warnings.warn("No valid processing times found in event log!")
            self.fallback_mean = 3600.0  # Default: 1 hour
            self.fallback_std = 1800.0   # Default: 30 minutes

    def _extract_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract training data with features and target (processing times) from event log.
        
        Returns:
            Tuple of (features DataFrame, target Series with processing times in seconds)
        """
        # Ensure timestamp column is datetime
        if "time:timestamp" in self.data_log_df.columns:
            self.data_log_df["time:timestamp"] = pd.to_datetime(
                self.data_log_df["time:timestamp"], errors="coerce"
            )
        
        # Sort by case and timestamp
        df_sorted = self.data_log_df.sort_values(
            ["case:concept:name", "time:timestamp"]
        ).copy()
        
        required_cols = ["case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"]
        missing_cols = [col for col in required_cols if col not in df_sorted.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing timestamps
        df_sorted = df_sorted.dropna(subset=["time:timestamp"])
        
        training_samples = []
        
        for case_id, case_data in df_sorted.groupby("case:concept:name"):
            case_data = case_data.reset_index(drop=True)
            
            # Skip cases with only one event (no processing time)
            if len(case_data) < 2:
                continue
            
            # Get case-level attributes (constant for the case)
            case_attrs = {}
            case_attr_cols = ["case:RequestedAmount", "case:LoanGoal", "case:ApplicationType"]
            for col in case_attr_cols:
                if col in case_data.columns:
                    val = case_data[col].iloc[0] if len(case_data) > 0 else None
                    case_attrs[col] = val if not pd.isna(val) else None
                else:
                    case_attrs[col] = None
            
            # Calculate case start time for position features
            case_start_time = case_data["time:timestamp"].min()
            case_duration_total = (case_data["time:timestamp"].max() - case_start_time).total_seconds()
            
            for i in range(len(case_data) - 1):
                prev_event = case_data.iloc[i]
                curr_event = case_data.iloc[i + 1]
                
                # Skip if missing critical information
                if pd.isna(prev_event["time:timestamp"]) or pd.isna(curr_event["time:timestamp"]):
                    continue
                
                # Calculate processing time in seconds
                time_diff = (curr_event["time:timestamp"] - prev_event["time:timestamp"]).total_seconds()
                
                # Skip negative, zero, or extreme outliers (more than 1 year)
                if time_diff <= 0 or time_diff > 31536000:
                    continue
                
                # Extract features
                sample = {}
                
                # Activity and lifecycle features
                sample['prev_activity'] = str(prev_event["concept:name"]) if not pd.isna(prev_event["concept:name"]) else "unknown"
                sample['prev_lifecycle'] = str(prev_event["lifecycle:transition"]) if not pd.isna(prev_event.get("lifecycle:transition")) else "complete"
                sample['curr_activity'] = str(curr_event["concept:name"]) if not pd.isna(curr_event["concept:name"]) else "unknown"
                sample['curr_lifecycle'] = str(curr_event["lifecycle:transition"]) if not pd.isna(curr_event.get("lifecycle:transition")) else "complete"
                
                # Resource features
                sample['prev_resource'] = str(prev_event.get("org:resource", "unknown")) if not pd.isna(prev_event.get("org:resource")) else "unknown"
                sample['curr_resource'] = str(curr_event.get("org:resource", "unknown")) if not pd.isna(curr_event.get("org:resource")) else "unknown"
                
                # Case-level attributes
                for col, val in case_attrs.items():
                    sample[col] = val
                
                # Event-level attributes (from current event)
                event_attr_cols = ["CreditScore", "FirstWithdrawalAmount", "NumberOfTerms", 
                                  "MonthlyCost", "Accepted", "Selected", "OfferedAmount"]
                for col in event_attr_cols:
                    if col in curr_event.index:
                        val = curr_event[col]
                        sample[col] = val if not pd.isna(val) else None
                    else:
                        sample[col] = None
                
                # Temporal features from current event timestamp
                timestamp = curr_event["time:timestamp"]
                sample['hour'] = timestamp.hour
                sample['weekday'] = timestamp.weekday()
                sample['month'] = timestamp.month
                sample['day_of_year'] = timestamp.timetuple().tm_yday
                
                # Case position features
                time_since_start = (prev_event["time:timestamp"] - case_start_time).total_seconds()
                sample['event_position_in_case'] = i + 1
                sample['case_duration_so_far'] = time_since_start
                sample['case_duration_total'] = case_duration_total if case_duration_total > 0 else 1
                sample['case_progress'] = time_since_start / case_duration_total if case_duration_total > 0 else 0
                
                # Target: processing time
                sample['processing_time'] = time_diff
                
                training_samples.append(sample)
        
        if not training_samples:
            raise ValueError("No valid training samples found in event log!")
        
        df_training = pd.DataFrame(training_samples)
        
        # Separate features and target
        target = df_training['processing_time']
        features = df_training.drop(columns=['processing_time'])
        
        print(f"Extracted {len(features)} training samples")
        print(f"Features shape: {features.shape}")
        
        return features, target

    def _prepare_features(self, df_features: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare features for ML model: encoding, scaling, imputation.
        
        Args:
            df_features: DataFrame with raw features
            is_training: If True, fit encoders/scalers; if False, use existing ones
        
        Returns:
            DataFrame with prepared features
        """
        df = df_features.copy()
        
        # Define feature types
        categorical_cols = ['prev_activity', 'prev_lifecycle', 'curr_activity', 'curr_lifecycle',
                           'prev_resource', 'curr_resource', 'case:LoanGoal', 'case:ApplicationType']
        numerical_cols = ['case:RequestedAmount', 'CreditScore', 'FirstWithdrawalAmount', 
                        'NumberOfTerms', 'MonthlyCost', 'OfferedAmount',
                        'hour', 'weekday', 'month', 'day_of_year',
                        'event_position_in_case', 'case_duration_so_far', 'case_duration_total', 'case_progress']
        boolean_cols = ['Accepted', 'Selected']
        
        # Filter to only columns that exist
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        numerical_cols = [c for c in numerical_cols if c in df.columns]
        boolean_cols = [c for c in boolean_cols if c in df.columns]
        
        # Handle boolean columns - fill None values first, then convert to int
        for col in boolean_cols:
            if col in df.columns:
                # Fill None/NaN with 0 (False), then convert to int
                df[col] = df[col].fillna(0).astype(int)
                if col not in numerical_cols:
                    numerical_cols.append(col)
        
        # Impute missing values
        for col in numerical_cols:
            if col in df.columns:
                if is_training:
                    median_val = df[col].median()
                    self.feature_defaults[col] = median_val if not pd.isna(median_val) else 0.0
                df[col] = df[col].fillna(self.feature_defaults.get(col, 0.0))
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    mode_val = df[col].mode()
                    default_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                    self.feature_defaults[col] = default_val
                df[col] = df[col].fillna(self.feature_defaults.get(col, "unknown"))
                df[col] = df[col].astype(str)
        
        # Encode categorical features
        if is_training:
            self.categorical_features = categorical_cols
            self.numerical_features = numerical_cols
            
            # Create label encoders for each categorical feature
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        else:
            # Use existing encoders
            for col in categorical_cols:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    unique_vals = set(df[col].unique())
                    known_classes = set(le.classes_)
                    unknown_vals = unique_vals - known_classes
                    
                    if unknown_vals:
                        # Map unknown values to a default (use most common class index)
                        default_idx = 0
                        df[col] = df[col].apply(
                            lambda x: le.transform([x])[0] if x in known_classes else default_idx
                        )
                    else:
                        df[col] = le.transform(df[col])
                else:
                    # If encoder doesn't exist, use default encoding
                    df[col] = 0
        
        # Scale numerical features
        if is_training:
            self.scaler = MinMaxScaler()
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            if self.scaler is not None:
                # Ensure all numerical columns exist
                missing_num_cols = [c for c in numerical_cols if c not in df.columns]
                for col in missing_num_cols:
                    df[col] = self.feature_defaults.get(col, 0.0)
                
                # Only scale columns that were in training
                cols_to_scale = [c for c in numerical_cols if c in df.columns]
                if cols_to_scale:
                    df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        # Store feature names for prediction
        if is_training:
            self.feature_names = list(df.columns)
        
        return df

    def _train_ml_model(self):
        """
        Train Random Forest Regressor model on extracted features.
        """
        print("="*80)
        print("Training ML Model for Processing Time Prediction")
        print("="*80)
        
        # Extract training data
        print("\n[1/4] Extracting training data from event log...")
        X_raw, y = self._extract_training_data()
        
        # Prepare features
        print("\n[2/4] Preparing features (encoding, scaling, imputation)...")
        X = self._prepare_features(X_raw, is_training=True)
        
        # Remove extreme outliers from target (beyond 3 standard deviations)
        mean_y = y.mean()
        std_y = y.std()
        outlier_threshold = mean_y + 3 * std_y
        valid_mask = y <= outlier_threshold
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"After outlier removal: {len(X)} samples")
        print(f"Target statistics: mean={y.mean():.2f}s, median={y.median():.2f}s, std={y.std():.2f}s")
        
        # Split into train and validation sets
        print("\n[3/4] Splitting data into train/validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Train Random Forest model
        print("\n[4/4] Training Random Forest Regressor...")
        self.ml_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=30,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.ml_model.fit(X_train, y_train)
        
        # Evaluate model
        print("\n" + "-"*80)
        print("Model Evaluation")
        print("-"*80)
        
        y_train_pred = self.ml_model.predict(X_train)
        y_val_pred = self.ml_model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"\nTraining Set:")
        print(f"  MAE:  {train_mae:.2f} seconds ({train_mae/3600:.2f} hours)")
        print(f"  RMSE: {train_rmse:.2f} seconds ({train_rmse/3600:.2f} hours)")
        print(f"  R²:   {train_r2:.4f}")
        
        print(f"\nValidation Set:")
        print(f"  MAE:  {val_mae:.2f} seconds ({val_mae/3600:.2f} hours)")
        print(f"  RMSE: {val_rmse:.2f} seconds ({val_rmse/3600:.2f} hours)")
        print(f"  R²:   {val_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Calculate fallback statistics for edge cases
        self.fallback_mean = float(y.median())  # Use median as it's more robust
        self.fallback_std = float(y.std())
        
        print(f"\nFallback statistics: mean={self.fallback_mean:.2f}s, std={self.fallback_std:.2f}s")
        print("="*80)
        print("Model training completed!")
        print("="*80)

    def save_model(self, filepath: str):
        """
        Save trained model and preprocessors to disk.
        
        Args:
            filepath: Base path for saving (will create multiple files)
        """
        if self.ml_model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_path = f"{filepath}_model.joblib"
        encoders_path = f"{filepath}_encoders.joblib"
        scaler_path = f"{filepath}_scaler.joblib"
        metadata_path = f"{filepath}_metadata.joblib"
        
        joblib.dump(self.ml_model, model_path)
        joblib.dump(self.label_encoders, encoders_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'feature_defaults': self.feature_defaults,
            'fallback_mean': self.fallback_mean,
            'fallback_std': self.fallback_std
        }, metadata_path)
        
        print(f"Model saved to {filepath}_*.joblib")

    def load_model(self, filepath: str):
        """
        Load trained model and preprocessors from disk.
        
        Args:
            filepath: Base path for loading (will load multiple files)
        """
        model_path = f"{filepath}_model.joblib"
        encoders_path = f"{filepath}_encoders.joblib"
        scaler_path = f"{filepath}_scaler.joblib"
        metadata_path = f"{filepath}_metadata.joblib"
        
        if not all(os.path.exists(p) for p in [model_path, encoders_path, scaler_path, metadata_path]):
            raise FileNotFoundError(f"Model files not found at {filepath}")
        
        self.ml_model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
        self.scaler = joblib.load(scaler_path)
        metadata = joblib.load(metadata_path)
        
        self.feature_names = metadata['feature_names']
        self.categorical_features = metadata['categorical_features']
        self.numerical_features = metadata['numerical_features']
        self.feature_defaults = metadata['feature_defaults']
        self.fallback_mean = metadata['fallback_mean']
        self.fallback_std = metadata['fallback_std']
        
        print(f"Model loaded from {filepath}_*.joblib")

    def predict(
        self,
        prev_activity: str,
        prev_lifecycle: str,
        curr_activity: str,
        curr_lifecycle: str,
        context: Optional[Dict] = None
    ) -> float:
        """
        Predict processing time for a given transition.
        
        Args:
            prev_activity: Previous activity name
            prev_lifecycle: Previous lifecycle transition
            curr_activity: Current/next activity name
            curr_lifecycle: Current/next lifecycle transition
            context: Optional context dictionary (not used for distribution method, but kept for API compatibility)
        
        Returns:
            Predicted processing time in seconds (sampled from fitted distribution)
        """
        transition_key = (str(prev_activity), str(prev_lifecycle), str(curr_activity), str(curr_lifecycle))
        
        if self.method == "distribution":
            # Try exact match first
            if transition_key in self.distributions:
                dist_info = self.distributions[transition_key]
                # Sample from the distribution
                sample = dist_info['distribution'].rvs(size=1)[0]
                # Ensure positive value
                return max(0.0, float(sample))
            
            # Try fallback: activity-only matching (without lifecycle)
            activity_only_key = (prev_activity, "*", curr_activity, "*")
            for key, dist_info in self.distributions.items():
                if key[0] == prev_activity and key[2] == curr_activity:
                    sample = dist_info['distribution'].rvs(size=1)[0]
                    return max(0.0, float(sample))
            
            # Final fallback: use overall statistics with log-normal distribution
            if self.fallback_mean and self.fallback_std:
                # Approximate log-normal parameters from mean and std
                # For log-normal: mean = exp(mu + sigma^2/2), var = exp(2*mu + sigma^2) * (exp(sigma^2) - 1)
                # We can estimate mu and sigma from mean and std
                cv = self.fallback_std / self.fallback_mean  # coefficient of variation
                sigma_approx = np.sqrt(np.log(1 + cv**2))
                mu_approx = np.log(self.fallback_mean) - 0.5 * sigma_approx**2
                
                fallback_dist = stats.lognorm(s=sigma_approx, scale=np.exp(mu_approx))
                sample = fallback_dist.rvs(size=1)[0]
                return max(0.0, float(sample))
            
            # Ultimate fallback: return mean (or default)
            return self.fallback_mean if self.fallback_mean else 3600.0
        
        else:  # ML method
            if self.ml_model is None:
                warnings.warn("ML model not trained. Using fallback.")
                return self.fallback_mean if self.fallback_mean else 3600.0
            
            try:
                # Prepare feature vector from context
                feature_dict = self._context_to_features(
                    prev_activity, prev_lifecycle, curr_activity, curr_lifecycle, context
                )
                
                # Convert to DataFrame
                df_features = pd.DataFrame([feature_dict])
                
                # Prepare features (encoding, scaling)
                df_prepared = self._prepare_features(df_features, is_training=False)
                
                # Ensure all required features are present
                missing_features = set(self.feature_names) - set(df_prepared.columns)
                if missing_features:
                    for feat in missing_features:
                        df_prepared[feat] = self.feature_defaults.get(feat, 0.0)
                
                # Reorder columns to match training order
                df_prepared = df_prepared[self.feature_names]
                
                # Predict
                prediction = self.ml_model.predict(df_prepared)[0]
                
                # Ensure positive prediction
                prediction = max(0.0, float(prediction))
                
                return prediction
                
            except Exception as e:
                warnings.warn(f"Error in ML prediction: {e}. Using fallback.")
                return self.fallback_mean if self.fallback_mean else 3600.0

    def _context_to_features(
        self,
        prev_activity: str,
        prev_lifecycle: str,
        curr_activity: str,
        curr_lifecycle: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Convert prediction context to feature dictionary matching training data format.
        
        Args:
            prev_activity: Previous activity name
            prev_lifecycle: Previous lifecycle transition
            curr_activity: Current/next activity name
            curr_lifecycle: Current/next lifecycle transition
            context: Context dictionary with additional features
        
        Returns:
            Dictionary with feature names and values
        """
        if context is None:
            context = {}
        
        # Build feature dictionary
        features = {}
        
        # Activity and lifecycle features
        features['prev_activity'] = str(prev_activity) if prev_activity else "unknown"
        features['prev_lifecycle'] = str(prev_lifecycle) if prev_lifecycle else "complete"
        features['curr_activity'] = str(curr_activity) if curr_activity else "unknown"
        features['curr_lifecycle'] = str(curr_lifecycle) if curr_lifecycle else "complete"
        
        # Resource features
        features['prev_resource'] = str(context.get('resource_1', 'unknown'))
        features['curr_resource'] = str(context.get('resource_2', 'unknown'))
        
        # Case-level attributes
        features['case:RequestedAmount'] = context.get('case:RequestedAmount', None)
        features['case:LoanGoal'] = context.get('case:LoanGoal', None)
        features['case:ApplicationType'] = context.get('case:ApplicationType', None)
        
        # Event-level attributes
        features['CreditScore'] = context.get('CreditScore', None)
        features['FirstWithdrawalAmount'] = context.get('FirstWithdrawalAmount', None)
        features['NumberOfTerms'] = context.get('NumberOfTerms', None)
        features['MonthlyCost'] = context.get('MonthlyCost', None)
        features['Accepted'] = context.get('Accepted', None)
        features['Selected'] = context.get('Selected', None)
        features['OfferedAmount'] = context.get('OfferedAmount', None)
        
        # Temporal features
        from datetime import datetime
        if 'hour' in context:
            features['hour'] = context['hour']
            features['weekday'] = context.get('weekday', 0)
            features['month'] = context.get('month', 1)
            # Estimate day_of_year (approximate)
            features['day_of_year'] = context.get('day_of_year', (context.get('month', 1) - 1) * 30 + 15)
        else:
            # Use current time if not provided
            now = datetime.now()
            features['hour'] = now.hour
            features['weekday'] = now.weekday()
            features['month'] = now.month
            features['day_of_year'] = now.timetuple().tm_yday
        
        # Case position features (use defaults if not available)
        features['event_position_in_case'] = context.get('event_position_in_case', 1)
        features['case_duration_so_far'] = context.get('case_duration_so_far', 0.0)
        features['case_duration_total'] = context.get('case_duration_total', 1.0)
        features['case_progress'] = context.get('case_progress', 0.0)
        
        return features

    def get_distribution_info(self, transition_key: Optional[Tuple[str, str, str, str]] = None) -> Dict:
        """
        Get information about fitted distributions.
        
        Args:
            transition_key: Optional specific transition to get info for.
                          If None, returns info for all distributions.
        
        Returns:
            Dictionary with distribution information
        """
        if transition_key is None:
            return {
                'num_distributions': len(self.distributions),
                'distributions': {
                    str(k): {
                        'mu': v['mu'],
                        'sigma': v['sigma'],
                        'count': v['count'],
                        'mean': v['mean'],
                        'std': v['std'],
                        'median': v['median']
                    }
                    for k, v in self.distributions.items()
                },
                'fallback_mean': self.fallback_mean,
                'fallback_std': self.fallback_std
            }
        else:
            if transition_key in self.distributions:
                info = self.distributions[transition_key].copy()
                info.pop('distribution')  # Remove the scipy object
                return info
            else:
                return {'error': f'Transition {transition_key} not found'}
