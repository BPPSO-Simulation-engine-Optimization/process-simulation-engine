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
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ProcessingTimeTrainer:
    """
    training and fitting of processing time prediction models.
    """

    def __init__(
        self, 
        data_log_df: pd.DataFrame, 
        method: str = "distribution", 
        min_observations: int = 2,
        n_estimators: int = 500,
        max_depth: Optional[int] = 30,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = 'sqrt'
    ):
        """
        Args:
            data_log_df: DataFrame with event log data 
            method: "distribution", "ml", or "probabilistic_ml"
            min_observations: Minimum number of observations required to fit a distribution
            n_estimators: Number of trees in the forest (higher = better performance but slower training)
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
        """
        self.data_log_df = data_log_df.copy()
        self.method = method
        self.min_observations = min_observations
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
        self.distributions: Dict[Tuple[str, str, str, str], Dict] = {}
        self.fallback_mean: Optional[float] = None
        self.fallback_std: Optional[float] = None
        
        self.ml_model: Optional[RandomForestRegressor] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.feature_defaults: Dict[str, any] = {}
        
        self.lstm_model: Optional[keras.Model] = None
        self.sequence_length: int = 10
        self.activity_encoder: Optional[LabelEncoder] = None
        self.lifecycle_encoder: Optional[LabelEncoder] = None
        self.resource_encoder: Optional[LabelEncoder] = None

    def fit_distributions(self):
        """
        Fit log-normal probability distributions for processing times between consecutive events.
        Processing time is the time until the next activity, grouped by activity pairs and lifecycle transitions.
        """
        if "time:timestamp" in self.data_log_df.columns:
            self.data_log_df["time:timestamp"] = pd.to_datetime(
                self.data_log_df["time:timestamp"], errors="coerce"
            )
        
        df_sorted = self.data_log_df.sort_values(
            ["case:concept:name", "time:timestamp"]
        ).copy()
        
        required_cols = ["case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"]
        missing_cols = [col for col in required_cols if col not in df_sorted.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_sorted = df_sorted.dropna(subset=["time:timestamp"])
        
        processing_times_by_transition: Dict[Tuple[str, str, str, str], list] = {}
        
        for case_id, case_data in df_sorted.groupby("case:concept:name"):
            case_data = case_data.reset_index(drop=True)
            
            if len(case_data) < 2:
                continue
            
            for i in range(len(case_data) - 1):
                prev_event = case_data.iloc[i]
                curr_event = case_data.iloc[i + 1]
                
                if pd.isna(prev_event["time:timestamp"]) or pd.isna(curr_event["time:timestamp"]):
                    continue
                
                prev_activity = str(prev_event["concept:name"])
                prev_lifecycle = "complete" if pd.isna(prev_event.get("lifecycle:transition")) else str(prev_event["lifecycle:transition"])
                curr_activity = str(curr_event["concept:name"])
                curr_lifecycle = "complete" if pd.isna(curr_event.get("lifecycle:transition")) else str(curr_event["lifecycle:transition"])
                
                time_diff = (curr_event["time:timestamp"] - prev_event["time:timestamp"]).total_seconds()
                
                if time_diff <= 0:
                    continue
                
                transition_key = (prev_activity, prev_lifecycle, curr_activity, curr_lifecycle)
                
                if transition_key not in processing_times_by_transition:
                    processing_times_by_transition[transition_key] = []
                
                processing_times_by_transition[transition_key].append(time_diff)
        
        print(f"Found {len(processing_times_by_transition)} unique transition patterns")
        
        all_processing_times = []
        
        for transition_key, times in processing_times_by_transition.items():
            if len(times) < self.min_observations:
                continue
            
            all_processing_times.extend(times)
            
            log_times = np.log(times)
            mu = np.mean(log_times)
            sigma = np.std(log_times, ddof=1)
            
            if sigma < 1e-6:
                sigma = 1e-6
            
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
        
        if all_processing_times:
            self.fallback_mean = np.mean(all_processing_times)
            self.fallback_std = np.std(all_processing_times)
            print(f"Fitted {len(self.distributions)} distributions")
            print(f"Fallback statistics: mean={self.fallback_mean:.2f}s, std={self.fallback_std:.2f}s")
        else:
            warnings.warn("No valid processing times found in event log!")
            self.fallback_mean = 3600.0
            self.fallback_std = 1800.0

    def _extract_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract training data with features and target (processing times) from event log.
        
        Returns:
            Tuple of (features DataFrame, target Series with processing times in seconds)
        """
        if "time:timestamp" in self.data_log_df.columns:
            self.data_log_df["time:timestamp"] = pd.to_datetime(
                self.data_log_df["time:timestamp"], errors="coerce"
            )
        
        df_sorted = self.data_log_df.sort_values(
            ["case:concept:name", "time:timestamp"]
        ).copy()
        
        required_cols = ["case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"]
        missing_cols = [col for col in required_cols if col not in df_sorted.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_sorted = df_sorted.dropna(subset=["time:timestamp"])
        
        training_samples = []
        
        for case_id, case_data in df_sorted.groupby("case:concept:name"):
            case_data = case_data.reset_index(drop=True)
            
            if len(case_data) < 2:
                continue
            
            case_attrs = {}
            case_attr_cols = ["case:LoanGoal", "case:ApplicationType"]
            for col in case_attr_cols:
                if col in case_data.columns:
                    val = case_data[col].iloc[0] if len(case_data) > 0 else None
                    case_attrs[col] = val if not pd.isna(val) else None
                else:
                    case_attrs[col] = None
            
            case_start_time = case_data["time:timestamp"].min()
            
            for i in range(len(case_data) - 1):
                prev_event = case_data.iloc[i]
                curr_event = case_data.iloc[i + 1]
                
                if pd.isna(prev_event["time:timestamp"]) or pd.isna(curr_event["time:timestamp"]):
                    continue
                
                time_diff = (curr_event["time:timestamp"] - prev_event["time:timestamp"]).total_seconds()
                
                if time_diff <= 0 or time_diff > 31536000:
                    continue
                
                sample = {}
                
                sample['prev_activity'] = str(prev_event["concept:name"]) if not pd.isna(prev_event["concept:name"]) else "unknown"
                sample['prev_lifecycle'] = str(prev_event["lifecycle:transition"]) if not pd.isna(prev_event.get("lifecycle:transition")) else "complete"
                sample['curr_activity'] = str(curr_event["concept:name"]) if not pd.isna(curr_event["concept:name"]) else "unknown"
                sample['curr_lifecycle'] = str(curr_event["lifecycle:transition"]) if not pd.isna(curr_event.get("lifecycle:transition")) else "complete"
                
                sample['prev_resource'] = str(prev_event.get("org:resource", "unknown")) if not pd.isna(prev_event.get("org:resource")) else "unknown"
                sample['curr_resource'] = str(curr_event.get("org:resource", "unknown")) if not pd.isna(curr_event.get("org:resource")) else "unknown"
                
                for col, val in case_attrs.items():
                    sample[col] = val
                
                event_attr_cols = ["Accepted", "Selected"]
                for col in event_attr_cols:
                    if col in curr_event.index:
                        val = curr_event[col]
                        sample[col] = val if not pd.isna(val) else None
                    else:
                        sample[col] = None
                
                timestamp = curr_event["time:timestamp"]
                sample['hour'] = timestamp.hour
                sample['weekday'] = timestamp.weekday()
                sample['month'] = timestamp.month
                sample['day_of_year'] = timestamp.timetuple().tm_yday
                
                time_since_start = (prev_event["time:timestamp"] - case_start_time).total_seconds()
                sample['event_position_in_case'] = i + 1
                sample['case_duration_so_far'] = time_since_start
                
                sample['processing_time'] = time_diff
                
                training_samples.append(sample)
        
        if not training_samples:
            raise ValueError("No valid training samples found in event log!")
        
        df_training = pd.DataFrame(training_samples)
        
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
        
        categorical_cols = ['prev_activity', 'prev_lifecycle', 'curr_activity', 'curr_lifecycle',
                           'prev_resource', 'curr_resource', 'case:LoanGoal', 'case:ApplicationType']
        numerical_cols = ['hour', 'weekday', 'month', 'day_of_year',
                        'event_position_in_case', 'case_duration_so_far']
        boolean_cols = ['Accepted', 'Selected']
        
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        numerical_cols = [c for c in numerical_cols if c in df.columns]
        boolean_cols = [c for c in boolean_cols if c in df.columns]
        
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
                if col not in numerical_cols:
                    numerical_cols.append(col)
        
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
        
        if is_training:
            self.categorical_features = categorical_cols
            self.numerical_features = numerical_cols
            
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        else:
            for col in categorical_cols:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str)
                    unique_vals = set(df[col].unique())
                    known_classes = set(le.classes_)
                    unknown_vals = unique_vals - known_classes
                    
                    if unknown_vals:
                        default_idx = 0
                        df[col] = df[col].apply(
                            lambda x: le.transform([x])[0] if x in known_classes else default_idx
                        )
                    else:
                        df[col] = le.transform(df[col])
                else:
                    df[col] = 0
        
        if is_training:
            self.scaler = MinMaxScaler()
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            if self.scaler is not None:
                missing_num_cols = [c for c in numerical_cols if c not in df.columns]
                for col in missing_num_cols:
                    df[col] = self.feature_defaults.get(col, 0.0)
                
                cols_to_scale = [c for c in numerical_cols if c in df.columns]
                if cols_to_scale:
                    df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        if is_training:
            self.feature_names = list(df.columns)
        
        return df

    def train_ml_model(self):
        """
        Train Random Forest Regressor model on extracted features.
        """
        print("="*80)
        print("Training ML Model for Processing Time Prediction")
        print("="*80)
        
        print("\n[1/4] Extracting training data from event log...")
        X_raw, y = self._extract_training_data()
        
        print("\n[2/4] Preparing features (encoding, scaling, imputation)...")
        X = self._prepare_features(X_raw, is_training=True)
        
        mean_y = y.mean()
        std_y = y.std()
        outlier_threshold = mean_y + 3 * std_y
        valid_mask = y <= outlier_threshold
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"After outlier removal: {len(X)} samples")
        print(f"Target statistics: mean={y.mean():.2f}s, median={y.median():.2f}s, std={y.std():.2f}s")
        
        print("\n[3/4] Splitting data into train/validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        print(f"\n[4/4] Training Random Forest Regressor (n_estimators={self.n_estimators})...")
        self.ml_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.ml_model.fit(X_train, y_train)
        
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
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.fallback_mean = float(y.median())
        self.fallback_std = float(y.std())
        
        print(f"\nFallback statistics: mean={self.fallback_mean:.2f}s, std={self.fallback_std:.2f}s")
        print("="*80)
        print("Model training completed!")
        print("="*80)

    def _extract_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for probabilistic_ml method. Install with: pip install tensorflow")
        
        if "time:timestamp" in self.data_log_df.columns:
            self.data_log_df["time:timestamp"] = pd.to_datetime(
                self.data_log_df["time:timestamp"], errors="coerce"
            )
        
        df_sorted = self.data_log_df.sort_values(
            ["case:concept:name", "time:timestamp"]
        ).copy()
        
        required_cols = ["case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"]
        missing_cols = [col for col in required_cols if col not in df_sorted.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_sorted = df_sorted.dropna(subset=["time:timestamp"])
        
        all_activities = []
        all_lifecycles = []
        all_resources = []
        
        for case_id, case_data in df_sorted.groupby("case:concept:name"):
            case_data = case_data.reset_index(drop=True)
            for _, event in case_data.iterrows():
                all_activities.append(str(event["concept:name"]) if not pd.isna(event["concept:name"]) else "unknown")
                all_lifecycles.append(str(event.get("lifecycle:transition", "complete")) if not pd.isna(event.get("lifecycle:transition")) else "complete")
                all_resources.append(str(event.get("org:resource", "unknown")) if not pd.isna(event.get("org:resource")) else "unknown")
        
        self.activity_encoder = LabelEncoder()
        self.lifecycle_encoder = LabelEncoder()
        self.resource_encoder = LabelEncoder()
        
        activity_encoded = self.activity_encoder.fit_transform(all_activities)
        lifecycle_encoded = self.lifecycle_encoder.fit_transform(all_lifecycles)
        resource_encoded = self.resource_encoder.fit_transform(all_resources)
        
        sequences = []
        contexts = []
        targets = []
        
        for case_id, case_data in df_sorted.groupby("case:concept:name"):
            case_data = case_data.reset_index(drop=True)
            
            if len(case_data) < 2:
                continue
            
            case_attrs = {}
            case_attr_cols = ["case:LoanGoal", "case:ApplicationType"]
            for col in case_attr_cols:
                if col in case_data.columns:
                    val = case_data[col].iloc[0] if len(case_data) > 0 else None
                    case_attrs[col] = val if not pd.isna(val) else None
                else:
                    case_attrs[col] = None
            
            case_start_time = case_data["time:timestamp"].min()
            
            case_sequence = []
            case_contexts = []
            case_targets = []
            
            for i in range(len(case_data)):
                event = case_data.iloc[i]
                
                if pd.isna(event["time:timestamp"]):
                    continue
                
                activity = str(event["concept:name"]) if not pd.isna(event["concept:name"]) else "unknown"
                lifecycle = str(event.get("lifecycle:transition", "complete")) if not pd.isna(event.get("lifecycle:transition")) else "complete"
                resource = str(event.get("org:resource", "unknown")) if not pd.isna(event.get("org:resource")) else "unknown"
                
                activity_idx = self.activity_encoder.transform([activity])[0]
                lifecycle_idx = self.lifecycle_encoder.transform([lifecycle])[0]
                resource_idx = self.resource_encoder.transform([resource])[0]
                
                case_sequence.append([activity_idx, lifecycle_idx, resource_idx])
                
                timestamp = event["time:timestamp"]
                time_since_start = (timestamp - case_start_time).total_seconds()
                
                context = [
                    timestamp.hour / 24.0,
                    timestamp.weekday() / 7.0,
                    timestamp.month / 12.0,
                    timestamp.timetuple().tm_yday / 365.0,
                    (i + 1) / 100.0,
                    time_since_start / 86400.0
                ]
                
                if case_attrs.get("case:LoanGoal"):
                    context.append(1.0 if str(case_attrs["case:LoanGoal"]) == "Car" else 0.0)
                else:
                    context.append(0.0)
                
                if case_attrs.get("case:ApplicationType"):
                    context.append(1.0 if str(case_attrs["case:ApplicationType"]) == "New" else 0.0)
                else:
                    context.append(0.0)
                
                case_contexts.append(context)
                
                if i < len(case_data) - 1:
                    next_event = case_data.iloc[i + 1]
                    if not pd.isna(next_event["time:timestamp"]):
                        time_diff = (next_event["time:timestamp"] - timestamp).total_seconds()
                        if 0 < time_diff <= 31536000:
                            case_targets.append(time_diff)
                        else:
                            case_targets.append(None)
                    else:
                        case_targets.append(None)
                else:
                    case_targets.append(None)
            
            for i in range(len(case_sequence) - 1):
                if case_targets[i] is None:
                    continue
                
                seq_start = max(0, i + 1 - self.sequence_length)
                seq = case_sequence[seq_start:i+1]
                
                while len(seq) < self.sequence_length:
                    seq = [[0, 0, 0]] + seq
                
                sequences.append(seq)
                contexts.append(case_contexts[i])
                targets.append(case_targets[i])
        
        if not sequences:
            raise ValueError("No valid sequences found in event log!")
        
        X_seq = np.array(sequences)
        X_ctx = np.array(contexts)
        y = np.array(targets)
        
        mean_y = np.mean(y)
        std_y = np.std(y)
        outlier_threshold = mean_y + 3 * std_y
        valid_mask = y <= outlier_threshold
        X_seq = X_seq[valid_mask]
        X_ctx = X_ctx[valid_mask]
        y = y[valid_mask]
        
        return X_seq, X_ctx, y

    def train_probabilistic_ml_model(self):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for probabilistic_ml method. Install with: pip install tensorflow")
        
        print("="*80)
        print("Training Gaussian LSTM Model for Processing Time Prediction")
        print("="*80)
        
        print("\n[1/4] Extracting sequences from event log...")
        X_seq, X_ctx, y = self._extract_sequences()
        
        print(f"Extracted {len(X_seq)} sequences")
        print(f"Sequence shape: {X_seq.shape}, Context shape: {X_ctx.shape}")
        print(f"Target statistics: mean={np.mean(y):.2f}s, median={np.median(y):.2f}s, std={np.std(y):.2f}s")
        
        print("\n[2/4] Splitting data into train/validation sets...")
        X_seq_train, X_seq_val, X_ctx_train, X_ctx_val, y_train, y_val = train_test_split(
            X_seq, X_ctx, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training samples: {len(X_seq_train)}")
        print(f"Validation samples: {len(X_seq_val)}")
        
        print("\n[3/4] Building LSTM model...")
        
        num_activities = len(self.activity_encoder.classes_)
        num_lifecycles = len(self.lifecycle_encoder.classes_)
        num_resources = len(self.resource_encoder.classes_)
        
        seq_input = keras.Input(shape=(self.sequence_length, 3), name='sequence')
        
        activity_embed = layers.Embedding(num_activities, 16, mask_zero=True)(seq_input[:, :, 0])
        lifecycle_embed = layers.Embedding(num_lifecycles, 8, mask_zero=True)(seq_input[:, :, 1])
        resource_embed = layers.Embedding(num_resources, 16, mask_zero=True)(seq_input[:, :, 2])
        
        combined = layers.Concatenate()([activity_embed, lifecycle_embed, resource_embed])
        
        lstm_out = layers.LSTM(64, return_sequences=False)(combined)
        
        ctx_input = keras.Input(shape=(X_ctx.shape[1],), name='context')
        ctx_dense = layers.Dense(32, activation='relu')(ctx_input)
        
        merged = layers.Concatenate()([lstm_out, ctx_dense])
        hidden = layers.Dense(64, activation='relu')(merged)
        hidden = layers.Dropout(0.2)(hidden)
        hidden = layers.Dense(32, activation='relu')(hidden)
        
        mean_output = layers.Dense(1, activation='relu', name='mean')(hidden)
        log_var_output = layers.Dense(1, name='log_variance')(hidden)
        
        combined_output = layers.Concatenate()([mean_output, log_var_output])
        model = keras.Model(inputs=[seq_input, ctx_input], outputs=combined_output)
        
        def gaussian_loss(y_true, y_pred):
            mean_pred = y_pred[:, 0:1]
            log_var_pred = y_pred[:, 1:2]
            var_pred = tf.exp(log_var_pred) + 1e-6
            return tf.reduce_mean(0.5 * (tf.math.log(var_pred) + tf.square(y_true - mean_pred) / var_pred))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=gaussian_loss
        )
        
        print("\n[4/4] Training LSTM model...")
        y_train_combined = np.column_stack([y_train, np.zeros_like(y_train)])
        y_val_combined = np.column_stack([y_val, np.zeros_like(y_val)])
        
        history = model.fit(
            [X_seq_train, X_ctx_train],
            y_train_combined,
            validation_data=([X_seq_val, X_ctx_val], y_val_combined),
            epochs=20,
            batch_size=128,
            verbose=1
        )
        
        print("\n" + "-"*80)
        print("Model Evaluation")
        print("-"*80)
        
        train_pred = model.predict([X_seq_train, X_ctx_train], verbose=0)
        train_pred_mean = train_pred[:, 0]
        train_pred_logvar = train_pred[:, 1]
        train_pred_std = np.sqrt(np.exp(train_pred_logvar) + 1e-6)
        
        val_pred = model.predict([X_seq_val, X_ctx_val], verbose=0)
        val_pred_mean = val_pred[:, 0]
        val_pred_logvar = val_pred[:, 1]
        val_pred_std = np.sqrt(np.exp(val_pred_logvar) + 1e-6)
        
        train_mae = mean_absolute_error(y_train, train_pred_mean)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred_mean))
        val_mae = mean_absolute_error(y_val, val_pred_mean)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred_mean))
        
        print(f"\nTraining Set:")
        print(f"  MAE:  {train_mae:.2f} seconds ({train_mae/3600:.2f} hours)")
        print(f"  RMSE: {train_rmse:.2f} seconds ({train_rmse/3600:.2f} hours)")
        print(f"  Mean predicted std: {np.mean(train_pred_std):.2f}s")
        
        print(f"\nValidation Set:")
        print(f"  MAE:  {val_mae:.2f} seconds ({val_mae/3600:.2f} hours)")
        print(f"  RMSE: {val_rmse:.2f} seconds ({val_rmse/3600:.2f} hours)")
        print(f"  Mean predicted std: {np.mean(val_pred_std):.2f}s")
        
        self.lstm_model = model
        self.fallback_mean = float(np.median(y))
        self.fallback_std = float(np.std(y))
        
        print(f"\nFallback statistics: mean={self.fallback_mean:.2f}s, std={self.fallback_std:.2f}s")
        print("="*80)
        print("Model training completed!")
        print("="*80)

    def train(self):
        """
        Train/fit the model based on the specified method.
        """
        if self.method == "distribution":
            self.fit_distributions()
        elif self.method == "ml":
            self.train_ml_model()
        elif self.method == "probabilistic_ml":
            self.train_probabilistic_ml_model()
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'distribution', 'ml', or 'probabilistic_ml'.")

    def save_model(self, filepath: str):
        """
        Save trained model and preprocessors to disk.
        
        Args:
            filepath: Base path for saving (will create multiple files)
        """
        if self.method == "distribution":
            distributions_path = f"{filepath}_distributions.joblib"
            metadata_path = f"{filepath}_metadata.joblib"
            
            distributions_serializable = {}
            for key, value in self.distributions.items():
                distributions_serializable[key] = {
                    'mu': value['mu'],
                    'sigma': value['sigma'],
                    'count': value['count'],
                    'mean': value['mean'],
                    'std': value['std'],
                    'median': value['median']
                }
            
            joblib.dump(distributions_serializable, distributions_path)
            joblib.dump({
                'fallback_mean': self.fallback_mean,
                'fallback_std': self.fallback_std,
                'method': 'distribution'
            }, metadata_path)
            
            print(f"Model saved to {filepath}_*.joblib")
            
        elif self.method == "probabilistic_ml":
            if self.lstm_model is None:
                raise ValueError("No model to save. Train model first.")
            
            model_path = f"{filepath}_lstm_model.h5"
            encoders_path = f"{filepath}_encoders.joblib"
            metadata_path = f"{filepath}_metadata.joblib"
            
            self.lstm_model.save(model_path)
            joblib.dump({
                'activity_encoder': self.activity_encoder,
                'lifecycle_encoder': self.lifecycle_encoder,
                'resource_encoder': self.resource_encoder
            }, encoders_path)
            joblib.dump({
                'sequence_length': self.sequence_length,
                'fallback_mean': self.fallback_mean,
                'fallback_std': self.fallback_std,
                'method': 'probabilistic_ml'
            }, metadata_path)
            
            print(f"Model saved to {filepath}_*.joblib and {filepath}_lstm_model.h5")
            
        elif self.method == "ml":
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
                'fallback_std': self.fallback_std,
                'method': 'ml'
            }, metadata_path)
            
            print(f"Model saved to {filepath}_*.joblib")

