"""
Processing Time Trainer - Prefix-based Cumulative Time Prediction

Trains models to predict cumulative elapsed time given a prefix of events.
Input: Sequence of activities from the start of a trace (prefix)
Output: Total elapsed time from case start to current event
"""

from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def setup_gpu():
    """Configure GPU if available."""
    if not TF_AVAILABLE:
        return False
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detected: {len(gpus)} device(s)")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    print("Using CPU")
    return False


class ProcessingTimeTrainer:
    """
    Trains models to predict cumulative time given event prefixes.
    
    Methods:
        - 'ml': Random Forest on prefix features
        - 'lstm': LSTM on prefix sequences
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        method: str = "lstm",
        max_prefix_length: int = 50,
        embedding_dim: int = 32,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        batch_size: int = 128,
        epochs: int = 100,
        learning_rate: float = 0.001
    ):
        self.df = df.copy()
        self.method = method
        self.max_prefix_length = max_prefix_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.activity_encoder: Optional[LabelEncoder] = None
        self.num_activities = 0
        
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        
        self.y_mean = 0.0
        self.y_std = 1.0
        self.fallback_mean = 0.0

    def _extract_prefixes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract prefixes and cumulative times from event log.
        
        For each event in each case, creates a training sample:
        - X: prefix of activities up to and including current event
        - y: cumulative time from case start to current event
        """
        print("Extracting prefixes from event log...")
        
        self.df["time:timestamp"] = pd.to_datetime(self.df["time:timestamp"], errors="coerce")
        df_sorted = self.df.sort_values(["case:concept:name", "time:timestamp"]).copy()
        df_sorted = df_sorted.dropna(subset=["time:timestamp", "concept:name"])
        
        # Collect all activities
        all_activities = set(["<PAD>"])
        for act in df_sorted["concept:name"].unique():
            all_activities.add(str(act))
        
        self.activity_encoder = LabelEncoder()
        self.activity_encoder.fit(list(all_activities))
        self.num_activities = len(self.activity_encoder.classes_)
        
        pad_idx = self.activity_encoder.transform(["<PAD>"])[0]
        
        prefixes = []
        prefix_lengths = []
        cumulative_times = []
        
        for case_id, case_data in df_sorted.groupby("case:concept:name"):
            case_data = case_data.reset_index(drop=True)
            
            if len(case_data) < 2:
                continue
            
            case_start = case_data["time:timestamp"].iloc[0]
            
            # For each position in the trace, create a prefix sample
            for i in range(1, len(case_data)):
                current_time = case_data["time:timestamp"].iloc[i]
                elapsed = (current_time - case_start).total_seconds()
                
                if elapsed < 0 or elapsed > 365 * 24 * 3600:
                    continue
                
                # Get activities up to position i (prefix)
                activities = []
                for j in range(i + 1):
                    act = str(case_data["concept:name"].iloc[j])
                    activities.append(self.activity_encoder.transform([act])[0])
                
                # Truncate or pad
                if len(activities) > self.max_prefix_length:
                    activities = activities[-self.max_prefix_length:]
                
                prefix_len = len(activities)
                
                while len(activities) < self.max_prefix_length:
                    activities = [pad_idx] + activities
                
                prefixes.append(activities)
                prefix_lengths.append(prefix_len)
                cumulative_times.append(elapsed)
        
        X = np.array(prefixes, dtype=np.int32)
        lengths = np.array(prefix_lengths, dtype=np.int32)
        y = np.array(cumulative_times, dtype=np.float32)
        
        print(f"Extracted {len(y)} prefix samples")
        print(f"Cumulative times: mean={np.mean(y)/3600:.1f}h, median={np.median(y)/3600:.1f}h")
        
        return X, lengths, y

    def _build_lstm_model(self):
        """Build LSTM model for prefix-based prediction."""
        
        prefix_input = keras.Input(shape=(self.max_prefix_length,), dtype='int32', name='prefix')
        
        # Embedding layer
        x = layers.Embedding(
            self.num_activities, 
            self.embedding_dim,
            mask_zero=True,
            name='embedding'
        )(prefix_input)
        
        # LSTM layers
        x = layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)(x)
        x = layers.LSTM(self.lstm_units, dropout=self.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output
        output = layers.Dense(1, name='cumulative_time')(x)
        
        model = keras.Model(inputs=prefix_input, outputs=output)
        return model

    def train(self, save_path: Optional[str] = None):
        """Train the model."""
        
        print("=" * 70)
        print(f"Training {self.method.upper()} Model - Prefix Cumulative Time")
        print("=" * 70)
        
        # Extract data
        X, lengths, y = self._extract_prefixes()
        
        # Remove outliers
        mean_y, std_y = np.mean(y), np.std(y)
        valid = y <= mean_y + 3 * std_y
        X, lengths, y = X[valid], lengths[valid], y[valid]
        print(f"After outlier removal: {len(y)} samples")
        
        # Split
        X_train, X_val, len_train, len_val, y_train, y_val = train_test_split(
            X, lengths, y, test_size=0.2, random_state=42
        )
        print(f"Train: {len(y_train)}, Validation: {len(y_val)}")
        
        # Normalize targets (log transform)
        y_train_log = np.log(y_train + 1.0)
        y_val_log = np.log(y_val + 1.0)
        self.y_mean = float(np.mean(y_train_log))
        self.y_std = float(np.std(y_train_log)) + 1e-6
        y_train_norm = (y_train_log - self.y_mean) / self.y_std
        y_val_norm = (y_val_log - self.y_mean) / self.y_std
        
        self.fallback_mean = float(np.median(y))
        
        if self.method == "lstm":
            self._train_lstm(X_train, X_val, y_train_norm, y_val_norm, y_train, y_val)
        else:
            self._train_rf(X_train, X_val, len_train, len_val, y_train, y_val)
        
        if save_path:
            self.save(save_path)

    def _train_lstm(self, X_train, X_val, y_train_norm, y_val_norm, y_train_raw, y_val_raw):
        """Train LSTM model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM")
        
        gpu_available = setup_gpu()
        batch_size = self.batch_size * 2 if gpu_available else self.batch_size
        
        print("\nBuilding LSTM model...")
        self.model = self._build_lstm_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        self.model.summary()
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_norm))
        train_ds = train_ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_norm))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        print("\nTraining...")
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_pred_norm = self.model.predict(X_val, verbose=0).flatten()
        val_pred = np.exp(val_pred_norm * self.y_std + self.y_mean) - 1.0
        
        self._print_metrics(y_val_raw, val_pred)

    def _train_rf(self, X_train, X_val, len_train, len_val, y_train, y_val):
        """Train Random Forest model."""
        print("\nTraining Random Forest...")
        
        # Flatten prefixes + add length as feature
        X_train_flat = np.hstack([X_train, len_train.reshape(-1, 1)])
        X_val_flat = np.hstack([X_val, len_val.reshape(-1, 1)])
        
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        self.model.fit(X_train_flat, y_train)
        val_pred = self.model.predict(X_val_flat)
        
        self._print_metrics(y_val, val_pred)

    def _print_metrics(self, y_true, y_pred):
        """Print evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        rel_errors = np.abs(y_true - y_pred) / (y_true + 1e-6)
        within_25 = np.mean(rel_errors <= 0.25) * 100
        within_50 = np.mean(rel_errors <= 0.50) * 100
        
        print("\n" + "-" * 50)
        print("Validation Results")
        print("-" * 50)
        print(f"MAE:        {mae/3600:.2f} hours ({mae:.0f}s)")
        print(f"RMSE:       {rmse/3600:.2f} hours ({rmse:.0f}s)")
        print(f"R²:         {r2:.4f}")
        print(f"Within 25%: {within_25:.1f}%")
        print(f"Within 50%: {within_50:.1f}%")
        print("=" * 70)

    def save(self, filepath: str):
        """Save model and config."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        if self.method == "lstm" and TF_AVAILABLE:
            self.model.save(f"{filepath}_model.keras")
        else:
            joblib.dump(self.model, f"{filepath}_model.joblib")
        
        joblib.dump({
            'method': self.method,
            'activity_encoder': self.activity_encoder,
            'num_activities': self.num_activities,
            'max_prefix_length': self.max_prefix_length,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'fallback_mean': self.fallback_mean
        }, f"{filepath}_config.joblib")
        
        print(f"Model saved to {filepath}")
