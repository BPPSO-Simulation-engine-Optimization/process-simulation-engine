"""
Unified LSTM model with dual output heads for activity and lifecycle prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional


# Register custom layer for serialization
try:
    _register = keras.saving.register_keras_serializable
except AttributeError:
    try:
        _register = keras.utils.register_keras_serializable
    except AttributeError:
        def _register(package=None):
            return lambda fn: fn


@_register(package="UnifiedLayers")
def expand_and_cast(x):
    """Expand dimensions and cast to float32 for duration input."""
    return tf.cast(tf.expand_dims(x, axis=-1), tf.float32)


def build_unified_model(
    num_activities: int,
    num_lifecycles: int,
    num_resources: int,
    context_dim: int,
    max_seq_len: int,
    num_target_activities: int,
    num_target_lifecycles: int,
    activity_embed_dim: int = 64,
    lifecycle_embed_dim: int = 16,
    resource_embed_dim: int = 32,
    lstm_units: int = 256,
    hidden_units: int = 256,
    dropout: float = 0.3,
) -> keras.Model:
    """
    Build unified LSTM model with dual outputs and attention.

    Inputs: [activities, lifecycles, durations, resources, context]
    Outputs: [activity_probs, lifecycle_probs]
    """
    # Input layers
    act_input = keras.Input(shape=(max_seq_len,), name="activities")
    lc_input = keras.Input(shape=(max_seq_len,), name="lifecycles")
    dur_input = keras.Input(shape=(max_seq_len,), name="durations")
    res_input = keras.Input(shape=(max_seq_len,), name="resources")
    ctx_input = keras.Input(shape=(context_dim,), name="context")

    # Embeddings with larger dimensions
    act_embed = layers.Embedding(num_activities + 1, activity_embed_dim, mask_zero=True)(act_input)
    lc_embed = layers.Embedding(num_lifecycles + 1, lifecycle_embed_dim, mask_zero=True)(lc_input)
    res_embed = layers.Embedding(num_resources + 1, resource_embed_dim, mask_zero=True)(res_input)
    
    # Normalize durations (log transform + batch norm)
    dur_expanded = layers.Lambda(expand_and_cast)(dur_input)
    dur_normalized = layers.BatchNormalization()(dur_expanded)

    # Concatenate sequence features
    seq_concat = layers.Concatenate()([act_embed, lc_embed, dur_normalized, res_embed])

    # Stacked Bidirectional LSTM with return sequences for attention
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(seq_concat)
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(lstm_out)
    
    # Self-attention layer
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
    attention = layers.LayerNormalization()(attention + lstm_out)
    
    # Global pooling
    pooled = layers.GlobalAveragePooling1D()(attention)

    # Context processing with more capacity
    ctx_dense = layers.Dense(64, activation="relu")(ctx_input)
    ctx_dense = layers.BatchNormalization()(ctx_dense)

    # Merge sequence and context
    merged = layers.Concatenate()([pooled, ctx_dense])
    
    # Deeper hidden layers
    hidden = layers.Dense(hidden_units, activation="relu")(merged)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(dropout)(hidden)
    hidden = layers.Dense(hidden_units // 2, activation="relu")(hidden)
    hidden = layers.Dropout(dropout / 2)(hidden)

    # Dual output heads
    activity_output = layers.Dense(
        num_target_activities, activation="softmax", name="activity"
    )(hidden)
    lifecycle_output = layers.Dense(
        num_target_lifecycles, activation="softmax", name="lifecycle"
    )(hidden)

    model = keras.Model(
        inputs=[act_input, lc_input, dur_input, res_input, ctx_input],
        outputs=[activity_output, lifecycle_output],
    )

    # Use label smoothing to help with class imbalance
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "activity": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "lifecycle": keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        },
        loss_weights={"activity": 1.0, "lifecycle": 0.3},
        metrics={"activity": "accuracy", "lifecycle": "accuracy"},
    )

    return model


class UnifiedEncoder:
    """Encodes sequences for the unified model."""

    UNKNOWN_TOKEN = "UNKNOWN"

    def __init__(self, context_keys: List[str]):
        self.activity_encoder = LabelEncoder()
        self.lifecycle_encoder = LabelEncoder()
        self.resource_encoder = LabelEncoder()
        self.target_activity_encoder = LabelEncoder()
        self.target_lifecycle_encoder = LabelEncoder()
        self.context_keys = context_keys
        self.context_encoders = {}
        self.max_seq_len = None
        self.context_dim = 0

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Fit encoders and transform data."""
        # Collect all values for fitting
        all_activities = [act for seq in df["sequence_activities"] for act in seq]
        all_lifecycles = [lc for seq in df["sequence_lifecycles"] for lc in seq]
        all_resources = [res for seq in df["sequence_resources"] for res in seq]

        # Include targets in fitting
        all_activities.extend(df["target_activity"].tolist())
        all_lifecycles.extend(df["target_lifecycle"].tolist())

        # Fit encoders
        self.activity_encoder.fit(all_activities)
        self.lifecycle_encoder.fit(all_lifecycles)
        self.resource_encoder.fit(all_resources)
        self.target_activity_encoder.fit(df["target_activity"])
        self.target_lifecycle_encoder.fit(df["target_lifecycle"])

        # Add UNKNOWN token
        self._add_unknown_tokens()

        # Get max sequence length
        self.max_seq_len = df["sequence_activities"].apply(len).max()

        # Transform sequences
        X_acts = [self.activity_encoder.transform(seq) for seq in df["sequence_activities"]]
        X_lcs = [self.lifecycle_encoder.transform(seq) for seq in df["sequence_lifecycles"]]
        X_res = [self.resource_encoder.transform(seq) for seq in df["sequence_resources"]]
        X_durs = df["sequence_durations"].tolist()

        # Pad sequences
        X_acts = pad_sequences(X_acts, maxlen=self.max_seq_len, padding="pre")
        X_lcs = pad_sequences(X_lcs, maxlen=self.max_seq_len, padding="pre")
        X_res = pad_sequences(X_res, maxlen=self.max_seq_len, padding="pre")
        X_durs = pad_sequences(X_durs, maxlen=self.max_seq_len, padding="pre", dtype="float32")

        # Transform context
        X_ctx = self._transform_context(df)

        # Transform targets
        y_activity = self.target_activity_encoder.transform(df["target_activity"])
        y_lifecycle = self.target_lifecycle_encoder.transform(df["target_lifecycle"])

        return [X_acts, X_lcs, X_durs, X_res, X_ctx], y_activity, y_lifecycle

    def _add_unknown_tokens(self):
        """Add UNKNOWN token to all encoders."""
        for encoder in [
            self.activity_encoder,
            self.lifecycle_encoder,
            self.resource_encoder,
        ]:
            if self.UNKNOWN_TOKEN not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, self.UNKNOWN_TOKEN)

    def _transform_context(self, df: pd.DataFrame) -> np.ndarray:
        """Transform context attributes."""
        results = []

        for key in self.context_keys:
            if key not in df.columns:
                continue

            col = df[key]
            if col.dtype == "object" or str(col.dtype) == "category":
                enc = LabelEncoder()
                encoded = enc.fit_transform(col.fillna("MISSING").astype(str))
                self.context_encoders[key] = enc
                results.append(encoded.reshape(-1, 1))
            else:
                vals = col.fillna(0).values.reshape(-1, 1).astype("float32")
                self.context_encoders[key] = None
                results.append(vals)

        if results:
            result = np.hstack(results).astype("float32")
            self.context_dim = result.shape[1]
            return result

        self.context_dim = 0
        return np.zeros((len(df), 0), dtype="float32")

    def transform_single(
        self,
        activities: List[str],
        lifecycles: List[str],
        resources: List[str],
        durations: List[float],
        context: Dict,
    ) -> List[np.ndarray]:
        """Transform a single sequence for inference."""
        def safe_encode(values, encoder):
            known = set(encoder.classes_)
            cleaned = [v if v in known else self.UNKNOWN_TOKEN for v in values]
            return encoder.transform(cleaned)

        X_act = safe_encode(activities, self.activity_encoder)
        X_lc = safe_encode(lifecycles, self.lifecycle_encoder)
        X_res = safe_encode(resources, self.resource_encoder)
        X_dur = np.array(durations, dtype="float32")

        X_act = pad_sequences([X_act], maxlen=self.max_seq_len, padding="pre")
        X_lc = pad_sequences([X_lc], maxlen=self.max_seq_len, padding="pre")
        X_res = pad_sequences([X_res], maxlen=self.max_seq_len, padding="pre")
        X_dur = pad_sequences([X_dur], maxlen=self.max_seq_len, padding="pre", dtype="float32")

        # Context
        ctx_vals = []
        for key in self.context_keys:
            if key in context:
                enc = self.context_encoders.get(key)
                if enc is not None:
                    val = context[key]
                    if val in enc.classes_:
                        ctx_vals.append(float(enc.transform([val])[0]))
                    else:
                        ctx_vals.append(0.0)
                else:
                    ctx_vals.append(float(context[key]) if context[key] else 0.0)
            else:
                ctx_vals.append(0.0)

        X_ctx = np.array([ctx_vals], dtype="float32")

        return [X_act, X_lc, X_dur, X_res, X_ctx]

    @property
    def num_activities(self) -> int:
        return len(self.activity_encoder.classes_)

    @property
    def num_lifecycles(self) -> int:
        return len(self.lifecycle_encoder.classes_)

    @property
    def num_resources(self) -> int:
        return len(self.resource_encoder.classes_)

    @property
    def num_target_activities(self) -> int:
        return len(self.target_activity_encoder.classes_)

    @property
    def num_target_lifecycles(self) -> int:
        return len(self.target_lifecycle_encoder.classes_)


class UnifiedPredictor:
    """High-level predictor class for training and inference."""

    def __init__(
        self,
        context_keys: List[str] = None,
        max_seq_len: int = 15,
        lstm_units: int = 128,
        hidden_units: int = 128,
    ):
        self.context_keys = context_keys or [
            "case:LoanGoal",
            "case:ApplicationType",
            "case:RequestedAmount",
        ]
        self.max_seq_len = max_seq_len
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.encoder = None
        self.model = None

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 128,
        validation_split: float = 0.1,
        checkpoint_path: Optional[str] = None,
        use_class_weights: bool = True,
    ) -> keras.callbacks.History:
        """Train the unified model."""
        # Create encoder and transform data
        self.encoder = UnifiedEncoder(self.context_keys)
        X, y_act, y_lc = self.encoder.fit_transform(df)

        # Build model
        self.model = build_unified_model(
            num_activities=self.encoder.num_activities,
            num_lifecycles=self.encoder.num_lifecycles,
            num_resources=self.encoder.num_resources,
            context_dim=self.encoder.context_dim,
            max_seq_len=self.encoder.max_seq_len,
            num_target_activities=self.encoder.num_target_activities,
            num_target_lifecycles=self.encoder.num_target_lifecycles,
            lstm_units=self.lstm_units,
            hidden_units=self.hidden_units,
        )

        # Print class distribution info
        unique, counts = np.unique(y_act, return_counts=True)
        print(f"Activity classes: {len(unique)}, samples: {len(y_act)}")
        print(f"  Most common: {counts.max()} samples, Least common: {counts.min()} samples")

        # Callbacks with learning rate reduction
        callbacks = [
            EarlyStopping(
                monitor="val_activity_accuracy",
                patience=7,
                restore_best_weights=True,
                mode="max",
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_activity_accuracy",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                )
            )

        # Train
        history = self.model.fit(
            X,
            {"activity": y_act, "lifecycle": y_lc},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
        )

        return history

    def predict(
        self,
        activities: List[str],
        lifecycles: List[str],
        resources: List[str],
        durations: List[float],
        context: Dict,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Predict next activity and lifecycle."""
        X = self.encoder.transform_single(activities, lifecycles, resources, durations, context)
        preds = self.model.predict(X, verbose=0)

        activity_probs = preds[0][0]
        lifecycle_probs = preds[1][0]

        # Get top-k activities
        act_indices = np.argsort(activity_probs)[::-1][:top_k]
        activity_results = [
            (self.encoder.target_activity_encoder.classes_[i], float(activity_probs[i]))
            for i in act_indices
        ]

        # Get top-k lifecycles
        lc_indices = np.argsort(lifecycle_probs)[::-1][:top_k]
        lifecycle_results = [
            (self.encoder.target_lifecycle_encoder.classes_[i], float(lifecycle_probs[i]))
            for i in lc_indices
        ]

        return activity_results, lifecycle_results

