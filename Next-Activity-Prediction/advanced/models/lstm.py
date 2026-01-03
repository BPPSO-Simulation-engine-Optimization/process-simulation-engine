import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Lambda
from tensorflow.keras.models import Model


def _expand_and_cast(x):
    return tf.cast(tf.expand_dims(x, axis=-1), tf.float32)


class LSTMPredictor:
    """Builds, compiles, and trains LSTM models for next-activity prediction."""

    def __init__(
        self,
        num_activities,
        num_resources,
        context_dim,
        max_seq_len,
        num_classes,
        activity_embed_dim=64,
        resource_embed_dim=16,
        lstm_units=128,
        dense_units=64,
    ):
        self.num_activities = num_activities
        self.num_resources = num_resources
        self.context_dim = context_dim
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.activity_embed_dim = activity_embed_dim
        self.resource_embed_dim = resource_embed_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.model = None

    def build(self):
        activity_in = Input(shape=(self.max_seq_len,), name="activity_input")
        duration_in = Input(shape=(self.max_seq_len,), name="duration_input", dtype="float32")
        resource_in = Input(shape=(self.max_seq_len,), name="resource_input")
        context_in = Input(shape=(self.context_dim,), name="context_input")

        act_emb = Embedding(
            input_dim=self.num_activities,
            output_dim=self.activity_embed_dim,
            name="activity_embedding",
        )(activity_in)

        res_emb = Embedding(
            input_dim=self.num_resources,
            output_dim=self.resource_embed_dim,
            name="resource_embedding",
        )(resource_in)

        dur_expanded = Lambda(
            _expand_and_cast,
            output_shape=lambda s: (s[0], s[1], 1),
            name="expand_and_cast",
        )(duration_in)

        seq_combined = Concatenate(axis=-1)([act_emb, dur_expanded, res_emb])
        seq_out = LSTM(units=self.lstm_units, return_sequences=False)(seq_combined)

        merged = Concatenate()([seq_out, context_in])
        hidden = Dense(self.dense_units, activation="relu")(merged)
        output = Dense(self.num_classes, activation="softmax")(hidden)

        self.model = Model(
            inputs=[activity_in, duration_in, resource_in, context_in],
            outputs=output,
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self

    def train(self, X_train, y_train, epochs=10, batch_size=128, patience=3, baseline=0.95):
        early_stop = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            min_delta=1e-4,
            baseline=baseline,
            restore_best_weights=True,
            verbose=1,
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            callbacks=[early_stop],
            epochs=epochs,
            batch_size=batch_size,
        )
        return history

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_classes(self, X_test):
        probs = self.predict(X_test)
        return probs.argmax(axis=1)


def expand_and_cast(x):
    """Legacy function wrapper for backward compatibility."""
    return _expand_and_cast(x)


def build_lstm_model(num_activities, num_resources, context_dim, max_seq_len, num_classes):
    """Legacy function wrapper for backward compatibility."""
    predictor = LSTMPredictor(
        num_activities=num_activities,
        num_resources=num_resources,
        context_dim=context_dim,
        max_seq_len=max_seq_len,
        num_classes=num_classes,
    )
    predictor.build()
    return predictor.model


def train_model(model, X_train, y_train):
    """Legacy function wrapper for backward compatibility."""
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        min_delta=1e-4,
        baseline=0.95,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        callbacks=[early_stop],
        epochs=10,
        batch_size=128,
    )
    return history

