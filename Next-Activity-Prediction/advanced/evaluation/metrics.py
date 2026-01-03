import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences


class _ContextEncoder:
    """Minimal context encoder for evaluation."""

    def __init__(self, keys):
        self.keys = keys
        self.encoders = {}

    def fit_transform(self, df):
        result = df[self.keys].copy()
        for col in self.keys:
            col_data = df[col]
            if col_data.dtype == "object" or col_data.dtype.name == "string":
                enc = LabelEncoder()
                result[col] = enc.fit_transform(col_data.astype(str))
                self.encoders[col] = enc
            else:
                scaler = StandardScaler()
                result[col] = scaler.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                self.encoders[col] = scaler
        return result.to_numpy().astype("float32")


class ModelEvaluator:
    """Evaluates LSTM models and compares against baselines."""

    def __init__(self, model_bundle):
        self.model = model_bundle["model"]
        self.activity_enc = model_bundle["activity_encoder"]
        self.resource_enc = model_bundle["resource_encoder"]
        self.label_enc = model_bundle["label_encoder"]
        self.context_keys = model_bundle["context_keys"]
        self.max_seq_len = model_bundle["max_seq_len"]

    def _ensure_unknown_token(self, encoder):
        if "UNKNOWN" not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, "UNKNOWN")

    def _clean_sequences(self, sequences, known_set, encoder):
        self._ensure_unknown_token(encoder)
        cleaned = [[item if item in known_set else "UNKNOWN" for item in seq] for seq in sequences]
        return [encoder.transform(seq) for seq in cleaned]

    def prepare_holdout(self, df_holdout):
        known_acts = set(self.activity_enc.classes_)
        known_res = set(self.resource_enc.classes_)

        X_acts = self._clean_sequences(df_holdout["sequence"], known_acts, self.activity_enc)
        X_res = self._clean_sequences(df_holdout["sequence_resources"], known_res, self.resource_enc)
        X_durs = df_holdout["sequence_durations"].tolist()

        ctx_enc = _ContextEncoder(self.context_keys)
        X_ctx = ctx_enc.fit_transform(df_holdout)

        y_true = self.label_enc.transform(df_holdout["label"])

        X_acts_pad = pad_sequences(X_acts, maxlen=self.max_seq_len, padding="pre")
        X_res_pad = pad_sequences(X_res, maxlen=self.max_seq_len, padding="pre")
        X_durs_pad = pad_sequences(X_durs, maxlen=self.max_seq_len, padding="pre", dtype="float32")

        return [X_acts_pad, X_durs_pad, X_res_pad, X_ctx], y_true

    def evaluate(self, X_test, y_test, print_report=True):
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        if print_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_enc.classes_, zero_division=0))

            cm = confusion_matrix(y_test, y_pred)
            per_class_acc = cm.diagonal() / cm.sum(axis=1)

            print("\nPer-Class Accuracy:")
            for label, acc in zip(self.label_enc.classes_, per_class_acc):
                print(f"{label}: {acc:.2%}")

        return classification_report(y_test, y_pred, target_names=self.label_enc.classes_, output_dict=True, zero_division=0)

    def baseline_report(self, y_true, seed=42):
        np.random.seed(seed)

        counts = np.bincount(y_true)
        probs = counts / counts.sum()

        y_pred = np.random.choice(np.arange(len(probs)), size=len(y_true), p=probs)

        return classification_report(
            y_true,
            y_pred,
            target_names=self.label_enc.classes_,
            output_dict=True,
            zero_division=0,
        )

    def compare_with_baseline(self, df_holdout, print_result=True):
        if df_holdout.empty:
            return None

        X_test, y_true = self.prepare_holdout(df_holdout)

        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        lstm_report = classification_report(
            y_true, y_pred, target_names=self.label_enc.classes_, output_dict=True, zero_division=0
        )
        baseline = self.baseline_report(y_true)

        f1_lstm = lstm_report["weighted avg"]["f1-score"]
        f1_base = baseline["weighted avg"]["f1-score"]
        improvement = (f1_lstm - f1_base) / f1_base * 100 if f1_base > 0 else float("inf")

        if print_result:
            print(f"Weighted F1 (LSTM):     {f1_lstm:.3f}")
            print(f"Weighted F1 (Baseline): {f1_base:.3f}")
            print(f"Relative Improvement:   {improvement:.2f}%")

        return {
            "f1_lstm": f1_lstm,
            "f1_baseline": f1_base,
            "relative_improvement": improvement,
        }


def evaluate_model(model, X_test, y_test, label_encoder):
    """Legacy function wrapper for backward compatibility."""
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_), zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\nPer-Class Accuracy:")
    for label, acc in zip(label_encoder.classes_, per_class_acc):
        print(f"{label}: {acc:.2%}")


def evaluate_baseline(y_true, label_encoder, random_seed=42):
    """Legacy function wrapper for backward compatibility."""
    np.random.seed(random_seed)

    counts = np.bincount(y_true)
    probs = counts / counts.sum()

    y_pred = np.random.choice(np.arange(len(probs)), size=len(y_true), p=probs)

    return classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )


def compare_f1_for_trained_model(dp, data_per_dp, decision_point_models):
    """Legacy function wrapper for backward compatibility."""
    if dp not in data_per_dp or dp not in decision_point_models:
        print(f"Missing data or model for {dp}")
        return None

    df_holdout = data_per_dp[dp]
    if df_holdout.empty:
        print(f"Holdout set for {dp} is empty.")
        return None

    evaluator = ModelEvaluator(decision_point_models[dp])
    print(f"\nComparison for {dp} (Holdout Set)")
    result = evaluator.compare_with_baseline(df_holdout, print_result=True)

    if result:
        result["dp"] = dp
    return result

