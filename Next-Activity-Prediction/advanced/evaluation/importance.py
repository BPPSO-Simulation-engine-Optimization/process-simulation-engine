import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class FeatureImportanceAnalyzer:
    """Computes and visualizes permutation importance for LSTM model features."""

    def __init__(self, model, metric=accuracy_score):
        self.model = model
        self.metric = metric

    def _flatten_labels(self, y):
        if hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] == 1:
            return y.flatten()
        return y

    def _baseline_score(self, X_inputs, y_true):
        y_pred = np.argmax(self.model.predict(X_inputs, verbose=0), axis=1)
        return self.metric(y_true, y_pred)

    def context_importance(self, X_acts, X_durs, X_res, X_ctx, y_true, feature_names, n_repeats=5, seed=42):
        rng = np.random.RandomState(seed)
        y_flat = self._flatten_labels(y_true)
        baseline = self._baseline_score([X_acts, X_durs, X_res, X_ctx], y_flat)

        importances = []
        for i in range(len(feature_names)):
            scores = []
            for _ in range(n_repeats):
                X_perm = X_ctx.copy()
                perm_idx = rng.permutation(X_perm.shape[0])
                X_perm[:, i] = X_perm[perm_idx, i]

                y_pred = np.argmax(self.model.predict([X_acts, X_durs, X_res, X_perm], verbose=0), axis=1)
                scores.append(baseline - self.metric(y_flat, y_pred))

            importances.append(float(np.mean(scores)))

        return np.array(importances, dtype=float)

    def all_features_importance(self, X_inputs, y_true, context_feature_names, n_repeats=3, seed=42):
        rng = np.random.RandomState(seed)
        X_acts, X_durs, X_res, X_ctx = X_inputs
        y_flat = self._flatten_labels(y_true)
        baseline = self._baseline_score(X_inputs, y_flat)

        importances = []
        names = []

        for i, feat in enumerate(context_feature_names):
            scores = []
            for _ in range(n_repeats):
                X_perm = X_ctx.copy()
                perm_idx = rng.permutation(X_perm.shape[0])
                X_perm[:, i] = X_perm[perm_idx, i]

                y_pred = np.argmax(self.model.predict([X_acts, X_durs, X_res, X_perm], verbose=0), axis=1)
                scores.append(baseline - self.metric(y_flat, y_pred))

            importances.append(float(np.mean(scores)))
            names.append(feat)

        seq_features = [
            (0, X_acts, "activity_sequence"),
            (1, X_durs, "duration_sequence"),
            (2, X_res, "resource_sequence"),
        ]

        for idx, X_seq, name in seq_features:
            scores = []
            for _ in range(n_repeats):
                perm_idx = rng.permutation(X_seq.shape[0])
                X_seq_perm = X_seq[perm_idx]

                inputs_perm = [X_acts, X_durs, X_res, X_ctx]
                inputs_perm[idx] = X_seq_perm

                y_pred = np.argmax(self.model.predict(inputs_perm, verbose=0), axis=1)
                scores.append(baseline - self.metric(y_flat, y_pred))

            importances.append(float(np.mean(scores)))
            names.append(name)

        return np.array(importances, dtype=float), names, float(baseline)

    @staticmethod
    def plot(feature_names, importances, title="Feature Importance"):
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(9, 6))
        plt.barh(np.array(feature_names)[sorted_idx], np.array(importances)[sorted_idx])
        plt.xlabel("Permutation Importance (Δ accuracy)")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


def add_unknown_label(enc, token="UNKNOWN"):
    """Legacy function wrapper for backward compatibility."""
    if hasattr(enc, "classes_"):
        if token not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, token)
    return enc


def model_score(model, X, y_true, metric=accuracy_score):
    """Legacy function wrapper for backward compatibility."""
    y_pred = model.predict(X, verbose=0)
    y_pred_classes = y_pred.argmax(axis=-1)
    if hasattr(y_true, "shape") and len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    return metric(y_true, y_pred_classes)


def permutation_importance_context(model, X_acts, X_durs, X_res, X_ctx, y, feature_names, n_repeats=5, random_state=42, metric=accuracy_score):
    """Legacy function wrapper for backward compatibility."""
    analyzer = FeatureImportanceAnalyzer(model, metric)
    return analyzer.context_importance(X_acts, X_durs, X_res, X_ctx, y, feature_names, n_repeats, random_state)


def permutation_importance_all_features(model, X_input_list, y_true, context_feature_names, n_repeats=3, metric=accuracy_score, random_state=42):
    """Legacy function wrapper for backward compatibility."""
    analyzer = FeatureImportanceAnalyzer(model, metric)
    return analyzer.all_features_importance(X_input_list, y_true, context_feature_names, n_repeats, random_state)


def plot_feature_importance(feature_names, importances, title):
    """Legacy function wrapper for backward compatibility."""
    FeatureImportanceAnalyzer.plot(feature_names, importances, title)

