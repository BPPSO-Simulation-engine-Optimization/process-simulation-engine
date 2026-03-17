import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Evaluator:
    """Handles model evaluation and performance analysis."""

    def __init__(self):
        """Initialize the Evaluator."""
        pass

    def calculate_regression_metrics(
        self,
        y_true_log: np.ndarray,
        y_pred_log: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics for log-transformed predictions.

        Args:
            y_true_log: True values (log-transformed)
            y_pred_log: Predicted values (log-transformed)

        Returns:
            Dictionary with MAE and RMSE in original scale (hours)
        """
        # Convert back from log10(x+1) scale to original scale
        y_true = np.power(10, y_true_log) - 1
        y_pred = np.power(10, y_pred_log) - 1

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        metrics = {
            "mae_hours": mae,
            "rmse_hours": rmse,
            "mae_log": mean_absolute_error(y_true_log, y_pred_log),
            "rmse_log": np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        }

        return metrics

    def evaluate_model(
        self,
        model_trainer,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Args:
            model_trainer: Trained ModelTrainer instance (or wrapper with predict/get_feature_importance)
            X_test: Test feature matrix
            y_test: Test target vector

        Returns:
            Dictionary with evaluation results
        """
        print("Evaluating model on test set...")

        y_pred_log = model_trainer.predict(X_test)

        metrics = self.calculate_regression_metrics(y_test.values, y_pred_log)

        feature_importance = model_trainer.get_feature_importance()

        results = {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "predictions": {
                "y_true_log": y_test.values,
                "y_pred_log": y_pred_log,
                "y_true_original": np.power(10, y_test.values) - 1,
                "y_pred_original": np.power(10, y_pred_log) - 1
            }
        }

        return results

    def print_evaluation_results(self, evaluation_results: Dict[str, Any]) -> None:
        """
        Print evaluation results in a formatted way.

        Args:
            evaluation_results: Results from evaluate_model
        """
        metrics = evaluation_results["metrics"]

        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)

        print("\nRegression Metrics (in hours):")
        print(f"MAE:  {metrics['mae_hours']:.2f}")
        print(f"RMSE: {metrics['rmse_hours']:.2f}")

        print("\nRegression Metrics (log scale):")
        print(f"MAE:  {metrics['mae_log']:.4f}")
        print(f"RMSE: {metrics['rmse_log']:.4f}")

        print("\nTop 15 Feature Importance:")
        feature_importance = evaluation_results["feature_importance"]
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

    def analyze_prediction_errors(
        self,
        evaluation_results: Dict[str, Any],
        X_test: pd.DataFrame,
        n_worst: int = 10
    ) -> pd.DataFrame:
        """
        Analyze prediction errors and return worst predictions.

        Args:
            evaluation_results: Results from evaluate_model
            X_test: Test feature matrix
            n_worst: Number of worst predictions to return

        Returns:
            DataFrame with worst predictions and their features
        """
        preds = evaluation_results["predictions"]
        y_true_orig = preds["y_true_original"]
        y_pred_orig = preds["y_pred_original"]

        errors = np.abs(y_true_orig - y_pred_orig)

        error_analysis = X_test.copy()
        error_analysis["true_processing_time"] = y_true_orig
        error_analysis["predicted_processing_time"] = y_pred_orig
        error_analysis["absolute_error"] = errors
        error_analysis["relative_error"] = errors / (y_true_orig + 1e-6)

        error_analysis = error_analysis.sort_values("absolute_error", ascending=False)

        return error_analysis.head(n_worst)

    def save_evaluation_results(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: str = "results"
    ) -> None:
        """
        Save evaluation results to files.

        Args:
            evaluation_results: Results from evaluate_model
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        fi_path = os.path.join(output_dir, "feature_importance.csv")
        evaluation_results["feature_importance"].to_csv(fi_path, index=False)
        print(f"Feature importance saved to: {fi_path}")

        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("="*30 + "\n\n")
            for key, value in evaluation_results["metrics"].items():
                f.write(f"{key}: {value:.4f}\n")
        print(f"Metrics saved to: {metrics_path}")

        print(f"All evaluation results saved to: {output_dir}/")
