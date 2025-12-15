from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from scipy import stats
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ProcessingTimePredictionClass:
    """
    Predicts processing times between consecutive events.
    """

    def __init__(
        self,
        method: str = "ml",
        model_path: Optional[str] = None,
    ):
        """
        Initialize the prediction class by loading a model from disk.

        Args:
            method: Method to use ("distribution", "ml", or "probabilistic_ml").
            model_path: Base path of the saved model (without suffixes like _model.joblib);
                        if None, a default path ``models/processing_time_model`` is used.
        """
        self.method = method

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
        self.event_history: List[Dict] = []

        base_path = model_path or "models/processing_time_model"
        self.load_model(base_path)

    def _prepare_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model prediction: encoding, scaling, imputation.
        
        Args:
            df_features: DataFrame with raw features
        
        Returns:
            DataFrame with prepared features
        """
        df = df_features.copy()
        
        categorical_cols = ['prev_activity', 'prev_lifecycle', 'curr_activity', 'curr_lifecycle',
                           'prev_resource', 'curr_resource', 'case:LoanGoal', 'case:ApplicationType']
        numerical_cols = ['hour', 'weekday', 'month', 'day_of_year',
                        'event_position_in_case', 'case_duration_so_far',]
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
                df[col] = df[col].fillna(self.feature_defaults.get(col, 0.0))
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.feature_defaults.get(col, "unknown"))
                df[col] = df[col].astype(str)
        
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
        
        if self.scaler is not None:
            missing_num_cols = [c for c in numerical_cols if c not in df.columns]
            for col in missing_num_cols:
                df[col] = self.feature_defaults.get(col, 0.0)
            
            cols_to_scale = [c for c in numerical_cols if c in df.columns]
            if cols_to_scale:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df

    def load_model(self, filepath: str):
        """
        Load trained model and preprocessors from disk.
        
        Args:
            filepath: Base path for loading (will load multiple files)
        """
        metadata_path = f"{filepath}_metadata.joblib"
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata file not found at {metadata_path}")
        
        metadata = joblib.load(metadata_path)
        self.method = metadata.get('method', 'ml')
        self.fallback_mean = metadata['fallback_mean']
        self.fallback_std = metadata['fallback_std']
        
        if self.method == "distribution":
            distributions_path = f"{filepath}_distributions.joblib"
            if not os.path.exists(distributions_path):
                raise FileNotFoundError(f"Distributions file not found at {distributions_path}")
            
            distributions_serializable = joblib.load(distributions_path)
            
            self.distributions = {}
            for key, value in distributions_serializable.items():
                dist = stats.lognorm(s=value['sigma'], scale=np.exp(value['mu']))
                self.distributions[key] = {
                    'distribution': dist,
                    'mu': value['mu'],
                    'sigma': value['sigma'],
                    'count': value['count'],
                    'mean': value['mean'],
                    'std': value['std'],
                    'median': value['median']
                }
            
            print(f"Distribution model loaded from {filepath}_*.joblib")
            
        elif self.method == "ml":
            model_path = f"{filepath}_model.joblib"
            encoders_path = f"{filepath}_encoders.joblib"
            scaler_path = f"{filepath}_scaler.joblib"
            
            if not all(os.path.exists(p) for p in [model_path, encoders_path, scaler_path]):
                raise FileNotFoundError(f"ML model files not found at {filepath}")
            
            self.ml_model = joblib.load(model_path)
            self.label_encoders = joblib.load(encoders_path)
            self.scaler = joblib.load(scaler_path)
            
            self.feature_names = metadata['feature_names']
            self.categorical_features = metadata['categorical_features']
            self.numerical_features = metadata['numerical_features']
            self.feature_defaults = metadata['feature_defaults']
            
            print(f"ML model loaded from {filepath}_*.joblib")
            
        elif self.method == "probabilistic_ml":
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is required for probabilistic_ml method. Install with: pip install tensorflow")
            
            model_path = f"{filepath}_lstm_model.h5"
            encoders_path = f"{filepath}_encoders.joblib"
            
            if not all(os.path.exists(p) for p in [model_path, encoders_path]):
                raise FileNotFoundError(f"Probabilistic ML model files not found at {filepath}")
            
            self.lstm_model = keras.models.load_model(model_path, compile=False)
            encoders = joblib.load(encoders_path)
            self.activity_encoder = encoders['activity_encoder']
            self.lifecycle_encoder = encoders['lifecycle_encoder']
            self.resource_encoder = encoders['resource_encoder']
            self.sequence_length = metadata.get('sequence_length', 10)
            
            print(f"Probabilistic ML model loaded from {filepath}_*.joblib and {filepath}_lstm_model.h5")

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
            if transition_key in self.distributions:
                dist_info = self.distributions[transition_key]
                sample = dist_info['distribution'].rvs(size=1)[0]
                return max(0.0, float(sample))
            
            for key, dist_info in self.distributions.items():
                if key[0] == prev_activity and key[2] == curr_activity:
                    sample = dist_info['distribution'].rvs(size=1)[0]
                    return max(0.0, float(sample))
            
            if self.fallback_mean and self.fallback_std:
                cv = self.fallback_std / self.fallback_mean
                sigma_approx = np.sqrt(np.log(1 + cv**2))
                mu_approx = np.log(self.fallback_mean) - 0.5 * sigma_approx**2
                
                fallback_dist = stats.lognorm(s=sigma_approx, scale=np.exp(mu_approx))
                sample = fallback_dist.rvs(size=1)[0]
                return max(0.0, float(sample))
            
            return self.fallback_mean if self.fallback_mean else 3600.0
        
        elif self.method == "probabilistic_ml":
            if self.lstm_model is None:
                warnings.warn("LSTM model not loaded. Using fallback.")
                return self.fallback_mean if self.fallback_mean else 3600.0
            
            try:
                seq_input, ctx_input = self._prepare_lstm_input(
                    prev_activity, prev_lifecycle, curr_activity, curr_lifecycle, context
                )
                
                pred = self.lstm_model.predict([seq_input, ctx_input], verbose=0)
                mean_pred_val = max(0.0, float(pred[0, 0]))
                log_var_pred = pred[0, 1]
                std_pred = np.sqrt(np.exp(log_var_pred) + 1e-6)
                
                sample = np.random.normal(mean_pred_val, std_pred)
                sample = max(0.0, float(sample))
                
                event_info = {
                    'activity': prev_activity,
                    'lifecycle': prev_lifecycle,
                    'resource': context.get('resource_1', 'unknown') if context else 'unknown'
                }
                self.event_history.append(event_info)
                if len(self.event_history) > self.sequence_length * 2:
                    self.event_history = self.event_history[-self.sequence_length:]
                
                return sample
                
            except Exception as e:
                warnings.warn(f"Error in probabilistic ML prediction: {e}. Using fallback.")
                return self.fallback_mean if self.fallback_mean else 3600.0
        
        else:
            if self.ml_model is None:
                warnings.warn("ML model not trained. Using fallback.")
                return self.fallback_mean if self.fallback_mean else 3600.0
            
            try:
                feature_dict = self._context_to_features(
                    prev_activity, prev_lifecycle, curr_activity, curr_lifecycle, context
                )
                
                df_features = pd.DataFrame([feature_dict])
                df_prepared = self._prepare_features(df_features)
                
                missing_features = set(self.feature_names) - set(df_prepared.columns)
                if missing_features:
                    for feat in missing_features:
                        df_prepared[feat] = self.feature_defaults.get(feat, 0.0)
                
                df_prepared = df_prepared[self.feature_names]
                
                prediction = self.ml_model.predict(df_prepared)[0]
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
        
        features = {}
        
        features['prev_activity'] = str(prev_activity) if prev_activity else "unknown"
        features['prev_lifecycle'] = str(prev_lifecycle) if prev_lifecycle else "complete"
        features['curr_activity'] = str(curr_activity) if curr_activity else "unknown"
        features['curr_lifecycle'] = str(curr_lifecycle) if curr_lifecycle else "complete"
        
        features['prev_resource'] = str(context.get('resource_1', 'unknown'))
        features['curr_resource'] = str(context.get('resource_2', 'unknown'))
        
        features['case:LoanGoal'] = context.get('case:LoanGoal', None)
        features['case:ApplicationType'] = context.get('case:ApplicationType', None)
        
        features['Accepted'] = context.get('Accepted', None)
        features['Selected'] = context.get('Selected', None)
        
        from datetime import datetime
        if 'hour' in context:
            features['hour'] = context['hour']
            features['weekday'] = context.get('weekday', 0)
            features['month'] = context.get('month', 1)
            features['day_of_year'] = context.get('day_of_year', (context.get('month', 1) - 1) * 30 + 15)
        else:
            now = datetime.now()
            features['hour'] = now.hour
            features['weekday'] = now.weekday()
            features['month'] = now.month
            features['day_of_year'] = now.timetuple().tm_yday
        
        features['event_position_in_case'] = context.get('event_position_in_case', 1)
        features['case_duration_so_far'] = context.get('case_duration_so_far', 0.0)
        
        return features

    def _prepare_lstm_input(
        self,
        prev_activity: str,
        prev_lifecycle: str,
        curr_activity: str,
        curr_lifecycle: str,
        context: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if context is None:
            context = {}
        
        sequence = []
        for event in self.event_history[-self.sequence_length:]:
            try:
                activity_idx = self.activity_encoder.transform([event['activity']])[0]
                lifecycle_idx = self.lifecycle_encoder.transform([event['lifecycle']])[0]
                resource_idx = self.resource_encoder.transform([event['resource']])[0]
                sequence.append([activity_idx, lifecycle_idx, resource_idx])
            except:
                sequence.append([0, 0, 0])
        
        try:
            activity_idx = self.activity_encoder.transform([prev_activity])[0]
        except:
            activity_idx = 0
        try:
            lifecycle_idx = self.lifecycle_encoder.transform([prev_lifecycle])[0]
        except:
            lifecycle_idx = 0
        try:
            resource = context.get('resource_1', 'unknown') if context else 'unknown'
            resource_idx = self.resource_encoder.transform([resource])[0]
        except:
            resource_idx = 0
        
        sequence.append([activity_idx, lifecycle_idx, resource_idx])
        
        while len(sequence) < self.sequence_length:
            sequence = [[0, 0, 0]] + sequence
        
        sequence = sequence[-self.sequence_length:]
        seq_array = np.array([sequence])
        
        from datetime import datetime
        if 'hour' in context:
            hour = context['hour'] / 24.0
            weekday = context.get('weekday', 0) / 7.0
            month = context.get('month', 1) / 12.0
            day_of_year = context.get('day_of_year', (context.get('month', 1) - 1) * 30 + 15) / 365.0
        else:
            now = datetime.now()
            hour = now.hour / 24.0
            weekday = now.weekday() / 7.0
            month = now.month / 12.0
            day_of_year = now.timetuple().tm_yday / 365.0
        
        event_pos = context.get('event_position_in_case', 1) / 100.0
        case_duration = context.get('case_duration_so_far', 0.0) / 86400.0
        
        loan_goal = 1.0 if str(context.get('case:LoanGoal', '')) == 'Car' else 0.0
        app_type = 1.0 if str(context.get('case:ApplicationType', '')) == 'New' else 0.0
        
        ctx_array = np.array([[hour, weekday, month, day_of_year, event_pos, case_duration, loan_goal, app_type]])
        
        return seq_array, ctx_array

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
                info.pop('distribution')
                return info
            else:
                return {'error': f'Transition {transition_key} not found'}
    
    def get_probabilistic_distribution(
        self,
        prev_activity: str,
        prev_lifecycle: str,
        curr_activity: str,
        curr_lifecycle: str,
        context: Optional[Dict] = None
    ) -> Dict:
        if self.method != "probabilistic_ml":
            return {'error': 'Method is not probabilistic_ml'}
        
        if self.lstm_model is None:
            return {'error': 'LSTM model not loaded'}
        
        try:
            seq_input, ctx_input = self._prepare_lstm_input(
                prev_activity, prev_lifecycle, curr_activity, curr_lifecycle, context
            )
            
            pred = self.lstm_model.predict([seq_input, ctx_input], verbose=0)
            mean_pred_val = max(0.0, float(pred[0, 0]))
            log_var_pred = pred[0, 1]
            std_pred = np.sqrt(np.exp(log_var_pred) + 1e-6)
            
            return {
                'mean': mean_pred_val,
                'std': float(std_pred),
                'distribution_type': 'gaussian'
            }
        except Exception as e:
            return {'error': str(e)}
