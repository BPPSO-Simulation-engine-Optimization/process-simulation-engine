from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
import warnings


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
        
        # Extract and fit distributions if using distribution method
        if method == "distribution":
            self._extract_and_fit_distributions()
        else:
            # For "ml" method, we would load/initialize an ML model here
            # For now, this is a placeholder
            pass

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
        
        else:
            # For ML method, we would use a trained model here
            # For now, return fallback
            return self.fallback_mean if self.fallback_mean else 3600.0

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
