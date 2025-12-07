import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import holidays
from sklearn.cluster import KMeans


class ResourceAvailabilityModel:
    """
    Basic model for resource availabilities based on interval patterns.
    
    Simulates resource availability using:
    - 2-week (14-day) cyclic pattern
    - Configurable working hours (default: 8:00-17:00)
    - Configurable working days in the 2-week cycle
    - Dutch public holidays awareness
    """

    def __init__(
        self,
        event_log_df: pd.DataFrame,
        interval_days: int = 14,
        workday_start_hour: int = 8,
        workday_end_hour: int = 17,
        working_cycle_days: Optional[Set[int]] = None,
    ):
        """
        Initialize the basic resource availability model.
        
        Args:
            event_log_df: Event log as DataFrame with 'org:resource' and 'time:timestamp'
            interval_days: Length of the work cycle (default: 14 days)
            workday_start_hour: Start of working hours (default: 8)
            workday_end_hour: End of working hours (default: 17)
            working_cycle_days: Set of working days in the cycle (0-indexed)
                               Default: Mon-Fri both weeks (0-4, 7-11 in 14-day cycle)
        """
        self.event_log_df = event_log_df.copy()
        self.interval_days = interval_days
        self.workday_start_hour = workday_start_hour
        self.workday_end_hour = workday_end_hour

        # Default: Mon-Fri in both weeks of 2-week cycle
        if working_cycle_days is None:
            working_cycle_days = {0, 1, 2, 3, 4, 7, 8, 9, 10, 11}
        self.working_cycle_days = working_cycle_days

        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(self.event_log_df["time:timestamp"]):
            self.event_log_df["time:timestamp"] = pd.to_datetime(
                self.event_log_df["time:timestamp"], errors="coerce"
            )

        # Set cycle anchor to Monday of first week
        first_ts = self.event_log_df["time:timestamp"].min()
        first_date = first_ts.normalize().date()
        monday_date = first_date - timedelta(days=first_ts.weekday())  # Mon=0
        self.cycle_start_date = monday_date

        # Dutch public holidays
        years = range(self.cycle_start_date.year, self.cycle_start_date.year + 20)
        self.nl_holidays = set(holidays.Netherlands(years=years).keys())

        self.resources = sorted(self.event_log_df["org:resource"].dropna().unique())

    def _cycle_day_index(self, current_time: datetime) -> int:
        """Calculate the day index within the cycle (0 to interval_days-1)."""
        delta_days = (current_time.date() - self.cycle_start_date).days
        return delta_days % self.interval_days

    def is_working_time(self, current_time: datetime) -> bool:
        """
        Check if current_time is within general working time.
        
        Checks:
        1. Not a Dutch public holiday
        2. Within the working days of the cycle
        3. Within working hours
        """
        current_date = current_time.date()

        # 1) Holiday check
        if current_date in self.nl_holidays:
            return False

        # 2) Cycle pattern check
        cycle_day = self._cycle_day_index(current_time)
        if cycle_day not in self.working_cycle_days:
            return False

        # 3) Hour check
        hour = current_time.hour
        return self.workday_start_hour <= hour < self.workday_end_hour

    def is_available(self, resource_id: str, current_time: datetime) -> bool:
        """Check if a resource is available at the given time."""
        if resource_id not in self.resources:
            return False
        return self.is_working_time(current_time)

    def get_next_available_time(self, resource_id: str, current_time: datetime) -> Optional[datetime]:
        """Find the next available time for a resource starting from current_time."""
        if resource_id not in self.resources:
            return None
        
        # Search up to 30 days ahead
        check_time = current_time
        for _ in range(30 * 24):  # Check hour by hour for 30 days
            if self.is_available(resource_id, check_time):
                return check_time
            check_time += timedelta(hours=1)
        
        return None


class AdvancedResourceAvailabilityModel(ResourceAvailabilityModel):
    """
    Advanced model with resource mining capabilities.
    
    Learns individual resource patterns from historical data including:
    - Resource-specific working hours
    - Day-of-week preferences
    - Activity intensity patterns (peak/off-peak hours)
    - Resource clustering based on similar work patterns
    - Probabilistic availability based on historical activity
    """

    def __init__(
        self,
        event_log_df: pd.DataFrame,
        interval_days: int = 14,
        workday_start_hour: int = 8,
        workday_end_hour: int = 17,
        working_cycle_days: Optional[Set[int]] = None,
        enable_pattern_mining: bool = True,
        min_activity_threshold: int = 10,
    ):
        """
        Initialize the advanced resource availability model with pattern mining.
        
        Args:
            event_log_df: Event log DataFrame
            interval_days: Cycle length in days
            workday_start_hour: Default start hour
            workday_end_hour: Default end hour
            working_cycle_days: Default working days
            enable_pattern_mining: Whether to mine resource patterns from data
            min_activity_threshold: Minimum activities required to mine patterns
        """
        super().__init__(
            event_log_df, 
            interval_days, 
            workday_start_hour, 
            workday_end_hour, 
            working_cycle_days
        )
        
        self.enable_pattern_mining = enable_pattern_mining
        self.min_activity_threshold = min_activity_threshold
        
        # Resource-specific patterns
        self.resource_patterns: Dict[str, Dict] = {}
        self.resource_clusters: Dict[str, int] = {}
        self.cluster_profiles: Dict[int, Dict] = {}
        
        if enable_pattern_mining:
            self._mine_resource_patterns()
            self._cluster_resources()

    def _mine_resource_patterns(self):
        """Mine work patterns for each resource from historical event log."""
        print("Mining resource patterns...")
        
        df = self.event_log_df.copy()
        df['hour'] = df['time:timestamp'].dt.hour
        df['day_of_week'] = df['time:timestamp'].dt.dayofweek
        df['date'] = df['time:timestamp'].dt.date
        
        for resource in self.resources:
            resource_df = df[df['org:resource'] == resource]
            
            if len(resource_df) < self.min_activity_threshold:
                # Not enough data, use defaults
                self.resource_patterns[resource] = self._get_default_pattern()
                continue
            
            # Extract temporal patterns
            hour_counts = resource_df['hour'].value_counts()
            dow_counts = resource_df['day_of_week'].value_counts()
            
            # Working hours (hours with activity)
            active_hours = sorted(hour_counts.index.tolist())
            working_start = min(active_hours) if active_hours else self.workday_start_hour
            working_end = max(active_hours) + 1 if active_hours else self.workday_end_hour
            
            # Peak hours (top 25% of activity)
            hour_threshold = hour_counts.quantile(0.75)
            peak_hours = set(hour_counts[hour_counts >= hour_threshold].index.tolist())
            
            # Working days (days with activity)
            working_days = set(dow_counts.index.tolist())
            
            # Activity probability by hour (normalized)
            hour_probs = (hour_counts / hour_counts.sum()).to_dict()
            
            # Activity probability by day of week (normalized)
            dow_probs = (dow_counts / dow_counts.sum()).to_dict()
            
            # Date range
            first_activity = resource_df['time:timestamp'].min()
            last_activity = resource_df['time:timestamp'].max()
            
            # Activity intensity (activities per day)
            active_days = resource_df['date'].nunique()
            activity_intensity = len(resource_df) / active_days if active_days > 0 else 0
            
            self.resource_patterns[resource] = {
                'working_start': working_start,
                'working_end': working_end,
                'working_days': working_days,
                'peak_hours': peak_hours,
                'hour_probabilities': hour_probs,
                'dow_probabilities': dow_probs,
                'first_activity': first_activity,
                'last_activity': last_activity,
                'total_activities': len(resource_df),
                'activity_intensity': activity_intensity,
                'active_days': active_days,
            }
        
        print(f"Mined patterns for {len(self.resource_patterns)} resources")

    def _get_default_pattern(self) -> Dict:
        """Get default pattern for resources with insufficient data."""
        return {
            'working_start': self.workday_start_hour,
            'working_end': self.workday_end_hour,
            'working_days': {0, 1, 2, 3, 4},  # Mon-Fri
            'peak_hours': {9, 10, 11, 13, 14},
            'hour_probabilities': {},
            'dow_probabilities': {},
            'first_activity': None,
            'last_activity': None,
            'total_activities': 0,
            'activity_intensity': 0,
            'active_days': 0,
        }

    def _cluster_resources(self, n_clusters: int = 5):
        """Cluster resources based on their work patterns."""
        print(f"Clustering resources into {n_clusters} groups...")
        
        # Create feature vectors for clustering
        features = []
        resource_list = []
        
        for resource, pattern in self.resource_patterns.items():
            if pattern['total_activities'] < self.min_activity_threshold:
                continue
            
            # Feature vector: [working_start, working_end, num_working_days, 
            #                  activity_intensity, hour_spread, dow_spread]
            feature_vec = [
                pattern['working_start'],
                pattern['working_end'],
                len(pattern['working_days']),
                np.log1p(pattern['activity_intensity']),  # Log transform for scale
                pattern['working_end'] - pattern['working_start'],  # Hour spread
                len(pattern['dow_probabilities']),  # Day spread
            ]
            
            features.append(feature_vec)
            resource_list.append(resource)
        
        if len(features) < n_clusters:
            print(f"Not enough resources for clustering ({len(features)} < {n_clusters})")
            # Assign all to cluster 0
            for resource in resource_list:
                self.resource_clusters[resource] = 0
            self.cluster_profiles[0] = self._compute_cluster_profile(resource_list)
            return
        
        # Perform K-means clustering
        features_array = np.array(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_array)
        
        # Assign clusters
        for resource, cluster_id in zip(resource_list, cluster_labels):
            self.resource_clusters[resource] = int(cluster_id)
        
        # Compute cluster profiles
        for cluster_id in range(n_clusters):
            cluster_resources = [r for r, c in self.resource_clusters.items() if c == cluster_id]
            self.cluster_profiles[cluster_id] = self._compute_cluster_profile(cluster_resources)
        
        print(f"Resources distributed across clusters: {dict(pd.Series(cluster_labels).value_counts().sort_index())}")

    def _compute_cluster_profile(self, resources: List[str]) -> Dict:
        """Compute aggregate profile for a cluster of resources."""
        if not resources:
            return self._get_default_pattern()
        
        patterns = [self.resource_patterns[r] for r in resources if r in self.resource_patterns]
        
        if not patterns:
            return self._get_default_pattern()
        
        # Aggregate statistics
        working_starts = [p['working_start'] for p in patterns]
        working_ends = [p['working_end'] for p in patterns]
        all_working_days = set()
        all_peak_hours = set()
        
        for p in patterns:
            all_working_days.update(p['working_days'])
            all_peak_hours.update(p['peak_hours'])
        
        return {
            'avg_working_start': np.mean(working_starts),
            'avg_working_end': np.mean(working_ends),
            'common_working_days': all_working_days,
            'common_peak_hours': all_peak_hours,
            'avg_activity_intensity': np.mean([p['activity_intensity'] for p in patterns]),
            'num_resources': len(resources),
        }

    def is_available(self, resource_id: str, current_time: datetime, 
                     use_probabilistic: bool = False) -> bool:
        """
        Check if a resource is available at the given time.
        
        Args:
            resource_id: Resource identifier
            current_time: Time to check
            use_probabilistic: If True, uses probability-based availability
        
        Returns:
            True if resource is available, False otherwise
        """
        if resource_id not in self.resources:
            return False
        
        # Check holidays first
        if current_time.date() in self.nl_holidays:
            return False
        
        # Use pattern mining if enabled
        if self.enable_pattern_mining and resource_id in self.resource_patterns:
            pattern = self.resource_patterns[resource_id]
            
            # Check if resource was active during this time period
            if pattern['first_activity'] and pattern['last_activity']:
                # Make current_time timezone-aware if needed for comparison
                first_act = pd.Timestamp(pattern['first_activity'])
                last_act = pd.Timestamp(pattern['last_activity'])
                curr_time = pd.Timestamp(current_time)
                
                # Convert to timezone-naive for comparison if needed
                if first_act.tz is not None and curr_time.tz is None:
                    first_act = first_act.tz_localize(None)
                    last_act = last_act.tz_localize(None)
                elif first_act.tz is None and curr_time.tz is not None:
                    curr_time = curr_time.tz_localize(None)
                
                if curr_time < first_act or curr_time > last_act:
                    return False
            
            hour = current_time.hour
            dow = current_time.weekday()
            
            # Check working days
            if dow not in pattern['working_days']:
                return False
            
            # Check working hours
            if hour < pattern['working_start'] or hour >= pattern['working_end']:
                return False
            
            # Probabilistic mode: use historical probability
            if use_probabilistic and pattern['hour_probabilities']:
                prob = pattern['hour_probabilities'].get(hour, 0)
                # Higher probability in peak hours
                if hour in pattern['peak_hours']:
                    prob = min(1.0, prob * 1.2)
                return np.random.random() < prob
            
            return True
        
        # Fallback to basic model
        return super().is_available(resource_id, current_time)

    def get_resource_info(self, resource_id: str) -> Optional[Dict]:
        """Get detailed information about a resource's pattern."""
        if resource_id not in self.resource_patterns:
            return None
        
        pattern = self.resource_patterns[resource_id].copy()
        
        # Add cluster information
        if resource_id in self.resource_clusters:
            pattern['cluster_id'] = self.resource_clusters[resource_id]
            pattern['cluster_profile'] = self.cluster_profiles.get(
                self.resource_clusters[resource_id], {}
            )
        
        return pattern

    def get_available_resources(self, current_time: datetime, 
                               use_probabilistic: bool = False) -> List[str]:
        """Get list of all resources available at the given time."""
        return [
            resource for resource in self.resources
            if self.is_available(resource, current_time, use_probabilistic)
        ]

    def get_resource_workload(self, resource_id: str) -> Dict:
        """Get workload statistics for a resource."""
        if resource_id not in self.resource_patterns:
            return {}
        
        pattern = self.resource_patterns[resource_id]
        
        return {
            'total_activities': pattern['total_activities'],
            'activity_intensity': pattern['activity_intensity'],
            'active_days': pattern['active_days'],
            'avg_activities_per_day': pattern['activity_intensity'],
            'working_hours_per_day': pattern['working_end'] - pattern['working_start'],
            'working_days_per_week': len(pattern['working_days']),
        }

    def predict_availability_probability(self, resource_id: str, 
                                        current_time: datetime) -> float:
        """
        Predict probability that a resource is available at given time.
        
        Returns value between 0 and 1.
        """
        if resource_id not in self.resource_patterns:
            return 0.5  # Unknown, assume 50%
        
        pattern = self.resource_patterns[resource_id]
        
        # Base checks
        if current_time.date() in self.nl_holidays:
            return 0.0
        
        if pattern['first_activity'] and pattern['last_activity']:
            # Make timezone-aware for comparison
            first_act = pd.Timestamp(pattern['first_activity'])
            last_act = pd.Timestamp(pattern['last_activity'])
            curr_time = pd.Timestamp(current_time)
            
            # Convert to timezone-naive for comparison if needed
            if first_act.tz is not None and curr_time.tz is None:
                first_act = first_act.tz_localize(None)
                last_act = last_act.tz_localize(None)
            elif first_act.tz is None and curr_time.tz is not None:
                curr_time = curr_time.tz_localize(None)
            
            if curr_time < first_act or curr_time > last_act:
                return 0.0
        
        hour = current_time.hour
        dow = current_time.weekday()
        
        # Day of week probability
        dow_prob = pattern['dow_probabilities'].get(dow, 0)
        
        # Hour probability
        hour_prob = pattern['hour_probabilities'].get(hour, 0)
        
        # Combined probability (geometric mean)
        if dow_prob > 0 and hour_prob > 0:
            combined_prob = np.sqrt(dow_prob * hour_prob)
        else:
            combined_prob = 0.0
        
        # Boost for peak hours
        if hour in pattern['peak_hours']:
            combined_prob = min(1.0, combined_prob * 1.3)
        
        return combined_prob
