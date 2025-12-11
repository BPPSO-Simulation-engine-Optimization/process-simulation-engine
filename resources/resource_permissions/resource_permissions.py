from typing import List, Optional, Dict, Set, Any
import pandas as pd
import pm4py
import os
import logging
import pickle
from sklearn.cluster import AgglomerativeClustering  # Still used by OrdinoR internally via group_discovery.ahc
from ordinor.org_model_miner import resource_features, group_discovery, group_profiling
from ordinor.org_model_miner.models import base

from resources.resource_permissions.data_preparation import ResourceDataPreparation

logger = logging.getLogger(__name__)


class BasicResourcePermissions:
    """
    Manages resource permissions based on historical event log data.
    Maps activities to the set of resources that have performed them.
    
    By default, preprocesses data to include only completed activities
    following OrdinoR (2022) methodology.
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        filter_completed: bool = True,
        exclude_resources: Optional[List[str]] = None
    ):
        """
        Initialize the ResourcePermissions model.

        Args:
            log_path: Path to the XES event log file.
            df: Existing pandas DataFrame containing the event log.
            filter_completed: If True, only include 'complete' lifecycle events.
            exclude_resources: Resources to exclude (e.g., ['User_1'] for system users).
        
        Raises:
            ValueError: If neither log_path nor df is provided.
        """
        self.activity_resource_map: Dict[str, Set[str]] = {}
        self.filter_completed = filter_completed
        self.exclude_resources = exclude_resources or []

        if df is not None:
            self.df = df
            # Apply preprocessing if filtering is enabled and df hasn't been preprocessed
            if filter_completed or exclude_resources:
                self.df = self._preprocess(df)
        elif log_path is not None:
            self.df = self._load_and_preprocess(log_path)
        else:
            raise ValueError("Either 'log_path' or 'df' must be provided.")

        self._build_mapping()

    def _load_and_preprocess(self, path: str) -> pd.DataFrame:
        """Load and preprocess event log."""
        prep = ResourceDataPreparation(log_path=path)
        return prep.prepare(
            filter_completed=self.filter_completed,
            exclude_resources=self.exclude_resources if self.exclude_resources else None,
            drop_na=True
        )
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to existing DataFrame."""
        prep = ResourceDataPreparation(df=df)
        return prep.prepare(
            filter_completed=self.filter_completed,
            exclude_resources=self.exclude_resources if self.exclude_resources else None,
            drop_na=True
        )

    def _build_mapping(self):
        """
        Parses the event log to build a map of Activity -> Set[Resource].
        Assumes standard column names: 'concept:name' for activity, 'org:resource' for resource.
        """
        if self.df is None or self.df.empty:
            return

        activity_col = "concept:name"
        resource_col = "org:resource"

        if activity_col not in self.df.columns or resource_col not in self.df.columns:
             raise ValueError(f"DataFrame missing required columns: {activity_col}, {resource_col}")
        grouped = self.df.dropna(subset=[activity_col, resource_col])
        
        mapping_series = grouped.groupby(activity_col)[resource_col].apply(set)
        
        self.activity_resource_map = mapping_series.to_dict()

    def get_eligible_resources(self, activity_name: str, timestamp: pd.Timestamp = None, case_type: str = None) -> List[str]:
        """
        Returns a list of all resources that have historically performed the given activity.
        
        Args:
            activity_name: The name of the activity.
            timestamp: Ignored (Basic model is not context-aware).
            case_type: Ignored (Basic model is not context-aware).

        Returns:
            List[str]: A list of resource IDs. Returns empty list if activity is unknown.
        """
        resources = self.activity_resource_map.get(activity_name, set())
        return list(resources)


class OrdinoRResourcePermissions:
    """
    Advanced resource permission system using OrdinoR library (Yang et al. 2022).
    
    Implements the best performing configuration for BPIC2017:
    Resource permissions using OrdinoR library with Trace Clustering.
    """
    
    def __init__(self, log_path: str = None, df: pd.DataFrame = None, filter_completed: bool = True, exclude_resources: List[str] = None):
        """
        Initialize OrdinoR permissions.
        
        Args:
            log_path: Path to event log XES.
            df: Optional Pandas DataFrame (if log_path not provided).
            filter_completed: Whether to keep only completed events.
            exclude_resources: list of resources to exclude.
        """
        self.model = None
        self.log_path = log_path
        self.df = None
        
        self.filter_completed = filter_completed
        self.exclude_resources = exclude_resources or []
        
        # Mapping from case_id to case_type (cluster ID)
        self._case_to_cluster = {}
        
        # Load and preprocess
        if df is not None:
             self.df = self._preprocess(df)
        elif log_path is not None:
             prep = ResourceDataPreparation(log_path=log_path)
             self.df = prep.prepare(
                 filter_completed=filter_completed,
                 exclude_resources=self.exclude_resources,
                 drop_na=True
             )
        else:
             raise ValueError("Either log_path or df must be provided to OrdinoRResourcePermissions")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing manually."""
        df_out = df.copy()
        if self.filter_completed and 'lifecycle:transition' in df_out.columns:
            df_out = df_out[df_out['lifecycle:transition'].str.lower() == 'complete']
            
        if self.exclude_resources:
            df_out = df_out[~df_out['org:resource'].isin(self.exclude_resources)]
            
        # Ensure critical columns exist and are not null
        cols = ['org:resource', 'concept:name'] 
        if all(c in df_out.columns for c in cols):
             df_out = df_out.dropna(subset=cols)
             
        return df_out

    def discover_model(self, n_resource_clusters: int = 10, w1: float = 0.5, p: float = 0.5, 
                        case_type_column: str = 'case:LoanGoal'):
        """
        Discover organizational model using OrdinoR pipeline.
        
        Uses case attributes (e.g., LoanGoal) directly for Case Type dimension
        instead of trace clustering. This approach is suitable for simulation because
        case attributes are known at case instantiation (before any activities occur).
        
        Args:
            n_resource_clusters: Number of resource groups to discover (AHC).
            w1: Profiling weight for Relative Stake (default 0.5).
            p: Profiling threshold (default 0.5).
            case_type_column: Column name for case type (default 'case:LoanGoal').
        """
        import time
        import threading
        import sys
        
        n_events = len(self.df)
        n_resources = self.df['org:resource'].nunique()
        n_activities = self.df['concept:name'].nunique()
        
        print(f"\n{'='*60}")
        print(f"OrdinoR Discovery Pipeline")
        print(f"{'='*60}")
        print(f"Dataset: {n_events:,} events, {n_resources} resources, {n_activities} activities")
        print(f"Config: Case Type from '{case_type_column}', {n_resource_clusters} resource clusters")
        print(f"{'='*60}\n")
        
        logger.info("Starting OrdinoR discovery pipeline...")
        
        # Validate case type column
        if case_type_column not in self.df.columns:
            available = [c for c in self.df.columns if c.startswith('case:')]
            raise ValueError(f"Column '{case_type_column}' not found. Available case columns: {available}")
        
        # Step 1: Extract Case Types from attributes
        print(f"[1/4] Extracting Case Types from '{case_type_column}'...")
        start = time.time()
        
        rl_df = self.df.copy()
        rl_df['case_type'] = rl_df[case_type_column].astype(str)
        
        n_case_types = rl_df['case_type'].nunique()
        case_type_values = rl_df['case_type'].unique().tolist()
        print(f"      ✓ Found {n_case_types} case types: {case_type_values[:5]}{'...' if n_case_types > 5 else ''}")
        print(f"      ✓ Completed in {time.time() - start:.1f}s\n")
        
        # Step 2: Construct Execution Contexts (CT+AT+TT)
        print("[2/4] Constructing Execution Contexts (CT+AT+TT)...")
        start = time.time()
        
        rl_df['activity_type'] = rl_df['concept:name']
        
        if 'time:timestamp' in rl_df.columns:
            rl_df['time:timestamp'] = pd.to_datetime(rl_df['time:timestamp'], utc=True)
            rl_df['time_type'] = rl_df['time:timestamp'].dt.day_name()
        else:
            logger.warning("No timestamp found, using default time type")
            rl_df['time_type'] = "AnyTime"
        
        n_contexts = rl_df.groupby(['case_type', 'activity_type', 'time_type']).ngroups
        print(f"      ✓ Created {n_contexts} unique execution contexts in {time.time() - start:.1f}s\n")
        
        # Step 3: Resource Profiling & Clustering
        print("[3/4] Building Resource Profiles & Clustering (AHC)...")
        start = time.time()
        profiles = resource_features.direct_count(rl_df)
        
        groups = group_discovery.ahc(
            profiles, 
            n_groups=n_resource_clusters, 
            method='ward', 
            metric='euclidean'
        )
        print(f"      ✓ Discovered {len(groups)} groups in {time.time() - start:.1f}s\n")
        
        # Step 4: Group Profiling (computationally intensive)
        print("[4/4] Profiling Groups (OverallScore)...")
        est_minutes = n_events / 8000  # Empirical: ~8000 events/minute
        print(f"      ⚠ Estimated time: ~{est_minutes:.0f} minutes for {n_events:,} events")
        start = time.time()
        
        # Progress monitoring for long-running operations
        stop_monitor = threading.Event()
        
        def monitor_progress():
            elapsed = 0
            while not stop_monitor.is_set():
                time.sleep(5)
                elapsed += 5
                mins = elapsed / 60
                progress = min(99, (mins / est_minutes) * 100)
                eta = max(0, est_minutes - mins)
                sys.stdout.write(f"\r      ⏱ Elapsed: {mins:.1f} min | Progress: ~{progress:.0f}% | ETA: ~{eta:.1f} min")
                sys.stdout.flush()
        
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        try:
            self.model = group_profiling.overall_score(
                groups, rl_df, w1=w1, p=p, auto_search=True
            )
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=1)
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()
        
        elapsed = time.time() - start
        print(f"      ✓ Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
        
        print(f"{'='*60}")
        print(f"Discovery Complete! Model has {len(groups)} resource groups.")
        print(f"{'='*60}\n")
        logger.info(f"Discovery complete. Model has {len(groups)} groups.")

    def get_eligible_resources(self, activity: str, timestamp: pd.Timestamp = None, case_type: str = None) -> List[str]:
        """
        Get resources eligible for a given activity, respecting context if provided.
        
        Args:
            activity: The activity name.
            timestamp: Optional timestamp to enforce Time Type context (derives day of week).
            case_type: Optional Case Type (e.g., "Home improvement", "Car").
            
        Returns:
            List of resource IDs.
        """
        if self.model is None:
            logger.warning("Model not discovered yet!")
            return []
        
        # OrdinoR Execution Context is a tuple: (case_type, activity_type, time_type)
        
        # 1. Determine constraints
        target_time_type = None
        if timestamp:
            # Match the time type definition in discovery (Day Name)
            try:
                if isinstance(timestamp, str):
                    ts = pd.to_datetime(timestamp)
                else:
                    ts = timestamp
                target_time_type = ts.day_name()
            except Exception as e:
                logger.warning(f"Failed to derive time type from {timestamp}: {e}")

        # Case Type: Use explicit case_type if provided
        target_case_type = None
        if case_type:
            target_case_type = str(case_type)
        
        all_contexts = self.model.find_all_execution_contexts()
        
        matching_contexts = []
        for ctx in all_contexts:
            # OrdinoR uses tuples for contexts: (case_type, activity_type, time_type)
            # activity_type is at index 1
            if isinstance(ctx, tuple) and len(ctx) >= 3:
                c_case, c_act, c_time = ctx[0], ctx[1], ctx[2]
                
                # Check Activity
                if c_act != activity:
                    continue
                    
                # Check Time Context (if provided)
                if target_time_type and c_time != "AnyTime" and c_time != target_time_type:
                    continue
                    
                # Check Case Context (if provided and known)
                if target_case_type and c_case != target_case_type:
                    continue
                    
                matching_contexts.append(ctx)
            elif isinstance(ctx, str):
                # Fallback
                if activity in ctx:
                    matching_contexts.append(ctx)
        
        eligible_resources = set()
        for ctx in matching_contexts:
            group_ids = self.model.find_candidate_groups(ctx)
            for gid in group_ids:
                # OrdinoR might return the group (set of resources) or an ID
                if isinstance(gid, (set, frozenset, list, tuple)):
                    eligible_resources.update(gid)
                else:
                    try:
                        members = self.model.find_group_members(gid)
                        eligible_resources.update(members)
                    except KeyError:
                        # Fallback or strict error? 
                        # If gid is not in _mem, maybe it's invalid ID?
                        logger.warning(f"Group ID {gid} not found in model members.")
        
        # Initialize cache if needed (e.g. if accessed before discovery, though unlikely)
        if not hasattr(self, '_eligible_cache'):
            self._eligible_cache = {}
            
        result = list(eligible_resources)
        self._eligible_cache[activity] = result
        return result

    def get_coverage_stats(self) -> Dict[str, Any]:
        """
        Compute coverage statistics.
        Returns dict with covered_activities, total_activities, coverage_ratio.
        """
        if self.df is None:
             return {"covered_activities": 0, "total_activities": 0, "coverage_ratio": 0.0}
             
        # Use simple unique check
        activities = self.df['concept:name'].unique()
        total = len(activities)
        covered = 0
        
        for act in activities:
            if self.get_eligible_resources(act):
                covered += 1
                
        return {
            "covered_activities": covered,
            "total_activities": total,
            "coverage_ratio": covered / total if total > 0 else 0.0
        }

    def save_model(self, path: str):
        """Save the discovered model to a file."""
        if self.model is None:
            logger.warning("No model to save.")
            return
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, path: str):
        """Load a discovered model from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model cache not found: {path}")
            
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {path}")
            # Reset cache
            self._eligible_cache = {}
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

