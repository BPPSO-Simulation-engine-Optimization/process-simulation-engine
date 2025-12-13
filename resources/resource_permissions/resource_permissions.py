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

    Supports two profiling modes:
    - 'full_recall': Groups inherit ALL activities from ANY member (recommended for simulation)
    - 'overall_score': Original OrdinoR profiling with threshold (precision-optimized)

    FullRecall provides principled generalization: if resources X and Y are clustered
    together based on similar behavioral profiles, and Y performed activity A but X
    never did, X is still permitted to do A based on role inference.
    """

    def __init__(self, log_path: str = None, df: pd.DataFrame = None,
                 filter_completed: bool = True, exclude_resources: List[str] = None,
                 profiling_mode: str = 'full_recall'):
        """
        Initialize OrdinoR permissions.

        Args:
            log_path: Path to event log XES.
            df: Optional Pandas DataFrame (if log_path not provided).
            filter_completed: Whether to keep only completed events.
            exclude_resources: list of resources to exclude.
            profiling_mode: 'full_recall' (default, for simulation) or 'overall_score' (strict).
        """
        if profiling_mode not in ('full_recall', 'overall_score'):
            raise ValueError(f"profiling_mode must be 'full_recall' or 'overall_score', got '{profiling_mode}'")

        self.profiling_mode = profiling_mode
        self.model = None  # OrdinoR OrganizationalModel (for overall_score mode)
        self.log_path = log_path
        self.df = None

        self.filter_completed = filter_completed
        self.filter_completed = filter_completed
        self.exclude_resources = exclude_resources or []
        
        # Activities where system user (User_1) must be included
        self.SYSTEM_USER_ACTIVITIES = {
            "A_Cancelled", "A_Concept", "A_Create Application", "A_Submitted",
            "O_Cancelled",
            "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files", 
            "W_Complete application", "W_Handle leads", "W_Validate application"
        }

        # Activities with ONLY incomplete events (hardcoded to avoid parsing raw log)
        self.INCOMPLETE_ACTIVITY_MD = {
            "W_Shortened completion ": {
                "User_43", "User_11", "User_18", "User_42", "User_2", "User_28", 
                "User_49", "User_5", "User_53", "User_106", "User_75", "User_124", 
                "User_30", "User_77", "User_79"
            },
            "W_Personal Loan collection": {
                "User_50", "User_138", "User_5", "User_119", "User_24", "User_69", "User_99"
            }
        }

        # FullRecall mode structures
        self._groups: List[Set[str]] = []  # List of resource groups
        self._activity_to_groups: Dict[str, Set[int]] = {}  # activity -> set of group indices
        self._resource_to_group: Dict[str, int] = {}  # resource -> group index

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
            w1: Profiling weight for Relative Stake (default 0.5, only for overall_score mode).
            p: Profiling threshold (default 0.5, only for overall_score mode).
            case_type_column: Column name for case type (default 'case:LoanGoal').
        """
        import time
        import threading
        import sys

        n_events = len(self.df)
        n_resources = self.df['org:resource'].nunique()
        n_activities = self.df['concept:name'].nunique()

        mode_label = "FullRecall" if self.profiling_mode == 'full_recall' else "OverallScore"

        print(f"\n{'='*60}")
        print(f"OrdinoR Discovery Pipeline ({mode_label})")
        print(f"{'='*60}")
        print(f"Dataset: {n_events:,} events, {n_resources} resources, {n_activities} activities")
        print(f"Config: {n_resource_clusters} resource clusters, mode={self.profiling_mode}")
        print(f"{'='*60}\n")

        logger.info(f"Starting OrdinoR discovery pipeline (mode={self.profiling_mode})...")

        # Step 1: Prepare DataFrame
        print("[1/3] Preparing data...")
        start = time.time()

        rl_df = self.df.copy()
        rl_df['activity_type'] = rl_df['concept:name']

        if 'time:timestamp' in rl_df.columns:
            rl_df['time:timestamp'] = pd.to_datetime(rl_df['time:timestamp'], utc=True)
            rl_df['time_type'] = rl_df['time:timestamp'].dt.day_name()
        else:
            logger.warning("No timestamp found, using default time type")
            rl_df['time_type'] = "AnyTime"

        # Case type (only needed for overall_score context matching)
        if case_type_column in self.df.columns:
            rl_df['case_type'] = rl_df[case_type_column].astype(str)
        else:
            rl_df['case_type'] = "Default"

        print(f"      ✓ Completed in {time.time() - start:.1f}s\n")

        # Step 2: Resource Profiling & Clustering (shared between modes)
        print("[2/3] Building Resource Profiles & Clustering (AHC)...")
        start = time.time()
        profiles = resource_features.direct_count(rl_df)

        groups = group_discovery.ahc(
            profiles,
            n_groups=n_resource_clusters,
            method='ward',
            metric='euclidean'
        )
        print(f"      ✓ Discovered {len(groups)} groups in {time.time() - start:.1f}s\n")

        # Step 3: Profiling (mode-dependent)
        if self.profiling_mode == 'full_recall':
            print("[3/3] Building FullRecall capability map...")
            start = time.time()
            self._build_full_recall_model(groups, rl_df)
            print(f"      ✓ Completed in {time.time() - start:.1f}s\n")
        else:
            # overall_score mode - computationally intensive
            print("[3/3] Profiling Groups (OverallScore)...")
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

    def _build_full_recall_model(self, groups: List, rl_df: pd.DataFrame):
        """
        Build FullRecall capability model from discovered groups.

        For each group, computes the union of all activities performed by any member.
        This enables role-based generalization: if X and Y are in the same group and
        Y did activity A, then X is also permitted to do A.
        """
        # Extract groups as sets of resources
        self._groups = []
        self._resource_to_group = {}

        for idx, group in enumerate(groups):
            # Group structure from OrdinoR: frozenset of resources
            if isinstance(group, (set, frozenset)):
                resources = set(group)
            else:
                # Might be a tuple or other structure
                resources = set(group) if hasattr(group, '__iter__') else {group}

            self._groups.append(resources)
            for resource in resources:
                self._resource_to_group[resource] = idx

        # Build activity -> groups mapping based on historical performance
        # NOTE: We use rl_df (which might have been preprocessed/filtered) or even better,
        # we try to check if we can inspect the raw data to catch activities with no complete events.
        # But rl_df passed here usually comes from _preprocess which filters completed if configured.
        
        # KEY UPDATE: To handle activities with no 'complete' events (like W_Personal Loan collection),
        # we should relax the filter for this specific mapping step if we want TRUE Full Recall.
        # However, rl_df is passed in. We'll trust rl_df for now, but if the user wants to include these,
        # they must have been included in the DF passed to discover_model.

        # Wait, self.df is used in discover_model and then copied to rl_df. 
        # self.df is filtered in __init__.
        # So we need to access the UNFILTERED data if we want to include these activities.
        # But we don't have it stored.
        
        # Strategy: Rely on the fact that if we want to capture these, we should probably
        # rely on the clustering grouping resources who likely performed OTHER tasks too.
        # But if an activity ONLY has incomplete events, it might not even appear in rl_df.
        
        # Let's rebuild the map using the current rl_df. 
        # If the user wants to include incomplete events, we should change how rl_df is prepared in discover_model
        # OR how self.df is prepared in __init__.
        
        # Actually, looking at discover_model:
        # rl_df = self.df.copy()
        # self.df comes from __init__ -> _load_and_preprocess -> filter_completed=True by default.
        
        # To fix this without breaking the clustering (which might rely on complete events),
        # we should ideally build the capability map using a LESS filtered view.
        # But we don't have access to the raw log here easily unless we reload or store it.
        
        self._activity_to_groups = {}
        
        # Use available dataframe
        activity_resource_map = rl_df.groupby('concept:name')['org:resource'].apply(set).to_dict()
        
        for activity, performers in activity_resource_map.items():
            eligible_groups = set()
            for resource in performers:
                if resource in self._resource_to_group:
                    eligible_groups.add(self._resource_to_group[resource])
            self._activity_to_groups[activity] = eligible_groups

        # Log statistics
        n_activities = len(self._activity_to_groups)
        avg_groups_per_activity = sum(len(g) for g in self._activity_to_groups.values()) / n_activities if n_activities else 0
        logger.info(f"FullRecall model: {len(self._groups)} groups, {n_activities} activities, "
                   f"avg {avg_groups_per_activity:.1f} groups per activity")

    def get_eligible_resources(self, activity: str, timestamp: pd.Timestamp = None, case_type: str = None) -> List[str]:
        """
        Get resources eligible for a given activity, respecting context if provided.

        Args:
            activity: The activity name.
            timestamp: Optional timestamp (used for context in overall_score mode).
            case_type: Optional Case Type (used for context in overall_score mode).

        Returns:
            List of resource IDs.
        """
        # FullRecall mode: simple group-based lookup
        if self.profiling_mode == 'full_recall':
            resources = self._get_eligible_full_recall(activity)
        # Apply exceptions (for activities with no complete events)
        if activity in self.INCOMPLETE_ACTIVITY_MD:
            return list(set(resources).union(self.INCOMPLETE_ACTIVITY_MD[activity]))
            
        return self._apply_system_user_logic(activity, resources)

    def _get_eligible_full_recall(self, activity: str) -> List[str]:
        """Get eligible resources using FullRecall mode (group-based)."""
        if not self._groups:
            logger.warning("FullRecall model not built yet!")
            return []

        eligible_groups = self._activity_to_groups.get(activity, set())
        if not eligible_groups:
            return []

        # Return all resources from all eligible groups
        eligible_resources = set()
        for group_idx in eligible_groups:
            eligible_resources.update(self._groups[group_idx])

        return list(eligible_resources)

    def _apply_system_user_logic(self, activity: str, resources: List[str]) -> List[str]:
        """Ensure User_1 is included if required for this activity."""
        if activity in self.SYSTEM_USER_ACTIVITIES:
            if "User_1" not in resources:
                # Create a new list to avoid modifying the input if it's a reference
                resources = list(resources) + ["User_1"]
        return resources

    def _get_eligible_overall_score(self, activity: str, timestamp: pd.Timestamp = None, case_type: str = None) -> List[str]:
        """Get eligible resources using OverallScore mode (context-aware)."""
        if self.model is None:
            logger.warning("OverallScore model not discovered yet!")
            return []

        # OrdinoR Execution Context is a tuple: (case_type, activity_type, time_type)

        # 1. Determine constraints
        target_time_type = None
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    ts = pd.to_datetime(timestamp)
                else:
                    ts = timestamp
                target_time_type = ts.day_name()
            except Exception as e:
                logger.warning(f"Failed to derive time type from {timestamp}: {e}")

        target_case_type = str(case_type) if case_type else None

        all_contexts = self.model.find_all_execution_contexts()

        matching_contexts = []
        for ctx in all_contexts:
            if isinstance(ctx, tuple) and len(ctx) >= 3:
                c_case, c_act, c_time = ctx[0], ctx[1], ctx[2]

                if c_act != activity:
                    continue
                if target_time_type and c_time != "AnyTime" and c_time != target_time_type:
                    continue
                if target_case_type and c_case != target_case_type:
                    continue

                matching_contexts.append(ctx)
            elif isinstance(ctx, str) and activity in ctx:
                matching_contexts.append(ctx)

        eligible_resources = set()
        for ctx in matching_contexts:
            group_ids = self.model.find_candidate_groups(ctx)
            for gid in group_ids:
                if isinstance(gid, (set, frozenset, list, tuple)):
                    eligible_resources.update(gid)
                else:
                    try:
                        members = self.model.find_group_members(gid)
                        eligible_resources.update(members)
                    except KeyError:
                        logger.warning(f"Group ID {gid} not found in model members.")

        return list(eligible_resources)

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
        if self.profiling_mode == 'full_recall':
            if not self._groups:
                logger.warning("No FullRecall model to save.")
                return
            model_data = {
                'profiling_mode': 'full_recall',
                'groups': self._groups,
                'activity_to_groups': self._activity_to_groups,
                'resource_to_group': self._resource_to_group
            }
        else:
            if self.model is None:
                logger.warning("No OverallScore model to save.")
                return
            model_data = {
                'profiling_mode': 'overall_score',
                'model': self.model
            }

        try:
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {path} (mode={self.profiling_mode})")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, path: str):
        """Load a discovered model from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model cache not found: {path}")

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Handle both old format (direct OrdinoR model) and new format (dict with mode)
            if isinstance(data, dict) and 'profiling_mode' in data:
                saved_mode = data['profiling_mode']

                if saved_mode == 'full_recall':
                    self._groups = data['groups']
                    self._activity_to_groups = data['activity_to_groups']
                    self._resource_to_group = data['resource_to_group']
                    self.profiling_mode = 'full_recall'
                    logger.info(f"FullRecall model loaded from {path}")
                else:
                    self.model = data['model']
                    self.profiling_mode = 'overall_score'
                    logger.info(f"OverallScore model loaded from {path}")
            else:
                # Legacy format: direct OrdinoR OrganizationalModel
                self.model = data
                self.profiling_mode = 'overall_score'
                logger.info(f"Legacy OverallScore model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

