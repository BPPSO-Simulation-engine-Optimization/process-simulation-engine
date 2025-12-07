from typing import List, Optional, Dict, Set, Any
import pandas as pd
import pm4py
import os
import logging
from sklearn.cluster import KMeans
from ordinor.org_model_miner import resource_features, group_discovery, group_profiling
from ordinor.org_model_miner.models import base

from resources.resource_permissions.resource_features import ResourceActivityMatrix
from resources.resource_permissions.resource_clustering import ResourceClusterer
from resources.resource_permissions.group_profiling import GroupProfiler
from resources.resource_permissions.organizational_model import OrganizationalModel
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
            log_path: Path to the XES/CSV event log file.
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

    def get_eligible_resources(self, activity_name: str) -> List[str]:
        """
        Returns a list of all resources that have historically performed the given activity.

        Args:
            activity_name: The name of the activity.

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
            log_path: Path to event log XES/CS.
            df: Optional Pandas DataFrame (if log_path not provided).
            filter_completed: Whether to keep only completed events.
            exclude_resources: list of resources to exclude.
        """
        self.model = None
        self.log_path = log_path
        self.df = None
        
        self.filter_completed = filter_completed
        self.exclude_resources = exclude_resources or []
        
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

    def discover_model(self, n_trace_clusters: int = 5, n_resource_clusters: int = 10, w1: float = 0.5, p: float = 0.5):
        """
        Discover organizational model using OrdinoR pipeline.
        
        Args:
            n_trace_clusters: Number of case types to discover (K-Means).
            n_resource_clusters: Number of resource groups to discover (AHC).
            w1: Profiling weight for Relative Stake (default 0.5).
            p: Profiling threshold (default 0.5).
        """
        logger.info("Starting OrdinoR discovery pipeline...")
        
        # 1. Trace Clustering
        logger.info(f"Running Trace Clustering (k={n_trace_clusters})...")
        df_clustered = self._apply_trace_clustering(self.df, n_clusters=n_trace_clusters)
        
        # 2. Construct Execution Contexts (CT + AT + TT)
        logger.info("Constructing Execution Contexts (CT+AT+TT)...")
        
        # OrdinoR direct_count expects:
        # const.CASE_TYPE -> 'case_type'
        # const.ACTIVITY_TYPE -> 'activity_type'
        # const.TIME_TYPE -> 'time_type'
        # const.RESOURCE -> 'org:resource'
        
        rl_df = df_clustered.copy()
        
        # Map CT (Case Type)
        rl_df['case_type'] = rl_df['case:cluster'].astype(str)
        
        # Map AT (Activity Type)
        rl_df['activity_type'] = rl_df['concept:name']
        
        # Map TT (Time Type)
        if 'time:timestamp' in rl_df.columns:
            rl_df['time:timestamp'] = pd.to_datetime(rl_df['time:timestamp'], utc=True)
            rl_df['time_type'] = rl_df['time:timestamp'].dt.day_name()
        else:
             logger.warning("No timestamp found, using default time type")
             rl_df['time_type'] = "AnyTime"
        
        # 3. Resource Profiling
        logger.info("Building Resource Profiles...")
        profiles = resource_features.direct_count(rl_df)
        
        # 4. Resource Clustering (AHC)
        logger.info(f"Clustering Resources (AHC, n={n_resource_clusters})...")
        groups = group_discovery.ahc(
            profiles, 
            n_groups=n_resource_clusters, 
            method='ward', 
            metric='euclidean'
        )
        
        # 5. Group Profiling (OverallScore)
        logger.info(f"Profiling Groups (OverallScore, w1={w1}, p={p})...")
        self.model = group_profiling.overall_score(
            groups, 
            rl_df, 
            w1=w1, 
            p=p,
            auto_search=False
        )
        
        # Clear cache whenever model changes
        self._eligible_cache = {}
        
        
        logger.info(f"Discovery complete. Model has {len(groups)} groups.")

    def _apply_trace_clustering(self, df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
        """
        Apply K-Means clustering on traces based on activity occurrence (Bag-of-Activities).
        Adds 'case:cluster' column to the DataFrame.
        """
        # Create Bag-of-Activities matrix
        # Rows: Case IDs, Columns: Activities, Values: Count
        case_id_col = "case:concept:name"
        activity_col = "concept:name"
        
        if case_id_col not in df.columns:
            # Fallback if case id is different or index
            logger.warning("Standard case ID column not found. Using numeric index as cases?")
            # This would probably be wrong for process mining. Assuming standard XES.
            raise ValueError(f"Column {case_id_col} required for trace clustering")

        # Pivot table (fill_value=0)
        matrix = pd.pivot_table(
            df, 
            index=case_id_col, 
            columns=activity_col, 
            aggfunc='size', 
            fill_value=0
        )
        
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(matrix)
        
        # Map clusters back to DataFrame
        cluster_map = dict(zip(matrix.index, clusters))
        
        df_out = df.copy()
        df_out['case:cluster'] = df_out[case_id_col].map(cluster_map)
        
        return df_out

    def get_eligible_resources(self, activity: str) -> List[str]:
        """
        Get resources eligible for a given activity.
        
        Since OrdinoR model stores 'Contexts' (CaseType, Activity, TimeType),
        we return resources that have capability for ANY context involving this activity.
        """
        if self.model is None:
            logger.warning("Model not discovered yet!")
            return []
            
        # Check cache
        if hasattr(self, '_eligible_cache') and activity in self._eligible_cache:
            return self._eligible_cache[activity]
        
        # OrdinoR Execution Context is a tuple: (case_type, activity_type, time_type)
        # We want to find all contexts where activity matches
        
        all_contexts = self.model.find_all_execution_contexts()
        
        matching_contexts = []
        for ctx in all_contexts:
            # OrdinoR uses tuples for contexts: (case_type, activity_type, time_type)
            # activity_type is at index 1
            if isinstance(ctx, tuple):
                if len(ctx) >= 2 and ctx[1] == activity:
                    matching_contexts.append(ctx)
            elif isinstance(ctx, str):
                # Fallback if OrdinoR ever stringifies keys (unlikely in this ver)
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

