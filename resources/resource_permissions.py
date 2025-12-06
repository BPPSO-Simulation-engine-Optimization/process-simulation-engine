from typing import List, Optional, Dict, Set
import pandas as pd
import pm4py
import os
import logging

from resources.resource_features import ResourceActivityMatrix
from resources.resource_clustering import ResourceClusterer
from resources.group_profiling import GroupProfiler
from resources.organizational_model import OrganizationalModel

logger = logging.getLogger(__name__)


class BasicResourcePermissions:
    """
    Manages resource permissions based on historical event log data.
    Maps activities to the set of resources that have performed them.
    """

    def __init__(self, log_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        Initialize the ResourcePermissions model.

        Args:
            log_path: Path to the XES/CSV event log file.
            df: Existing pandas DataFrame containing the event log.
        
        Raises:
            ValueError: If neither log_path nor df is provided.
        """
        self.activity_resource_map: Dict[str, Set[str]] = {}

        if df is not None:
            self.df = df
        elif log_path is not None:
            self.df = self._load_log(log_path)
        else:
            raise ValueError("Either 'log_path' or 'df' must be provided.")

        self._build_mapping()

    def _load_log(self, path: str) -> pd.DataFrame:
        """
        Loads the event log from the given path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Event log not found at: {path}")
        try:
            log = pm4py.read_xes(path, return_legacy_log_object=True)
            df = pm4py.convert_to_dataframe(log)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load event log: {e}")

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


class AdvancedResourcePermissions:
    """
    Advanced resource permission system using organizational model mining.
    
    Discovers resource groups (roles) via clustering and assigns permissions
    at the group level based on activity profiles.
    Source - (2022) OrdinoR: A Framework for Discovering, Evaluating, and Analyzing Organizational Models using Event Logs
    """
    
    def __init__(
        self,
        log_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        model: Optional[OrganizationalModel] = None
    ):
        """
        Initialize the advanced permissions system.
        
        Args:
            log_path: Path to the XES event log file.
            df: Existing pandas DataFrame containing the event log.
            model: Pre-built OrganizationalModel (if provided, skips discovery).
        
        Raises:
            ValueError: If neither log_path, df, nor model is provided.
        """
        self.df: Optional[pd.DataFrame] = None
        self.model: Optional[OrganizationalModel] = model
        self._all_activities: Set[str] = set()
        
        if model is not None:
            # Use provided model directly
            logger.info("Using pre-built organizational model")
        elif df is not None:
            self.df = df
            self._extract_all_activities()
        elif log_path is not None:
            self.df = self._load_log(log_path)
            self._extract_all_activities()
        else:
            raise ValueError("Either 'log_path', 'df', or 'model' must be provided.")
    
    def _load_log(self, path: str) -> pd.DataFrame:
        """Load event log from XES file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Event log not found at: {path}")
        try:
            log = pm4py.read_xes(path, return_legacy_log_object=True)
            df = pm4py.convert_to_dataframe(log)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load event log: {e}")
    
    def _extract_all_activities(self) -> None:
        """Extract all unique activities from the log."""
        if self.df is not None and "concept:name" in self.df.columns:
            self._all_activities = set(self.df["concept:name"].dropna().unique())
    
    def discover_model(
        self,
        n_clusters: int = 5,
        min_frequency: int = 5,
        min_coverage: float = 0.3
    ) -> OrganizationalModel:
        """
        Run the full organizational model discovery pipeline.
        
        Args:
            n_clusters: Number of resource groups to discover.
            min_frequency: Minimum activity occurrences for group capability.
            min_coverage: Minimum fraction of group members for capability.
        
        Returns:
            Discovered OrganizationalModel.
        
        Raises:
            ValueError: If no DataFrame is available for discovery.
        """
        if self.df is None:
            raise ValueError("No DataFrame available. Initialize with log_path or df.")
        
        logger.info(f"Starting model discovery: n_clusters={n_clusters}, "
                   f"min_frequency={min_frequency}, min_coverage={min_coverage}")
        
        # Step 1: Build feature matrix
        matrix_builder = ResourceActivityMatrix(self.df)
        matrix = matrix_builder.build_matrix()
        resource_ids = matrix_builder.get_resource_ids()
        
        # Step 2: Cluster resources
        clusterer = ResourceClusterer(n_clusters=n_clusters)
        resource_to_group = clusterer.cluster(matrix.values, resource_ids)
        
        # Step 3: Profile groups
        profiler = GroupProfiler(min_frequency=min_frequency, min_coverage=min_coverage)
        group_capabilities = profiler.profile_groups(matrix, resource_to_group)
        
        # Step 4: Build organizational model
        resource_groups = clusterer.get_group_members(resource_to_group)
        
        self.model = OrganizationalModel(
            resource_groups={k: set(v) for k, v in resource_groups.items()},
            group_capabilities=group_capabilities,
            resource_to_group=resource_to_group
        )
        
        # Log summary
        logger.info(self.model.summary())
        
        # Log coverage stats
        if self._all_activities:
            stats = self.model.get_coverage_stats(self._all_activities)
            logger.info(f"Activity coverage: {stats['coverage_ratio']:.1%} "
                       f"({stats['covered_activities']}/{stats['total_activities']})")
        
        return self.model
    
    def get_eligible_resources(self, activity_name: str) -> List[str]:
        """
        Get all resources eligible to perform the given activity.
        
        Uses group-based lookup: returns all members of groups that have
        this activity as a capability.
        
        Args:
            activity_name: The name of the activity.
        
        Returns:
            List of resource IDs. Empty list if no groups handle this activity.
        
        Raises:
            ValueError: If no model has been discovered or loaded.
        """
        if self.model is None:
            raise ValueError("No organizational model available. Call discover_model() first.")
        
        groups = self.model.get_groups_for_activity(activity_name)
        members = self.model.get_members_of_groups(groups)
        return list(members)
    
    def save_model(self, path: str) -> None:
        """
        Save the discovered model to a JSON file.
        
        Args:
            path: File path to save to.
        """
        if self.model is None:
            raise ValueError("No model to save. Call discover_model() first.")
        self.model.save(path)
    
    def load_model(self, path: str) -> None:
        """
        Load a model from a JSON file.
        
        Args:
            path: File path to load from.
        """
        self.model = OrganizationalModel.load(path)
    
    def get_coverage_stats(self) -> Dict:
        """
        Get coverage statistics for the current model.
        
        Returns:
            Dict with coverage metrics.
        """
        if self.model is None:
            raise ValueError("No model available.")
        return self.model.get_coverage_stats(self._all_activities)

