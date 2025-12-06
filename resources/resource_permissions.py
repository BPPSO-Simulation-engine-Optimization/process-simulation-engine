from typing import List, Optional, Dict, Set
import pandas as pd
import pm4py
import os

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
        # print(self.activity_resource_map)

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
