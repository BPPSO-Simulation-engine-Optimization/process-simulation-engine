"""
Resource Feature Engineering Module.

Builds resource-activity frequency matrices for organizational model mining.
"""
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ResourceActivityMatrix:
    """
    Constructs a resource-activity frequency matrix from event log data.
    
    The matrix has:
    - Rows: Resources (unique resource IDs)
    - Columns: Activities (unique activity names)
    - Values: Count of (resource, activity) occurrences in the log
    """
    
    def __init__(self, df: pd.DataFrame, 
                 activity_col: str = "concept:name",
                 resource_col: str = "org:resource"):
        """
        Initialize the matrix builder.
        
        Args:
            df: Event log DataFrame.
            activity_col: Column name for activities.
            resource_col: Column name for resources.
        """
        self.df = df
        self.activity_col = activity_col
        self.resource_col = resource_col
        
        self.matrix: Optional[pd.DataFrame] = None
        self.resource_ids: List[str] = []
        self.activity_names: List[str] = []
    
    def build_matrix(self) -> pd.DataFrame:
        """
        Build the resource-activity frequency matrix.
        
        Returns:
            DataFrame with resources as rows, activities as columns, counts as values.
        
        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        if self.activity_col not in self.df.columns:
            raise ValueError(f"Activity column '{self.activity_col}' not found in DataFrame")
        if self.resource_col not in self.df.columns:
            raise ValueError(f"Resource column '{self.resource_col}' not found in DataFrame")
        
        # Drop rows with missing activity or resource
        clean_df = self.df.dropna(subset=[self.activity_col, self.resource_col])
        
        if clean_df.empty:
            logger.warning("No valid (activity, resource) pairs found in the log")
            self.matrix = pd.DataFrame()
            return self.matrix
        
        # Build frequency matrix using pivot_table
        self.matrix = pd.pivot_table(
            clean_df,
            index=self.resource_col,
            columns=self.activity_col,
            aggfunc='size',
            fill_value=0
        )
        
        self.resource_ids = list(self.matrix.index)
        self.activity_names = list(self.matrix.columns)
        
        logger.info(f"Built resource-activity matrix: {len(self.resource_ids)} resources x {len(self.activity_names)} activities")
        
        return self.matrix
    
    def get_numpy_array(self) -> np.ndarray:
        """
        Get the matrix as a numpy array (for sklearn).
        
        Returns:
            2D numpy array of shape (n_resources, n_activities).
        """
        if self.matrix is None:
            self.build_matrix()
        return self.matrix.values
    
    def get_resource_ids(self) -> List[str]:
        """Get ordered list of resource IDs (row labels)."""
        return self.resource_ids
    
    def get_activity_names(self) -> List[str]:
        """Get ordered list of activity names (column labels)."""
        return self.activity_names
