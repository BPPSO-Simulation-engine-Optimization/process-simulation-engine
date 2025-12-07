"""
Resource Feature Engineering Module.

Builds resource feature matrices for organizational model mining.
Supports multiple feature dimensions per OrdinoR (2022) paper:
- AT (Activity Type): Which activities a resource performs
- CT (Case Type): Which case types a resource handles
- TT (Time Type): When a resource works (temporal patterns)
"""
from typing import List, Tuple, Optional, Literal
from enum import Enum
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureDimension(Enum):
    """Feature dimensions for resource profiling (OrdinoR paper)."""
    AT = "activity_type"      # Activity dimension
    CT = "case_type"          # Case type dimension  
    TT = "time_type"          # Temporal dimension


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


class ResourceFeatureMatrix:
    """
    Multi-dimensional feature matrix for resource profiling.
    
    Implements CT+AT+TT feature engineering from OrdinoR (2022):
    - AT (Activity Type): Resource × Activity frequency matrix
    - CT (Case Type): Resource × CaseType frequency matrix
    - TT (Time Type): Resource × TimeSlot frequency matrix
    
    Features are L2-normalized to ensure each dimension contributes equally.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        resource_col: str = "org:resource",
        activity_col: str = "concept:name",
        timestamp_col: str = "time:timestamp",
        case_type_cols: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None
    ):
        """
        Initialize the multi-dimensional feature builder.
        
        Args:
            df: Event log DataFrame.
            resource_col: Column name for resources.
            activity_col: Column name for activities.
            timestamp_col: Column name for timestamps.
            case_type_cols: Columns to use for Case Type dimension.
                           Default: ['case:ApplicationType', 'case:LoanGoal']
            dimensions: Which dimensions to include. 
                       Options: ['AT', 'CT', 'TT'] or subset.
                       Default: ['AT'] (activity only, like basic approach)
        """
        self.df = df.copy()
        self.resource_col = resource_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.case_type_cols = case_type_cols or ['case:ApplicationType', 'case:LoanGoal']
        self.dimensions = dimensions or ['AT']
        
        self.matrix: Optional[pd.DataFrame] = None
        self.resource_ids: List[str] = []
        self.feature_names: List[str] = []
        self._dimension_ranges: dict = {}  # Track which columns belong to which dimension
    
    def build_matrix(self, normalize: bool = True) -> pd.DataFrame:
        """
        Build the multi-dimensional feature matrix.
        
        Args:
            normalize: If True, L2-normalize each dimension separately
                      so they contribute equally to distance calculations.
        
        Returns:
            DataFrame with resources as rows, features as columns.
        """
        matrices = []
        
        # Clean data
        clean_df = self.df.dropna(subset=[self.resource_col])
        
        if 'AT' in self.dimensions:
            at_matrix = self._build_activity_dimension(clean_df)
            matrices.append(('AT', at_matrix))
            logger.info(f"AT dimension: {at_matrix.shape[1]} features")
        
        if 'CT' in self.dimensions:
            ct_matrix = self._build_case_type_dimension(clean_df)
            matrices.append(('CT', ct_matrix))
            logger.info(f"CT dimension: {ct_matrix.shape[1]} features")
        
        if 'TT' in self.dimensions:
            tt_matrix = self._build_time_type_dimension(clean_df)
            matrices.append(('TT', tt_matrix))
            logger.info(f"TT dimension: {tt_matrix.shape[1]} features")
        
        if not matrices:
            raise ValueError("No dimensions specified")
        
        # Align all matrices to same resources
        all_resources = set(matrices[0][1].index)
        for _, m in matrices[1:]:
            all_resources &= set(m.index)
        
        aligned_matrices = []
        col_offset = 0
        for dim_name, m in matrices:
            aligned = m.loc[list(all_resources)]
            
            # Normalize dimension if requested
            if normalize:
                aligned = self._l2_normalize(aligned)
            
            # Track dimension ranges
            self._dimension_ranges[dim_name] = (col_offset, col_offset + aligned.shape[1])
            col_offset += aligned.shape[1]
            
            aligned_matrices.append(aligned)
        
        # Concatenate all dimensions
        self.matrix = pd.concat(aligned_matrices, axis=1)
        self.resource_ids = list(self.matrix.index)
        self.feature_names = list(self.matrix.columns)
        
        logger.info(f"Built feature matrix: {len(self.resource_ids)} resources x {len(self.feature_names)} features")
        logger.info(f"Dimensions: {list(self._dimension_ranges.keys())}")
        
        return self.matrix
    
    def _build_activity_dimension(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Activity Type (AT) feature matrix."""
        if self.activity_col not in df.columns:
            raise ValueError(f"Activity column '{self.activity_col}' not found")
        
        clean = df.dropna(subset=[self.activity_col])
        
        matrix = pd.pivot_table(
            clean,
            index=self.resource_col,
            columns=self.activity_col,
            aggfunc='size',
            fill_value=0
        )
        
        # Prefix columns
        matrix.columns = [f"AT:{c}" for c in matrix.columns]
        return matrix
    
    def _build_case_type_dimension(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Case Type (CT) feature matrix."""
        ct_matrices = []
        
        for col in self.case_type_cols:
            if col not in df.columns:
                logger.warning(f"Case type column '{col}' not found, skipping")
                continue
            
            clean = df.dropna(subset=[col])
            
            # For numeric columns, bin them
            if pd.api.types.is_numeric_dtype(clean[col]):
                col_name = col.replace('case:', '')
                clean = clean.copy()
                clean[f'{col}_bin'] = pd.qcut(clean[col], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
                pivot_col = f'{col}_bin'
            else:
                pivot_col = col
                col_name = col.replace('case:', '')
            
            matrix = pd.pivot_table(
                clean,
                index=self.resource_col,
                columns=pivot_col,
                aggfunc='size',
                fill_value=0
            )
            
            # Prefix columns
            matrix.columns = [f"CT:{col_name}:{c}" for c in matrix.columns]
            ct_matrices.append(matrix)
        
        if not ct_matrices:
            # Return empty matrix with same index
            return pd.DataFrame(index=df[self.resource_col].unique())
        
        # Combine all case type features
        combined = pd.concat(ct_matrices, axis=1).fillna(0)
        return combined
    
    def _build_time_type_dimension(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Time Type (TT) feature matrix."""
        if self.timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{self.timestamp_col}' not found")
            return pd.DataFrame(index=df[self.resource_col].unique())
        
        clean = df.dropna(subset=[self.timestamp_col]).copy()
        
        # Extract time features
        timestamps = pd.to_datetime(clean[self.timestamp_col])
        
        # Hour bins: morning (6-12), afternoon (12-18), evening (18-22), night (22-6)
        hours = timestamps.dt.hour
        clean['time_period'] = pd.cut(
            hours, 
            bins=[-1, 6, 12, 18, 22, 24],
            labels=['night', 'morning', 'afternoon', 'evening', 'late_night']
        )
        
        # Day of week: weekday vs weekend
        clean['day_type'] = timestamps.dt.dayofweek.apply(
            lambda x: 'weekend' if x >= 5 else 'weekday'
        )
        
        matrices = []
        
        # Time period matrix
        time_matrix = pd.pivot_table(
            clean.dropna(subset=['time_period']),
            index=self.resource_col,
            columns='time_period',
            aggfunc='size',
            fill_value=0
        )
        time_matrix.columns = [f"TT:period:{c}" for c in time_matrix.columns]
        matrices.append(time_matrix)
        
        # Day type matrix
        day_matrix = pd.pivot_table(
            clean,
            index=self.resource_col,
            columns='day_type',
            aggfunc='size',
            fill_value=0
        )
        day_matrix.columns = [f"TT:day:{c}" for c in day_matrix.columns]
        matrices.append(day_matrix)
        
        combined = pd.concat(matrices, axis=1).fillna(0)
        return combined
    
    def _l2_normalize(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """L2-normalize rows of a matrix."""
        norms = np.linalg.norm(matrix.values, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = matrix.values / norms
        return pd.DataFrame(normalized, index=matrix.index, columns=matrix.columns)
    
    def get_numpy_array(self) -> np.ndarray:
        """Get the matrix as a numpy array."""
        if self.matrix is None:
            self.build_matrix()
        return self.matrix.values
    
    def get_resource_ids(self) -> List[str]:
        """Get ordered list of resource IDs."""
        return self.resource_ids
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return self.feature_names
    
    def get_dimension_info(self) -> dict:
        """Get information about which features belong to which dimension."""
        return self._dimension_ranges

