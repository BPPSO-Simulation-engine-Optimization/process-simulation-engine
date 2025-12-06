"""
Group Profiling Module.

Determines group capabilities based on activity frequency and coverage thresholds.
"""
from typing import Dict, Set, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GroupProfiler:
    """
    Profiles resource groups to determine their activity capabilities.
    
    An activity is considered a group capability if:
    1. The group performed it at least `min_frequency` times total
    2. At least `min_coverage` fraction of group members performed it
    """
    
    def __init__(self, min_frequency: int = 5, min_coverage: float = 0.3):
        """
        Initialize the profiler.
        
        Args:
            min_frequency: Minimum total occurrences for an activity to be a capability.
            min_coverage: Minimum fraction of group members who performed the activity.
        """
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be between 0.0 and 1.0")
        
        self.min_frequency = min_frequency
        self.min_coverage = min_coverage
    
    def profile_groups(
        self,
        matrix: pd.DataFrame,
        resource_to_group: Dict[str, int]
    ) -> Dict[int, Set[str]]:
        """
        Determine capabilities for each group.
        
        Args:
            matrix: Resource-activity frequency matrix (DataFrame).
                    Rows are resources, columns are activities.
            resource_to_group: Dict mapping resource_id -> group_id.
        
        Returns:
            Dict mapping group_id -> set of activity names (capabilities).
        """
        if matrix.empty:
            logger.warning("Empty matrix provided, returning empty capabilities")
            return {}
        
        activity_names = list(matrix.columns)
        
        # Build group members lookup
        group_members: Dict[int, List[str]] = {}
        for resource_id, group_id in resource_to_group.items():
            if group_id not in group_members:
                group_members[group_id] = []
            group_members[group_id].append(resource_id)
        
        group_capabilities: Dict[int, Set[str]] = {}
        
        for group_id, members in group_members.items():
            capabilities: Set[str] = set()
            group_size = len(members)
            
            # Get submatrix for this group's members
            # Handle case where some members might not be in the matrix
            valid_members = [m for m in members if m in matrix.index]
            if not valid_members:
                logger.warning(f"Group {group_id} has no valid members in matrix")
                group_capabilities[group_id] = capabilities
                continue
            
            group_matrix = matrix.loc[valid_members]
            
            for activity in activity_names:
                # Total count for this activity in the group
                total_count = group_matrix[activity].sum()
                
                # Number of members who performed this activity (count > 0)
                members_performed = (group_matrix[activity] > 0).sum()
                coverage = members_performed / len(valid_members)
                
                # Check thresholds
                if total_count >= self.min_frequency and coverage >= self.min_coverage:
                    capabilities.add(activity)
            
            group_capabilities[group_id] = capabilities
            logger.info(f"Group {group_id} ({group_size} members): {len(capabilities)} capabilities")
        
        return group_capabilities
