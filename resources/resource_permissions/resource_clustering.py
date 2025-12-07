"""
Resource Clustering Module.

Implements Agglomerative Hierarchical Clustering to discover resource groups (roles).
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import logging

logger = logging.getLogger(__name__)


class ResourceClusterer:
    """
    Clusters resources into groups based on their activity profiles.
    
    Uses Agglomerative Hierarchical Clustering (AHC) with Ward linkage
    to create non-overlapping resource groups.
    """
    
    def __init__(self, n_clusters: int = 5, linkage: str = 'ward'):
        """
        Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters to create.
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single').
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_: Optional[np.ndarray] = None
    
    def cluster(self, matrix: np.ndarray, resource_ids: List[str]) -> Dict[str, int]:
        """
        Perform clustering on the resource-activity matrix.
        
        Args:
            matrix: 2D numpy array of shape (n_resources, n_activities).
            resource_ids: List of resource IDs corresponding to matrix rows.
        
        Returns:
            Dict mapping resource_id -> group_id.
        
        Raises:
            ValueError: If matrix is empty or has fewer rows than clusters.
        """
        if matrix.size == 0:
            raise ValueError("Cannot cluster empty matrix")
        
        n_resources = matrix.shape[0]
        if n_resources < self.n_clusters:
            logger.warning(f"Fewer resources ({n_resources}) than clusters ({self.n_clusters}). "
                          f"Reducing n_clusters to {n_resources}")
            actual_n_clusters = n_resources
        else:
            actual_n_clusters = self.n_clusters
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=actual_n_clusters,
            linkage=self.linkage
        )
        self.labels_ = clustering.fit_predict(matrix)
        
        # Build resource -> group mapping
        resource_to_group = {
            resource_ids[i]: int(self.labels_[i])
            for i in range(len(resource_ids))
        }
        
        # Log cluster sizes
        cluster_sizes = {}
        for group_id in self.labels_:
            cluster_sizes[group_id] = cluster_sizes.get(group_id, 0) + 1
        
        logger.info(f"Discovered {actual_n_clusters} groups with sizes: {dict(sorted(cluster_sizes.items()))}")
        
        return resource_to_group
    
    def get_group_members(self, resource_to_group: Dict[str, int]) -> Dict[int, List[str]]:
        """
        Invert the resource-to-group mapping to get group members.
        
        Args:
            resource_to_group: Dict mapping resource_id -> group_id.
        
        Returns:
            Dict mapping group_id -> list of resource_ids.
        """
        group_members: Dict[int, List[str]] = {}
        for resource_id, group_id in resource_to_group.items():
            if group_id not in group_members:
                group_members[group_id] = []
            group_members[group_id].append(resource_id)
        return group_members
