"""
Organizational Model Module.

Data structure for storing and persisting discovered organizational models.
"""
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrganizationalModel:
    """
    Represents a discovered organizational model with resource groups and capabilities.
    
    Attributes:
        resource_groups: Mapping of group_id -> set of resource_ids in that group.
        group_capabilities: Mapping of group_id -> set of activity names the group can perform.
        resource_to_group: Mapping of resource_id -> group_id for quick lookup.
    """
    resource_groups: Dict[int, Set[str]] = field(default_factory=dict)
    group_capabilities: Dict[int, Set[str]] = field(default_factory=dict)
    resource_to_group: Dict[str, int] = field(default_factory=dict)
    
    def get_groups_for_activity(self, activity: str) -> List[int]:
        """
        Find all groups that have the given activity as a capability.
        
        Args:
            activity: Activity name to search for.
        
        Returns:
            List of group IDs that can perform this activity.
        """
        return [
            group_id
            for group_id, capabilities in self.group_capabilities.items()
            if activity in capabilities
        ]
    
    def get_members_of_groups(self, group_ids: List[int]) -> Set[str]:
        """
        Get all resources that belong to any of the given groups.
        
        Args:
            group_ids: List of group IDs.
        
        Returns:
            Set of resource IDs.
        """
        members: Set[str] = set()
        for group_id in group_ids:
            if group_id in self.resource_groups:
                members.update(self.resource_groups[group_id])
        return members
    
    def get_all_activities(self) -> Set[str]:
        """Get all activities that are capabilities of at least one group."""
        all_activities: Set[str] = set()
        for capabilities in self.group_capabilities.values():
            all_activities.update(capabilities)
        return all_activities
    
    def get_coverage_stats(self, all_activities: Set[str]) -> Dict:
        """
        Calculate coverage statistics.
        
        Args:
            all_activities: Set of all activities in the log.
        
        Returns:
            Dict with coverage metrics.
        """
        covered = self.get_all_activities()
        covered_count = len(covered.intersection(all_activities))
        total_count = len(all_activities)
        
        return {
            "total_activities": total_count,
            "covered_activities": covered_count,
            "coverage_ratio": covered_count / total_count if total_count > 0 else 0.0,
            "uncovered_activities": list(all_activities - covered)
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to a JSON file.
        
        Args:
            path: File path to save to.
        """
        # Convert sets to lists for JSON serialization
        data = {
            "resource_groups": {
                str(k): list(v) for k, v in self.resource_groups.items()
            },
            "group_capabilities": {
                str(k): list(v) for k, v in self.group_capabilities.items()
            },
            "resource_to_group": self.resource_to_group
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved organizational model to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'OrganizationalModel':
        """
        Load a model from a JSON file.
        
        Args:
            path: File path to load from.
        
        Returns:
            OrganizationalModel instance.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        model = cls(
            resource_groups={
                int(k): set(v) for k, v in data["resource_groups"].items()
            },
            group_capabilities={
                int(k): set(v) for k, v in data["group_capabilities"].items()
            },
            resource_to_group=data["resource_to_group"]
        )
        
        logger.info(f"Loaded organizational model from {path}")
        return model
    
    def summary(self) -> str:
        """Generate a human-readable summary of the model."""
        lines = [
            f"Organizational Model Summary:",
            f"  Groups: {len(self.resource_groups)}",
            f"  Total Resources: {len(self.resource_to_group)}",
            f"  Total Capabilities: {len(self.get_all_activities())}",
            "",
            "  Group Details:"
        ]
        
        for group_id in sorted(self.resource_groups.keys()):
            members = self.resource_groups.get(group_id, set())
            capabilities = self.group_capabilities.get(group_id, set())
            lines.append(f"    Group {group_id}: {len(members)} members, {len(capabilities)} capabilities")
        
        return "\n".join(lines)
