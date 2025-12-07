from datetime import datetime
from typing import Optional, List
import random
import logging

logger = logging.getLogger(__name__)

class ResourceAllocator:
    """
    Allocates resources to activities based on permissions and availability.
    """

    def __init__(self, permissions_model, availability_model):
        """
        Initialize the ResourceAllocator.

        Args:
            permissions_model: An object with a get_eligible_resources(activity_name) method.
            availability_model: An object with an is_available(resource_id, timestamp) method.
        """
        self.permissions = permissions_model
        self.availability = availability_model

    def allocate(self, activity: str, timestamp: datetime) -> Optional[str]:
        """
        Allocates a suitable resource for the given activity at the given timestamp.

        1. Finds all resources eligible for the activity.
        2. Filters them by availability at the timestamp.
        3. Returns a randomly selected available resource, or None if none are found.

        Args:
            activity: The name of the activity.
            timestamp: The timestamp when the activity is to be performed.

        Returns:
            The ID of the allocated resource, or None if no resource could be allocated.
        """
        eligible_resources = self.permissions.get_eligible_resources(activity)
        if not eligible_resources:
            logger.debug(f"No eligible resources found for activity '{activity}'.")
            return None

        available_resources = []
        for resource in eligible_resources:
            if self.availability.is_available(resource, timestamp):
                available_resources.append(resource)

        if not available_resources:
            logger.debug(f"No available resources found for activity '{activity}' at {timestamp}. Eligible: {eligible_resources}")
            return None

        selected_resource = random.choice(available_resources)
        return selected_resource
