import pandas as pd
import pm4py
from datetime import datetime
from typing import Optional, List
import random
import logging
import os

from resources.resource_permissions.resource_permissions import BasicResourcePermissions, OrdinoRResourcePermissions
from resources.resource_availabilities.resource_availabilities import ResourceAvailabilityModel

logger = logging.getLogger(__name__)

class ResourceAllocator:
    """
    Allocates resources to activities based on permissions and availability.
    
    Serves as the central entry point for resource management in the simulation.
    Orchestrates:
    1. Permission Model (Basic or OrdinoR) - Who is qualified?
    2. Availability Model (Working Hours) - Who is free?
    """

    def __init__(self, log_path: str = None, permission_method: str = 'ordinor', 
                 n_trace_clusters: int = 5, n_resource_clusters: int = 10,
                 use_sample: int = None, cache_path: str = None, df: pd.DataFrame = None,
                 permissions_model = None):
        """
        Initialize the ResourceAllocator.

        Args:
            log_path: Path to the XES event log.
            permission_method: Strategy for permissions ('basic' or 'ordinor').
            n_trace_clusters: Parameters for OrdinoR (if used).
            n_resource_clusters: Parameters for OrdinoR (if used).
            use_sample: If set, load only a sample of the log (for testing/speed).
            cache_path: Path to save/load the permission model cache.
            df: Optional pre-loaded DataFrame. If provided, log_path is ignored.
            permissions_model: Optional pre-initialized permissions object (Dependency Injection).
        """
        self.log_path = log_path
        self.permission_method = permission_method.lower()
        
        # 1. Unified Log Loading
        logger.info(f"Initializing ResourceAllocator with method='{self.permission_method}'")
        if df is not None:
             self.df = df
        elif log_path:
             self.df = self._load_log(log_path, use_sample)
        elif permissions_model is None:
             # Only raise if we need to load data ourselves. 
             # If permissions_model is provided, we might not strictly need df/log_path immediately 
             # unless availability model needs it.
             raise ValueError("Either log_path, df, or permissions_model must be provided.")
        else:
             self.df = None
        
        # 2. Initialize Availability Model
        logger.info("Initializing Resource Availability Model...")
        if self.df is not None:
            self.availability = ResourceAvailabilityModel(event_log_df=self.df)
        elif permissions_model:
            # If mocking, we might mock availability too, or need to handle this case.
            # For now, let's assume if DI is used, we might want to inject availability too?
            # Keeping it simple: If DI is used (tests), we might mock availability or 
            # the availability definition below might fail if self.df is None.
            # But the Original test mocks availability separately.
            pass

        # 3. Initialize Permission Model
        if permissions_model:
             self.permissions = permissions_model
        else:
            logger.info("Initializing Resource Permission Model...")
            if self.permission_method == 'ordinor':
                self.permissions = OrdinoRResourcePermissions(df=self.df)
                
                # Check for cache
                loaded_from_cache = False
                if cache_path and os.path.exists(cache_path):
                    try:
                        logger.info(f"Loading cached model from {cache_path}...")
                        self.permissions.load_model(cache_path)
                        loaded_from_cache = True
                    except Exception as e:
                        logger.warning(f"Failed to load cache: {e}. Falling back to discovery.")
                
                if not loaded_from_cache:
                    logger.info("Discovering OrdinoR model (this may take time)...")
                    self.permissions.discover_model(
                        n_trace_clusters=n_trace_clusters, 
                        n_resource_clusters=n_resource_clusters
                    )
                    if cache_path:
                        self.permissions.save_model(cache_path)
                        
            elif self.permission_method == 'basic':
                self.permissions = BasicResourcePermissions(df=self.df)
            else:
                raise ValueError(f"Unknown permission method: {permission_method}")
            
    def _load_log(self, log_path: str, sample_size: Optional[int]) -> pd.DataFrame:
        """Load and preprocess the event log into a DataFrame."""
        if not os.path.exists(log_path):
             raise FileNotFoundError(f"Log file not found: {log_path}")
             
        logger.info(f"Loading log from {log_path}...")
        log = pm4py.read_xes(log_path)
        df = pm4py.convert_to_dataframe(log)
        
        if sample_size and sample_size < len(df):
            logger.info(f"Sampling first {sample_size} events...")
            df = df.head(sample_size)
            
        return df

    def allocate(self, activity: str, timestamp: datetime, case_id: str = None) -> Optional[str]:
        """
        Allocates a suitable resource for the given activity at the given timestamp.

        1. Finds all resources eligible for the activity (considering context).
        2. Filters them by availability at the timestamp.
        3. Returns a randomly selected available resource, or None if none are found.
        
        Args:
            activity: Name of the activity.
            timestamp: Time when the activity is to be performed.
            case_id: Optional Case ID (for context-aware permissions).
        """
        # 1. Eligibility (context-aware)
        try:
             # Check if the permission model supports context (OrdinoR does, Basic might not fully)
             # Inspect signature or just try passing kwargs if flexible, 
             # but here we know OrdinoRResourcePermissions has the new signature.
             # BasicResourcePermissions needs to be updated too or use kwargs.
             
             # Safest is to check signature or assume updated interface
             eligible_resources = self.permissions.get_eligible_resources(
                 activity, timestamp=timestamp, case_id=case_id
             )
        except TypeError:
             # Fallback for models without context support (e.g. unmodified Basic)
             eligible_resources = self.permissions.get_eligible_resources(activity)

        if not eligible_resources:
            # logger.debug(f"No eligible resources found for activity '{activity}'.")
            return None

        # 2. Availability
        # Optimization: Filter list first
        available_resources = [
            res for res in eligible_resources 
            if self.availability.is_available(res, timestamp)
        ]

        if not available_resources:
            # logger.debug(f"No available resources found for activity '{activity}' at {timestamp}.")
            return None

        # 3. Selection (Random for now)
        selected_resource = random.choice(available_resources)
        return selected_resource
