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
    1. Permission Model (Basic, OrdinoR-FullRecall, or OrdinoR-OverallScore) - Who is qualified?
    2. Availability Model (Working Hours) - Who is free?

    Permission Methods:
    - 'basic': Simple historical lookup (resource did activity before)
    - 'ordinor': OrdinoR with FullRecall profiling (default, recommended for simulation)
    - 'ordinor-strict': OrdinoR with OverallScore profiling (precision-optimized)
    """

    def __init__(self, log_path: str = None, permission_method: str = 'ordinor',
                 n_resource_clusters: int = 10,
                 use_sample: int = None, cache_path: str = None, df: pd.DataFrame = None,
                 permissions_model = None):
        """
        Initialize the ResourceAllocator.

        Args:
            log_path: Path to the XES event log.
            permission_method: Strategy for permissions:
                - 'basic': Simple historical lookup
                - 'ordinor': OrdinoR with FullRecall (default, role-based generalization)
                - 'ordinor-strict': OrdinoR with OverallScore (precision-optimized)
            n_resource_clusters: Number of resource clusters for OrdinoR (default 10).
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

            # Determine profiling mode from permission_method
            if self.permission_method in ('ordinor', 'ordinor-fullrecall'):
                profiling_mode = 'full_recall'
            elif self.permission_method == 'ordinor-strict':
                profiling_mode = 'overall_score'
            else:
                profiling_mode = None  # Not OrdinoR

            if self.permission_method.startswith('ordinor'):
                self.permissions = OrdinoRResourcePermissions(
                    df=self.df,
                    profiling_mode=profiling_mode
                )

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
                    logger.info(f"Discovering OrdinoR model (mode={profiling_mode})...")
                    self.permissions.discover_model(
                        n_resource_clusters=n_resource_clusters
                    )
                    if cache_path:
                        self.permissions.save_model(cache_path)

            elif self.permission_method == 'basic':
                self.permissions = BasicResourcePermissions(df=self.df)
            else:
                raise ValueError(f"Unknown permission method: '{permission_method}'. "
                               f"Valid options: 'basic', 'ordinor', 'ordinor-strict'")
            
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

    def allocate(self, activity: str, timestamp: datetime, case_type: str = None) -> Optional[str]:
        """
        Allocates a suitable resource for the given activity at the given timestamp.

        1. Finds all resources eligible for the activity (considering context).
        2. Filters them by availability at the timestamp.
        3. Returns a randomly selected available resource, or None if none are found.
        
        Args:
            activity: Name of the activity.
            timestamp: Time when the activity is to be performed.
            case_type: Optional Case Type (e.g., "Home improvement", "Car").
        """
        # 1. Eligibility (context-aware)
        try:
             eligible_resources = self.permissions.get_eligible_resources(
                 activity, timestamp=timestamp, case_type=case_type
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
