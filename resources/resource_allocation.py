import pandas as pd
import pm4py
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import random
import logging
import os
from pathlib import Path

from resources.resource_permissions.resource_permissions import BasicResourcePermissions, OrdinoRResourcePermissions
from resources.resource_availabilities.resource_availabilities import AdvancedResourceAvailabilityModel

logger = logging.getLogger(__name__)

# Default cache paths
DEFAULT_AVAILABILITY_CACHE = Path(__file__).parent / "resource_availabilities" / "bpic2017_resource_model.pkl"
DEFAULT_PERMISSIONS_CACHE = Path(__file__).parent / "resource_permissions" / "ordinor_fullrecall.pkl"

class ResourceAllocator:
    """
    Allocates resources to activities based on permissions and availability.

    Serves as the central entry point for resource management in the simulation.
    Orchestrates:
    1. Permission Model (Basic, OrdinoR-FullRecall, or OrdinoR-OverallScore) - Who is qualified?
    2. Availability Model (Advanced) - Who is free based on mined patterns?

    Permission Methods:
    - 'basic': Simple historical lookup (resource did activity before)
    - 'ordinor': OrdinoR with FullRecall profiling (default, recommended for simulation)
    - 'ordinor-strict': OrdinoR with OverallScore profiling (precision-optimized)
    """

    def __init__(self, log_path: str = None, permission_method: str = 'ordinor',
                 n_resource_clusters: int = 10,
                 use_sample: int = None, cache_path: str = None, df: pd.DataFrame = None,
                 permissions_model = None, availability_model = None,
                 availability_config: dict = None,
                 availability_cache_path: str = None):
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
                        Defaults to 'resources/resource_permissions/ordinor_fullrecall.pkl' for 'ordinor' method.
            df: Optional pre-loaded DataFrame. If provided, log_path is ignored.
            permissions_model: Optional pre-initialized permissions object (Dependency Injection).
            availability_model: Optional pre-initialized availability object (Dependency Injection).
            availability_config: Config dict for AdvancedResourceAvailabilityModel.
            availability_cache_path: Path to availability model cache.
        """
        self.log_path = log_path
        self.permission_method = permission_method.lower()

        # Set default cache paths if not provided
        if cache_path is None and self.permission_method in ('ordinor', 'ordinor-fullrecall'):
            cache_path = str(DEFAULT_PERMISSIONS_CACHE)
        if availability_cache_path is None:
            availability_cache_path = str(DEFAULT_AVAILABILITY_CACHE)

        # Check if we can load from cache (skip log loading if both caches exist)
        availability_cache_exists = Path(availability_cache_path).exists()
        permissions_cache_exists = cache_path and Path(cache_path).exists()
        can_use_permissions_cache = self.permission_method.startswith('ordinor') and permissions_cache_exists

        # Determine if we need to load the log
        need_log = False
        if permissions_model is None and availability_model is None:
            if df is not None:
                need_log = False  # DataFrame already provided
            elif not availability_cache_exists:
                need_log = True  # Need log to train availability model
            elif not can_use_permissions_cache and self.permission_method != 'basic':
                need_log = True  # Need log to train permissions model
            elif self.permission_method == 'basic':
                need_log = True  # Basic method always needs the log

        # 1. Unified Log Loading (only if needed)
        logger.info(f"Initializing ResourceAllocator with method='{self.permission_method}'")
        if df is not None:
            self.df = df
        elif need_log and log_path:
            self.df = self._load_log(log_path, use_sample)
        elif need_log and not log_path:
            raise ValueError("log_path is required when cache files are not available")
        else:
            # Both caches exist, no need to load log
            logger.info("Using cached models - skipping event log loading")
            self.df = None
        
        # 2. Initialize Availability Model
        if availability_model:
            self.availability = availability_model
        else:
            logger.info("Initializing Advanced Resource Availability Model...")
            config = availability_config or {}
            # Pass cache path; model will load from cache if available (even without df)
            config.setdefault('model_cache_path', availability_cache_path)
            self.availability = AdvancedResourceAvailabilityModel(
                event_log_df=self.df,  # Can be None if loading from cache
                **config
            )

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
                # Check if we can load from cache (without needing df)
                loaded_from_cache = False
                if can_use_permissions_cache:
                    try:
                        logger.info(f"Loading cached permissions model from {cache_path}...")
                        # Create instance without loading data, then load from cache
                        self.permissions = OrdinoRResourcePermissions(
                            profiling_mode=profiling_mode,
                            _skip_init=True
                        )
                        self.permissions.load_model(cache_path)
                        loaded_from_cache = True
                    except Exception as e:
                        logger.warning(f"Failed to load cache: {e}. Falling back to discovery.")

                if not loaded_from_cache:
                    # Need to train from data
                    if self.df is None:
                        raise ValueError("DataFrame is required to train permissions model (no cache available)")
                    self.permissions = OrdinoRResourcePermissions(
                        df=self.df,
                        profiling_mode=profiling_mode
                    )
                    logger.info(f"Discovering OrdinoR model (mode={profiling_mode})...")
                    self.permissions.discover_model(
                        n_resource_clusters=n_resource_clusters
                    )
                    if cache_path:
                        self.permissions.save_model(cache_path)

            elif self.permission_method == 'basic':
                if self.df is None:
                    raise ValueError("DataFrame is required for 'basic' permission method")
                self.permissions = BasicResourcePermissions(df=self.df)
            else:
                raise ValueError(f"Unknown permission method: '{self.permission_method}'. "
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

    def allocate(self, activity: str, timestamp: datetime, case_type: str = None) -> Tuple[Optional[str], datetime]:
        """
        Allocates a suitable resource for the given activity at the given timestamp.

        1. Finds all resources eligible for the activity (considering context).
        2. Filters them by availability at the timestamp.
        3. If no resource is available, waits until the next available time.
        4. Returns the selected resource and the actual start time.
        
        Args:
            activity: Name of the activity.
            timestamp: Desired time when the activity should be performed.
            case_type: Optional Case Type (e.g., "Home improvement", "Car").
            
        Returns:
            Tuple of (resource_id, actual_start_time):
            - If resources are available at timestamp: (resource, timestamp)
            - If no resources available: (resource, next_available_time)
            - If no eligible resources exist: (None, timestamp)
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
            return (None, timestamp)

        # 2. Availability at requested time
        available_resources = [
            res for res in eligible_resources 
            if self.availability.is_available(res, timestamp)
        ]

        if available_resources:
            # Resources available now - select one randomly
            selected_resource = random.choice(available_resources)
            return (selected_resource, timestamp)
        
        # 3. No resources available - find next available time
        # Check each eligible resource for their next available time
        next_available_times = []
        
        for resource in eligible_resources:
            next_time = self._find_next_available_time(resource, timestamp)
            if next_time:
                next_available_times.append((resource, next_time))
        
        if not next_available_times:
            # No resource will ever be available (shouldn't happen normally)
            logger.warning(f"No resource found that will become available for activity '{activity}'")
            return (None, timestamp)
        
        # Select the resource that becomes available earliest
        next_available_times.sort(key=lambda x: x[1])
        selected_resource, actual_start_time = next_available_times[0]
        
        # logger.info(f"Activity '{activity}' delayed: resource '{selected_resource}' available at {actual_start_time} instead of {timestamp}")
        return (selected_resource, actual_start_time)
    
    def _find_next_available_time(self, resource_id: str, start_time: datetime, max_days: int = 30) -> Optional[datetime]:
        """
        Find the next time a resource becomes available after start_time.
        
        Args:
            resource_id: The resource to check
            start_time: Start searching from this time
            max_days: Maximum number of days to search ahead (default: 30)
            
        Returns:
            Next available datetime, or None if not found within max_days
        """
        check_time = start_time
        end_time = start_time + timedelta(days=max_days)
        
        # Check hour by hour
        while check_time < end_time:
            if self.availability.is_available(resource_id, check_time):
                return check_time
            check_time += timedelta(hours=1)
        
        return None
