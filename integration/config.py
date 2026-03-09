"""
Configuration for the simulation engine.

Provides a dataclass to configure basic vs advanced mode for each prediction component.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class SimulationConfig:
    """
    Configuration for the DES simulation engine.

    Each component can be set to 'basic' or 'advanced'.
    """

    # Processing time prediction
    processing_time_mode: Literal["basic", "advanced"] = "basic"
    processing_time_method: Literal["distribution", "ml", "probabilistic_ml"] = "ml"
    processing_time_model_path: Optional[str] = "models/processing_time_model"

    # Next activity prediction
    next_activity_class: str = "lstm"  # "lstm", "process_transformer", "lifecycle_dual"
    next_activity_model_path: Optional[str] = None
    next_activity_lifecycle_variant: Optional[str] = None  # "start_complete" or "full_lifecycle"
    next_activity_temperature: float = 1.0

    # Case arrival times (advanced uses CaseInterarrivalPipeline)
    # NOTE: These defaults must match the parameters used to train case_arrival_model.pkl
    # (see case_arrival_times_prediction/runner.py run() defaults)
    case_arrival_mode: Literal["basic", "advanced"] = "basic"
    arrival_train_ratio: float = 0.8
    arrival_window_size: int = 21
    arrival_kmax: int = 5
    arrival_z_values: tuple = (0.9, 0.725, 0.55, 0.375, 0.2)
    arrival_L: int = 4
    arrival_kernel: str = "gaussian"
    arrival_min_samples_kde: int = 2
    arrival_dbscan_eps: float = 0.8
    arrival_dbscan_min_samples: int = 2

    # Case attributes (uses AttributeSimulationEngine)
    case_attribute_mode: Literal["basic", "advanced"] = "basic"
    case_attribute_seed: int = 42
    case_attribute_offer_activity: str = "O_Create Offer"
    case_attribute_monthly_artifact_path: Optional[str] = None
    case_attribute_retrain: bool = False  # If True, retrain from df instead of using cached artifacts

    # Global settings
    event_log_path: Optional[str] = None
    num_cases: int = 100
    random_seed: int = 42
    verbose: bool = False
