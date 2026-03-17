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
    processing_time_method: Literal["distribution", "ml", "probabilistic_ml"] = "probabilistic_ml"
    processing_time_model_path: Optional[str] = "models/processing_time_model"

    # Next activity prediction
    next_activity_class: str = "lstm"  # "lstm", "process_transformer", "lifecycle_dual"
    next_activity_model_path: Optional[str] = None
    next_activity_lifecycle_variant: Optional[str] = None  # "start_complete" or "full_lifecycle"
    next_activity_hf_repo: Optional[str] = None  # Override HuggingFace repo for lifecycle_dual download
    next_activity_temperature: float = 1.0
    # PT-only lifecycle logging mode (ignored unless next_activity_class is process_transformer)
    pt_lifecycle_mode: Literal["native", "gt_activity_gated"] = "native"
    # Max PT duration cap in seconds (prevents outlier durations from cascading queue buildup)
    pt_max_duration_seconds: Optional[float] = None

    # Case arrival times (advanced uses CaseInterarrivalPipeline)
    # NOTE: These defaults must match the parameters used to train case_arrival_model.pkl
    # (see case_arrival_times_prediction/runner.py run() defaults)
    case_arrival_mode: Literal["basic", "advanced"] = "advanced"
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
    case_attribute_seed: int = 42
    case_attribute_offer_activity: str = "O_Create Offer"
    case_attribute_monthly_artifact_path: Optional[str] = None
    case_attribute_retrain: bool = False  # If True, retrain from df instead of using cached artifacts

    # Resource selection strategy (R-RMA=random, R-RRA=round_robin, R-SHQ=shortest_queue)
    resource_selection_strategy: Literal["random", "round_robin", "shortest_queue"] = "random"

    # Resource allocation mode: "greedy" uses selection_strategy, "batch" uses batch_policy,
    # "drl" uses trained PPO, "pmsp" uses PMSP optimizer (CP-SAT + JV fallback)
    resource_allocation_mode: Literal["greedy", "batch", "drl", "pmsp"] = "greedy"
    # Batch policy (only when resource_allocation_mode="batch")
    batch_policy: Literal["1_batch_1"] = "1_batch_1"
    # DRL policy settings (only when resource_allocation_mode="drl")
    drl_model_path: Optional[str] = "models/drl_allocation/drl_allocation_model"
    drl_deterministic: bool = True
    drl_reward_tau: float = 100.0
    drl_max_postpone_wait_hours: float = 4.0
    # PMSP settings (only when resource_allocation_mode="pmsp")
    pmsp_dummy_delta: float = 1.0
    pmsp_solver_time_limit_seconds: Optional[float] = 2.0
    pmsp_prediction_batch_size: int = 0  # 0 = unlimited
    pmsp_optimization_batch_size: int = 0  # Min waiting tasks to trigger optimization (0 = always optimize)
    pmsp_park_song_lookahead: bool = True  # Park & Song [13] look-ahead: add predicted next tasks to optimization

    # Global settings
    event_log_path: Optional[str] = None
    num_cases: int = 100
    random_seed: int = 42
    verbose: bool = False
