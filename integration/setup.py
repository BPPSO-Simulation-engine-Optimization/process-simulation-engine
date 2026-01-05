"""
Setup functions for wiring up simulation components.

Provides factory functions to create and configure prediction components
based on SimulationConfig settings.
"""

import random
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Any
import os
import pandas as pd

from .config import SimulationConfig

logger = logging.getLogger(__name__)


def setup_simulation(
    config: SimulationConfig,
    df: Optional[pd.DataFrame] = None,
    start_date: Optional[datetime] = None,
) -> Tuple[List[datetime], Any, Any, Any, Any]:
    """
    Set up all simulation components based on configuration.

    Args:
        config: SimulationConfig specifying basic/advanced modes.
        df: Event log DataFrame for training advanced predictors.
            Required if any component is in 'advanced' mode.
        start_date: Start date for arrival timestamp generation.
            Defaults to now() if not provided.

    Returns:
        Tuple of (arrival_timestamps, next_activity_predictor, processing_time_predictor, case_attribute_predictor)

    Raises:
        ValueError: If advanced mode requires df but df is None.
    """
    if start_date is None:
        start_date = datetime.now()

    # Validate: advanced modes require training data
    needs_df = (
        config.case_arrival_mode == "advanced" or
        config.case_attribute_mode == "advanced"
    )
    if needs_df and df is None:
        raise ValueError(
            "Event log DataFrame (df) is required for advanced mode predictors. "
            "Either provide df or set all modes to 'basic'."
        )

    # 1. Case arrival timestamps
    arrival_timestamps = _setup_arrivals(config, df, start_date)

    # 2. Next activity predictor
    next_activity_pred = _setup_next_activity(config)

    # 3. Processing time predictor
    processing_time_pred = _setup_processing_time(config)

    # 4. Case attribute predictor
    case_attr_pred = _setup_case_attributes(config, df)

    return arrival_timestamps, next_activity_pred, processing_time_pred, case_attr_pred


def _setup_arrivals(
    config: SimulationConfig,
    df: Optional[pd.DataFrame],
    start_date: datetime,
) -> List[datetime]:
    """Set up case arrival timestamps using the new run() API."""
    if config.case_arrival_mode == "advanced":
        logger.info("Setting up advanced case arrival using runner API...")
        try:
            from case_arrival_times_prediction import run
            from case_arrival_times_prediction.config import SimulationConfig as ArrivalConfig

            # Build config for the arrival pipeline
            arr_config = ArrivalConfig(
                train_ratio=config.arrival_train_ratio,
                window_size=config.arrival_window_size,
                kmax=config.arrival_kmax,
                z_values=config.arrival_z_values,
                L=config.arrival_L,
                kernel=config.arrival_kernel,
                min_samples_kde=config.arrival_min_samples_kde,
                dbscan_eps=config.arrival_dbscan_eps,
                dbscan_min_samples=config.arrival_dbscan_min_samples,
                verbose=config.verbose,
                random_state=config.random_seed,
            )

            # Determine whether to retrain or load cached model
            model_path = "models/case_arrival_model.pkl"
            
            # Use cached model if it exists, otherwise retrain if data is available
            if os.path.exists(model_path):
                logger.info(f"Found existing case arrival model at {model_path}, using cached version.")
                retrain_model = False
            else:
                retrain_model = df is not None

            # The run() API generates by DAYS, not by case count.
            # BPIC17 event log statistics (analyzed from eventlog.xes.gz):
            #   - Mean daily arrivals: 86.1, Median: 88.0
            #   - Min: 20, Max: 178, Std: 32.2
            #   - Total cases: 31,509 over 366 days
            # Using 86 cases/day based on actual mean, with 1.5x buffer for variance.
            avg_cases_per_day = 86
            estimated_days = int((config.num_cases / avg_cases_per_day) * 1.5) + 1
            max_retries = 5

            # Use the new run() API which handles model caching automatically
            # Retry loop: if insufficient timestamps, increase estimated_days and regenerate
            for attempt in range(max_retries):
                timestamps = run(
                    df=df if attempt == 0 else None,  # Only pass df on first attempt
                    retrain_model=retrain_model if attempt == 0 else False,
                    model_path=model_path,
                    n_days_to_simulate=estimated_days,
                    config=arr_config,
                    start_date=start_date,
                )

                # Convert to datetime objects
                arrival_timestamps = [ts.to_pydatetime() for ts in timestamps]

                if len(arrival_timestamps) >= config.num_cases:
                    # Success: slice to exact num_cases and return
                    if len(arrival_timestamps) > config.num_cases:
                        arrival_timestamps = arrival_timestamps[:config.num_cases]
                        logger.info(
                            f"Generated {len(timestamps)} timestamps, "
                            f"sliced to {config.num_cases} (attempt {attempt + 1})"
                        )
                    else:
                        logger.info(
                            f"Generated exactly {len(arrival_timestamps)} arrival timestamps "
                            f"(attempt {attempt + 1})"
                        )
                    return arrival_timestamps

                # Insufficient timestamps: increase days and retry
                old_days = estimated_days
                estimated_days = int(estimated_days * 1.5)
                logger.info(
                    f"Insufficient timestamps ({len(arrival_timestamps)} < {config.num_cases}), "
                    f"increasing estimated_days from {old_days} to {estimated_days} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

            # All retries exhausted: fail loudly instead of silent degradation
            raise ValueError(
                f"Failed to generate sufficient arrival timestamps after {max_retries} attempts. "
                f"Generated {len(arrival_timestamps)} timestamps but {config.num_cases} requested. "
                f"Final estimated_days was {estimated_days}. "
                f"Check arrival model configuration or reduce num_cases."
            )

        except FileNotFoundError as e:
            logger.warning(f"Case arrival model not found: {e}")
            logger.warning("Falling back to basic mode. Train model first with df provided.")
        except ImportError as e:
            logger.warning(f"Could not import case arrival module: {e}")
            logger.warning("Falling back to basic mode")
        except Exception as e:
            logger.warning(f"Error in case arrival pipeline: {e}")
            logger.warning("Falling back to basic mode")

    # Basic mode: generate uniform random arrivals
    logger.info("Using basic (stub) case arrival generation...")
    return _generate_basic_arrivals(config.num_cases, start_date, config.random_seed)


def _generate_basic_arrivals(
    num_cases: int,
    start_date: datetime,
    seed: int = 42,
) -> List[datetime]:
    """Generate basic random arrival timestamps."""
    rng = random.Random(seed)
    timestamps = []
    current_time = start_date

    for _ in range(num_cases):
        # Random 1-30 minutes between cases
        minutes = rng.randint(1, 30)
        current_time = current_time + timedelta(minutes=minutes)
        timestamps.append(current_time)

    return timestamps


def _setup_processing_time(config: SimulationConfig) -> Any:
    """
    Set up processing time predictor.

    Always returns a ProcessingTimePredictionClass instance.
    Requires a trained model at the configured path.
    """
    from processing_time_prediction.ProcessingTimePredictionClass import (
        ProcessingTimePredictionClass
    )

    logger.info("Setting up processing time predictor (ProcessingTimePredictionClass)...")

    predictor = ProcessingTimePredictionClass(
        method=config.processing_time_method,
        model_path=config.processing_time_model_path,
    )
    logger.info(f"Loaded processing time model: {config.processing_time_method}")
    return predictor


def _setup_case_attributes(
    config: SimulationConfig,
    df: Optional[pd.DataFrame],
) -> Any:
    """
    Set up case attribute predictor.

    Always returns an AttributeSimulationEngine instance.
    By default loads from cached artifacts. Only retrains if explicitly requested.
    """
    from case_attribute_prediction.simulator import AttributeSimulationEngine

    logger.info("Setting up case attribute predictor (AttributeSimulationEngine)...")

    # Load monthly artifact if specified
    monthly_artifact = None
    if config.case_attribute_monthly_artifact_path:
        try:
            import joblib
            monthly_artifact = joblib.load(config.case_attribute_monthly_artifact_path)
        except Exception as e:
            logger.warning(f"Could not load monthly artifact: {e}")

    # By default, load from cached artifacts (df=None, retrain_models=False)
    # Only pass df if retraining is explicitly requested
    retrain = getattr(config, 'case_attribute_retrain', False)

    predictor = AttributeSimulationEngine(
        df=df if retrain else None,
        seed=config.case_attribute_seed,
        monthly_artifact=monthly_artifact,
        offer_create_activity=config.case_attribute_offer_activity,
        retrain_models=retrain,
    )

    mode_desc = "retrained from event log" if retrain else "from cached artifacts"
    logger.info(f"Loaded AttributeSimulationEngine ({mode_desc})")
    return predictor


def _setup_next_activity(config: SimulationConfig) -> Any:
    """
    Set up next activity predictor based on config.
    
    Returns predictor based on mode:
    - "advanced": UnifiedNextActivityPredictor (preferred) or LSTMNextActivityPredictor
    - "basic": None (engine will auto-load)
    """
    from pathlib import Path
    
    if config.next_activity_mode == "advanced":
        # Try Unified model first (best for avoiding loops)
        unified_model_dir = Path("models/unified_next_activity")
        if unified_model_dir.exists() and (unified_model_dir / "model.keras").exists():
            logger.info("Setting up advanced next activity predictor (Unified)...")
            try:
                from simulation.engine import UnifiedNextActivityPredictor
                predictor = UnifiedNextActivityPredictor(
                    model_path=str(unified_model_dir),
                    max_history=15,
                    seed=config.random_seed,
                )
                logger.info("✓ UNIFIED ADVANCED MODEL LOADED")
                return predictor
            except Exception as e:
                import traceback
                logger.warning(f"Unified predictor load failed: {e}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
        
        # Fall back to LSTM models
        logger.info("Setting up advanced next activity predictor (LSTM)...")
        from simulation.engine import LSTMNextActivityPredictor
        
        lstm_models_dir = Path("Next-Activity-Prediction/advanced/models_lstm_new")
        if not lstm_models_dir.exists():
            logger.warning(f"LSTM models directory not found: {lstm_models_dir}")
            logger.info("Falling back to auto-load (will try BranchPredictor or Stub)")
            return None
        
        try:
            predictor = LSTMNextActivityPredictor(
                models_dir=str(lstm_models_dir),
                max_history=15,
                seed=config.random_seed,
            )
            logger.info(f"✓ LSTM ADVANCED MODEL LOADED: {len(predictor.models)} models")
            logger.info(f"✓ Process graph: {len(predictor.process_graph)} nodes")
            logger.info(f"✓ Decision points: {len(predictor.decision_point_map)} DPs")
            return predictor
        except Exception as e:
            import traceback
            logger.error(f"!!! LSTM predictor load FAILED: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.info("Falling back to auto-load")
            return None
    
    # Basic mode - return None to trigger auto-load in engine
    logger.info("Using engine auto-load for next activity prediction")
    return None
