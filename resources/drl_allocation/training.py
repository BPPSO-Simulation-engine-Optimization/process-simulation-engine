"""
DRL resource allocation training script.

Trains a MaskablePPO agent to allocate resources in the BPIC17 simulation.

Usage:
    # Smoke test (~minutes)
    python -m resources.drl_allocation.training --event-log "Dataset/BPI Challenge 2017.xes" \
        --num-cases 50 --total-timesteps 1000 --output-dir models/drl_test

    # Real training (~hours)
    python -m resources.drl_allocation.training --event-log "Dataset/BPI Challenge 2017.xes" \
        --num-cases 500 --total-timesteps 2000000

    # Monitor with TensorBoard:
    tensorboard --logdir models/drl_allocation/tb_logs/
"""

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def load_event_log(path: str) -> pd.DataFrame:
    """Load event log from XES or CSV."""
    if path.endswith('.xes') or path.endswith('.xes.gz'):
        import pm4py
        log = pm4py.read_xes(path)
        df = pm4py.convert_to_dataframe(log)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path}")
    print(f"Loaded event log: {len(df)} events, {df['case:concept:name'].nunique()} cases")
    return df


def build_state_builder(allocator):
    """
    Build a DRLStateBuilder from the OrdinoR permission model.

    Returns:
        DRLStateBuilder instance.
    """
    from resources.drl_allocation.state import DRLStateBuilder

    permissions = allocator.permissions

    # Extract OrdinoR groups
    if not hasattr(permissions, '_groups') or not permissions._groups:
        raise ValueError(
            "DRL training requires OrdinoR permissions with FullRecall mode. "
            "Ensure the allocator uses OrdinoRResourcePermissions."
        )

    role_groups = permissions._groups
    resource_to_role = permissions._resource_to_group
    activity_to_roles = permissions._activity_to_groups

    # Build activity list: all activities known to the permission model, sorted
    activity_list = sorted(activity_to_roles.keys())

    print(f"State builder: {len(role_groups)} roles, {len(activity_list)} activities")
    print(f"  Observation size: {3*len(role_groups) + 3*len(activity_list) + 5}")
    print(f"  Action space: {len(activity_list) + 1} (activities + postpone)")

    return DRLStateBuilder(
        activity_list=activity_list,
        role_groups=role_groups,
        resource_to_role=resource_to_role,
        activity_to_roles=activity_to_roles,
    )


def create_engine_factory(allocator, df, config, state_builder):
    """
    Create a closure that builds a TrainingDESEngine with a given bridge.

    Returns:
        Callable[InteractiveBatchPolicy -> TrainingDESEngine]
    """
    from integration.setup import setup_simulation
    from simulation.engine import TrainingDESEngine, NextActivityPredictorType
    from resources.selection_strategies import create_strategy

    # Setup predictors once (they're reusable across episodes)
    start_date = pd.to_datetime(df['time:timestamp']).min().to_pydatetime()

    arrivals, next_act_pred, proc_pred, attr_pred = setup_simulation(
        config, df=df, start_date=start_date,
    )

    resource_strategy = create_strategy(config.resource_selection_strategy)

    # If setup_simulation returned None for next_activity (no model files found),
    # let the engine auto-load using STUB type
    pred_type = None
    if next_act_pred is None:
        pred_type = NextActivityPredictorType.STUB

    def factory(bridge):
        engine = TrainingDESEngine(
            resource_allocator=allocator,
            arrival_timestamps=arrivals[:config.num_cases],
            next_activity_predictor=next_act_pred,
            next_activity_predictor_type=pred_type,
            processing_time_predictor=proc_pred,
            case_attribute_predictor=attr_pred,
            start_time=start_date,
            resource_selection_strategy=resource_strategy,
            drl_policy=bridge,
        )
        return engine

    return factory


def main():
    parser = argparse.ArgumentParser(
        description="Train DRL resource allocation agent (MaskablePPO)"
    )
    parser.add_argument(
        "--event-log",
        default="Dataset/BPI Challenge 2017.xes",
        help="Path to event log file",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=500,
        help="Number of cases per training episode",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Total PPO training timesteps",
    )
    parser.add_argument(
        "--reward-tau",
        type=float,
        default=100.0,
        help="Reward scaling reference time (hours): r = 1/(1 + CT/tau)",
    )
    parser.add_argument(
        "--eval-cases",
        type=int,
        default=100,
        help="Number of cases per evaluation episode",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluate every N training steps",
    )
    parser.add_argument(
        "--output-dir",
        default="models/drl_allocation",
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")
    # Keep engine quiet during training
    logging.getLogger("simulation").setLevel(logging.WARNING)

    print("=" * 60)
    print("DRL Resource Allocation Training")
    print("=" * 60)
    print(f"  Event log: {args.event_log}")
    print(f"  Cases/episode: {args.num_cases}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Reward tau: {args.reward_tau}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    # Load data
    print("\nLoading event log...")
    df = load_event_log(args.event_log)

    # Create allocator
    print("Creating resource allocator...")
    from resources import ResourceAllocator
    allocator = ResourceAllocator(log_path=args.event_log)

    # Build state builder from OrdinoR groups
    print("Building state builder...")
    state_builder = build_state_builder(allocator)

    # Create config (basic mode for fast training)
    from integration.config import SimulationConfig
    config = SimulationConfig.all_basic()
    config.num_cases = args.num_cases
    config.event_log_path = args.event_log
    # setup_simulation reads this dynamically-set attribute
    config.next_activity_class = "lstm"

    # Create engine factory
    print("Setting up engine factory...")
    engine_factory = create_engine_factory(allocator, df, config, state_builder)

    # Create training env
    print("Creating training environment...")
    from resources.drl_allocation.env import ResourceAllocationEnv
    train_env = ResourceAllocationEnv(
        engine_factory=engine_factory,
        state_builder=state_builder,
        num_cases=args.num_cases,
        reward_tau=args.reward_tau,
    )

    # Create eval env (fewer cases for speed)
    eval_env = ResourceAllocationEnv(
        engine_factory=engine_factory,
        state_builder=state_builder,
        num_cases=args.eval_cases,
        reward_tau=args.reward_tau,
    )

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    tb_log_dir = os.path.join(args.output_dir, "tb_logs")
    eval_log_dir = os.path.join(args.output_dir, "eval_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    # Create MaskablePPO model
    print("Creating MaskablePPO model...")
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import EvalCallback

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=tb_log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )

    # Setup eval callback
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=3,
        log_path=eval_log_dir,
        best_model_save_path=os.path.join(args.output_dir, "best_model"),
        deterministic=True,
    )

    # Train
    print(f"\nStarting training ({args.total_timesteps:,} timesteps)...")
    print(f"Monitor with: tensorboard --logdir {tb_log_dir}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
    )

    # Save model
    model_path = os.path.join(args.output_dir, "drl_allocation_model")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save state builder config for inference reconstruction
    config_path = os.path.join(args.output_dir, "state_builder_config.pkl")
    sb_config = {
        "activity_list": state_builder.activity_list,
        "role_groups": state_builder.role_groups,
        "resource_to_role": state_builder.resource_to_role,
        "activity_to_roles": state_builder.activity_to_roles,
        "num_roles": state_builder.num_roles,
        "reward_tau": args.reward_tau,
    }
    with open(config_path, "wb") as f:
        pickle.dump(sb_config, f)
    print(f"State builder config saved to: {config_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
