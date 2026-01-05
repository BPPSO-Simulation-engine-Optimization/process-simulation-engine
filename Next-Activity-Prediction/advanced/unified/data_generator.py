"""
Training data generator for unified activity-lifecycle prediction.

Creates sequences where each step is (activity, lifecycle) and the target
is the next (activity, lifecycle) pair.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class UnifiedDataGenerator:
    """Generates training data with lifecycle information for all transitions."""

    CONTEXT_KEYS = ["case:LoanGoal", "case:ApplicationType", "case:RequestedAmount"]
    DEFAULT_LIFECYCLE = "complete"

    def __init__(
        self,
        df_log: pd.DataFrame,
        max_history: int = 15,
        min_samples: int = 10,
    ):
        self.df_log = df_log.copy()
        self.max_history = max_history
        self.min_samples = min_samples
        self._prepare_log()

    def _prepare_log(self):
        """Prepare event log with lifecycle column."""
        # Ensure timestamp is datetime
        self.df_log["time:timestamp"] = pd.to_datetime(self.df_log["time:timestamp"])

        # Add lifecycle column if missing
        if "lifecycle:transition" not in self.df_log.columns:
            self.df_log["lifecycle:transition"] = self.DEFAULT_LIFECYCLE

        # Fill missing lifecycles
        self.df_log["lifecycle:transition"] = (
            self.df_log["lifecycle:transition"]
            .fillna(self.DEFAULT_LIFECYCLE)
            .str.lower()
        )

        # Sort by case and timestamp
        self.df_log = self.df_log.sort_values(
            ["case:concept:name", "time:timestamp"]
        ).reset_index(drop=True)

    def _compute_durations(self, timestamps: List) -> List[float]:
        """Compute log-scaled duration between consecutive events."""
        import math
        
        if len(timestamps) <= 1:
            return [0.0] * len(timestamps)

        durations = [0.0]
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
            # Log transform to handle wide range of durations (seconds to days)
            log_dur = math.log1p(max(0.0, delta))
            durations.append(log_dur)
        return durations

    def _extract_case_context(self, case_id: str, group: pd.DataFrame) -> Dict:
        """Extract case-level context attributes."""
        context = {}
        for key in self.CONTEXT_KEYS:
            if key in group.columns:
                context[key] = group[key].iloc[0]
        return context

    def generate(self) -> pd.DataFrame:
        """Generate training dataset with sequences and lifecycle targets."""
        rows = []

        for case_id, group in self.df_log.groupby("case:concept:name"):
            activities = group["concept:name"].tolist()
            lifecycles = group["lifecycle:transition"].tolist()
            resources = group["org:resource"].fillna("Unknown").tolist()
            timestamps = group["time:timestamp"].tolist()

            if len(activities) < 2:
                continue

            context = self._extract_case_context(case_id, group)

            # Create sliding windows
            for i in range(1, len(activities)):
                # Sequence up to position i-1
                start_idx = max(0, i - self.max_history)
                seq_activities = activities[start_idx:i]
                seq_lifecycles = lifecycles[start_idx:i]
                seq_resources = resources[start_idx:i]
                seq_timestamps = timestamps[start_idx:i]
                seq_durations = self._compute_durations(seq_timestamps)

                # Target is the next (activity, lifecycle) pair
                target_activity = activities[i]
                target_lifecycle = lifecycles[i]

                rows.append({
                    "case_id": case_id,
                    "sequence_activities": seq_activities,
                    "sequence_lifecycles": seq_lifecycles,
                    "sequence_resources": seq_resources,
                    "sequence_durations": seq_durations,
                    "target_activity": target_activity,
                    "target_lifecycle": target_lifecycle,
                    **context,
                })

        df = pd.DataFrame(rows)

        # Filter rare activity-lifecycle combinations
        df["_target_pair"] = df["target_activity"] + "|" + df["target_lifecycle"]
        pair_counts = df["_target_pair"].value_counts()
        valid_pairs = pair_counts[pair_counts >= self.min_samples].index
        df = df[df["_target_pair"].isin(valid_pairs)].drop(columns=["_target_pair"])

        return df.reset_index(drop=True)


def generate_unified_training_data(
    df_log: pd.DataFrame,
    max_history: int = 15,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Convenience function to generate unified training data."""
    generator = UnifiedDataGenerator(df_log, max_history, min_samples)
    return generator.generate()

