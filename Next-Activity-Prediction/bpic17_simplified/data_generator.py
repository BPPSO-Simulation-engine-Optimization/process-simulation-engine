"""
Training data generator for BPIC17 simplified model.

Generates sequences of (activity, lifecycle) pairs from filtered event log.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class BPIC17SimplifiedDataGenerator:
    """Generates training data with simplified lifecycle (start/complete only)."""

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
        if "time:timestamp" in self.df_log.columns:
            self.df_log["time:timestamp"] = pd.to_datetime(self.df_log["time:timestamp"])

        if "lifecycle:transition" not in self.df_log.columns:
            self.df_log["lifecycle:transition"] = self.DEFAULT_LIFECYCLE

        self.df_log["lifecycle:transition"] = (
            self.df_log["lifecycle:transition"]
            .fillna(self.DEFAULT_LIFECYCLE)
            .str.lower()
            .str.strip()
        )

        self.df_log = self.df_log.sort_values(
            ["case:concept:name", "time:timestamp"]
        ).reset_index(drop=True)

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

            if len(activities) < 2:
                continue

            context = self._extract_case_context(case_id, group)

            for i in range(1, len(activities)):
                start_idx = max(0, i - self.max_history)
                seq_activities = activities[start_idx:i]
                seq_lifecycles = lifecycles[start_idx:i]
                seq_resources = resources[start_idx:i]

                target_activity = activities[i]
                target_lifecycle = lifecycles[i]

                rows.append({
                    "case_id": case_id,
                    "sequence_activities": seq_activities,
                    "sequence_lifecycles": seq_lifecycles,
                    "sequence_resources": seq_resources,
                    "target_activity": target_activity,
                    "target_lifecycle": target_lifecycle,
                    **context,
                })

        df = pd.DataFrame(rows)

        if len(df) == 0:
            raise ValueError("No training sequences generated. Check event log format.")

        df["_target_pair"] = df["target_activity"] + "|" + df["target_lifecycle"]
        pair_counts = df["_target_pair"].value_counts()
        valid_pairs = pair_counts[pair_counts >= self.min_samples].index
        df = df[df["_target_pair"].isin(valid_pairs)].drop(columns=["_target_pair"])

        return df.reset_index(drop=True)


def generate_training_data(
    df_log: pd.DataFrame,
    max_history: int = 15,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Convenience function to generate training data."""
    generator = BPIC17SimplifiedDataGenerator(df_log, max_history, min_samples)
    return generator.generate()


