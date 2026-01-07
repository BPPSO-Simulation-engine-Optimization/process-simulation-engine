"""
Lifecycle filtering module for event log preprocessing.

Filters event logs to keep only 'start' and 'complete' lifecycle transitions,
then collapses them into single activity instances with computed processing times.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class LifecycleFilter:
    """
    Filters and transforms event logs by lifecycle transitions.

    Keeps only 'start' and 'complete' events, pairs them, and computes
    actual processing times (complete - start) for activities.

    Activities without start events (e.g., O_* activities) are treated as
    instant (0 duration) since they represent automatic/system events.
    """

    LIFECYCLE_COLUMN = "lifecycle:transition"
    ACTIVITY_COLUMN = "concept:name"
    CASE_COLUMN = "case:concept:name"
    TIMESTAMP_COLUMN = "time:timestamp"
    RESOURCE_COLUMN = "org:resource"

    VALID_LIFECYCLES = {"start", "complete"}

    def __init__(self, df_log: pd.DataFrame):
        """
        Initialize the filter with an event log DataFrame.

        Args:
            df_log: Event log DataFrame with lifecycle:transition column
        """
        self._log = df_log.copy()
        self._validate_columns()

    def _validate_columns(self):
        """Validate required columns exist."""
        required = [self.LIFECYCLE_COLUMN, self.ACTIVITY_COLUMN,
                    self.CASE_COLUMN, self.TIMESTAMP_COLUMN]
        missing = [col for col in required if col not in self._log.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_lifecycle_distribution(self) -> pd.Series:
        """Return distribution of lifecycle transitions."""
        return self._log[self.LIFECYCLE_COLUMN].value_counts()

    def get_activity_lifecycle_summary(self) -> pd.DataFrame:
        """
        Return summary of which activities have start/complete events.

        Returns:
            DataFrame with columns: activity, has_start, has_complete,
            start_count, complete_count
        """
        pivot = self._log.groupby([self.ACTIVITY_COLUMN, self.LIFECYCLE_COLUMN]).size().unstack(fill_value=0)

        summary = pd.DataFrame({
            "activity": pivot.index,
            "has_start": pivot.get("start", 0) > 0,
            "has_complete": pivot.get("complete", 0) > 0,
            "start_count": pivot.get("start", 0),
            "complete_count": pivot.get("complete", 0),
        }).reset_index(drop=True)

        return summary

    def filter_lifecycle(self) -> pd.DataFrame:
        """
        Filter to keep only start and complete lifecycle transitions.

        Returns:
            Filtered DataFrame with only start/complete events
        """
        mask = self._log[self.LIFECYCLE_COLUMN].isin(self.VALID_LIFECYCLES)
        filtered = self._log[mask].copy()

        print(f"Lifecycle filter: {len(self._log)} -> {len(filtered)} events "
              f"({len(filtered)/len(self._log)*100:.1f}% retained)")

        return filtered

    def collapse_to_activity_instances(self, filtered_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Collapse start/complete pairs into single activity instances.

        For activities with both start and complete:
            - processing_time = complete_timestamp - start_timestamp
            - resource = resource from complete event (who finished it)
            - timestamp = complete_timestamp (when activity finished)

        For activities with only complete (O_*/A_* activities):
            - processing_time = 0 (instant/automatic)
            - resource = resource from complete event
            - timestamp = complete_timestamp

        Args:
            filtered_df: Pre-filtered DataFrame (optional, will filter if not provided)

        Returns:
            DataFrame with one row per activity instance, including processing_time column
        """
        if filtered_df is None:
            filtered_df = self.filter_lifecycle()

        df = filtered_df.copy()

        # Identify activities that have start events (only W_* activities)
        activities_with_start = set(
            df[df[self.LIFECYCLE_COLUMN] == "start"][self.ACTIVITY_COLUMN].unique()
        )

        # Split into complete events (our output base) and start events (for pairing)
        completes = df[df[self.LIFECYCLE_COLUMN] == "complete"].copy()
        starts = df[df[self.LIFECYCLE_COLUMN] == "start"].copy()

        # Initialize processing_time to 0 for all complete events
        completes["processing_time"] = 0.0

        # Only process activities that have start events (W_* activities)
        if len(activities_with_start) > 0 and len(starts) > 0:
            # Filter to only work activities for pairing
            work_completes = completes[completes[self.ACTIVITY_COLUMN].isin(activities_with_start)].copy()
            work_starts = starts[starts[self.ACTIVITY_COLUMN].isin(activities_with_start)].copy()

            if len(work_completes) > 0 and len(work_starts) > 0:
                # Store original index for updating later
                work_completes["_orig_idx"] = work_completes.index

                # Sort both by case, activity, timestamp for merge_asof
                work_completes_sorted = work_completes.sort_values(
                    [self.CASE_COLUMN, self.ACTIVITY_COLUMN, self.TIMESTAMP_COLUMN]
                ).reset_index(drop=True)

                work_starts_sorted = work_starts[[self.CASE_COLUMN, self.ACTIVITY_COLUMN, self.TIMESTAMP_COLUMN]].copy()
                work_starts_sorted = work_starts_sorted.rename(columns={self.TIMESTAMP_COLUMN: "start_timestamp"})
                work_starts_sorted = work_starts_sorted.sort_values(
                    [self.CASE_COLUMN, self.ACTIVITY_COLUMN, "start_timestamp"]
                ).reset_index(drop=True)

                # Use merge_asof to find the closest start event before each complete
                # This is a vectorized backward search
                paired = pd.merge_asof(
                    work_completes_sorted.sort_values(self.TIMESTAMP_COLUMN),
                    work_starts_sorted.sort_values("start_timestamp"),
                    left_on=self.TIMESTAMP_COLUMN,
                    right_on="start_timestamp",
                    by=[self.CASE_COLUMN, self.ACTIVITY_COLUMN],
                    direction="backward"
                )

                # Calculate processing time where we have a matching start
                has_start = paired["start_timestamp"].notna()
                paired.loc[has_start, "processing_time"] = (
                    paired.loc[has_start, self.TIMESTAMP_COLUMN] - paired.loc[has_start, "start_timestamp"]
                ).dt.total_seconds()

                # Update the completes dataframe with computed processing times
                completes.loc[paired["_orig_idx"].values, "processing_time"] = paired["processing_time"].values

        # Remove lifecycle column
        result = completes.drop(columns=[self.LIFECYCLE_COLUMN])

        # Sort by case and timestamp
        result = result.sort_values(
            [self.CASE_COLUMN, self.TIMESTAMP_COLUMN]
        ).reset_index(drop=True)

        print(f"Collapsed to {len(result)} activity instances")

        return result

    def transform(self) -> pd.DataFrame:
        """
        Full transformation: filter lifecycle and collapse to activity instances.

        Returns:
            Transformed DataFrame ready for TrainingDataGenerator
        """
        filtered = self.filter_lifecycle()
        collapsed = self.collapse_to_activity_instances(filtered)
        return collapsed

    def get_processing_time_stats(self, collapsed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute processing time statistics per activity.

        Args:
            collapsed_df: DataFrame from collapse_to_activity_instances()

        Returns:
            DataFrame with processing time statistics per activity
        """
        stats = collapsed_df.groupby(self.ACTIVITY_COLUMN)["processing_time"].agg([
            "count", "mean", "std", "min", "median", "max"
        ]).round(2)

        # Add flag for instant activities
        stats["is_instant"] = stats["max"] == 0

        return stats.sort_values("mean", ascending=False)
