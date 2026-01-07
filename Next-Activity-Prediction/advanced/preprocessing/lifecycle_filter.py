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

        For activities with only complete (O_* activities):
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

        # Sort by case, activity, timestamp
        df = filtered_df.sort_values(
            [self.CASE_COLUMN, self.ACTIVITY_COLUMN, self.TIMESTAMP_COLUMN]
        ).reset_index(drop=True)

        # Identify activities that have start events
        activities_with_start = set(
            df[df[self.LIFECYCLE_COLUMN] == "start"][self.ACTIVITY_COLUMN].unique()
        )

        collapsed_rows = []

        # Process each case
        for case_id, case_group in df.groupby(self.CASE_COLUMN):
            # Separate start and complete events
            starts = case_group[case_group[self.LIFECYCLE_COLUMN] == "start"]
            completes = case_group[case_group[self.LIFECYCLE_COLUMN] == "complete"]

            # Index starts by activity for quick lookup
            # Store as list of (timestamp, row_dict) tuples for proper comparison
            start_events = {}
            for _, row in starts.iterrows():
                activity = row[self.ACTIVITY_COLUMN]
                if activity not in start_events:
                    start_events[activity] = []
                # Store as tuple (timestamp, row_dict) for easier comparison
                start_events[activity].append((row[self.TIMESTAMP_COLUMN], row.to_dict()))

            # Process each complete event
            for _, complete_row in completes.iterrows():
                activity = complete_row[self.ACTIVITY_COLUMN]
                complete_ts = complete_row[self.TIMESTAMP_COLUMN]

                # Check if this activity has start events
                if activity in activities_with_start and activity in start_events:
                    # Find matching start event (closest start before this complete)
                    matching_idx = None
                    matching_ts = None
                    for idx, (start_ts, _) in enumerate(start_events[activity]):
                        if start_ts <= complete_ts:
                            if matching_ts is None or start_ts > matching_ts:
                                matching_idx = idx
                                matching_ts = start_ts

                    if matching_idx is not None:
                        # Compute processing time
                        processing_time = (complete_ts - matching_ts).total_seconds()
                        # Remove used start event to handle multiple instances
                        start_events[activity].pop(matching_idx)
                        if not start_events[activity]:
                            del start_events[activity]
                    else:
                        # No matching start found, treat as instant
                        processing_time = 0.0
                else:
                    # Activity doesn't have start events (O_*/A_* activities), treat as instant
                    processing_time = 0.0

                # Create collapsed row
                row_dict = complete_row.to_dict()
                row_dict["processing_time"] = processing_time
                # Remove lifecycle column as it's no longer needed
                row_dict.pop(self.LIFECYCLE_COLUMN, None)
                collapsed_rows.append(row_dict)

        result = pd.DataFrame(collapsed_rows)

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
