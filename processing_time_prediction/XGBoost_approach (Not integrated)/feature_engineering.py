import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import time, timedelta


class FeatureEngineering:
    """Handles feature engineering for process mining prediction tasks."""

    def __init__(
        self,
        events_of_interest: Optional[List[str]] = None,
        case_features: Optional[List[str]] = None,
        base_features: Optional[List[str]] = None
    ):
        """
        Initialize the FeatureEngineering class.
        
        Args:
            events_of_interest: List of events to include in analysis
            case_features: List of case-level features to use
            base_features: List of base features to use (default: ["event", "lifecycle:transition", "event_index", "hour", "weekday"])
        """
        # Define events of interest (can be customized)
        self.events_of_interest = events_of_interest or [
            "O_Sent (mail and online)",
            "O_Sent (online only)",
            "O_Returned",
            "O_Refused",
            "O_Created",
            "A_Validating",
            "A_Incomplete",
            "A_Concept",
            "A_Complete"
        ]

        # Define case features (can be customized)
        self.case_features = case_features or [
            "case:LoanGoal",
            "case:ApplicationType",
            "case:RequestedAmount"
        ]
        
        # Define base features (can be customized)
        # These are the core features that are always available
        self.base_features = base_features or [
            "event",
            "lifecycle:transition",
            "event_index",
            "hour",
            "weekday"
        ]

        # Business hours configuration
        self.work_start = time(5, 0)  # 9:00 AM
        self.work_end = time(22, 0)   # 5:00 PM

    def business_seconds_between(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """
        Calculate business seconds between two timestamps (only during work hours).

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            Business seconds between start and end
        """
        if start >= end:
            return 0.0

        total = 0.0
        current = start

        while current.date() <= end.date():
            day_start = pd.Timestamp.combine(current.date(), self.work_start)
            day_end = pd.Timestamp.combine(current.date(), self.work_end)

            interval_start = max(current, day_start)
            interval_end = min(end, day_end)

            if interval_start < interval_end:
                total += (interval_end - interval_start).total_seconds()

            current = pd.Timestamp.combine(
                current.date() + timedelta(days=1),
                time(0, 0)
            )

        return total

    def calculate_processing_time(
        self,
        df: pd.DataFrame,
        use_business_time: bool = False
    ) -> pd.DataFrame:
        """
        Calculate processing time for events.
        
        For W_ activities (W_Call after offers, W_Call incomplete files, W_Complete application,
        W_Handle leads, W_Validate application): processing_time is the time between lifecycle
        event "schedule" and "complete" or "ate_abort" for the same activity in the same case.
        Note: schedule always comes before start.
        
        For all other activities: processing_time is the time between consecutive events
        (event n to event n+1) in each case.
        
        Args:
            df: DataFrame with case_id, event, timestamp columns (and lifecycle:transition for W_ activities)
            use_business_time: If True, only count time during business hours (5:00-22:00)

        Returns:
            DataFrame with processing_time column (in hours)
        """
        
        # Define W_ activities that need special handling
        w_activities = [
            "W_Call after offers",
            "W_Call incomplete files",
            "W_Complete application",
            "W_Handle leads",
            "W_Validate application"
        ]
        
        if use_business_time:
            print("Calculating business-time processing times (5:00-22:00)...")
        else:
            print("Calculating processing times...")
            print(f"Special logic for W_ activities: {w_activities}")

        # Sort by case and timestamp
        df = df.sort_values(["case_id", "timestamp"]).copy()
        
        # Check if lifecycle:transition column exists (required for W_ activities)
        has_lifecycle = "lifecycle:transition" in df.columns
        
        if not has_lifecycle:
            print("Warning: lifecycle:transition column not found. Cannot apply special logic for W_ activities.")
            print("Falling back to standard consecutive event logic for all activities.")

        # Separate W_ activities from other activities
        w_mask = df["event"].isin(w_activities)
        df_w = df[w_mask].copy() if w_mask.any() else pd.DataFrame(columns=df.columns)
        df_other = df[~w_mask].copy() if w_mask.any() else df.copy()
        
        w_results = []
        
        # Process W_ activities with special logic: schedule -> complete/ate_abort
        # Note: schedule always comes before start, so we use schedule as the beginning event
        if len(df_w) > 0 and has_lifecycle:
            print(f"Processing {len(df_w)} W_ activity events with special lifecycle logic...")
            
            # Filter to only schedule, complete, and ate_abort lifecycle events for W_ activities
            # Note: schedule always comes before start, so we use schedule as the beginning event
            df_w_lifecycle = df_w[df_w["lifecycle:transition"].isin(["schedule", "complete", "ate_abort"])].copy()
            
            # Group by case_id and event to find schedule -> complete/ate_abort pairs
            for (case_id, event_name), group in df_w_lifecycle.groupby(["case_id", "event"]):
                group = group.sort_values("timestamp").copy()
                
                # Iterate through events in chronological order to match schedule-end pairs
                # Track which end events have been used
                used_end_indices = set()
                
                for idx, row in group.iterrows():
                    lifecycle = row["lifecycle:transition"]
                    
                    if lifecycle == "schedule":
                        schedule_time = row["timestamp"]
                        
                        # Find the first unused complete/ate_abort after this schedule
                        remaining_ends = group[
                            (group["lifecycle:transition"].isin(["complete", "ate_abort"])) &
                            (group["timestamp"] > schedule_time) &
                            (~group.index.isin(used_end_indices))
                        ]
                        
                        if len(remaining_ends) > 0:
                            end_row = remaining_ends.iloc[0]
                            end_time = end_row["timestamp"]
                            
                            # Mark this end event as used
                            used_end_indices.add(end_row.name)
                            
                            # Calculate processing time
                            if use_business_time:
                                time_seconds = self.business_seconds_between(schedule_time, end_time)
                                if time_seconds > 0:
                                    processing_time = time_seconds / 3600
                                else:
                                    continue  # Skip if no business time
                            else:
                                processing_time = (end_time - schedule_time).total_seconds() / 3600
                            
                            # Create result row as a dictionary and append to list
                            result_row = row.to_dict()
                            result_row["processing_time"] = processing_time
                            w_results.append(result_row)
        
        # Convert W_ results to DataFrame if we have any
        df_w_result = pd.DataFrame(w_results) if w_results else pd.DataFrame(columns=df.columns)
        
        # Process other activities with standard logic: consecutive events
        if len(df_other) > 0:
            print(f"Processing {len(df_other)} other activity events with standard consecutive event logic...")
            
            # Calculate next timestamp for consecutive events
            df_other["next_timestamp"] = df_other.groupby("case_id")["timestamp"].shift(-1)
            
            # Remove rows without next timestamp (last event in case)
            df_other = df_other[df_other["next_timestamp"].notna()].copy()
            
            if len(df_other) > 0:
                if use_business_time:
                    # Calculate business-time processing time in hours
                    df_other["time_to_next_event_seconds"] = df_other.apply(
                        lambda r: self.business_seconds_between(
                            r["timestamp"], r["next_timestamp"]
                        ),
                        axis=1
                    )
                    df_other["processing_time"] = df_other["time_to_next_event_seconds"] / 3600
                    df_other = df_other[df_other["time_to_next_event_seconds"] > 0].copy()
                    
                    # Drop intermediate columns
                    if "time_to_next_event_seconds" in df_other.columns:
                        df_other = df_other.drop(columns=["time_to_next_event_seconds"])
                else:
                    # Calculate processing time in hours (total time)
                    df_other["processing_time"] = (
                        df_other["next_timestamp"] - df_other["timestamp"]
                    ).dt.total_seconds() / 3600
                
                # Drop next_timestamp column
                if "next_timestamp" in df_other.columns:
                    df_other = df_other.drop(columns=["next_timestamp"])
        
        # Combine results from W_ activities and other activities
        results_to_combine = []
        if len(df_w_result) > 0:
            results_to_combine.append(df_w_result)
        if len(df_other) > 0:
            results_to_combine.append(df_other)
        
        if results_to_combine:
            df = pd.concat(results_to_combine, ignore_index=True)
        else:
            df = pd.DataFrame(columns=df.columns)
        
        # Sort by case and timestamp for consistency
        if len(df) > 0:
            df = df.sort_values(["case_id", "timestamp"]).copy()
            
            # Check for invalid processing_time values (should not occur - indicates a problem)
            if "processing_time" in df.columns:
                nan_mask = df["processing_time"].isna()
                inf_mask = ~np.isfinite(df["processing_time"])
                negative_mask = df["processing_time"] < 0
                invalid_mask = nan_mask | inf_mask | negative_mask
                
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum()
                    print(f"ERROR: Found {invalid_count} rows with invalid processing_time values:")
                    print(f"  NaN: {nan_mask.sum()}, Inf: {(inf_mask & ~nan_mask).sum()}, Negative: {negative_mask.sum()}")
                    print(f"  This indicates a problem in processing_time calculation!")
                    raise ValueError(f"Invalid processing_time values found. This should not happen - check processing_time calculation logic.")
            
            # Note: Negative processing_time can occur if timestamps are out of order
            # This should be investigated rather than silently removed

        print(f"Total events with processing time: {len(df)}")

        return df

    def filter_events_of_interest(self, df: pd.DataFrame, additional_events: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter DataFrame to only include events of interest.
        Can also include additional events (e.g., fixed activities).

        Args:
            df: Input DataFrame
            additional_events: Optional list of additional events to include (e.g., fixed activities)

        Returns:
            Filtered DataFrame
        """
        events_to_keep = self.events_of_interest.copy()
        if additional_events:
            events_to_keep.extend(additional_events)
            events_to_keep = list(set(events_to_keep))  # Remove duplicates
        
        print(f"Filtering to {len(events_to_keep)} events of interest...")

        original_count = len(df)
        df_filtered = df[df["event"].isin(events_to_keep)].copy()

        print(f"Filtered from {original_count} to {len(df_filtered)} events")

        return df_filtered

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features like hour, weekday, event index.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional temporal features
        """
        print("Adding temporal features...")

        # Event position in case
        df["event_index"] = df.groupby("case_id").cumcount()

        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["second"] = df["timestamp"].dt.second
        df["microsecond"] = df["timestamp"].dt.microsecond
        df["weekday"] = df["timestamp"].dt.weekday
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear

        return df

    def log_transform_target(self, df: pd.DataFrame, target_column: str = "processing_time") -> pd.DataFrame:
        """
        Apply log transformation to target variable.

        Args:
            df: Input DataFrame
            target_column: Name of target column to transform

        Returns:
            DataFrame with log-transformed target
        """
        print("Applying log10 transformation to target variable...")

        # Check for invalid values before transformation (should not occur - indicates a problem)
        nan_mask = df[target_column].isna()
        inf_mask = ~np.isfinite(df[target_column])
        negative_mask = df[target_column] < 0
        invalid_mask = nan_mask | inf_mask | negative_mask
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            print(f"ERROR: Found {invalid_count} rows with invalid {target_column} values before log transformation:")
            print(f"  NaN: {nan_mask.sum()}, Inf: {(inf_mask & ~nan_mask).sum()}, Negative: {negative_mask.sum()}")
            print(f"  This indicates a problem in data processing!")
            raise ValueError(f"Invalid {target_column} values found before log transformation. This should not happen - check data processing pipeline.")

        # Use log10(x+1) to handle zero values and be consistent with log1p
        df["log_processing_time"] = np.log10(df[target_column] + 1)
        
        # Check for NaN/Inf in transformed values (should not occur)
        invalid_mask = ~np.isfinite(df["log_processing_time"])
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            print(f"ERROR: Found {invalid_count} invalid values in log_processing_time after transformation")
            print(f"  This indicates a problem in log transformation!")
            raise ValueError(f"Invalid log_processing_time values after transformation. This should not happen - check log transformation logic.")

        return df


    def save_distribution_to_csv(self, distribution: pd.DataFrame, output_path: str) -> None:
        """
        Save processing time distribution to CSV.

        Args:
            distribution: Distribution DataFrame
            output_path: Output file path
        """
        distribution.to_csv(output_path)
        print(f"Saved distribution to: {output_path}")

    def prepare_features_and_target(
        self,
        df: pd.DataFrame,
        target_column: str = "log_processing_time"
    ) -> tuple:
        """
        Prepare feature matrix X and target vector y for machine learning.

        Args:
            df: Input DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (X, y, case_ids) where X is features, y is target, case_ids for grouping
        """
        print("Preparing features and target...")

        # Get existing case features
        existing_case_features = [
            c for c in self.case_features if c in df.columns
        ]

        # Define feature columns using base_features + case_features
        feature_columns = (
            [f for f in self.base_features if f in df.columns]  # Only include base features that exist
            + existing_case_features
        )

        # Check if all required columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns}")

        # Create feature matrix
        X = df[feature_columns]
        y = df[target_column]

        # Get case IDs for grouped splitting
        case_ids = df["case_id"]

        print(f"Prepared {len(X)} samples with {len(feature_columns)} features")

        return X, y, case_ids
