import pandas as pd
from datetime import datetime, timedelta


class ResourceAvailabilityModel:
    """
    Basic model for resource availabilities:
    - 2-week interval as repeating calendar pattern (14 days)
    - All resources share the same working pattern (e.g. Mon–Fri, 08:00–17:00 in both weeks)
    """

    def __init__(
        self,
        event_log_df: pd.DataFrame,
        interval_days: int = 14,
        workday_start_hour: int = 8,
        workday_end_hour: int = 17,
        working_cycle_days=None,
    ):
        self.event_log_df = event_log_df.copy()
        self.interval_days = interval_days
        self.workday_start_hour = workday_start_hour
        self.workday_end_hour = workday_end_hour

        # Default: Mon–Fri in both weeks
        if working_cycle_days is None:
            working_cycle_days = {0, 1, 2, 3, 4, 7, 8, 9, 10, 11}
        self.working_cycle_days = working_cycle_days

        # Ensure proper datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.event_log_df["time:timestamp"]):
            self.event_log_df["time:timestamp"] = pd.to_datetime(
                self.event_log_df["time:timestamp"], errors="coerce"
            )

        # Use the Monday of the week of the first timestamp as cycle anchor
        first_ts = self.event_log_df["time:timestamp"].min()
        first_date = first_ts.normalize().date()
        monday_date = first_date - timedelta(days=first_ts.weekday())  # Mon=0

        self.cycle_start_date = monday_date

        # Resources in the log
        self.resources = sorted(self.event_log_df["org:resource"].dropna().unique())

    def _cycle_day_index(self, current_time: datetime) -> int:
        """
        Map current_time to a day index in the 2-week cycle [0..interval_days-1].
        The cycle repeats infinitely.
        """
        delta_days = (current_time.date() - self.cycle_start_date).days
        return delta_days % self.interval_days

    def is_working_time(self, current_time: datetime) -> bool:
        """
        Check if current_time falls on a working day in the cycle
        and within the working hours.
        """
        cycle_day = self._cycle_day_index(current_time)
        if cycle_day not in self.working_cycle_days:
            return False

        hour = current_time.hour
        return self.workday_start_hour <= hour < self.workday_end_hour

    def is_available(self, resource_id: str, current_time: datetime) -> bool:
        """
        Basic: all resources share the same 2-week calendar.
        """
        if resource_id not in self.resources:
            return False

        return self.is_working_time(current_time)
