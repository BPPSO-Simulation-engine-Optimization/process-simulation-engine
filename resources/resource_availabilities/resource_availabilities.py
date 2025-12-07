import pandas as pd
from datetime import datetime, timedelta


class ResourceAvailabilityModel:
    """
    Basic model for resource availabilities:
    - Uses a two-week interval starting from the first timestamp in the log
    - All resources share the same working hours (e.g. Mon–Fri, 08:00–17:00)
    """

    def __init__(
        self,
        event_log_df: pd.DataFrame,
        interval_days: int = 14,
        workday_start_hour: int = 8,
        workday_end_hour: int = 17,
        working_weekdays=None,
    ):
        if working_weekdays is None:
            # 0 = Monday, ..., 6 = Sunday
            working_weekdays = {0, 1, 2, 3, 4}

        self.event_log_df = event_log_df.copy()
        self.interval_days = interval_days
        self.workday_start_hour = workday_start_hour
        self.workday_end_hour = workday_end_hour
        self.working_weekdays = working_weekdays

        # wir brauchen echte Datumswerte
        if not pd.api.types.is_datetime64_any_dtype(self.event_log_df["time:timestamp"]):
            self.event_log_df["time:timestamp"] = pd.to_datetime(
                self.event_log_df["time:timestamp"], errors="coerce"
            )

        # Simulationshorizont: zwei Wochen ab dem ersten Event
        self.start_time: datetime = self.event_log_df["time:timestamp"].min().normalize()
        self.end_time: datetime = self.start_time + timedelta(days=self.interval_days)

        # Menge der Ressourcen aus dem Log
        self.resources = sorted(self.event_log_df["org:resource"].dropna().unique())

    def is_within_interval(self, current_time: datetime) -> bool:
        """Check if current_time is inside the global availability interval."""
        return self.start_time <= current_time < self.end_time

    def is_working_time(self, current_time: datetime) -> bool:
        """Check if current_time is during working hours on a working day."""
        weekday = current_time.weekday()
        hour = current_time.hour
        if weekday not in self.working_weekdays:
            return False
        return self.workday_start_hour <= hour < self.workday_end_hour

    def is_available(self, resource_id: str, timestamp: datetime) -> bool:
        """
        Basic model: all resources share the same calendar.
        You could extend this later with resource-specific calendars.
        """
        if resource_id not in self.resources:
            # unbekannte Ressource -> nicht verfügbar
            return False

        if not self.is_within_interval(timestamp):
            return False

        return self.is_working_time(timestamp)

