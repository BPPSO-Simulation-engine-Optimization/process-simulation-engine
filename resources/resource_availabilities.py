import pandas as pd
from datetime import datetime, timedelta
import holidays  # neu

class ResourceAvailabilityModel:
    """
    Advanced model for resource availabilities:
    - Mon–Fri, 08:00–17:00
    - Plus Dutch public holidays as non-working days
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

        if working_cycle_days is None:
            working_cycle_days = {0, 1, 2, 3, 4, 7, 8, 9, 10, 11}
        self.working_cycle_days = working_cycle_days

        if not pd.api.types.is_datetime64_any_dtype(self.event_log_df["time:timestamp"]):
            self.event_log_df["time:timestamp"] = pd.to_datetime(
                self.event_log_df["time:timestamp"], errors="coerce"
            )

        first_ts = self.event_log_df["time:timestamp"].min()
        first_date = first_ts.normalize().date()
        monday_date = first_date - timedelta(days=first_ts.weekday())  # Mon=0
        self.cycle_start_date = monday_date

        # Dutch public holidays (for a range of years, z.B. 2015–2030)
        years = range(self.cycle_start_date.year, self.cycle_start_date.year + 20)
        self.nl_holidays = set(holidays.Netherlands(years=years).keys())

        self.resources = sorted(self.event_log_df["org:resource"].dropna().unique())

    def _cycle_day_index(self, current_time: datetime) -> int:
        delta_days = (current_time.date() - self.cycle_start_date).days
        return delta_days % self.interval_days

    def is_working_time(self, current_time: datetime) -> bool:
        """
        Check if current_time falls on:
        - a working day in the 2-week cycle
        - within the working hours
        - and is not a Dutch public holiday
        """
        current_date = current_time.date()

        # 1) Feiertag in NL -> nie arbeiten
        if current_date in self.nl_holidays:
            return False

        # 2) 2-Wochen-Muster (Mo–Fr in beiden Wochen)
        cycle_day = self._cycle_day_index(current_time)
        if cycle_day not in self.working_cycle_days:
            return False

        # 3) Uhrzeit
        hour = current_time.hour
        return self.workday_start_hour <= hour < self.workday_end_hour

    def is_available(self, resource_id: str, current_time: datetime) -> bool:
        if resource_id not in self.resources:
            return False
        return self.is_working_time(current_time)
