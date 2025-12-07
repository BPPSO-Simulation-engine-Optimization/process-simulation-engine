import pandas as pd
from datetime import datetime, time, timedelta
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe

from resources.resource_availabilities import ResourceAvailabilityModel

LOG_FILE = r"Dataset\BPI Challenge 2017.xes"
print(">>> TEST STARTED")


def print_check(model: ResourceAvailabilityModel, resource_id: str, label: str, dt: datetime):
    cycle_day = model._cycle_day_index(dt)
    weekday = dt.weekday()  # 0=Mon,...,6=Sun
    print(
        f"[{label}] dt={dt} | weekday={weekday} | cycle_day={cycle_day} "
        f"| available={model.is_available(resource_id, dt)}"
    )


def main():
    print("Loading XES log...")
    log = xes_importer.apply(LOG_FILE)
    df = convert_to_dataframe(log)

    print("Number of events:", len(df))
    print("Columns:", df.columns.tolist()[:10], "...")

    assert "org:resource" in df.columns, "Missing column: org:resource"
    assert "time:timestamp" in df.columns, "Missing column: time:timestamp"

    model = ResourceAvailabilityModel(df)

    print("Number of resources:", len(model.resources))
    example_resource = model.resources[0] if model.resources else None
    print("First resource:", example_resource)

    if example_resource is None:
        print("No resources found, aborting test.")
        return

    print("Cycle start date (Monday anchor):", model.cycle_start_date)

    # Base datetime = cycle_start_date 00:00
    base_dt = datetime.combine(model.cycle_start_date, time(0, 0))

    # 1) Weekday in working hours (should be True)
    weekday_10 = base_dt.replace(hour=10)  # Monday 10:00
    print_check(model, example_resource, "Weekday 10:00 (expected True)", weekday_10)

    # 2) Weekday before working hours (should be False)
    weekday_07 = base_dt.replace(hour=7)
    print_check(model, example_resource, "Weekday 07:00 (expected False)", weekday_07)

    # 3) Weekday at boundary 16:00 (still inside, expected True)
    weekday_16 = base_dt.replace(hour=16)
    print_check(model, example_resource, "Weekday 16:00 (expected True)", weekday_16)

    # 4) Weekday after working hours (18:00, expected False)
    weekday_18 = base_dt.replace(hour=18)
    print_check(model, example_resource, "Weekday 18:00 (expected False)", weekday_18)

    # 5) Saturday 10:00 (expected False)
    saturday_dt = base_dt
    while saturday_dt.weekday() != 5:  # 5 = Saturday
        saturday_dt += timedelta(days=1)
    saturday_10 = saturday_dt.replace(hour=10)
    print_check(model, example_resource, "Saturday 10:00 (expected False)", saturday_10)

    # 6) Saturday 16:00 (expected False)
    saturday_16 = saturday_dt.replace(hour=16)
    print_check(model, example_resource, "Saturday 16:00 (expected False)", saturday_16)

    # 7) Sunday 10:00 (expected False)
    sunday_dt = base_dt
    while sunday_dt.weekday() != 6:  # 6 = Sunday
        sunday_dt += timedelta(days=1)
    sunday_10 = sunday_dt.replace(hour=10)
    print_check(model, example_resource, "Sunday 10:00 (expected False)", sunday_10)

    # 8) Sunday 07:00 (expected False)
    sunday_07 = sunday_dt.replace(hour=7)
    print_check(model, example_resource, "Sunday 07:00 (expected False)", sunday_07)

    # 9) Far in the future: some Monday in 2026 at 10:00 (should still follow pattern -> True)
    future_monday = datetime(2026, 5, 18, 10, 0)  # 18.05.2026 ist ein Montag
    print_check(model, example_resource, "Future Monday 2026-05-18 10:00 (expected True)", future_monday)

    # 10) Future Saturday in 2026 at 10:00 (should be False)
    future_saturday = datetime(2026, 5, 16, 10, 0)  # 16.05.2026 ist ein Samstag
    print_check(model, example_resource, "Future Saturday 2026-05-16 10:00 (expected False)", future_saturday)

    # 11) Future Sunday in 2026 at 16:00 (should be False)
    future_sunday = datetime(2026, 5, 17, 16, 0)  # 17.05.2026 ist ein Sonntag
    print_check(model, example_resource, "Future Sunday 2026-05-17 16:00 (expected False)", future_sunday)

    # 12) Future weekday at 07:00 (should be False)
    future_weekday_07 = datetime(2026, 5, 19, 7, 0)  # Tuesday
    print_check(model, example_resource, "Future Weekday 07:00 (expected False)", future_weekday_07)

    kings_day_2016 = datetime(2016, 4, 27, 10, 0)
    print_check(model, example_resource, "King's Day 2016 (expected False)", kings_day_2016)



if __name__ == "__main__":
    print(">>> MAIN EXECUTED")
    main()
