import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe

from resources.resource_availabilities import ResourceAvailabilityModel

LOG_FILE = r"Dataset\BPI Challenge 2017.xes"
print(">>> TEST STARTED")


def main():
    print("Loading XES log...")
    log = xes_importer.apply(LOG_FILE)

    # Convert pm4py event log to pandas DataFrame
    df = convert_to_dataframe(log)

    print("Number of events:", len(df))
    print("Columns:", df.columns.tolist()[:10], "...")

    # Basic validation
    assert "org:resource" in df.columns, "Missing column: org:resource"
    assert "time:timestamp" in df.columns, "Missing column: time:timestamp"

    model = ResourceAvailabilityModel(df)

    print("Number of resources:", len(model.resources))
    print("First resource:", model.resources[0] if model.resources else "None")

    example_time = model.start_time.replace(hour=10, minute=0, second=0)
    example_resource = model.resources[0]

    print("Simulation start:", model.start_time)
    print("Simulation end:", model.end_time)
    print(
        f"Is resource '{example_resource}' available at {example_time}? ->",
        model.is_available(example_resource, example_time),
    )

    # Sunday test (should be False)
    sunday_time = model.start_time
    while sunday_time.weekday() != 6:
        sunday_time += pd.Timedelta(days=1)
    sunday_time = sunday_time.replace(hour=10)

    print(
        f"Is resource '{example_resource}' available on Sunday {sunday_time}? ->",
        model.is_available(example_resource, sunday_time),
    )


if __name__ == "__main__":
    print(">>> MAIN EXECUTED")
    main()