import pm4py
import pandas as pd

LOG_FILE = r"Dataset\BPI Challenge 2017.xes"

def prefix_generation():
    """
    Generates all  prefixes from an event log
    """
    log = pm4py.read_xes(LOG_FILE) 
    event_log = pm4py.convert_to_event_log(log)

    prefixes = []
    labels = []

    for trace in event_log:
        activities = [event for event in trace]
        for i in range(1, len(activities)):
            prefix = activities[:i]
            label = activities[i]
            prefixes.append(prefix)
            labels.append(label)

    df = pd.DataFrame({
        "prefix": prefixes,
        "next_activity": labels
    })

    df.to_csv(r'Dataset\Prefix_Label_Data.csv', index=False)




if __name__ == "__main__":
    prefix_generation()