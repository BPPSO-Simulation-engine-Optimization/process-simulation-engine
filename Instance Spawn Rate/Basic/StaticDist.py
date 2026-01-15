import pm4py
import numpy as np
import pandas as pd
from datetime import datetime


LOG_FILE = r"Dataset\BPI Challenge 2017.xes"
LAMBDA = 0.000997770693997961

def fit_static_exponential_distribution():
    log = pm4py.read_xes(LOG_FILE) 
    
    df = pm4py.convert_to_dataframe(log)

    case_starts = df.groupby('case:concept:name')['time:timestamp'].min().reset_index()
    case_starts.rename(columns={'time:timestamp': 'start_time'}, inplace=True)
    case_starts.sort_values('start_time', inplace=True)

    case_starts['interarrival_time'] = case_starts['start_time'].diff().dt.total_seconds().fillna(0)
    print(case_starts[['case:concept:name', 'start_time', 'interarrival_time']].head(10))
    interarrivals = case_starts['interarrival_time'][1:]  # drop first NaN
    print(interarrivals.head())

    LAMBDA = 1 / interarrivals.mean()
    print(f"Estimated lambda (rate): {LAMBDA} cases/sec")

def next_case_time_lambda(current_time):
    delta = np.random.exponential(scale=1/LAMBDA)
    return current_time + pd.Timedelta(seconds=delta)


if __name__ == "__main__":
    print(next_case_time_lambda(datetime.now()))