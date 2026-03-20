import os
import pm4py
import numpy as np
import pandas as pd
from datetime import datetime


LOG_FILE = os.path.join("Dataset", "BPI Challenge 2017.xes")
LAMBDA = 0.000997770693997961  # pre-fitted value (MLE: 1/mean_interarrival)


def fit_static_exponential_distribution():
    """Re-fit λ from the event log (offline estimator)."""
    global LAMBDA
    log = pm4py.read_xes(LOG_FILE)
    df = pm4py.convert_to_dataframe(log)

    case_starts = (
        df.groupby('case:concept:name')['time:timestamp']
        .min()
        .sort_values()
        .reset_index(name='start_time')
    )

    # Interarrival times in seconds (first row has no predecessor → NaN)
    case_starts['interarrival_time'] = case_starts['start_time'].diff().dt.total_seconds()

    # Drop the first NaN row (positional, not label-based)
    interarrivals = case_starts['interarrival_time'].iloc[1:]

    LAMBDA = 1 / interarrivals.mean()
    print(f"Estimated lambda (rate): {LAMBDA} cases/sec")
    print(f"Mean interarrival time:  {1/LAMBDA:.1f} sec  ({1/LAMBDA/60:.1f} min)")


def next_case_time_lambda(current_time):
    """Sample the next case arrival time from Exp(λ)."""
    delta = np.random.exponential(scale=1 / LAMBDA)
    return current_time + pd.Timedelta(seconds=delta)


if __name__ == "__main__":
    fit_static_exponential_distribution()
    print(f"\nSample next arrival: {next_case_time_lambda(datetime.now())}")
