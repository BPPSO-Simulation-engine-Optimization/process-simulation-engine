from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

from .preprocessing import DailySequence


def flatten_days(D: DailySequence) -> List[pd.Timestamp]:
    return [pd.to_datetime(ts) for day in D for ts in day]


def cadd_distance(arrivals_true, arrivals_sim) -> float:
    arrivals_true = [pd.to_datetime(ts) for ts in arrivals_true]
    arrivals_sim = [pd.to_datetime(ts) for ts in arrivals_sim]

    if len(arrivals_true) == 0 and len(arrivals_sim) == 0:
        return 0.0

    all_ts = arrivals_true + arrivals_sim
    ref_start = min(all_ts)

    def to_hour_bins(timestamps):
        if len(timestamps) == 0:
            return np.array([], dtype=int)
        ts_series = pd.to_datetime(timestamps)
        deltas = ts_series - ref_start
        hours = np.floor(deltas.total_seconds() / 3600.0).astype(int)
        return hours

    bins_true = to_hour_bins(arrivals_true)
    bins_sim = to_hour_bins(arrivals_sim)

    if len(bins_true) == 0 or len(bins_sim) == 0:
        return 0.0

    max_bin = max(bins_true.max(), bins_sim.max())
    n_bins = max_bin + 1

    counts_true = np.bincount(bins_true, minlength=n_bins).astype(float)
    counts_sim = np.bincount(bins_sim, minlength=n_bins).astype(float)

    total_true = counts_true.sum()
    total_sim = counts_sim.sum()
    if total_true == 0 or total_sim == 0:
        return 0.0

    p = counts_true / total_true
    q = counts_sim / total_sim

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    emd = float(np.sum(np.abs(cdf_p - cdf_q)))
    return emd


def sqrt_cadd(D_true: DailySequence, D_sim: DailySequence) -> float:
    arr_true = flatten_days(D_true)
    arr_sim = flatten_days(D_sim)
    cadd = cadd_distance(arr_true, arr_sim)
    return float(np.sqrt(cadd))


@dataclass(frozen=True)
class InterarrivalStats:
    mean: float
    std: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float


def interarrival_statistics_from_timestamps(timestamps: List[pd.Timestamp], unit: str = "seconds") -> InterarrivalStats:
    """
    Berechnet Interarrival-Statistiken aus einer flachen Timestamp-Liste.
    unit: "seconds" (default) oder "hours".
    """
    ts = sorted(pd.to_datetime(timestamps))
    if len(ts) < 2:
        return InterarrivalStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.array(ts, dtype="datetime64[ns]")
    diffs_sec = np.diff(arr).astype("timedelta64[s]").astype(float)
    diffs_sec = diffs_sec[diffs_sec > 0]

    if len(diffs_sec) == 0:
        return InterarrivalStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if unit == "hours":
        diffs = diffs_sec / 3600.0
    else:
        diffs = diffs_sec

    q05, q25, q50, q75, q95 = np.percentile(diffs, [5, 25, 50, 75, 95])

    return InterarrivalStats(
        mean=float(np.mean(diffs)),
        std=float(np.std(diffs)),
        q05=float(q05),
        q25=float(q25),
        q50=float(q50),
        q75=float(q75),
        q95=float(q95),
    )