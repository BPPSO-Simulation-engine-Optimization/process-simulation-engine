from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from .preprocessing import DayArrivals


def compute_statistics_for_days(days: List[DayArrivals]) -> np.ndarray | None:
    daily_counts = np.array([len(day) for day in days], dtype=float)
    if len(daily_counts) == 0:
        return None

    avg_daily = daily_counts.mean()
    p25_daily, p75_daily = np.percentile(daily_counts, [25, 75])

    timestamps = [ts for day in days for ts in day]
    if len(timestamps) < 2:
        std_ia = 0.0
        p25_ia = 0.0
        p75_ia = 0.0
    else:
        ts_series = pd.to_datetime(timestamps)
        if ts_series.tz is not None:
            ts_series = ts_series.tz_localize(None)
        arr = np.sort(np.array(ts_series, dtype="datetime64[ns]"))
        diffs = np.diff(arr).astype("timedelta64[s]").astype(float) / 3600.0
        std_ia = np.std(diffs)
        p25_ia, p75_ia = np.percentile(diffs, [25, 75])

    return np.array([avg_daily, p25_daily, p75_daily, std_ia, p25_ia, p75_ia])


class WeekdayClusterer:
    """
    Entspricht deiner cluster_weekdays(G, dates).
    """
    def cluster(self, G: List[list], dates=None) -> Dict[int, List[List[DayArrivals]]]:
        W: Dict[int, List[List[DayArrivals]]] = {}

        for j, Dj in enumerate(G, start=1):
            if len(Dj) == 0:
                W[j] = []
                continue

            first = Dj[0]

            # Fall 1: Dj ist eine Liste von Tagen (Fallback G=[D])
            if isinstance(first, list) and (len(first) == 0 or isinstance(first[0], (pd.Timestamp, np.datetime64))):
                Dj_days = Dj
            else:
                # Fall 2: Dj ist Liste von Segmenten -> flatten
                Dj_days = [day for segment in Dj for day in segment]

            T_w = {w: [] for w in range(1, 8)}
            for day in Dj_days:
                if len(day) == 0:
                    continue
                ts0 = pd.to_datetime(day[0])
                w = ts0.weekday() + 1
                T_w[w].append(day)

            F = []
            valid_weekdays = []
            for w in range(1, 8):
                if len(T_w[w]) > 0:
                    stats = compute_statistics_for_days(T_w[w])
                    F.append(stats)
                    valid_weekdays.append(w)

            if len(valid_weekdays) == 0:
                W[j] = [Dj_days]
                continue

            F = np.array(F)

            if len(valid_weekdays) > 1:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1e-6,
                    linkage="ward"
                )
                weekday_labels = model.fit_predict(F)
            else:
                weekday_labels = np.array([0])

            unique_labels = np.unique(weekday_labels)
            label_map = {lab: i for i, lab in enumerate(unique_labels)}
            Kj = len(unique_labels)

            Wj: List[List[DayArrivals]] = [[] for _ in range(Kj)]
            for w, lab in zip(valid_weekdays, weekday_labels):
                cid = label_map[lab]
                Wj[cid].extend(T_w[w])

            empty_weekdays = [w for w in range(1, 8) if len(T_w[w]) == 0]
            if len(empty_weekdays) > 0:
                Wj.append([])

            W[j] = Wj

        return W
