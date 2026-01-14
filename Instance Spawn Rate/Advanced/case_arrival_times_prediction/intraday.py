from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .preprocessing import DayArrivals


class IntradayBinner:
    """
    Entspricht deiner intraday_binning(weekday_clusters, L).
    """
    def bin(self, weekday_clusters: Dict[int, List[List[DayArrivals]]], L: int) -> Dict[int, List[List[List[list]]]]:
        seconds_per_day = 24 * 60 * 60
        bin_length = seconds_per_day / L

        intraday_binned = {}

        for j, Wj in weekday_clusters.items():
            Wj_binned = []

            for Wjk in Wj:
                days_binned = []

                for day in Wjk:
                    bins = [[] for _ in range(L)]
                    for ts in day:
                        ts = pd.to_datetime(ts)
                        midnight = ts.normalize()
                        seconds_since_midnight = (ts - midnight).total_seconds()
                        idx = int(seconds_since_midnight // bin_length)
                        if idx >= L:
                            idx = L - 1
                        bins[idx].append(ts)

                    days_binned.append(bins)

                Wj_binned.append(days_binned)

            intraday_binned[j] = Wj_binned

        return intraday_binned
