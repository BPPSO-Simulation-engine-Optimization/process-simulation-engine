from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .preprocessing import DayArrivals


class WeekdayClusterMapper:
    """
    Entspricht deiner build_weekday_cluster_mapping(weekday_clusters).
    """
    def build(self, weekday_clusters: Dict[int, List[List[DayArrivals]]]) -> Dict[Tuple[int, int], Optional[int]]:
        mapping: Dict[Tuple[int, int], Optional[int]] = {}

        for j, Wj in weekday_clusters.items():
            counts = {w: {} for w in range(1, 8)}

            for k, days in enumerate(Wj, start=1):
                for day in days:
                    if len(day) == 0:
                        continue
                    ts0 = pd.to_datetime(day[0])
                    w = ts0.weekday() + 1
                    counts[w][k] = counts[w].get(k, 0) + 1

            for w in range(1, 8):
                if len(counts[w]) == 0:
                    mapping[(j, w)] = None
                else:
                    k_best = max(counts[w].items(), key=lambda x: x[1])[0]
                    mapping[(j, w)] = k_best

        return mapping
