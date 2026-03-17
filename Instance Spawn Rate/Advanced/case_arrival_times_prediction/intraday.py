from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocessing import DayArrivals


@dataclass(frozen=True)
class IntradayBounds:
    """Working-hour bounds as seconds since midnight (Paper: DetermineBounds)."""
    lower: float  # earliest observed arrival (seconds since midnight)
    upper: float  # latest observed arrival (seconds since midnight)


def determine_bounds(weekday_clusters: Dict[int, List[List[DayArrivals]]]) -> IntradayBounds:
    """
    Paper Algorithm 3 – DetermineBounds(D):
    Scan all observed arrival timestamps and return the earliest / latest
    time-of-day (as seconds since midnight).  These define the working-hour
    window within which time-bins are created.
    """
    earliest = float("inf")
    latest = float("-inf")

    for Wj in weekday_clusters.values():
        for Wjk in Wj:
            for day in Wjk:
                for ts in day:
                    ts = pd.to_datetime(ts)
                    midnight = ts.normalize()
                    sec = (ts - midnight).total_seconds()
                    if sec < earliest:
                        earliest = sec
                    if sec > latest:
                        latest = sec

    if earliest == float("inf") or latest == float("-inf"):
        # No data at all → fall back to full day
        return IntradayBounds(lower=0.0, upper=24 * 60 * 60)

    # Small safety margin: don't let lower==upper (would give zero-length bins)
    if latest <= earliest:
        latest = earliest + 1.0

    return IntradayBounds(lower=earliest, upper=latest)


class IntradayBinner:
    """
    Paper Algorithm 3 – CreateTimeBins(lower_time, upper_time, L):
    Splits each day's arrivals into L equal-width bins within the observed
    working-hour window [lower, upper], NOT over the full 00:00–24:00 range.
    """
    def bin(
        self,
        weekday_clusters: Dict[int, List[List[DayArrivals]]],
        L: int,
        bounds: Optional[IntradayBounds] = None,
    ) -> Tuple[Dict[int, List[List[List[list]]]], IntradayBounds]:
        """
        Bin arrivals into L intraday bins within the working-hour bounds.

        Args:
            weekday_clusters: Output of WeekdayClusterer.cluster().
            L: Number of intraday bins.
            bounds: Pre-computed bounds. If None, DetermineBounds is called
                    automatically from the data.

        Returns:
            Tuple of (intraday_binned dict, bounds used).
        """
        if bounds is None:
            bounds = determine_bounds(weekday_clusters)

        bin_length = (bounds.upper - bounds.lower) / L

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

                        # Map into the bounds window
                        offset = seconds_since_midnight - bounds.lower
                        if offset < 0:
                            # Arrival before working hours → first bin
                            idx = 0
                        elif offset >= (bounds.upper - bounds.lower):
                            # Arrival after working hours → last bin
                            idx = L - 1
                        else:
                            idx = int(offset // bin_length)
                            if idx >= L:
                                idx = L - 1

                        bins[idx].append(ts)

                    days_binned.append(bins)

                Wj_binned.append(days_binned)

            intraday_binned[j] = Wj_binned

        return intraday_binned, bounds
