from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


DayArrivals = List[pd.Timestamp]          # ein Tag -> Liste von Timestamps
DailySequence = List[DayArrivals]        # D -> Liste von Tagen


@dataclass(frozen=True)
class DailyArrivalsResult:
    first_events: pd.DataFrame
    D: DailySequence
    dates: List[pd.Timestamp]
    M: np.ndarray


class DailyArrivalBuilder:
    """
    Entspricht deiner build_daily_arrivals(df).
    """
    def build(self, df: pd.DataFrame) -> DailyArrivalsResult:
        if "time:timestamp" not in df.columns or "case:concept:name" not in df.columns:
            raise KeyError("df muss Spalten 'time:timestamp' und 'case:concept:name' enthalten.")

        df = df.copy()
        # Ensure we always end up with tz-naive timestamps (UTC-based) regardless of
        # whether the input is a Series or a DatetimeIndex (older pandas may not
        # expose .dt on DatetimeIndex).
        ts = pd.to_datetime(df["time:timestamp"])
        # Series path (has .dt accessor)
        if hasattr(ts, "dt"):
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            # DatetimeIndex or array-like with .tz attribute
            if getattr(ts, "tz", None) is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        df["time:timestamp"] = ts

        first_events = (
            df.sort_values("time:timestamp")
              .groupby("case:concept:name", as_index=False)["time:timestamp"]
              .first()
        )
        first_events = first_events.sort_values("time:timestamp", kind="mergesort").reset_index(drop=True)

        if first_events.empty:
            raise ValueError("Keine Events gefunden (first_events ist leer).")

        first_events["date"] = first_events["time:timestamp"].dt.floor("D")
        grouped = first_events.groupby("date")["time:timestamp"].apply(list)

        all_dates = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")

        D: DailySequence = []
        dates: List[pd.Timestamp] = []

        for d in all_dates:
            d = pd.to_datetime(d)
            dates.append(d)
            if d in grouped.index:
                D.append(sorted(pd.to_datetime(grouped.loc[d]).tolist()))
            else:
                D.append([])

        M = np.array([len(ti) for ti in D], dtype=float)
        return DailyArrivalsResult(first_events=first_events, D=D, dates=dates, M=M)
