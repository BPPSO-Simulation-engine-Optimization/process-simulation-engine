from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def _silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0

    std = np.std(x)
    if std == 0:
        return max(x.mean() * 0.1, 1e-6)

    h = 1.06 * std * n ** (-1.0 / 5.0)
    return max(h, 1e-6)


@dataclass(frozen=True)
class KDETrainingResult:
    last_diffs: np.ndarray
    models: Dict[Tuple[int, int, int], Optional[KernelDensity]]
    info: Dict[Tuple[int, int, int], Dict[str, Any]]


class InterarrivalKDETrainer:
    """
    Entspricht deiner learn_interarrival_kde(intraday_binned, ...).
    Trainiert KDE auf Interarrivals in Sekunden (wie im Notebook).
    """
    def __init__(self, kernel: str = "gaussian", min_samples: int = 2):
        self.kernel = kernel
        self.min_samples = min_samples

    def fit(self, intraday_binned: dict, L: int) -> KDETrainingResult:
        models: Dict[Tuple[int, int, int], Optional[KernelDensity]] = {}
        info: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        last_diffs = np.array([], dtype=float)

        for j, Wj_binned in intraday_binned.items():
            for k, days_binned in enumerate(Wj_binned, start=1):

                if len(days_binned) == 0:
                    for l in range(1, L + 1):
                        models[(j, k, l)] = None
                        info[(j, k, l)] = {"n_arrivals": 0, "n_inters": 0, "bandwidth": None}
                    continue

                for l in range(L):
                    diffs_all = []
                    n_arrivals = 0

                    for day_bins in days_binned:
                        if l >= len(day_bins):
                            continue

                        ts_day = sorted(day_bins[l])
                        n_arrivals += len(ts_day)

                        if len(ts_day) < 2:
                            continue

                        ts_series = pd.to_datetime(ts_day)
                        if ts_series.tz is not None:
                            ts_series = ts_series.tz_localize(None)
                        arr_day = np.array(ts_series, dtype="datetime64[ns]")
                        diffs_day = np.diff(arr_day).astype("timedelta64[s]").astype(float)
                        diffs_day = diffs_day[diffs_day > 0]

                        if len(diffs_day) > 0:
                            diffs_all.extend(diffs_day)

                    diffs = np.array(diffs_all, dtype=float)
                    if len(diffs) > 0:
                        last_diffs = diffs

                    if len(diffs) < self.min_samples:
                        models[(j, k, l + 1)] = None
                        info[(j, k, l + 1)] = {
                            "n_arrivals": n_arrivals,
                            "n_inters": len(diffs),
                            "bandwidth": None,
                        }
                        continue

                    h = _silverman_bandwidth(diffs)
                    kde = KernelDensity(kernel=self.kernel, bandwidth=h)
                    kde.fit(diffs.reshape(-1, 1))

                    models[(j, k, l + 1)] = kde
                    info[(j, k, l + 1)] = {
                        "n_arrivals": n_arrivals,
                        "n_inters": len(diffs),
                        "bandwidth": h,
                    }

        return KDETrainingResult(last_diffs=last_diffs, models=models, info=info)
