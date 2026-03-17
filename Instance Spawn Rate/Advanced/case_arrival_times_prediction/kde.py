from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, Sequence

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
    def __init__(
        self,
        kernel: str = "gaussian",
        min_samples: int = 2,
        bandwidth_k_values: Sequence[float] = (1.0,),
        bandwidth_val_ratio: float = 0.2,
    ):
        self.kernel = kernel
        self.min_samples = min_samples
        self.bandwidth_k_values = tuple(k for k in bandwidth_k_values if k > 0)
        if len(self.bandwidth_k_values) == 0:
            self.bandwidth_k_values = (1.0,)
        self.bandwidth_val_ratio = float(min(max(bandwidth_val_ratio, 0.05), 0.5))

    def _fit_kde_with_bandwidth_search(self, diffs: np.ndarray) -> tuple[KernelDensity, float, float]:
        n = len(diffs)
        base_h = _silverman_bandwidth(diffs)

        split = int(n * (1.0 - self.bandwidth_val_ratio))
        split = min(max(split, self.min_samples), n - 1)
        train = diffs[:split]
        val = diffs[split:]

        best_h = max(base_h, 1e-6)
        best_nll = float("inf")
        best_model: Optional[KernelDensity] = None

        for k in self.bandwidth_k_values:
            h = max(base_h * float(k), 1e-6)
            kde = KernelDensity(kernel=self.kernel, bandwidth=h)
            kde.fit(train.reshape(-1, 1))
            nll = float(-kde.score(val.reshape(-1, 1)) / len(val))
            if nll < best_nll:
                best_nll = nll
                best_h = h
                best_model = kde

        if best_model is None:
            best_model = KernelDensity(kernel=self.kernel, bandwidth=best_h)
            best_model.fit(train.reshape(-1, 1))
            best_nll = float(-best_model.score(val.reshape(-1, 1)) / len(val))
        return best_model, best_h, best_nll

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

                    if len(diffs) > self.min_samples:
                        kde, h, val_nll = self._fit_kde_with_bandwidth_search(diffs)
                    else:
                        h = _silverman_bandwidth(diffs)
                        kde = KernelDensity(kernel=self.kernel, bandwidth=h)
                        kde.fit(diffs.reshape(-1, 1))
                        val_nll = float("nan")

                    models[(j, k, l + 1)] = kde
                    info[(j, k, l + 1)] = {
                        "n_arrivals": n_arrivals,
                        "n_inters": len(diffs),
                        "bandwidth": h,
                        "val_nll": val_nll,
                    }

        return KDETrainingResult(last_diffs=last_diffs, models=models, info=info)
