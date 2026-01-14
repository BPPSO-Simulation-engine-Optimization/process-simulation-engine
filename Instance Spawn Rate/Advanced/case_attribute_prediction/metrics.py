from __future__ import annotations
import numpy as np


def ks_statistic_1d(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")

    x = np.sort(x)
    y = np.sort(y)
    data_all = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, data_all, side="right") / x.size
    cdf_y = np.searchsorted(y, data_all, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def wasserstein_approx_1d(x, y, grid: int = 1000) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")

    qs = np.linspace(0.0, 1.0, grid)
    xq = np.quantile(x, qs)
    yq = np.quantile(y, qs)
    return float(np.mean(np.abs(xq - yq)))
