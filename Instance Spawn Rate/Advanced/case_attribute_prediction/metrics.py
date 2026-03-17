from __future__ import annotations

import numpy as np


def ks_statistic_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Zweistichproben-KS-Statistik (empirische CDF-Differenz)."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    combined = np.concatenate([x, y])
    combined = np.unique(combined)
    cdf_x = np.searchsorted(x, combined, side="right") / len(x)
    cdf_y = np.searchsorted(y, combined, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def wasserstein_approx_1d(x: np.ndarray, y: np.ndarray, n_quantiles: int = 1000) -> float:
    """Näherung der Wasserstein-1-Distanz über Quantile."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, n_quantiles)
    qx = np.quantile(x, q)
    qy = np.quantile(y, q)
    return float(np.mean(np.abs(qx - qy)))
