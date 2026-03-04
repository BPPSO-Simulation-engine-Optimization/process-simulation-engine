from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional


@dataclass(frozen=True)
class SimulationConfig:
    # Temporaler Hold-out auf Case-Ebene (erste train_ratio Cases train, Rest test)
    train_ratio: float = 0.8

    # Step 1: Global Segmentation
    window_size: int = 10
    kmax: int = 6
    z_values: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1)

    # Step 3: Intraday Bins
    L: int = 5

    # KDE
    kernel: str = "gaussian"
    min_samples_kde: int = 2
    bandwidth_k_values: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
    bandwidth_val_ratio: float = 0.3

    # DBSCAN
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 2

    # Logging/Debug
    verbose: bool = False

    # Reproduzierbarkeit (KernelDensity.sample hat random_state Parameter)
    random_state: Optional[int] = None
