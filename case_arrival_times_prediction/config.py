from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional


@dataclass(frozen=True)
class SimulationConfig:
    # Train/Test auf Tagesebene
    train_ratio: float = 0.8

    # Step 1: Global Segmentation
    window_size: int = 14
    kmax: int = 3
    z_values: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2)

    # Step 3: Intraday Bins
    L: int = 1

    # KDE
    kernel: str = "gaussian"
    min_samples_kde: int = 2

    # DBSCAN
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 2

    # Logging/Debug
    verbose: bool = False

    # Reproduzierbarkeit (KernelDensity.sample hat random_state Parameter)
    random_state: Optional[int] = None
