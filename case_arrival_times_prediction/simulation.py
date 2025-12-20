from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from .forecasting import SegmentForecaster
from .preprocessing import DailySequence, DayArrivals


@dataclass(frozen=True)
class SimulationResult:
    D_sim: DailySequence


class ArrivalGenerator:
    """
    Generiert synthetische Arrivals je Tag auf Basis:
    - globaler Cluster pro Tag (SegmentForecaster)
    - Weekday-Cluster Mapping (j, weekday) -> k
    - KDE-Modelle pro (j, k, l) mit Interarrivals in Sekunden

    Wichtig: Für reproduzierbare, aber nicht degenerierte Samples wird ein RNG
    EINMALIG erzeugt und als Objekt an kde.sample(...) weitergereicht.
    """
    def __init__(self, L: int, verbose: bool = False, random_state: Optional[int] = None):
        self.L = int(L)
        self.verbose = bool(verbose)

        self.forecaster = SegmentForecaster()

        # EINMALIGER RNG: sorgt für reproduzierbare, aber nicht identische Samples pro call
        self._rng = np.random.RandomState(random_state) if random_state is not None else None

    def generate(
        self,
        N_hat: int,
        D_train: DailySequence,
        day_labels: np.ndarray,
        weekday_cluster_map: Dict[Tuple[int, int], Optional[int]],
        kde_models: Dict[Tuple[int, int, int], Optional[KernelDensity]],
        start_date: Optional[pd.Timestamp] = None,
        max_resample: int = 20,
    ) -> SimulationResult:

        # 1) Startdatum bestimmen
        if start_date is None:
            all_train_ts = [ts for day in D_train for ts in day]
            if len(all_train_ts) == 0:
                raise ValueError("D_train enthält keine Timestamps; start_date kann nicht abgeleitet werden.")
            last_ts = max(pd.to_datetime(ts) for ts in all_train_ts)
            start_date = last_ts.normalize() + pd.Timedelta(days=1)
        else:
            start_date = pd.to_datetime(start_date).normalize()

        # 2) Globale Cluster (j) pro Tag schätzen
        est_segments_per_day = self.forecaster.estimate(N_hat, day_labels)

        # 3) Bin-Länge in Sekunden
        seconds_per_day = 24 * 60 * 60
        bin_length_seconds = seconds_per_day / self.L

        D_sim: DailySequence = []

        for i in range(N_hat):
            current_date = start_date + pd.Timedelta(days=i)
            weekday = current_date.weekday() + 1  # 1..7

            # Globaler Cluster j für diesen Tag
            j = int(est_segments_per_day[i])

            # Weekday-Cluster k
            k = weekday_cluster_map.get((j, weekday), None)
            if k is None:
                D_sim.append([])
                continue

            seq_day: DayArrivals = []

            # 4) Für jeden Intraday-Bin l (1..L) Arrivals generieren
            for l in range(1, self.L + 1):
                kde = kde_models.get((j, k, l), None)
                if kde is None:
                    if self.verbose:
                        print(f"kde is none for (j={j}, k={k}, l={l})")
                    continue

                bin_start = current_date + pd.Timedelta(seconds=(l - 1) * bin_length_seconds)
                max_duration = bin_length_seconds

                t = 0.0  # kumulierte Zeit in Sekunden seit Bin-Start

                while True:
                    ia = None

                    # positive Interarrival samplen (max_resample Versuche)
                    for _ in range(max_resample):
                        if self._rng is not None:
                            sample = kde.sample(1, random_state=self._rng)[0, 0]
                        else:
                            sample = kde.sample(1)[0, 0]

                        if sample > 0:
                            ia = float(sample)
                            break

                    if ia is None:
                        # zu oft <=0 gezogen -> Bin abbrechen
                        break

                    t_next = t + ia
                    if t_next > max_duration:
                        # nächste Ankunft läge außerhalb des Bins
                        break

                    ts = bin_start + pd.Timedelta(seconds=t_next)
                    seq_day.append(ts)
                    t = t_next

            seq_day.sort()
            D_sim.append(seq_day)

        return SimulationResult(D_sim=D_sim)
