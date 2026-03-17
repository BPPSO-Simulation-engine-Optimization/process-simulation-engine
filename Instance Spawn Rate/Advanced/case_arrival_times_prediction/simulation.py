from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from .forecasting import SegmentForecaster
from .intraday import IntradayBounds
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

        # Working-hour bounds (set by generate(); falls back to full-day if None)
        self._bounds: Optional[IntradayBounds] = None

    def generate(
        self,
        N_hat: int,
        D_train: DailySequence,
        day_labels: np.ndarray,
        weekday_cluster_map: Dict[Tuple[int, int], Optional[int]],
        kde_models: Dict[Tuple[int, int, int], Optional[KernelDensity]],
        start_date: Optional[pd.Timestamp] = None,
        max_resample: int = 20,
        bounds: Optional[IntradayBounds] = None,
    ) -> SimulationResult:

        # 1) Startzeitpunkt bestimmen (exakter Timestamp)
        if start_date is None:
            all_train_ts = [ts for day in D_train for ts in day]
            if len(all_train_ts) == 0:
                raise ValueError("D_train enthält keine Timestamps; start_date kann nicht abgeleitet werden.")
            last_ts = max(pd.to_datetime(ts) for ts in all_train_ts)
            start_date = last_ts.normalize() + pd.Timedelta(days=1)
        else:
            start_date = pd.to_datetime(start_date)

        sim_start_ts = pd.to_datetime(start_date)
        sim_start_day = sim_start_ts.floor("D")

        # 2) Globale Cluster (j) pro Tag schätzen
        est_segments_per_day = self.forecaster.estimate(N_hat, day_labels)

        # 3) Bin-Länge in Sekunden – Paper: CreateTimeBins(lower, upper, L)
        #    Bins span only the observed working-hour window, not 00:00–24:00.
        if bounds is None:
            bounds = IntradayBounds(lower=0.0, upper=24 * 60 * 60)
        self._bounds = bounds
        bin_length_seconds = (bounds.upper - bounds.lower) / self.L

        D_sim: DailySequence = []

        for i in range(N_hat):
            current_day = sim_start_day + pd.Timedelta(days=i)
            weekday = current_day.weekday() + 1  # 1..7

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

                # Calendar-day anchoring: bins are tied to midnight-based day windows.
                # If simulation starts mid-day, day 0 is treated as partial.
                bin_start = current_day + pd.Timedelta(seconds=bounds.lower + (l - 1) * bin_length_seconds)
                bin_end = current_day + pd.Timedelta(seconds=bounds.lower + l * bin_length_seconds)
                if i == 0:
                    if bin_end <= sim_start_ts:
                        continue
                    effective_start = max(bin_start, sim_start_ts)
                else:
                    effective_start = bin_start

                max_duration = (bin_end - effective_start).total_seconds()
                if max_duration <= 0:
                    continue

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

                    ts = effective_start + pd.Timedelta(seconds=t_next)
                    seq_day.append(ts)
                    t = t_next

            seq_day.sort()
            D_sim.append(seq_day)

        return SimulationResult(D_sim=D_sim)
