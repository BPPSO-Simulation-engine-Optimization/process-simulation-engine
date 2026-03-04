from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from .config import SimulationConfig
from .preprocessing import DailyArrivalBuilder, DailySequence, DailyArrivalsResult
from .global_segmentation import GlobalSegmentClusterer
from .weekday_clustering import WeekdayClusterer
from .intraday import IntradayBinner, IntradayBounds
from .kde import InterarrivalKDETrainer
from .mapping import WeekdayClusterMapper
from .simulation import ArrivalGenerator
from .metrics import flatten_days, sqrt_cadd, interarrival_statistics_from_timestamps


@dataclass
class FitArtifacts:
    D_train: DailySequence
    D_test: DailySequence
    dates_train: List[pd.Timestamp]
    dates_test: List[pd.Timestamp]

    day_labels_train: object  # np.ndarray
    weekday_cluster_map: dict
    kde_models: dict
    test_start_time: Optional[pd.Timestamp] = None
    intraday_bounds: IntradayBounds = None  # Paper: DetermineBounds(D)


class CaseInterarrivalPipeline:
    """
    Orchestriert exakt deine Notebook-Pipeline:
      build_daily_arrivals -> train/test split ->
      cluster_global_segments -> cluster_weekdays -> intraday_binning ->
      learn_interarrival_kde -> weekday mapping -> generate_arrivals

    Hauptoutput: flache Liste case_timestamps (pd.Timestamp).
    """
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        self._builder = DailyArrivalBuilder()
        self._global = GlobalSegmentClusterer(
            dbscan_eps=config.dbscan_eps,
            dbscan_min_samples=config.dbscan_min_samples,
            verbose=config.verbose
        )
        self._weekday = WeekdayClusterer()
        self._binner = IntradayBinner()
        self._kde = InterarrivalKDETrainer(
            kernel=config.kernel,
            min_samples=config.min_samples_kde,
            bandwidth_k_values=config.bandwidth_k_values,
            bandwidth_val_ratio=config.bandwidth_val_ratio,
        )
        self._mapper = WeekdayClusterMapper()
        self._generator = ArrivalGenerator(L=config.L, verbose=config.verbose, random_state=config.random_state)

        self.artifacts: Optional[FitArtifacts] = None

    def _split_daily_sequence_by_cases(
        self, daily_result: DailyArrivalsResult
    ) -> Tuple[DailySequence, DailySequence, List[pd.Timestamp], List[pd.Timestamp], pd.Timestamp]:
        first_events = daily_result.first_events.sort_values("time:timestamp", kind="mergesort").reset_index(drop=True)
        n_cases = len(first_events)
        if n_cases < 2:
            raise ValueError("Zu wenige Cases fuer einen Train/Test-Split (mindestens 2 erforderlich).")

        split_idx = int(n_cases * self.cfg.train_ratio)
        split_idx = min(max(split_idx, 1), n_cases - 1)

        first_events = first_events.copy()
        first_events["is_train"] = False
        first_events.loc[: split_idx - 1, "is_train"] = True
        first_events["date"] = pd.to_datetime(first_events["time:timestamp"]).dt.floor("D")

        grouped_train = first_events[first_events["is_train"]].groupby("date")["time:timestamp"].apply(list)
        grouped_test = first_events[~first_events["is_train"]].groupby("date")["time:timestamp"].apply(list)
        test_start_time = pd.to_datetime(first_events.loc[split_idx, "time:timestamp"])

        D_train_all: DailySequence = []
        D_test_all: DailySequence = []
        dates_all: List[pd.Timestamp] = [pd.to_datetime(d) for d in daily_result.dates]

        for d in dates_all:
            if d in grouped_train.index:
                D_train_all.append(sorted(pd.to_datetime(grouped_train.loc[d]).tolist()))
            else:
                D_train_all.append([])

            if d in grouped_test.index:
                D_test_all.append(sorted(pd.to_datetime(grouped_test.loc[d]).tolist()))
            else:
                D_test_all.append([])

        train_non_empty = [i for i, day in enumerate(D_train_all) if len(day) > 0]
        test_non_empty = [i for i, day in enumerate(D_test_all) if len(day) > 0]
        if not train_non_empty or not test_non_empty:
            raise ValueError("Train/Test-Split nach Cases ergab eine leere Seite.")

        i_train_last = train_non_empty[-1]
        i_test_first = test_non_empty[0]

        D_train = D_train_all[: i_train_last + 1]
        dates_train = dates_all[: i_train_last + 1]
        D_test = D_test_all[i_test_first:]
        dates_test = dates_all[i_test_first:]
        return D_train, D_test, dates_train, dates_test, test_start_time

    def fit(self, df: pd.DataFrame) -> FitArtifacts:
        daily = self._builder.build(df)
        D_train, D_test, dates_train, dates_test, test_start_time = self._split_daily_sequence_by_cases(daily)

        seg_res = self._global.cluster(
            D_train,
            window_size=self.cfg.window_size,
            kmax=self.cfg.kmax,
            z_values=self.cfg.z_values
        )

        weekday_clusters = self._weekday.cluster(seg_res.G, dates_train)
        intraday_binned, intraday_bounds = self._binner.bin(weekday_clusters, L=self.cfg.L)
        kde_res = self._kde.fit(intraday_binned, L=self.cfg.L)
        weekday_cluster_map = self._mapper.build(weekday_clusters)

        self.artifacts = FitArtifacts(
            D_train=D_train,
            D_test=D_test,
            dates_train=dates_train,
            dates_test=dates_test,
            day_labels_train=seg_res.day_labels,
            weekday_cluster_map=weekday_cluster_map,
            kde_models=kde_res.models,
            test_start_time=test_start_time,
            intraday_bounds=intraday_bounds,
        )
        return self.artifacts

    def simulate_days(self, N_hat: Optional[int] = None, start_date: Optional[pd.Timestamp] = None) -> DailySequence:
        if self.artifacts is None:
            raise RuntimeError("Pipeline ist nicht gefittet. Erst fit(df) ausführen.")

        if N_hat is None:
            N_hat = len(self.artifacts.D_test)

        if start_date is None:
            if self.artifacts.test_start_time is not None:
                start_date = self.artifacts.test_start_time
            elif len(self.artifacts.dates_test) > 0:
                start_date = self.artifacts.dates_test[0]
            else:
                start_date = None

        sim_res = self._generator.generate(
            N_hat=N_hat,
            D_train=self.artifacts.D_train,
            day_labels=self.artifacts.day_labels_train,
            weekday_cluster_map=self.artifacts.weekday_cluster_map,
            kde_models=self.artifacts.kde_models,
            start_date=start_date,
            bounds=self.artifacts.intraday_bounds,
        )
        return sim_res.D_sim

    def simulate_case_timestamps(self, N_hat: Optional[int] = None, start_date: Optional[pd.Timestamp] = None) -> List[pd.Timestamp]:
        D_sim = self.simulate_days(N_hat=N_hat, start_date=start_date)
        return flatten_days(D_sim)

    def evaluate_sqrt_cadd(self, D_sim: DailySequence) -> float:
        if self.artifacts is None:
            raise RuntimeError("Pipeline ist nicht gefittet. Erst fit(df) ausführen.")
        return sqrt_cadd(self.artifacts.D_test, D_sim)

    def print_simulated_interarrival_statistics(self, case_timestamps: List[pd.Timestamp], unit: str = "seconds") -> None:
        stats = interarrival_statistics_from_timestamps(case_timestamps, unit=unit)
        print("simulierte Interarrival-Statistiken:")
        print(f"  mean: {stats.mean:.4f}")
        print(f"  std:  {stats.std:.4f}")
        print(f"  q05:  {stats.q05:.4f}")
        print(f"  q25:  {stats.q25:.4f}")
        print(f"  q50:  {stats.q50:.4f}")
        print(f"  q75:  {stats.q75:.4f}")
        print(f"  q95:  {stats.q95:.4f}")
