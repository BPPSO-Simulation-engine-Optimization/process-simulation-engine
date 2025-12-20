from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .preprocessing import DailySequence, DayArrivals


@dataclass(frozen=True)
class GlobalSegmentationResult:
    G: List[list]                 # entspricht deiner Struktur (Cluster -> Segmente oder Tage)
    seg_labels: np.ndarray
    day_labels: np.ndarray        # Länge = #Trainingstage
    cp_days: List[int]


class SegmentFeatureExtractor:
    """
    Entspricht deiner _extract_segment_features(segment).
    """
    @staticmethod
    def extract(segment: List[DayArrivals]) -> Optional[np.ndarray]:
        daily_counts = np.array([len(day) for day in segment], dtype=float)
        if daily_counts.size == 0:
            return None

        avg_daily = daily_counts.mean()
        p25_daily, p75_daily = np.percentile(daily_counts, [25, 75])

        arrivals = [ts for day in segment for ts in day]
        if len(arrivals) < 2:
            std_ia = 0.0
            p25_ia = 0.0
            p75_ia = 0.0
        else:
            arr = np.sort(np.array(arrivals, dtype="datetime64[ns]"))
            diffs = np.diff(arr).astype("timedelta64[s]").astype(float) / 3600.0  # hours
            std_ia = np.std(diffs)
            p25_ia, p75_ia = np.percentile(diffs, [25, 75])

        return np.array([avg_daily, p25_daily, p75_daily, std_ia, p25_ia, p75_ia], dtype=float)


class GlobalSegmentClusterer:
    """
    Entspricht deiner cluster_global_segments(D, window_size, kmax, z_values).
    """
    def __init__(self, dbscan_eps: float = 0.8, dbscan_min_samples: int = 2, verbose: bool = False):
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.verbose = verbose

    def cluster(
        self,
        D: DailySequence,
        window_size: int = 14,
        kmax: int = 5,
        z_values: Optional[Sequence[float]] = None,
    ) -> GlobalSegmentationResult:

        if z_values is None:
            z_values = (1.0, 0.8, 0.6, 0.4, 0.2)

        N = len(D)
        if N < 2 * window_size:
            G = [D]
            seg_labels = np.array([1], dtype=int)
            day_labels = np.ones(N, dtype=int)
            return GlobalSegmentationResult(G=G, seg_labels=seg_labels, day_labels=day_labels, cp_days=[])

        M = np.array([len(ti) for ti in D], dtype=float)
        M_series = pd.Series(M)
        MA = (
            M_series
            .rolling(window=window_size, min_periods=window_size)
            .mean()
            .dropna()
            .to_numpy()
        )

        N_ma = len(MA)
        if N_ma < 2 * window_size:
            G = [D]
            seg_labels = np.array([1], dtype=int)
            day_labels = np.ones(N, dtype=int)
            return GlobalSegmentationResult(G=G, seg_labels=seg_labels, day_labels=day_labels, cp_days=[])

        Lambda = MA[window_size:] - MA[:-window_size]

        best_result = None

        for z in z_values:
            q1 = np.quantile(Lambda, 0.25)
            q3 = np.quantile(Lambda, 0.75)
            iqr = q3 - q1

            cf = 1.5 * iqr * z
            lower = q1 - cf
            upper = q3 + cf

            cand_idx = np.where((Lambda < lower) | (Lambda > upper))[0]
            if cand_idx.size == 0:
                continue

            groups = np.split(cand_idx, np.where(np.diff(cand_idx) != 1)[0] + 1)

            cp_lambda_idx = []
            for g in groups:
                if g.size == 0:
                    continue
                max_idx = g[np.argmax(np.abs(Lambda[g]))]
                cp_lambda_idx.append(max_idx)

            cp_lambda_idx = np.array(sorted(cp_lambda_idx), dtype=int)
            cp_days = cp_lambda_idx + window_size
            cp_days = cp_days[(cp_days > 0) & (cp_days < N)]
            cp_days = np.unique(cp_days)

            if cp_days.size == 0:
                continue

            segments = []
            segment_day_ranges = []
            start = 0
            for cp in cp_days:
                segments.append(D[start:cp])
                segment_day_ranges.append((start, cp))
                start = cp
            segments.append(D[start:N])
            segment_day_ranges.append((start, N))

            S = segments
            lens = np.array([end - start for (start, end) in segment_day_ranges], dtype=int)

            if len(S) < 2:
                continue

            feats = []
            valid_idx = []
            for idx, seg in enumerate(S):
                f = SegmentFeatureExtractor.extract(seg)
                if f is not None:
                    feats.append(f)
                    valid_idx.append(idx)

            feats = np.array(feats)
            if feats.shape[0] < 2:
                continue

            scaler = StandardScaler()
            X = scaler.fit_transform(feats)

            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            raw_labels = db.fit_predict(X)

            unique_raw = np.unique(raw_labels)
            if set(unique_raw) == {-1}:
                continue

            unique_raw_sorted = sorted(unique_raw)
            label_map = {lab: i + 1 for i, lab in enumerate(unique_raw_sorted)}

            seg_labels_all = np.zeros(len(S), dtype=int)
            for idx in range(len(S)):
                if idx in valid_idx:
                    lab = raw_labels[valid_idx.index(idx)]
                    seg_labels_all[idx] = label_map[lab]
                else:
                    seg_labels_all[idx] = label_map[unique_raw_sorted[-1]] + idx

            J = len(np.unique(seg_labels_all))

            if lens.min() >= window_size and J < kmax:
                best_result = (S, seg_labels_all, segment_day_ranges, cp_days)
                break

        if best_result is None:
            if self.verbose:
                print("Fallback on single Cluster!")
            G = [D]
            seg_labels = np.array([1], dtype=int)
            day_labels = np.ones(N, dtype=int)
            return GlobalSegmentationResult(G=G, seg_labels=seg_labels, day_labels=day_labels, cp_days=[])

        S, seg_labels, segment_day_ranges, cp_days = best_result

        unique_seg_labels = sorted(np.unique(seg_labels))
        cluster_id_map = {lab: i for i, lab in enumerate(unique_seg_labels)}  # 0..J-1
        J = len(unique_seg_labels)
        G: List[list] = [[] for _ in range(J)]

        for seg_idx, seg in enumerate(S):
            cid = cluster_id_map[seg_labels[seg_idx]]
            G[cid].append(seg)

        day_labels = np.zeros(N, dtype=int)
        for seg_idx, (start, end) in enumerate(segment_day_ranges):
            cid = cluster_id_map[seg_labels[seg_idx]] + 1
            day_labels[start:end] = cid

        return GlobalSegmentationResult(G=G, seg_labels=seg_labels, day_labels=day_labels, cp_days=list(cp_days))
