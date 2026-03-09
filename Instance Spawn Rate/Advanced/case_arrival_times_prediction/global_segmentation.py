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
    Get Features from Segments -> Calculate Statistics
    """
    @staticmethod
    def extract(segment: List[DayArrivals]) -> Optional[np.ndarray]:
        daily_counts = np.array([len(day) for day in segment], dtype=float)
        if daily_counts.size == 0:
            return None

        avg_daily = daily_counts.mean()
        p25_daily, p75_daily = np.percentile(daily_counts, [25, 75])

        diffs_hours = []
        for day in segment:
            if len(day) < 2:
                continue
            timestamps = pd.to_datetime(day)
            if timestamps.tz is not None:
                timestamps = timestamps.tz_localize(None)
            arr = np.sort(np.array(timestamps, dtype="datetime64[ns]"))
            diffs = np.diff(arr).astype("timedelta64[s]").astype(float) / 3600.0  # hours
            diffs = diffs[diffs > 0]
            if diffs.size > 0:
                diffs_hours.extend(diffs.tolist())

        if len(diffs_hours) == 0:
            std_ia = 0.0
            p25_ia = 0.0
            p75_ia = 0.0
        else:
            diffs = np.array(diffs_hours, dtype=float)
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
        """
        Segments the daily arrival sequence D into contiguous time periods and
        clusters them by similarity, producing a global segmentation of the
        process timeline.

        High-level steps:
          1. Compute a moving-average (MA) of daily case counts.
          2. Derive a change-rate signal (Lambda) from the MA.
          3. For each sensitivity level z, detect outlier change points via
             a modified IQR rule on Lambda.
          4. Split the timeline at the detected change points into segments.
          5. Extract statistical features per segment and cluster them with
             DBSCAN to group similar periods (e.g. "high season" vs "low season").
          6. Return the first valid segmentation that satisfies quality
             constraints (min segment length >= window_size, #clusters < kmax).

        Args:
            D:           Daily arrival sequence — list of lists, one entry per day,
                         each containing the arrival timestamps of that day.
            window_size: Window width (in days) for the rolling mean and the
                         change-rate calculation.
            kmax:        Upper bound on the number of allowed clusters. A
                         segmentation is only accepted if #clusters < kmax.
            z_values:    Sequence of sensitivity multipliers for the IQR-based
                         outlier detection, tried from most conservative (1.0)
                         to most aggressive (0.2). The first z that yields a
                         valid segmentation wins.

        Returns:
            GlobalSegmentationResult with cluster groups G, segment labels,
            per-day labels, and the detected change-point days.
        """

        if z_values is None:
            # Sensitivity levels: 1.0 = conservative (few CPs), 0.2 = aggressive (many CPs)
            z_values = (1.0, 0.8, 0.6, 0.4, 0.2)

        N = len(D)  # total number of days in the sequence

        # --- Early exit: not enough data to compute even one full MA window ---
        if N < 2 * window_size:
            G = [D]
            seg_labels = np.array([1], dtype=int)
            day_labels = np.ones(N, dtype=int)
            return GlobalSegmentationResult(G=G, seg_labels=seg_labels, day_labels=day_labels, cp_days=[])

        # =====================================================================
        # STEP 1: Daily case counts → Rolling mean (MA)
        # M[i] = number of case arrivals on day i
        # MA   = centered moving average of M over `window_size` days
        # =====================================================================
        M = np.array([len(ti) for ti in D], dtype=float)
        # Paper formula: MA_i = (1/w) * sum_{k=i}^{i+w-1} M_k
        # for i = 0..N-w. This avoids rolling-window alignment ambiguity.
        csum = np.concatenate(([0.0], np.cumsum(M)))
        MA = (csum[window_size:] - csum[:-window_size]) / float(window_size)

        N_ma = len(MA)
        # Not enough smoothed values to derive a meaningful change signal
        # Das bezieht sich auf den Mangel an Moving average Daten
        if N_ma < 2 * window_size:
            G = [D]
            seg_labels = np.array([1], dtype=int)
            day_labels = np.ones(N, dtype=int)
            return GlobalSegmentationResult(G=G, seg_labels=seg_labels, day_labels=day_labels, cp_days=[])

        # =====================================================================
        # STEP 2: Change-rate signal (Lambda)
        # Lambda[i] = MA[i + window_size] - MA[i]
        # Positive Lambda → arrival rate is increasing over that window
        # Negative Lambda → arrival rate is decreasing
        # =====================================================================
        # z.B. np.array([0.2, 0.1, -0.3, 0.5, -0.1, 8.5, 0.3, -7.2, 0.0, 0.4])
        Lambda = MA[window_size:] - MA[:-window_size]

        best_result = None

        # =====================================================================
        # STEP 3: Iterate over sensitivity levels z to find change points
        # For each z we apply a modified IQR outlier rule on Lambda to detect
        # indices where the change rate is unusually large (= regime shifts).
        # =====================================================================
        for z in z_values:
            # --- 3a: IQR-based outlier bounds on Lambda ---
            q1 = np.quantile(Lambda, 0.25)
            q3 = np.quantile(Lambda, 0.75)
            iqr = q3 - q1

            # z scales the fence width: smaller z → narrower fence → more outliers
            cf = 1.5 * iqr * z
            lower = q1 - cf
            upper = q3 + cf

            # --- 3b: Find indices in Lambda that exceed the fences ---
            # Outlier aus Lambda Liste oben wählen
            cand_idx = np.where((Lambda < lower) | (Lambda > upper))[0]
            if cand_idx.size == 0:
                # No outliers at this sensitivity → try a more aggressive z
                continue

            # --- 3c: Group consecutive outlier indices together ---
            # Dort wo die Sequenz aus Outlier brüche von +1 hat, wird eine neue Gruppe erstellt
            groups = np.split(cand_idx, np.where(np.diff(cand_idx) != 1)[0] + 1)

            # --- 3d: Pick the single strongest change point per group ---
            # (the index with the largest absolute Lambda value in the group)
            cp_lambda_idx = []
            for g in groups:
                if g.size == 0:
                    continue
                max_idx = g[np.argmax(np.abs(Lambda[g]))]
                cp_lambda_idx.append(max_idx)

            # --- 3e: Map Lambda-indices back to original day indices ---
            # Lambda is offset by `window_size` relative to the day sequence D
            cp_lambda_idx = np.array(sorted(cp_lambda_idx), dtype=int)
            cp_days = cp_lambda_idx + window_size
            cp_days = cp_days[(cp_days > 0) & (cp_days < N)]  # clamp to valid range
            cp_days = np.unique(cp_days)

            if cp_days.size == 0:
                continue

            # =================================================================
            # STEP 4: Split the daily sequence D at the change points
            # Each segment is a contiguous slice D[start:end].
            # =================================================================
            # Dann werden diese maximalen Outlier pro Gruppe als Segmentgrenzen definiert und auf Tage gemappt
            segments = []
            segment_day_ranges = []
            start = 0
            for cp in cp_days:
                segments.append(D[start:cp])
                segment_day_ranges.append((start, cp))
                start = cp
            segments.append(D[start:N])          # last segment until end
            segment_day_ranges.append((start, N))

            S = segments
            lens = np.array([end - start for (start, end) in segment_day_ranges], dtype=int)

            # Need at least 2 segments for clustering to make sense
            if len(S) < 2:
                continue

            # =================================================================
            # STEP 5: Feature extraction & DBSCAN clustering of segments
            # Each segment is described by 6 features (daily count stats +
            # inter-arrival time stats) and then clustered to group similar
            # periods together (e.g. "busy" vs "quiet" phases).
            # =================================================================

            # --- 5a: Extract feature vectors per segment ---
            feats = []
            valid_idx = []  # track which segments had enough data for features
            for idx, seg in enumerate(S):
                f = SegmentFeatureExtractor.extract(seg)
                if f is not None:
                    feats.append(f)
                    valid_idx.append(idx)

            feats = np.array(feats)
            if feats.shape[0] < 2:
                # Not enough valid segments to cluster → try next z
                continue

            # --- 5b: Standardize features (zero mean, unit variance) ---
            scaler = StandardScaler()
            X = scaler.fit_transform(feats)

            # --- 5c: Density-based clustering with DBSCAN ---
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            raw_labels = db.fit_predict(X)

            # If all segments are noise (-1), this z-level is not useful
            unique_raw = np.unique(raw_labels)
            if set(unique_raw) == {-1}:
                continue

            # --- 5d: Re-map DBSCAN labels to 1-based contiguous IDs ---
            # DBSCAN labels can be -1 (noise), 0, 1, … → remap to 1, 2, 3, …
            unique_raw_sorted = sorted(unique_raw)
            label_map = {lab: i + 1 for i, lab in enumerate(unique_raw_sorted)}

            # Assign labels to ALL segments (including ones without features)
            seg_labels_all = np.zeros(len(S), dtype=int)
            for idx in range(len(S)):
                if idx in valid_idx:
                    lab = raw_labels[valid_idx.index(idx)]
                    seg_labels_all[idx] = label_map[lab]
                else:
                    # Segments without features get a unique fallback label
                    seg_labels_all[idx] = label_map[unique_raw_sorted[-1]] + idx

            J = len(np.unique(seg_labels_all))  # number of distinct clusters

            # =================================================================
            # STEP 6: Quality check — accept this segmentation if:
            #   - Every segment has at least `window_size` days (stable enough)
            #   - The number of clusters is below the limit kmax
            # =================================================================
            if lens.min() >= window_size and J < kmax:
                best_result = (S, seg_labels_all, segment_day_ranges, cp_days)
                break  # first valid z wins (most conservative that works)

        # =====================================================================
        # FALLBACK: No valid segmentation found at any z → treat entire
        # timeline as a single cluster.
        # =====================================================================
        # Beispiel: Wenn N=100 Tage und keine Segmentierung gefunden wurde:
        #   G = [D]  # Liste mit einem Cluster, der alle 100 Tage enthält
        #   seg_labels = [1]  # Ein einziges Segment mit Label 1
        #   day_labels = [1, 1, 1, ..., 1]  # Alle 100 Tage haben Label 1
        #   cp_days = []  # Keine Change Points
        if best_result is None:
            if self.verbose:
                print("Fallback on single Cluster!")
            G = [D]
            seg_labels = np.array([1], dtype=int)
            day_labels = np.ones(N, dtype=int)
            return GlobalSegmentationResult(G=G, seg_labels=seg_labels, day_labels=day_labels, cp_days=[])

        # =====================================================================
        # POST-PROCESSING: Build the final output structures
        # =====================================================================
        # Beispiel: Angenommen wir haben 4 Segmente gefunden:
        #   S = [seg0, seg1, seg2, seg3]  # 4 Segmente (jeweils Liste von Tagen)
        #   seg_labels = [1, 2, 1, 2]  # Segment 0,2 → Cluster 1; Segment 1,3 → Cluster 2
        #   segment_day_ranges = [(0, 30), (30, 60), (60, 80), (80, 100)]  # Tag-Bereiche
        #   cp_days = [30, 60, 80]  # Change Points an Tag 30, 60, 80
        S, seg_labels, segment_day_ranges, cp_days = best_result

        # --- Map segment cluster labels to 0-based contiguous IDs ---
        # Beispiel: seg_labels = [1, 2, 1, 2] → unique_seg_labels = [1, 2]
        #   cluster_id_map = {1: 0, 2: 1}  # Label 1 → Cluster 0, Label 2 → Cluster 1
        #   J = 2  # Zwei Cluster
        unique_seg_labels = sorted(np.unique(seg_labels))
        cluster_id_map = {lab: i for i, lab in enumerate(unique_seg_labels)}  # 0..J-1
        J = len(unique_seg_labels)

        # --- G[j] collects all segments that belong to cluster j ---
        # Beispiel: Mit seg_labels = [1, 2, 1, 2] und cluster_id_map = {1: 0, 2: 1}
        #   G[0] = [seg0, seg2]  # Cluster 0 enthält Segment 0 und 2
        #   G[1] = [seg1, seg3]  # Cluster 1 enthält Segment 1 und 3
        G: List[list] = [[] for _ in range(J)]
        for seg_idx, seg in enumerate(S):
            cid = cluster_id_map[seg_labels[seg_idx]]
            G[cid].append(seg)

        # --- day_labels: assign each original day its cluster ID (1-based) ---
        # Beispiel: Mit segment_day_ranges = [(0, 30), (30, 60), (60, 80), (80, 100)]
        #   und seg_labels = [1, 2, 1, 2] → cluster_id_map = {1: 0, 2: 1}
        #   Tag 0-29:   cid = cluster_id_map[1] + 1 = 0 + 1 = 1
        #   Tag 30-59:  cid = cluster_id_map[2] + 1 = 1 + 1 = 2
        #   Tag 60-79:  cid = cluster_id_map[1] + 1 = 0 + 1 = 1
        #   Tag 80-99:  cid = cluster_id_map[2] + 1 = 1 + 1 = 2
        #   day_labels = [1,1,...,1, 2,2,...,2, 1,1,...,1, 2,2,...,2]
        #                (30x)      (30x)      (20x)      (20x)
        day_labels = np.zeros(N, dtype=int)
        for seg_idx, (start, end) in enumerate(segment_day_ranges):
            cid = cluster_id_map[seg_labels[seg_idx]] + 1  # 1-based for output
            day_labels[start:end] = cid

        return GlobalSegmentationResult(G=G, seg_labels=seg_labels, day_labels=day_labels, cp_days=list(cp_days))
