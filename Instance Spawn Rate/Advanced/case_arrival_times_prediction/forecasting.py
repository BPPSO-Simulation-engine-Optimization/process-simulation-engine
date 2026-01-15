from __future__ import annotations

import numpy as np


class SegmentForecaster:
    """
    Entspricht deiner estimate_segments_per_day(N_hat, day_labels).
    """
    def estimate(self, N_hat: int, day_labels: np.ndarray) -> np.ndarray:
        day_labels = np.asarray(day_labels, dtype=int)
        N_train = len(day_labels)
        if N_train == 0:
            return np.zeros(N_hat, dtype=int)

        segments = []
        current_label = day_labels[0]
        current_len = 1

        for lab in day_labels[1:]:
            if lab == current_label:
                current_len += 1
            else:
                segments.append((current_label, current_len))
                current_label = lab
                current_len = 1
        segments.append((current_label, current_len))

        M = len(segments)
        seg_labels = [lab for (lab, _) in segments]

        pattern = None
        for p in range(1, M // 2 + 1):
            candidate = seg_labels[:p]
            ok = True
            for i in range(M):
                if seg_labels[i] != candidate[i % p]:
                    ok = False
                    break
            if ok:
                pattern = segments[:p]
                break

        if pattern is None:
            last_label = segments[-1][0]
            return np.full(N_hat, last_label, dtype=int)

        result = []
        while len(result) < N_hat:
            for lab, length in pattern:
                result.extend([lab] * length)
                if len(result) >= N_hat:
                    break

        return np.array(result[:N_hat], dtype=int)
