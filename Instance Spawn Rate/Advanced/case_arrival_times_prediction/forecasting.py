from __future__ import annotations

import numpy as np


class SegmentForecaster:
    """
    Global Clusters
    """
    def estimate(self, N_hat: int, day_labels: np.ndarray) -> np.ndarray:
        # Beispiel: day_labels = [1, 1, 1, 2, 2, 1, 1, 2, 2, 2] (N_train = 10)
        #           N_hat = 12 (wir wollen 12 Tage vorhersagen)
        day_labels = np.asarray(day_labels, dtype=int)
        N_train = len(day_labels)
        if N_train == 0:
            # Fallback: Wenn keine Trainingsdaten, gib N_hat Nullen zurück
            # Beispiel: return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] für N_hat=12
            return np.zeros(N_hat, dtype=int)

        # --- Schritt 1: Finde aufeinanderfolgende Segmente mit gleichem Label ---
        # Das macht nichts anderes, als dass es die Cluster-ID der Tage an die trainingstage verteilt
        # Beispiel: day_labels = [1, 1, 1, 2, 2, 1, 1, 2, 2, 2]
        #   → segments = [(1, 3), (2, 2), (1, 2), (2, 3)]
        #   Jedes Tupel: (label, länge) = (Cluster-ID, Anzahl aufeinanderfolgender Tage)
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
        # Nach der Schleife: segments = [(1, 3), (2, 2), (1, 2), (2, 3)]

        M = len(segments)  # M = 4 im Beispiel
        # Extrahiere nur die Labels: seg_labels = [1, 2, 1, 2]
        seg_labels = [lab for (lab, _) in segments]

        # --- Schritt 2: Suche nach wiederkehrendem Muster in den Segment-Labels ---
        # Beispiel: seg_labels = [1, 2, 1, 2]
        #   p=1: candidate = [1], prüfe ob [1, 2, 1, 2] == [1, 1, 1, 1]? → Nein
        #   p=2: candidate = [1, 2], prüfe ob [1, 2, 1, 2] == [1, 2, 1, 2]? → Ja!
        #   → pattern = [(1, 3), (2, 2)] (die ersten 2 Segmente)
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

        # --- Schritt 3: Extrapoliere das Muster für N_hat Tage ---
        if pattern is None:
            # Kein Muster gefunden → verwende das letzte Label für alle Tage
            # Beispiel: segments = [(1, 3), (2, 2), (1, 2), (2, 3)]
            #   → last_label = 2
            #   → return [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] für N_hat=12
            last_label = segments[-1][0]
            return np.full(N_hat, last_label, dtype=int)

        # Muster gefunden → wiederhole es bis N_hat Tage erreicht sind
        # Beispiel: pattern = [(1, 3), (2, 2)], N_hat = 12
        #   Iteration 1: result = [1, 1, 1, 2, 2] (5 Tage)
        #   Iteration 2: result = [1, 1, 1, 2, 2, 1, 1, 1, 2, 2] (10 Tage)
        #   Iteration 3: result = [1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1] (12 Tage)
        #   → return [1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1]
        result = []
        while len(result) < N_hat:
            for lab, length in pattern:
                result.extend([lab] * length)
                if len(result) >= N_hat:
                    break

        return np.array(result[:N_hat], dtype=int)
