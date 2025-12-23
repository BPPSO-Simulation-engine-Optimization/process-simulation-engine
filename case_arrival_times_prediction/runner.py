"""
API-Modul zum Trainieren und Ausführen von Case Arrival Time Predictions.

Dieses Modul stellt eine einfache Schnittstelle bereit, um:
- Modelle zu trainieren und zu speichern
- Gespeicherte Modelle zu laden
- Case-Timestamps zu simulieren
"""
import os
os.environ["THREADPOOLCTL_DISABLE"] = "1"  # muss vor sklearn-Import gesetzt werden
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd

# Workaround für macOS/Accelerate + threadpoolctl Bug:
# wir überschreiben sklearn.utils.fixes.threadpool_limits mit einem No-Op-Contextmanager,
# damit beim Aufruf von DBSCAN kein threadpoolctl verwendet wird.
try:
    import sklearn.utils.fixes as _sk_fixes  # type: ignore

    class _NoOpThreadpoolLimits:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    _sk_fixes.threadpool_limits = _NoOpThreadpoolLimits  # type: ignore
except Exception:
    # Falls sklearn nicht importierbar ist, ignorieren wir den Workaround stillschweigend.
    pass

from .config import SimulationConfig
from .pipeline import CaseInterarrivalPipeline
from .preprocessing import DailySequence


def interarrival_stats_intraday_only(D_sim: DailySequence, unit: str = "seconds"):
    """
    Berechnet Interarrival-Statistiken nur innerhalb von Tagen (intraday).
    
    Args:
        D_sim: Liste von Tagen, wobei jeder Tag eine Liste von Timestamps ist
        unit: "seconds" (default) oder "hours"
    
    Returns:
        Dictionary mit Statistiken oder None wenn keine Daten vorhanden
    """
    diffs = []
    for day in D_sim:
        if len(day) < 2:
            continue
        arr = np.array(sorted(pd.to_datetime(day)), dtype="datetime64[ns]")
        d = np.diff(arr).astype("timedelta64[ns]").astype(float) / 1e9  # Sekunden (float)
        d = d[d > 0]
        diffs.extend(d.tolist())

    diffs = np.array(diffs, dtype=float)
    if diffs.size == 0:
        return None

    if unit == "hours":
        diffs = diffs / 3600.0

    q05, q25, q50, q75, q95 = np.percentile(diffs, [5, 25, 50, 75, 95])

    return {
        "mean": float(diffs.mean()),
        "std": float(diffs.std()),
        "q05": float(q05),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q95": float(q95),
        "n": int(diffs.size),
    }


def run(
    df: Optional[pd.DataFrame],
    retrain_model: bool,
    model_path: str = "case_arrival_model.pkl",
    n_days_to_simulate: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
) -> List[pd.Timestamp]:
    """
    Trainiert oder lädt ein Modell und simuliert Case-Timestamps.
    
    Args:
        df: DataFrame mit Event-Log Daten (benötigt wenn retrain_model=True).
            Muss eine Spalte "time:timestamp" enthalten.
        retrain_model: Wenn True, wird das Modell neu trainiert und gespeichert.
                      Wenn False, wird ein gespeichertes Modell geladen.
        model_path: Pfad zur Modell-Datei (.pkl)
        n_days_to_simulate: Anzahl der Tage, die simuliert werden sollen.
                           None = gleiche Länge wie Testperiode
        config: Optional SimulationConfig. Wenn None, werden Standardwerte verwendet.
    
    Returns:
        Liste von simulierten Case-Timestamps (pd.Timestamp)
    
    Raises:
        ValueError: Wenn retrain_model=True aber df=None
        FileNotFoundError: Wenn retrain_model=False aber Modell-Datei nicht existiert
    """
    if config is None:
        cfg = SimulationConfig(
            train_ratio=0.8,
            window_size=21,
            kmax=5,
            z_values=(0.9, 0.725, 0.55, 0.375, 0.2),
            L=4,
            random_state=42,
            verbose=False
        )
    else:
        cfg = config

    if retrain_model:
        if df is None:
            raise ValueError("DataFrame df must be provided when retrain_model=True")
        pipe = CaseInterarrivalPipeline(cfg)
        pipe.fit(df)

        # Modell speichern
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
    else:
        # Nur Modell laden
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. "
                f"Set retrain_model=True once to train and store the model."
            )
        with open(model_path, "rb") as f:
            pipe = pickle.load(f)

    art = pipe.artifacts
    print("Train days:", len(art.D_train), "Test days:", len(art.D_test))
    print("Train arrivals:", sum(len(d) for d in art.D_train))
    print("Test arrivals:", sum(len(d) for d in art.D_test))

    # Wenn n_days_to_simulate gesetzt ist, wird diese Anzahl an Tagen simuliert.
    # Die tatsächliche Anzahl der Timestamps ergibt sich dann aus dem Modell.
    case_timestamps = pipe.simulate_case_timestamps(N_hat=n_days_to_simulate)

    D_sim = pipe.simulate_days(N_hat=n_days_to_simulate)
    stats = interarrival_stats_intraday_only(D_sim, unit="seconds")
    print("simulierte Interarrival-Statistiken (intraday only):")
    for k in ["mean","std","q05","q25","q50","q75","q95"]:
        print(f"  {k}: {stats[k]:.4f}")

    score = pipe.evaluate_sqrt_cadd(D_sim)
    print(f"√CADD = {score:.4f}")

    pipe.print_simulated_interarrival_statistics(
        case_timestamps, unit="seconds"
    )

    return case_timestamps

