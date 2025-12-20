import pandas as pd
from case_arrival_times_prediction import SimulationConfig, CaseInterarrivalPipeline
import os
os.environ["THREADPOOLCTL_DISABLE"] = "1"
import pandas as pd
from typing import List

import numpy as np
import pandas as pd

def interarrival_stats_intraday_only(D_sim, unit="seconds"):
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


def run(df: pd.DataFrame) -> List[pd.Timestamp]:

    cfg = SimulationConfig(
        train_ratio=0.8,
        window_size=21,
        kmax=5,
        z_values=(0.9, 0.725, 0.55, 0.375, 0.2),
        L=4,
        random_state=42,
        verbose=False
    )

    pipe = CaseInterarrivalPipeline(cfg)
    pipe.fit(df)

    art = pipe.artifacts
    print("Train days:", len(art.D_train), "Test days:", len(art.D_test))
    print("Train arrivals:", sum(len(d) for d in art.D_train))
    print("Test arrivals:", sum(len(d) for d in art.D_test))


    case_timestamps = pipe.simulate_case_timestamps()

    D_sim = pipe.simulate_days()
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


if __name__ == "__main__":
    df = pd.read_csv("event_log.csv")
    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"],
        utc=True,
        errors="raise",
        format="mixed"     # pandas erkennt je Element das Format
    )

    case_timestamps = run(df)

    print(f"Anzahl simulierter Case-Timestamps: {len(case_timestamps)}")
    print("Erste 10:", case_timestamps[:10])
