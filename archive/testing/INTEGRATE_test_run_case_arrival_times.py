from typing import Optional

import pandas as pd

from case_arrival_times_prediction import run
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter


if __name__ == "__main__":
    # Flag: soll das Modell neu trainiert werden?
    RETRAIN_MODEL = False  # auf False setzen, um nur ein gespeichertes Modell zu laden
    MODEL_PATH = "case_arrival_model.pkl"
    # Wie viele Tage sollen simuliert werden? (None = gleiche Länge wie Testperiode)
    N_DAYS_TO_SIMULATE: Optional[int] = None

    if RETRAIN_MODEL:
        # XES nur laden, wenn das Modell neu trainiert werden soll
        log = xes_importer.apply("eventlog.xes.gz")
        df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

        # Sicherstellen, dass der Zeitstempel korrekt als Datetime vorliegt
        df["time:timestamp"] = pd.to_datetime(
            df["time:timestamp"],
            utc=True,
            errors="raise",
            format="mixed",  # pandas erkennt je Element das Format
        )
    else:
        df = None

    case_timestamps = run(
        df=df,
        retrain_model=RETRAIN_MODEL,
        model_path=MODEL_PATH,
        n_days_to_simulate=N_DAYS_TO_SIMULATE,
    )

    print(f"Anzahl simulierter Case-Timestamps: {len(case_timestamps)}")
    print("Erste 10:", case_timestamps[:10])
