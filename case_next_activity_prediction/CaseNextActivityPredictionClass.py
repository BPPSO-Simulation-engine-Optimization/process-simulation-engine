import pandas as pd
import pm4py

class CaseNextActivityPredictionClass:

    def __init__(self):
        pass

    def predict(self):
        # TODO: Modell bauen
        # TODO: Predicten

        predicted_next_activity = "A_Create Application"
        isCaseEnded = False

        return predicted_next_activity, isCaseEnded


def prefix_generation():
    """
    Generates all prefixes from a CSV event log
    Assumes CSV contains at least: case_id, activity
    """

    # --- CSV statt XES laden ---
    df = pd.read_csv("event_log.csv")

    # Falls Timestamp existiert → sortieren
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["case_id", "timestamp"])
    else:
        df = df.sort_values(["case_id"])

    prefixes = []
    labels = []

    # --- Pro Case Präfixe generieren ---
    for case_id, case_trace in df.groupby("case_id"):
        activities = list(case_trace["activity"])

        for i in range(1, len(activities)):
            prefix = activities[:i]
            label = activities[i]

            prefixes.append(prefix)
            labels.append(label)

    # --- DataFrame exportieren ---
    out_df = pd.DataFrame({
        "prefix": prefixes,
        "next_activity": labels
    })

    out_df.to_csv(r"Dataset/Prefix_Label_Data.csv", index=False)
    print("Prefix dataset created at Dataset/Prefix_Label_Data.csv")


if __name__ == "__main__":
    prefix_generation()
