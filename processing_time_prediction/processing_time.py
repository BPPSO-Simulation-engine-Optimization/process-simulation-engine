import pandas as pd
import pm4py
from processing_time_prediction.ProcessingTimeTrainer import ProcessingTimeTrainer
from processing_time_prediction.ProcessingTimePredictionClass import ProcessingTimePredictionClass

log = pm4py.read_xes("eventlog/eventlog.xes.gz")
df = pm4py.convert_to_dataframe(log)

MODEL_BASE = "models/processing_time_model"

# --- Train inter-event time ML model on start+complete only ---
print("=" * 80)
print("Filtering to start+complete events for inter-event time model...")
print("=" * 80)
sc_lifecycles = {"start", "complete"}
df_sc = df[df["lifecycle:transition"].str.lower().isin(sc_lifecycles)].copy()
print(f"Full log: {len(df)} events -> start+complete only: {len(df_sc)} events")

print("\nTraining ML model on start+complete filtered data...")
trainer = ProcessingTimeTrainer(df_sc, method="ml")
trainer.train()
trainer.save_model(MODEL_BASE)
print("Inter-event time model saved!")

# --- Extract and save resource hold time model from FULL log ---
print("\n" + "=" * 80)
print("Extracting resource hold times from full event log (all lifecycle transitions)...")
print("=" * 80)
full_trainer = ProcessingTimeTrainer(df, method="ml")  # method doesn't matter for hold time extraction
full_trainer.save_resource_hold_model(f"{MODEL_BASE}_resource_hold.joblib")

# --- Quick sanity check ---
print("\n" + "=" * 80)
print("Sanity check: loading models and making predictions...")
print("=" * 80)
predictor = ProcessingTimePredictionClass(method="ml", model_path=MODEL_BASE)

prediction = predictor.predict(
    prev_activity="A_Submitted",
    prev_lifecycle="complete",
    curr_activity="A_PartiallySubmitted",
    curr_lifecycle="start",
    context={
        'resource_1': 'User_1',
        'resource_2': 'User_2',
        'hour': 14,
        'weekday': 2
    }
)
print(f"Inter-event time prediction: {prediction:.2f}s ({prediction/3600:.2f}h)")

hold_time = predictor.predict_resource_hold_time("W_Complete application", "User_2")
print(f"Resource hold time (W_Complete application, User_2): {hold_time:.1f}s ({hold_time/60:.1f}min)")

hold_time_a = predictor.predict_resource_hold_time("A_Accepted", "User_1")
print(f"Resource hold time (A_Accepted, User_1): {hold_time_a:.1f}s ({hold_time_a/60:.1f}min)")
