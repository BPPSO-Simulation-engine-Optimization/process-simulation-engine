import pandas as pd
import pm4py
from ProcessingTimeTrainer import ProcessingTimeTrainer
from ProcessingTimePredictionClass import ProcessingTimePredictionClass

print("Loading event log...")
log = pm4py.read_xes("Dataset/BPI Challenge 2017.xes")
df = pm4py.convert_to_dataframe(log)
print(f"Loaded {len(df)} events")

print("\nTraining probabilistic LSTM model with one-hot encoding...")
trainer = ProcessingTimeTrainer(df, method="probabilistic_ml")
trainer.train(cache_path="models/processing_time_model_lstm", force_recompute=False)
trainer.save_model("models/processing_time_model_lstm")
print("Model saved!")

print("\nLoading model for predictions...")
predictor = ProcessingTimePredictionClass(method="probabilistic_ml", model_path="models/processing_time_model_lstm")

print("\nTesting predictions:")
test_cases = [
    {
        "prev_activity": "A_Submitted",
        "prev_lifecycle": "complete",
        "curr_activity": "W_Complete application",
        "curr_lifecycle": "scheduled",
        "context": {
            'resource_1': 'User_1',
            'resource_2': 'User_2',
            'hour': 14,
            'weekday': 2,
            'month': 3,
            'day_of_year': 75,
            'event_position_in_case': 1,
            'case_duration_so_far': 3600.0
        }
    },
    {
        "prev_activity": "W_Complete application",
        "prev_lifecycle": "complete",
        "curr_activity": "A_Accepted",
        "curr_lifecycle": "complete",
        "context": {
            'resource_1': 'User_2',
            'resource_2': 'User_3',
            'hour': 15,
            'weekday': 2,
            'month': 3,
            'day_of_year': 75,
            'event_position_in_case': 2,
            'case_duration_so_far': 7200.0
        }
    }
]

for i, test in enumerate(test_cases, 1):
    prediction = predictor.predict(
        prev_activity=test["prev_activity"],
        prev_lifecycle=test["prev_lifecycle"],
        curr_activity=test["curr_activity"],
        curr_lifecycle=test["curr_lifecycle"],
        context=test["context"]
    )
    
    dist_info = predictor.get_probabilistic_distribution(
        prev_activity=test["prev_activity"],
        prev_lifecycle=test["prev_lifecycle"],
        curr_activity=test["curr_activity"],
        curr_lifecycle=test["curr_lifecycle"],
        context=test["context"]
    )
    
    print(f"\nTest {i}:")
    print(f"  Transition: {test['prev_activity']} -> {test['curr_activity']}")
    print(f"  Predicted time: {prediction:.2f} seconds ({prediction/3600:.2f} hours)")
    if 'mean' in dist_info:
        print(f"  Distribution mean: {dist_info['mean']:.2f}s, std: {dist_info['std']:.2f}s")

print("\nDone!")

