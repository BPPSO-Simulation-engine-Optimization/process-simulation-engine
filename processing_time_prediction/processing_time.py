import pandas as pd
import pm4py
from ProcessingTimeTrainer import ProcessingTimeTrainer
from ProcessingTimePredictionClass import ProcessingTimePredictionClass

log = pm4py.read_xes("Dataset/BPI Challenge 2017.xes")
df = pm4py.convert_to_dataframe(log)

print("Training model...")
trainer = ProcessingTimeTrainer(df, method="ml")
trainer.train()
trainer.save_model("models/processing_time_model")
print("Model saved!")

print("Loading model for predictions...")
predictor = ProcessingTimePredictionClass(method="distribution", model_path="models/processing_time_model_tuned")

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

print(f"Predicted processing time: {prediction:.2f} seconds ({prediction/3600:.2f} hours)")