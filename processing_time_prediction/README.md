# Processing Time Prediction

Prefix-based cumulative time prediction for process traces.

## How It Works

Given a **prefix** of activities from the start of a case:
```
[Activity A, Activity B, Activity C]
```

The model predicts the **cumulative elapsed time** from the case start to the current event.

## Usage

### Training

```bash
cd processing_time_prediction
python test_cumulative_time.py --train --method lstm
```

### Evaluation

```bash
python test_cumulative_time.py --evaluate --traces 200
```

### In Code

```python
from processing_time_prediction import ProcessingTimeTrainer, ProcessingTimePrediction

# Train
trainer = ProcessingTimeTrainer(df=event_log_df, method="lstm")
trainer.train(save_path="models/cumulative_time_lstm")

# Predict
predictor = ProcessingTimePrediction()
predictor.load("models/cumulative_time_lstm")

# Get cumulative time for a prefix
elapsed_seconds = predictor.predict(["Activity A", "Activity B", "Activity C"])
print(f"Elapsed: {elapsed_seconds / 3600:.1f} hours")
```

## Methods

- **lstm**: LSTM neural network with embeddings (recommended)
- **ml**: Random Forest on encoded prefixes

## Files

- `ProcessingTimeTrainer.py` - Training logic
- `ProcessingTimePredictionClass.py` - Prediction class
- `test_cumulative_time.py` - Training and evaluation script

