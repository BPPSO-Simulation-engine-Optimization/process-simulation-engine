# BPIC17 Simplified Next Activity Prediction Model

A next activity prediction model specifically designed for BPIC 2017 event log with simplified lifecycle transitions (start/complete only) and END token support for trace termination prediction.

## Features

- **Simplified Lifecycle**: Filters event log to only "start" and "complete" lifecycle transitions
- **END Token Support**: Automatically adds END tokens at trace endings for termination prediction
- **Dual Output**: Predicts both next activity and next lifecycle transition
- **Simulation Integration**: Implements `NextActivityPredictor` protocol for use with DESEngine

## Installation

Ensure you have the required dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn pm4py joblib
```

## Usage

### Training

#### Command Line

```bash
python -m Next-Activity-Prediction.bpic17_simplified.train \
    --log-path "Dataset/BPI Challenge 2017.xes" \
    --output-dir "models/bpic17_simplified" \
    --max-history 15 \
    --epochs 30 \
    --batch-size 128
```

#### Python Script

```python
from Next-Activity-Prediction.bpic17_simplified import (
    load_and_filter_bpic17,
    add_end_tokens,
    BPIC17SimplifiedDataGenerator,
    BPIC17SimplifiedModel,
    BPIC17SimplifiedPersistence
)

# Load and preprocess data
df_log = load_and_filter_bpic17()
df_log = add_end_tokens(df_log)

# Generate training data
generator = BPIC17SimplifiedDataGenerator(df_log, max_history=15)
df_train = generator.generate()

# Train model
model = BPIC17SimplifiedModel(max_seq_len=15, lstm_units=256, hidden_units=256)
history = model.fit(df_train, epochs=30, batch_size=128, validation_split=0.1)

# Save model
BPIC17SimplifiedPersistence.save(model, "models/bpic17_simplified")
```

### Inference / Simulation

```python
from Next-Activity-Prediction.bpic17_simplified import BPIC17SimplifiedPredictor
from simulation.case_manager import CaseState

# Load predictor
predictor = BPIC17SimplifiedPredictor(model_path="models/bpic17_simplified")

# Create case state
case_state = CaseState(
    case_id="case_1",
    case_type="Home improvement",
    application_type="New credit",
    requested_amount=10000.0
)

# Predict next activity
next_activity, is_ended = predictor.predict(case_state)
print(f"Next activity: {next_activity}, Case ended: {is_ended}")
```

### Integration with Simulation Engine

```python
from simulation.engine import DESEngine
from Next-Activity-Prediction.bpic17_simplified import BPIC17SimplifiedPredictor

# Create predictor
predictor = BPIC17SimplifiedPredictor(model_path="models/bpic17_simplified")

# Use with simulation engine
engine = DESEngine(
    resource_allocator=resource_allocator,
    next_activity_predictor=predictor,
    # ... other parameters
)

# Run simulation
results = engine.run(num_cases=100)
```

## Model Architecture

- **Input**: 
  - Activities: Sequence of activity names
  - Lifecycles: Sequence of lifecycle transitions (start/complete)
  - Resources: Sequence of resource names
  - Context: Case-level context attributes (LoanGoal, ApplicationType, RequestedAmount)
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Output**: 
  - Activity probability distribution (including END token)
  - Lifecycle probability distribution (start/complete)

## Data Preprocessing

1. **Filter Lifecycle**: Keeps only "start" and "complete" transitions
2. **Add END Tokens**: Appends END token at the end of each trace
3. **Generate Sequences**: Creates sliding windows of activity-lifecycle pairs

## Parameters

- `max_history`: Maximum sequence length for prediction (default: 15)
- `lstm_units`: Number of LSTM units (default: 256)
- `hidden_units`: Number of hidden units (default: 256)
- `repetition_penalty`: Penalty for repeating activities (default: 0.5)
- `min_samples`: Minimum samples per activity-lifecycle pair (default: 10)

## Files

- `data_preprocessing.py`: Event log loading and filtering
- `data_generator.py`: Training data generation
- `model.py`: LSTM model architecture and encoder
- `predictor.py`: Inference predictor for simulation
- `persistence.py`: Model save/load utilities
- `train.py`: Training script

## Notebooks

See the notebooks directory for:
- `01_train.ipynb`: Interactive training notebook
- `02_benchmark.ipynb`: Benchmarking with SimulationBenchmark


