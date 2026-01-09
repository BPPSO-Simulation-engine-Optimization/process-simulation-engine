# Next Activity Prediction with One-Hot Encoding

This module provides an LSTM-based next activity prediction model for process simulation using **one-hot encoding** instead of embedding vectors. The model filters event logs to only include "start" and "complete" lifecycle transitions and integrates seamlessly with the simulation engine.

## Table of Contents

- [Overview](#overview)
- [Key Differences from Embedding-Based Module](#key-differences-from-embedding-based-module)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Configuration](#configuration)
- [Integration with Simulation Engine](#integration-with-simulation-engine)
- [File Structure](#file-structure)

---

## Overview

The **Next Activity Predictor with One-Hot Encoding** is a single-step prediction model that predicts the next activity in a case sequence one activity at a time. It uses a stacked LSTM architecture with one-hot encoded inputs instead of learned embeddings.

**Key Features:**
- Predicts one activity at a time
- Uses one-hot encoding for activity representation
- Maintains case history for stateful predictions
- Automatically detects case completion via END token
- No embedding layer - direct one-hot input to LSTM
- Real-time prediction during simulation

**When to Use One-Hot Encoding:**
- When you want explicit, interpretable activity representations
- When vocabulary size is manageable (typically < 100 activities)
- When you prefer simpler model architecture without embedding parameters
- For comparison with embedding-based approaches

---

## Key Differences from Embedding-Based Module

### Input Representation

**Embedding-Based (`next_activity_prediction/`):**
- Input: Integer indices `(sequence_length,)`
- Embedding layer converts indices to dense vectors `(sequence_length, embedding_dim)`
- Embedding dimension: typically 128

**One-Hot Based (`next_activity_prediction_onehot/`):**
- Input: One-hot encoded vectors `(sequence_length, vocab_size)`
- No embedding layer - direct input to LSTM
- Each activity represented as a binary vector of size `vocab_size`

### Model Architecture

**Embedding-Based:**
```
Input (sequence_length,) → Embedding → LSTM → Output
```

**One-Hot Based:**
```
Input (sequence_length, vocab_size) → LSTM → Output
```

### Memory and Computation

**One-Hot Encoding:**
- **Input size**: `sequence_length × vocab_size` (e.g., 50 × 28 = 1,400 values)
- **Memory per sample**: Higher than embedding (sparse but explicit)
- **Computation**: Similar LSTM computation, but larger input dimension

**Embedding-Based:**
- **Input size**: `sequence_length × embedding_dim` (e.g., 50 × 128 = 6,400 values)
- **Memory per sample**: Lower input size, but embedding parameters add to model size
- **Computation**: Embedding lookup + LSTM computation

---

## Architecture

The Next Activity Predictor uses a **stacked LSTM architecture with one-hot input**:

```
Input Sequence (one-hot encoded, shape: sequence_length × vocab_size)
    ↓
Stacked LSTM Layers (processes sequence context)
    ↓
Dense Output Layer (softmax over vocabulary)
    ↓
Next Activity Prediction (including END token)
```

**Components:**
- **One-Hot Input**: Each activity represented as a binary vector where exactly one position is 1
- **LSTM Layers**: Stacked LSTM layers that process the one-hot encoded sequence context
- **Output Layer**: Dense layer with softmax activation predicting the next activity
- **Vocabulary**: Includes all activities plus special tokens (`<PAD>`, `END`)

### How It Works

#### Training Phase

1. **Sequence Preparation**: 
   - For each case sequence `[a₁, a₂, ..., aₙ, END]`
   - Create training samples: `(prefix, next_activity)` pairs
   - Example: `([a₁], a₂)`, `([a₁, a₂], a₃)`, ..., `([a₁, ..., aₙ], END)`

2. **One-Hot Encoding**:
   - Convert activity sequences to integer indices using vocabulary
   - Pad/truncate to fixed `sequence_length`
   - Convert each integer index to one-hot vector of size `vocab_size`
   - Result: Input shape `(n_samples, sequence_length, vocab_size)`

3. **Model Learning**:
   - Learn mapping: `f: onehot_prefix → P(next_activity | prefix)`
   - Captures temporal patterns and activity dependencies
   - Learns to predict END when case should complete

#### Prediction Phase

**Algorithm:**

```
Input: case_state with activity_history = [a₁, a₂, ..., aₖ]

1. Handle empty history:
   If activity_history is empty:
     Return ("A_Create Application", False)

2. Prepare input sequence:
   sequence_indices = pad_or_truncate(activity_history, sequence_length)
   sequence_onehot = onehot_encode(sequence_indices, vocab_size)
   X = [sequence_onehot]  # Shape: (1, sequence_length, vocab_size)

3. Forward pass:
   probs = model.predict(X)  # Shape: (1, vocab_size)
   probs = probs[0]  # Shape: (vocab_size,)

4. Greedy prediction:
   next_activity_idx = argmax(probs)
   next_activity = idx_to_activity[next_activity_idx]

5. END detection:
   If next_activity == "END" or next_activity_idx == end_token_idx:
     Return (last_activity_from_history, True)
   Else:
     Return (next_activity, False)
```

**One-Hot Encoding Details:**
- **Vocabulary size**: Number of unique activities + 2 (PAD and END tokens)
- **One-hot vector**: Binary vector of length `vocab_size` with exactly one 1
- **PAD token**: Index 0, represented as `[1, 0, 0, ..., 0]`
- **Activity tokens**: Each activity has unique index, e.g., activity at index 5 is `[0, 0, 0, 0, 0, 1, 0, ..., 0]`

**Example:**
- Vocabulary: `{'<PAD>': 0, 'A_Create': 1, 'A_Submitted': 2, 'A_Concept': 3, ..., 'END': 27}`
- Sequence: `["A_Create", "A_Submitted"]` (padded to length 50)
- One-hot encoding:
  - Position 0-47: `[1, 0, 0, ..., 0]` (PAD)
  - Position 48: `[0, 1, 0, 0, ..., 0]` (A_Create)
  - Position 49: `[0, 0, 1, 0, ..., 0]` (A_Submitted)

#### Computational Complexity

**Prediction Time:**
- **One-hot encoding**: `O(sequence_length × vocab_size)` ≈ `O(50 × 28)` = `O(1,400)`
- **LSTM forward pass**: `O(sequence_length × lstm_units²)` ≈ `O(50 × 256²)` = `O(3,276,800)`
- **Dense output**: `O(lstm_units × vocab_size)` ≈ `O(256 × 28)` = `O(7,168)`
- **Total**: `O(sequence_length × lstm_units²)` - dominated by LSTM computation
- **Typical time**: ~1-5ms per prediction on CPU, <1ms on GPU

**Memory Requirements:**
- **Model weights**: ~850K parameters × 4 bytes = ~3.4 MB (no embedding layer)
- **Input data**: `sequence_length × vocab_size × 4 bytes` = ~5.6 KB per sample
- **Intermediate activations**: ~50KB per prediction
- **Case histories**: ~100 bytes per case

---

## Installation

### Requirements

```bash
pip install tensorflow pandas numpy pm4py
```

### Optional Dependencies

- `matplotlib`, `seaborn`: For evaluation and visualization

---

## Quick Start

### Training Model

```python
from next_activity_prediction_onehot import NextActivityConfigOneHot, train_model_onehot

config = NextActivityConfigOneHot(
    event_log_path="eventlog/eventlog.xes.gz",
    model_dir="models/next_activity_lstm_onehot",
    epochs=50,
    batch_size=64,
    sequence_length=50
)

train_model_onehot(config)
```

### Command Line Training

```bash
python -m next_activity_prediction_onehot.trainer eventlog/eventlog.xes.gz
```

### Using the Predictor

```python
from next_activity_prediction_onehot import LSTMNextActivityPredictorOneHot
from simulation.engine import CaseState

# Initialize predictor
predictor = LSTMNextActivityPredictorOneHot(
    model_path="models/next_activity_lstm_onehot"
)

# Create a case state
case_state = CaseState(
    case_id="case_1",
    case_type="Home improvement",
    application_type="New credit",
    requested_amount=10000.0
)

# Set initial activity history
case_state.activity_history = ["A_Create Application", "A_Submitted"]

# Predict next activity
next_activity, is_end = predictor.predict(case_state)
print(f"Next activity: {next_activity}, Is end: {is_end}")

# Continue prediction
case_state.activity_history.append(next_activity)
next_activity, is_end = predictor.predict(case_state)
```

---

## Detailed Usage

### Initialization

```python
from next_activity_prediction_onehot import LSTMNextActivityPredictorOneHot

predictor = LSTMNextActivityPredictorOneHot(
    model_path="models/next_activity_lstm_onehot",  # Path to model directory
    seed=42  # Optional: random seed for reproducibility
)
```

### Prediction

The `predict()` method returns a tuple `(next_activity, is_case_ended)`:

```python
next_activity, is_end = predictor.predict(case_state)

if is_end:
    print(f"Case completed with final activity: {next_activity}")
else:
    print(f"Next activity: {next_activity}")
```

### Case Management

```python
# Reset a specific case
predictor.reset_case("case_1")

# Clear all case histories
predictor.clear()
```

---

## Model Architecture

### Stacked LSTM with One-Hot Input

The model architecture consists of:

1. **Input Layer**: Accepts one-hot encoded sequences of shape `(sequence_length, vocab_size)`
2. **LSTM Layers**: Stacked LSTM layers (default: 2 layers, 256 units each)
3. **Dropout**: Applied between LSTM layers and before output
4. **Output Layer**: Dense layer with softmax activation over vocabulary

**Model Parameters:**
- No embedding layer (unlike embedding-based version)
- LSTM parameters: `4 × lstm_units × (lstm_units + vocab_size + 1)` per layer
- For default (256 units, 28 vocab): ~290K parameters per LSTM layer

**Data Flow:**
```
Input: (batch_size, sequence_length, vocab_size)
  ↓
LSTM Layer 1 (return_sequences=True)
  → (batch_size, sequence_length, lstm_units)
  ↓
Dropout
  ↓
LSTM Layer 2 (return_sequences=False)
  → (batch_size, lstm_units)
  ↓
Final Dropout
  ↓
Dense Output (softmax)
  → (batch_size, vocab_size)
```

---

## Data Preprocessing

### One-Hot Encoding Process

1. **Load Event Log**: Load from XES or CSV file
2. **Filter Lifecycles**: Keep only "start" and "complete" transitions
3. **Extract Sequences**: Group activities by case, append END token
4. **Create Vocabulary**: Map activities to integer indices
5. **Pad/Truncate**: Normalize sequences to fixed length
6. **One-Hot Encode**: Convert integer sequences to one-hot vectors

### Example

```python
from next_activity_prediction_onehot.data_preprocessing import (
    load_event_log,
    filter_lifecycles,
    extract_case_sequences,
    create_vocabulary,
    prepare_training_data
)

# Load and preprocess
df = load_event_log("eventlog/eventlog.xes.gz")
df = filter_lifecycles(df)
sequences = extract_case_sequences(df, min_length=2, max_length=200)
activity_to_idx, idx_to_activity = create_vocabulary(sequences)

# Prepare training data with one-hot encoding
X_onehot, y_activity, _ = prepare_training_data(
    sequences,
    activity_to_idx,
    sequence_length=50,
    return_position_info=False
)

print(f"Input shape: {X_onehot.shape}")  # (n_samples, 50, vocab_size)
print(f"Output shape: {y_activity.shape}")  # (n_samples,)
```

---

## Training

### Configuration

```python
from next_activity_prediction_onehot import NextActivityConfigOneHot

config = NextActivityConfigOneHot(
    # Model architecture
    sequence_length=50,
    lstm_units=256,
    lstm_layers=2,
    dropout_rate=0.3,
    
    # Training
    batch_size=64,
    learning_rate=0.001,
    epochs=50,
    validation_split=0.2,
    early_stopping_patience=10,
    
    # Paths
    model_dir="models/next_activity_lstm_onehot",
    event_log_path="eventlog/eventlog.xes.gz",
    
    # Data preprocessing
    min_case_length=2,
    max_case_length=200,
    
    # Class weighting
    use_class_weights=True,
    end_token_weight=None,
    class_weight_method="balanced",
    
    # Position weighting
    use_position_weights=False,
    position_weight_power=1.5
)
```

### Training Process

```python
from next_activity_prediction_onehot import train_model_onehot

history = train_model_onehot(config)
```

The training process:
1. Loads and preprocesses the event log
2. Creates vocabulary and one-hot encodes sequences
3. Splits data into train/validation sets
4. Builds model with one-hot input architecture
5. Trains with early stopping and model checkpointing
6. Saves model and metadata

---

## Configuration

### NextActivityConfigOneHot

All configuration parameters:

- **sequence_length** (int): Input sequence length (default: 50)
- **lstm_units** (int): Number of LSTM units per layer (default: 256)
- **lstm_layers** (int): Number of LSTM layers (default: 2)
- **dropout_rate** (float): Dropout rate (default: 0.3)
- **batch_size** (int): Training batch size (default: 64)
- **learning_rate** (float): Learning rate (default: 0.001)
- **epochs** (int): Maximum training epochs (default: 50)
- **validation_split** (float): Validation split ratio (default: 0.2)
- **early_stopping_patience** (int): Early stopping patience (default: 10)
- **model_dir** (str): Model directory path
- **event_log_path** (str): Path to event log file
- **min_case_length** (int): Minimum case length (default: 2)
- **max_case_length** (int): Maximum case length (default: 200)
- **use_class_weights** (bool): Use class weights (default: True)
- **end_token_weight** (float): Manual END token weight (default: None)
- **class_weight_method** (str): Class weight method (default: "balanced")
- **use_position_weights** (bool): Use position weights (default: False)
- **position_weight_power** (float): Position weight power (default: 1.5)

**Note**: Unlike the embedding-based module, there is no `embedding_dim` parameter since one-hot encoding uses the vocabulary size directly.

---

## Integration with Simulation Engine

The predictor implements the `NextActivityPredictor` protocol and integrates seamlessly with the simulation engine:

```python
from next_activity_prediction_onehot import LSTMNextActivityPredictorOneHot
from integration.setup import SimulationSetup

# Initialize predictor
predictor = LSTMNextActivityPredictorOneHot(
    model_path="models/next_activity_lstm_onehot"
)

# Use in simulation setup
setup = SimulationSetup(
    next_activity_predictor=predictor,
    # ... other components
)
```

The predictor automatically:
- Maintains case histories
- Handles empty histories (returns starting activity)
- Detects case completion via END token
- Clears histories when cases end

---

## File Structure

```
next_activity_prediction_onehot/
├── __init__.py                 # Module exports
├── config.py                   # Configuration class
├── data_preprocessing.py       # One-hot encoding and preprocessing
├── model.py                    # Model architecture (no embedding)
├── predictor.py                # Predictor class
├── trainer.py                  # Training script
├── utils.py                    # Utility functions
└── README.md                   # This file
```

---

## Comparison: One-Hot vs Embedding

| Aspect | One-Hot Encoding | Embedding Vectors |
|--------|------------------|-------------------|
| **Input Shape** | `(sequence_length, vocab_size)` | `(sequence_length,)` |
| **Embedding Layer** | None | Required |
| **Input Size** | Larger (sparse) | Smaller (dense) |
| **Model Parameters** | Fewer (no embedding) | More (embedding weights) |
| **Interpretability** | High (explicit) | Lower (learned) |
| **Vocabulary Size** | Works best < 100 | Scales better |
| **Memory (Input)** | Higher | Lower |
| **Memory (Model)** | Lower | Higher |

**Recommendation:**
- Use **one-hot encoding** for smaller vocabularies (< 50 activities) and when interpretability is important
- Use **embedding vectors** for larger vocabularies (> 50 activities) and when you want learned representations

---

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Vocabularies**
   - One-hot encoding can be memory-intensive for large vocabularies
   - Consider using embedding-based module for vocabularies > 100 activities

2. **Model Not Learning**
   - Check vocabulary size matches between training and prediction
   - Verify one-hot encoding is correct (exactly one 1 per timestep)

3. **END Token Not Detected**
   - Ensure END token is in vocabulary
   - Check `end_token_idx` in metadata

---

## License

Same as parent project.

