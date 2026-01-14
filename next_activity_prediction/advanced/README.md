# Next Activity Prediction and Suffix Prediction

This module provides two LSTM-based prediction models for process simulation: **Next Activity Prediction** and **Suffix Prediction**. Both models filter event logs to only include "start" and "complete" lifecycle transitions and integrate seamlessly with the simulation engine.

## Table of Contents

- [Overview](#overview)
- [Next Activity Predictor](#next-activity-predictor)
- [Suffix Predictor](#suffix-predictor)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Model Architectures](#model-architectures)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Configuration](#configuration)
- [Integration with Simulation Engine](#integration-with-simulation-engine)
- [Evaluation](#evaluation)
- [File Structure](#file-structure)

---

## Overview

### Next Activity Predictor

The **Next Activity Predictor** is a single-step prediction model that predicts the next activity in a case sequence one activity at a time. It uses a stacked LSTM architecture to learn patterns from historical event logs and predicts the next activity given the current case history. The model treats "END" as a special activity token that indicates case completion.

**Key Features:**
- Predicts one activity at a time
- Maintains case history for stateful predictions
- Automatically detects case completion via END token
- Simple, efficient architecture
- Real-time prediction during simulation

### Suffix Predictor

The **Suffix Predictor** is a sequence-to-sequence model that predicts the entire remaining sequence (suffix) of activities for a case in a single prediction. It uses an encoder-decoder LSTM architecture to generate complete activity sequences. The predicted suffix is cached and returned one activity at a time as needed by the simulation engine.

**Key Features:**
- Predicts entire remaining sequence at once
- Encoder-decoder architecture for sequence generation
- Caches predicted suffixes for efficient retrieval
- Better long-term sequence coherence
- Reduces prediction calls during simulation

### When to Use Which?

- **Next Activity Predictor**: Use when you need fine-grained control, want to adapt predictions based on intermediate results, or prefer simpler model architecture.
- **Suffix Predictor**: Use when you want better sequence coherence, need to reduce prediction overhead, or want to generate complete traces more efficiently.

---

## Next Activity Predictor

### Architecture

The Next Activity Predictor uses a **stacked LSTM architecture**:

```
Input Sequence (padded/truncated)
    ↓
Embedding Layer (maps activity tokens to dense vectors)
    ↓
Stacked LSTM Layers (processes sequence context)
    ↓
Dense Output Layer (softmax over vocabulary)
    ↓
Next Activity Prediction (including END token)
```

**Components:**
- **Embedding Layer**: Maps activity tokens to dense vector representations
- **LSTM Layers**: Stacked LSTM layers that process the sequence context
- **Output Layer**: Dense layer with softmax activation predicting the next activity
- **Vocabulary**: Includes all activities plus special tokens (`<PAD>`, `END`)

### How It Works

#### Training Phase

1. **Sequence Preparation**: 
   - For each case sequence `[a₁, a₂, ..., aₙ, END]`
   - Create training samples: `(prefix, next_activity)` pairs
   - Example: `([a₁], a₂)`, `([a₁, a₂], a₃)`, ..., `([a₁, ..., aₙ], END)`

2. **Model Learning**:
   - Learn mapping: `f: prefix → P(next_activity | prefix)`
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
   sequence = pad_or_truncate(activity_history, sequence_length)
   X = [sequence]  # Shape: (1, sequence_length)

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

**Padding/Truncation Details:**
- **Left padding**: Shorter sequences padded with 0 (`<PAD>`) on the left
- **Right truncation**: Longer sequences keep most recent `sequence_length` activities
- **Example**: 
  - History: `["A_Create", "A_Submitted", "A_Concept"]` (length 3)
  - `sequence_length = 50`
  - Padded: `[0, 0, ..., 0, idx("A_Create"), idx("A_Submitted"), idx("A_Concept")]` (47 zeros + 3 activities)

**State Management:**
- Maintains `case_histories` dictionary: `{case_id: [activity_list]}`
- Syncs with `case_state.activity_history` (source of truth)
- Clears history when case ends

#### Computational Complexity

**Prediction Time:**
- **Embedding lookup**: `O(sequence_length × embedding_dim)` ≈ `O(50 × 128)` = `O(6,400)`
- **LSTM forward pass**: `O(sequence_length × lstm_units²)` ≈ `O(50 × 256²)` = `O(3,276,800)`
- **Dense output**: `O(lstm_units × vocab_size)` ≈ `O(256 × 28)` = `O(7,168)`
- **Total**: `O(sequence_length × lstm_units²)` - dominated by LSTM computation
- **Typical time**: ~1-5ms per prediction on CPU, <1ms on GPU

**Memory Requirements:**
- **Model weights**: ~930K parameters × 4 bytes = ~3.7 MB
- **Intermediate activations**: ~50KB per prediction
- **Case histories**: ~100 bytes per case (depends on history length)

### Example Usage

```python
from next_activity_prediction import LSTMNextActivityPredictor
from simulation.engine import CaseState

# Initialize predictor
predictor = LSTMNextActivityPredictor(
    model_path="models/next_activity_lstm"
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

## Suffix Predictor

### Architecture

The Suffix Predictor uses a **sequence-to-sequence (encoder-decoder) architecture**:

```
Input Prefix (padded/truncated)
    ↓
Embedding Layer
    ↓
Encoder LSTM Layers (processes prefix)
    ↓
Encoder State (context vector)
    ↓
Repeat Vector (for decoder timesteps)
    ↓
Decoder LSTM Layers (generates suffix)
    ↓
Time-Distributed Dense Output (softmax for each timestep)
    ↓
Predicted Suffix Sequence (including END token)
```

**Components:**
- **Encoder**: Stacked LSTM layers that encode the prefix into a context vector
- **Decoder**: Stacked LSTM layers that decode the context into a suffix sequence
- **Time-Distributed Output**: Dense layer applied at each decoder timestep
- **Caching**: Predicted suffixes are cached and returned incrementally

### How It Works

#### Training Phase

1. **Prefix-Suffix Pair Generation**:
   - For each case sequence `[a₁, a₂, ..., aₙ, END]`
   - Create training samples: `(prefix, suffix)` pairs
   - Example: 
     - `([a₁], [a₂, a₃, ..., END])`
     - `([a₁, a₂], [a₃, a₄, ..., END])`
     - `([a₁, a₂, a₃], [a₄, ..., END])`

2. **Model Learning**:
   - Learn mapping: `f: prefix → P(suffix | prefix)`
   - Encoder captures prefix context
   - Decoder generates suffix sequence
   - Learns to predict END token in suffix

#### Prediction Phase

**Algorithm:**

```
Input: case_state with activity_history = [a₁, a₂, ..., aₖ]

1. Handle empty history:
   If activity_history is empty:
     Return ("A_Create Application", False)

2. Check if new prediction needed:
   If case_id not in predicted_suffixes OR
      suffix_positions[case_id] >= len(predicted_suffixes[case_id]):
     → Need new prediction

3. Predict suffix (if needed):
   a. Prepare prefix:
      prefix = pad_or_truncate(activity_history, prefix_length)
      X = [prefix]  # Shape: (1, prefix_length)
   
   b. Forward pass:
      suffix_probs = model.predict(X)  # Shape: (1, suffix_length, vocab_size)
      suffix_probs = suffix_probs[0]  # Shape: (suffix_length, vocab_size)
   
   c. Greedy decoding:
      suffix = []
      For t in range(suffix_length):
        activity_idx = argmax(suffix_probs[t])
        activity = idx_to_activity[activity_idx]
        suffix.append(activity)
        If activity == "END":
          break
   
   d. Cache suffix:
      predicted_suffixes[case_id] = suffix
      suffix_positions[case_id] = 0

4. Retrieve next activity from cache:
   pos = suffix_positions[case_id]
   If pos >= len(predicted_suffixes[case_id]):
     → Suffix exhausted, end case
   
   next_activity = predicted_suffixes[case_id][pos]
   
   If next_activity == "END":
     → Clear cache, end case
   
   suffix_positions[case_id] = pos + 1
   Return (next_activity, False)
```

**Caching Strategy:**
- **Cache key**: `case_id`
- **Cache value**: `predicted_suffixes[case_id]` = list of activity names
- **Position tracking**: `suffix_positions[case_id]` = current index in suffix
- **Cache invalidation**: When suffix exhausted or END encountered

**Example Prediction Flow:**

```
Initial state:
  activity_history = ["A_Create", "A_Submitted"]
  case_id = "case_1"

Step 1: Predict suffix
  prefix = ["A_Create", "A_Submitted"] (padded to 50)
  model.predict() → suffix_probs (shape: (30, 28))
  Greedy decode → suffix = ["A_Concept", "W_Complete", "A_Accepted", ..., "END"]
  Cache: predicted_suffixes["case_1"] = suffix
         suffix_positions["case_1"] = 0

Step 2: Return first activity
  next_activity = suffix[0] = "A_Concept"
  suffix_positions["case_1"] = 1
  Return ("A_Concept", False)

Step 3: Return second activity
  next_activity = suffix[1] = "W_Complete"
  suffix_positions["case_1"] = 2
  Return ("W_Complete", False)

... (continue until END or suffix exhausted)
```

**Re-prediction Triggers:**
- Case history changes (new activities added)
- Suffix exhausted (reached end of predicted sequence)
- Case reset (explicitly cleared)

#### Computational Complexity

**Suffix Prediction (when needed):**
- **Encoder**: `O(prefix_length × encoder_lstm_units²)` ≈ `O(50 × 256²)` = `O(3,276,800)`
- **Decoder**: `O(suffix_length × decoder_lstm_units²)` ≈ `O(30 × 256²)` = `O(1,966,080)`
- **Time-distributed output**: `O(suffix_length × decoder_lstm_units × vocab_size)` ≈ `O(30 × 256 × 28)` = `O(215,040)`
- **Total**: `O(prefix_length × encoder_units² + suffix_length × decoder_units²)`
- **Typical time**: ~5-20ms per suffix prediction on CPU, ~1-3ms on GPU

**Activity Retrieval (from cache):**
- **Time**: `O(1)` - Simple array lookup
- **Memory**: `O(suffix_length)` per case (typically 10-30 activities)

**Memory Requirements:**
- **Model weights**: ~1.98M parameters × 4 bytes = ~7.9 MB
- **Intermediate activations**: ~200KB per suffix prediction
- **Cached suffixes**: ~500 bytes per active case (depends on suffix length)

### Example Usage

```python
from next_activity_prediction import LSTMSuffixPredictor
from simulation.engine import CaseState

# Initialize predictor
predictor = LSTMSuffixPredictor(
    model_path="models/suffix_prediction_lstm"
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

# Predict next activity (suffix is predicted and cached internally)
next_activity, is_end = predictor.predict(case_state)
print(f"Next activity: {next_activity}, Is end: {is_end}")

# Continue prediction (uses cached suffix)
case_state.activity_history.append(next_activity)
next_activity, is_end = predictor.predict(case_state)

# Access cached suffix if needed
if case_state.case_id in predictor.predicted_suffixes:
    suffix = predictor.predicted_suffixes[case_state.case_id]
    print(f"Predicted suffix: {suffix}")
```

---

## Installation

### Requirements

```bash
pip install tensorflow pandas numpy pm4py
```

### Optional Dependencies

- `matplotlib`, `seaborn`: For evaluation and visualization (see `evaluate_models.ipynb`)

---

## Quick Start

### Training Next Activity Model

```python
from next_activity_prediction import NextActivityConfig, train_model

config = NextActivityConfig(
    event_log_path="eventlog/eventlog.xes.gz",
    model_dir="models/next_activity_lstm",
    epochs=50,
    batch_size=64,
    sequence_length=50
)

train_model(config)
```

### Training Suffix Model

```python
from next_activity_prediction import train_suffix_model

model, metadata = train_suffix_model(
    event_log_path="eventlog/eventlog.xes.gz",
    model_dir="models/suffix_prediction_lstm",
    prefix_length=50,
    suffix_length=30,
    epochs=50,
    batch_size=64
)
```

### Command Line Training

**Next Activity Model:**
```bash
python -m next_activity_prediction.trainer eventlog/eventlog.xes.gz
```

**Suffix Model:**
Use the Jupyter notebook `train_suffix_model.ipynb` or call `train_suffix_model()` directly.

---

## Detailed Usage

### Next Activity Predictor

#### Initialization

```python
from next_activity_prediction import LSTMNextActivityPredictor

predictor = LSTMNextActivityPredictor(
    model_path="models/next_activity_lstm",  # Path to model directory
    seed=42  # Optional: random seed for reproducibility
)
```

#### Prediction

The `predict()` method returns a tuple `(next_activity, is_case_ended)`:

```python
next_activity, is_end = predictor.predict(case_state)

if is_end:
    print(f"Case completed with final activity: {next_activity}")
else:
    print(f"Next activity: {next_activity}")
```

#### Case Management

```python
# Reset a specific case
predictor.reset_case("case_1")

# Clear all case histories
predictor.clear()
```

### Suffix Predictor

#### Initialization

```python
from next_activity_prediction import LSTMSuffixPredictor

predictor = LSTMSuffixPredictor(
    model_path="models/suffix_prediction_lstm",
    seed=42
)
```

#### Prediction

The `predict()` method works similarly but caches the suffix internally:

```python
next_activity, is_end = predictor.predict(case_state)
```

#### Accessing Cached Suffix

```python
# Check if suffix is cached
if case_state.case_id in predictor.predicted_suffixes:
    suffix = predictor.predicted_suffixes[case_state.case_id]
    position = predictor.suffix_positions[case_state.case_id]
    print(f"Suffix: {suffix}")
    print(f"Current position: {position}")
```

#### Case Management

```python
# Reset a specific case (clears cached suffix)
predictor.reset_case("case_1")

# Clear all cached suffixes
predictor.clear()
```

---

## Model Architectures

### Next Activity Model Architecture

#### Input Layer

**Input Tensor:**
- Shape: `(batch_size, sequence_length)`
- Type: `int32` - Integer indices representing activity sequence
- Example: `[[0, 0, 0, 5, 12, 8, 3, ...], ...]` where 0 is padding, others are activity indices
- Padding: Left-padded with 0 (`<PAD>` token) for sequences shorter than `sequence_length`
- Truncation: Right-truncated (keeps most recent activities) for sequences longer than `sequence_length`

#### Embedding Layer

**Purpose:** Maps discrete activity tokens to dense vector representations in a continuous space.

**Technical Details:**
- **Input dimension**: `vocab_size` (typically 20-50 activities + special tokens)
- **Output dimension**: `embedding_dim` (default: 128)
- **Embedding matrix**: `W_emb ∈ R^(vocab_size × embedding_dim)`
- **Operation**: `E[i] = W_emb[i, :]` where `i` is the activity index
- **Mask zero**: `True` - Padding tokens (index 0) are masked and don't contribute to gradients
- **Output shape**: `(batch_size, sequence_length, embedding_dim)`

**Mathematical Formulation:**
```
For input sequence x = [x₁, x₂, ..., xₜ] where xᵢ ∈ {0, 1, ..., V-1}:
E = [W_emb[x₁], W_emb[x₂], ..., W_emb[xₜ]]
```

**Why Embeddings:**
- Captures semantic relationships between activities
- Allows the model to learn activity similarities
- Reduces dimensionality compared to one-hot encoding
- Enables transfer learning from pre-trained embeddings

#### LSTM Layers (Stacked)

**Purpose:** Process sequential context to capture temporal dependencies and long-range patterns.

**LSTM Cell Architecture:**

Each LSTM cell maintains:
- **Hidden state** `h_t`: Current output (shape: `(batch_size, lstm_units)`)
- **Cell state** `c_t`: Long-term memory (shape: `(batch_size, lstm_units)`)

**LSTM Gates (for each timestep t):**

1. **Forget Gate** `f_t`: Decides what to forget from cell state
   ```
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   ```
   where `σ` is sigmoid, `W_f` is weight matrix, `b_f` is bias

2. **Input Gate** `i_t`: Decides what new information to store
   ```
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   ```

3. **Cell State Update**:
   ```
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t
   ```
   where `⊙` is element-wise multiplication

4. **Output Gate** `o_t`: Decides what parts of cell state to output
   ```
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t ⊙ tanh(c_t)
   ```

**Stacked LSTM Configuration:**
- **Number of layers**: `lstm_layers` (default: 2)
- **Units per layer**: `lstm_units` (default: 256)
- **Return sequences**: 
  - Intermediate layers: `True` (pass full sequence to next layer)
  - Final layer: `False` (only return final hidden state)
- **Dropout**: Applied to inputs (`dropout_rate`) and recurrent connections (`recurrent_dropout`)
- **Output shape (final layer)**: `(batch_size, lstm_units)`

**Data Flow Through Stacked LSTMs:**
```
Input: (batch_size, sequence_length, embedding_dim)
  ↓
LSTM Layer 1 (return_sequences=True)
  → (batch_size, sequence_length, lstm_units)
  ↓
Dropout (dropout_rate)
  ↓
LSTM Layer 2 (return_sequences=False)
  → (batch_size, lstm_units)
  ↓
Final Dropout (dropout_rate)
```

**Memory and Computation:**
- **Parameters per LSTM layer**: `4 × lstm_units × (lstm_units + input_dim + 1)`
  - For default (256 units, 128 embedding): ~394K parameters per layer
- **Memory per sample**: `O(sequence_length × lstm_units)`
- **Computation**: `O(sequence_length × lstm_units²)` per layer

#### Dense Output Layer

**Purpose:** Maps LSTM hidden state to probability distribution over vocabulary.

**Technical Details:**
- **Input shape**: `(batch_size, lstm_units)`
- **Weight matrix**: `W_out ∈ R^(lstm_units × vocab_size)`
- **Bias vector**: `b_out ∈ R^(vocab_size)`
- **Activation**: Softmax
- **Output shape**: `(batch_size, vocab_size)`

**Mathematical Formulation:**
```
z = W_out · h_final + b_out
P(activity = i | sequence) = softmax(z)[i] = exp(z_i) / Σⱼ exp(z_j)
```

**Softmax Properties:**
- Outputs valid probability distribution: `Σᵢ P(i) = 1`
- All probabilities are non-negative: `P(i) ≥ 0`
- Amplifies differences: largest logit gets highest probability

#### Complete Data Flow

```
Input: (batch_size, sequence_length)
  ↓ [Embedding lookup]
Embeddings: (batch_size, sequence_length, embedding_dim)
  ↓ [LSTM Layer 1]
LSTM1 Output: (batch_size, sequence_length, lstm_units)
  ↓ [LSTM Layer 2]
LSTM2 Hidden: (batch_size, lstm_units)
  ↓ [Dense + Softmax]
Output: (batch_size, vocab_size) - Probability distribution
```

#### Total Model Parameters

For default configuration (vocab_size=28, embedding_dim=128, lstm_units=256, lstm_layers=2, sequence_length=50):

- **Embedding**: `28 × 128 = 3,584` parameters
- **LSTM Layer 1**: `4 × 256 × (256 + 128 + 1) = 394,240` parameters
- **LSTM Layer 2**: `4 × 256 × (256 + 256 + 1) = 525,312` parameters
- **Dense Output**: `256 × 28 + 28 = 7,196` parameters
- **Total**: ~930,000 trainable parameters

#### Output Interpretation

**Output Tensor:**
- Shape: `(batch_size, vocab_size)`
- Values: Probabilities (sum to 1.0 per sample)
- Prediction: `argmax(output)` gives the most likely next activity
- Example: `[0.01, 0.05, 0.80, 0.10, 0.04]` → Activity index 2 (80% confidence)

### Suffix Model Architecture

The suffix model uses a **sequence-to-sequence (seq2seq) encoder-decoder architecture** to predict entire activity sequences rather than single activities.

#### Input Layer

**Input Tensor:**
- Shape: `(batch_size, prefix_length)`
- Type: `int32` - Integer indices representing prefix sequence
- Example: `[[0, 0, 5, 12, 8], ...]` - Left-padded prefix
- Padding: Left-padded with 0 for shorter prefixes
- Truncation: Right-truncated for longer prefixes

#### Encoder Architecture

**Purpose:** Encode the input prefix into a fixed-size context vector that captures the entire sequence information.

##### Embedding Layer

Same as next activity model:
- **Input dimension**: `vocab_size`
- **Output dimension**: `embedding_dim` (default: 128)
- **Output shape**: `(batch_size, prefix_length, embedding_dim)`

##### Encoder LSTM Layers

**Configuration:**
- **Number of layers**: `encoder_lstm_layers` (default: 2)
- **Units per layer**: `encoder_lstm_units` (default: 256)
- **Return sequences**: 
  - Intermediate layers: `True` (pass full sequence)
  - Final layer: `False` (only final hidden state)
- **Return state**: `True` (returns both hidden state `h` and cell state `c`)
- **Dropout**: Applied to inputs and recurrent connections

**Encoder Data Flow:**
```
Input: (batch_size, prefix_length)
  ↓ [Embedding]
Embeddings: (batch_size, prefix_length, embedding_dim)
  ↓ [Encoder LSTM Layer 1]
LSTM1 Output: (batch_size, prefix_length, encoder_lstm_units)
LSTM1 States: (h₁, c₁) - (batch_size, encoder_lstm_units) each
  ↓ [Encoder LSTM Layer 2]
LSTM2 Hidden: (batch_size, encoder_lstm_units)  ← Final context vector
LSTM2 States: (h₂, c₂) - (batch_size, encoder_lstm_units) each
```

**Context Vector:**
- **Shape**: `(batch_size, encoder_lstm_units)`
- **Represents**: Encoded information about the entire prefix sequence
- **Properties**: Fixed-size representation regardless of prefix length

#### Decoder Architecture

**Purpose:** Generate the suffix sequence from the encoded context vector.

##### Repeat Vector Layer

**Purpose:** Convert the single context vector into a sequence for the decoder.

**Operation:**
- **Input**: `(batch_size, encoder_lstm_units)` - Single context vector
- **Output**: `(batch_size, suffix_length, encoder_lstm_units)` - Repeated context
- **Function**: `RepeatVector(suffix_length)(context)`

**Mathematical Formulation:**
```
If context = [h₁, h₂, ..., h_d] (shape: (batch_size, d))
Then RepeatVector output = [[h₁, h₂, ..., h_d],
                            [h₁, h₂, ..., h_d],
                            ...
                            [h₁, h₂, ..., h_d]]  (shape: (batch_size, suffix_length, d))
```

**Why Repeat Vector:**
- Decoder LSTM needs input at each timestep
- Same context is used to initialize each decoder timestep
- Simpler than teacher forcing during inference

##### Decoder LSTM Layers

**Configuration:**
- **Number of layers**: `decoder_lstm_layers` (default: 2)
- **Units per layer**: `decoder_lstm_units` (default: 256)
- **Return sequences**: `True` (always, for all layers)
- **Input**: Repeated context vector from encoder
- **Output shape**: `(batch_size, suffix_length, decoder_lstm_units)`

**Decoder Data Flow:**
```
Repeated Context: (batch_size, suffix_length, encoder_lstm_units)
  ↓ [Decoder LSTM Layer 1]
LSTM1 Output: (batch_size, suffix_length, decoder_lstm_units)
  ↓ [Decoder LSTM Layer 2]
LSTM2 Output: (batch_size, suffix_length, decoder_lstm_units)
```

**Note:** Unlike traditional seq2seq with attention, this implementation uses a simpler approach where the encoder's final state is repeated. This works well for process prediction where the entire prefix context is relevant for generating the suffix.

##### Time-Distributed Dense Output

**Purpose:** Apply the same dense layer to each timestep of the decoder output.

**Technical Details:**
- **Input shape**: `(batch_size, suffix_length, decoder_lstm_units)`
- **Operation**: For each timestep `t`, apply:
  ```
  z_t = W_out · h_t + b_out
  P_t = softmax(z_t)
  ```
- **Weight matrix**: `W_out ∈ R^(decoder_lstm_units × vocab_size)` (shared across timesteps)
- **Bias vector**: `b_out ∈ R^(vocab_size)` (shared across timesteps)
- **Output shape**: `(batch_size, suffix_length, vocab_size)`

**Time-Distributed Operation:**
```
For each timestep t in [0, suffix_length-1]:
  Apply Dense layer to decoder_output[:, t, :]
  Result: predictions[:, t, :]  (shape: (batch_size, vocab_size))
```

#### Complete Encoder-Decoder Data Flow

```
Encoder:
  Input: (batch_size, prefix_length)
    ↓ [Embedding]
  Embeddings: (batch_size, prefix_length, embedding_dim)
    ↓ [Encoder LSTM Layers]
  Context: (batch_size, encoder_lstm_units)

Decoder:
  Context: (batch_size, encoder_lstm_units)
    ↓ [RepeatVector]
  Repeated: (batch_size, suffix_length, encoder_lstm_units)
    ↓ [Decoder LSTM Layers]
  Decoder Output: (batch_size, suffix_length, decoder_lstm_units)
    ↓ [TimeDistributed Dense + Softmax]
  Output: (batch_size, suffix_length, vocab_size)
```

#### Total Model Parameters

For default configuration (vocab_size=28, embedding_dim=128, encoder_lstm_units=256, decoder_lstm_units=256, encoder_lstm_layers=2, decoder_lstm_layers=2, prefix_length=50, suffix_length=30):

- **Embedding**: `28 × 128 = 3,584` parameters
- **Encoder LSTM Layer 1**: `4 × 256 × (256 + 128 + 1) = 394,240` parameters
- **Encoder LSTM Layer 2**: `4 × 256 × (256 + 256 + 1) = 525,312` parameters
- **Decoder LSTM Layer 1**: `4 × 256 × (256 + 256 + 1) = 525,312` parameters
- **Decoder LSTM Layer 2**: `4 × 256 × (256 + 256 + 1) = 525,312` parameters
- **Time-Distributed Dense**: `256 × 28 + 28 = 7,196` parameters
- **Total**: ~1,980,000 trainable parameters

#### Output Interpretation

**Output Tensor:**
- Shape: `(batch_size, suffix_length, vocab_size)`
- Values: Probability distributions (each timestep sums to 1.0)
- Structure: `output[b, t, :]` = probability distribution for activity at position `t` in suffix for batch `b`

**Prediction Process:**
1. For each timestep `t` in `[0, suffix_length-1]`:
   - Extract: `probs_t = output[:, t, :]` (shape: `(batch_size, vocab_size)`)
   - Predict: `activity_t = argmax(probs_t)` (greedy decoding)
2. Stop early if `activity_t == END_token_idx`
3. Return sequence: `[activity_0, activity_1, ..., activity_k, END]`

**Example Output:**
```
Output shape: (1, 30, 28)
At timestep 0: [0.01, 0.05, 0.80, 0.10, 0.04, ...] → Activity 2
At timestep 1: [0.02, 0.70, 0.10, 0.15, 0.03, ...] → Activity 1
At timestep 2: [0.05, 0.10, 0.75, 0.05, 0.05, ...] → Activity 2
...
At timestep 8: [0.01, 0.01, 0.01, 0.01, 0.90, ...] → END token
```

**Greedy vs. Beam Search:**
- **Current implementation**: Greedy decoding (argmax at each step)
- **Alternative**: Beam search (maintain top-k candidates) - not implemented but possible
- **Trade-off**: Greedy is faster, beam search may find better sequences

---

## Data Preprocessing

### Event Log Format

Both models expect event logs in **XES** or **CSV** format with the following columns:
- `case:concept:name`: Case identifier
- `concept:name`: Activity name
- `time:timestamp`: Event timestamp
- `lifecycle:transition`: Lifecycle transition (optional, but recommended)

### Preprocessing Pipeline

1. **Load Event Log**
   - Supports `.xes`, `.xes.gz`, `.csv`, `.csv.gz`
   - Uses `pm4py` for XES files

2. **Filter Lifecycles**
   - Keeps only `start` and `complete` transitions
   - Removes `suspend`, `resume`, `schedule`, etc.

3. **Extract Case Sequences**
   - Groups events by case ID
   - Sorts by timestamp
   - Filters by `min_case_length` and `max_case_length`
   - Appends `END` token to each sequence

4. **Create Vocabulary**
   - Maps activity names to integer indices
   - Includes special tokens: `<PAD>` (0), `END` (last index)
   - Sorted alphabetically (except END)

5. **Prepare Training Data**

   **Next Activity Model:**
   
   **Algorithm:**
   ```
   For each sequence seq = [a₁, a₂, ..., aₙ, END]:
     For each position i in [1, len(seq)-1]:
       prefix = seq[:i]  # [a₁, a₂, ..., aᵢ]
       next_activity = seq[i]  # a_{i+1} or END
       
       prefix_padded = pad_or_truncate(prefix, sequence_length)
       next_idx = activity_to_idx[next_activity]
       
       X.append(prefix_padded)
       y.append(next_idx)
   ```
   
   **Example:**
   - Sequence: `["A_Create", "A_Submitted", "A_Concept", "END"]` (length 4)
   - Samples created:
     - `(prefix=["A_Create"], next="A_Submitted")`
     - `(prefix=["A_Create", "A_Submitted"], next="A_Concept")`
     - `(prefix=["A_Create", "A_Submitted", "A_Concept"], next="END")`
   - Total: 3 samples from 1 sequence
   
   **Output Shapes:**
   - `X`: `(n_samples, sequence_length)` - integer indices
   - `y`: `(n_samples,)` - integer indices (next activity)

   **Suffix Model:**
   
   **Algorithm:**
   ```
   For each sequence seq = [a₁, a₂, ..., aₙ, END]:
     For each position i in [min_prefix_length, len(seq)-1]:
       prefix = seq[:i]  # [a₁, a₂, ..., aᵢ]
       suffix = seq[i:]  # [a_{i+1}, a_{i+2}, ..., END]
       
       prefix_padded = pad_or_truncate(prefix, prefix_length)
       suffix_padded = pad_or_truncate(suffix, suffix_length)
       
       prefix_indices = [activity_to_idx[a] for a in prefix_padded]
       suffix_indices = [activity_to_idx[a] for a in suffix_padded]
       
       X_prefix.append(prefix_indices)
       y_suffix.append(suffix_indices)
   ```
   
   **Example:**
   - Sequence: `["A_Create", "A_Submitted", "A_Concept", "A_Accepted", "END"]` (length 5)
   - `min_prefix_length = 1`
   - Samples created:
     - `(prefix=["A_Create"], suffix=["A_Submitted", "A_Concept", "A_Accepted", "END"])`
     - `(prefix=["A_Create", "A_Submitted"], suffix=["A_Concept", "A_Accepted", "END"])`
     - `(prefix=["A_Create", "A_Submitted", "A_Concept"], suffix=["A_Accepted", "END"])`
     - `(prefix=["A_Create", "A_Submitted", "A_Concept", "A_Accepted"], suffix=["END"])`
   - Total: 4 samples from 1 sequence
   
   **Output Shapes:**
   - `X_prefix`: `(n_samples, prefix_length)` - integer indices
   - `y_suffix`: `(n_samples, suffix_length)` - integer indices

6. **Padding/Truncation Algorithm**

   **Padding Function:**
   ```
   Function pad_sequence(sequence, max_length, pad_value=0):
     If len(sequence) < max_length:
       padding = [pad_value] × (max_length - len(sequence))
       Return padding + sequence  # Left padding
     Else:
       Return sequence[-max_length:]  # Right truncation
   ```
   
   **Example - Padding:**
   - Sequence: `[5, 12, 8]` (length 3)
   - `max_length = 50`
   - Result: `[0, 0, ..., 0, 5, 12, 8]` (47 zeros + 3 activities)
   
   **Example - Truncation:**
   - Sequence: `[1, 2, 3, ..., 60]` (length 60)
   - `max_length = 50`
   - Result: `[11, 12, 13, ..., 60]` (last 50 activities)
   
   **Why Left Padding:**
   - Keeps most recent activities at the end (right side)
   - LSTM processes left-to-right, so recent context is at the end
   - Padding tokens are masked (don't affect computation)

**Data Statistics After Preprocessing:**

**Next Activity Model:**
- **Training samples**: Typically 10-20× number of cases (multiple samples per case)
- **Sample distribution**: More samples from longer cases
- **END token frequency**: ~5-15% of samples (depends on case lengths)

**Suffix Model:**
- **Training samples**: Similar to next activity (multiple prefix-suffix pairs per case)
- **Prefix length distribution**: Varies from `min_prefix_length` to case length
- **Suffix length distribution**: Varies from 1 to case length (decreasing as prefix grows)

### Data Statistics

After preprocessing, you'll see logs like:
```
Loaded 1,202,267 events, 31,509 cases
Filtered to start/complete lifecycles: 1,202,267 -> 603,533 (50.2%)
Extracted 31,509 case sequences
Average sequence length: 20.2 (including END)
Created vocabulary with 28 activities (including END)
Prepared 603,533 training samples
```

---

## Training

### Training Next Activity Model

#### Using Configuration Object

```python
from next_activity_prediction import NextActivityConfig, train_model

config = NextActivityConfig(
    event_log_path="eventlog/eventlog.xes.gz",
    model_dir="models/next_activity_lstm",
    
    # Architecture
    sequence_length=50,
    embedding_dim=128,
    lstm_units=256,
    lstm_layers=2,
    dropout_rate=0.3,
    
    # Training
    batch_size=64,
    learning_rate=0.001,
    epochs=50,
    validation_split=0.2,
    early_stopping_patience=10,
    
    # Data preprocessing
    min_case_length=2,
    max_case_length=200,
    
    # Advanced options
    use_class_weights=True,
    use_position_weights=False
)

history = train_model(config)
```

#### Training Process

**Detailed Training Pipeline:**

1. **Data Loading and Preprocessing**
   - Load event log from XES/CSV
   - Filter to start/complete lifecycles
   - Extract case sequences
   - Create vocabulary mapping
   - Generate training samples (input-output pairs)
   - Split into train/validation sets (default: 80/20)

2. **Model Building**
   - Initialize embedding layer with random weights
   - Initialize LSTM layers with Glorot uniform initialization
   - Initialize dense output layer
   - Compile model with optimizer and loss function

3. **Training Loop** (for each epoch):
   - **Forward Pass**: 
     - Process batch through model
     - Compute predictions: `ŷ = model(X)`
   - **Loss Computation**:
     - Calculate loss: `L = loss(y_true, ŷ)`
     - Apply class weights or sample weights if enabled
   - **Backward Pass**:
     - Compute gradients: `∇θ L` via backpropagation
     - Update parameters: `θ ← θ - α · ∇θ L` (Adam optimizer)
   - **Validation**:
     - Evaluate on validation set
     - Monitor validation loss and accuracy

4. **Early Stopping Check**
   - If validation loss doesn't improve for `patience` epochs:
     - Restore best model weights
     - Stop training

5. **Model Saving**
   - Save final model to `model.keras`
   - Save best checkpoint to `checkpoints/best_model.keras`
   - Save metadata (vocabulary, config, training history) to `metadata.json`

#### Loss Function: Sparse Categorical Crossentropy

**Mathematical Formulation:**

For a single sample with true label `y` (integer index) and predicted probabilities `ŷ`:

```
L(y, ŷ) = -log(ŷ[y])
```

For a batch of size `N`:

```
L = -(1/N) Σᵢ log(ŷᵢ[yᵢ])
```

**Properties:**
- **Range**: `[0, +∞)` (0 = perfect prediction, higher = worse)
- **Sparse**: Uses integer labels instead of one-hot vectors (memory efficient)
- **Interpretation**: Negative log-likelihood of the true class

**Example:**
- True label: `y = 5` (activity index 5)
- Predicted probabilities: `ŷ = [0.01, 0.02, 0.05, 0.10, 0.15, 0.60, 0.07]`
- Loss: `-log(0.60) ≈ 0.511`

#### Optimizer: Adam (Adaptive Moment Estimation)

**Adam Update Rule:**

For each parameter `θ`:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · ∇θ L_t      (first moment estimate)
v_t = β₂ · v_{t-1} + (1 - β₂) · (∇θ L_t)²   (second moment estimate)

m̂_t = m_t / (1 - β₁^t)                      (bias correction)
v̂_t = v_t / (1 - β₂^t)                      (bias correction)

θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε)
```

**Default Hyperparameters:**
- `α = 0.001` (learning rate)
- `β₁ = 0.9` (first moment decay)
- `β₂ = 0.999` (second moment decay)
- `ε = 1e-7` (numerical stability)

**Advantages:**
- Adaptive learning rate per parameter
- Handles sparse gradients well
- Good default choice for most problems

#### Training Callbacks

**EarlyStopping:**
- **Monitor**: `val_loss` (validation loss)
- **Patience**: Number of epochs to wait (default: 10)
- **Mode**: `min` (stop when validation loss stops decreasing)
- **Restore best weights**: `True` (revert to best model)
- **Effect**: Prevents overfitting by stopping when model stops improving

**ModelCheckpoint:**
- **Monitor**: `val_loss`
- **Save best only**: `True`
- **Filepath**: `checkpoints/best_model.keras`
- **Effect**: Saves model checkpoint after each epoch if validation loss improves

**ReduceLROnPlateau:**
- **Monitor**: `val_loss`
- **Factor**: `0.5` (reduce learning rate by 50%)
- **Patience**: `5` epochs
- **Min learning rate**: `1e-6`
- **Effect**: Reduces learning rate when validation loss plateaus, allowing fine-tuning

#### Class Weighting

**Purpose:** Handle imbalanced classes (e.g., END token is rare but important).

**Calculation Methods:**

1. **Balanced (sklearn-style)**:
   ```
   weight[i] = n_samples / (n_classes × count[i])
   ```
   - Normalizes by number of classes
   - Example: If class 5 appears 10 times and there are 28 classes with 1000 total samples:
     `weight[5] = 1000 / (28 × 10) = 3.57`

2. **Inverse Frequency**:
   ```
   weight[i] = n_samples / count[i]
   ```
   - Directly inverse to frequency
   - Example: `weight[5] = 1000 / 10 = 100`

3. **Custom END Token Weight**:
   - Override END token weight manually
   - Example: `end_token_weight = 2.0` (double the importance)

**Application:**
- During loss computation: `weighted_loss = weight[y] × loss(y, ŷ)`
- Rare classes get higher weights, encouraging the model to predict them more often

#### Position-Based Sample Weighting

**Purpose:** Emphasize later positions in sequences (more likely to predict END).

**Calculation:**
```
position_weight[i] = (relative_position[i] + ε)^power
normalized_weight[i] = position_weight[i] / mean(position_weights)
```

**Parameters:**
- `relative_position`: `[0.0, 1.0]` (0.0 = start, 1.0 = end)
- `power`: Default 1.5 (higher = more emphasis on later positions)
- `ε`: 0.1 (prevents zero weights)

**Example:**
- Position 0.2 (early): `weight = (0.2 + 0.1)^1.5 ≈ 0.16`
- Position 0.8 (late): `weight = (0.8 + 0.1)^1.5 ≈ 0.85`
- After normalization: weights scaled so mean = 1.0

**Combined with Class Weights:**
- Final sample weight: `sample_weight[i] = class_weight[y[i]] × position_weight[i]`
- Normalized to have mean = 1.0

### Training Suffix Model

#### Using Function Call

```python
from next_activity_prediction import train_suffix_model

model, metadata = train_suffix_model(
    event_log_path="eventlog/eventlog.xes.gz",
    model_dir="models/suffix_prediction_lstm",
    
    # Sequence lengths
    prefix_length=50,
    suffix_length=30,
    min_prefix_length=1,
    
    # Architecture
    embedding_dim=128,
    encoder_lstm_units=256,
    decoder_lstm_units=256,
    encoder_lstm_layers=2,
    decoder_lstm_layers=2,
    dropout_rate=0.3,
    
    # Training
    batch_size=64,
    learning_rate=0.001,
    epochs=50,
    validation_split=0.2,
    early_stopping_patience=10,
    
    # Data preprocessing
    min_case_length=2,
    max_case_length=200,
    random_seed=42
)
```

#### Training Process

**Detailed Training Pipeline:**

1. **Data Loading and Preprocessing**
   - Same as next activity model (filter lifecycles, extract sequences)
   - Create vocabulary mapping

2. **Prefix-Suffix Pair Generation**
   - For each case sequence `[a₁, a₂, ..., aₙ, END]`:
     - For each split position `i` in `[min_prefix_length, n]`:
       - Prefix: `[a₁, a₂, ..., aᵢ]` (padded/truncated to `prefix_length`)
       - Suffix: `[a_{i+1}, a_{i+2}, ..., END]` (padded/truncated to `suffix_length`)
   - Creates multiple training samples per case
   - Example: Case with 10 activities → 9 prefix-suffix pairs

3. **Model Building**
   - Initialize encoder (embedding + stacked LSTM)
   - Initialize decoder (stacked LSTM + time-distributed dense)
   - Compile with Adam optimizer and sparse categorical crossentropy

4. **Training Loop**
   - **Forward Pass**:
     - Encode prefix: `context = encoder(prefix)`
     - Decode suffix: `ŷ = decoder(context)`
     - Shape: `ŷ ∈ R^(batch_size, suffix_length, vocab_size)`
   - **Loss Computation**:
     - For each timestep `t` in suffix:
       - True label: `y_t` (integer index)
       - Predicted: `ŷ_t` (probability distribution)
       - Loss: `L_t = -log(ŷ_t[y_t])`
     - Total loss: `L = (1/suffix_length) Σₜ L_t` (average over timesteps)
   - **Backward Pass**: Backpropagation through time (BPTT)
   - **Parameter Update**: Adam optimizer

5. **Early Stopping and Saving**: Same as next activity model

#### Loss Function: Sparse Categorical Crossentropy (Sequence)

**Mathematical Formulation:**

For a sequence of length `T` with true labels `y = [y₁, y₂, ..., yₜ]` and predictions `ŷ = [ŷ₁, ŷ₂, ..., ŷₜ]`:

```
L = -(1/T) Σₜ log(ŷₜ[yₜ])
```

**Per-Timestep Loss:**
- Each timestep contributes equally to the total loss
- Loss is averaged over sequence length
- END token prediction is part of the sequence loss

**Example:**
- True suffix: `[5, 12, 8, END]` (length 4)
- Predicted probabilities at each timestep:
  - `ŷ₁[5] = 0.60` → `L₁ = -log(0.60) ≈ 0.511`
  - `ŷ₂[12] = 0.75` → `L₂ = -log(0.75) ≈ 0.288`
  - `ŷ₃[8] = 0.50` → `L₃ = -log(0.50) ≈ 0.693`
  - `ŷ₄[END] = 0.80` → `L₄ = -log(0.80) ≈ 0.223`
- Total loss: `(0.511 + 0.288 + 0.693 + 0.223) / 4 ≈ 0.429`

#### Training Challenges

**Teacher Forcing:**
- During training: Use true previous tokens as decoder input
- During inference: Use predicted tokens (exposure bias)
- Current implementation: Uses RepeatVector (simpler, no teacher forcing)

**Sequence Length Mismatch:**
- True suffix may be shorter than `suffix_length`
- Padding with `<PAD>` token (index 0)
- Loss masked for padding positions (via sparse categorical crossentropy)

**Gradient Flow:**
- Long sequences can cause vanishing/exploding gradients
- LSTM gates help with gradient flow
- Gradient clipping not implemented but recommended for very long sequences

### Training Output

Both training functions save:
- **Model file**: `model.keras` (final trained model)
- **Checkpoint**: `checkpoints/best_model.keras` (best model during training)
- **Metadata**: `metadata.json` (vocabulary, configuration, training history)

### Monitoring Training

Training logs include:
- Model architecture summary
- Training/validation loss and accuracy per epoch
- Early stopping notifications
- Final training metrics

Example output:
```
Epoch 1/50
482826/482826 [==============================] - 245s 508us/step - loss: 2.1234 - sparse_categorical_accuracy: 0.4567 - val_loss: 1.9876 - val_sparse_categorical_accuracy: 0.5123

Epoch 2/50
...
Epoch 15/50
Restoring model weights from the end of the best epoch.
Epoch 15: early stopping

Final train loss: 0.9503, val loss: 0.8882
Final train accuracy: 0.7331, val accuracy: 0.7466
```

---

## Configuration

### Next Activity Config

```python
@dataclass
class NextActivityConfig:
    # Model architecture
    sequence_length: int = 50          # Input sequence length
    embedding_dim: int = 128           # Embedding dimension
    lstm_units: int = 256              # LSTM units per layer
    lstm_layers: int = 2                # Number of LSTM layers
    dropout_rate: float = 0.3          # Dropout rate
    
    # Training
    batch_size: int = 64               # Batch size
    learning_rate: float = 0.001       # Learning rate
    epochs: int = 50                   # Maximum epochs
    validation_split: float = 0.2      # Validation split
    early_stopping_patience: int = 10  # Early stopping patience
    
    # Paths
    model_dir: str = "models/next_activity_lstm"
    event_log_path: Optional[str] = None
    
    # Data preprocessing
    min_case_length: int = 2           # Minimum case length
    max_case_length: int = 200         # Maximum case length
    
    # Class weighting
    use_class_weights: bool = True     # Use class weights
    end_token_weight: Optional[float] = None  # Manual END token weight
    class_weight_method: str = "balanced"  # Weight calculation method
    
    # Position weighting
    use_position_weights: bool = False  # Weight by position in case
    position_weight_power: float = 1.5  # Position weight power
```

### Suffix Model Parameters

```python
train_suffix_model(
    # Sequence lengths
    prefix_length: int = 50,           # Input prefix length
    suffix_length: int = 30,           # Output suffix length
    min_prefix_length: int = 1,       # Minimum prefix length
    
    # Architecture
    embedding_dim: int = 128,
    encoder_lstm_units: int = 256,
    decoder_lstm_units: int = 256,
    encoder_lstm_layers: int = 2,
    decoder_lstm_layers: int = 2,
    dropout_rate: float = 0.3,
    
    # Training
    batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 50,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    
    # Data preprocessing
    min_case_length: int = 2,
    max_case_length: int = 200,
    random_seed: int = 42
)
```

### Hyperparameter Tuning Tips

**Sequence Length:**
- Longer sequences capture more context but require more memory
- Typical range: 30-100 for next activity, 20-50 for suffix

**LSTM Units:**
- More units = more capacity but slower training
- Typical range: 128-512

**LSTM Layers:**
- Deeper networks learn more complex patterns
- Typical range: 1-3 layers

**Dropout:**
- Prevents overfitting
- Typical range: 0.2-0.5

**Batch Size:**
- Larger batches = more stable gradients but more memory
- Typical range: 32-128

---

## Integration with Simulation Engine

Both predictors implement the `NextActivityPredictor` protocol, making them compatible with the simulation engine's auto-load mechanism.

### Auto-Loading

The simulation engine automatically loads predictors if trained models exist:
- Next Activity: `models/next_activity_lstm/`
- Suffix: `models/suffix_prediction_lstm/`

### Manual Integration

```python
from simulation.engine import SimulationEngine
from next_activity_prediction import LSTMNextActivityPredictor, LSTMSuffixPredictor

# Create predictor
predictor = LSTMNextActivityPredictor(model_path="models/next_activity_lstm")

# Or use suffix predictor
predictor = LSTMSuffixPredictor(model_path="models/suffix_prediction_lstm")

# Pass to simulation engine
engine = SimulationEngine(next_activity_predictor=predictor)
```

### Case State Requirements

The predictors expect `CaseState` objects with:
- `case_id`: Unique case identifier
- `activity_history`: List of activity names (strings)

Optional attributes (not used by predictors but may be used by engine):
- `case_type`
- `application_type`
- `requested_amount`
- Other case attributes

---

## Evaluation

### Evaluation Notebook

Use `evaluate_models.ipynb` to compare model performance:

1. **Load Models**: Load both trained models
2. **Generate Traces**: Generate traces using both models
3. **Compare Metrics**:
   - Case length distribution
   - Activity distribution
   - Sequence patterns (n-grams)
   - Trace similarity (Jaccard, LCS)

### Evaluation Metrics

**Case Length Distribution:**
- Mean, median, standard deviation
- Histogram comparison

**Activity Distribution:**
- Frequency of each activity
- Comparison with original log

**Sequence Patterns:**
- Bigram/trigram analysis
- Pattern overlap with original log

**Trace Similarity:**
- Jaccard similarity (set-based)
- Sequence similarity (LCS-based)

### Running Evaluation

```python
# See evaluate_models.ipynb for complete example
from next_activity_prediction import LSTMNextActivityPredictor, LSTMSuffixPredictor

# Load models
next_activity_predictor = LSTMNextActivityPredictor("models/next_activity_lstm")
suffix_predictor = LSTMSuffixPredictor("models/suffix_prediction_lstm")

# Generate traces and compare
# (See notebook for full implementation)
```

---

## File Structure

```
next_activity_prediction/
├── README.md                    # This file
├── __init__.py                  # Module exports
├── config.py                    # NextActivityConfig class
├── data_preprocessing.py        # Next activity data preprocessing
├── model.py                     # Next activity model architecture
├── trainer.py                   # Next activity training
├── predictor.py                 # Next activity predictor class
├── suffix_data_preprocessing.py # Suffix data preprocessing
├── suffix_model.py              # Suffix model architecture
├── suffix_trainer.py            # Suffix training
├── suffix_predictor.py          # Suffix predictor class
├── utils.py                     # Utility functions (weights, etc.)
├── train_model.ipynb            # Next activity training notebook
├── train_suffix_model.ipynb     # Suffix training notebook
└── evaluate_models.ipynb        # Model evaluation notebook
```

### Key Files

**Next Activity Predictor:**
- `model.py`: LSTM model architecture
- `trainer.py`: Training pipeline
- `predictor.py`: Prediction class
- `data_preprocessing.py`: Data loading and preprocessing

**Suffix Predictor:**
- `suffix_model.py`: Encoder-decoder model architecture
- `suffix_trainer.py`: Training pipeline
- `suffix_predictor.py`: Prediction class with caching
- `suffix_data_preprocessing.py`: Data loading and preprocessing

**Shared:**
- `config.py`: Configuration dataclass
- `utils.py`: Utility functions (class weights, position weights)

---

## Troubleshooting

### Common Issues

**Import Errors:**
```python
# Ensure TensorFlow is installed
pip install tensorflow
```

**Model Not Found:**
```python
# Check model directory exists and contains:
# - model.keras
# - metadata.json
```

**Vocabulary Mismatch:**
- Ensure training and prediction use the same event log vocabulary
- Check that `metadata.json` contains correct vocabulary mappings

**Memory Issues:**
- Reduce `batch_size`
- Reduce `sequence_length` or `suffix_length`
- Reduce `lstm_units`

**Poor Performance:**
- Increase training epochs
- Adjust learning rate
- Try different architecture (more layers/units)
- Check data quality and preprocessing

### Getting Help

- Check model metadata: `metadata.json` in model directory
- Review training logs for warnings
- Use evaluation notebook to diagnose issues
- Verify event log format and preprocessing

---

## Advanced Topics

### Class Weighting

The next activity model supports class weighting to handle imbalanced classes (e.g., END token):

```python
config = NextActivityConfig(
    use_class_weights=True,
    class_weight_method="balanced",  # or "inverse_freq"
    end_token_weight=2.0  # Manual weight for END token
)
```

**Technical Details:**
- Applied during loss computation: `weighted_loss = class_weight[y] × loss(y, ŷ)`
- Keras applies weights per sample: `loss_with_weights = mean(weight[y] × loss(y, ŷ))`
- Effective sample size: Rare classes contribute more to gradient updates

### Position Weighting

Weight samples by position in case (later positions more likely to be END):

```python
config = NextActivityConfig(
    use_position_weights=True,
    position_weight_power=1.5  # Higher = more emphasis on later positions
)
```

**Mathematical Formulation:**
```
relative_position = i / sequence_length  # 0.0 to 1.0
position_weight = (relative_position + ε)^power
normalized_weight = position_weight / mean(position_weights)
```

**Combined Weighting:**
- When both class and position weights enabled:
  - `sample_weight[i] = class_weight[y[i]] × position_weight[i]`
  - Normalized to mean = 1.0
- Keras uses `sample_weight` OR `class_weight`, not both
- If `sample_weight` provided, `class_weight` is ignored

### Gradient Flow and Vanishing Gradients

**LSTM Design for Gradient Flow:**

LSTM cells are specifically designed to mitigate vanishing gradients:

1. **Cell State Highway**: 
   - Cell state `c_t` has direct connection: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t`
   - Gradient can flow directly through cell state without attenuation

2. **Gate Mechanisms**:
   - Forget gate controls gradient flow: `f_t` can be close to 1 (preserve gradient)
   - Input gate controls new information: `i_t` can be close to 0 (preserve old state)

3. **Gradient Clipping** (not implemented but recommended):
   ```python
   # In training loop
   gradients = tape.gradient(loss, model.trainable_variables)
   gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
   optimizer.apply_gradients(zip(gradients, model.trainable_variables))
   ```

**Monitoring Gradient Flow:**
- Use TensorBoard to visualize gradient magnitudes
- Check for gradients near zero (vanishing) or very large (exploding)
- Typical healthy range: `1e-5` to `1e-1`

### Memory Management

**Model Memory:**
- **Parameters**: ~930K (next activity) or ~1.98M (suffix) × 4 bytes = 3.7-7.9 MB
- **Activations during training**: 
  - Next activity: `batch_size × sequence_length × lstm_units × 4 bytes`
  - Suffix: `batch_size × (prefix_length + suffix_length) × lstm_units × 4 bytes`
  - Example (batch_size=64, sequence_length=50, lstm_units=256): ~3.3 MB per batch

**Prediction Memory:**
- **Case histories**: ~100 bytes per case (depends on history length)
- **Cached suffixes**: ~500 bytes per active case (suffix predictor)
- **Intermediate activations**: ~50-200 KB per prediction

**Optimization Tips:**
- Reduce `batch_size` if running out of memory
- Use `sequence_length` / `prefix_length` / `suffix_length` appropriate for your data
- Clear case histories periodically: `predictor.clear()`

### Performance Optimization

**Training Speed:**
- **GPU acceleration**: 10-50× faster than CPU
- **Batch size**: Larger batches = better GPU utilization (but more memory)
- **Mixed precision**: Use `tf.keras.mixed_precision` for 2× speedup on modern GPUs

**Prediction Speed:**
- **Batch prediction**: Process multiple cases at once
  ```python
  # Instead of:
  for case in cases:
      activity, _ = predictor.predict(case)
  
  # Use:
  batch_histories = [case.activity_history for case in cases]
  batch_predictions = model.predict(batch_sequences)
  ```
- **Model quantization**: Reduce precision (FP32 → FP16) for 2× speedup
- **TensorFlow Lite**: Convert to TFLite for mobile/edge deployment

**Caching Strategy (Suffix Predictor):**
- **Cache hit rate**: Typically 80-95% (most predictions use cached suffix)
- **Cache invalidation**: Only when case history changes or suffix exhausted
- **Memory trade-off**: Cache uses ~500 bytes per active case

### Numerical Stability

**Softmax Numerical Issues:**
- Large logits can cause overflow: `exp(1000) = inf`
- Solution: Subtract max before softmax (doesn't change result):
  ```python
  # Stable softmax
  z_max = max(z)
  z_stable = z - z_max
  softmax = exp(z_stable) / sum(exp(z_stable))
  ```
- TensorFlow/Keras handles this automatically

**Loss Function Stability:**
- `log(0) = -inf` can occur if predicted probability is exactly 0
- Solution: Add small epsilon: `loss = -log(max(ŷ[y], ε))` where `ε = 1e-7`
- TensorFlow handles this automatically

### Custom Vocabulary

The vocabulary is automatically created from the event log. To use a custom vocabulary:

**Option 1: Modify Preprocessing**
```python
# In data_preprocessing.py
def create_custom_vocabulary(activities):
    # Your custom mapping logic
    activity_to_idx = {...}
    idx_to_activity = {...}
    return activity_to_idx, idx_to_activity
```

**Option 2: Load Pre-trained Embeddings**
```python
# Initialize embedding layer with pre-trained weights
embedding_layer = layers.Embedding(vocab_size, embedding_dim)
embedding_layer.set_weights([pretrained_embeddings])
```

### Model Ensembling

Combine predictions from both models for improved accuracy:

**Simple Averaging:**
```python
def ensemble_predict(case_state, next_activity_predictor, suffix_predictor, alpha=0.5):
    next_act, is_end_na = next_activity_predictor.predict(case_state)
    suffix_act, is_end_suf = suffix_predictor.predict(case_state)
    
    # Weighted combination
    if is_end_na or is_end_suf:
        return next_act if is_end_na else suffix_act, True
    
    # Could combine probabilities if both models expose them
    return next_act if alpha > 0.5 else suffix_act, False
```

**Probability Averaging** (requires model modifications):
- Get probability distributions from both models
- Average: `P_ensemble = α × P_next_activity + (1-α) × P_suffix`
- Predict: `argmax(P_ensemble)`

### Transfer Learning

**Pre-training on Large Dataset:**
1. Train on large event log (e.g., 1M+ cases)
2. Save model weights
3. Fine-tune on smaller, domain-specific dataset

**Fine-tuning Strategy:**
```python
# Load pre-trained model
base_model = load_model("pretrained_model.keras")

# Freeze early layers
for layer in base_model.layers[:-2]:
    layer.trainable = False

# Fine-tune on new data
base_model.fit(new_X, new_y, epochs=10)
```

### Hyperparameter Search

**Grid Search Example:**
```python
from itertools import product

lstm_units_options = [128, 256, 512]
lstm_layers_options = [1, 2, 3]
dropout_options = [0.2, 0.3, 0.4]

best_score = -float('inf')
best_config = None

for units, layers, dropout in product(lstm_units_options, lstm_layers_options, dropout_options):
    config = NextActivityConfig(
        lstm_units=units,
        lstm_layers=layers,
        dropout_rate=dropout
    )
    history = train_model(config)
    val_acc = max(history['val_sparse_categorical_accuracy'])
    
    if val_acc > best_score:
        best_score = val_acc
        best_config = (units, layers, dropout)
```

**Random Search** (more efficient):
- Sample hyperparameters from distributions
- Typically finds good configurations faster than grid search

**Bayesian Optimization** (most efficient):
- Use libraries like `optuna` or `hyperopt`
- Learns which hyperparameters matter most
- Requires fewer trials than random search

---

## Mathematical Foundations

### Probability Model

Both models learn a conditional probability distribution:

**Next Activity Model:**
```
P(a_{t+1} | a₁, a₂, ..., aₜ) = f_LSTM([a₁, a₂, ..., aₜ])
```
where `f_LSTM` is the learned LSTM function mapping sequence to probability distribution.

**Suffix Model:**
```
P(a_{t+1}, a_{t+2}, ..., a_{t+k} | a₁, a₂, ..., aₜ) = f_EncoderDecoder([a₁, a₂, ..., aₜ])
```
where `f_EncoderDecoder` generates a sequence of probability distributions.

### Maximum Likelihood Estimation

**Objective Function:**

For next activity model with training set `D = {(xᵢ, yᵢ)}`:

```
L(θ) = -Σᵢ log P(yᵢ | xᵢ; θ)
```

where `θ` are model parameters and `P(yᵢ | xᵢ; θ)` is the model's predicted probability.

**Gradient Descent Update:**

```
θ_{t+1} = θ_t - α · ∇θ L(θ_t)
```

where `α` is learning rate and `∇θ L` is computed via backpropagation.

### LSTM Mathematical Formulation

**Complete LSTM Equations:**

For input `x_t` and previous states `h_{t-1}`, `c_{t-1}`:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)           (Forget gate)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)           (Input gate)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)        (Candidate values)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t               (Cell state)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)           (Output gate)
h_t = o_t ⊙ tanh(c_t)                         (Hidden state)
```

**Parameter Count:**

For LSTM with `n` input units and `m` hidden units:
- Weight matrices: `W_f, W_i, W_C, W_o` each of size `(m, n+m)`
- Bias vectors: `b_f, b_i, b_C, b_o` each of size `(m,)`
- Total: `4 × (m × (n + m) + m) = 4m(n + m + 1)` parameters

### Embedding Learning

**Embedding Matrix:**

The embedding layer learns a matrix `E ∈ R^(V × d)` where:
- `V` = vocabulary size
- `d` = embedding dimension

**Learning Objective:**

Embeddings are learned jointly with the model to minimize prediction error. The embedding for activity `i` is updated via:

```
∇E[i] = Σ gradients from all occurrences of activity i
E[i] ← E[i] - α · ∇E[i]
```

**Embedding Properties:**
- Similar activities (frequently co-occurring) have similar embeddings
- Embeddings capture activity relationships in a continuous space
- Can be visualized using dimensionality reduction (t-SNE, PCA)

### Sequence-to-Sequence Theory

**Encoder-Decoder Architecture:**

The encoder maps input sequence to fixed-size representation:
```
h_enc = Encoder(x₁, x₂, ..., x_T)
```

The decoder generates output sequence from representation:
```
y₁, y₂, ..., y_S = Decoder(h_enc)
```

**Information Bottleneck:**

The encoder must compress all information from the prefix into a fixed-size vector. This is the "information bottleneck" that forces the model to learn efficient representations.

**Attention Mechanism (Future Enhancement):**

Current implementation uses RepeatVector. Future enhancement could use attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

where:
- `Q` = decoder query
- `K, V` = encoder keys and values
- Allows decoder to focus on relevant parts of prefix

### Loss Function Properties

**Cross-Entropy Loss:**

```
L = -Σᵢ y_i · log(ŷ_i)
```

where `y` is one-hot true label and `ŷ` is predicted probabilities.

**Properties:**
- **Convex**: In the space of logits (before softmax)
- **Differentiable**: Smooth gradient everywhere
- **Information-theoretic**: Minimizes KL divergence between true and predicted distributions

**Sparse Categorical Crossentropy:**

Same as cross-entropy but uses integer labels instead of one-hot:
```
L = -log(ŷ[y])
```

More memory efficient for large vocabularies.

### Regularization

**Dropout:**

Randomly sets activations to zero during training:
```
h_dropout = dropout(h, rate=p)
```

**Effect:**
- Prevents co-adaptation of neurons
- Forces model to learn robust features
- Reduces overfitting

**Recurrent Dropout:**

Applied to recurrent connections in LSTM:
```
h_t = LSTM(x_t, dropout(h_{t-1}, rate=p))
```

**Effect:**
- Prevents LSTM from relying too heavily on previous hidden state
- Encourages learning from current input

### Evaluation Metrics

**Accuracy:**

```
Accuracy = (Number of correct predictions) / (Total predictions)
```

**Limitations:**
- Doesn't account for class imbalance
- END token predictions are rare but important

**Per-Class Accuracy:**

```
Accuracy_class[i] = (Correct predictions for class i) / (Total samples of class i)
```

**Sequence-Level Metrics (Suffix Model):**

- **Exact Match**: Percentage of suffixes predicted exactly correctly
- **BLEU Score**: Measures n-gram overlap with true suffix
- **Edit Distance**: Levenshtein distance between predicted and true suffix

### Theoretical Limitations

**Markov Assumption:**

Models assume that future depends only on recent history (up to `sequence_length`). Very long-range dependencies may be lost.

**Greedy Decoding:**

Suffix model uses greedy decoding (argmax at each step). This may not find globally optimal sequence. Beam search could improve results.

**Fixed Sequence Length:**

Models use fixed `sequence_length` / `prefix_length`. Cases longer than this are truncated, potentially losing information.

**No External Context:**

Models only use activity sequences. Additional context (case attributes, timestamps, resources) could improve predictions but are not currently used.

---

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

[Add citation information here]
