# Basic Next Activity Prediction

A probabilistic branch prediction system for business process simulation that learns decision point probabilities from historical event logs and BPMN process models.

## Overview

This module implements a lightweight next-activity prediction approach that combines BPMN process structure with historical event log data to predict branch selections at XOR gateways. The system learns transition probabilities based on the preceding activity context and uses weighted random sampling for prediction.

## Architecture

### Core Components

#### 1. **BPMNParser** (`bpmn_parser.py`)
Parses BPMN 2.0 XML files to extract process structure:
- **Tasks**: Activity nodes and their names
- **Gateways**: Exclusive gateways (XOR) with direction and flow connections
- **Sequence Flows**: Connections between process elements

**Key Methods:**
- `get_xor_decision_points()`: Identifies diverging XOR gateways (decision points)
- `get_gateway_connections()`: Maps each gateway to:
  - `preceding`: Activities that can lead to this gateway
  - `branches`: Possible next activities after the gateway

**Technical Details:**
- Uses XML namespace detection for BPMN compatibility
- Recursively traces flows to find all preceding/succeeding activities
- Handles nested gateway structures through depth-first traversal

#### 2. **LogAnalyzer** (`log_analyzer.py`)
Analyzes XES event logs to extract transition patterns:
- Reads event logs using `pm4py`
- Counts transitions from preceding activities through gateways to branch activities
- Calculates empirical probabilities from transition frequencies

**Key Methods:**
- `count_transitions()`: Counts how often each (gateway, preceding_activity) pair leads to each branch
- `calculate_probabilities()`: Converts counts to probability distributions

**Technical Details:**
- Uses `defaultdict` for efficient counting
- Handles both DataFrame and EventLog formats from pm4py
- Probability calculation: `P(branch | gateway, preceding) = count(branch) / total_count`

#### 3. **BranchPredictor** (`predictor.py`)
Main prediction class that combines BPMN structure with learned probabilities:

**Training (`fit` method):**
1. Parses BPMN to identify decision points
2. Analyzes event log to count transitions
3. Calculates probability distributions for each (gateway, preceding_activity) pair

**Prediction (`predict` method):**
- Input: `(gateway_id, preceding_activity)`
- Output: Next activity (sampled from probability distribution)
- Fallback: Random selection from available branches if no training data exists

**Technical Details:**
- Uses `random.choices()` with probability weights for sampling
- Stores probabilities as nested dictionaries: `{(gateway_id, preceding_activity): {branch: probability}}`
- Model persistence via `joblib` serialization

## Data Flow

```
BPMN File → BPMNParser → Gateway Connections
                                    ↓
Event Log → LogAnalyzer → Transition Counts → Probabilities
                                    ↓
                            BranchPredictor.fit()
                                    ↓
                            Trained Model (joblib)
                                    ↓
                            BranchPredictor.predict()
```

## Usage

### Training

Train a model from BPMN and event log:

```python
from Next_Activity_Prediction.basic_prediction.predictor import BranchPredictor

predictor = BranchPredictor()
predictor.fit(
    bpmn_path="process_model/LoanApplicationProcess.bpmn",
    log_path="Dataset/BPI Challenge 2017.xes"
)
predictor.save("models/branch_predictor.joblib")
```

Or use the training script:

```bash
python train.py
```

### Prediction

Load and use a trained model:

```python
from Next_Activity_Prediction.basic_prediction.predictor import BranchPredictor

predictor = BranchPredictor.load("models/branch_predictor.joblib")

# Predict next activity at a decision point
next_activity = predictor.predict(
    gateway_id="Gateway_abc123",
    preceding_activity="A_Create Application"
)

# Get probability distribution
probs = predictor.get_probabilities(
    gateway_id="Gateway_abc123",
    preceding_activity="A_Create Application"
)
# Returns: {"Activity_A": 0.6, "Activity_B": 0.4}
```

### Testing

Test model accuracy against actual traces:

```bash
python test_predictions.py
```

This script:
- Loads the trained model
- Iterates through event log traces
- Compares predictions to actual next activities
- Reports accuracy metrics

### Simulation Example

See `example_simulation.py` for integration into process simulation:

```python
from Next_Activity_Prediction.basic_prediction.example_simulation import BranchPredictor

predictor = BranchPredictor.load("models/branch_predictor.joblib")

# During simulation, at each decision point:
next_activity = predictor.predict(gateway_id, current_activity)
```

## File Structure

```
basic_prediction/
├── bpmn_parser.py          # BPMN XML parsing and gateway extraction
├── log_analyzer.py         # Event log analysis and probability calculation
├── predictor.py            # Main BranchPredictor class
├── train.py                # Training script
├── test_predictions.py     # Model evaluation script
├── example_simulation.py   # Simulation integration example
├── DataPreProcessing.py    # Utility for prefix generation
└── README.md              # This file
```

## Technical Specifications

### Model Format

The saved model (joblib) contains:
```python
{
    'probabilities': {
        (gateway_id, preceding_activity): {
            branch_activity: probability,
            ...
        },
        ...
    },
    'gateway_branches': {
        gateway_id: [branch1, branch2, ...],
        ...
    },
    'gateway_connections': {
        gateway_id: {
            'preceding': [activity1, activity2, ...],
            'branches': [branch1, branch2, ...]
        },
        ...
    }
}
```

### Probability Calculation

For each decision point `(gateway_id, preceding_activity)`:
1. Count occurrences: `count(branch_i)` = number of times branch_i was taken
2. Total count: `total = sum(count(branch_i))`
3. Probability: `P(branch_i) = count(branch_i) / total`

### Prediction Algorithm

1. Lookup probability distribution for `(gateway_id, preceding_activity)`
2. If found: sample from distribution using `random.choices(branches, weights=probabilities)`
3. If not found: fallback to uniform random selection from available branches
4. If no branches available: return `None`

## Dependencies

- `pm4py`: Event log reading and processing
- `joblib`: Model serialization
- `xml.etree.ElementTree`: BPMN parsing (standard library)
- `collections.defaultdict`: Efficient counting (standard library)
- `random`: Probability sampling (standard library)

## Limitations

1. **Context**: Only considers immediate preceding activity, not full trace history
2. **Gateway Types**: Only supports XOR (exclusive) gateways
3. **Cold Start**: No probabilities for unseen (gateway, activity) combinations
4. **Determinism**: Uses random sampling, not deterministic selection

## Extensions

For more advanced prediction capabilities, see:
- `Next-Activity-Prediction/advanced/`: LSTM-based sequence models
- `Next-Activity-Prediction/bpic17_simplified/`: Dataset-specific implementations

## Performance Considerations

- **Training**: O(n × m) where n = traces, m = average trace length
- **Prediction**: O(1) lookup + O(k) sampling where k = number of branches
- **Memory**: O(g × a × b) where g = gateways, a = preceding activities, b = branches

Typical model sizes: 10-100 KB for medium-sized processes.

