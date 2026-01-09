# Processing Time Prediction

This module provides three different methods for predicting processing times between consecutive events in process execution logs. Processing time is defined as the time elapsed from the completion of one activity to the start of the next activity in a case.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Methods](#methods)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Technical Details](#technical-details)
- [API Reference](#api-reference)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)

---

## Overview

Processing time prediction is a critical component of process simulation engines. This module implements three distinct approaches:

1. **Distribution Method**: Fits log-normal probability distributions for activity transition pairs
2. **ML Method**: Uses Random Forest regression with engineered features
3. **Probabilistic ML Method**: Employs LSTM neural networks with Gaussian output for uncertainty quantification

### Key Features

- **Multiple prediction methods** with different trade-offs
- **Context-aware predictions** using temporal, resource, and case attributes
- **Uncertainty quantification** (probabilistic ML method)
- **Robust preprocessing** with outlier removal and feature engineering
- **Efficient prediction** with optimized feature pipelines
- **Model persistence** with full serialization support

---

## Architecture

### Data Flow

```
Event Log (XES/CSV)
    ↓
Data Extraction
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Model Serialization
    ↓
Prediction API
```

### Processing Time Definition

Processing time is calculated as:
```
processing_time = timestamp(next_event) - timestamp(previous_event)
```

**Constraints:**
- Must be positive (> 0 seconds)
- Must be reasonable (< 1 year = 31,536,000 seconds)
- Measured in seconds (floating point)

---

## Methods

### 1. Distribution Method

**Approach**: Fits log-normal probability distributions for each unique activity transition pair.

**Key Characteristics:**
- **Grouping**: By `(prev_activity, prev_lifecycle, curr_activity, curr_lifecycle)`
- **Distribution**: Log-normal (fits well to positive, right-skewed data)
- **Prediction**: Samples from fitted distribution
- **Fallback**: Hierarchical fallback to activity pairs, then global statistics

**When to Use:**
- Simple, interpretable predictions
- Sufficient historical data for each transition
- No need for context features
- Fast training and prediction

**Limitations:**
- Requires minimum observations per transition (default: 2)
- Cannot leverage contextual information
- May have sparse coverage for rare transitions

### 2. ML Method (Random Forest)

**Approach**: Random Forest regression with engineered features from event context.

**Key Characteristics:**
- **Model**: Random Forest Regressor (scikit-learn)
- **Features**: 14+ engineered features (activities, resources, temporal, case attributes)
- **Preprocessing**: Label encoding, MinMax scaling, imputation
- **Prediction**: Point estimate (mean prediction)

**When to Use:**
- Need context-aware predictions
- Rich feature set available
- Want feature importance insights
- Good balance of accuracy and interpretability

**Limitations:**
- No uncertainty quantification
- Requires feature engineering
- Training time scales with data size

### 3. Probabilistic ML Method (LSTM)

**Approach**: LSTM neural network with Gaussian output for mean and variance prediction.

**Key Characteristics:**
- **Model**: LSTM + Dense layers (TensorFlow/Keras)
- **Input**: Event sequences (one-hot encoded) + context features
- **Output**: Mean and log-variance (Gaussian distribution)
- **Prediction**: Samples from predicted distribution
- **Uncertainty**: Quantifies prediction uncertainty

**When to Use:**
- Need uncertainty quantification
- Sequential patterns matter
- Large datasets available
- State-of-the-art accuracy desired

**Limitations:**
- Requires TensorFlow
- Longer training time
- More complex model
- Higher memory usage

---

## Installation

### Requirements

**Core Dependencies:**
```bash
pip install pandas numpy scipy scikit-learn joblib pm4py
```

**For Probabilistic ML Method:**
```bash
pip install tensorflow
```

**Optional (for visualization):**
```bash
pip install matplotlib seaborn
```

### Module Structure

```
processing_time_prediction/
├── __init__.py
├── ProcessingTimeTrainer.py      # Training class
├── ProcessingTimePredictionClass.py  # Prediction class
├── compare_methods.py             # Method comparison script
├── processing_time.py             # Example usage
└── train_test_lstm.py             # LSTM training example
```

---

## Quick Start

### Training a Model

```python
import pandas as pd
import pm4py
from processing_time_prediction import ProcessingTimeTrainer

# Load event log
log = pm4py.read_xes("eventlog/eventlog.xes.gz")
df = pm4py.convert_to_dataframe(log)

# Train ML model
trainer = ProcessingTimeTrainer(df, method="ml")
trainer.train()
trainer.save_model("models/processing_time_model")
```

### Making Predictions

```python
from processing_time_prediction import ProcessingTimePredictionClass

# Load trained model
predictor = ProcessingTimePredictionClass(
    method="ml",
    model_path="models/processing_time_model"
)

# Predict processing time
prediction = predictor.predict(
    prev_activity="A_Submitted",
    prev_lifecycle="complete",
    curr_activity="A_Concept",
    curr_lifecycle="start",
    context={
        'resource_1': 'User_123',
        'resource_2': 'User_456',
        'hour': 14,
        'weekday': 2,
        'month': 6,
        'day_of_year': 150,
        'event_position_in_case': 3,
        'case_duration_so_far': 3600.0,
        'case:LoanGoal': 'Car',
        'case:ApplicationType': 'New',
        'Accepted': True,
        'Selected': False
    }
)

print(f"Predicted processing time: {prediction:.2f} seconds ({prediction/3600:.2f} hours)")
```

---

## Technical Details

### Feature Engineering

#### Categorical Features

- **prev_activity**: Previous activity name
- **prev_lifecycle**: Previous lifecycle transition (start/complete)
- **curr_activity**: Current/next activity name
- **curr_lifecycle**: Current/next lifecycle transition
- **prev_resource**: Resource executing previous activity
- **curr_resource**: Resource executing current activity
- **case:LoanGoal**: Case attribute (e.g., "Car", "Home improvement")
- **case:ApplicationType**: Case attribute (e.g., "New", "Renewal")

**Encoding**: Label encoding (integer mapping)

#### Numerical Features

- **hour**: Hour of day (0-23)
- **weekday**: Day of week (0=Monday, 6=Sunday)
- **month**: Month of year (1-12)
- **day_of_year**: Day of year (1-365)
- **event_position_in_case**: Position in case sequence (1-indexed)
- **case_duration_so_far**: Time since case start (seconds)

**Scaling**: MinMax normalization (0-1 range)

#### Boolean Features

- **Accepted**: Event attribute (0/1)
- **Selected**: Event attribute (0/1)

**Encoding**: Direct integer (0/1)

### Distribution Method Details

#### Log-Normal Distribution

The log-normal distribution is used because:
1. Processing times are always positive
2. Real-world processing times are typically right-skewed
3. Log transformation normalizes the data

**Parameters:**
- **μ (mu)**: Mean of log-transformed values
- **σ (sigma)**: Standard deviation of log-transformed values

**Fitting Process:**
```python
# For each transition pair (prev_activity, prev_lifecycle, curr_activity, curr_lifecycle):
times = [t1, t2, ..., tn]  # Observed processing times
log_times = np.log(times)
mu = np.mean(log_times)
sigma = np.std(log_times)
distribution = stats.lognorm(s=sigma, scale=np.exp(mu))
```

**Prediction:**
```python
sample = distribution.rvs(size=1)[0]  # Random sample
```

**Fallback Strategy:**
1. Exact transition match: `(prev_activity, prev_lifecycle, curr_activity, curr_lifecycle)`
2. Activity pair match: `(prev_activity, *, curr_activity, *)`
3. Global fallback: Use overall mean/std with log-normal approximation

### ML Method Details

#### Random Forest Architecture

**Hyperparameters (defaults):**
- `n_estimators`: 500 trees
- `max_depth`: 30 (None for unlimited)
- `min_samples_split`: 10
- `min_samples_leaf`: 5
- `max_features`: 'sqrt' (√n_features)

**Training Process:**
1. Extract training samples from event log
2. Engineer features (14+ features)
3. Encode categorical variables
4. Scale numerical features
5. Remove outliers (mean + 3×std)
6. Train/test split (80/20)
7. Train Random Forest
8. Evaluate on validation set

**Feature Importance:**
The model provides feature importance scores indicating which features contribute most to predictions.

#### Optimized Prediction Pipeline

For fast single predictions, the ML method uses an optimized pipeline:

1. **Pre-computed feature plan**: Maps feature names to processing types
2. **Fast encoders**: Dictionary lookups instead of LabelEncoder.transform()
3. **Pre-fetched scaler parameters**: Direct scaling without sklearn overhead
4. **Vectorized operations**: NumPy arrays instead of DataFrames

**Performance**: ~0.1-1ms per prediction (vs ~10-50ms with DataFrame approach)

### Probabilistic ML Method Details

#### LSTM Architecture

**Input Layers:**
1. **Sequence Input**: `(sequence_length, feature_dim)`
   - One-hot encoded activity sequences
   - Feature dimension = num_activities + num_lifecycles + num_resources
   - Sequence length: 10 (configurable)

2. **Context Input**: `(8,)`
   - Normalized temporal and case features
   - [hour, weekday, month, day_of_year, event_position, case_duration, loan_goal, app_type]

**Model Architecture:**
```
Sequence Input (10, feature_dim)
    ↓
LSTM(128 units, dropout=0.3, recurrent_dropout=0.2)
    ↓
    └─→ Concatenate
Context Input (8,)                    ↓
    ↓                                 ↓
Dense(64, ReLU) + Dropout(0.3)       ↓
    └─→ Concatenate ────────────────→ ↓
                                      ↓
                              Dense(128, ReLU) + Dropout(0.4)
                                      ↓
                              Dense(64, ReLU) + Dropout(0.3)
                                      ↓
                              ┌───────┴───────┐
                              ↓               ↓
                        Dense(1, 'mean')  Dense(1, 'log_variance')
```

**Output:**
- **Mean**: Predicted mean processing time (log-space, normalized)
- **Log-Variance**: Log of predicted variance (for numerical stability)

**Loss Function:**
Gaussian negative log-likelihood:
```python
loss = 0.5 * log_var + 0.5 * (target - mean)² / exp(log_var)
```

**Prediction Process:**
1. Normalize target: `y_log = log(y + 1)`
2. Standardize: `y_norm = (y_log - mean) / std`
3. Predict: `[mean_norm, log_var] = model.predict([sequence, context])`
4. Denormalize: `mean = exp(mean_norm * std + mean) - 1`
5. Calculate std: `std = sqrt(exp(log_var)) * exp(mean_log)`
6. Sample: `prediction = normal(mean, std)`

**Sequence Construction:**
- Maintains event history per case
- Pads sequences to fixed length (left padding with zeros)
- Truncates to most recent `sequence_length` events

**Caching:**
- Sequences are cached to disk for faster re-training
- Cache file: `{cache_path}_sequences_cache.joblib`
- Includes: X_seq, X_ctx, y, encoders

---

## API Reference

### ProcessingTimeTrainer

**Initialization:**
```python
trainer = ProcessingTimeTrainer(
    data_log_df: pd.DataFrame,
    method: str = "distribution",  # "distribution", "ml", or "probabilistic_ml"
    min_observations: int = 2,     # For distribution method
    n_estimators: int = 500,        # For ML method
    max_depth: Optional[int] = 30,  # For ML method
    min_samples_split: int = 10,    # For ML method
    min_samples_leaf: int = 5,      # For ML method
    max_features: str = 'sqrt'      # For ML method
)
```

**Methods:**
- `train(cache_path=None, force_recompute=False)`: Train the model
- `save_model(filepath)`: Save model to disk
- `fit_distributions()`: Fit distributions (distribution method)
- `train_ml_model()`: Train Random Forest (ML method)
- `train_probabilistic_ml_model(cache_path, force_recompute)`: Train LSTM (probabilistic ML)

### ProcessingTimePredictionClass

**Initialization:**
```python
predictor = ProcessingTimePredictionClass(
    method: str = "ml",  # "distribution", "ml", or "probabilistic_ml"
    model_path: Optional[str] = None  # Default: "models/processing_time_model"
)
```

**Methods:**
- `predict(prev_activity, prev_lifecycle, curr_activity, curr_lifecycle, context=None)`: Predict processing time
- `get_distribution_info(transition_key=None)`: Get distribution information (distribution method)
- `get_probabilistic_distribution(...)`: Get mean/std prediction (probabilistic ML method)
- `load_model(filepath)`: Load model from disk

**Prediction Parameters:**
- `prev_activity` (str): Previous activity name
- `prev_lifecycle` (str): Previous lifecycle transition
- `curr_activity` (str): Current/next activity name
- `curr_lifecycle` (str): Current/next lifecycle transition
- `context` (dict, optional): Context dictionary with:
  - `resource_1`, `resource_2`: Resource names
  - `hour`, `weekday`, `month`, `day_of_year`: Temporal features
  - `event_position_in_case`: Position in case
  - `case_duration_so_far`: Time since case start (seconds)
  - `case:LoanGoal`, `case:ApplicationType`: Case attributes
  - `Accepted`, `Selected`: Event attributes

**Returns:**
- `float`: Predicted processing time in seconds

---

## Training

### Distribution Method Training

```python
trainer = ProcessingTimeTrainer(df, method="distribution", min_observations=2)
trainer.train()
trainer.save_model("models/processing_time_model_dist")
```

**Output Files:**
- `{filepath}_metadata.joblib`: Metadata (method, fallback stats)
- `{filepath}_distributions.joblib`: Fitted distributions

### ML Method Training

```python
trainer = ProcessingTimeTrainer(
    df,
    method="ml",
    n_estimators=500,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=5
)
trainer.train()
trainer.save_model("models/processing_time_model_ml")
```

**Output Files:**
- `{filepath}_metadata.joblib`: Metadata (method, features, defaults)
- `{filepath}_model.joblib`: Random Forest model
- `{filepath}_encoders.joblib`: Label encoders
- `{filepath}_scaler.joblib`: MinMax scaler

### Probabilistic ML Method Training

```python
trainer = ProcessingTimeTrainer(df, method="probabilistic_ml")
trainer.train(
    cache_path="models/processing_time_model_lstm",
    force_recompute=False
)
trainer.save_model("models/processing_time_model_lstm")
```

**Output Files:**
- `{filepath}_metadata.joblib`: Metadata (method, sequence_length, normalization)
- `{filepath}_lstm_model.keras` or `.h5`: LSTM model
- `{filepath}_encoders.joblib`: Activity/lifecycle/resource encoders
- `{filepath}_sequences_cache.joblib`: Cached sequences (optional)

**Training Time:**
- Distribution: < 1 minute
- ML: 5-30 minutes (depends on data size)
- Probabilistic ML: 30 minutes - 2 hours (depends on data size and GPU)

---

## Evaluation Metrics

### Metrics Computed

1. **MAE (Mean Absolute Error)**: Average absolute difference
   ```
   MAE = (1/n) * Σ|y_true - y_pred|
   ```

2. **RMSE (Root Mean Squared Error)**: Penalizes large errors
   ```
   RMSE = √[(1/n) * Σ(y_true - y_pred)²]
   ```

3. **R² (Coefficient of Determination)**: Proportion of variance explained
   ```
   R² = 1 - (SS_res / SS_tot)
   ```

4. **MAPE (Mean Absolute Percentage Error)**: Relative error
   ```
   MAPE = (100/n) * Σ|y_true - y_pred| / (y_true + ε)
   ```

### Typical Performance

**Distribution Method:**
- MAE: ~2-5 hours
- RMSE: ~5-10 hours
- R²: 0.3-0.5

**ML Method:**
- MAE: ~1-3 hours
- RMSE: ~3-6 hours
- R²: 0.5-0.7

**Probabilistic ML Method:**
- MAE: ~1-2 hours
- RMSE: ~2-4 hours
- R²: 0.6-0.8

*Note: Performance varies significantly with dataset characteristics*

---

## File Structure

### Saved Model Files

**Distribution Method:**
```
models/processing_time_model_dist/
├── processing_time_model_dist_metadata.joblib
└── processing_time_model_dist_distributions.joblib
```

**ML Method:**
```
models/processing_time_model_ml/
├── processing_time_model_ml_metadata.joblib
├── processing_time_model_ml_model.joblib
├── processing_time_model_ml_encoders.joblib
└── processing_time_model_ml_scaler.joblib
```

**Probabilistic ML Method:**
```
models/processing_time_model_lstm/
├── processing_time_model_lstm_metadata.joblib
├── processing_time_model_lstm_lstm_model.keras
├── processing_time_model_lstm_encoders.joblib
└── processing_time_model_lstm_sequences_cache.joblib (optional)
```

---

## Configuration

### Training Configuration

**Distribution Method:**
- `min_observations`: Minimum samples per transition (default: 2)

**ML Method:**
- `n_estimators`: Number of trees (default: 500)
- `max_depth`: Maximum tree depth (default: 30, None for unlimited)
- `min_samples_split`: Minimum samples to split (default: 10)
- `min_samples_leaf`: Minimum samples at leaf (default: 5)
- `max_features`: Features per split (default: 'sqrt')

**Probabilistic ML Method:**
- `sequence_length`: Event history length (default: 10)
- `cache_path`: Path for sequence caching
- `force_recompute`: Re-extract sequences even if cached

### Preprocessing Configuration

**Outlier Removal:**
- Threshold: `mean + 3 × std`
- Applied to all methods

**Feature Defaults:**
- Numerical: Median value
- Categorical: Mode value
- Boolean: 0 (False)

---

## Advanced Usage

### Comparing Methods

Use the `compare_methods.py` script to evaluate all three methods:

```python
python processing_time_prediction/compare_methods.py
```

This script:
1. Loads event log
2. Creates test set
3. Trains all three methods
4. Evaluates on test set
5. Prints comparison table

### Custom Feature Engineering

You can extend feature engineering by modifying `_extract_training_data()` and `_context_to_features()` methods.

### Uncertainty Quantification

The probabilistic ML method provides uncertainty estimates:

```python
# Get distribution parameters
dist_info = predictor.get_probabilistic_distribution(
    prev_activity="A_Submitted",
    prev_lifecycle="complete",
    curr_activity="A_Concept",
    curr_lifecycle="start",
    context={...}
)

print(f"Mean: {dist_info['mean']:.2f}s")
print(f"Std: {dist_info['std']:.2f}s")
print(f"95% CI: [{dist_info['mean'] - 1.96*dist_info['std']:.2f}, {dist_info['mean'] + 1.96*dist_info['std']:.2f}]")
```

### Batch Prediction

For multiple predictions, prepare features in batch:

```python
predictions = []
for sample in test_samples:
    pred = predictor.predict(
        prev_activity=sample['prev_activity'],
        prev_lifecycle=sample['prev_lifecycle'],
        curr_activity=sample['curr_activity'],
        curr_lifecycle=sample['curr_lifecycle'],
        context=sample['context']
    )
    predictions.append(pred)
```

### Model Inspection

**Distribution Method:**
```python
# Get all distributions
info = predictor.get_distribution_info()
print(f"Number of distributions: {info['num_distributions']}")

# Get specific transition
transition_key = ("A_Submitted", "complete", "A_Concept", "start")
info = predictor.get_distribution_info(transition_key)
print(f"Mean: {info['mean']:.2f}s, Std: {info['std']:.2f}s")
```

**ML Method:**
```python
# Feature importance
importances = predictor.ml_model.feature_importances_
feature_names = predictor.feature_names
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")
```

---

## Troubleshooting

### Common Issues

1. **"No valid training samples found"**
   - Check event log has required columns
   - Verify timestamps are valid
   - Ensure cases have at least 2 events

2. **"Model file not found"**
   - Verify model path is correct
   - Check all required files exist
   - Ensure file extensions match

3. **"TensorFlow is required"**
   - Install TensorFlow: `pip install tensorflow`
   - Only needed for probabilistic_ml method

4. **Memory errors (probabilistic ML)**
   - Reduce sequence_length
   - Use caching to avoid re-extraction
   - Process in smaller batches

5. **Poor prediction accuracy**
   - Check feature quality
   - Increase training data
   - Tune hyperparameters
   - Try different method

### Performance Optimization

**For ML Method:**
- Use optimized prediction pipeline (automatic)
- Batch predictions when possible
- Cache feature preparation

**For Probabilistic ML Method:**
- Use GPU for training
- Enable sequence caching
- Reduce sequence_length if memory constrained

---

## Integration with Simulation Engine

The module integrates seamlessly with the simulation engine:

```python
from processing_time_prediction import ProcessingTimePredictionClass
from integration.setup import SimulationConfig

# Configure
config = SimulationConfig(
    processing_time_mode="advanced",
    processing_time_method="ml",  # or "distribution" or "probabilistic_ml"
    processing_time_model_path="models/processing_time_model"
)

# Engine automatically loads and uses the predictor
```

---

## References

### Log-Normal Distribution
- Fits positive, right-skewed data
- Parameters: μ (location), σ (scale)
- Common in reliability and processing time modeling

### Random Forest
- Ensemble of decision trees
- Handles non-linear relationships
- Provides feature importance

### LSTM Networks
- Captures sequential dependencies
- Handles variable-length sequences
- Gaussian output for uncertainty

---

## License

Same as parent project.

