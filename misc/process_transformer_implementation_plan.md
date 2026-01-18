# ProcessTransformer Integration: Implementation Plan

**Document Version:** 1.0
**Created:** 2024-01-09
**Status:** Planning
**Repository:** https://github.com/Zaharah/processtransformer

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Data Preparation](#3-phase-1-data-preparation)
4. [Phase 2: Model Training (Google Colab)](#4-phase-2-model-training-google-colab)
5. [Phase 3: Model Publishing (HuggingFace Hub)](#5-phase-3-model-publishing-huggingface-hub)
6. [Phase 4: Local Integration](#6-phase-4-local-integration)
7. [Phase 5: Testing & Validation](#7-phase-5-testing--validation)
8. [Pitfalls & Mitigations](#8-pitfalls--mitigations)
9. [Maintenance & Versioning](#9-maintenance--versioning)
10. [Appendix: Code Templates](#10-appendix-code-templates)

---

## 1. Executive Summary

### Goal
Integrate ProcessTransformer as a next-activity predictor in the BPIC17 simulation engine, enabling probabilistic activity prediction using self-attention mechanisms.

### Key Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training environment | Google Colab (Pro recommended) | Free GPU, reproducible notebooks |
| Model hosting | HuggingFace Hub | Version control, easy download, community standard |
| Inference mode | Probabilistic sampling | Matches simulation stochasticity requirements |
| Activity encoding | Simplified (no lifecycle) | Matches existing predictor patterns |

### Deliverables
1. Training notebook (`notebooks/train_process_transformer.ipynb`)
2. HuggingFace model repository
3. `ProcessTransformerPredictor` class in engine
4. Integration tests
5. Benchmark comparison with existing predictors

---

## 2. Architecture Overview

### 2.1 System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION ENGINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│   │  DESEngine   │───▶│     NextActivityPredictor (Protocol)    │   │
│   └──────────────┘    └─────────────────────────────────────────┘   │
│                                        │                             │
│                        ┌───────────────┼───────────────┐            │
│                        ▼               ▼               ▼            │
│               ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│               │   LSTM      │  │  Unified    │  │  Process    │     │
│               │  Predictor  │  │  Predictor  │  │ Transformer │     │
│               └─────────────┘  └─────────────┘  └──────┬──────┘     │
│                                                         │            │
└─────────────────────────────────────────────────────────│────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │   HuggingFace Hub    │
                                              │  (Model Repository)  │
                                              └──────────────────────┘
```

### 2.2 Model Artifacts Structure

```
huggingface.co/your-org/bpic17-process-transformer/
├── model.keras                    # Trained model weights
├── vocab.json                     # Activity → Integer mapping
├── config.json                    # Model configuration
├── training_metadata.json         # Training run details
├── README.md                      # Model card
└── requirements.txt               # Inference dependencies
```

### 2.3 Local Cache Structure

```
~/.cache/bpic17-simulation/
└── process_transformer/
    └── v1.0.0/
        ├── model.keras
        ├── vocab.json
        └── config.json
```

---

## 3. Phase 1: Data Preparation

### 3.1 Objective
Transform BPIC17 event log into ProcessTransformer-compatible format.

### 3.2 Schema Mapping

**Source (BPIC17):**
```
case:concept:name | concept:name          | lifecycle:transition | time:timestamp
Application_123   | A_Create Application  | complete             | 2016-01-04 08:15:00
Application_123   | A_Submitted           | complete             | 2016-01-04 08:16:30
```

**Target (ProcessTransformer):**
```
Case ID           | Activity              | Complete Timestamp
Application_123   | A_Create Application  | 2016-01-04 08:15:00
Application_123   | A_Submitted           | 2016-01-04 08:16:30
```

### 3.3 Preprocessing Script

Create file: `Next-Activity-Prediction/process_transformer/prepare_data.py`

```python
"""
Prepare BPIC17 data for ProcessTransformer training.

Usage:
    python prepare_data.py --input eventlog/eventlog.xes.gz --output data/bpic17_pt.csv
"""

import pandas as pd
import pm4py
from pathlib import Path


def prepare_bpic17_for_process_transformer(
    input_path: str,
    output_path: str,
    lifecycle_filter: str = "complete",
    min_case_length: int = 2,
    max_case_length: int = 100,
) -> pd.DataFrame:
    """
    Transform BPIC17 to ProcessTransformer format.

    Args:
        input_path: Path to XES/XES.GZ file
        output_path: Path for output CSV
        lifecycle_filter: Only include events with this lifecycle (None = all)
        min_case_length: Exclude cases shorter than this
        max_case_length: Truncate cases longer than this

    Returns:
        Prepared DataFrame
    """
    # Load event log
    log = pm4py.read_xes(input_path)
    df = pm4py.convert_to_dataframe(log)

    print(f"Loaded: {len(df)} events, {df['case:concept:name'].nunique()} cases")

    # Filter by lifecycle if specified
    if lifecycle_filter:
        df = df[df['lifecycle:transition'] == lifecycle_filter]
        print(f"After lifecycle filter: {len(df)} events")

    # Sort by case and timestamp
    df = df.sort_values(['case:concept:name', 'time:timestamp'])

    # Calculate case lengths
    case_lengths = df.groupby('case:concept:name').size()

    # Filter by case length
    valid_cases = case_lengths[
        (case_lengths >= min_case_length) &
        (case_lengths <= max_case_length)
    ].index
    df = df[df['case:concept:name'].isin(valid_cases)]
    print(f"After length filter: {len(df)} events, {len(valid_cases)} cases")

    # Truncate long cases (keep first max_case_length events)
    df = df.groupby('case:concept:name').head(max_case_length)

    # Select and rename columns
    result = df[['case:concept:name', 'concept:name', 'time:timestamp']].copy()
    result.columns = ['Case ID', 'Activity', 'Complete Timestamp']

    # Ensure timestamp format
    result['Complete Timestamp'] = pd.to_datetime(result['Complete Timestamp'])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"Final: {len(result)} events, {result['Case ID'].nunique()} cases")
    print(f"Activities: {result['Activity'].nunique()} unique")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eventlog/eventlog.xes.gz")
    parser.add_argument("--output", default="data/bpic17_pt.csv")
    parser.add_argument("--lifecycle", default="complete")
    args = parser.parse_args()

    prepare_bpic17_for_process_transformer(
        args.input,
        args.output,
        lifecycle_filter=args.lifecycle,
    )
```

### 3.4 Data Validation Checklist

- [ ] No missing values in required columns
- [ ] Timestamps are properly parsed
- [ ] Case IDs are consistent (no whitespace issues)
- [ ] Activity names match expected vocabulary
- [ ] Case lengths are within expected range
- [ ] Train/test split maintains case integrity (no case in both splits)

### 3.5 Known Data Issues

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Lifecycle variants (start/complete/schedule) | Doubled activities if not filtered | Filter to `complete` only |
| Very long cases (>200 events) | Memory issues, rare patterns | Truncate or exclude |
| Duplicate timestamps | Ordering ambiguity | Add secondary sort by activity |
| Missing activities in some cases | Incomplete traces | Keep as-is (realistic) |

---

## 4. Phase 2: Model Training (Google Colab)

### 4.1 Colab Notebook Structure

Create: `notebooks/train_process_transformer.ipynb`

The notebook should have these sections:

```
1. Setup & Dependencies
2. Data Loading
3. Data Preprocessing
4. Model Configuration
5. Training
6. Evaluation
7. Export to HuggingFace
```

### 4.2 Complete Colab Notebook

```python
# =============================================================================
# Cell 1: Setup & Dependencies
# =============================================================================
# Run this first to install dependencies

!pip install tensorflow==2.15.0
!pip install pm4py
!pip install huggingface_hub
!pip install scikit-learn

# Clone ProcessTransformer repository
!git clone https://github.com/Zaharah/processtransformer.git
%cd processtransformer

# =============================================================================
# Cell 2: Mount Google Drive (for data persistence)
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

# Create working directory
import os
WORK_DIR = '/content/drive/MyDrive/BPIC17_ProcessTransformer'
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(f'{WORK_DIR}/data', exist_ok=True)
os.makedirs(f'{WORK_DIR}/models', exist_ok=True)

# =============================================================================
# Cell 3: Upload BPIC17 Data
# =============================================================================
# Option A: Upload from local machine
from google.colab import files
print("Upload your prepared bpic17_pt.csv file:")
uploaded = files.upload()

# Move to work directory
import shutil
for filename in uploaded.keys():
    shutil.move(filename, f'{WORK_DIR}/data/{filename}')
    print(f"Moved {filename} to {WORK_DIR}/data/")

# Option B: If already in Drive, skip upload and set path
DATA_PATH = f'{WORK_DIR}/data/bpic17_pt.csv'

# =============================================================================
# Cell 4: Data Preprocessing
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {len(df)} events, {df['Case ID'].nunique()} cases")

# Build vocabulary
activities = df['Activity'].unique().tolist()
activities = ['<PAD>', '<START>', '<END>'] + sorted(activities)  # Special tokens

activity_to_idx = {act: idx for idx, act in enumerate(activities)}
idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}

vocab_size = len(activities)
print(f"Vocabulary size: {vocab_size}")

# Save vocabulary
vocab_path = f'{WORK_DIR}/models/vocab.json'
with open(vocab_path, 'w') as f:
    json.dump({
        'activity_to_idx': activity_to_idx,
        'idx_to_activity': {str(k): v for k, v in idx_to_activity.items()},
        'vocab_size': vocab_size,
    }, f, indent=2)
print(f"Saved vocabulary to: {vocab_path}")

# Convert activities to indices
df['activity_idx'] = df['Activity'].map(activity_to_idx)

# Group by case and create sequences
cases = df.groupby('Case ID')['activity_idx'].apply(list).reset_index()
cases.columns = ['case_id', 'sequence']

# Calculate max case length
max_case_length = cases['sequence'].apply(len).max()
print(f"Max case length: {max_case_length}")

# Use reasonable max length (cap at 100 for memory)
MAX_SEQ_LEN = min(max_case_length + 2, 100)  # +2 for START/END tokens
print(f"Using max sequence length: {MAX_SEQ_LEN}")

# Create training examples (prefix → next activity)
def create_training_examples(sequences, max_len):
    """Generate prefix-target pairs for each case."""
    X, y = [], []
    start_idx = activity_to_idx['<START>']
    end_idx = activity_to_idx['<END>']
    pad_idx = activity_to_idx['<PAD>']

    for seq in sequences:
        # Add START token at beginning
        full_seq = [start_idx] + seq + [end_idx]

        # Generate prefixes of increasing length
        for i in range(1, len(full_seq)):
            prefix = full_seq[:i]
            target = full_seq[i]

            # Pad prefix to max length
            if len(prefix) < max_len:
                prefix = [pad_idx] * (max_len - len(prefix)) + prefix
            else:
                prefix = prefix[-max_len:]  # Take last max_len tokens

            X.append(prefix)
            y.append(target)

    return np.array(X), np.array(y)

X, y = create_training_examples(cases['sequence'].tolist(), MAX_SEQ_LEN)
print(f"Training examples: {len(X)}")

# Train/validation split (by case, not by example)
train_cases, val_cases = train_test_split(
    cases['sequence'].tolist(),
    test_size=0.2,
    random_state=42
)

X_train, y_train = create_training_examples(train_cases, MAX_SEQ_LEN)
X_val, y_val = create_training_examples(val_cases, MAX_SEQ_LEN)

print(f"Training: {len(X_train)} examples")
print(f"Validation: {len(X_val)} examples")

# =============================================================================
# Cell 5: Model Definition
# =============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_process_transformer(
    max_seq_len: int,
    vocab_size: int,
    embed_dim: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    dropout_rate: float = 0.1,
):
    """
    Create ProcessTransformer model for next activity prediction.

    Based on: https://github.com/Zaharah/processtransformer
    """
    # Input layer
    inputs = layers.Input(shape=(max_seq_len,), dtype=tf.int32)

    # Embedding layer
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
    )(inputs)

    # Positional encoding (learned)
    positions = tf.range(start=0, limit=max_seq_len, delta=1)
    position_embedding = layers.Embedding(
        input_dim=max_seq_len,
        output_dim=embed_dim,
    )(positions)

    x = embedding + position_embedding
    x = layers.Dropout(dropout_rate)(x)

    # Transformer blocks
    for _ in range(num_blocks):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate,
        )(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])
        ffn_output = ffn(x)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global average pooling (or take last non-padded position)
    x = layers.GlobalAveragePooling1D()(x)

    # Output layer
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(vocab_size, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create model
model = create_process_transformer(
    max_seq_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout_rate=0.1,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# =============================================================================
# Cell 6: Training
# =============================================================================
# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
    ),
    keras.callbacks.ModelCheckpoint(
        f'{WORK_DIR}/models/checkpoint.keras',
        monitor='val_loss',
        save_best_only=True,
    ),
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)

# Plot training history
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Training Loss')

axes[1].plot(history.history['accuracy'], label='Train')
axes[1].plot(history.history['val_accuracy'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].set_title('Training Accuracy')

plt.tight_layout()
plt.savefig(f'{WORK_DIR}/models/training_history.png')
plt.show()

# =============================================================================
# Cell 7: Evaluation
# =============================================================================
# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Test probabilistic sampling
def predict_next_activity(model, prefix_indices, vocab, temperature=1.0):
    """Predict next activity with probabilistic sampling."""
    # Pad prefix
    pad_idx = vocab['activity_to_idx']['<PAD>']
    max_len = model.input_shape[1]

    if len(prefix_indices) < max_len:
        padded = [pad_idx] * (max_len - len(prefix_indices)) + prefix_indices
    else:
        padded = prefix_indices[-max_len:]

    # Predict
    probs = model.predict(np.array([padded]), verbose=0)[0]

    # Apply temperature
    if temperature != 1.0:
        probs = np.power(probs, 1/temperature)
        probs = probs / probs.sum()

    # Sample
    next_idx = np.random.choice(len(probs), p=probs)
    next_activity = vocab['idx_to_activity'][str(next_idx)]

    return next_activity, probs[next_idx]

# Test on a few examples
print("\nSample predictions:")
test_prefixes = [
    ['A_Create Application'],
    ['A_Create Application', 'A_Submitted'],
    ['A_Create Application', 'A_Submitted', 'W_Handle leads'],
]

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

for prefix in test_prefixes:
    prefix_indices = [vocab['activity_to_idx'][act] for act in prefix]
    next_act, prob = predict_next_activity(model, prefix_indices, vocab)
    print(f"  {prefix[-1]} → {next_act} (p={prob:.3f})")

# =============================================================================
# Cell 8: Save Model Locally
# =============================================================================
# Save final model
model_path = f'{WORK_DIR}/models/model.keras'
model.save(model_path)
print(f"Saved model to: {model_path}")

# Save configuration
config = {
    'max_seq_len': MAX_SEQ_LEN,
    'vocab_size': vocab_size,
    'embed_dim': 64,
    'num_heads': 4,
    'ff_dim': 128,
    'num_blocks': 2,
    'dropout_rate': 0.1,
    'training_examples': len(X_train),
    'validation_accuracy': float(val_acc),
    'validation_loss': float(val_loss),
}

config_path = f'{WORK_DIR}/models/config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved config to: {config_path}")

# Save training metadata
from datetime import datetime

metadata = {
    'trained_at': datetime.now().isoformat(),
    'dataset': 'BPIC17',
    'tensorflow_version': tf.__version__,
    'final_val_accuracy': float(val_acc),
    'final_val_loss': float(val_loss),
    'epochs_trained': len(history.history['loss']),
    'training_cases': len(train_cases),
    'validation_cases': len(val_cases),
}

metadata_path = f'{WORK_DIR}/models/training_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to: {metadata_path}")

# =============================================================================
# Cell 9: Upload to HuggingFace Hub
# =============================================================================
from huggingface_hub import HfApi, login, create_repo

# Login to HuggingFace (you'll need to enter your token)
login()

# Repository settings
REPO_NAME = "bpic17-process-transformer"
REPO_ID = f"your-username/{REPO_NAME}"  # Change to your username!

# Create repository (if it doesn't exist)
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"Repository may already exist: {e}")

# Upload files
api = HfApi()

files_to_upload = [
    (model_path, "model.keras"),
    (vocab_path, "vocab.json"),
    (config_path, "config.json"),
    (metadata_path, "training_metadata.json"),
]

for local_path, repo_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded: {repo_path}")

# Create model card (README.md)
model_card = f"""---
license: apache-2.0
tags:
- process-mining
- next-activity-prediction
- transformer
- bpic17
datasets:
- BPIC17
metrics:
- accuracy
---

# BPIC17 Process Transformer

A Transformer-based next activity predictor trained on the BPIC17 event log.

## Model Description

This model predicts the next activity in a business process trace using self-attention mechanisms.
It outputs a probability distribution over all possible activities, enabling stochastic simulation.

## Training Data

- **Dataset:** BPIC17 (Business Process Intelligence Challenge 2017)
- **Events:** Loan application process
- **Preprocessing:** Complete lifecycle events only

## Usage

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf
import json
import numpy as np

# Download model files
model_path = hf_hub_download("{REPO_ID}", "model.keras")
vocab_path = hf_hub_download("{REPO_ID}", "vocab.json")
config_path = hf_hub_download("{REPO_ID}", "config.json")

# Load model and vocabulary
model = tf.keras.models.load_model(model_path)
with open(vocab_path) as f:
    vocab = json.load(f)

# Predict
prefix = ["A_Create Application", "A_Submitted"]
prefix_indices = [vocab['activity_to_idx'][a] for a in prefix]
# ... (see full example in simulation engine)
```

## Performance

- **Validation Accuracy:** {val_acc:.4f}
- **Validation Loss:** {val_loss:.4f}

## Limitations

- Trained on BPIC17 only; may not generalize to other processes
- Does not consider case attributes (LoanGoal, Amount, etc.)
- Requires TensorFlow >= 2.15.0

## Citation

If you use this model, please cite the original ProcessTransformer paper and BPIC17 dataset.
"""

# Save and upload model card
readme_path = f'{WORK_DIR}/models/README.md'
with open(readme_path, 'w') as f:
    f.write(model_card)

api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"Uploaded model card")

print(f"\n{'='*60}")
print(f"Model published to: https://huggingface.co/{REPO_ID}")
print(f"{'='*60}")
```

### 4.3 Colab Tips & Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not available | Runtime → Change runtime type → GPU |
| Session timeout | Enable "Prevent going to sleep" extension |
| Out of memory | Reduce batch_size or max_seq_len |
| Drive mount fails | Re-authenticate or restart runtime |
| HuggingFace login fails | Generate new token at huggingface.co/settings/tokens |

### 4.4 Training Hyperparameters

**Recommended starting configuration:**

```python
HYPERPARAMETERS = {
    # Model architecture
    'embed_dim': 64,        # Embedding dimension
    'num_heads': 4,         # Attention heads
    'ff_dim': 128,          # Feed-forward dimension
    'num_blocks': 2,        # Transformer blocks
    'dropout_rate': 0.1,    # Dropout rate

    # Training
    'batch_size': 64,       # Batch size (reduce if OOM)
    'learning_rate': 1e-4,  # Adam learning rate
    'epochs': 50,           # Max epochs (early stopping)
    'patience': 5,          # Early stopping patience

    # Data
    'max_seq_len': 100,     # Max sequence length
    'test_split': 0.2,      # Validation split ratio
}
```

**For larger datasets (>100k cases):**
- Increase `embed_dim` to 128
- Increase `num_blocks` to 3-4
- Use `batch_size` of 128-256

---

## 5. Phase 3: Model Publishing (HuggingFace Hub)

### 5.1 Account Setup

1. Create account at [huggingface.co](https://huggingface.co)
2. Generate access token:
   - Go to Settings → Access Tokens
   - Create token with "Write" permission
   - Save token securely (you'll need it in Colab)

### 5.2 Repository Naming Convention

```
your-org/bpic17-process-transformer-v{major}.{minor}.{patch}

Examples:
- bpso25/bpic17-process-transformer-v1.0.0  (initial release)
- bpso25/bpic17-process-transformer-v1.1.0  (new features)
- bpso25/bpic17-process-transformer-v1.0.1  (bug fixes)
```

### 5.3 Model Card Requirements

Every model MUST have a README.md with:

- [ ] License information
- [ ] Model description
- [ ] Training data description
- [ ] Usage example
- [ ] Performance metrics
- [ ] Known limitations
- [ ] Citation information

### 5.4 Versioning Strategy

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Breaking API change | Major (X.0.0) | New vocabulary format |
| New feature | Minor (0.X.0) | Add temperature parameter |
| Bug fix | Patch (0.0.X) | Fix padding issue |
| Retrain (same config) | None | Use Git tags |

---

## 6. Phase 4: Local Integration

### 6.1 Directory Structure

```
process-simulation-engine/
├── Next-Activity-Prediction/
│   └── process_transformer/
│       ├── __init__.py
│       ├── predictor.py          # Main predictor class
│       ├── downloader.py         # HuggingFace download utilities
│       └── prepare_data.py       # Data preparation script
├── models/
│   └── process_transformer/      # Local cache (auto-created)
│       └── v1.0.0/
│           ├── model.keras
│           ├── vocab.json
│           └── config.json
└── simulation/
    └── engine.py                 # Updated with PT predictor
```

### 6.2 Predictor Implementation

Create: `Next-Activity-Prediction/process_transformer/predictor.py`

```python
"""
ProcessTransformer predictor for simulation engine.

Downloads model from HuggingFace Hub and provides next-activity predictions.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ProcessTransformerPredictor:
    """
    Next activity predictor using ProcessTransformer model.

    Downloads model from HuggingFace Hub on first use.
    Provides probabilistic predictions for stochastic simulation.
    """

    # HuggingFace repository (update with your org/repo)
    DEFAULT_REPO_ID = "bpso25/bpic17-process-transformer"
    DEFAULT_REVISION = "main"  # or specific version tag like "v1.0.0"

    # Local cache directory
    CACHE_DIR = Path.home() / ".cache" / "bpic17-simulation" / "process_transformer"

    # End-of-case activities
    END_ACTIVITIES = {"A_Cancelled", "A_Complete", "<END>"}
    START_ACTIVITY = "A_Create Application"

    def __init__(
        self,
        repo_id: str = None,
        revision: str = None,
        cache_dir: str = None,
        temperature: float = 1.0,
        seed: int = 42,
        offline: bool = False,
    ):
        """
        Initialize the ProcessTransformer predictor.

        Args:
            repo_id: HuggingFace repository ID (default: DEFAULT_REPO_ID)
            revision: Git revision/tag to use (default: main)
            cache_dir: Local cache directory for model files
            temperature: Sampling temperature (1.0 = neutral, <1 = more deterministic)
            seed: Random seed for reproducibility
            offline: If True, only use cached model (don't download)
        """
        self.repo_id = repo_id or self.DEFAULT_REPO_ID
        self.revision = revision or self.DEFAULT_REVISION
        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE_DIR / self.revision
        self.temperature = temperature
        self.offline = offline
        self.rng = np.random.default_rng(seed)

        # Load model and vocabulary
        self._load_model()

        logger.info(
            f"ProcessTransformerPredictor initialized: "
            f"vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len}"
        )

    def _load_model(self):
        """Download and load model from HuggingFace Hub."""
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths
        model_path = self.cache_dir / "model.keras"
        vocab_path = self.cache_dir / "vocab.json"
        config_path = self.cache_dir / "config.json"

        # Download if not cached (or not offline mode)
        if not self.offline:
            self._download_if_needed(model_path, vocab_path, config_path)

        # Verify files exist
        for path in [model_path, vocab_path, config_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}. "
                    f"Set offline=False to download from HuggingFace."
                )

        # Load TensorFlow model
        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(model_path))

        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        self.activity_to_idx = vocab_data['activity_to_idx']
        self.idx_to_activity = {int(k): v for k, v in vocab_data['idx_to_activity'].items()}
        self.vocab_size = vocab_data['vocab_size']

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.max_seq_len = config['max_seq_len']
        self.pad_idx = self.activity_to_idx.get('<PAD>', 0)
        self.start_idx = self.activity_to_idx.get('<START>', 1)
        self.end_idx = self.activity_to_idx.get('<END>', 2)

    def _download_if_needed(self, model_path: Path, vocab_path: Path, config_path: Path):
        """Download model files from HuggingFace Hub if not cached."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.warning(
                "huggingface_hub not installed. "
                "Install with: pip install huggingface_hub"
            )
            return

        files_to_download = [
            ("model.keras", model_path),
            ("vocab.json", vocab_path),
            ("config.json", config_path),
        ]

        for repo_file, local_path in files_to_download:
            if not local_path.exists():
                logger.info(f"Downloading {repo_file} from {self.repo_id}...")
                try:
                    downloaded = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=repo_file,
                        revision=self.revision,
                        local_dir=str(self.cache_dir),
                        local_dir_use_symlinks=False,
                    )
                    logger.info(f"Downloaded to: {downloaded}")
                except Exception as e:
                    logger.error(f"Failed to download {repo_file}: {e}")
                    raise

    def predict(self, case_state) -> Tuple[str, bool]:
        """
        Predict the next activity for a case.

        Implements the NextActivityPredictor protocol from simulation/engine.py.

        Args:
            case_state: CaseState object with activity_history attribute.

        Returns:
            Tuple of (next_activity_name, is_case_ended).
        """
        # Handle first activity
        if not case_state.activity_history:
            return self.START_ACTIVITY, False

        # Check if already at end
        current = case_state.activity_history[-1]
        if current in self.END_ACTIVITIES:
            return current, True

        # Convert history to indices
        prefix_indices = self._encode_prefix(case_state.activity_history)

        # Get probability distribution
        probs = self._get_probabilities(prefix_indices)

        # Apply repetition penalty to avoid loops
        probs = self._apply_repetition_penalty(probs, case_state.activity_history)

        # Sample next activity
        next_idx = self._sample(probs)
        next_activity = self.idx_to_activity.get(next_idx, self.START_ACTIVITY)

        # Check for end
        is_end = next_activity in self.END_ACTIVITIES

        return next_activity, is_end

    def _encode_prefix(self, history: List[str]) -> List[int]:
        """Convert activity history to token indices."""
        indices = [self.start_idx]  # Start with <START> token

        for activity in history:
            if activity in self.activity_to_idx:
                indices.append(self.activity_to_idx[activity])
            else:
                # Unknown activity - skip or use special token
                logger.warning(f"Unknown activity: {activity}")

        return indices

    def _get_probabilities(self, prefix_indices: List[int]) -> np.ndarray:
        """Get probability distribution from model."""
        # Pad to max sequence length
        if len(prefix_indices) < self.max_seq_len:
            padded = [self.pad_idx] * (self.max_seq_len - len(prefix_indices)) + prefix_indices
        else:
            padded = prefix_indices[-self.max_seq_len:]

        # Predict
        input_array = np.array([padded])
        probs = self.model.predict(input_array, verbose=0)[0]

        return probs

    def _apply_repetition_penalty(
        self,
        probs: np.ndarray,
        history: List[str],
        penalty: float = 0.5,
        window: int = 3,
    ) -> np.ndarray:
        """
        Reduce probability of recently seen activities to avoid loops.

        Args:
            probs: Original probability distribution
            history: Activity history
            penalty: Multiplicative penalty (0.5 = halve probability)
            window: How many recent activities to penalize

        Returns:
            Adjusted probability distribution (normalized)
        """
        probs = probs.copy()

        # Get recent activities
        recent = history[-window:] if len(history) >= window else history

        for activity in recent:
            if activity in self.activity_to_idx:
                idx = self.activity_to_idx[activity]
                probs[idx] *= penalty

        # Also penalize PAD and START tokens
        probs[self.pad_idx] = 0
        probs[self.start_idx] = 0

        # Renormalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            # Fallback: uniform over non-special tokens
            probs = np.ones_like(probs)
            probs[self.pad_idx] = 0
            probs[self.start_idx] = 0
            probs = probs / probs.sum()

        return probs

    def _sample(self, probs: np.ndarray) -> int:
        """Sample from probability distribution with temperature."""
        if self.temperature != 1.0:
            # Apply temperature
            probs = np.power(probs, 1.0 / self.temperature)
            probs = probs / probs.sum()

        return self.rng.choice(len(probs), p=probs)

    def get_distribution(self, case_state) -> Dict[str, float]:
        """
        Get full probability distribution for next activity.

        Useful for analysis and debugging.

        Args:
            case_state: CaseState object.

        Returns:
            Dictionary mapping activity names to probabilities.
        """
        if not case_state.activity_history:
            prefix_indices = [self.start_idx]
        else:
            prefix_indices = self._encode_prefix(case_state.activity_history)

        probs = self._get_probabilities(prefix_indices)

        return {
            self.idx_to_activity[i]: float(p)
            for i, p in enumerate(probs)
            if p > 0.001  # Only include non-negligible probabilities
        }
```

### 6.3 Downloader Utility

Create: `Next-Activity-Prediction/process_transformer/downloader.py`

```python
"""
Utility for downloading ProcessTransformer models from HuggingFace Hub.

Can be used standalone to pre-download models before simulation.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(
    repo_id: str,
    revision: str = "main",
    cache_dir: str = None,
    force: bool = False,
):
    """
    Download ProcessTransformer model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        revision: Git revision/tag
        cache_dir: Local cache directory
        force: If True, re-download even if cached
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "bpic17-simulation" / "process_transformer" / revision
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    files = ["model.keras", "vocab.json", "config.json", "training_metadata.json"]

    for filename in files:
        local_path = cache_dir / filename

        if local_path.exists() and not force:
            logger.info(f"Already cached: {filename}")
            continue

        logger.info(f"Downloading: {filename}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Downloaded to: {local_path}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            if filename == "training_metadata.json":
                logger.info("(training_metadata.json is optional)")
            else:
                raise

    logger.info(f"\nModel cached at: {cache_dir}")
    return cache_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ProcessTransformer model")
    parser.add_argument(
        "--repo-id",
        default="bpso25/bpic17-process-transformer",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision or tag",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local cache directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download",
    )

    args = parser.parse_args()
    download_model(args.repo_id, args.revision, args.cache_dir, args.force)
```

### 6.4 Engine Integration

Update: `simulation/engine.py`

Add to imports (around line 44):
```python
# Import ProcessTransformerPredictor (lazy load to avoid TensorFlow startup time)
```

Add to `_create_next_activity_predictor()` method (around line 370):
```python
# Try ProcessTransformer model (from HuggingFace Hub)
pt_cache_dir = Path.home() / ".cache" / "bpic17-simulation" / "process_transformer"
if (pt_cache_dir / "main" / "model.keras").exists():
    try:
        logger.info("Loading ProcessTransformerPredictor...")
        import sys
        project_root = Path(__file__).parent.parent
        pt_path = project_root / "Next-Activity-Prediction" / "process_transformer"
        if str(pt_path) not in sys.path:
            sys.path.insert(0, str(pt_path))

        from predictor import ProcessTransformerPredictor
        return ProcessTransformerPredictor(offline=True)
    except Exception as e:
        logger.warning(f"Could not load ProcessTransformer: {e}")
```

### 6.5 Package Init File

Create: `Next-Activity-Prediction/process_transformer/__init__.py`

```python
"""
ProcessTransformer integration for BPIC17 simulation.
"""

from .predictor import ProcessTransformerPredictor
from .downloader import download_model

__all__ = ['ProcessTransformerPredictor', 'download_model']
```

---

## 7. Phase 5: Testing & Validation

### 7.1 Unit Tests

Create: `tests/test_process_transformer.py`

```python
"""
Unit tests for ProcessTransformer predictor.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch


class TestProcessTransformerPredictor:
    """Tests for ProcessTransformerPredictor class."""

    @pytest.fixture
    def mock_case_state(self):
        """Create mock CaseState for testing."""
        case = Mock()
        case.activity_history = []
        return case

    def test_first_activity_prediction(self, mock_case_state):
        """First prediction should return START_ACTIVITY."""
        # Skip if model not available
        pytest.importorskip("tensorflow")

        # This test requires the model to be downloaded
        # Skip in CI environments
        from process_transformer import ProcessTransformerPredictor

        try:
            predictor = ProcessTransformerPredictor(offline=True)
        except FileNotFoundError:
            pytest.skip("Model not cached locally")

        activity, is_end = predictor.predict(mock_case_state)
        assert activity == "A_Create Application"
        assert is_end is False

    def test_end_activity_detection(self, mock_case_state):
        """Should detect end activities correctly."""
        pytest.importorskip("tensorflow")
        from process_transformer import ProcessTransformerPredictor

        try:
            predictor = ProcessTransformerPredictor(offline=True)
        except FileNotFoundError:
            pytest.skip("Model not cached locally")

        mock_case_state.activity_history = ["A_Create Application", "A_Complete"]
        activity, is_end = predictor.predict(mock_case_state)
        assert is_end is True

    def test_vocabulary_loading(self):
        """Vocabulary should load correctly from cache."""
        pytest.importorskip("tensorflow")
        from process_transformer import ProcessTransformerPredictor

        try:
            predictor = ProcessTransformerPredictor(offline=True)
        except FileNotFoundError:
            pytest.skip("Model not cached locally")

        assert predictor.vocab_size > 0
        assert '<PAD>' in predictor.activity_to_idx
        assert '<START>' in predictor.activity_to_idx
        assert '<END>' in predictor.activity_to_idx

    def test_probability_distribution(self, mock_case_state):
        """get_distribution should return valid probabilities."""
        pytest.importorskip("tensorflow")
        from process_transformer import ProcessTransformerPredictor

        try:
            predictor = ProcessTransformerPredictor(offline=True)
        except FileNotFoundError:
            pytest.skip("Model not cached locally")

        mock_case_state.activity_history = ["A_Create Application"]
        dist = predictor.get_distribution(mock_case_state)

        assert isinstance(dist, dict)
        assert len(dist) > 0
        assert all(0 <= p <= 1 for p in dist.values())
```

### 7.2 Integration Tests

Add to: `integration/test_integration.py`

```python
def test_process_transformer_integration():
    """Test simulation with ProcessTransformer predictor."""
    from process_transformer import ProcessTransformerPredictor

    try:
        predictor = ProcessTransformerPredictor(offline=True)
    except FileNotFoundError:
        pytest.skip("ProcessTransformer model not cached")

    # Create minimal simulation
    config = SimulationConfig.all_advanced(
        event_log_path="eventlog/eventlog.xes.gz",
        num_cases=10,
    )

    # Run simulation
    events = run_simulation(config, ...)

    # Verify outputs
    assert len(events) > 0
    assert all('concept:name' in e for e in events)
```

### 7.3 Benchmark Comparison

Create: `benchmarks/compare_predictors.py`

```python
"""
Compare ProcessTransformer against existing predictors.

Metrics:
- Accuracy (next activity prediction on held-out cases)
- Simulation fidelity (trace similarity to ground truth)
- Inference latency (time per prediction)
"""

import time
import pandas as pd
from collections import defaultdict


def benchmark_predictors(
    event_log_path: str,
    num_cases: int = 100,
    predictors: list = None,
):
    """Run benchmark comparison of predictors."""
    results = defaultdict(dict)

    # Load ground truth
    df = load_event_log(event_log_path)
    test_cases = df.groupby('case:concept:name').head(num_cases)

    for predictor_name, predictor in predictors:
        # Measure accuracy
        correct, total = 0, 0
        latencies = []

        for case_id, case_df in test_cases.groupby('case:concept:name'):
            history = []
            for _, row in case_df.iterrows():
                if history:
                    # Predict
                    start = time.perf_counter()
                    predicted, _ = predictor.predict(MockCaseState(history))
                    latencies.append(time.perf_counter() - start)

                    # Compare to actual
                    actual = row['concept:name']
                    if predicted == actual:
                        correct += 1
                    total += 1

                history.append(row['concept:name'])

        results[predictor_name] = {
            'accuracy': correct / total if total > 0 else 0,
            'avg_latency_ms': sum(latencies) / len(latencies) * 1000,
            'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
        }

    return pd.DataFrame(results).T
```

---

## 8. Pitfalls & Mitigations

### 8.1 Critical Issues

| Issue | Impact | Mitigation | Detection |
|-------|--------|------------|-----------|
| **Vocabulary mismatch** | Predictions fail for unknown activities | Include all activities in training vocab, use UNK token | Unit test vocabulary coverage |
| **Infinite loops** | Simulation hangs | Repetition penalty, max_activities_per_case limit | Integration test loop detection |
| **TensorFlow version conflict** | Model loading fails | Pin TensorFlow version, use SavedModel format | CI tests on multiple TF versions |
| **Memory exhaustion** | OOM during inference | Limit batch size, use CPU inference | Monitor memory in benchmarks |

### 8.2 Data Issues

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Lifecycle filtering** | Missing events if wrong filter | Document and validate lifecycle choice |
| **Case length outliers** | Training instability | Cap max_case_length, document truncation |
| **Rare activities** | Poor prediction accuracy | Ensure minimum frequency in training data |
| **Timestamp parsing** | Incorrect sequence order | Validate timestamp format before training |

### 8.3 Inference Issues

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Cold start latency** | First prediction slow (~5s) | Pre-load model at engine init |
| **GPU/CPU mismatch** | Different behavior on different machines | Test on CPU, document GPU benefits |
| **Temperature tuning** | Too deterministic or too random | Provide sensible default (1.0), allow override |
| **Probability underflow** | Zero probabilities after softmax | Add epsilon to probabilities before sampling |

### 8.4 Deployment Issues

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **HuggingFace rate limits** | Download failures | Cache models locally, document offline mode |
| **Model versioning** | Breaking changes | Use semantic versioning, pin revision in engine |
| **Large model size** | Slow downloads, storage concerns | Document model size (~50MB), support compression |
| **Authentication** | Private repo access fails | Document HF_TOKEN environment variable |

---

## 9. Maintenance & Versioning

### 9.1 Model Update Workflow

```
1. Identify improvement opportunity
   └── New training data, hyperparameter tuning, architecture change

2. Train new model in Colab
   └── Use same notebook, document changes

3. Evaluate against previous version
   └── Run benchmarks, compare metrics

4. Publish to HuggingFace (new version tag)
   └── git tag v1.x.x, update model card

5. Update engine to use new version
   └── Update DEFAULT_REVISION in predictor.py

6. Run integration tests
   └── Verify simulation still works

7. Document in CHANGELOG.md
   └── What changed, why, performance impact
```

### 9.2 Version Compatibility Matrix

| Engine Version | ProcessTransformer Version | TensorFlow | Notes |
|----------------|---------------------------|------------|-------|
| 1.0.0 | v1.0.0 | 2.15.x | Initial release |
| 1.1.0 | v1.0.0 - v1.1.x | 2.15.x | Added temperature param |
| 2.0.0 | v2.0.0+ | 2.16.x | New vocabulary format |

### 9.3 Deprecation Policy

1. **Announce deprecation** in model card 1 month before removal
2. **Log warning** in predictor when using deprecated version
3. **Remove support** after 2 engine releases
4. **Archive old models** (don't delete from HuggingFace)

---

## 10. Appendix: Code Templates

### 10.1 Quick Start Script

Save as: `scripts/setup_process_transformer.sh`

```bash
#!/bin/bash
# Quick setup for ProcessTransformer integration

set -e

echo "=== Setting up ProcessTransformer ==="

# 1. Install dependencies
echo "Installing Python dependencies..."
pip install tensorflow==2.15.0 huggingface_hub

# 2. Download model
echo "Downloading model from HuggingFace..."
python -c "
from Next-Activity-Prediction.process_transformer import download_model
download_model('bpso25/bpic17-process-transformer', 'main')
"

# 3. Verify installation
echo "Verifying installation..."
python -c "
from Next-Activity-Prediction.process_transformer import ProcessTransformerPredictor
p = ProcessTransformerPredictor(offline=True)
print(f'Loaded model with vocab_size={p.vocab_size}')
"

echo "=== Setup complete ==="
```

### 10.2 Colab Quick Start Cell

```python
# One-cell setup for Colab training
!pip install -q tensorflow==2.15.0 pm4py huggingface_hub scikit-learn
!git clone -q https://github.com/Zaharah/processtransformer.git

from google.colab import drive, files
drive.mount('/content/drive')

print("Upload your BPIC17 data (bpic17_pt.csv):")
uploaded = files.upload()

print("\n✓ Environment ready! Continue with training cells.")
```

### 10.3 Local Testing Script

Save as: `scripts/test_process_transformer.py`

```python
"""
Quick local test of ProcessTransformer predictor.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Next-Activity-Prediction" / "process_transformer"))

from predictor import ProcessTransformerPredictor
from simulation.case_manager import CaseState


def main():
    print("Loading ProcessTransformerPredictor...")
    predictor = ProcessTransformerPredictor(offline=True)

    print(f"Vocabulary size: {predictor.vocab_size}")
    print(f"Max sequence length: {predictor.max_seq_len}")

    # Simulate a case
    case = CaseState(
        case_id="Test_001",
        case_type="Home improvement",
        start_time=None,
    )

    print("\nSimulating case:")
    for i in range(10):
        activity, is_end = predictor.predict(case)
        print(f"  {i+1}. {activity}" + (" [END]" if is_end else ""))

        if is_end:
            break

        case.add_activity(activity, "User_1")

    print("\nDone!")


if __name__ == "__main__":
    main()
```

---

## Checklist Summary

### Before Training
- [ ] BPIC17 event log available
- [ ] Data preparation script tested
- [ ] HuggingFace account created
- [ ] Colab notebook ready

### After Training
- [ ] Model uploaded to HuggingFace
- [ ] Model card complete
- [ ] Validation accuracy acceptable (>60%)
- [ ] Training metadata saved

### Before Deployment
- [ ] ProcessTransformerPredictor tested locally
- [ ] Engine integration complete
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Benchmark comparison documented

### Ongoing
- [ ] Monitor simulation quality
- [ ] Track inference latency
- [ ] Update model when data changes
- [ ] Maintain version compatibility

---

*Document maintained by: BPSO25 Team*
*Last updated: 2024-01-09*
