"""
Data preprocessing for suffix prediction.

Filters event logs to start/complete lifecycles and prepares sequences for suffix prediction training.
"""

import os
import logging
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import pm4py
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_event_log(log_path: str) -> pd.DataFrame:
    """
    Load event log from XES or CSV file.
    
    Args:
        log_path: Path to event log file
        
    Returns:
        DataFrame with event log
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Event log not found: {log_path}")
    
    logger.info(f"Loading event log from {log_path}")
    
    if log_path.endswith('.csv') or log_path.endswith('.csv.gz'):
        df = pd.read_csv(log_path)
    elif log_path.endswith('.xes') or log_path.endswith('.xes.gz'):
        if not PM4PY_AVAILABLE:
            raise ImportError("pm4py is required to load .xes files. Install with: pip install pm4py")
        log = pm4py.read_xes(log_path)
        df = pm4py.convert_to_dataframe(log)
    else:
        raise ValueError(f"Unsupported file format: {log_path}")
    
    logger.info(f"Loaded {len(df)} events, {df['case:concept:name'].nunique()} cases")
    return df


def filter_lifecycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter event log to only include 'start' and 'complete' lifecycle transitions.
    
    Args:
        df: Event log DataFrame
        
    Returns:
        Filtered DataFrame
    """
    if 'lifecycle:transition' not in df.columns:
        logger.warning("No lifecycle:transition column found, using all events")
        return df
    
    before = len(df)
    df_filtered = df[df['lifecycle:transition'].isin(['start', 'complete'])].copy()
    after = len(df_filtered)
    
    logger.info(f"Filtered to start/complete lifecycles: {before:,} -> {after:,} ({after/before:.1%})")
    return df_filtered


def extract_case_sequences(df: pd.DataFrame, min_length: int = 2, max_length: int = 200) -> List[List[str]]:
    """
    Extract activity sequences from event log, grouped by case.
    Each sequence ends with an "END" token.
    
    Args:
        df: Event log DataFrame (already filtered to start/complete)
        min_length: Minimum case length (excluding END token)
        max_length: Maximum case length (excluding END token)
        
    Returns:
        List of activity sequences, each ending with "END" token
    """
    required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    sequences = []
    case_counts = {}
    
    df_sorted = df.sort_values(['case:concept:name', 'time:timestamp'])
    
    for case_id, case_df in df_sorted.groupby('case:concept:name'):
        activities = case_df['concept:name'].tolist()
        
        if len(activities) < min_length:
            continue
        
        if len(activities) > max_length:
            activities = activities[:max_length]
        
        # Append END token to each sequence
        activities.append("END")
        sequences.append(activities)
        case_counts[case_id] = len(activities)
    
    logger.info(f"Extracted {len(sequences)} case sequences")
    logger.info(f"Average sequence length: {np.mean([len(s) for s in sequences]):.1f} (including END)")
    logger.info(f"Min/Max sequence length: {min(len(s) for s in sequences)}/{max(len(s) for s in sequences)}")
    
    return sequences


def create_vocabulary(sequences: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create vocabulary mapping from activity names to indices.
    Includes "END" as a special activity token.
    
    Args:
        sequences: List of activity sequences (which include END tokens)
        
    Returns:
        Tuple of (activity_to_idx, idx_to_activity) dictionaries
    """
    all_activities = set()
    for seq in sequences:
        all_activities.update(seq)
    
    # Ensure END is in the vocabulary
    all_activities.add("END")
    
    # Sort activities, but put END at the end for easier handling
    activities_without_end = sorted([a for a in all_activities if a != "END"])
    activities_sorted = activities_without_end + ["END"]
    
    activity_to_idx = {act: idx + 1 for idx, act in enumerate(activities_sorted)}
    activity_to_idx['<PAD>'] = 0
    idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}
    
    logger.info(f"Created vocabulary with {len(activity_to_idx)} activities (including END)")
    logger.info(f"END token index: {activity_to_idx.get('END', 'NOT FOUND')}")
    
    return activity_to_idx, idx_to_activity


def pad_sequence(sequence: List[int], max_length: int, pad_value: int = 0) -> List[int]:
    """
    Pad or truncate sequence to fixed length.
    
    Args:
        sequence: Sequence of indices
        max_length: Target length
        pad_value: Value to use for padding (default: 0 for PAD token)
        
    Returns:
        Padded/truncated sequence as list of indices
    """
    if len(sequence) < max_length:
        sequence = sequence + [pad_value] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    
    return sequence


def prepare_suffix_training_data(
    sequences: List[List[str]],
    activity_to_idx: Dict[str, int],
    prefix_length: int,
    suffix_length: int,
    min_prefix_length: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for suffix prediction training.
    
    For each sequence [a1, a2, ..., an, END], creates training samples:
    - Input (prefix): [a1, a2, ..., a_i] (padded/truncated to prefix_length)
    - Output (suffix): [a_{i+1}, a_{i+2}, ..., END] (padded/truncated to suffix_length)
    
    Args:
        sequences: List of activity sequences (ending with END token)
        activity_to_idx: Vocabulary mapping (including END)
        prefix_length: Target length for input prefix
        suffix_length: Target length for output suffix
        min_prefix_length: Minimum prefix length to consider
        
    Returns:
        Tuple of (X_prefix, y_suffix) numpy arrays
        - X_prefix: (n_samples, prefix_length) - input prefixes
        - y_suffix: (n_samples, suffix_length) - target suffixes
    """
    X_prefix = []
    y_suffix = []
    
    for seq in sequences:
        if len(seq) < min_prefix_length + 1:  # Need at least prefix + 1 activity
            continue
        
        # Create training samples by splitting at different positions
        for i in range(min_prefix_length, len(seq)):
            prefix = seq[:i]
            suffix = seq[i:]  # This includes END if it exists
            
            # Convert to indices
            prefix_indices = [activity_to_idx.get(act, 0) for act in prefix]
            suffix_indices = [activity_to_idx.get(act, 0) for act in suffix]
            
            # Pad/truncate
            prefix_padded = pad_sequence(prefix_indices, prefix_length)
            suffix_padded = pad_sequence(suffix_indices, suffix_length)
            
            X_prefix.append(prefix_padded)
            y_suffix.append(suffix_padded)
    
    X_prefix = np.array(X_prefix, dtype=np.int32)
    y_suffix = np.array(y_suffix, dtype=np.int32)
    
    logger.info(f"Prepared {len(X_prefix)} training samples for suffix prediction")
    logger.info(f"Prefix shape: {X_prefix.shape}, Suffix shape: {y_suffix.shape}")
    
    # Count samples that predict END
    end_token_idx = activity_to_idx.get("END", -1)
    samples_with_end = (y_suffix == end_token_idx).any(axis=1).sum()
    logger.info(f"Samples with END token: {samples_with_end} ({samples_with_end/len(y_suffix):.2%})")
    
    return X_prefix, y_suffix


def preprocess_event_log_for_suffix(
    log_path: str,
    prefix_length: int = 50,
    suffix_length: int = 30,
    min_case_length: int = 2,
    max_case_length: int = 200,
    min_prefix_length: int = 1,
    validation_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Complete preprocessing pipeline from event log to suffix prediction training data.
    
    Args:
        log_path: Path to event log file
        prefix_length: Target length for input prefix
        suffix_length: Target length for output suffix
        min_case_length: Minimum case length
        max_case_length: Maximum case length
        min_prefix_length: Minimum prefix length to consider
        validation_split: Fraction of data for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, activity_to_idx, idx_to_activity)
    """
    np.random.seed(random_seed)
    
    df = load_event_log(log_path)
    df = filter_lifecycles(df)
    sequences = extract_case_sequences(df, min_case_length, max_case_length)
    activity_to_idx, idx_to_activity = create_vocabulary(sequences)
    
    X, y = prepare_suffix_training_data(
        sequences, 
        activity_to_idx, 
        prefix_length, 
        suffix_length,
        min_prefix_length
    )
    
    split_idx = int(len(X) * (1 - validation_split))
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    return (X_train, y_train, 
            X_val, y_val,
            activity_to_idx, idx_to_activity)

