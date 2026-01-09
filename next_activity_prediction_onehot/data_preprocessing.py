"""
Data preprocessing for next activity prediction with one-hot encoding.

Filters event logs to start/complete lifecycles and prepares sequences for training.
Uses one-hot encoding instead of embedding vectors.
"""

import os
import logging
from typing import List, Tuple, Dict, Optional, Union
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
    
    all_activities.add("END")
    
    activities_without_end = sorted([a for a in all_activities if a != "END"])
    activities_sorted = activities_without_end + ["END"]
    
    activity_to_idx = {act: idx + 1 for idx, act in enumerate(activities_sorted)}
    activity_to_idx['<PAD>'] = 0
    idx_to_activity = {idx: act for act, idx in activity_to_idx.items()}
    
    logger.info(f"Created vocabulary with {len(activity_to_idx)} activities (including END)")
    logger.info(f"END token index: {activity_to_idx.get('END', 'NOT FOUND')}")
    
    return activity_to_idx, idx_to_activity


def onehot_encode_sequence(sequence: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Convert integer sequence to one-hot encoded sequence.
    
    Args:
        sequence: Integer sequence of shape (sequence_length,)
        vocab_size: Size of vocabulary (including PAD token)
        
    Returns:
        One-hot encoded sequence of shape (sequence_length, vocab_size)
    """
    sequence_length = len(sequence)
    onehot = np.zeros((sequence_length, vocab_size), dtype=np.float32)
    
    for i, idx in enumerate(sequence):
        if 0 <= idx < vocab_size:
            onehot[i, idx] = 1.0
    
    return onehot


def pad_sequence(sequence: List[str], activity_to_idx: Dict[str, int], max_length: int) -> List[int]:
    """
    Pad or truncate sequence to fixed length.
    
    Args:
        sequence: Activity sequence
        activity_to_idx: Vocabulary mapping
        max_length: Target length
        
    Returns:
        Padded/truncated sequence as list of indices
    """
    indices = [activity_to_idx.get(act, 0) for act in sequence]
    
    if len(indices) < max_length:
        indices = [0] * (max_length - len(indices)) + indices
    else:
        indices = indices[-max_length:]
    
    return indices


def prepare_training_data(
    sequences: List[List[str]],
    activity_to_idx: Dict[str, int],
    sequence_length: int,
    return_position_info: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Prepare sequences for training by creating input/output pairs with one-hot encoding.
    
    For each sequence [a1, a2, ..., an, END], creates:
    - Input: One-hot encoded [a1, a2, ..., a_i] (padded/truncated)
    - Output activity: a_{i+1} (which can be a regular activity or END)
    
    Args:
        sequences: List of activity sequences (ending with END token)
        activity_to_idx: Vocabulary mapping (including END)
        sequence_length: Target sequence length for padding/truncation
        return_position_info: If True, also return position information for weighting
        
    Returns:
        Tuple of (X_onehot, y_activity, position_info) numpy arrays
        X_onehot has shape (n_samples, sequence_length, vocab_size)
        position_info is None if return_position_info=False, otherwise contains relative positions
    """
    X_indices = []
    y_activity = []
    positions = [] if return_position_info else None
    
    for seq in sequences:
        if len(seq) < 2:
            continue
        
        seq_length = len(seq) - 1
        for i in range(1, len(seq)):
            input_seq = seq[:i]
            next_activity = seq[i]
            
            input_padded = pad_sequence(input_seq, activity_to_idx, sequence_length)
            X_indices.append(input_padded)
            
            next_idx = activity_to_idx.get(next_activity, 0)
            y_activity.append(next_idx)
            
            if return_position_info:
                rel_position = i / seq_length if seq_length > 0 else 0.0
                positions.append(rel_position)
    
    X_indices = np.array(X_indices, dtype=np.int32)
    y_activity = np.array(y_activity, dtype=np.int32)
    
    vocab_size = len(activity_to_idx)
    
    X_onehot = np.zeros((len(X_indices), sequence_length, vocab_size), dtype=np.float32)
    for i, seq_indices in enumerate(X_indices):
        X_onehot[i] = onehot_encode_sequence(seq_indices, vocab_size)
    
    end_token_idx = activity_to_idx.get("END", -1)
    end_count = (y_activity == end_token_idx).sum()
    
    logger.info(f"Prepared {len(X_onehot)} training samples")
    logger.info(f"Input shape: {X_onehot.shape} (samples, sequence_length, vocab_size)")
    logger.info(f"END token count: {end_count} ({end_count/len(y_activity):.2%})")
    
    if return_position_info and positions:
        positions = np.array(positions, dtype=np.float32)
        logger.info(f"Position range: [{positions.min():.3f}, {positions.max():.3f}]")
        return X_onehot, y_activity, positions
    
    return X_onehot, y_activity, None


def preprocess_event_log(
    log_path: str,
    sequence_length: int = 50,
    min_case_length: int = 2,
    max_case_length: int = 200,
    validation_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Complete preprocessing pipeline from event log to training data with one-hot encoding.
    
    Args:
        log_path: Path to event log file
        sequence_length: Target sequence length
        min_case_length: Minimum case length
        max_case_length: Maximum case length
        validation_split: Fraction of data for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_activity_train, X_val, y_activity_val, activity_to_idx, idx_to_activity)
        X_train and X_val are one-hot encoded arrays of shape (n_samples, sequence_length, vocab_size)
    """
    np.random.seed(random_seed)
    
    df = load_event_log(log_path)
    df = filter_lifecycles(df)
    sequences = extract_case_sequences(df, min_case_length, max_case_length)
    activity_to_idx, idx_to_activity = create_vocabulary(sequences)
    
    X, y_activity, _ = prepare_training_data(sequences, activity_to_idx, sequence_length, return_position_info=False)
    
    split_idx = int(len(X) * (1 - validation_split))
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    y_activity_train = y_activity[train_indices]
    
    X_val = X[val_indices]
    y_activity_val = y_activity[val_indices]
    
    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    return (X_train, y_activity_train, 
            X_val, y_activity_val,
            activity_to_idx, idx_to_activity)

