"""
Utility functions for next activity prediction training.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_class_weights(
    y: np.ndarray,
    method: str = "balanced",
    end_token_idx: Optional[int] = None,
    end_token_weight: Optional[float] = None,
    vocab_size: Optional[int] = None
) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced classification.
    
    Args:
        y: Array of class indices (labels)
        method: Weighting method - "balanced" (sklearn style), "inverse_freq", or "custom"
        end_token_idx: Index of END token (if None, auto-detect as least frequent)
        end_token_weight: Manual weight for END token (overrides auto-calculation if provided)
        vocab_size: Total vocabulary size (if None, inferred from y)
        
    Returns:
        Dictionary mapping class index to weight
    """
    if vocab_size is None:
        vocab_size = int(y.max() + 1)
    
    # Count class frequencies
    unique, counts = np.unique(y, return_counts=True)
    class_counts = np.zeros(vocab_size, dtype=np.int64)
    class_counts[unique] = counts
    
    total_samples = len(y)
    
    if method == "balanced":
        # sklearn-style balanced weights: n_samples / (n_classes * count)
        n_classes = len(unique)
        class_weights = {}
        for class_idx in range(vocab_size):
            if class_counts[class_idx] > 0:
                class_weights[class_idx] = total_samples / (n_classes * class_counts[class_idx])
            else:
                class_weights[class_idx] = 1.0
                
    elif method == "inverse_freq":
        # Inverse frequency weighting
        class_weights = {}
        for class_idx in range(vocab_size):
            if class_counts[class_idx] > 0:
                class_weights[class_idx] = total_samples / class_counts[class_idx]
            else:
                class_weights[class_idx] = 1.0
                
    elif method == "custom":
        # Use provided weights or default to balanced
        class_weights = {}
        for class_idx in range(vocab_size):
            if class_counts[class_idx] > 0:
                class_weights[class_idx] = total_samples / (len(unique) * class_counts[class_idx])
            else:
                class_weights[class_idx] = 1.0
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Override END token weight if specified
    if end_token_idx is not None and end_token_weight is not None:
        if end_token_idx in class_weights:
            old_weight = class_weights[end_token_idx]
            class_weights[end_token_idx] = end_token_weight
            logger.info(f"Overriding END token (idx={end_token_idx}) weight: {old_weight:.3f} -> {end_token_weight:.3f}")
        else:
            class_weights[end_token_idx] = end_token_weight
    
    # Log class weights summary
    end_count = class_counts[end_token_idx] if end_token_idx is not None and end_token_idx < len(class_counts) else 0
    end_weight = class_weights.get(end_token_idx, 1.0) if end_token_idx is not None else 1.0
    
    logger.info(f"Class weights (method={method}):")
    logger.info(f"  END token (idx={end_token_idx}): count={end_count}, weight={end_weight:.3f}")
    logger.info(f"  Min weight: {min(class_weights.values()):.3f}, Max weight: {max(class_weights.values()):.3f}")
    logger.info(f"  Average weight: {np.mean(list(class_weights.values())):.3f}")
    
    return class_weights


def calculate_position_weights(
    positions: np.ndarray,
    power: float = 1.5
) -> np.ndarray:
    """
    Calculate sample weights based on position in case.
    
    Later positions (closer to END) get higher weights.
    
    Args:
        positions: Array of relative positions (0.0 = start, 1.0 = end)
        power: Power for position weighting (higher = more emphasis on later positions)
        
    Returns:
        Array of position-based weights (normalized to have mean=1.0)
    """
    # Weight increases with position: weight = (position + epsilon)^power
    epsilon = 0.1  # Small value to avoid zero weights for early positions
    weights = np.power(positions + epsilon, power)
    
    # Normalize to have mean=1.0
    weights = weights / np.mean(weights)
    
    logger.info(f"Position weights (power={power}):")
    logger.info(f"  Min: {weights.min():.3f}, Max: {weights.max():.3f}, Mean: {weights.mean():.3f}")
    
    return weights


def combine_sample_weights(
    class_weights: Optional[Dict[int, float]],
    position_weights: Optional[np.ndarray],
    y: np.ndarray
) -> np.ndarray:
    """
    Combine class weights and position weights into final sample weights.
    
    Args:
        class_weights: Dictionary mapping class index to weight (None to skip)
        position_weights: Array of position-based weights (None to skip)
        y: Array of class labels
        
    Returns:
        Combined sample weights array
    """
    n_samples = len(y)
    weights = np.ones(n_samples, dtype=np.float32)
    
    # Apply class weights
    if class_weights is not None:
        for i, class_idx in enumerate(y):
            weights[i] *= class_weights.get(int(class_idx), 1.0)
    
    # Apply position weights
    if position_weights is not None:
        if len(position_weights) != n_samples:
            raise ValueError(f"Position weights length ({len(position_weights)}) != y length ({n_samples})")
        weights *= position_weights
    
    # Normalize to have mean=1.0
    weights = weights / np.mean(weights)
    
    logger.info(f"Combined sample weights: Min={weights.min():.3f}, Max={weights.max():.3f}, Mean={weights.mean():.3f}")
    
    return weights

