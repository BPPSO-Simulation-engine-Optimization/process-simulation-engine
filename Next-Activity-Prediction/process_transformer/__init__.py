"""
ProcessTransformer integration for BPIC17 simulation.

Provides a Transformer-based next activity predictor.
"""

from .predictor import ProcessTransformerPredictor
from .downloader import download_model

__all__ = ['ProcessTransformerPredictor', 'download_model']
