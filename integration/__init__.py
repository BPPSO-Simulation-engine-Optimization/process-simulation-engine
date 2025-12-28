"""
Integration module for connecting the DES simulation engine with prediction components.

This module provides:
- SimulationConfig: Configuration dataclass for basic/advanced mode selection
- setup_simulation: Factory function to wire up all components
"""

from .config import SimulationConfig
from .setup import setup_simulation

__all__ = ['SimulationConfig', 'setup_simulation']
