"""
Simulation Module - Discrete Event Simulation Engine for BPIC17.

MVP Components:
- events: SimulationEvent, EventType
- event_queue: Priority queue implementation
- clock: Virtual time management
- engine: DES main loop
- case_manager: Active case tracking
- log_exporter: CSV and XES export
"""

from .events import SimulationEvent, EventType
from .event_queue import EventQueue
from .clock import SimulationClock
from .engine import DESEngine
from .case_manager import CaseState
from .log_exporter import LogExporter

# Import advanced simulation utilities
import sys
from pathlib import Path
next_activity_path = Path(__file__).parent.parent / "Next-Activity-Prediction"
if str(next_activity_path) not in sys.path:
    sys.path.insert(0, str(next_activity_path))

try:
    from advanced.simulation import load_simulation_assets, decision_function_advanced
except ImportError:
    # Fallback if advanced module is not available
    load_simulation_assets = None
    decision_function_advanced = None

__all__ = [
    'SimulationEvent',
    'EventType', 
    'EventQueue',
    'SimulationClock',
    'DESEngine',
    'CaseState',
    'LogExporter',
    'load_simulation_assets',
    'decision_function_advanced',
]
