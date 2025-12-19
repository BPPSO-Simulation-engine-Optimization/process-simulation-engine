"""
Simulation Clock - Virtual time management for DES.
"""

from datetime import datetime, timedelta


class SimulationClock:
    """
    Manages simulation virtual time.
    
    Time advances only when events are processed.
    """
    
    def __init__(self, start_time: datetime = None):
        """
        Initialize the clock.
        
        Args:
            start_time: Initial simulation time. Defaults to now.
        """
        self._current_time = start_time or datetime.now()
        self._start_time = self._current_time
    
    @property
    def now(self) -> datetime:
        """Current simulation time."""
        return self._current_time
    
    def advance_to(self, timestamp: datetime) -> None:
        """
        Advance the clock to the given timestamp.
        
        Args:
            timestamp: New simulation time (must be >= current time).
            
        Raises:
            ValueError: If timestamp is in the past.
        """
        if timestamp < self._current_time:
            raise ValueError(
                f"Cannot go back in time: {timestamp} < {self._current_time}"
            )
        self._current_time = timestamp
    
    def elapsed(self) -> timedelta:
        """Time elapsed since simulation start."""
        return self._current_time - self._start_time
    
    def reset(self, start_time: datetime = None) -> None:
        """Reset the clock to a new start time."""
        self._current_time = start_time or datetime.now()
        self._start_time = self._current_time
