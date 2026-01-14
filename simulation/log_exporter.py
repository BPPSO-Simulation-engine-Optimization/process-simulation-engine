"""
Log Exporter - Export simulation results to CSV and XES formats.
"""

import pandas as pd
from typing import List, Dict
import pm4py


class LogExporter:
    """Export simulated events to standard event log formats."""
    
    @staticmethod
    def to_dataframe(events: List[Dict]) -> pd.DataFrame:
        """Convert events to a pandas DataFrame."""
        return pd.DataFrame(events)
    
    @staticmethod
    def to_csv(events: List[Dict], path: str) -> None:
        """
        Export events to CSV format.
        
        Args:
            events: List of event dictionaries.
            path: Output file path.
        """
        df = LogExporter.to_dataframe(events)
        df.to_csv(path, index=False)
    
    @staticmethod
    def to_xes(events: List[Dict], path: str) -> None:
        """
        Export events to XES format.
        
        Args:
            events: List of event dictionaries with XES-compatible column names.
            path: Output file path (.xes).
        """
        df = LogExporter.to_dataframe(events)
        
        # Ensure timestamp is datetime
        if 'time:timestamp' in df.columns:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        
        # Convert to event log and export
        log = pm4py.convert_to_event_log(df)
        pm4py.write_xes(log, path)
    
    @staticmethod
    def validate_xes_columns(events: List[Dict]) -> List[str]:
        """
        Check for required XES columns.
        
        Returns list of missing columns.
        """
        required = ['case:concept:name', 'concept:name', 'time:timestamp']
        if not events:
            return required
        
        first_event = events[0]
        return [col for col in required if col not in first_event]
