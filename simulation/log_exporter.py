"""
Log Exporter - Export simulation results to CSV and XES formats.
"""

import csv
import os
import pandas as pd
from typing import List, Dict, Optional
import pm4py

# Canonical column order for the simulated event log.
# All possible columns are listed here so that every appended batch uses an
# identical schema regardless of which case attributes have been populated.
_CANONICAL_COLUMNS = [
    'case:concept:name',
    'concept:name',
    'org:resource',
    'time:timestamp',
    'lifecycle:transition',
    'case:LoanGoal',
    'case:ApplicationType',
    'case:RequestedAmount',
    'CreditScore',
    'OfferedAmount',
    'FirstWithdrawalAmount',
    'NumberOfTerms',
    'MonthlyCost',
    'Selected',
    'Accepted',
]


class LogExporter:
    """Export simulated events to standard event log formats."""
    
    @staticmethod
    def to_dataframe(events: List[Dict], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert events to a pandas DataFrame.
        
        Args:
            events: List of event dictionaries.
            columns: If provided, reindex the DataFrame to exactly these columns
                     (missing ones become NaN, extra ones are dropped).
        """
        df = pd.DataFrame(events)
        if columns is not None:
            # Add any missing columns as NaN, then select in canonical order.
            for col in columns:
                if col not in df.columns:
                    df[col] = None
            df = df[columns]
        return df
    
    @staticmethod
    def to_csv(events: List[Dict], path: str) -> None:
        """
        Export events to CSV format.
        
        Args:
            events: List of event dictionaries.
            path: Output file path.
        """
        df = LogExporter.to_dataframe(events, columns=_CANONICAL_COLUMNS)
        df.to_csv(path, index=False)
    
    @staticmethod
    def append_to_csv(events: List[Dict], path: str, write_header: bool = False) -> None:
        """
        Append events to CSV file (incremental writing).

        All batches are normalised to _CANONICAL_COLUMNS so that every row in
        the output file has the same number of fields, regardless of whether
        offer-level attributes have been populated yet.
        
        Args:
            events: List of event dictionaries to append.
            path: Output file path.
            write_header: If True, write header row. If False, append without header.
        """
        if not events:
            return

        # Derive columns: if file already exists and we are NOT writing the
        # header, read the header from disk to guarantee exact column match.
        if not write_header and os.path.exists(path):
            with open(path, 'r', newline='', encoding='utf-8') as fh:
                existing_columns = next(csv.reader(fh))
        else:
            existing_columns = _CANONICAL_COLUMNS

        df = LogExporter.to_dataframe(events, columns=existing_columns)
        df.to_csv(path, mode='a', index=False, header=write_header)
    
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
