"""
Tests for LogExporter.
"""

import pytest
import tempfile
import os
from datetime import datetime

from simulation.log_exporter import LogExporter


class TestLogExporter:
    """Tests for LogExporter class."""
    
    @pytest.fixture
    def sample_events(self):
        """Sample events for testing."""
        return [
            {
                'case:concept:name': 'case_1',
                'concept:name': 'A_Create Application',
                'org:resource': 'User_42',
                'time:timestamp': datetime(2024, 1, 1, 9, 0),
                'lifecycle:transition': 'complete',
                'case:LoanGoal': 'Home improvement',
            },
            {
                'case:concept:name': 'case_1',
                'concept:name': 'A_Submitted',
                'org:resource': 'User_42',
                'time:timestamp': datetime(2024, 1, 1, 9, 30),
                'lifecycle:transition': 'complete',
                'case:LoanGoal': 'Home improvement',
            },
        ]
    
    def test_to_dataframe(self, sample_events):
        """Events should convert to DataFrame."""
        df = LogExporter.to_dataframe(sample_events)
        
        assert len(df) == 2
        assert 'case:concept:name' in df.columns
        assert 'concept:name' in df.columns
        assert df.iloc[0]['case:concept:name'] == 'case_1'
    
    def test_to_csv(self, sample_events):
        """Events should export to CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        
        try:
            LogExporter.to_csv(sample_events, path)
            
            assert os.path.exists(path)
            
            # Read back and verify
            import pandas as pd
            df = pd.read_csv(path)
            assert len(df) == 2
            assert 'case:concept:name' in df.columns
        finally:
            os.unlink(path)
    
    def test_to_xes(self, sample_events):
        """Events should export to XES."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xes', delete=False) as f:
            path = f.name
        
        try:
            LogExporter.to_xes(sample_events, path)
            
            assert os.path.exists(path)
            
            # Read back with pm4py and verify
            import pm4py
            log = pm4py.read_xes(path)
            df = pm4py.convert_to_dataframe(log)
            
            assert len(df) == 2
            assert 'concept:name' in df.columns
        finally:
            os.unlink(path)
    
    def test_validate_xes_columns_valid(self, sample_events):
        """Validation should pass for valid events."""
        missing = LogExporter.validate_xes_columns(sample_events)
        assert len(missing) == 0
    
    def test_validate_xes_columns_missing(self):
        """Validation should detect missing columns."""
        incomplete_events = [
            {'case:concept:name': 'case_1', 'concept:name': 'Activity'}
            # Missing time:timestamp
        ]
        
        missing = LogExporter.validate_xes_columns(incomplete_events)
        assert 'time:timestamp' in missing
    
    def test_validate_xes_columns_empty(self):
        """Validation should handle empty list."""
        missing = LogExporter.validate_xes_columns([])
        assert len(missing) == 3  # All required columns
