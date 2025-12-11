
import unittest
import os
import pandas as pd
from datetime import datetime
from resources.resource_allocation import ResourceAllocator

# Path to the log file (assuming run from project root)
LOG_PATH = "eventlog.xes.gz"
SAMPLE_SIZE = 1000

class TestResourceAllocator(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LOG_PATH):
            raise unittest.SkipTest(f"Log file not found at {LOG_PATH}")
            
        print("\nSetting up Basic Allocator...")
        cls.allocator_basic = ResourceAllocator(
            log_path=LOG_PATH, 
            permission_method='basic',
            use_sample=SAMPLE_SIZE
        )
        
        print("Setting up OrdinoR Allocator...")
        cls.allocator_ordinor = ResourceAllocator(
            log_path=LOG_PATH, 
            permission_method='ordinor',
            n_trace_clusters=2,
            n_resource_clusters=5,
            use_sample=SAMPLE_SIZE
        )

    def test_initialization_basic(self):
        self.assertIsNotNone(self.allocator_basic.df)
        self.assertLessEqual(len(self.allocator_basic.df), SAMPLE_SIZE)
        self.assertIsNotNone(self.allocator_basic.permissions)
        self.assertIsNotNone(self.allocator_basic.availability)

    def test_allocation_success(self):
        """Test successful allocation during working hours."""
        # Find a valid activity from the log
        activity = self.allocator_basic.df['concept:name'].iloc[0]
        
        # Pick a Monday at 10:00 (inside working hours)
        # 2017-01-02 was a Monday
        timestamp = datetime(2017, 1, 2, 10, 0, 0)
        
        # Adjust timestamp to be within the log's interval
        log_start = self.allocator_basic.availability.start_time
        timestamp = log_start.replace(hour=10)
        # Ensure it's a weekday
        while timestamp.weekday() > 4:
            timestamp += pd.Timedelta(days=1)
            
        resource = self.allocator_basic.allocate(activity, timestamp)
        
        eligible = self.allocator_basic.permissions.get_eligible_resources(activity)
        if eligible:
            self.assertIsNotNone(resource)
            self.assertIn(resource, eligible)

    def test_allocation_failure_sunday(self):
        """Test allocation failure on Sunday (non-working day)."""
        activity = self.allocator_basic.df['concept:name'].iloc[0]
        
        # Find a Sunday within range
        log_start = self.allocator_basic.availability.start_time
        sunday = log_start
        while sunday.weekday() != 6:
            sunday += pd.Timedelta(days=1)
        
        sunday = sunday.replace(hour=10)
        
        resource = self.allocator_basic.allocate(activity, sunday)
        self.assertIsNone(resource, "Should not allocate on Sunday")

    def test_ordinor_discovery(self):
        """Test that OrdinoR model covers activities."""
        activity = self.allocator_ordinor.df['concept:name'].iloc[0]
        eligible = self.allocator_ordinor.permissions.get_eligible_resources(activity)
        
        self.assertIsInstance(eligible, list)

if __name__ == "__main__":
    unittest.main()
