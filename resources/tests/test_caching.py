
import unittest
import os
import time
import shutil
from resources.resource_allocation import ResourceAllocator

LOG_PATH = "eventlog.xes.gz"
SAMPLE_SIZE = 1000
CACHE_FILE = "test_ordinor_cache.pkl"

class TestCaching(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(LOG_PATH):
            raise unittest.SkipTest(f"Log file not found at {LOG_PATH}")
        # Clean up previous cache
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)

    @classmethod
    def tearDownClass(cls):
        # Clean up cache after tests
        if os.path.exists(CACHE_FILE):
             os.remove(CACHE_FILE)

    def test_caching_speedup(self):
        """Verify that loading from cache is faster than discovery."""
        
        # Load log once to isolate discovery time
        print("Pre-loading log for fair comparison...")
        temp_alloc = ResourceAllocator(log_path=LOG_PATH, permission_method='basic', use_sample=SAMPLE_SIZE)
        df = temp_alloc.df
        
        print("\n[Run 1] Initializing without cache (Discovery)...")
        start_time = time.time()
        allocator1 = ResourceAllocator(
            df=df,
            permission_method='ordinor',
            n_trace_clusters=2,
            n_resource_clusters=5,
            cache_path=CACHE_FILE
        )
        duration_1 = time.time() - start_time
        print(f"Run 1 completed in {duration_1:.2f} seconds")
        
        self.assertTrue(os.path.exists(CACHE_FILE), "Cache file should be created")
        
        print("\n[Run 2] Initializing with cache (Load)...")
        start_time = time.time()
        allocator2 = ResourceAllocator(
            df=df,
            permission_method='ordinor',
            n_trace_clusters=2,
            n_resource_clusters=5,
            cache_path=CACHE_FILE
        )
        duration_2 = time.time() - start_time
        print(f"Run 2 completed in {duration_2:.2f} seconds")
        
        # It should be significantly faster
        # (Discovery involves KMeans and AHC, loading is just unpickling)
        self.assertLess(duration_2, duration_1, "Loading from cache should be faster")
        print(f"Speedup: {duration_1 / duration_2:.2f}x")
        
        # Verify functionality still works
        activity = allocator2.df['concept:name'].iloc[0]
        eligible = allocator2.permissions.get_eligible_resources(activity)
        self.assertIsInstance(eligible, list)
        self.assertTrue(len(eligible) > 0)

if __name__ == "__main__":
    unittest.main()
