"""
Test suite for Advanced Resource Permissions.

Tests the full organizational model mining pipeline including:
- Feature matrix construction
- Resource clustering
- Group profiling
- Advanced permission lookups
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resources.resource_features import ResourceActivityMatrix
from resources.resource_clustering import ResourceClusterer
from resources.group_profiling import GroupProfiler
from resources.organizational_model import OrganizationalModel
from resources.resource_permissions import AdvancedResourcePermissions, BasicResourcePermissions

# Enable logging for tests
logging.basicConfig(level=logging.INFO)


class TestResourceActivityMatrix(unittest.TestCase):
    """Tests for the feature matrix construction."""
    
    def setUp(self):
        self.test_df = pd.DataFrame({
            "concept:name": ["A", "A", "B", "B", "C", "A", "B", "C"],
            "org:resource": ["R1", "R2", "R1", "R2", "R3", "R1", "R3", "R3"],
            "time:timestamp": pd.date_range(start='1/1/2017', periods=8)
        })
    
    def test_matrix_shape(self):
        builder = ResourceActivityMatrix(self.test_df)
        matrix = builder.build_matrix()
        
        # 3 resources (R1, R2, R3), 3 activities (A, B, C)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], 3)
    
    def test_matrix_values(self):
        builder = ResourceActivityMatrix(self.test_df)
        matrix = builder.build_matrix()
        
        # R1 did A twice, B once, C zero times
        self.assertEqual(matrix.loc["R1", "A"], 2)
        self.assertEqual(matrix.loc["R1", "B"], 1)
        self.assertEqual(matrix.loc["R1", "C"], 0)
        
        # R3 did C twice, B once, A zero times
        self.assertEqual(matrix.loc["R3", "C"], 2)
        self.assertEqual(matrix.loc["R3", "B"], 1)
        self.assertEqual(matrix.loc["R3", "A"], 0)
    
    def test_missing_column_error(self):
        bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        builder = ResourceActivityMatrix(bad_df)
        with self.assertRaises(ValueError):
            builder.build_matrix()
    
    def test_resource_ids_list(self):
        builder = ResourceActivityMatrix(self.test_df)
        builder.build_matrix()
        self.assertEqual(set(builder.get_resource_ids()), {"R1", "R2", "R3"})


class TestResourceClusterer(unittest.TestCase):
    """Tests for resource clustering."""
    
    def test_clustering_basic(self):
        # Simple 4x3 matrix: 4 resources, 3 activities
        matrix = np.array([
            [10, 0, 0],   # R0: specializes in activity 0
            [8, 1, 0],    # R1: similar to R0
            [0, 0, 10],   # R2: specializes in activity 2
            [0, 1, 9],    # R3: similar to R2
        ])
        resource_ids = ["R0", "R1", "R2", "R3"]
        
        clusterer = ResourceClusterer(n_clusters=2)
        mapping = clusterer.cluster(matrix, resource_ids)
        
        # R0 and R1 should be in the same group
        self.assertEqual(mapping["R0"], mapping["R1"])
        # R2 and R3 should be in the same group
        self.assertEqual(mapping["R2"], mapping["R3"])
        # Groups should be different
        self.assertNotEqual(mapping["R0"], mapping["R2"])
    
    def test_clustering_too_few_resources(self):
        matrix = np.array([[1, 2], [3, 4]])
        resource_ids = ["R0", "R1"]
        
        # Request 5 clusters but only 2 resources
        clusterer = ResourceClusterer(n_clusters=5)
        mapping = clusterer.cluster(matrix, resource_ids)
        
        # Should still work with reduced clusters
        self.assertEqual(len(mapping), 2)
    
    def test_get_group_members(self):
        mapping = {"R0": 0, "R1": 0, "R2": 1, "R3": 1}
        clusterer = ResourceClusterer()
        members = clusterer.get_group_members(mapping)
        
        self.assertEqual(set(members[0]), {"R0", "R1"})
        self.assertEqual(set(members[1]), {"R2", "R3"})


class TestGroupProfiler(unittest.TestCase):
    """Tests for group capability profiling."""
    
    def setUp(self):
        # Matrix: 3 resources, 3 activities
        self.matrix = pd.DataFrame({
            "A": [10, 8, 0],
            "B": [2, 1, 10],
            "C": [0, 0, 5]
        }, index=["R0", "R1", "R2"])
        
        # R0, R1 in group 0; R2 in group 1
        self.resource_to_group = {"R0": 0, "R1": 0, "R2": 1}
    
    def test_group_capabilities(self):
        profiler = GroupProfiler(min_frequency=5, min_coverage=0.5)
        capabilities = profiler.profile_groups(self.matrix, self.resource_to_group)
        
        # Group 0: A (high freq, both members), B (low coverage)
        self.assertIn("A", capabilities[0])
        # B has freq=3 which is < 5, so not a capability
        self.assertNotIn("B", capabilities[0])
        
        # Group 1: B and C (R2 does both)
        self.assertIn("B", capabilities[1])
        self.assertIn("C", capabilities[1])
    
    def test_min_frequency_threshold(self):
        profiler = GroupProfiler(min_frequency=15, min_coverage=0.0)
        capabilities = profiler.profile_groups(self.matrix, self.resource_to_group)
        
        # Group 0: A has 18 total, meets threshold
        self.assertIn("A", capabilities[0])
        # Group 1: B has 10, C has 5, neither meets 15
        self.assertEqual(len(capabilities[1]), 0)
    
    def test_min_coverage_threshold(self):
        profiler = GroupProfiler(min_frequency=1, min_coverage=1.0)
        capabilities = profiler.profile_groups(self.matrix, self.resource_to_group)
        
        # Group 0: A done by both (100%), B done by both (100%)
        self.assertIn("A", capabilities[0])
        self.assertIn("B", capabilities[0])
        
        # Group 1: All activities done by R2 (100% coverage since only 1 member)
        self.assertIn("B", capabilities[1])
        self.assertIn("C", capabilities[1])


class TestOrganizationalModel(unittest.TestCase):
    """Tests for organizational model data structure."""
    
    def setUp(self):
        self.model = OrganizationalModel(
            resource_groups={0: {"R0", "R1"}, 1: {"R2", "R3"}},
            group_capabilities={0: {"A", "B"}, 1: {"C", "D"}},
            resource_to_group={"R0": 0, "R1": 0, "R2": 1, "R3": 1}
        )
    
    def test_get_groups_for_activity(self):
        self.assertEqual(self.model.get_groups_for_activity("A"), [0])
        self.assertEqual(self.model.get_groups_for_activity("C"), [1])
        self.assertEqual(self.model.get_groups_for_activity("Unknown"), [])
    
    def test_get_members_of_groups(self):
        members = self.model.get_members_of_groups([0, 1])
        self.assertEqual(members, {"R0", "R1", "R2", "R3"})
        
        members = self.model.get_members_of_groups([0])
        self.assertEqual(members, {"R0", "R1"})
    
    def test_save_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            self.model.save(path)
            loaded = OrganizationalModel.load(path)
            
            self.assertEqual(loaded.resource_groups, self.model.resource_groups)
            self.assertEqual(loaded.group_capabilities, self.model.group_capabilities)
            self.assertEqual(loaded.resource_to_group, self.model.resource_to_group)
        finally:
            os.unlink(path)
    
    def test_coverage_stats(self):
        all_activities = {"A", "B", "C", "D", "E"}
        stats = self.model.get_coverage_stats(all_activities)
        
        self.assertEqual(stats["total_activities"], 5)
        self.assertEqual(stats["covered_activities"], 4)  # A, B, C, D
        self.assertAlmostEqual(stats["coverage_ratio"], 0.8)
        self.assertEqual(stats["uncovered_activities"], ["E"])


class TestAdvancedResourcePermissions(unittest.TestCase):
    """Tests for the advanced permissions system."""
    
    def setUp(self):
        # Create a test DataFrame that will cluster nicely
        data = []
        # Group 1: Resources that do A and B
        for r in ["R1", "R2", "R3"]:
            for _ in range(10):
                data.append({"concept:name": "A", "org:resource": r})
            for _ in range(8):
                data.append({"concept:name": "B", "org:resource": r})
        
        # Group 2: Resources that do C and D
        for r in ["R4", "R5", "R6"]:
            for _ in range(10):
                data.append({"concept:name": "C", "org:resource": r})
            for _ in range(8):
                data.append({"concept:name": "D", "org:resource": r})
        
        self.test_df = pd.DataFrame(data)
        self.test_df["time:timestamp"] = pd.date_range(start='1/1/2017', periods=len(data))
    
    def test_discover_model(self):
        perms = AdvancedResourcePermissions(df=self.test_df)
        model = perms.discover_model(n_clusters=2, min_frequency=5, min_coverage=0.5)
        
        # Should have 2 groups
        self.assertEqual(len(model.resource_groups), 2)
        # Each group should have 3 members
        for group_id, members in model.resource_groups.items():
            self.assertEqual(len(members), 3)
    
    def test_get_eligible_resources(self):
        perms = AdvancedResourcePermissions(df=self.test_df)
        perms.discover_model(n_clusters=2, min_frequency=5, min_coverage=0.5)
        
        # Activity A should return group 1 members
        eligible_a = set(perms.get_eligible_resources("A"))
        self.assertTrue({"R1", "R2", "R3"}.issubset(eligible_a) or 
                       {"R4", "R5", "R6"}.issubset(eligible_a))
        
        # Unknown activity should return empty
        eligible_unknown = perms.get_eligible_resources("Unknown")
        self.assertEqual(eligible_unknown, [])
    
    def test_coverage_stats(self):
        perms = AdvancedResourcePermissions(df=self.test_df)
        perms.discover_model(n_clusters=2, min_frequency=5, min_coverage=0.5)
        
        stats = perms.get_coverage_stats()
        # All 4 activities should be covered
        self.assertEqual(stats["coverage_ratio"], 1.0)


class TestAdvancedPermissionsRealLog(unittest.TestCase):
    """Integration tests against the actual BPIC2017 event log."""
    
    rp_advanced = None
    rp_basic = None

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        load_dotenv()
        
        log_path = os.getenv("EVENT_LOG_PATH")
        
        if not log_path or not os.path.exists(log_path):
            raise unittest.SkipTest(f"Event log not found at {log_path} (set EVENT_LOG_PATH in .env)")
        
        print(f"\n[RealLog] Loading log from {log_path} for advanced permissions test...")
        
        # Initialize both systems
        cls.rp_basic = BasicResourcePermissions(log_path=log_path)
        cls.rp_advanced = AdvancedResourcePermissions(log_path=log_path)
        cls.rp_advanced.discover_model(n_clusters=5, min_frequency=5, min_coverage=0.3)
    
    def test_groups_non_empty(self):
        """All discovered groups should have members."""
        model = self.rp_advanced.model
        for group_id, members in model.resource_groups.items():
            self.assertGreater(len(members), 0, f"Group {group_id} is empty")
    
    def test_groups_reasonable_size(self):
        """No group should have more than 80% of all resources."""
        model = self.rp_advanced.model
        total_resources = len(model.resource_to_group)
        
        for group_id, members in model.resource_groups.items():
            ratio = len(members) / total_resources
            self.assertLess(ratio, 0.8, f"Group {group_id} has {ratio:.0%} of all resources")
    
    def test_coverage_threshold(self):
        """At least 90% of activities should have eligible resources."""
        stats = self.rp_advanced.get_coverage_stats()
        self.assertGreaterEqual(stats["coverage_ratio"], 0.9,
                               f"Coverage is only {stats['coverage_ratio']:.1%}")
    
    def test_compare_with_basic(self):
        """Compare advanced vs basic approach."""
        test_activities = ["A_Create Application", "W_Validate application"]
        
        for activity in test_activities:
            basic = set(self.rp_basic.get_eligible_resources(activity))
            advanced = set(self.rp_advanced.get_eligible_resources(activity))
            
            print(f"\n{activity}:")
            print(f"  Basic: {len(basic)} resources")
            print(f"  Advanced: {len(advanced)} resources")
            
            # Advanced should return resources (may be more or fewer than basic)
            # but shouldn't be empty if basic has resources
            if basic:
                self.assertGreater(len(advanced), 0,
                                  f"Advanced returned 0 resources for {activity}")


if __name__ == '__main__':
    unittest.main()
