import unittest
from unittest.mock import MagicMock
from datetime import datetime
from resources.resource_allocation import ResourceAllocator

class TestResourceAllocator(unittest.TestCase):

    def setUp(self):
        self.permissions = MagicMock()
        self.availability = MagicMock()
        # Initialize with mocked permissions
        self.allocator = ResourceAllocator(permissions_model=self.permissions)
        # Mock availability injection manually since we bypassed standard init
        self.allocator.availability = self.availability
        
        self.test_time = datetime(2023, 10, 27, 10, 0, 0)

    def test_allocate_success(self):
        # Setup
        self.permissions.get_eligible_resources.return_value = ["res1", "res2", "res3"]
        # Make res1 and res3 available, res2 unavailable
        self.availability.is_available.side_effect = lambda res, time: res in ["res1", "res3"]

        # Execute
        allocated_resource = self.allocator.allocate("test_activity", self.test_time)

        # Verify
        self.assertIn(allocated_resource, ["res1", "res3"])
        self.permissions.get_eligible_resources.assert_called_with("test_activity", timestamp=self.test_time, case_type=None)
        
        # Check that is_available was called for at least some candidates
        self.assertTrue(self.availability.is_available.called)

    def test_allocate_with_case_type(self):
        # Verify that case_type is passed to permissions
        self.permissions.get_eligible_resources.return_value = ["res1"]
        self.availability.is_available.return_value = True
        
        self.allocator.allocate("test_activity", self.test_time, case_type="Home improvement")
        
        self.permissions.get_eligible_resources.assert_called_with(
            "test_activity", 
            timestamp=self.test_time, 
            case_type="Home improvement"
        )

    def test_allocate_no_permission(self):
        # Setup
        self.permissions.get_eligible_resources.return_value = []
        
        # Execute
        allocated_resource = self.allocator.allocate("test_activity", self.test_time)

        # Verify
        self.assertIsNone(allocated_resource)
        self.availability.is_available.assert_not_called()

    def test_allocate_no_availability(self):
        # Setup
        self.permissions.get_eligible_resources.return_value = ["res1", "res2"]
        self.availability.is_available.return_value = False

        # Execute
        allocated_resource = self.allocator.allocate("test_activity", self.test_time)

        # Verify
        self.assertIsNone(allocated_resource)
        self.availability.is_available.assert_called()

if __name__ == '__main__':
    unittest.main()
