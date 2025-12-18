"""
BPIC17 Integration Test for Resource Allocation Module

This test validates resource allocation against the real BPIC17 event log
to understand the module's effectiveness and identify potential issues.

Key Question: How many activities go unallocated and why?
- Permission failures: Activity not in the discovered organizational model
- Availability failures: All eligible resources are outside working hours
"""
import pytest
import pandas as pd
import pm4py
import os
from collections import defaultdict
from datetime import datetime
from resources.resource_allocation import ResourceAllocator


# Path to the BPIC17 event log
EVENT_LOG_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "..", "eventlog", "eventlog.xes.gz"
)


@pytest.fixture(scope="module")
def event_log_df():
    """Load BPIC17 event log once for all tests."""
    if not os.path.exists(EVENT_LOG_PATH):
        pytest.skip(f"Event log not found: {EVENT_LOG_PATH}")
    
    log = pm4py.read_xes(EVENT_LOG_PATH)
    df = pm4py.convert_to_dataframe(log)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    
    # Filter to complete events only
    if 'lifecycle:transition' in df.columns:
        df = df[df['lifecycle:transition'] == 'complete']
    
    return df


@pytest.fixture(scope="module")
def allocator(event_log_df):
    """Initialize ResourceAllocator with OrdinoR model."""
    return ResourceAllocator(
        df=event_log_df,
        permission_method='ordinor',
        availability_config={
            'enable_pattern_mining': True,
            'min_activity_threshold': 5
        }
    )


class TestBPIC17ResourceAllocation:
    """
    Integration tests for resource allocation using BPIC17 event log.
    
    These tests replay historical events to understand allocation behavior.
    """
    
    def test_allocation_coverage_analysis(self, allocator, event_log_df):
        """
        Core test: Analyze how many activities can be allocated.
        
        This answers: Is it expected to have unallocated activities?
        
        Interpretation:
        - PERMISSION failures = Model doesn't know this activity-context combination
          → May need to enrich the organizational model or handle as "any resource" 
        - AVAILABILITY failures = Known activity but no one is working at that time
          → Simulation would need to WAIT for next available resource (queue behavior)
        """
        # Sample for performance (full log is ~1.2M events)
        sample_size = 2000
        sample_df = event_log_df.sample(n=min(len(event_log_df), sample_size), random_state=42)
        
        # Metrics
        results = {
            'total': 0,
            'allocated': 0,
            'no_permission': 0,  # Activity unknown to model
            'no_availability': 0,  # Known activity but no one available
        }
        
        # Track failure patterns
        permission_failures_by_activity = defaultdict(int)
        availability_failures_by_hour = defaultdict(int)
        
        for _, event in sample_df.iterrows():
            activity = event['concept:name']
            timestamp = event['time:timestamp']
            case_type = event.get('case:LoanGoal', None)
            
            if pd.isna(event.get('org:resource')):
                continue
                
            results['total'] += 1
            
            # Attempt allocation
            allocated = allocator.allocate(activity, timestamp, case_type)
            
            if allocated:
                results['allocated'] += 1
            else:
                # Diagnose failure reason
                eligible = allocator.permissions.get_eligible_resources(
                    activity, timestamp=timestamp, case_type=case_type
                )
                
                if not eligible:
                    results['no_permission'] += 1
                    permission_failures_by_activity[activity] += 1
                else:
                    results['no_availability'] += 1
                    availability_failures_by_hour[timestamp.hour] += 1
        
        # Calculate rates
        total = results['total']
        allocation_rate = (results['allocated'] / total * 100) if total > 0 else 0
        permission_fail_rate = (results['no_permission'] / total * 100) if total > 0 else 0
        availability_fail_rate = (results['no_availability'] / total * 100) if total > 0 else 0
        
        # Print summary report
        print("\n" + "=" * 60)
        print("BPIC17 RESOURCE ALLOCATION ANALYSIS")
        print("=" * 60)
        print(f"\nTotal events analyzed: {total}")
        print(f"\n📊 ALLOCATION RESULTS:")
        print(f"   ✅ Successfully allocated: {results['allocated']:,} ({allocation_rate:.1f}%)")
        print(f"   ❌ Permission failures:    {results['no_permission']:,} ({permission_fail_rate:.1f}%)")
        print(f"   ⏰ Availability failures:  {results['no_availability']:,} ({availability_fail_rate:.1f}%)")
        
        # Show top permission failures
        if permission_failures_by_activity:
            print(f"\n🚫 TOP PERMISSION FAILURES (activity not in org model):")
            for act, count in sorted(permission_failures_by_activity.items(), 
                                     key=lambda x: -x[1])[:5]:
                print(f"   - {act}: {count}")
        
        # Show availability failure pattern
        if availability_failures_by_hour:
            print(f"\n⏰ AVAILABILITY FAILURES BY HOUR (consider for schedule tuning):")
            peak_hours = sorted(availability_failures_by_hour.items(), key=lambda x: -x[1])[:5]
            for hour, count in peak_hours:
                print(f"   - Hour {hour:02d}:00: {count} failures")
        
        print("\n" + "=" * 60)
        print("INTERPRETATION:")
        print("-" * 60)
        if availability_fail_rate > 0:
            print("⏰ Availability failures are EXPECTED in simulation!")
            print("   → Real simulation handles this via QUEUING/WAITING")
            print("   → Activity waits until a qualified resource is available")
        if permission_fail_rate > 0:
            print("🚫 Permission failures indicate GAPS in org model.")
            print("   → Consider: fallback to 'any qualified resource'")
            print("   → Or: enrich the organizational model discovery")
        print("=" * 60 + "\n")
        
        # Assertions - these are flexible as some failures are expected
        assert allocation_rate > 0, "At least some allocations should succeed"
        
        # Store for other tests to reference
        self._results = results
        self._permission_failures = permission_failures_by_activity
    
    def test_availability_model_sanity(self, allocator):
        """Verify the availability model produces reasonable working hours."""
        # Sample resources 
        resources = list(allocator.availability.resource_patterns.keys())[:5]
        
        print("\n📅 SAMPLE RESOURCE AVAILABILITY PATTERNS:")
        for resource in resources[:3]:
            pattern = allocator.availability.resource_patterns.get(resource)
            if pattern:
                print(f"\n   {resource}:")
                # Show a sample of pattern info (varies by implementation)
                if hasattr(pattern, 'items'):
                    for k, v in list(pattern.items())[:3]:
                        print(f"      {k}: {v}")
                else:
                    print(f"      Pattern type: {type(pattern).__name__}")
        
        # Basic sanity: at least some resources should have patterns
        assert len(resources) > 0, "Should have discovered some resource patterns"
    
    def test_permission_model_coverage(self, allocator, event_log_df):
        """Check what percentage of unique activities are covered by the org model."""
        unique_activities = event_log_df['concept:name'].unique()
        
        # Check each activity
        covered = 0
        uncovered = []
        
        for activity in unique_activities:
            # Check if any resource is eligible for this activity
            sample_timestamp = event_log_df['time:timestamp'].iloc[0]
            eligible = allocator.permissions.get_eligible_resources(activity, timestamp=sample_timestamp)
            if eligible:
                covered += 1
            else:
                uncovered.append(activity)
        
        coverage_rate = (covered / len(unique_activities) * 100)
        
        print(f"\n📋 PERMISSION MODEL COVERAGE:")
        print(f"   Activities in log:     {len(unique_activities)}")
        print(f"   Covered by org model:  {covered} ({coverage_rate:.1f}%)")
        
        if uncovered:
            print(f"\n   ⚠️  Uncovered activities ({len(uncovered)}):")
            for act in uncovered[:10]:
                print(f"      - {act}")
        
        # At least 50% coverage expected for a reasonable model
        assert coverage_rate >= 50, f"Permission model covers only {coverage_rate:.1f}% of activities"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
