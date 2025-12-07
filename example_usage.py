"""
Example usage of Resource Availability Models
Demonstrates both basic and advanced models with the BPIC 2017 dataset
"""

import sys
sys.path.append('pm4py-release')

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe
from resources.resource_availabilities import (
    ResourceAvailabilityModel,
    AdvancedResourceAvailabilityModel
)
from datetime import datetime
import pandas as pd


def example_basic_model():
    """Example: Using the basic resource availability model."""
    
    print("=" * 80)
    print("EXAMPLE 1: Basic Resource Availability Model")
    print("=" * 80)
    
    # Load dataset
    log = xes_importer.apply('Dataset/BPI Challenge 2017.xes')
    df = convert_to_dataframe(log)
    
    # Initialize basic model
    model = ResourceAvailabilityModel(
        event_log_df=df,
        interval_days=14,          # 2-week cycle
        workday_start_hour=8,      # Start at 8 AM
        workday_end_hour=17,       # End at 5 PM
        working_cycle_days={0, 1, 2, 3, 4, 7, 8, 9, 10, 11}  # Mon-Fri both weeks
    )
    
    # Example usage
    resource = "User_3"
    test_time = datetime(2016, 6, 15, 10, 0)  # Wednesday 10 AM
    
    print(f"\nChecking availability for {resource}")
    print(f"Time: {test_time}")
    print(f"Available: {model.is_available(resource, test_time)}")
    
    # Find next available time from a weekend
    weekend_time = datetime(2016, 6, 18, 14, 0)  # Saturday 2 PM
    next_available = model.get_next_available_time(resource, weekend_time)
    print(f"\nFrom {weekend_time} (Saturday)")
    print(f"Next available: {next_available}")


def example_advanced_model():
    """Example: Using the advanced model with pattern mining."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Advanced Resource Availability Model")
    print("=" * 80)
    
    # Load dataset
    log = xes_importer.apply('Dataset/BPI Challenge 2017.xes')
    df = convert_to_dataframe(log)
    
    # Initialize advanced model with pattern mining
    model = AdvancedResourceAvailabilityModel(
        event_log_df=df,
        enable_pattern_mining=True,
        min_activity_threshold=10
    )
    
    # Get most active resources
    top_resources = df['org:resource'].value_counts().head(3).index.tolist()
    
    for resource in top_resources:
        print(f"\n{'-' * 80}")
        print(f"Resource: {resource}")
        print(f"{'-' * 80}")
        
        # Get resource pattern information
        info = model.get_resource_info(resource)
        if info:
            print(f"Working hours: {info['working_start']}:00 - {info['working_end']}:00")
            print(f"Working days: {sorted(info['working_days'])}")
            print(f"Peak hours: {sorted(list(info['peak_hours']))[:5]}...")
            print(f"Total activities: {info['total_activities']:,}")
            print(f"Activity intensity: {info['activity_intensity']:.2f} activities/day")
            
            if 'cluster_id' in info:
                print(f"Cluster: {info['cluster_id']}")
        
        # Get workload statistics
        workload = model.get_resource_workload(resource)
        print(f"\nWorkload: {workload['avg_activities_per_day']:.1f} activities/day")
        
        # Test availability prediction
        test_time = datetime(2016, 6, 15, 9, 0)
        is_avail = model.is_available(resource, test_time)
        prob = model.predict_availability_probability(resource, test_time)
        
        print(f"\nAt {test_time}:")
        print(f"  Available: {is_avail}")
        print(f"  Probability: {prob:.1%}")


def example_simulation_integration():
    """Example: Using models in a simulation context."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Simulation Integration")
    print("=" * 80)
    
    # Load dataset
    log = xes_importer.apply('Dataset/BPI Challenge 2017.xes')
    df = convert_to_dataframe(log)
    
    # Use advanced model
    model = AdvancedResourceAvailabilityModel(
        event_log_df=df,
        enable_pattern_mining=True
    )
    
    # Simulate resource selection at different times
    test_times = [
        datetime(2016, 6, 15, 8, 0),   # Wednesday 8 AM
        datetime(2016, 6, 15, 14, 0),  # Wednesday 2 PM
        datetime(2016, 6, 15, 18, 0),  # Wednesday 6 PM
        datetime(2016, 6, 18, 10, 0),  # Saturday 10 AM
    ]
    
    print("\nSimulating resource availability across different times:\n")
    
    for test_time in test_times:
        available = model.get_available_resources(test_time)
        print(f"{test_time.strftime('%A %I:%M %p')}:")
        print(f"  Available resources: {len(available)}/{len(model.resources)} ({len(available)/len(model.resources)*100:.1f}%)")
        
        # Show probability distribution for a sample resource
        sample_resource = model.resources[10] if len(model.resources) > 10 else model.resources[0]
        prob = model.predict_availability_probability(sample_resource, test_time)
        print(f"  Sample ({sample_resource}) probability: {prob:.1%}\n")


def example_cluster_analysis():
    """Example: Analyzing resource clusters."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Resource Cluster Analysis")
    print("=" * 80)
    
    # Load dataset
    log = xes_importer.apply('Dataset/BPI Challenge 2017.xes')
    df = convert_to_dataframe(log)
    
    # Initialize model
    model = AdvancedResourceAvailabilityModel(
        event_log_df=df,
        enable_pattern_mining=True
    )
    
    print(f"\nFound {len(model.cluster_profiles)} resource clusters:")
    
    for cluster_id, profile in model.cluster_profiles.items():
        print(f"\n{'=' * 60}")
        print(f"Cluster {cluster_id}: {profile['num_resources']} resources")
        print(f"{'=' * 60}")
        print(f"Average working hours: {profile['avg_working_start']:.1f}:00 - {profile['avg_working_end']:.1f}:00")
        print(f"Common working days: {sorted(profile['common_working_days'])}")
        print(f"Peak hours: {sorted(list(profile['common_peak_hours']))[:8]}...")
        print(f"Average intensity: {profile['avg_activity_intensity']:.2f} activities/day")
        
        # Find resources in this cluster
        cluster_resources = [r for r, c in model.resource_clusters.items() if c == cluster_id]
        print(f"Example resources: {cluster_resources[:5]}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RESOURCE AVAILABILITY MODELS - USAGE EXAMPLES")
    print("=" * 80)
    
    try:
        example_basic_model()
        example_advanced_model()
        example_simulation_integration()
        example_cluster_analysis()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
