import pandas as pd
import sys
sys.path.append('pm4py-release')

from datetime import datetime, time, timedelta
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe

from resources.resource_availabilities import (
    ResourceAvailabilityModel, 
    AdvancedResourceAvailabilityModel
)

LOG_FILE = r"Dataset\BPI Challenge 2017.xes"
print("=" * 80)
print("RESOURCE AVAILABILITY MODEL TESTS")
print("=" * 80)


def print_check(model, resource_id: str, label: str, dt: datetime):
    """Helper function to print availability check results."""
    cycle_day = model._cycle_day_index(dt)
    weekday = dt.weekday()  # 0=Mon,...,6=Sun
    available = model.is_available(resource_id, dt)
    print(
        f"[{label}]\n"
        f"  Time: {dt} | Weekday: {weekday} | Cycle Day: {cycle_day}\n"
        f"  Available: {available}"
    )


def test_basic_model(df):
    """Test the basic resource availability model."""
    print("\n" + "=" * 80)
    print("TESTING BASIC MODEL")
    print("=" * 80)
    
    model = ResourceAvailabilityModel(df, interval_days=14)
    
    print(f"\nTotal resources: {len(model.resources)}")
    example_resource = model.resources[0] if model.resources else None
    print(f"Example resource: {example_resource}")
    
    if example_resource is None:
        print("No resources found, skipping basic model tests.")
        return
    
    print(f"Cycle start date (Monday anchor): {model.cycle_start_date}")
    print(f"Working hours: {model.workday_start_hour}:00 - {model.workday_end_hour}:00")
    print(f"Working cycle days: {sorted(model.working_cycle_days)}")
    
    # Base datetime = cycle_start_date 00:00
    base_dt = datetime.combine(model.cycle_start_date, time(0, 0))
    
    print("\n" + "-" * 80)
    print("Test Cases:")
    print("-" * 80)
    
    # 1) Weekday in working hours (should be True)
    weekday_10 = base_dt.replace(hour=10)  # Monday 10:00
    print_check(model, example_resource, "Monday 10:00 (Expected: True)", weekday_10)

    # 2) Weekday before working hours (should be False)
    weekday_07 = base_dt.replace(hour=7)
    print_check(model, example_resource, "Weekday 07:00 (expected False)", weekday_07)

    # 3) Weekday at boundary 16:00 (still inside, expected True)
    weekday_16 = base_dt.replace(hour=16)
    print_check(model, example_resource, "Weekday 16:00 (expected True)", weekday_16)

    # 4) Weekday after working hours (18:00, expected False)
    weekday_18 = base_dt.replace(hour=18)
    print_check(model, example_resource, "Weekday 18:00 (expected False)", weekday_18)

    # 5) Saturday 10:00 (expected False)
    saturday_dt = base_dt
    while saturday_dt.weekday() != 5:  # 5 = Saturday
        saturday_dt += timedelta(days=1)
    saturday_10 = saturday_dt.replace(hour=10)
    print_check(model, example_resource, "Saturday 10:00 (expected False)", saturday_10)

    # 6) Saturday 16:00 (expected False)
    saturday_16 = saturday_dt.replace(hour=16)
    print_check(model, example_resource, "Saturday 16:00 (expected False)", saturday_16)

    # 7) Sunday 10:00 (expected False)
    sunday_dt = base_dt
    while sunday_dt.weekday() != 6:  # 6 = Sunday
        sunday_dt += timedelta(days=1)
    sunday_10 = sunday_dt.replace(hour=10)
    print_check(model, example_resource, "Sunday 10:00 (expected False)", sunday_10)

    # 8) Sunday 07:00 (expected False)
    sunday_07 = sunday_dt.replace(hour=7)
    print_check(model, example_resource, "Sunday 07:00 (expected False)", sunday_07)

    # 9) Far in the future: some Monday in 2026 at 10:00 (should still follow pattern -> True)
    future_monday = datetime(2026, 5, 18, 10, 0)  # 18.05.2026 ist ein Montag
    print_check(model, example_resource, "Future Monday 2026-05-18 10:00 (expected True)", future_monday)

    # 10) Future Saturday in 2026 at 10:00 (should be False)
    future_saturday = datetime(2026, 5, 16, 10, 0)  # 16.05.2026 ist ein Samstag
    print_check(model, example_resource, "Future Saturday 2026-05-16 10:00 (expected False)", future_saturday)

    # 11) Future Sunday in 2026 at 16:00 (should be False)
    future_sunday = datetime(2026, 5, 17, 16, 0)  # 17.05.2026 ist ein Sonntag
    print_check(model, example_resource, "Future Sunday 2026-05-17 16:00 (expected False)", future_sunday)

    # 12) Future weekday at 07:00 (should be False)
    future_tuesday_07 = datetime(2026, 5, 19, 7, 0)  # 19.05.2026 ist ein Dienstag
    print_check(model, example_resource, "Future Tuesday 2026-05-19 07:00 (expected False)", future_tuesday_07)
    
    # 13) Dutch holiday test
    kings_day_2016 = datetime(2016, 4, 27, 10, 0)
    print_check(model, example_resource, "King's Day 2016 (expected False)", kings_day_2016)
    
    # Test next available time
    print("\n" + "-" * 80)
    print("Next Available Time Tests:")
    print("-" * 80)
    
    # From a Saturday, find next available time
    saturday_dt = base_dt
    while saturday_dt.weekday() != 5:
        saturday_dt += timedelta(days=1)
    saturday_noon = saturday_dt.replace(hour=12)
    
    next_time = model.get_next_available_time(example_resource, saturday_noon)
    print(f"From {saturday_noon} -> Next available: {next_time}")


def test_advanced_model(df):
    """Test the advanced resource availability model with pattern mining."""
    print("\n" + "=" * 80)
    print("TESTING ADVANCED MODEL WITH RESOURCE MINING")
    print("=" * 80)
    
    model = AdvancedResourceAvailabilityModel(
        df, 
        interval_days=14,
        enable_pattern_mining=True,
        min_activity_threshold=10
    )
    
    print(f"\nTotal resources: {len(model.resources)}")
    
    if not model.resources:
        print("No resources found, skipping advanced model tests.")
        return
    
    # Test with top 5 most active resources
    df_temp = df.copy()
    top_resources = df_temp['org:resource'].value_counts().head(5).index.tolist()
    
    print(f"\nTesting with top 5 resources: {top_resources[:5]}")
    
    for i, resource in enumerate(top_resources[:3], 1):  # Test first 3
        print(f"\n{'-' * 80}")
        print(f"Resource {i}: {resource}")
        print(f"{'-' * 80}")
        
        # Get resource pattern info
        info = model.get_resource_info(resource)
        if info:
            print(f"Total activities: {info['total_activities']}")
            print(f"Working hours: {info['working_start']}:00 - {info['working_end']}:00")
            print(f"Working days: {sorted(info['working_days'])}")
            print(f"Peak hours: {sorted(info['peak_hours'])}")
            print(f"Activity intensity: {info['activity_intensity']:.2f} activities/day")
            print(f"Active from: {info['first_activity']} to {info['last_activity']}")
            
            if 'cluster_id' in info:
                print(f"Cluster ID: {info['cluster_id']}")
        
        # Get workload statistics
        workload = model.get_resource_workload(resource)
        if workload:
            print(f"\nWorkload Statistics:")
            print(f"  Average activities per day: {workload.get('avg_activities_per_day', 0):.2f}")
            print(f"  Working hours per day: {workload.get('working_hours_per_day', 0)}")
            print(f"  Working days per week: {workload.get('working_days_per_week', 0)}")
        
        # Test availability at different times
        print(f"\nAvailability Tests:")
        
        # Test during peak hour
        test_date = datetime(2016, 6, 15, 9, 0)  # Mid-dataset Wednesday 9 AM
        avail = model.is_available(resource, test_date)
        prob = model.predict_availability_probability(resource, test_date)
        print(f"  Wednesday 9:00 AM: Available={avail}, Probability={prob:.2%}")
        
        # Test during off-peak hour
        test_date = datetime(2016, 6, 15, 6, 0)  # Wednesday 6 AM
        avail = model.is_available(resource, test_date)
        prob = model.predict_availability_probability(resource, test_date)
        print(f"  Wednesday 6:00 AM: Available={avail}, Probability={prob:.2%}")
        
        # Test weekend
        test_date = datetime(2016, 6, 18, 10, 0)  # Saturday 10 AM
        avail = model.is_available(resource, test_date)
        prob = model.predict_availability_probability(resource, test_date)
        print(f"  Saturday 10:00 AM: Available={avail}, Probability={prob:.2%}")
    
    # Test cluster information
    print(f"\n{'=' * 80}")
    print("Cluster Analysis")
    print(f"{'=' * 80}")
    
    print(f"\nTotal clusters: {len(model.cluster_profiles)}")
    
    for cluster_id, profile in model.cluster_profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Number of resources: {profile.get('num_resources', 0)}")
        print(f"  Avg working hours: {profile.get('avg_working_start', 0):.1f} - {profile.get('avg_working_end', 0):.1f}")
        print(f"  Common working days: {sorted(profile.get('common_working_days', []))}")
        print(f"  Common peak hours: {sorted(profile.get('common_peak_hours', []))}")
        print(f"  Avg activity intensity: {profile.get('avg_activity_intensity', 0):.2f}")
    
    # Test getting available resources at specific time
    print(f"\n{'=' * 80}")
    print("Available Resources Query")
    print(f"{'=' * 80}")
    
    test_time = datetime(2016, 6, 15, 10, 0)  # Wednesday 10 AM
    available = model.get_available_resources(test_time)
    print(f"\nAt {test_time}:")
    print(f"  Total available resources: {len(available)}")
    print(f"  Sample (first 10): {available[:10]}")


def test_lifecycle_tracking(df):
    """Test lifecycle-aware availability."""
    print("\n" + "=" * 80)
    print("TESTING LIFECYCLE-AWARE AVAILABILITY")
    print("=" * 80)
    
    model = AdvancedResourceAvailabilityModel(
        df, 
        enable_pattern_mining=True,
        enable_lifecycle_tracking=True
    )
    
    if not model.enable_lifecycle_tracking:
        print("[WARNING] Lifecycle tracking not enabled")
        return
    
    if not model.resource_busy_periods:
        print("[WARNING] No busy periods extracted (no lifecycle data)")
        return
    
    print(f"\n[INFO] Resources with busy period data: {len(model.resource_busy_periods)}")
    
    # Pick resources with busy periods
    resources_with_periods = [r for r in model.resources if r in model.resource_busy_periods and len(model.resource_busy_periods[r]) > 0]
    
    if not resources_with_periods:
        print("[WARNING] No resources with busy periods found")
        return
    
    # Test with top 3 resources that have busy periods
    test_resources = resources_with_periods[:3]
    
    for i, resource in enumerate(test_resources, 1):
        busy_periods = model.resource_busy_periods[resource]
        
        print(f"\n{'-' * 80}")
        print(f"Resource {i}: {resource}")
        print(f"{'-' * 80}")
        print(f"Total busy periods: {len(busy_periods)}")
        
        # Show first few busy periods
        print(f"\nFirst 3 busy periods:")
        for j, (start, end, activity) in enumerate(busy_periods[:3]):
            duration = (end - start).total_seconds() / 3600
            print(f"  {j+1}. {activity[:40]:40s}: {start} to {end} ({duration:.2f}h)")
        
        # Get busy period stats
        stats = model.get_busy_period_stats(resource)
        print(f"\nBusy Period Statistics:")
        print(f"  Total periods: {stats['total_busy_periods']}")
        print(f"  Avg duration: {stats['avg_duration_hours']:.2f}h")
        print(f"  Min duration: {stats['min_duration_hours']:.2f}h")
        print(f"  Max duration: {stats['max_duration_hours']:.2f}h")
        print(f"  Total busy time: {stats['total_busy_hours']:.2f}h")
        
        # Test availability during busy period
        if busy_periods:
            test_time = busy_periods[0][0] + timedelta(minutes=30)  # 30 min into first busy period
            
            print(f"\n[TEST] Testing availability at {test_time}:")
            is_busy = model.is_resource_busy_at(resource, test_time)
            current_activity = model.get_current_activity(resource, test_time)
            is_available = model.is_available(resource, test_time)
            probability = model.predict_availability_probability(resource, test_time)
            
            print(f"  Time: {test_time}")
            print(f"  Is busy: {is_busy}")
            print(f"  Current activity: {current_activity}")
            print(f"  Available: {is_available}")
            print(f"  Probability: {probability:.2%}")
            
            # Test workload
            workload = model.get_resource_workload_at(resource, test_time, window_hours=2)
            print(f"  Workload (±1h window): {workload} overlapping activities")
            
            # Test after busy period ends
            test_time_after = busy_periods[0][1] + timedelta(minutes=10)
            is_busy_after = model.is_resource_busy_at(resource, test_time_after)
            is_available_after = model.is_available(resource, test_time_after)
            
            print(f"\n[TEST] Testing 10 min after busy period ends:")
            print(f"  Time: {test_time_after}")
            print(f"  Is busy: {is_busy_after}")
            print(f"  Available: {is_available_after}")


def main():
    """Main test function."""
    print("\nLoading XES log...")
    log = xes_importer.apply(LOG_FILE)
    df = convert_to_dataframe(log)
    
    print(f"Number of events: {len(df)}")
    print(f"Number of cases: {df['case:concept:name'].nunique()}")
    print(f"Number of resources: {df['org:resource'].nunique()}")
    
    assert "org:resource" in df.columns, "Missing column: org:resource"
    assert "time:timestamp" in df.columns, "Missing column: time:timestamp"
    
    # Run tests
    test_basic_model(df)
    test_advanced_model(df)
    test_lifecycle_tracking(df)
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
