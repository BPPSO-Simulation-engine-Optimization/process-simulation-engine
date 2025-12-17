"""Debug the specific NO AVAIL case for User_1."""
import pandas as pd
import pm4py
from datetime import datetime
from resources.resource_allocation import ResourceAllocator

# Load log
log = pm4py.read_xes("eventlog.xes.gz")
df = pm4py.convert_to_dataframe(log)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

# Find the problematic event
problem_ts = pd.Timestamp('2016-03-28 06:00:32.008', tz='UTC')
event = df[df['time:timestamp'] == problem_ts].iloc[0]

print(f"=== DEBUGGING NO AVAIL CASE ===")
print(f"Timestamp: {event['time:timestamp']}")
print(f"Activity: {event['concept:name']}")
print(f"Actual Resource: {event['org:resource']}")

# Initialize allocator
allocator = ResourceAllocator(
    log_path="eventlog.xes.gz",
    permission_method='ordinor',
    availability_config={
        'enable_pattern_mining': True,
        'min_activity_threshold': 5
    }
)

# Check permissions
activity = event['concept:name']
timestamp = event['time:timestamp']
eligible = allocator.permissions.get_eligible_resources(activity, timestamp, None)
print(f"\nEligible resources: {len(eligible)}")
print(f"User_1 in eligible: {'User_1' in eligible}")

# Check availability
is_avail = allocator.availability.is_available('User_1', timestamp)
print(f"\nUser_1 available at {timestamp}: {is_avail}")

if not is_avail:
    # Check why not available
    pattern = allocator.availability.resource_patterns.get('User_1', {})
    print(f"\nUser_1 pattern:")
    print(f"  working_days: {pattern.get('working_days', 'N/A')}")
    print(f"  working_start: {pattern.get('working_start', 'N/A')}")
    print(f"  working_end: {pattern.get('working_end', 'N/A')}")
    print(f"  first_activity: {pattern.get('first_activity', 'N/A')}")
    print(f"  last_activity: {pattern.get('last_activity', 'N/A')}")
    
    # Check time range
    if 'first_activity' in pattern and 'last_activity' in pattern:
        in_range = pattern['first_activity'] <= timestamp <= pattern['last_activity']
        print(f"  In time range: {in_range}")
        
        if not in_range:
            print(f"\n✗ PROBLEM: Event timestamp {timestamp} is OUTSIDE User_1's time range!")
            print(f"  First activity: {pattern['first_activity']}")
            print(f"  Last activity: {pattern['last_activity']}")
            print(f"  Event time: {timestamp}")
            
            # Check if it's before or after
            if timestamp < pattern['first_activity']:
                print(f"  → Event is {pattern['first_activity'] - timestamp} BEFORE first activity")
            elif timestamp > pattern['last_activity']:
                print(f"  → Event is {timestamp - pattern['last_activity']} AFTER last activity")

# Test allocation
allocated = allocator.allocate(activity, timestamp, None, actual_resource='User_1')
print(f"\nAllocated resource: {allocated}")
print(f"Result: {'✓ SUCCESS' if allocated == 'User_1' else '✗ FAILED'}")
