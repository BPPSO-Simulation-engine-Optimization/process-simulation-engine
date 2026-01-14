"""Debug script to trace why User_1 is never allocated."""
import pandas as pd
import pm4py
import logging
from datetime import datetime
from resources.resource_allocation import ResourceAllocator

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s: %(message)s')

# Load the log
log_path = "eventlog.xes.gz"
log = pm4py.read_xes(log_path)
df = pm4py.convert_to_dataframe(log)

# Find a User_1 event
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
user1_events = df[df['org:resource'] == 'User_1'].head(1)

if user1_events.empty:
    print("ERROR: No User_1 events found in log!")
    exit(1)

event = user1_events.iloc[0]
print("\n=== DEBUGGING USER_1 ALLOCATION ===")
print(f"Activity: {event['concept:name']}")
print(f"Timestamp: {event['time:timestamp']}")
print(f"Actual Resource: {event['org:resource']}")
print(f"Case Type: {event.get('case:LoanGoal', 'Unknown')}")

# Initialize allocator
print("\n[1] Initializing ResourceAllocator...")
allocator = ResourceAllocator(
    log_path=log_path,
    permission_method='ordinor',
    availability_config={
        'enable_pattern_mining': True,
        'min_activity_threshold': 5
    }
)

# Test permissions
print("\n[2] Checking Permissions...")
activity = event['concept:name']
timestamp = event['time:timestamp']
case_type = event.get('case:LoanGoal', 'Unknown')

eligible_resources = allocator.permissions.get_eligible_resources(activity, timestamp, case_type)
print(f"Eligible resources for '{activity}': {len(eligible_resources)} resources")
print(f"User_1 in eligible list: {'User_1' in eligible_resources}")
if 'User_1' in eligible_resources:
    print("✓ User_1 IS eligible (permissions OK)")
else:
    print("✗ User_1 NOT eligible (PERMISSIONS PROBLEM)")
    print(f"Eligible list preview: {list(eligible_resources)[:10]}")

# Test availability
print("\n[3] Checking Availability...")
is_avail = allocator.availability.is_available('User_1', timestamp)
print(f"User_1 available at {timestamp}: {is_avail}")

if not is_avail:
    print("✗ User_1 NOT available (AVAILABILITY PROBLEM)")
    
    # Debug why not available
    if 'User_1' not in allocator.availability.resources:
        print("  - User_1 not in availability.resources list!")
    elif 'User_1' not in allocator.availability.resource_patterns:
        print("  - User_1 not in resource_patterns!")
    else:
        pattern = allocator.availability.resource_patterns['User_1']
        print(f"  - User_1 pattern:")
        print(f"    - working_days: {pattern.get('working_days', 'N/A')}")
        print(f"    - working_start: {pattern.get('working_start', 'N/A')}")
        print(f"    - working_end: {pattern.get('working_end', 'N/A')}")
        print(f"    - first_activity: {pattern.get('first_activity', 'N/A')}")
        print(f"    - last_activity: {pattern.get('last_activity', 'N/A')}")
        
        # Check time constraints
        ts = pd.Timestamp(timestamp)
        if pattern.get('first_activity') and pattern.get('last_activity'):
            print(f"  - Time range check:")
            print(f"    - First activity: {pattern['first_activity']}")
            print(f"    - Last activity: {pattern['last_activity']}")
            print(f"    - Event time: {ts}")
            in_range = pattern['first_activity'] <= ts <= pattern['last_activity']
            print(f"    - In range: {in_range}")
            if not in_range:
                print("    ✗ PROBLEM: Event outside User_1's time range!")
else:
    print("✓ User_1 IS available (availability OK)")

# Test full allocation WITHOUT actual_resource (old behavior)
print("\n[4] Testing Full Allocation (WITHOUT actual_resource)...")
allocated = allocator.allocate(activity, timestamp, case_type)
print(f"Allocated resource: {allocated}")

if allocated == 'User_1':
    print("✓ SUCCESS! User_1 was allocated")
elif allocated is None:
    print("✗ FAIL! No resource allocated")
else:
    print(f"✗ FAIL! Different resource allocated: {allocated}")
    
    # Check if allocated resource is in eligible list
    if allocated in eligible_resources:
        print(f"  - {allocated} IS in eligible list")
    else:
        print(f"  - {allocated} NOT in eligible list (SHOULD NOT HAPPEN)")

# Test full allocation WITH actual_resource (new behavior)
print("\n[4b] Testing Full Allocation (WITH actual_resource='User_1')...")
actual_resource = event['org:resource']
allocated_with_actual = allocator.allocate(activity, timestamp, case_type, actual_resource=actual_resource)
print(f"Allocated resource: {allocated_with_actual}")

if allocated_with_actual == 'User_1':
    print("✓ SUCCESS! User_1 was allocated when provided as actual_resource")
elif allocated_with_actual is None:
    print("✗ FAIL! No resource allocated")
else:
    print(f"✗ FAIL! Different resource allocated: {allocated_with_actual}")
    print(f"  - Expected: User_1 (actual resource)")
    print(f"  - Got: {allocated_with_actual}")

# Check allocation logic
print("\n[5] Checking Allocation Logic...")
available_eligible = [
    r for r in eligible_resources 
    if allocator.availability.is_available(r, timestamp)
]
print(f"Available AND eligible resources: {len(available_eligible)}")
print(f"User_1 in available+eligible: {'User_1' in available_eligible}")

if available_eligible:
    print(f"Preview of available+eligible: {available_eligible[:10]}")
    
print("\n=== DEBUG COMPLETE ===")
