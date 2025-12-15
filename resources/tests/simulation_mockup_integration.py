import pandas as pd
import pm4py
import logging
from datetime import datetime
from resources.resource_allocation import ResourceAllocator
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_simulation_mockup(log_path):
    print(f"--- Starting Simulation Mockup ---")
    print(f"Log: {log_path}")

    # 1. Initialize the Resource Allocator
    # In a real simulation, this happens once at startup.
    # We use a subset or the full log to "discover" the model.
    print("\n[1] Initializing Resource Allocator...")
    allocator = ResourceAllocator(
        log_path=log_path,
        permission_method='ordinor',  # FullRecall
        availability_config={
            'enable_pattern_mining': True,
            'min_activity_threshold': 5
        }
    )
    
    # Load the log for replay (ground truth)
    print("\n[2] Loading Evaluation Log...")
    if log_path.endswith('.gz'):
        # pm4py handles gz automatically
        log = pm4py.read_xes(log_path)
        df_replay = pm4py.convert_to_dataframe(log)
    else:
        # Fallback if uncompressed
        df_replay = pd.read_csv(log_path) if log_path.endswith('csv') else pm4py.convert_to_dataframe(pm4py.read_xes(log_path))

    # Preprocessing for the test replay
    # Ensure efficient datetime
    df_replay['time:timestamp'] = pd.to_datetime(df_replay['time:timestamp'], utc=True)
    
    # Filter for 'complete' events only for this test, as allocation usually happens at start or complete
    if 'lifecycle:transition' in df_replay.columns:
        df_replay = df_replay[df_replay['lifecycle:transition'].isin(['start', 'complete'])]

    # Metrics
    total_events = 0
    correct_picks = 0
    eligible_picks = 0
    availability_misses = 0 # Allocator returned None (no available resource)
    
    # Sampling for speed in this mockup
    sample_size = 500
    print(f"\n[3] Replaying a sample of {sample_size} events...")
    
    # Shuffle and pick sample to get variety
    eval_subset = df_replay.sample(n=min(len(df_replay), sample_size), random_state=42).sort_values('time:timestamp')
    
    print(f"{'Timestamp':<25} | {'Activity':<30} | {'Case Type':<15} | {'Actual':<15} | {'Allocated':<15} | {'Result'}")
    print("-" * 120)

    for _, event in eval_subset.iterrows():
        activity = event['concept:name']
        timestamp = event['time:timestamp']
        actual_resource = event['org:resource']
        
        # KEY: Extract Case Type (Loans Goal) as per interface
        # Adjust column name if your log uses a different one for Loan Goal
        case_type = event.get('case:LoanGoal', 'Unknown') 
        
        # Verify columns exist
        if pd.isna(actual_resource): continue

        total_events += 1

        # CALL THE INTERFACE
        allocated_resource = allocator.allocate(
            activity=activity,
            timestamp=timestamp,
            case_type=case_type
        )

        # Evaluate
        status = ""
        if allocated_resource == actual_resource:
            correct_picks += 1
            eligible_picks += 1 # If it's the actual one, it must be eligible
            status = "MATCH"
        elif allocated_resource is not None:
             # Check if the allocated resource was at least *eligible* to do this (even if not the actual one)
             # This requires peeking into permissions
             eligible_list = allocator.permissions.get_eligible_resources(activity, timestamp, case_type)
             if allocated_resource in eligible_list:
                 eligible_picks += 1
                 status = "VALID (DIFF)"
             else:
                 status = "INVALID"
        else:
            availability_misses += 1
            status = "NO AVAIL"

        # Print first 10 rows or specific interesting ones
        if total_events <= 10:
            print(f"{str(timestamp):<25} | {activity[:28]:<30} | {str(case_type)[:13]:<15} | {str(actual_resource):<15} | {str(allocated_resource):<15} | {status}")

    # Metrics Summary
    accuracy = (correct_picks / total_events * 100) if total_events > 0 else 0
    validity = (eligible_picks / total_events * 100) if total_events > 0 else 0
    miss_rate = (availability_misses / total_events * 100) if total_events > 0 else 0

    print("-" * 120)
    print("\n--- Simulation Mockup Metrics ---")
    print(f"Total Events Simulated: {total_events}")
    print(f"Direct Matches (Accuracy): {correct_picks} ({accuracy:.1f}%)")
    print(f"Valid Allocations (Validity): {eligible_picks} ({validity:.1f}%)")
    print(f"Failed to Allocate (No Avail): {availability_misses} ({miss_rate:.1f}%)")
    
    print("\nNOTE: 'Direct Match' is hard to achieve in stochastic simulation.")
    print("'Valid Allocation' is the key metric: did we pick *someone* who could/should do it?")

if __name__ == "__main__":
    # Assuming running from project root
    log_file = "eventlog.xes.gz"
    if os.path.exists(log_file):
        run_simulation_mockup(log_file)
    else:
        # Try full path if simple relative failed (based on previous findings)
        full_path = "/Users/lgk/git/uni/BPSO25/process-simulation-engine/eventlog.xes.gz"
        if os.path.exists(full_path):
            run_simulation_mockup(full_path)
        else:
            print(f"ERROR: Could not find {log_file}")
