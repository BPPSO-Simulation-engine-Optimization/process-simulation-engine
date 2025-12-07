"""
Analyze lifecycle transitions in BPIC 2017 dataset
"""
import sys
sys.path.append('pm4py-release')

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe
import pandas as pd
from datetime import timedelta

LOG_FILE = r"Dataset\BPI Challenge 2017.xes"

print("=" * 80)
print("LIFECYCLE TRANSITION ANALYSIS - BPIC 2017")
print("=" * 80)

# Load data
log = xes_importer.apply(LOG_FILE)
df = convert_to_dataframe(log)

print(f"\nTotal events: {len(df):,}")

# Check for lifecycle column
if 'lifecycle:transition' in df.columns:
    print("\n✅ Lifecycle column found!")
    
    # Count lifecycle states
    lifecycle_counts = df['lifecycle:transition'].value_counts()
    print("\nLifecycle State Distribution:")
    print("-" * 80)
    for state, count in lifecycle_counts.items():
        percentage = (count / len(df)) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {state:20s}: {count:8,} ({percentage:5.2f}%) {bar}")
    
    # Analyze by resource
    print("\n\nLifecycle States by Resource (Top 5 resources):")
    print("-" * 80)
    top_resources = df['org:resource'].value_counts().head(5).index
    
    for resource in top_resources:
        resource_df = df[df['org:resource'] == resource]
        states = resource_df['lifecycle:transition'].value_counts()
        print(f"\n{resource} (Total: {len(resource_df):,}):")
        for state, count in states.items():
            print(f"  {state}: {count:,}")
    
    # Check for complete lifecycle pairs (START -> COMPLETE)
    print("\n\nAnalyzing START → COMPLETE pairs:")
    print("-" * 80)
    
    # Group by case and activity
    grouped = df.groupby(['case:concept:name', 'concept:name'])
    
    pairs_found = 0
    total_durations = []
    
    for (case, activity), group in grouped:
        group_sorted = group.sort_values('time:timestamp')
        
        # Look for START/COMPLETE pairs
        has_start = 'START' in group_sorted['lifecycle:transition'].values
        has_complete = 'COMPLETE' in group_sorted['lifecycle:transition'].values
        
        if has_start and has_complete:
            start_rows = group_sorted[group_sorted['lifecycle:transition'] == 'START']
            complete_rows = group_sorted[group_sorted['lifecycle:transition'] == 'COMPLETE']
            
            if not start_rows.empty and not complete_rows.empty:
                start_time = start_rows.iloc[0]['time:timestamp']
                complete_time = complete_rows.iloc[-1]['time:timestamp']
                duration = (complete_time - start_time).total_seconds() / 3600  # hours
                
                pairs_found += 1
                total_durations.append({
                    'case': case,
                    'activity': activity,
                    'resource': group_sorted.iloc[0]['org:resource'],
                    'start_time': start_time,
                    'complete_time': complete_time,
                    'duration_hours': duration
                })
    
    print(f"Found {pairs_found:,} START→COMPLETE pairs")
    
    if total_durations:
        duration_df = pd.DataFrame(total_durations)
        
        print(f"\nDuration Statistics (hours):")
        print(duration_df['duration_hours'].describe())
        
        print(f"\nTop 10 longest activities:")
        top_durations = duration_df.nlargest(10, 'duration_hours')
        for idx, row in top_durations.iterrows():
            print(f"  {row['activity']:40s}: {row['duration_hours']:8.2f}h ({row['resource']})")
        
        print(f"\nTop 10 shortest activities:")
        short_durations = duration_df.nsmallest(10, 'duration_hours')
        for idx, row in short_durations.iterrows():
            print(f"  {row['activity']:40s}: {row['duration_hours']:8.2f}h ({row['resource']})")
        
        # Statistics by resource
        print(f"\n\nBusy Period Statistics by Resource (Top 10):")
        print("-" * 80)
        resource_stats = duration_df.groupby('resource').agg({
            'duration_hours': ['count', 'mean', 'sum']
        }).round(2)
        resource_stats.columns = ['Count', 'Avg Duration (h)', 'Total Busy Time (h)']
        resource_stats = resource_stats.sort_values('Count', ascending=False).head(10)
        print(resource_stats)
        
else:
    print("\n❌ No lifecycle:transition column found!")
    print("\nAvailable columns:")
    for col in df.columns:
        print(f"  - {col}")

print("\n" + "=" * 80)
