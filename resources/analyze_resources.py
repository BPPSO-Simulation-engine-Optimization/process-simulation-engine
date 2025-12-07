"""
Script to analyze resource patterns in BPIC 2017 dataset
"""
import sys
sys.path.append('pm4py-release')

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_dataframe
import pandas as pd
from datetime import datetime

# Load the dataset
log = xes_importer.apply('Dataset/BPI Challenge 2017.xes')
df = convert_to_dataframe(log)

print("=" * 80)
print("BPIC 2017 Dataset Analysis")
print("=" * 80)

print("\nColumns:", df.columns.tolist())
print("\nTotal events:", len(df))
print("Total cases:", df['case:concept:name'].nunique())
print("Total resources:", df['org:resource'].nunique())

print("\n" + "=" * 80)
print("Top 15 Resources by Activity Count:")
print("=" * 80)
print(df['org:resource'].value_counts().head(15))

print("\n" + "=" * 80)
print("Date Range:")
print("=" * 80)
print("Start:", df['time:timestamp'].min())
print("End:", df['time:timestamp'].max())

# Analyze temporal patterns
df['hour'] = pd.to_datetime(df['time:timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['time:timestamp']).dt.dayofweek
df['date'] = pd.to_datetime(df['time:timestamp']).dt.date

print("\n" + "=" * 80)
print("Activity by Hour of Day:")
print("=" * 80)
print(df['hour'].value_counts().sort_index())

print("\n" + "=" * 80)
print("Activity by Day of Week (0=Monday, 6=Sunday):")
print("=" * 80)
print(df['day_of_week'].value_counts().sort_index())

# Analyze resource-specific patterns
print("\n" + "=" * 80)
print("Resource-Specific Temporal Analysis (Top 5 Resources):")
print("=" * 80)

top_resources = df['org:resource'].value_counts().head(5).index

for resource in top_resources:
    resource_df = df[df['org:resource'] == resource].copy()
    resource_df['hour'] = pd.to_datetime(resource_df['time:timestamp']).dt.hour
    resource_df['day_of_week'] = pd.to_datetime(resource_df['time:timestamp']).dt.dayofweek
    
    print(f"\n{resource}:")
    print(f"  Total activities: {len(resource_df)}")
    print(f"  Active hours: {resource_df['hour'].min()}-{resource_df['hour'].max()}")
    print(f"  Most active hour: {resource_df['hour'].mode().values[0] if len(resource_df['hour'].mode()) > 0 else 'N/A'}")
    print(f"  Days active: {sorted(resource_df['day_of_week'].unique())}")
    print(f"  Date range: {resource_df['time:timestamp'].min()} to {resource_df['time:timestamp'].max()}")
