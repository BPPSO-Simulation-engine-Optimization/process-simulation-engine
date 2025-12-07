# Resource Availability Models for BPIC 2017

This document describes the implementation of resource availability models for the BPI Challenge 2017 dataset, including both a basic interval-based model and an advanced model with resource mining capabilities.

## Overview

The resource availability models simulate when resources (users/workers) are available to perform activities in a business process. The models support:

1. **Basic Model**: Interval-based availability with configurable working hours and cycles
2. **Advanced Model**: Pattern mining from historical data with individual resource profiles

## 1. Basic Resource Availability Model

### Features

The basic model implements a simple but flexible interval-based availability system:

- **Cyclic Pattern**: 2-week (14-day) repeating cycle by default
- **Working Hours**: Configurable working hours (default: 8:00-17:00)
- **Working Days**: Configurable working days within the cycle
- **Holiday Awareness**: Integration with Dutch public holidays
- **Timezone Support**: Handles timezone-aware timestamps

### Usage Example

```python
from resources.resource_availabilities import ResourceAvailabilityModel
import pandas as pd
from datetime import datetime

# Load event log
df = pd.read_csv('event_log.csv')

# Initialize model with 2-week cycle
model = ResourceAvailabilityModel(
    event_log_df=df,
    interval_days=14,
    workday_start_hour=8,
    workday_end_hour=17,
    working_cycle_days={0, 1, 2, 3, 4, 7, 8, 9, 10, 11}  # Mon-Fri both weeks
)

# Check availability
resource_id = "User_1"
check_time = datetime(2016, 6, 15, 10, 0)
is_available = model.is_available(resource_id, check_time)

# Find next available time
next_time = model.get_next_available_time(resource_id, check_time)
```

### Configuration

- `interval_days`: Length of work cycle (default: 14)
- `workday_start_hour`: Start of working hours (default: 8)
- `workday_end_hour`: End of working hours (default: 17)
- `working_cycle_days`: Set of active days in cycle (default: Mon-Fri in both weeks)

## 2. Advanced Resource Availability Model with Pattern Mining

### Features

The advanced model extends the basic model with sophisticated pattern mining capabilities:

#### 2.1 Resource Pattern Mining

Learns individual resource behavior from historical event logs:

- **Individual Working Hours**: Each resource's actual working hours based on activity timestamps
- **Day-of-Week Preferences**: Which days each resource typically works
- **Peak Hours Detection**: Identifies when resources are most active
- **Activity Intensity**: Calculates average activities per day
- **Temporal Coverage**: Tracks first and last activity dates

#### 2.2 Resource Clustering

Groups resources with similar work patterns using K-means clustering:

- **Feature Extraction**: Working hours, day spread, activity intensity
- **Cluster Profiles**: Aggregate statistics for each cluster
- **Pattern Discovery**: Identifies common work patterns across resources

#### 2.3 Probabilistic Availability

Predicts availability probability based on historical patterns:

- **Hour-based Probability**: Likelihood of availability by hour
- **Day-based Probability**: Likelihood of availability by day of week
- **Peak Hour Boosting**: Higher probability during peak activity times

### Usage Example

```python
from resources.resource_availabilities import AdvancedResourceAvailabilityModel
import pandas as pd
from datetime import datetime

# Load event log
df = pd.read_csv('event_log.csv')

# Initialize advanced model with pattern mining
model = AdvancedResourceAvailabilityModel(
    event_log_df=df,
    interval_days=14,
    enable_pattern_mining=True,
    min_activity_threshold=10
)

# Check availability (with learned patterns)
resource_id = "User_1"
check_time = datetime(2016, 6, 15, 10, 0)
is_available = model.is_available(resource_id, check_time)

# Get availability probability
probability = model.predict_availability_probability(resource_id, check_time)

# Get resource pattern information
pattern_info = model.get_resource_info(resource_id)
print(f"Working hours: {pattern_info['working_start']}-{pattern_info['working_end']}")
print(f"Peak hours: {pattern_info['peak_hours']}")
print(f"Activity intensity: {pattern_info['activity_intensity']}")

# Get workload statistics
workload = model.get_resource_workload(resource_id)
print(f"Avg activities/day: {workload['avg_activities_per_day']}")

# Get all available resources at a specific time
available_resources = model.get_available_resources(check_time)
print(f"Available resources: {len(available_resources)}")
```

## 3. Pattern Mining Details

### 3.1 Extracted Patterns

For each resource with sufficient data (≥10 activities by default), the system extracts:

- **working_start**: Earliest hour of activity
- **working_end**: Latest hour of activity + 1
- **working_days**: Set of active weekdays (0=Monday, 6=Sunday)
- **peak_hours**: Top 25% most active hours
- **hour_probabilities**: Normalized activity distribution by hour
- **dow_probabilities**: Normalized activity distribution by day of week
- **activity_intensity**: Average activities per active day
- **first_activity**: Timestamp of first observed activity
- **last_activity**: Timestamp of last observed activity

### 3.2 Clustering Features

Resources are clustered based on:

1. Working start hour
2. Working end hour
3. Number of working days per week
4. Activity intensity (log-transformed)
5. Hour spread (working_end - working_start)
6. Day spread (number of unique active days)

## 4. BPIC 2017 Analysis Results

### Dataset Overview

- **Events**: 1,202,267
- **Cases**: 31,509
- **Resources**: 149
- **Date Range**: Jan 1, 2016 - Feb 1, 2017

### Key Findings

#### 4.1 Temporal Patterns

- **Peak Hours**: 8-14 (morning and early afternoon)
- **Active Days**: Primarily Monday-Friday, some Saturday activity
- **Off-Peak**: Late evening and early morning hours

#### 4.2 Resource Patterns

**Top Resources by Activity**:
- User_1: 148,404 activities (24/7 availability - likely automated)
- User_3: 26,342 activities (5:00-21:00, Mon-Sat)
- User_5: 22,900 activities (5:00-21:00, Mon-Sat)

#### 4.3 Cluster Analysis

The model identified 5 distinct resource clusters:

**Cluster 0** (29 resources):
- Working hours: ~5:30-17:30
- Peak hours: 6-15
- Avg intensity: 95.39 activities/day
- Pattern: Early starters with moderate hours

**Cluster 1** (72 resources) - Largest cluster:
- Working hours: ~6:00-20:45
- Peak hours: 4-20 (wide range)
- Avg intensity: 92.06 activities/day
- Pattern: Flexible hours, high availability

**Cluster 2** (8 resources):
- Working hours: ~8:00-16:00
- Peak hours: 8-15
- Avg intensity: 55.04 activities/day
- Pattern: Standard office hours, Mon-Fri only

**Cluster 3** (3 resources) - High-intensity:
- Working hours: ~1:00-23:20
- Peak hours: 6-14, 18-19
- Avg intensity: 172.82 activities/day
- Pattern: Automated or high-volume processors

**Cluster 4** (34 resources):
- Working hours: ~7:00-20:00
- Peak hours: 5-19
- Avg intensity: 74.21 activities/day
- Pattern: Extended hours, Mon-Fri focused

## 5. Simulation Integration

### Using in Process Simulation

Both models can be integrated into process simulation engines:

```python
# During simulation, check if resource is available
def assign_activity(case_id, activity, current_time, simulation_model):
    # Get available resources for this activity
    candidate_resources = get_resources_for_activity(activity)
    
    # Filter by availability
    available = [
        r for r in candidate_resources 
        if simulation_model.is_available(r, current_time)
    ]
    
    if not available:
        # Find next available time
        next_times = [
            (r, simulation_model.get_next_available_time(r, current_time))
            for r in candidate_resources
        ]
        resource, next_time = min(next_times, key=lambda x: x[1])
        return resource, next_time
    
    # Select from available resources (e.g., least loaded)
    resource = select_resource(available)
    return resource, current_time
```

### Probabilistic Mode

For more realistic simulations, use probabilistic availability:

```python
# Use probability-based availability for stochastic simulation
is_available = model.is_available(resource_id, current_time, use_probabilistic=True)

# Or get explicit probability for decision-making
prob = model.predict_availability_probability(resource_id, current_time)
if random.random() < prob:
    assign_to_resource(resource_id)
```

## 6. Advantages and Limitations

### Basic Model

**Advantages**:
- Simple and predictable
- Easy to configure
- Low computational overhead
- Suitable when no historical data available

**Limitations**:
- No individual resource differences
- Fixed patterns may not reflect reality
- Cannot capture temporal trends

### Advanced Model

**Advantages**:
- Data-driven, reflects actual behavior
- Individual resource profiles
- Probabilistic predictions
- Identifies resource patterns and clusters
- More realistic simulation results

**Limitations**:
- Requires sufficient historical data
- Higher computational cost
- May overfit to historical patterns
- Assumes patterns remain stable

## 7. Best Practices

1. **Use Basic Model when**:
   - No historical data available
   - Need simple, predictable behavior
   - Defining future scenarios with new resources
   - Computational efficiency is critical

2. **Use Advanced Model when**:
   - Rich historical event logs available
   - Need realistic resource behavior
   - Analyzing existing processes
   - Comparing actual vs. simulated performance

3. **Data Requirements**:
   - Minimum 10 activities per resource (adjustable)
   - At least 2 weeks of data for pattern detection
   - Consistent timestamp formats
   - Valid resource identifiers

4. **Tuning Parameters**:
   - Adjust `min_activity_threshold` based on data sparsity
   - Modify cluster count based on resource population
   - Configure `interval_days` to match organizational cycles

## 8. Future Enhancements

Potential improvements to the models:

1. **Temporal Evolution**: Detect and model changes in availability patterns over time
2. **Skill-based Availability**: Different availability for different activity types
3. **Team Patterns**: Model team-based working patterns and dependencies
4. **Vacation/Leave**: Incorporate planned absences and holidays
5. **Learning Curves**: Model improving efficiency over time
6. **Workload Balancing**: Incorporate fatigue and workload limits
7. **Multi-timezone Support**: Better handling of global distributed teams

## 9. References

- BPI Challenge 2017: https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
- PM4Py Documentation: https://pm4py.fit.fraunhofer.de/
- Process Mining: Data Science in Action (van der Aalst, 2016)

## 10. Testing

Run the comprehensive test suite:

```bash
python test_resource_availabilities_new.py
```

This will:
- Load the BPIC 2017 dataset
- Test the basic model with various scenarios
- Mine resource patterns
- Cluster resources
- Display availability predictions
- Show cluster analysis
