# Resource Availability Models - Quick Start Guide

This folder contains two resource availability models for process simulation: a **Basic Model** and an **Advanced Model with Resource Mining**.

## 🚀 Quick Start

### Set-Up

```bash
pip install -r requirements.txt
```
```bash
.venv\Scripts\activate
```
### Run Tests

```bash
python test_resource_availabilities.py
```

### Run Examples

```bash
python example_usage.py
```

## 📁 Files

- `resources/resource_availabilities.py` - Main implementation
- `test_resource_availabilities_new.py` - Comprehensive test suite
- `example_usage.py` - Usage examples
- `RESOURCE_AVAILABILITY_DOCUMENTATION.md` - Detailed documentation
- `analyze_resources.py` - Dataset analysis script

## 🎯 Two Models

### 1. Basic Model (Interval-Based)

Simple, configurable availability based on time patterns:

```python
from resources.resource_availabilities import ResourceAvailabilityModel

model = ResourceAvailabilityModel(
    event_log_df=df,
    interval_days=14,           # 2-week cycle
    workday_start_hour=8,       # 8 AM start
    workday_end_hour=17         # 5 PM end
)

is_available = model.is_available("User_1", datetime(2016, 6, 15, 10, 0))
```

**Features:**
- ✅ 2-week cyclic patterns
- ✅ Configurable working hours
- ✅ Dutch holiday awareness
- ✅ Fast and predictable

### 2. Advanced Model (Pattern Mining)

Data-driven model that learns from historical behavior:

```python
from resources.resource_availabilities import AdvancedResourceAvailabilityModel

model = AdvancedResourceAvailabilityModel(
    event_log_df=df,
    enable_pattern_mining=True
)

# Get detailed resource info
info = model.get_resource_info("User_1")
print(f"Working hours: {info['working_start']}-{info['working_end']}")
print(f"Peak hours: {info['peak_hours']}")

# Probabilistic availability
probability = model.predict_availability_probability("User_1", check_time)

# Get all available resources
available = model.get_available_resources(check_time)
```

**Features:**
- ✅ Learns individual resource patterns
- ✅ Resource clustering (K-means)
- ✅ Probabilistic predictions
- ✅ Peak hour detection
- ✅ Activity intensity analysis
- ✅ Workload statistics
- ✅ **Lifecycle-aware availability** (NEW)
- ✅ Real-time busy period tracking
- ✅ Current activity detection

## 📊 BPIC 2017 Results

### Dataset Stats
- **Events**: 1,202,267
- **Resources**: 149
- **Date Range**: Jan 2016 - Feb 2017
- **Lifecycle States**: 7 (complete, suspend, schedule, start, resume, ate_abort, withdraw)
- **Busy Periods Extracted**: 36,776 (143 resources)
- **Avg Busy Period**: 59.93 hours

### Key Findings

**5 Resource Clusters Discovered:**

| Cluster | Resources | Working Hours | Pattern |
|---------|-----------|---------------|---------|
| 0 | 29 | 5:30-17:30 | Early starters |
| 1 | 72 | 6:00-20:45 | Flexible hours (largest) |
| 2 | 8 | 8:00-16:00 | Standard office (Mon-Fri) |
| 3 | 3 | 1:00-23:20 | High-intensity/automated |
| 4 | 34 | 7:00-20:00 | Extended hours |

**Top Resources:**
- User_1: 148,404 activities (24/7 - automated)
- User_3: 26,342 activities (5:00-21:00)
- User_5: 22,900 activities (5:00-21:00)

## 🔍 What the Models Do

### Basic Model
1. Defines a 2-week repeating availability pattern
2. Checks if a time falls within working hours/days
3. Respects Dutch public holidays
4. Finds next available time slot

### Advanced Model
Does everything in Basic Model **PLUS**:
1. **Mines patterns** from historical event data
2. **Learns** each resource's actual working hours
3. **Detects** peak activity hours (top 25%)
4. **Calculates** activity probabilities by hour/day
5. **Clusters** resources with similar patterns
6. **Predicts** availability probability (0-1)
7. **Provides** workload statistics
8. **Tracks lifecycle states** (start/complete events)
9. **Identifies busy periods** when resources are occupied
10. **Prevents double-booking** by checking current activities

## 💡 Use Cases

### Use Basic Model When:
- No historical data available
- Need simple, predictable behavior
- Defining future scenarios
- Computational efficiency critical

### Use Advanced Model When:
- Rich historical data available
- Need realistic behavior
- Analyzing existing processes
- Want probabilistic simulations

## 🔧 Configuration

### Basic Model Parameters

```python
model = ResourceAvailabilityModel(
    event_log_df=df,
    interval_days=14,                    # Cycle length
    workday_start_hour=8,                # Start hour
    workday_end_hour=17,                 # End hour
    working_cycle_days={0,1,2,3,4,7,8,9,10,11}  # Working days in cycle
)
```

### Advanced Model Parameters

```python
model = AdvancedResourceAvailabilityModel(
    event_log_df=df,
    interval_days=14,
    enable_pattern_mining=True,          # Enable mining
    min_activity_threshold=10,           # Min activities to mine
    enable_lifecycle_tracking=True       # Track busy periods (NEW)
)
```

## 📈 API Overview

### Common Methods (Both Models)

```python
# Check availability
is_available(resource_id, current_time) -> bool

# Find next available time
get_next_available_time(resource_id, current_time) -> datetime

# Check if time is working time (general)
is_working_time(current_time) -> bool
```

### Advanced Model Only

```python
# Get resource pattern details
get_resource_info(resource_id) -> dict

# Get workload statistics
get_resource_workload(resource_id) -> dict

# Predict availability probability
predict_availability_probability(resource_id, current_time) -> float

# Get all available resources at time
get_available_resources(current_time) -> list

# Check with probability
is_available(resource_id, current_time, use_probabilistic=True) -> bool

# Lifecycle-aware methods (NEW)
is_resource_busy_at(resource_id, current_time) -> bool
get_current_activity(resource_id, current_time) -> str | None
get_busy_period_stats(resource_id) -> dict
get_resource_workload_at(resource_id, current_time, window_hours=1) -> int
```

## 🎓 Example Output

```
Resource: User_3
--------------------------------------------------------------------------------
Working hours: 5:00 - 21:00
Working days: [0, 1, 2, 3, 4, 5]  # Mon-Sat
Peak hours: [8, 9, 10, 13]
Total activities: 26,342
Activity intensity: 212.44 activities/day
Cluster ID: 1

At Wednesday 9:00 AM:
  Available: True
  Probability: 19.32%

At Saturday 10:00 AM:
  Available: True
  Probability: 3.84%
```

## 🔬 Advanced Features Explained

### 1. Resource Pattern Mining
- Analyzes all activities for each resource
- Extracts temporal patterns (hours, days)
- Calculates activity intensity metrics
- Stores first/last activity timestamps

### 2. Resource Clustering
- Uses K-means (5 clusters by default)
- Features: work hours, activity intensity, day spread
- Creates cluster profiles with aggregate stats
- Helps identify resource groups

### 3. Probabilistic Availability
- Based on historical activity distribution
- Hour-based and day-based probabilities
- Boosts probability during peak hours
- Returns value between 0 and 1

### 4. Peak Hour Detection
- Identifies top 25% most active hours
- Useful for workload planning
- Can boost availability probability
- Reflects actual work patterns

### 5. Lifecycle-Aware Availability (NEW)
- Extracts busy periods from lifecycle events
- Tracks start/schedule → complete transitions
- Prevents assigning resources already working
- Returns 0% probability when resource is busy
- Provides current activity information
- Calculates workload (overlapping activities)

## 📚 Documentation

For detailed information, see [RESOURCE_AVAILABILITY_DOCUMENTATION.md](RESOURCE_AVAILABILITY_DOCUMENTATION.md)

## 🧪 Testing

The test suite (`test_resource_availabilities_new.py`) covers:
- ✅ Basic model cycle patterns
- ✅ Working hours boundaries
- ✅ Weekend/holiday handling
- ✅ Future date predictions
- ✅ Pattern mining accuracy
- ✅ Cluster analysis
- ✅ Probability predictions
- ✅ Next available time queries
- ✅ Lifecycle busy period extraction
- ✅ Real-time availability checking
- ✅ Current activity detection
- ✅ Workload calculation

## 🤝 Integration with DESEngine

The **DESEngine** integrates resource availability through the `ResourceAllocator` and a **waiting queue mechanism**:

```python
# DESEngine resource allocation flow:
# 1. Check eligibility (permission model)
# 2. Filter by availability (this model)
# 3. Filter by dynamic busy state (ResourcePool)
# 4. Either allocate or queue

# When no resource is available:
# - Work is added to per-activity waiting queue
# - When ANY activity completes, the freed resource checks the queue
# - Waiting work is dispatched FIFO to eligible freed resources
```

### Key Integration Points

1. **`is_available(resource_id, timestamp)`** - Called by ResourceAllocator to filter eligible resources by working hours

2. **Dynamic busy tracking** - The DESEngine's `ResourcePool` tracks which resources are currently working (separate from this model's static patterns)

3. **User_1 special handling** - User_1 returns `True` for `is_available()` at all times (24/7 system resource)

### No Retry Scheduling

The DESEngine does NOT use `get_next_available_time()` to schedule retries. Instead:
- Work waits in queue until a resource completes an activity
- The freed resource checks if it can handle any waiting work
- This creates realistic resource contention and waiting times

## 🎯 Summary

**Basic Model**: Simple interval-based availability with configurable patterns
- ✅ Easy to configure
- ✅ Predictable behavior
- ✅ No data requirements

**Advanced Model**: AI-powered pattern mining with resource clustering
- ✅ Data-driven insights
- ✅ Lifecycle-aware tracking
- ✅ Real-time busy detection
- ✅ Individual resource profiles
- ✅ Probabilistic predictions
- ✅ Cluster analysis



---


