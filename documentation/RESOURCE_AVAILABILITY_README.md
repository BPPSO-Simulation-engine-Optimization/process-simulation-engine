# Resource Availability Models - Quick Start Guide

This folder contains two resource availability models for process simulation: a **Basic Model** and an **Advanced Model with Resource Mining**.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
python test_resource_availabilities_new.py
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

## 📊 BPIC 2017 Results

### Dataset Stats
- **Events**: 1,202,267
- **Resources**: 149
- **Date Range**: Jan 2016 - Feb 2017

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
- User_1: 148,404 activities (24/7 - likely automated)
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
    min_activity_threshold=10            # Min activities to mine
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

## 🤝 Integration with Simulation

```python
# In your simulation loop
def assign_activity(activity, current_time):
    # Get candidate resources
    candidates = get_resources_for_activity(activity)
    
    # Filter by availability
    available = [r for r in candidates 
                 if model.is_available(r, current_time)]
    
    if not available:
        # Schedule for next available time
        next_times = [(r, model.get_next_available_time(r, current_time))
                      for r in candidates]
        resource, next_time = min(next_times, key=lambda x: x[1])
        return schedule_activity(activity, resource, next_time)
    
    # Select from available (e.g., least loaded)
    resource = select_optimal(available)
    return schedule_activity(activity, resource, current_time)
```

## 🎯 Summary

**Basic Model**: Simple interval-based availability with configurable patterns
- ✅ Easy to configure
- ✅ Predictable behavior
- ✅ No data requirements

**Advanced Model**: AI-powered pattern mining with resource clustering
- ✅ Data-driven insights
- ✅ Individual resource profiles
- ✅ Probabilistic predictions
- ✅ Cluster analysis

Both models support process simulation engines and can handle real-world datasets like BPIC 2017.

---

**Questions?** Check the full documentation or run the examples!
