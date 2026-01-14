# Process Simulation Engine -

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Predictors & ML Components](#predictors--ml-components)
5. [Resource Management](#resource-management)
6. [Integration & Benchmarking](#integration--benchmarking)
7. [Usage Guide](#usage-guide)
8. [Data Flow](#data-flow)
9. [Configuration](#configuration)

---

## Overview

A **data-driven discrete-event simulation engine** for business process modeling, specifically designed for the **BPIC 2017 Loan Application Process**. The engine combines:
- Discrete event simulation (DES)
- Machine learning predictors (LSTM-based next activity prediction)
- Resource allocation and availability modeling
- Event log generation (CSV/XES format)
- Comprehensive benchmarking and quality assessment

**Key Features:**
- ✅ Probabilistic processing times with ML-based prediction
- ✅ Next activity prediction using LSTM models
- ✅ Resource allocation with permission and availability models
- ✅ Case arrival time prediction with temporal patterns
- ✅ Case attribute prediction (credit score, amounts, etc.)
- ✅ Gateway branching prediction
- ✅ Event log export to XES and CSV
- ✅ Simulation quality benchmarking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Simulation Engine (DESEngine)               │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ EventQueue    │  │ SimClock     │  │ CaseManager      │    │
│  │ (Priority Q)  │  │ (Virtual T)  │  │ (Active Cases)   │    │
│  └───────────────┘  └──────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐   ┌────────────────┐   ┌─────────────────┐
│  Predictors  │   │    Resources   │   │   Log Export    │
├──────────────┤   ├────────────────┤   ├─────────────────┤
│• Next Act    │   │• Allocator     │   │• XES Format     │
│• Proc Time   │   │• Pool          │   │• CSV Format     │
│• Case Arrival│   │• Availability  │   │• PM4Py compat   │
│• Attributes  │   │• Permissions   │   └─────────────────┘
└──────────────┘   └────────────────┘
        ↓                   ↓
┌────────────────────────────────────────┐
│       Integration & Benchmarking       │
├────────────────────────────────────────┤
│• Ground Truth Extraction               │
│• SimulationBenchmark (Descriptive)     │
│• Next Activity Prediction Metrics      │
│• DFG/Variant/Throughput Comparison     │
└────────────────────────────────────────┘
```

---

## Core Components

### 1. Discrete Event Simulation (DES) Engine
**Location:** `simulation/engine.py`

The heart of the simulation system implementing a time-ordered discrete event simulation.

#### Key Classes:
- **`DESEngine`**: Main simulation orchestrator
  - Manages event queue, clock, and case states
  - Routes events to appropriate handlers
  - Coordinates predictors and resource allocation
  - Produces event logs

#### Event Types:
```python
class EventType:
    CASE_ARRIVAL      # New case enters system
    ACTIVITY_START    # Activity begins (optional)
    ACTIVITY_COMPLETE # Activity finishes (logged)
    CASE_END          # Case exits system
```

#### Simulation Flow:
```
1. CASE_ARRIVAL 
   → Create case state
   → Predict first activity
   → Schedule ACTIVITY_COMPLETE

2. ACTIVITY_COMPLETE
   → Log event to output
   → Release resource
   → Predict next activity
   → Schedule next or CASE_END

3. CASE_END
   → Cleanup case state
   → Update statistics
```

#### Main Methods:
```python
def run(num_cases: int = 100, max_time: datetime = None) -> List[Dict]:
    """Run simulation for N cases or until max_time"""
    
def _schedule_case_arrivals(num_cases: int) -> None:
    """Schedule initial case arrival events"""
    
def _handle_event(event: SimulationEvent) -> None:
    """Route event to appropriate handler"""
```

### 2. Event Management
**Location:** `simulation/events.py`, `simulation/event_queue.py`

#### SimulationEvent
```python
@dataclass
class SimulationEvent:
    timestamp: datetime
    event_type: EventType
    case_id: str
    activity: Optional[str] = None
    resource: Optional[str] = None
```

#### EventQueue
- Priority queue implementation using `heapq`
- Events ordered by timestamp
- O(log n) insertion and removal

### 3. Clock & Case Management
**Location:** `simulation/clock.py`, `simulation/case_manager.py`

#### SimulationClock
- Virtual time management
- Advances to each event timestamp
- Tracks simulation duration

#### CaseManager & CaseState
```python
@dataclass
class CaseState:
    case_id: str
    start_time: datetime
    current_activity: str
    activity_count: int
    attributes: Dict[str, Any]  # credit_score, amount, etc.
```

Tracks active cases and their progression through the process.

---

## Predictors & ML Components

### 1. Next Activity Prediction
**Location:** `Next-Activity-Prediction/`

#### Basic Prediction
- **Type:** Probabilistic sampling from DFG
- **Input:** Current activity
- **Output:** Next activity + transition probability
- **Usage:** Fallback when LSTM unavailable

#### Advanced Prediction (LSTM)
**Location:** `Next-Activity-Prediction/advanced/`

```python
LSTMNextActivityPredictor:
    - Encoder: OneHotEncoder for activities
    - Model: LSTM with 100 hidden units
    - Input: Activity sequence + case attributes
    - Output: Next activity prediction
    - Training: BPIC 2017 event log
```

**Key Features:**
- Sequence-based prediction (considers activity history)
- Case attribute incorporation (credit_score, amounts)
- Confidence scores for predictions
- Special handling for end activities

**Files:**
- `models/`: Pre-trained LSTM models
- `preprocessing/`: Data preparation
- `evaluation/`: Model performance metrics
- `simulation.py`: Integration with DES engine

### 2. Processing Time Prediction
**Location:** `processing_time_prediction/`

**Class:** `ProcessingTimePredictionClass`

```python
Features:
- Activity name
- Case attributes (credit_score, amounts, terms)
- Temporal features (hour, day_of_week)
- Previous activity (if available)

Model: Gradient Boosting Regressor
Output: Duration in seconds
```

**Training:**
- Fitted on BPIC 2017 historical data
- Captures activity-specific duration patterns
- Handles resource availability effects

**Files:**
- `processing_time_model_model.joblib`: Trained model
- `processing_time_model_encoders.joblib`: Feature encoders
- `processing_time_model_metadata.joblib`: Feature metadata
- `processing_time_model_scaler.joblib`: Feature scaling

### 3. Case Arrival Time Prediction
**Location:** `case_arrival_times_prediction/`

**Approach:** Time series forecasting with daily and weekly patterns

**Components:**
```python
GlobalSegmentation:
    - Identifies weekly arrival patterns
    - Clusters weekdays by similarity

IntradayModel:
    - KDE-based arrival distribution per weekday
    - Captures hourly patterns

Forecasting:
    - ARIMA/Prophet for daily volume prediction
    - Intraday KDE for time-of-day sampling
```

**Pipeline:**
1. Extract case arrivals from historical log
2. Segment by weekday
3. Fit intraday distributions
4. Generate arrival timestamps for simulation

### 4. Case Attribute Prediction
**Location:** `case_attribute_prediction/`

**Predictors:**
- `credit_score.py`: Loan applicant credit score
- `offered_amount.py`: Loan amount offered
- `first_withdrawal_amount.py`: Initial withdrawal
- `monthly_cost.py`: Monthly payment amount
- `number_of_terms.py`: Loan term length
- `accepted.py`: Application acceptance (binary)
- `selected.py`: Offer selection (binary)

**Registry Pattern:**
```python
CaseAttributeRegistry:
    - Registers all attribute predictors
    - Provides unified interface
    - Enables conditional dependencies
```

**Usage in Simulation:**
```python
case_attributes = {
    'CreditScore': 650,
    'FirstWithdrawalAmount': 5000,
    'OfferedAmount': 7500,
    # ...
}
```

### 5. Branching/Gateway Prediction
**Location:** `branching_prediction/`

**Purpose:** Predict which path to take at decision points (XOR gateways)

**Method:**
- Extract gateways from BPMN model
- Learn probability distribution from historical data
- Sample path at runtime based on case attributes

---

## Resource Management

### 1. Resource Allocation
**Location:** `resources/`

**Components:**

#### ResourceAllocator
```python
class ResourceAllocator:
    permissions: PermissionModel    # Who CAN do what
    availability: AvailabilityModel # Who is working when
    
    def allocate(activity, timestamp, case_attrs) -> str:
        """Find eligible, available, non-busy resource"""
```

#### PermissionModel
**File:** `resources/resource_permissions/`

Maps activities to eligible resources:
```python
{
    "W_Complete application": ["User_1", "User_3", "User_19", ...],
    "A_Accepted": ["User_1", "User_2", ...],
    # ...
}
```

#### AvailabilityModel
**File:** `resources/resource_availabilities/`

Defines working hours per resource:
```python
{
    "User_1": {
        "Monday": [(9, 17)],      # 9 AM - 5 PM
        "Tuesday": [(9, 17)],
        # ...
    }
}
```

### 2. Resource Pool
**Location:** `resources/resource_pool.py`

**Features:**
- Dynamic busy tracking (which resources are currently working)
- Per-activity waiting queues (FIFO)
- Automatic dispatch when resources become free
- Statistics on waiting time and resource contention

**Key Methods:**
```python
def mark_busy(resource: str, activity: str, timestamp: datetime)
def mark_free(resource: str, timestamp: datetime)
def has_waiting_work() -> bool
def get_waiting_work(activity: str) -> List[WaitingWork]
def add_waiting_work(work: WaitingWork)
```

**Resource Allocation Logic:**
```
1. Check Eligibility (permissions)
   ↓ NO → Add to waiting queue
2. Check Availability (working hours)
   ↓ NO → Add to waiting queue
3. Check Busy State (currently working?)
   ↓ NO → Allocate
   ↓ YES → Add to waiting queue
```

---

## Integration & Benchmarking

### 1. Integration Testing
**Location:** `integration/`

#### test_integration.py
End-to-end simulation test with:
- Case arrival time prediction
- Case attribute prediction
- Next activity prediction (LSTM)
- Processing time prediction
- Resource allocation
- Event log export
- Ground truth extraction

#### Ground Truth Creation
**File:** `integration/create_ground_truth.py`

Extracts first N cases from original log for benchmarking:
```python
def create_ground_truth_subset(log, num_cases=100):
    """Extract first N cases by arrival time"""
    # Sort by case start time
    # Select first N cases
    # Export to CSV and XES
```

### 2. Simulation Benchmarking
**Location:** `integration/SimulationBenchmark.py`

**Class:** `SimulationBenchmark`

Comprehensive comparison between original and simulated logs:

#### Metrics Categories:

**Basic Statistics:**
- Number of cases
- Number of events
- Number of variants
- Timespan (start/end dates)

**Events Per Case:**
- Mean, Median, P75, P90, P95, Max

**Case Throughput Time (Cycle Time):**
- Mean, Median, P75, P90, P95, Max (in hours)

**Case Arrivals & Completions:**
- Daily patterns
- Mean/Median daily rates

**Control Flow (DFG):**
- Directly-Follows Graph comparison
- Top edge frequencies
- Edge-by-edge comparison
- Missing/Extra edges

**Trace Variants:**
- Top variants comparison
- Variant frequency distribution
- Coverage analysis

**Start/End Activities:**
- Activity distribution at case start/end

**Activity Durations:**
- Time to next event per activity
- Mean, Median, P90 (in hours)

**Resource Statistics:**
- Number of resources
- Events per resource
- Activity-Resource pair frequencies

#### Next Activity Prediction Metrics (NEW!)
**Purpose:** Evaluate simulation quality by comparing activity sequences

**Metrics:**
- **Accuracy**: Overall prediction correctness
- **Precision (Macro/Weighted)**: Per-class precision
- **Recall (Macro/Weighted)**: Per-class recall
- **F1-Score (Macro/Weighted)**: Harmonic mean
- **AUC (Macro)**: Multiclass ROC AUC
- **AUC-PR (Macro)**: Multiclass Precision-Recall AUC
- **Classification Report**: Per-activity detailed metrics
- **Support/Weights**: Sample counts per activity

**Implementation:**
```python
def _compute_next_activity_metrics(self) -> Dict:
    # Align sequences between original and simulated logs
    y_true, y_pred = self._align_next_activity_sequences()
    
    # Compute sklearn metrics
    precision, recall, fscore, support = precision_recall_fscore_support(...)
    auc_score = multiclass_roc_auc_score(...)
    prauc_score = multiclass_pr_auc_score(...)
    
    return {
        'overall_metrics': {...},
        'per_class_metrics': DataFrame(...),
        'classification_report': ...
    }
```

**Alignment Strategy:**
- Compare cases by position (1st case vs 1st case)
- Within each case, align activities by event index
- Extract (current_activity → next_activity) pairs
- Use as (y_true, y_pred) for classification metrics

**Export:**
- Console output with formatted tables
- Excel export with multiple sheets:
  - `NAP Overall`: Summary metrics
  - `NAP Per Class`: Per-activity breakdown
  - `NAP Confusion`: Confusion matrix summary

**Usage:**
```python
benchmark = SimulationBenchmark(
    'ground_truth_log.csv',
    'simulated_log.csv'
)
results = benchmark.compute_all_metrics()
benchmark.print_summary()
benchmark.export_results('benchmark_results.xlsx')
```

---

## Usage Guide

### Installation

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 2. Install pm4py from local release
pip install pm4py-release/

# 3. Install requirements
pip install -r requirements.txt
pip install -r requirements_simengine.txt
```

### Running a Simulation

```python
from simulation import DESEngine, load_simulation_assets
from resources import ResourceAllocator
from datetime import datetime

# 1. Load trained models and data
assets = load_simulation_assets('Dataset/BPI Challenge 2017.xes')

# 2. Create resource allocator
allocator = ResourceAllocator(
    permissions=assets['permissions'],
    availability=assets['availability']
)

# 3. Initialize engine
engine = DESEngine(
    resource_allocator=allocator,
    next_activity_predictor=assets['next_activity_predictor'],
    processing_time_predictor=assets['processing_time_predictor'],
    case_arrival_predictor=assets['case_arrival_predictor'],
    case_attribute_predictor=assets['case_attribute_predictor'],
    arrival_timestamps=assets['arrival_timestamps'],
    start_time=datetime(2016, 1, 1, 9, 0)
)

# 4. Run simulation
events = engine.run(num_cases=100)

# 5. Export results
from simulation import LogExporter
exporter = LogExporter(events)
exporter.export_csv('simulated_log.csv')
exporter.export_xes('simulated_log.xes')
```

### Running Benchmark

```python
from integration.SimulationBenchmark import SimulationBenchmark

# Compare logs
benchmark = SimulationBenchmark(
    original_log='integration/output/ground_truth_log.csv',
    simulated_log='simulated_log_100.csv'
)

# Compute all metrics
results = benchmark.compute_all_metrics()

# Print summary to console
benchmark.print_summary()

# Export to Excel
benchmark.export_results('benchmark_results.xlsx')
```

---

## Data Flow

### End-to-End Simulation Data Flow

```
1. INPUT: Original Event Log (BPIC 2017.xes)
   ↓
2. PREPROCESSING & TRAINING
   ├─→ Extract Case Arrivals → CaseArrivalPredictor
   ├─→ Train LSTM Model → NextActivityPredictor
   ├─→ Train Processing Time Model → ProcessingTimePredictor
   ├─→ Train Attribute Models → CaseAttributePredictors
   ├─→ Extract Resource Permissions → PermissionModel
   └─→ Extract Resource Availability → AvailabilityModel
   ↓
3. SIMULATION RUNTIME
   │
   ├─→ [Event Queue] Schedule case arrivals
   │
   ├─→ [CASE_ARRIVAL Event]
   │   ├─→ Predict case attributes (credit_score, amounts, etc.)
   │   ├─→ Predict first activity
   │   ├─→ Allocate resource (check permissions + availability + busy)
   │   │   ├─ Available → Mark busy
   │   │   └─ Not available → Add to waiting queue
   │   ├─→ Predict processing time
   │   └─→ Schedule ACTIVITY_COMPLETE
   │
   ├─→ [ACTIVITY_COMPLETE Event]
   │   ├─→ Log event (timestamp, activity, resource, attributes)
   │   ├─→ Release resource (mark free)
   │   ├─→ Process waiting queue (dispatch waiting work)
   │   ├─→ Predict next activity (using LSTM + case history)
   │   │   ├─ Next activity exists → Schedule ACTIVITY_COMPLETE
   │   │   └─ End activity → Schedule CASE_END
   │   └─→ Loop
   │
   └─→ [CASE_END Event]
       ├─→ Cleanup case state
       └─→ Update statistics
   ↓
4. OUTPUT: Simulated Event Log
   ├─→ Export to CSV (simulated_log.csv)
   └─→ Export to XES (simulated_log.xes)
   ↓
5. BENCHMARKING
   ├─→ Load Original Log (ground truth)
   ├─→ Load Simulated Log
   ├─→ Compute Metrics:
   │   ├─ Basic Statistics
   │   ├─ Control Flow (DFG)
   │   ├─ Variants
   │   ├─ Throughput Time
   │   ├─ Resource Usage
   │   └─ Next Activity Prediction Quality ⭐
   └─→ Export Comparison Report (Excel)
```

### Detailed Resource Allocation Flow

```
Activity needs to be scheduled
↓
ResourceAllocator.allocate(activity, timestamp, case_attrs)
↓
1. Get eligible resources (PermissionModel)
   → Who CAN perform this activity?
   ↓
2. Filter by availability (AvailabilityModel)
   → Who is working at this timestamp?
   ↓
3. Check busy state (ResourcePool)
   → Who is NOT currently busy?
   ↓
   ┌─ Resource Available
   │  ├→ Mark as busy in ResourcePool
   │  ├→ Assign to activity
   │  └→ Schedule completion
   │
   └─ No Resource Available
      ├→ Create WaitingWork entry
      ├→ Add to per-activity waiting queue
      └→ Wait for resource to become free
      
Activity completes
↓
ResourcePool.mark_free(resource)
↓
Check waiting queue for activities this resource can handle
↓
┌─ Waiting work exists
│  ├→ Get next work from queue (FIFO)
│  ├→ Allocate resource
│  ├→ Mark busy again
│  └→ Schedule completion
│
└─ No waiting work
   └→ Resource stays idle
```

---

## Configuration

### Key Configuration Files

#### 1. Process Model
**Location:** `process_model/loan_application.bpmn`

BPMN 2.0 XML defining the loan application process structure:
- Activities
- Gateways (XOR, AND)
- Sequence flows
- Start/End events

#### 2. Trained Models
**Location:** `models/`

Pre-trained ML models:
- `branch_predictor.joblib`: Gateway decision model
- `processing_time_model_*.joblib`: Duration prediction
- LSTM models in `Next-Activity-Prediction/advanced/models/`

#### 3. Resource Configuration
**Location:** `resources/`

- `resource_permissions/`: Activity → Resource mapping
- `resource_availabilities/`: Resource → Working hours

#### 4. Requirements
- `requirements.txt`: Main dependencies
- `requirements_simengine.txt`: Simulation-specific deps

Key dependencies:
```
pm4py>=2.7.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.10.0  # For LSTM
joblib>=1.3.0
openpyxl>=3.1.0     # For Excel export
```

---

## Project Structure

```
process-simulation-engine/
│
├── simulation/                          # Core DES Engine
│   ├── engine.py                        # Main simulation loop
│   ├── events.py                        # Event definitions
│   ├── event_queue.py                   # Priority queue
│   ├── clock.py                         # Virtual time
│   ├── case_manager.py                  # Case state tracking
│   └── log_exporter.py                  # CSV/XES export
│
├── Next-Activity-Prediction/            # Next activity models
│   ├── basic_prediction/                # DFG-based fallback
│   └── advanced/                        # LSTM-based
│       ├── models/                      # Pre-trained models
│       ├── preprocessing/               # Data prep
│       ├── evaluation/                  # Metrics
│       └── simulation.py                # Integration
│
├── processing_time_prediction/          # Duration models
│   ├── ProcessingTimePredictionClass.py
│   └── processing_time.py
│
├── case_arrival_times_prediction/       # Arrival forecasting
│   ├── pipeline.py                      # Main pipeline
│   ├── global_segmentation.py           # Weekly patterns
│   ├── intraday.py                      # Hourly patterns
│   └── forecasting.py                   # ARIMA/Prophet
│
├── case_attribute_prediction/           # Case attributes
│   ├── registry.py                      # Central registry
│   ├── credit_score.py
│   ├── offered_amount.py
│   └── ...                              # Other attributes
│
├── branching_prediction/                # Gateway decisions
│   └── gateway_extractor.py
│
├── resources/                           # Resource management
│   ├── allocator.py                     # Resource allocation
│   ├── resource_pool.py                 # Busy tracking + queues
│   ├── resource_permissions/            # Permission models
│   └── resource_availabilities/         # Working hours
│
├── integration/                         # Integration tests
│   ├── test_integration.py              # End-to-end test
│   ├── create_ground_truth.py           # Ground truth extraction
│   ├── SimulationBenchmark.py           # Benchmarking ⭐
│   └── output/                          # Generated logs
│
├── process_model/                       # BPMN models
│   └── loan_application.bpmn
│
├── models/                              # Trained ML models
│   ├── branch_predictor.joblib
│   └── processing_time_model_*.joblib
│
├── Dataset/                             # Input data
│   └── BPI Challenge 2017.xes
│
├── README.md                            # Quick start guide
├── DOCUMENTATION.md                     # This file ⭐
├── requirements.txt
└── requirements_simengine.txt
```

---

## Key Algorithms

### 1. Event Queue Priority Ordering
```python
# Events ordered by (timestamp, event_type_priority)
priority = (event.timestamp, event.event_type.value)
heapq.heappush(queue, (priority, event))
```

### 2. Resource Allocation Decision Tree
```python
def allocate_resource(activity, timestamp, case_attrs):
    # 1. Eligibility filter
    eligible = permission_model.get_eligible(activity)
    if not eligible:
        return None  # No one can do this activity
    
    # 2. Availability filter
    available = [r for r in eligible 
                 if availability_model.is_working(r, timestamp)]
    if not available:
        return None  # No one is working now
    
    # 3. Busy filter
    free = [r for r in available 
            if not resource_pool.is_busy(r)]
    if not free:
        return None  # Everyone is busy
    
    # 4. Selection (first available)
    return free[0]
```

### 3. Next Activity Sequence Alignment (Benchmarking)
```python
def align_next_activity_sequences(original_log, simulated_log):
    y_true, y_pred = [], []
    
    # Get cases sorted by ID
    orig_cases = sorted(original_log['case_id'].unique())
    sim_cases = sorted(simulated_log['case_id'].unique())
    
    # Compare up to minimum number of cases
    for i in range(min(len(orig_cases), len(sim_cases))):
        orig_trace = get_trace(original_log, orig_cases[i])
        sim_trace = get_trace(simulated_log, sim_cases[i])
        
        # Align by event position
        min_len = min(len(orig_trace), len(sim_trace))
        for j in range(min_len - 1):
            y_true.append(orig_trace[j + 1])  # Actual next
            y_pred.append(sim_trace[j + 1])   # Simulated next
    
    return y_true, y_pred
```

---

## Performance Considerations

### Simulation Speed
- **Typical:** 100 cases in ~30-60 seconds
- **Bottlenecks:**
  - LSTM forward pass (next activity prediction)
  - Resource allocation checks
  - Waiting queue processing

### Optimization Tips
1. **Batch Processing:** Pre-generate case attributes
2. **Model Caching:** Cache LSTM predictions for common sequences
3. **Resource Pooling:** Limit resource availability checks
4. **Event Pruning:** Set max_activities_per_case to prevent infinite loops

### Memory Usage
- **Event Log:** ~100 bytes per event
- **Case State:** ~500 bytes per active case
- **Models in Memory:** ~50-200 MB (LSTM + other models)

---

## Troubleshooting

### Common Issues

#### 1. "No eligible resources" warnings
**Cause:** Permission model doesn't include mappings for all activities
**Solution:** Update resource_permissions/ to include all activities

#### 2. Low Next Activity Prediction accuracy
**Cause:** Simulated traces diverge significantly from original
**Solution:** 
- Retrain LSTM with more data
- Adjust gateway branching probabilities
- Review case attribute predictions

#### 3. Stuck cases (cases waiting indefinitely)
**Cause:** Resource availability gaps or permission conflicts
**Solution:**
- Check availability model for coverage
- Verify at least one resource is available 24/7 for critical activities
- Review waiting queue statistics

#### 4. Timestamp parsing errors
**Cause:** Mixed timestamp formats in CSV files
**Solution:** Use `pd.to_datetime(format='mixed')` in data loading

---

## Future Enhancements

### Potential Improvements
1. **Advanced Resource Selection:**
   - Skill-based allocation
   - Workload balancing
   - Priority queuing

2. **Better Next Activity Prediction:**
   - Transformer-based models
   - Attention mechanisms
   - Multi-task learning

3. **Real-time Simulation:**
   - Stream processing integration
   - Continuous model updates
   - Live dashboards

4. **Extended Benchmarking:**
   - Statistical significance tests
   - Conformance checking
   - Root cause analysis for discrepancies

5. **Optimization:**
   - Parallel case simulation
   - GPU acceleration for ML models
   - Distributed simulation

---

## References

### Papers & Resources
1. **BPIC 2017:** https://www.win.tue.nl/bpi/doku.php?id=2017:challenge
2. **PM4Py:** https://pm4py.fit.fraunhofer.de/
3. **Discrete Event Simulation:** Banks, J. et al. "Discrete-Event System Simulation"
4. **Process Mining:** van der Aalst, W. "Process Mining: Data Science in Action"

### Dataset
- **BPI Challenge 2017:** Loan Application Process
- **Source:** Dutch Financial Institution
- **Period:** 2016-2017
- **Cases:** 31,509
- **Events:** 1,202,267
- **Activities:** 26

---

## Contributors & Acknowledgments

**Project:** TUM Master - Business Process Prediction, Simulation and Optimization - Assignment 2

**Components:**
- Discrete Event Simulation Engine
- ML-based Predictors (LSTM, Gradient Boosting)
- Resource Management System
- Benchmarking Framework

**Key Technologies:**
- Python 3.12+
- PM4Py (Process Mining)
- TensorFlow/Keras (LSTM)
- scikit-learn (ML models)
- pandas (Data processing)

---

## Version History

### v1.0 (Current)
- ✅ Basic DES engine with event queue
- ✅ LSTM-based next activity prediction
- ✅ Processing time prediction
- ✅ Case arrival time prediction
- ✅ Case attribute prediction
- ✅ Resource allocation with permissions & availability
- ✅ Resource pool with waiting queues
- ✅ XES/CSV export
- ✅ Comprehensive benchmarking
- ✅ **Next Activity Prediction quality metrics** ⭐

---

**Last Updated:** January 14, 2026
**Status:** Production Ready ✅
