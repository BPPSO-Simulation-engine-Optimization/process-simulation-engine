# process-simulation-engine
A data-driven process simulation engine supporting discrete-event simulation, probabilistic processing times, next-activity prediction, resource modeling, and event-log generation. Built for coursework involving contextual spawn rates, advanced branching prediction, and extendable ML-based components.

Here is a clean, copy-paste-ready **README.md** snippet in Markdown based on what you provided:


# Setup

## 1. Activate Virtual Environment
```bat
venv\Scripts\activate
````

Or create one if it doesn't exist:

```bat
py -m venv .venv
```
## 2. Install pm4py
```bat
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
