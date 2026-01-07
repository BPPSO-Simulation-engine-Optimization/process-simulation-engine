# Lifecycle Filtering Design Document

> **Status**: Draft
> **Last Updated**: 2026-01-07
> **Purpose**: Document findings and track changes for lifecycle transition filtering in Next Activity Prediction

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Current Implementation Analysis](#current-implementation-analysis)
3. [Event Log Analysis](#event-log-analysis)
4. [Resource Behavior Analysis](#resource-behavior-analysis)
5. [Activity Classification](#activity-classification)
6. [Proposed Solution (Option C - Hybrid)](#proposed-solution-option-c---hybrid)
7. [Implementation Plan](#implementation-plan)
8. [Simulation Engine Integration](#simulation-engine-integration)
9. [Processing Time Prediction Integration](#processing-time-prediction-integration)
10. [Resolved Questions](#resolved-questions)
11. [Change Log](#change-log)

---

## Problem Statement

### Context
The Next Activity Prediction models are experiencing performance issues. Investigation revealed that the models are trained on the full event log including all lifecycle transitions, which introduces noise and patterns that aren't useful for predicting actual process flow.

### Goal
Filter event log to only **start** and **complete** lifecycle transitions to:
1. Reduce noise in training sequences
2. Capture accurate **processing times** for resource optimization
3. Improve model prediction accuracy
4. Enable realistic simulation with proper resource allocation/deallocation

---

## Current Implementation Analysis

### Architecture Overview
- **43 per-decision-point LSTM models** trained on BPI Challenge 2017 data
- **Inputs**: activity sequence + duration sequence + resource sequence + context features
- **Output**: next activity classification (probability distribution)

### Key Files
| File | Purpose |
|------|---------|
| `advanced/preprocessing/training_data.py` | `TrainingDataGenerator` - creates sequences from event log |
| `advanced/models/lstm.py` | `LSTMPredictor` - LSTM model architecture |
| `advanced/models/encoders.py` | `SequenceEncoder`, `ContextEncoder` - feature encoding |
| `advanced/notebooks/01_train.ipynb` | Training pipeline |
| `advanced/api.py` | Prediction API for inference |
| `advanced/simulation.py` | Simulation utilities |

### Current Duration Calculation
```python
# In training_data.py - _compute_durations()
# Currently computes INTER-EVENT time (time between consecutive events)
delta = (timestamps[j] - timestamps[j-1]).total_seconds()
```

**Problem**: This calculates time between any two consecutive events, not the actual processing time of an activity (start → complete).

### Current Lifecycle Handling
- **None** - all events treated equally regardless of lifecycle transition
- No filtering applied
- Sequences include suspend, resume, schedule, abort events

---

## Event Log Analysis

### Lifecycle Transition Distribution
| Transition | Count | Percentage |
|------------|-------|------------|
| complete | 475,306 | 39.4% |
| suspend | 215,402 | 17.9% |
| schedule | 149,104 | 12.4% |
| start | 128,227 | 10.6% |
| resume | 127,160 | 10.5% |
| ate_abort | 85,224 | 7.1% |
| withdraw | 21,844 | 1.8% |
| **Total** | **~1.2M** | **100%** |

### Key Observations
1. **Imbalance**: `complete` events (475k) significantly outnumber `start` events (128k)
2. **Not all activities have start events**: Some activities only log completion
3. **Noise sources**: suspend/resume cycles, schedule events, aborts add ~50% of events
4. **Impact**: Model learns patterns like "suspend → resume" which don't help predict actual process flow

---

## Resource Behavior Analysis

### Critical Finding: Suspend → Resume Resource Handoff

**Key Finding**: When work is suspended, a **different resource** picks it up 83% of the time.

| Metric | Value |
|--------|-------|
| Total suspend→resume pairs | 129,432 |
| Same resource resumes | 22,172 (17.1%) |
| Different resource resumes | 107,260 (82.9%) |

### Resource Consistency by Transition Type

| Transition | Same Resource | Different Resource | Interpretation |
|------------|---------------|--------------------|-----------------------------------------|
| start → suspend | 99.4% | 0.6% | Worker suspends own started work |
| resume → suspend | 99.6% | 0.4% | Worker suspends after resuming |
| resume → complete | 99.4% | 0.6% | Worker completes after resuming |
| schedule → start | 85.2% | 14.8% | Scheduled work started by same resource |
| **suspend → resume** | **17.4%** | **82.6%** | **Work handed off to different resource** |
| suspend → ate_abort | 3.9% | 96.1% | Aborted by different resource |

### Interpretation: Work Queue Model

The BPI Challenge 2017 process operates with a **work queue model**:
- **Within a work session**: Resources are highly consistent (~99%)
- **Between sessions (handoffs)**: Resources change frequently (suspend→resume only 17% consistent)

**Implication**: Suspended tasks go back into a pool and are picked up by whichever resource is available next, rather than being "reserved" for the original resource.

**For simulation**: This means we should NOT assume the same resource will complete work it started if there was a suspension. The simulation engine should model this handoff behavior.

---

## Activity Classification

### Analysis Results

Activities in BPI Challenge 2017 fall into two categories based on lifecycle patterns:

#### Activities WITH Start Events (Work Activities)
These are "W_" and "A_" prefix activities that represent actual work:
- Have both `start` and `complete` lifecycle transitions
- Processing time = `complete_timestamp - start_timestamp`
- Consume resources during execution
- Examples: `W_Complete application`, `A_Submitted`, `W_Call after offers`

#### Activities WITHOUT Start Events (Instant/Automatic)
These are "O_" prefix activities representing system events and status changes:

| Activity | Type | Description |
|----------|------|-------------|
| O_Create Offer | System/Automatic | System creates offer |
| O_Created | System/Automatic | Offer creation logged |
| O_Sent (mail and online) | Communication | Notification sent (instant) |
| O_Sent (online only) | Communication | Notification sent (instant) |
| O_Accepted | Status Update | Customer accepts (instant state change) |
| O_Cancelled | Status Update | Offer cancelled (instant) |
| O_Refused | Status Update | Offer refused (instant) |
| O_Returned | Status Update | Offer returned (instant) |

**Key characteristics**:
- Only have `complete` transitions (no `start`)
- Intrinsically instantaneous operations
- Execute atomically with no processing time
- Should be treated as **0-duration** in simulation

### Handling Strategy

| Activity Type | Has Start? | Duration Calculation | Resource Consumption |
|---------------|------------|---------------------|---------------------|
| W_* / A_* activities | Yes | `complete - start` | Yes (allocate → free) |
| O_* activities | No | 0 (instant) | No (or minimal) |

---

## Proposed Solution (Option C - Hybrid)

### Approach
1. **Filter** event log to keep only `start` and `complete` lifecycle transitions
2. **Compute processing time** for activities that have both start and complete events
3. **Treat O_* activities as instant** (0 duration) since they lack start events
4. **Collapse** start/complete pairs into single activity instances with duration attribute

### Sequence Transformation
```
BEFORE (raw event log):
A_schedule → A_start → A_suspend → A_resume → A_complete → O_Created → B_schedule → B_start → B_complete

AFTER (filtered + collapsed):
A (duration: complete - start) → O_Created (duration: 0) → B (duration: complete - start)
```

### Duration Handling Strategy

| Scenario | Duration Calculation |
|----------|---------------------|
| Activity has both start + complete | `complete_timestamp - start_timestamp` |
| Activity has only complete (O_* activities) | **0 seconds** (instant/automatic) |
| Activity has only start (incomplete) | Exclude from training |

### Benefits
- **Cleaner sequences**: ~60% reduction in events (from ~1.2M to ~600k start+complete)
- **Accurate processing times**: True activity durations for resource optimization
- **Better predictions**: Model learns actual process flow, not lifecycle mechanics
- **Simulation-ready**: Duration represents actual work time, not queue/wait time

---

## Implementation Plan

### Phase 1: Preprocessing Changes ✅ COMPLETE
**Files created/modified**:
- `advanced/preprocessing/lifecycle_filter.py` (NEW) - `LifecycleFilter` class
- `advanced/preprocessing/training_data.py` (MODIFIED) - Added processing_time support
- `advanced/preprocessing/__init__.py` (MODIFIED) - Export LifecycleFilter
- `advanced/notebooks/01_train.ipynb` (MODIFIED) - Uses lifecycle filtering

**Completed tasks**:
- [x] Add lifecycle filtering function to keep only `start` and `complete`
- [x] Implement start/complete event pairing logic (same case, same activity)
- [x] Add processing time calculation (`complete - start`)
- [x] Handle O_* activities as instant (duration = 0)
- [x] Update `TrainingDataGenerator` to use filtered events

### Phase 2: Data Validation
- [ ] Verify event pairing correctness (same case, same activity instance)
- [ ] Validate O_* activities have no start events (confirm classification)
- [ ] Compute duration statistics per activity type
- [ ] Compare sequence lengths before/after filtering

### Phase 3: Model Retraining
**File**: `advanced/notebooks/01_train.ipynb`

- [ ] Regenerate training data with filtered events
- [ ] Retrain all 43 decision point models
- [ ] Compare accuracy metrics vs baseline (current models)
- [ ] Validate on holdout set

### Phase 4: Integration Testing
- [ ] Ensure Next Activity Prediction works with simulation engine
- [ ] Verify processing time prediction receives correct inputs
- [ ] End-to-end simulation test with filtered data

---

## Simulation Engine Integration

### Engine Architecture Overview

The simulation engine (`simulation/engine.py`) is a **Discrete Event Simulation (DES)**:
- Event-driven time advancement using priority queue
- Three event types: `CASE_ARRIVAL`, `ACTIVITY_COMPLETE`, `CASE_END`
- Resource pool with waiting queues for contention handling

### Concurrency Model

**Yes, the simulation supports concurrent activities**:
- Multiple resources can execute different activities simultaneously
- Each resource can only execute ONE activity at a time
- `ResourcePool._busy_resources` tracks: `resource_id → (busy_until, case_id, activity)`
- Waiting queues (per-activity FIFO) handle resource unavailability

### Required Interfaces

#### NextActivityPredictor Protocol
```python
class NextActivityPredictor(Protocol):
    def predict(self, case_state: CaseState) -> tuple[str, bool]:
        """
        Args:
            case_state: Contains activity_history, case_type, application_type, etc.
        Returns:
            Tuple of (next_activity_name, is_case_ended)
        """
```

#### ProcessingTimePrediction Protocol
```python
def predict(
    self,
    prev_activity: str,
    prev_lifecycle: str,      # Currently hardcoded to "complete"
    curr_activity: str,
    curr_lifecycle: str,      # Currently hardcoded to "complete"
    context: Optional[Dict]
) -> float:  # Returns seconds
```

### Simulation Flow
```
1. Case arrives (CASE_ARRIVAL event)
2. NextActivityPredictor.predict(case) → next activity
3. Try to allocate resource (or queue if unavailable)
4. ProcessingTimePrediction.predict() → duration in seconds
5. Mark resource busy, schedule ACTIVITY_COMPLETE at current_time + duration
6. On ACTIVITY_COMPLETE: release resource, predict next activity
7. Repeat until case ends
```

### Data Flow
```
Event Log (XES)
    ↓ [Lifecycle Filtering - THIS WORK]
Filtered Events (start + complete only)
    ↓ [Pairing + Duration Calc]
Activity Instances with Processing Times
    ↓ [Sequence Generation]
Training Data per Decision Point
    ↓ [LSTM Training]
Trained Models
    ↓
    ├─→ Next Activity Prediction API
    │       ↓
    └─→ Processing Time Prediction (separate module)
            ↓
        Simulation Engine
            ↓
        Resource Optimization
```

---

## Processing Time Prediction Integration

### Assessment: Keep Existing Module

The `processing_time_prediction/` module is **well-designed and should be kept**:

#### Available Methods
1. **Distribution** - Samples from fitted log-normal distributions per transition (lightweight)
2. **ML** - Random Forest with feature engineering (balanced accuracy/speed)
3. **Probabilistic ML** - LSTM with uncertainty estimation (most complex)

**Recommendation**: Use **distribution** or **ml** method. The LSTM method adds TensorFlow dependency for marginal benefit.

#### Current Integration (simulation/engine.py:734-742)
```python
processing_seconds = self._processing_time.predict(
    prev_activity=prev_activity,
    prev_lifecycle="complete",    # Currently hardcoded
    curr_activity=activity,
    curr_lifecycle="complete",    # Currently hardcoded
    context=context,
)
```

#### Context Features Used
- Temporal: hour, weekday, month, day_of_year
- Case: LoanGoal, ApplicationType, event_position_in_case, case_duration_so_far
- Resource: previous resource, current resource
- Flags: Accepted, Selected

### No Changes Needed
The processing time module already handles duration prediction. This lifecycle filtering work focuses on **Next Activity Prediction** - the two modules are complementary.

---

## Resolved Questions

### 1. Activities without start events
**Resolution**: O_* activities (Offer-related) are instant/automatic events. Treat them as **0-duration**. They represent status changes, not work activities.

### 2. Duration distribution
**Resolution**: Use existing `processing_time_prediction/` module which already models distributions. Next Activity Prediction does not need to predict duration.

### 3. Concurrent activities
**Resolution**: Simulation engine already supports concurrency. Multiple resources can work in parallel; each resource handles one activity at a time.

### 4. Resource handoff on suspend/resume
**Resolution**: 83% of suspended work is picked up by a different resource. Simulation should model this work queue behavior, not assume same-resource completion.

### 5. Model architecture for duration
**Resolution**: Duration prediction is a **separate module** (processing_time_prediction/). Keep separation of concerns - Next Activity Prediction predicts *what*, Processing Time predicts *how long*.

---

## Change Log

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-07 | - | Initial document created |
| | | - Documented current implementation |
| | | - Analyzed lifecycle transition distribution |
| | | - Defined Option C (Hybrid) approach |
| | | - Outlined implementation plan |
| 2026-01-07 | - | Major update with exploration findings |
| | | - Added Resource Behavior Analysis (suspend→resume handoff) |
| | | - Added Activity Classification (O_* vs W_*/A_*) |
| | | - Documented simulation engine integration |
| | | - Assessed processing_time_prediction module (keep as-is) |
| | | - Resolved open questions |
| 2026-01-07 | - | **Phase 1 Implementation Complete** |
| | | - Created `LifecycleFilter` class in `lifecycle_filter.py` |
| | | - Updated `TrainingDataGenerator` for processing_time support |
| | | - Modified training notebook to use lifecycle filtering |

---

## References
- BPI Challenge 2017 Dataset
- Current models: `advanced/models_lstm_new/`
- Training notebook: `advanced/notebooks/01_train.ipynb`
- Simulation engine: `simulation/engine.py`
- Processing time prediction: `processing_time_prediction/ProcessingTimePredictionClass.py`
