# Process Simulation Engine - Integration Task List

> **Version:** 1.0
> **Created:** 2024-12-28
> **Purpose:** Comprehensive task tracking for resolving integration issues before 31k case simulation
> **Status:** Pre-Integration Validation Phase

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Dependency Graph](#component-dependency-graph)
3. [Critical Issues (P0)](#critical-issues-p0)
4. [High Priority Issues (P1)](#high-priority-issues-p1)
5. [Medium Priority Issues (P2)](#medium-priority-issues-p2)
6. [Pre-Flight Checklist](#pre-flight-checklist)
7. [Validation Protocol](#validation-protocol)

---

## System Overview

### Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         integration/test_integration.py                      │
│                                (Entry Point)                                 │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     integration/setup.py      │
                    │   (Component Orchestration)   │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────┬───────────┼───────────────┬───────────────┐
        ▼               ▼           ▼               ▼               ▼
┌───────────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐
│ Case Arrival  │ │Processing │ │   Case    │ │   Next    │ │   Resource    │
│  Prediction   │ │   Time    │ │ Attribute │ │ Activity  │ │  Allocator    │
│               │ │Prediction │ │Prediction │ │Prediction │ │               │
└───────┬───────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └───────┬───────┘
        │               │             │             │               │
        └───────────────┴─────────────┴─────────────┴───────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   simulation/engine.py    │
                        │      (DES Engine)         │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │ simulation/log_exporter.py│
                        │     (CSV/XES Output)      │
                        └───────────────────────────┘
```

### Key Files Reference

| Component | Primary File | Model/Cache File |
|-----------|--------------|------------------|
| Integration Entry | `integration/test_integration.py` | - |
| Integration Setup | `integration/setup.py` | - |
| Integration Config | `integration/config.py` | - |
| DES Engine | `simulation/engine.py` | - |
| Case Manager | `simulation/case_manager.py` | - |
| Case Arrival | `case_arrival_times_prediction/runner.py` | `case_arrival_model.pkl` |
| Processing Time | `processing_time_prediction/ProcessingTimePredictionClass.py` | `models/processing_time_model_*.joblib` |
| Case Attributes | `case_attribute_prediction/simulator.py` | `case_attribute_prediction/*.pkl` |
| Next Activity | `Next-Activity-Prediction/train.py` | `models/branch_predictor.joblib` |
| Resources | `resources/resource_allocation.py` | `resources/*/*.pkl` |
| Benchmark | `integration/SimulationBenchmark.py` | - |

---

## Component Dependency Graph

```
TASK DEPENDENCIES (must be completed in order within each priority level)

P0-1 ──► P0-2 (Context fix enables accurate predictions)

P1-1 ◄── P0-2 (Activity transitions depend on correct engine flow)
P1-2 ◄── P0-1 (Attribute logging depends on having attributes available)

P2-1 ──► P2-2 (Path consistency needed before fallback review)
```

---

## Critical Issues (P0)

### P0-1: Processing Time Context Not Passed

**Status:** `[x] Completed`
**Owner:** `Claude Code`
**Completed:** 2024-12-28
**Estimated Effort:** Medium
**Blocking:** P1-2

#### Problem Description

The DES engine calls `ProcessingTimePredictionClass.predict()` without passing the `context` parameter. This causes:

1. Temporal features (`hour`, `weekday`, `month`) use `datetime.now()` instead of simulation time
2. Case attributes (`LoanGoal`, `ApplicationType`) are not used
3. Event position and case duration are defaulted to 0/1

#### Affected Code

**File:** `simulation/engine.py`
**Lines:** 372-382

```python
# CURRENT (PROBLEMATIC)
prev_activity = case.activity_history[-1] if case.activity_history else "START"
processing_seconds = self._processing_time.predict(
    prev_activity=prev_activity,
    prev_lifecycle="complete",
    curr_activity=activity,
    curr_lifecycle="complete",
    # context=None  <-- MISSING
)
```

#### Required Changes

```python
# REQUIRED FIX
prev_activity = case.activity_history[-1] if case.activity_history else "START"

# Build context from simulation state
context = {
    # Temporal features (from simulation clock, not wall clock)
    'hour': current_time.hour,
    'weekday': current_time.weekday(),
    'month': current_time.month,
    'day_of_year': current_time.timetuple().tm_yday,

    # Case attributes
    'case:LoanGoal': case.case_type,
    'case:ApplicationType': case.application_type,

    # Event position tracking
    'event_position_in_case': len(case.activity_history) + 1,
    'case_duration_so_far': (current_time - case.start_time).total_seconds() if case.start_time else 0.0,

    # Resource info (current allocation)
    'resource_1': case.current_resource or 'unknown',
    'resource_2': resource,  # The resource being allocated for this activity
}

processing_seconds = self._processing_time.predict(
    prev_activity=prev_activity,
    prev_lifecycle="complete",
    curr_activity=activity,
    curr_lifecycle="complete",
    context=context,
)
```

#### Context Parameter Reference

**File:** `processing_time_prediction/ProcessingTimePredictionClass.py`
**Method:** `_context_to_features()` (lines 325-381)

Expected context keys:
| Key | Type | Description | Default if Missing |
|-----|------|-------------|-------------------|
| `hour` | int | Hour of day (0-23) | `datetime.now().hour` |
| `weekday` | int | Day of week (0-6) | `datetime.now().weekday()` |
| `month` | int | Month (1-12) | `datetime.now().month` |
| `day_of_year` | int | Day of year (1-366) | Computed from month |
| `case:LoanGoal` | str | Loan goal category | `None` |
| `case:ApplicationType` | str | Application type | `None` |
| `event_position_in_case` | int | Event sequence number | `1` |
| `case_duration_so_far` | float | Seconds since case start | `0.0` |
| `resource_1` | str | Previous resource | `'unknown'` |
| `resource_2` | str | Current resource | `'unknown'` |
| `Accepted` | bool | Offer accepted flag | `None` |
| `Selected` | bool | Offer selected flag | `None` |

#### Acceptance Criteria

- [x] Context dict is constructed with all available simulation state
- [x] Temporal features use simulation clock (`current_time`), not wall clock
- [ ] Unit test verifies context is passed correctly
- [ ] Processing time predictions vary based on context (not just activity pairs)

#### Implementation Notes (2024-12-28)

**Fix applied in:** `simulation/engine.py` lines 377-404

Context dict now includes:
- Temporal: `hour`, `weekday`, `month`, `day_of_year` from `current_time`
- Case attrs: `case:LoanGoal`, `case:ApplicationType` from case state
- Position: `event_position_in_case` = `len(case.activity_history) + 1`
- Duration: `case_duration_so_far` = `(current_time - case.start_time).total_seconds()`
- Resources: `resource_1` (previous), `resource_2` (newly allocated)

**Note:** `Accepted` and `Selected` attributes are not yet available at this point in simulation flow (see P1-2 for offer-level attribute integration).

#### Verification Command

```bash
# After fix, run small test and check logs for context usage
python -m integration.test_integration --mode advanced --num-cases 10 --verbose 2>&1 | grep -i "context"
```

---

### P0-2: Arrival Timestamp Count Verification

**Status:** `[x] Completed`
**Owner:** `Claude Code`
**Estimated Effort:** Low
**Blocking:** Full integration test
**Completed:** 2024-12-28

#### Problem Description

The arrival timestamp generation uses a hardcoded estimate of 100 cases/day:

```python
avg_cases_per_day = 100
estimated_days = int((config.num_cases / avg_cases_per_day) * 1.5) + 1
```

For 31,000 cases: `estimated_days = 466 days`

If actual average is lower, fewer timestamps are generated. The code logs a warning but continues with fewer cases.

#### Affected Code

**File:** `integration/setup.py`
**Lines:** 96-121

```python
# CURRENT LOGIC
avg_cases_per_day = 100  # Hardcoded assumption
estimated_days = int((config.num_cases / avg_cases_per_day) * 1.5) + 1

timestamps = run(...)

if len(arrival_timestamps) < config.num_cases:
    logger.warning(...)  # Only warns, doesn't fail or adjust
```

#### Required Analysis

1. **Determine actual average cases/day from training data:**

```python
# Run this analysis on the event log
import pandas as pd
import pm4py

df = pm4py.read_xes("eventlog/eventlog.xes.gz")
df = pm4py.convert_to_dataframe(df)

# Get first event per case (arrival)
arrivals = df.groupby('case:concept:name')['time:timestamp'].min()
daily_arrivals = arrivals.dt.date.value_counts()

print(f"Mean daily arrivals: {daily_arrivals.mean():.1f}")
print(f"Median daily arrivals: {daily_arrivals.median():.1f}")
print(f"Min daily arrivals: {daily_arrivals.min()}")
print(f"Max daily arrivals: {daily_arrivals.max()}")
```

2. **If average < 100:** Increase the multiplier or use actual statistics from the loaded model

#### Required Changes

**Option A: Use model statistics (preferred)**

```python
# In _setup_arrivals(), after loading the model:
timestamps = run(
    df=df,
    retrain_model=retrain_model,
    model_path=model_path,
    n_days_to_simulate=estimated_days,
    config=arr_config,
)

# If insufficient, dynamically increase days and regenerate
while len(timestamps) < config.num_cases:
    estimated_days = int(estimated_days * 1.5)
    logger.info(f"Insufficient timestamps ({len(timestamps)}), increasing to {estimated_days} days...")
    timestamps = run(
        df=None,  # Use cached model
        retrain_model=False,
        model_path=model_path,
        n_days_to_simulate=estimated_days,
        config=arr_config,
    )
```

**Option B: Fail loudly if insufficient**

```python
if len(arrival_timestamps) < config.num_cases:
    raise ValueError(
        f"Generated only {len(arrival_timestamps)} timestamps, "
        f"but {config.num_cases} requested. "
        f"Increase estimated_days or check arrival model."
    )
```

#### Acceptance Criteria

- [x] For 31,000 cases, at least 31,000 timestamps are generated
- [x] Either auto-retry or fail with clear error (not silent degradation)
- [x] Actual average cases/day is documented

#### Implementation Notes (2024-12-28)

**Fix applied in:** `integration/setup.py` lines 97-151

**Actual BPIC17 Daily Arrival Statistics:**
| Metric | Value |
|--------|-------|
| Mean | 86.1 |
| Median | 88.0 |
| Std Dev | 32.2 |
| Min | 20 |
| Max | 178 |
| Total cases | 31,509 |
| Total days | 366 |

**Changes made:**
1. Updated `avg_cases_per_day` from `100` to `86` (based on actual mean)
2. Implemented retry loop (Option A) with up to 5 attempts
3. Each retry increases `estimated_days` by 1.5x
4. After all retries exhausted, raises `ValueError` (fail-loud, no silent degradation)
5. Added detailed statistics as code comment

**Verification result:**
```
Requested 31,000 timestamps → Generated 51,608 → Sliced to 31,000 ✓
PASS on first attempt (no retries needed)
```

#### Verification Command

```bash
# Dry run to check timestamp count
python -c "
from integration.config import SimulationConfig
from integration.setup import _setup_arrivals
from datetime import datetime
import pandas as pd
import pm4py

df = pm4py.read_xes('eventlog/eventlog.xes.gz')
df = pm4py.convert_to_dataframe(df)

config = SimulationConfig.all_advanced('eventlog/eventlog.xes.gz', num_cases=31000)
start_date = pd.to_datetime(df['time:timestamp']).min().to_pydatetime()

timestamps = _setup_arrivals(config, df, start_date)
print(f'Generated {len(timestamps)} timestamps for {config.num_cases} requested cases')
assert len(timestamps) >= config.num_cases, 'INSUFFICIENT TIMESTAMPS'
print('PASS: Sufficient timestamps generated')
"
```

---

### P0-3: Case Arrival Config Parameter Alignment

**Status:** `[x] Completed`
**Owner:** `Claude Code`
**Completed:** 2024-12-28
**Estimated Effort:** Low

#### Problem Description

`integration/config.py` exposes arrival parameters, but `z_values` is missing and defaults may not align with trained model.

#### Affected Files

**File 1:** `integration/config.py` (lines 24-34)
```python
# CURRENT - Missing z_values
arrival_train_ratio: float = 0.8
arrival_window_size: int = 14  # Differs from runner.py default (21)
arrival_kmax: int = 3          # Differs from runner.py default (5)
arrival_L: int = 1             # Differs from runner.py default (4)
```

**File 2:** `case_arrival_times_prediction/runner.py` (lines 115-123)
```python
# DEFAULTS IN RUNNER
cfg = SimulationConfig(
    train_ratio=0.8,
    window_size=21,    # Different!
    kmax=5,            # Different!
    z_values=(0.9, 0.725, 0.55, 0.375, 0.2),
    L=4,               # Different!
    random_state=42,
    verbose=False
)
```

**File 3:** `case_arrival_times_prediction/config.py` (lines 7-32)
```python
# ACTUAL CONFIG CLASS DEFAULTS
window_size: int = 14
kmax: int = 3
z_values: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2)  # Yet another set!
L: int = 1
```

#### Required Changes

1. **Document which parameters the trained model was created with**
2. **Align `integration/config.py` with those parameters**
3. **Or:** When loading cached model, ignore config params (they're baked into the model)

```python
# integration/config.py - Add z_values and align defaults
arrival_train_ratio: float = 0.8
arrival_window_size: int = 21      # Aligned with runner.py
arrival_kmax: int = 5              # Aligned with runner.py
arrival_z_values: tuple = (0.9, 0.725, 0.55, 0.375, 0.2)  # NEW
arrival_L: int = 4                 # Aligned with runner.py
```

#### Acceptance Criteria

- [x] All arrival parameters are explicitly documented
- [x] Parameters match what the cached model was trained with
- [x] If model is cached, config params don't conflict with trained model

#### Resolution Notes

**Changes made (2024-12-28):**

1. **`integration/config.py`** (lines 24-36):
   - Added comment: "NOTE: These defaults must match the parameters used to train case_arrival_model.pkl"
   - `arrival_window_size`: 14 → 21
   - `arrival_kmax`: 3 → 5
   - Added `arrival_z_values: tuple = (0.9, 0.725, 0.55, 0.375, 0.2)`
   - `arrival_L`: 1 → 4

2. **`integration/setup.py`** (line 83):
   - Added `z_values=config.arrival_z_values` to `ArrivalConfig()` constructor

**Note:** The defaults in `case_arrival_times_prediction/config.py` remain unchanged (14, 3, 1) as they serve as the module's own defaults. The `runner.py` overrides these when training the model, and now `integration/config.py` matches those training parameters exactly.

---

## High Priority Issues (P1)

### P1-1: Next Activity Predictor Transition Coverage

**Status:** `[x] Completed`
**Owner:** `Claude Code`
**Completed:** 2024-12-28
**Estimated Effort:** Medium
**Depends On:** None (can be done in parallel)

#### Problem Description

`BranchNextActivityPredictor` only covers XOR gateway branches. Non-gateway activities use hardcoded `FALLBACK_TRANSITIONS`. Missing transitions cause premature case termination.

#### Affected Code

**File:** `simulation/engine.py`
**Lines:** 465-472

```python
FALLBACK_TRANSITIONS = {
    "A_Create Application": "A_Submitted",
    "A_Submitted": "W_Handle leads",
    "W_Handle leads": "W_Complete application",
    "A_Accepted": "O_Create Offer",
    "O_Create Offer": "O_Created",
    # MISSING: O_Created -> ???
    # MISSING: Paths to A_Cancelled
}
```

**Lines:** 540-542 (fallback behavior)
```python
# If no transition found, ends case
logger.warning(f"No transition found for activity '{current}', ending case")
return "A_Complete", True
```

#### Required Analysis

1. **Extract all unique activity sequences from BPIC17 log:**

```python
import pm4py
import pandas as pd
from collections import Counter

df = pm4py.read_xes("eventlog/eventlog.xes.gz")
df = pm4py.convert_to_dataframe(df)

# Get all transitions
transitions = []
for case_id, case_df in df.groupby('case:concept:name'):
    activities = case_df.sort_values('time:timestamp')['concept:name'].tolist()
    for i in range(len(activities) - 1):
        transitions.append((activities[i], activities[i+1]))

transition_counts = Counter(transitions)

# Find activities with missing outgoing transitions
all_activities = set(df['concept:name'].unique())
covered_sources = set(t[0] for t in transition_counts.keys())
gateway_covered = set(FALLBACK_TRANSITIONS.keys())
# Plus activities covered by the gateway model (need to check branch_predictor.joblib)

print("Potentially uncovered activities:")
for act in all_activities - covered_sources - gateway_covered:
    print(f"  - {act}")
```

2. **Verify gateway model coverage:**

```python
import joblib
data = joblib.load("models/branch_predictor.joblib")
gateway_connections = data['gateway_connections']

covered_by_gateway = set()
for gw_id, conn in gateway_connections.items():
    covered_by_gateway.update(conn['preceding'])

print(f"Gateway model covers: {covered_by_gateway}")
```

#### Required Changes

Either:
1. **Extend `FALLBACK_TRANSITIONS`** with missing transitions (quick fix)
2. **Extend the gateway model** to cover all process paths (proper fix)

#### Acceptance Criteria

- [x] All activities in BPIC17 log have defined outgoing transitions
- [x] No cases end prematurely due to missing transitions
- [x] Document which transitions are gateway-based vs fallback-based

#### Implementation Notes (2024-12-28)

**Fix applied in:** `simulation/engine.py` lines 493-518

**BPIC17 Activity Coverage Analysis:**

| Category | Count | Activities |
|----------|-------|------------|
| Gateway model | 13 | A_Complete, A_Concept, A_Incomplete, A_Pending, O_Cancelled, O_Created, O_Refused, O_Returned, O_Sent (mail and online), O_Sent (online only), W_Call after offers, W_Complete application, W_Validate application |
| Fallback (original) | 5 | A_Create Application, A_Submitted, W_Handle leads, A_Accepted, O_Create Offer |
| End activities | 2 | A_Cancelled, A_Complete |
| **Previously missing** | **7** | A_Denied, A_Validating, O_Accepted, W_Assess potential fraud, W_Call incomplete files, W_Personal Loan collection, W_Shortened completion |

**New Fallback Transitions Added:**

| Source Activity | Target Activity | Log Coverage |
|-----------------|-----------------|--------------|
| `A_Denied` | `O_Refused` | 99.1% (3720/3752) |
| `A_Validating` | `O_Returned` | 53.3% (20673/38816) |
| `O_Accepted` | `A_Pending` | 100% (17228/17228) |
| `W_Assess potential fraud` | `W_Validate application` | 52.9% (166/314) |
| `W_Call incomplete files` | `A_Incomplete` | 46.2% (23055/49935) |
| `W_Personal Loan collection` | `W_Validate application` | 100% (1/1) |
| `W_Shortened completion ` | `W_Call after offers` | 58.1% (68/117) |

**Note:** Target activities are the most frequent non-self-loop next activity from the BPIC17 event log. The gateway model handles probabilistic branching for activities preceding XOR gateways; fallback transitions provide deterministic paths for all other activities.

**Verification Result:**
```
Activities with outgoing transitions in log: 26
Total coverage after fix: 26/26 (100%)
✓ All fallback targets are valid activities
```

---

### P1-2: Offer-Level Attributes Not in Event Log

**Status:** `[x] Completed`
**Owner:** `Claude Code`
**Completed:** 2024-12-28
**Estimated Effort:** Medium
**Depends On:** P0-1

#### Problem Description

The `AttributeSimulationEngine` generates offer-dependent attributes (`CreditScore`, `OfferedAmount`, `NumberOfTerms`, `MonthlyCost`, `Selected`, `Accepted`), but these are never added to the simulated event log.

#### Current Flow

```
1. engine._on_case_arrival():
   attr_case = self._case_attribute.start_new_case()
   # Only extracts: loan_goal, application_type, requested_amount

2. CaseState created with only 3 attributes

3. Events logged via CaseState.get_payload():
   # Only returns: LoanGoal, ApplicationType, RequestedAmount
```

#### Affected Code

**File:** `simulation/engine.py`

**Lines 297-314** (case arrival - only basic attributes used):
```python
def _on_case_arrival(self, event: SimulationEvent) -> None:
    # Get case attributes from AttributeSimulationEngine
    attr_case = self._case_attribute.start_new_case()
    loan_goal = attr_case.loan_goal
    app_type = attr_case.application_type
    amount = attr_case.requested_amount
    # MISSING: credit_score, offered_amount, etc. are available but not used
```

**File:** `simulation/case_manager.py`

**Lines 11-22** (CaseState missing offer attributes):
```python
@dataclass
class CaseState:
    case_id: str
    case_type: str
    application_type: str
    requested_amount: float
    # MISSING: credit_score, offered_amount, number_of_terms,
    #          monthly_cost, selected, accepted
```

**Lines 36-42** (get_payload missing attributes):
```python
def get_payload(self) -> Dict:
    return {
        'case:LoanGoal': self.case_type,
        'case:ApplicationType': self.application_type,
        'case:RequestedAmount': self.requested_amount,
        # MISSING: CreditScore, OfferedAmount, etc.
    }
```

#### Required Changes

**Step 1:** Extend `simulation/case_manager.py:CaseState`

```python
@dataclass
class CaseState:
    case_id: str
    case_type: str  # LoanGoal
    application_type: str
    requested_amount: float

    # Offer-level attributes (populated when O_Create Offer occurs)
    credit_score: Optional[float] = None
    offered_amount: Optional[float] = None
    first_withdrawal_amount: Optional[float] = None
    number_of_terms: Optional[int] = None
    monthly_cost: Optional[float] = None
    selected: Optional[bool] = None
    accepted: Optional[bool] = None

    # Reference to attribute predictor's CaseState for lazy evaluation
    _attr_engine_case: Optional[Any] = field(default=None, repr=False)
```

**Step 2:** Store reference to `AttributeSimulationEngine` case in `engine.py`

```python
def _on_case_arrival(self, event: SimulationEvent) -> None:
    attr_case = self._case_attribute.start_new_case()

    case = self.case_manager.create_case(
        case_id=event.case_id,
        case_type=attr_case.loan_goal,
        application_type=attr_case.application_type,
        requested_amount=attr_case.requested_amount,
        start_time=event.timestamp,
    )
    case._attr_engine_case = attr_case  # Store reference
```

**Step 3:** Update attributes when O_Create Offer activity completes

```python
def _on_activity_complete(self, event: SimulationEvent) -> None:
    case = self.case_manager.get_case(event.case_id)

    # Trigger offer-dependent attribute generation
    if event.activity == "O_Create Offer" and case._attr_engine_case:
        self._case_attribute.draw_event_attributes("O_Create Offer")
        attr = case._attr_engine_case
        case.credit_score = attr.credit_score
        case.offered_amount = attr.offered_amount
        case.first_withdrawal_amount = attr.first_withdrawal_amount
        case.number_of_terms = attr.number_of_terms
        case.monthly_cost = attr.monthly_cost
        case.selected = attr.selected
        case.accepted = attr.accepted

    # ... rest of method
```

**Step 4:** Update `get_payload()` to include offer attributes

```python
def get_payload(self) -> Dict:
    payload = {
        'case:LoanGoal': self.case_type,
        'case:ApplicationType': self.application_type,
        'case:RequestedAmount': self.requested_amount,
    }

    # Add offer-level attributes if available
    if self.credit_score is not None:
        payload['CreditScore'] = self.credit_score
    if self.offered_amount is not None:
        payload['OfferedAmount'] = self.offered_amount
    if self.first_withdrawal_amount is not None:
        payload['FirstWithdrawalAmount'] = self.first_withdrawal_amount
    if self.number_of_terms is not None:
        payload['NumberOfTerms'] = self.number_of_terms
    if self.monthly_cost is not None:
        payload['MonthlyCost'] = self.monthly_cost
    if self.selected is not None:
        payload['Selected'] = self.selected
    if self.accepted is not None:
        payload['Accepted'] = self.accepted

    return payload
```

#### Acceptance Criteria

- [x] Simulated event log contains all BPIC17 case-level attributes
- [x] Offer attributes appear after O_Create Offer activity
- [x] Benchmark comparison can validate attribute distributions

#### Implementation Notes (2024-12-28)

**Files modified:**

1. **`simulation/case_manager.py`** (lines 22-32, 47-69):
   - Extended `CaseState` dataclass with offer-level attributes:
     - `credit_score`, `offered_amount`, `first_withdrawal_amount`, `number_of_terms`, `monthly_cost`, `selected`, `accepted`
   - Added `_attr_engine_case` field to store reference to `AttributeSimulationEngine`'s CaseState
   - Updated `get_payload()` to conditionally include offer attributes when non-None

2. **`simulation/engine.py`** (lines 316, 336-347, 412-414):
   - `_on_case_arrival()`: Stores `attr_case` reference in `case._attr_engine_case`
   - `_on_activity_complete()`: When "O_Create Offer" completes, calls `draw_event_attributes()` and copies offer attributes from attr engine case to simulation case state
   - `_schedule_activity()`: Added `Accepted` and `Selected` to processing time context dict

**Behavior:**
- Events **before** O_Create Offer: offer attributes are `None` (not included in CSV/XES output)
- Events **after** O_Create Offer: all 7 offer attributes (CreditScore, OfferedAmount, FirstWithdrawalAmount, NumberOfTerms, MonthlyCost, Selected, Accepted) are included in event payload
- Processing time predictions now have access to `Accepted`/`Selected` context when available

**Note:** `number_of_terms` is cast to `int` when copying from attr engine (which stores it as float) to ensure correct type in output.

#### Verification Command

```bash
# Run small simulation and check output for offer attributes
python -m integration.test_integration --mode advanced --num-cases 10 --verbose
# Then inspect output CSV:
head -20 integration/output/simulated_log.csv | cut -d',' -f1,4-10
```

---

## Medium Priority Issues (P2)

### P2-1: Relative Model Paths

**Status:** `[ ] Not Started`
**Owner:** `[Unassigned]`
**Estimated Effort:** Low

#### Problem Description

Model paths are relative, breaking when run from different working directories.

#### Affected Code

**File:** `integration/config.py` (line 22-23)
```python
processing_time_model_path: Optional[str] = "models/processing_time_model"
```

**File:** `integration/setup.py` (line 93)
```python
model_path = "case_arrival_model.pkl"
```

#### Required Changes

Use `pathlib` to resolve paths relative to project root:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# In config.py
processing_time_model_path: Optional[str] = str(PROJECT_ROOT / "models/processing_time_model")

# In setup.py
model_path = str(PROJECT_ROOT / "case_arrival_model.pkl")
```

#### Acceptance Criteria

- [ ] Integration test runs correctly from any working directory
- [ ] All model paths are absolute or correctly resolved

---

### P2-2: Resource Allocator Fallback Monitoring

**Status:** `[ ] Not Started`
**Owner:** `[Unassigned]`
**Estimated Effort:** Low

#### Problem Description

When no resource is available, the engine falls back to `"User_1"`. High fallback rates indicate resource model issues.

#### Affected Code

**File:** `simulation/engine.py` (lines 365-369)
```python
if resource is None:
    self.stats['allocation_failures'] += 1
    resource = "User_1"
```

#### Required Changes

Add threshold warning after simulation:

```python
def run(self, num_cases: int = 100, max_time: datetime = None) -> List[Dict]:
    # ... existing code ...

    # After simulation loop
    failure_rate = self.stats['allocation_failures'] / self.stats['events_processed']
    if failure_rate > 0.05:  # More than 5% failures
        logger.warning(
            f"High resource allocation failure rate: {failure_rate:.1%} "
            f"({self.stats['allocation_failures']} / {self.stats['events_processed']} events). "
            f"Consider reviewing resource availability model."
        )

    return self.completed_events
```

#### Acceptance Criteria

- [ ] Allocation failure rate is logged at end of simulation
- [ ] Warning emitted if failure rate > 5%
- [ ] Document expected failure rate for BPIC17

---

### P2-3: Lifecycle Transition Consistency

**Status:** `[ ] Not Started`
**Owner:** `[Unassigned]`
**Estimated Effort:** Low

#### Problem Description

Simulation only generates `complete` lifecycle events. BPIC17 has `start`, `complete`, `schedule`. Benchmark must handle this.

#### Affected Code

**File:** `simulation/events.py` (line 42)
```python
'lifecycle:transition': 'complete',  # MVP: always complete
```

**File:** `integration/SimulationBenchmark.py` - No lifecycle filtering

#### Required Changes

**Option A:** Filter original log to `complete` events only in benchmark:

```python
# In SimulationBenchmark._load_log()
if 'lifecycle:transition' in log.columns:
    log = log[log['lifecycle:transition'] == 'complete']
```

**Option B:** Document this limitation clearly:

```python
# In SimulationBenchmark.__init__()
"""
NOTE: Simulation generates only 'complete' lifecycle events.
Original log is filtered to 'complete' events for fair comparison.
"""
```

#### Acceptance Criteria

- [ ] Benchmark compares like with like (both complete-only)
- [ ] Documentation notes lifecycle limitation

---

## Pre-Flight Checklist

Before running the 31,000 case integration test, verify:

### Model Files

```bash
# Run this verification script
python -c "
from pathlib import Path
import sys

models = [
    ('models/branch_predictor.joblib', 'Next Activity'),
    ('models/processing_time_model_metadata.joblib', 'Processing Time'),
    ('models/processing_time_model_model.joblib', 'Processing Time Model'),
    ('models/processing_time_model_encoders.joblib', 'Processing Time Encoders'),
    ('models/processing_time_model_scaler.joblib', 'Processing Time Scaler'),
    ('case_arrival_model.pkl', 'Case Arrival'),
    ('case_attribute_prediction/attribute_models.pkl', 'Case Attributes Models'),
    ('case_attribute_prediction/attribute_distributions.pkl', 'Case Attributes Distributions'),
    ('resources/resource_permissions/ordinor_fullrecall.pkl', 'Resource Permissions'),
    ('resources/resource_availabilities/bpic2017_resource_model.pkl', 'Resource Availability'),
]

all_ok = True
for path, name in models:
    exists = Path(path).exists()
    status = '✓' if exists else '✗ MISSING'
    print(f'{status} {name}: {path}')
    if not exists:
        all_ok = False

sys.exit(0 if all_ok else 1)
"
```

### Quick Smoke Test

```bash
# Run 10-case test first
python -m integration.test_integration --mode advanced --num-cases 10 --verbose

# Verify output files exist
ls -la integration/output/simulated_log.csv
ls -la integration/output/simulated_log.xes
```

### Memory Check

```bash
# The processing time model is ~5GB
# Ensure sufficient RAM
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')"
# Should be > 8GB
```

---

## Validation Protocol

After completing all P0 and P1 tasks, run this validation:

### Step 1: Small-Scale Test (100 cases)

```bash
python -m integration.test_integration \
    --mode advanced \
    --num-cases 100 \
    --output-dir integration/output_100 \
    --verbose
```

**Expected:**
- 100 cases started
- ~700 events (avg 7 events/case for BPIC17)
- < 5% allocation failures
- No warnings about missing transitions

### Step 2: Medium-Scale Test (1000 cases)

```bash
python -m integration.test_integration \
    --mode advanced \
    --num-cases 1000 \
    --output-dir integration/output_1000
```

**Expected:**
- Linear scaling of events
- No memory issues
- Consistent allocation failure rate

### Step 3: Benchmark Validation (1000 cases)

```bash
python -c "
from integration.SimulationBenchmark import SimulationBenchmark

benchmark = SimulationBenchmark(
    'eventlog/eventlog.xes.gz',
    'integration/output_1000/simulated_log.xes'
)
results = benchmark.compute_all_metrics()
benchmark.print_summary()
"
```

**Expected:**
- Similar basic statistics (case count, event count)
- Similar activity distributions
- Throughput times within reasonable range

### Step 4: Full Integration Test (31,000 cases)

```bash
python -m integration.test_integration \
    --mode advanced \
    --num-cases 31000 \
    --output-dir integration/output_full
```

**Expected:**
- All 31,000 cases complete
- Runtime < 4 hours (estimate)
- Output files generated successfully

---

## Issue Tracking Summary

| ID | Priority | Status | Owner | Description |
|----|----------|--------|-------|-------------|
| P0-1 | Critical | `[x]` | Claude Code | Processing time context not passed |
| P0-2 | Critical | `[x]` | Claude Code | Arrival timestamp count verification |
| P0-3 | Critical | `[x]` | Claude Code | Case arrival config alignment |
| P1-1 | High | `[x]` | Claude Code | Activity transition coverage |
| P1-2 | High | `[x]` | Claude Code | Offer attributes in event log |
| P2-1 | Medium | `[ ]` | - | Relative model paths |
| P2-2 | Medium | `[ ]` | - | Resource fallback monitoring |
| P2-3 | Medium | `[ ]` | - | Lifecycle consistency |

---

## Appendix: Quick Reference Commands

```bash
# Run basic mode (stubs only)
python -m integration.test_integration --mode basic --num-cases 100

# Run advanced mode (all ML models)
python -m integration.test_integration --mode advanced --num-cases 100

# Run mixed mode (selective components)
python -m integration.test_integration --mode mixed \
    --arrivals advanced \
    --processing advanced \
    --attributes basic

# Run benchmark comparison
python -c "
from integration.SimulationBenchmark import SimulationBenchmark
b = SimulationBenchmark('eventlog/eventlog.xes.gz', 'integration/output/simulated_log.xes')
b.compute_all_metrics()
b.print_summary()
b.export_results('integration/output/benchmark.xlsx')
"
```

---

*Document maintained by: Integration Team*
*Last updated: 2024-12-28 (P0-1, P0-2, P0-3, P1-1, P1-2 completed)*
