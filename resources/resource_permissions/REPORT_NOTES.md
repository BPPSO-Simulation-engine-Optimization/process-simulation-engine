# Resource Permission Mining Implementation Notes

**Overview**
This document consolidates the implementation details of the resource permission mining system for the Process Simulation Engine. The implementation leverages the **OrdinoR (2022)** framework methodology.

## 1. Methodology: OrdinoR Framework

The implementation decomposes the permission discovery into three main stages: Context Construction (-> Learning Execution Contexts), Resource Clustering, and Group Profiling.

### 1.1 Configuration (BPIC2017 Optimized)
- **Context Construction:**
    - **Execution Context**: AT + TT + CT -> highest quality model for BPIC2017
        - **AT**: Activity Type - actions performed (raw activity labels)
        - **TT**: Time Type - when the work was done (day of week)
        - **CT**: Case Type - categorises the loan applications (by context aware trace clustering)
    - **Trace Clustering**: Agglomerative Hierarchical Clustering (AHC) with Ward's Linkage (-> minimum variance criteria) -> Bose & Van der Aalst, 2009
- **Resource Clustering**: AHC with Ward's Linkage (Euclidean distance)
- **Grouping Profiling**: OverallScore ($\omega_1=0.5, p=0.5$)
- **Data Preprocessing**:
    - Filter: `lifecycle:transition = 'complete'`
    - Exclude: System users (e.g., 'User_1')
    - Handling: Drop partial traces

---

## 2. Implementation Pipeline

### Step 1: Data Preparation
Located in `resources/resource_permissions/data_preparation.py`.
- **Filtering**: Keeps only completed events to capture actual work.
- **Cleaning**: Removes system actors to focus on human resource roles.

### Step 2: Trace Clustering (Case Types)
Located in `resources/resource_permissions/resource_permissions.py` -> `_apply_trace_clustering`.
- **Feature Extraction**: "Bag-of-Activities" frequency vector per case.
- **Algorithm**: **Agglomerative Hierarchical Clustering (AHC)**.
    - *Linkage*: **Ward's** (Minimum Variance).
    - *Distance*: **Euclidean**.
- **Output**: Assigns a `case_type` (cluster ID) to each historical trace.
- **Persistence**: A map of `{case_id: cluster_id}` is stored for later lookup.

### Step 3: Execution Contexts (CT+AT+TT)
Located in `resources/resource_permissions/resource_permissions.py` -> `discover_model`.
Defines the specific situation for a work item:
- **CT (Case Type)**: From trace clustering (e.g., "Cluster 1").
- **AT (Activity Type)**: The activity name (e.g., "Create Application").
- **TT (Time Type)**: Day of the week (e.g., "Monday").

### Step 4: Resource Clustering & Profiling
Located in `resources/resource_permissions/resource_permissions.py` -> `discover_model` (lines 226-278).
Uses OrdinoR library functions:
- **Resource Profiles**: `ordinor.resource_features.direct_count()` - Counts how often resources work in each context.
- **Clustering**: `ordinor.group_discovery.ahc()` - Groups resources with similar profiles using AHC (n=10 groups).
- **Profiling**: `ordinor.group_profiling.overall_score()` - Determines which contexts each *group* allows.

---

## 3. Runtime Permission Logic (Context-Aware)

The simulation queries permissions via `ResourceAllocator.allocate()`.

```python
allocator.allocate(activity="Review", timestamp=dt, case_id="case_123")
```

### Resolution Logic (`get_eligible_resources`)

1.  **Context Derivation**:
    *   **Time Context**: Derived from `timestamp.day_name()` (e.g., "Tuesday").
    *   **Case Context**: Looked up via `case_id` in the persisted training map.
        *   *Known Case*: Returns specific cluster ID.
        *   *New/Unknown Case*: Returns `None` (Permissive Fallback).

2.  **Filtering**:
    The system finds all execution contexts $(c_{case}, c_{act}, c_{time})$ in the discovered model where:
    *   $c_{act} == \text{Activity}$
    *   $c_{time} == \text{Time Context}$ (or "AnyTime")
    *   $c_{case} == \text{Case Context}$ (if Case Context is known; otherwise ignored)

3.  **Aggregation**:
    Returns the union of all resources belonging to groups capable of the matching contexts.

---

## 4. Key Design Decisions & Simplifications

| Feature | Paper | Implementation | Rationale |
| :--- | :--- | :--- | :--- |
| **Trace Features** | Complex (Transitions, etc.) | **Bag-of-Activities** | Simplification for initial version; sufficient for BPIC17. |
| **Trace Algorithm** | AHC (Ward's) | **AHC (Ward's)** | Aligned with "Context-Aware Trace Clustering" paper. |
| **Cluster Selection** | Cross-Validation (2-10) | **Fixed (5, 10)** | Pragmatic defaults; configurable by user. |
| **Profiling Threshold** | Grid Search ($\lambda \in 0.1..0.9$) | **Fixed ($p=0.5$)** | Balanced starting point. |
| **New Case Handling** | N/A (Static Analysis) | **Permissive Fallback** | Unknown cases match *any* case type to avoid blocking simulation. |

---

## 5. Performance & Caching

- **Discovery Time**: 15-60 mins for full BPIC2017 dataset.
- **Caching**: The discovered `OrganizationalModel` (including case map) is pickled to `ordinor_model_full.pkl`.
- **Allocating Time**: <1ms per query (in-memory lookup).
