# Resource Permission System

Implements resource eligibility checking for the BPIC2017 simulation engine following **OrdinoR (2022)** methodology.

> **Reference:** Yang, J., Ouyang, C., van der Aalst, W. M., ter Hofstede, A. H., & Yu, Y. (2022). OrdinoR: A framework for discovering, evaluating, and analyzing organizational models using event logs. Decision Support Systems, 158, 113771.

## Quick Start

```python
from resources.resource_permissions import BasicResourcePermissions, OrdinoRResourcePermissions

# Basic: direct historical lookup (preprocessing applied by default)
basic = BasicResourcePermissions(log_path="path/to/log.xes.gz")
eligible = basic.get_eligible_resources("A_Create Application")

# OrdinoR: Advanced trace clustering + AHC + Profiling
ordinor = OrdinoRResourcePermissions(log_path="path/to/log.xes.gz")
ordinor.discover_model(n_trace_clusters=5, n_resource_clusters=10)
eligible = ordinor.get_eligible_resources("A_Create Application")
```

## Data Preprocessing

Both approaches apply preprocessing by default:

| Step | Effect | BPIC2017 Impact |
|------|--------|-----------------|
| Filter completed | Keep only `lifecycle:transition = 'complete'` | 1,202,267 → 475,306 events (39.5%) |
| Exclude User_1 | Remove system/automation accounts | 475,306 → 399,356 events |
| Drop NA | Remove missing values | Minor reduction |

**Final dataset:** 399,356 events, 143 resources, 23 activities

## Two Approaches

### BasicResourcePermissions
Direct mapping from the event log. If a resource has ever performed an activity, they're eligible.

**Pros:** Simple, fast, no false positives  
**Cons:** Conservative—new resources have zero capabilities

### OrdinoRResourcePermissions
Uses the **OrdinoR** library with the paper's best-performing configuration for BPIC2017:

1.  **Trace Clustering (CT)**: K-Means clustering on bag-of-activities (k=5) to identify Case Types.
2.  **Execution Contexts**: Combines `CaseType + Activity + TimeType` (CT+AT+TT).
3.  **Resource Clustering**: Agglomerative Hierarchical Clustering (AHC) with Ward linkage (n=10).
4.  **Profiling**: OverallScore method (w1=0.5, p=0.5).

**Pros:** Advanced organizational mining, proven academic validity (F1=0.724).  
**Cons:** More complex pipeline.

## File Structure

```
resources/
├── resource_permissions.py    # Main classes (Basic & OrdinoR)
├── data_preparation.py        # Preprocessing (lifecycle filter, exclude resources)
├── resource_features.py       # (Internal) Resource profile building
├── resource_clustering.py     # (Internal) Clustering utilities
├── group_profiling.py         # (Internal) Profiling utilities
├── organizational_model.py    # (Internal) Model structure
└── docs/
    └── README.md              # This file
```

## Benchmarking

Compare Basic vs OrdinoR approaches:

```bash
python3 resources/benchmark_permissions.py \
    --log-path /path/to/eventlog.xes.gz \
    --n-trace-clusters 5 \
    --n-clusters 10 \
    --exclude-resources User_1 \
    --output-dir resources/benchmark_results_ordinor
```
