# Resource Permission System

This module implements resource eligibility checking for the BPIC2017 simulation engine. We support two approaches:

## Quick Start

```python
from resources.resource_permissions import BasicResourcePermissions, AdvancedResourcePermissions

# Basic: direct historical lookup
basic = BasicResourcePermissions(log_path="path/to/log.xes.gz")
eligible = basic.get_eligible_resources("A_Create Application")

# Advanced: group-based lookup via clustering
advanced = AdvancedResourcePermissions(log_path="path/to/log.xes.gz")
advanced.discover_model(n_clusters=5)
eligible = advanced.get_eligible_resources("A_Create Application")
```

## Two Approaches

### BasicResourcePermissions
Direct mapping from the event log. If a resource has ever performed an activity, they're eligible.

**Pros:** Simple, fast, no false positives  
**Cons:** Conservative—new resources have zero capabilities

### AdvancedResourcePermissions
Clusters resources into groups based on activity profiles (who does what). Assigns capabilities at the group level.

**Pros:** Better generalization, handles resource substitution  
**Cons:** May include resources who haven't actually done an activity but belong to a capable group

## File Structure

```
resources/
├── resource_permissions.py    # Main entry point (both classes)
├── resource_features.py       # Builds resource-activity matrix
├── resource_clustering.py     # Agglomerative clustering
├── group_profiling.py         # Determines group capabilities
├── organizational_model.py    # Model storage/persistence
└── docs/
    └── README.md              # This file
```

## Advanced Pipeline

1. **Feature extraction**: Build a matrix where rows=resources, columns=activities, values=counts
2. **Clustering**: Run AHC with Ward linkage to group similar resources
3. **Profiling**: For each group, determine which activities are "capabilities" based on:
   - `min_frequency`: at least N total occurrences in the group
   - `min_coverage`: at least X% of group members performed it
4. **Lookup**: Activity → groups with that capability → all members of those groups

## Parameters

| Param | Default | What it does |
|-------|---------|--------------|
| `n_clusters` | 5 | Number of resource groups |
| `min_frequency` | 5 | Min occurrences for capability |
| `min_coverage` | 0.3 | Min fraction of members |

## Model Persistence

```python
# Save after discovery
advanced.save_model("org_model.json")

# Load later (skip discovery)
from resources.organizational_model import OrganizationalModel
model = OrganizationalModel.load("org_model.json")
advanced = AdvancedResourcePermissions(model=model)
```

## Tests

```bash
python3 -m unittest tests/test_resource_permissions.py
python3 -m unittest tests/test_advanced_permissions.py
python3 -m unittest tests/test_benchmark.py
```

## Benchmarking

Compare Basic vs Advanced approaches:

```bash
python3 resources/benchmark_permissions.py \
    --log-path /path/to/eventlog.xes.gz \
    --n-clusters 5 \
    --output-dir benchmark_results
```

Outputs:
- `benchmark_report.md` - Human-readable analysis
- `activity_breakdown.csv` - Per-activity metrics
- `sensitivity_analysis.csv` - Metrics vs n_clusters
- `plots/` - Visualizations

Key metrics:
- **Precision**: Of advanced-eligible, how many actually performed the activity?
- **Recall**: Of actual performers, how many does advanced include?
- **Coverage**: What % of activities have eligible resources?

On BPIC2017: ~78% precision, ~96% coverage with n_clusters=5.

## Notes

- The advanced approach typically returns *more* eligible resources than basic (group members inherit capabilities)
- Set `EVENT_LOG_PATH` in `.env` to run integration tests
- Clustering uses sklearn's `AgglomerativeClustering`—make sure it's installed

