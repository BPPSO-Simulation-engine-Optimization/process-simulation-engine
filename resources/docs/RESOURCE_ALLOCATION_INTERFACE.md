# Resource Allocation Interface

The `ResourceAllocator` serves as the central component for assigning resources to simulated activities. It orchestrates two underlying sub-models to select the best resource:
1.  **Permissions**: Who is *qualified* to perform this task? (OrdinoR Model)
2.  **Availability**: Who is *available* at this time? (Advanced Mined Patterns)

## Core Usage

Initialize the allocator once with your event log:

```python
from resources.resource_allocation import ResourceAllocator

allocator = ResourceAllocator(
    log_path="path/to/event_log.xes",
    permission_method="ordinor",  # Uses OrdinoR (FullRecall)
    availability_config={
        "enable_pattern_mining": True, # Mines working hours/days from log
        "min_activity_threshold": 10   # Min events to mine a pattern
    }
)
```

### `allocate` Method

The main method called by the simulation engine during event processing:

```python
resource = allocator.allocate(
    activity="W_Complete application",
    timestamp=current_simulation_time,
    case_type="Home Improvement" 
)
```

### Parameters

*   **`activity`** (`str`): The name of the activity to perform.
*   **`timestamp`** (`datetime`): The simulation time when the activity is starting.
*   **`case_type`** (`str`): **CRITICAL**. This represents the context of the case. 
    *   Following the **OrdinoR** implementation paper, this MUST be mapped to the **Loan Goal** (e.g., "Home Improvement", "Car", "Personal Loan").
    *   Do *not* use trace variants or other derived types here. The permission model relies on the specific Loan Goal to determine role eligibility.

## Underlying Logic
1.  **Eligibility**: Retrieves a set of eligible resources for the `(activity, case_type)` context using the **OrdinoR FullRecall** model.
2.  **Availability**: Filters this set using the **AdvancedResourceAvailabilityModel**, which checks:
    *   Resource-specific working hours (mined from history).
    *   Resource-specific working days (e.g., Mon-Thu vs Mon-Fri).
    *   (Optional) Probabilistic availability if configured.
3.  **Selection**: Randomly selects one of the available, eligible resources.

## Integration with DESEngine

The `ResourceAllocator.allocate()` method returns `None` when no eligible resource is available. The **DESEngine** handles this through a **waiting queue mechanism**:

### Resource Pool & Waiting Queue

The DESEngine maintains a `ResourcePool` that tracks:
- **Dynamic busy state**: Which resources are currently working on activities
- **Per-activity waiting queues**: Work that couldn't be allocated (FIFO ordering)

### Allocation Flow in DESEngine

```
1. Activity needs resource
   ↓
2. Check eligibility (permission model)
   ↓
3. Filter by availability (working hours)
   ↓
4. Filter by busy state (not working on another activity)
   ↓
5a. Resource found → Schedule activity, mark resource busy
5b. No resource → Add to waiting queue
```

### When Resources Become Free

When an activity completes:
1. Resource is released (marked as free)
2. Waiting queue is checked for activities this resource can perform
3. If matching work found, it's dispatched to the freed resource (FIFO)

### Failure Reasons

The allocation can fail for three reasons:
- `'no_eligible'`: No resources qualified for this activity (permission model issue)
- `'outside_hours'`: Qualified resources exist but none are working at this timestamp
- `'all_busy'`: Qualified resources exist and are working, but all are busy with other activities

### User_1 (System Resource)

User_1 is a special 24/7 automated system resource that:
- Is always available (no working hour restrictions)
- Is eligible for system activities: `A_Create Application`, `A_Submitted`, `A_Concept`, `A_Cancelled`, `O_Cancelled`, `W_Complete application`, `W_Validate application`, `W_Call after offers`, `W_Call incomplete files`, `W_Handle leads`, `W_Assess potential fraud`

This ensures the simulation chain keeps moving even outside business hours for these activities.
