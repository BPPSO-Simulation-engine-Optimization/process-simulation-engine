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
