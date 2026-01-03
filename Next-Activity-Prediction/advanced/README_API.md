# Next-Activity API (Advanced Models)

```python
import pandas as pd
from Next-Activity-Prediction.advanced import load_models, predict_next_activity

models = load_models("Next-Activity-Prediction/advanced/models_lstm_new")

history = pd.DataFrame([
    {"concept:name": "A_Create Application", "org:resource": "User_1", "time:timestamp": "2025-08-18T10:00:00"},
    {"concept:name": "W_Complete application", "org:resource": "User_5", "time:timestamp": "2025-08-18T12:00:00"},
    {"concept:name": "A_Validating", "org:resource": "User_7", "time:timestamp": "2025-08-18T13:00:00"},
])

preds = predict_next_activity("DP 1", history, models, top_k=3)
print(preds)  # -> [('NextAct', 0.42), ...]
```

## Function signatures
- `load_models(models_dir: str) -> dict`
- `predict_next_activity(dp_name: str, history_df: pd.DataFrame, models: dict, top_k: int = 3, max_history: int = 15) -> list`

Inputs must include `concept:name`, `org:resource`, and `time:timestamp`. Context features listed in each model’s `context_keys` are read if present; missing ones default to zero.

## Simulation helpers (next-activity only)

```python
from advanced import (
    load_models,
    load_simulation_assets,
    simulate_cases_advanced,
    events_to_dataframe,
)

models = load_models("Next-Activity-Prediction/advanced/models_lstm_new")
process_graph, decision_map = load_simulation_assets()

cases = [
    {"loan_goal": "Car", "application_type": "New credit"},
    {"loan_goal": "Home improvement", "application_type": "Limit raise"},
]

events = simulate_cases_advanced(
    cases,
    models=models,
    process_graph=process_graph,
    decision_point_map=decision_map,
)

events_df = events_to_dataframe(events)
print(events_df.head())
```

- `load_simulation_assets` loads `named_transitions.json` and `bpmn_decision_point_map.pkl` from `advanced/assets` unless paths are provided.
- `simulate_cases_advanced` uses only next-activity predictions (no resource allocation).
- Outputs are lightweight `Event` objects; use `events_to_dataframe` for quick inspection.

