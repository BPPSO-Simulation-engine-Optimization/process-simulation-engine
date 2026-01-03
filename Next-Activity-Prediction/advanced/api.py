import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Support both package and script imports
try:
    from .storage.persistence import ModelPersistence  # type: ignore
except ImportError:  # pragma: no cover
    from storage.persistence import ModelPersistence


def load_models(models_dir: str):
    """Load all per-decision-point advanced models from a directory."""
    return ModelPersistence.load_all(models_dir)


def predict_next_activity(
    dp_name: str,
    history_df: pd.DataFrame,
    models: dict,
    top_k: int = 3,
    max_history: int = 15,
):
    """
    Predict next activities at a decision point using the advanced models.

    Args:
        dp_name: Decision point label, e.g. "DP 1".
        history_df: DataFrame with columns concept:name, org:resource, time:timestamp.
        models: Dict from load_models.
        top_k: Number of candidates to return.
        max_history: How many past events to keep.

    Returns:
        List of (activity, probability) tuples sorted descending.
    """
    if dp_name not in models:
        raise KeyError(f"No model for {dp_name}")

    bundle = models[dp_name]
    act_enc = bundle["activity_encoder"]
    res_enc = bundle["resource_encoder"]
    lbl_enc = bundle["label_encoder"]
    max_seq_len = bundle["max_seq_len"]
    ctx_keys = bundle["context_keys"]

    hist = history_df.sort_values("time:timestamp").tail(max_history)
    sequence = hist["concept:name"].tolist()
    resources = hist["org:resource"].tolist()
    timestamps = pd.to_datetime(hist["time:timestamp"]).tolist()
    durations = [0.0] + [
        (timestamps[i] - timestamps[i - 1]).total_seconds()
        for i in range(1, len(timestamps))
    ]

    def safe_encode(seq, enc, unknown="UNKNOWN"):
        known = set(enc.classes_)
        cleaned = [x if x in known else unknown for x in seq]
        return enc.transform(cleaned)

    X_acts = safe_encode(sequence, act_enc)
    X_res = safe_encode(resources, res_enc)
    X_durs = np.array(durations, dtype="float32")

    ctx_vals = []
    for key in ctx_keys:
        ctx_vals.append(hist[key].iloc[-1] if key in hist.columns else 0.0)
    X_ctx = np.array([ctx_vals], dtype="float32")

    X_acts = pad_sequences([X_acts], maxlen=max_seq_len, padding="pre")
    X_res = pad_sequences([X_res], maxlen=max_seq_len, padding="pre")
    X_durs = pad_sequences([X_durs], maxlen=max_seq_len, padding="pre", dtype="float32")

    probs = bundle["model"].predict([X_acts, X_durs, X_res, X_ctx], verbose=0)[0]
    classes = lbl_enc.classes_[: len(probs)]
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def load_simulation_assets(
    transitions_path: str | None = None,
    decision_map_path: str | None = None,
):
    """Wrapper for simulation asset loader (lazy import to avoid cycles)."""
    from .simulation import load_simulation_assets as _impl

    return _impl(transitions_path, decision_map_path)


def decision_function_advanced(
    current_dp: str,
    history: list,
    options: list,
    models: dict,
    decision_point_map: dict,
    process_graph: dict,
    max_history: int = 15,
):
    from .simulation import decision_function_advanced as _impl

    return _impl(
        current_dp,
        history,
        options,
        models,
        decision_point_map,
        process_graph,
        max_history=max_history,
    )


def simulate_cases_advanced(
    cases: list,
    models: dict,
    process_graph: dict,
    decision_point_map: dict,
    start_task: str = "A_Create Application",
    end_task: str = "End",
    max_history: int = 15,
):
    from .simulation import simulate_cases_advanced as _impl

    return _impl(
        cases,
        models,
        process_graph,
        decision_point_map,
        start_task=start_task,
        end_task=end_task,
        max_history=max_history,
    )


def events_to_dataframe(events: list):
    from .simulation import events_to_dataframe as _impl

    return _impl(events)
