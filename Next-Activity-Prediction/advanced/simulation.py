"""
Lightweight next-activity simulation utilities.

Ported from the simple simulator, trimmed to only use next-activity
prediction (no resource allocation). Designed to stay human-readable.
"""

from __future__ import annotations

import heapq
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import pandas as pd

from .api import predict_next_activity

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_TRANSITIONS = ASSETS_DIR / "named_transitions.json"
DEFAULT_DECISION_MAP = ASSETS_DIR / "bpmn_decision_point_map.pkl"


def load_simulation_assets(
    transitions_path: str | Path | None = None,
    decision_map_path: str | Path | None = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Load process graph and decision-point map."""
    transitions_file = Path(transitions_path or DEFAULT_TRANSITIONS)
    decision_map_file = Path(decision_map_path or DEFAULT_DECISION_MAP)

    with transitions_file.open() as f:
        process_graph = json.load(f)

    decision_point_map = joblib.load(decision_map_file)
    return process_graph, decision_point_map


def _clean(name: str) -> str:
    return re.sub(r"\s+\d+$", "", name)


def _history_to_df(history: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(history)
    if df.empty:
        return df

    df = df.rename(
        columns={
            "task": "concept:name",
            "resource": "org:resource",
            "timestamp": "time:timestamp",
        }
    )

    if "org:resource" not in df.columns:
        df["org:resource"] = "User_1"

    if "time:timestamp" not in df.columns:
        base = pd.Timestamp.utcnow()
        df["time:timestamp"] = [
            base + timedelta(seconds=i) for i in range(len(df))
        ]
    else:
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

    return df


def decision_function_advanced(
    current_dp: str,
    history: List[Dict[str, Any]],
    options: Iterable[str],
    models: Dict[str, Any],
    decision_point_map: Dict[str, Any],
    process_graph: Dict[str, List[str]],
    max_history: int = 15,
) -> Tuple[str, float]:
    """Choose next node at a decision point using advanced models."""
    opts = list(options)
    if len(opts) == 1:
        return opts[0], 1.0

    if current_dp not in models:
        if current_dp in decision_point_map:
            outs = decision_point_map[current_dp].get("outgoing", opts)
            choice = outs[0] if len(outs) == 1 else random.choice(outs)
        else:
            choice = random.choice(opts)
        return choice, 1.0 / len(opts)

    hist_df = _history_to_df(history)
    if hist_df.empty:
        fallback = decision_point_map.get(current_dp, {}).get("outgoing", opts)
        choice = fallback[0] if len(fallback) == 1 else random.choice(fallback)
        return choice, 1.0 / len(opts)

    hist_df = hist_df.sort_values("time:timestamp").tail(max_history)

    try:
        preds = predict_next_activity(
            current_dp, hist_df, models, top_k=max(len(opts) * 2, 5)
        )
    except Exception:
        fallback = decision_point_map.get(current_dp, {}).get("outgoing", opts)
        choice = fallback[0] if len(fallback) == 1 else random.choice(fallback)
        return choice, 1.0 / len(opts)

    prob_map = {_clean(act): float(prob) for act, prob in preds}

    def gather_leaves(node: str) -> List[str]:
        leaves, stack, seen = [], [node], {node}
        while stack:
            n = stack.pop()
            for nxt in process_graph.get(n, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if nxt.startswith(("DP", "PG")):
                    stack.append(nxt)
                else:
                    leaves.append(nxt)
        return leaves

    weights = {}
    for opt in opts:
        key = _clean(opt)
        if key in prob_map:
            weights[opt] = prob_map[key]
        else:
            leaves = gather_leaves(opt)
            leaf_probs = [prob_map.get(_clean(l), 0.0) for l in leaves]
            weights[opt] = max(leaf_probs, default=0.0)

    vals = pd.Series(weights)
    if (vals.sum() == 0) or vals.isna().any():
        choice = random.choice(opts)
        return choice, 1.0 / len(opts)

    probs = (vals / vals.sum()).fillna(0.0)
    choice = random.choices(list(probs.index), weights=probs.tolist(), k=1)[0]
    return choice, float(probs.loc[choice])


@dataclass(order=True)
class Event:
    timestamp: datetime
    case_id: str
    task: str
    context: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list, compare=False)
    probability: Any | None = field(default=None, compare=False)
    predicted: List[str] | None = field(default=None, compare=False)
    next_nodes: List[str] | None = field(default=None, compare=False)

    def to_history_entry(self) -> Dict[str, Any]:
        entry = {"task": self.task, "timestamp": self.timestamp, **self.context}
        if self.probability is not None:
            entry["probability"] = self.probability
        if self.predicted is not None:
            entry["predicted"] = self.predicted
        if self.next_nodes is not None:
            entry["next_nodes"] = self.next_nodes
        return entry


def simulate_cases_advanced(
    cases: List[Dict[str, Any]],
    models: Dict[str, Any],
    process_graph: Dict[str, List[str]],
    decision_point_map: Dict[str, Any],
    start_task: str = "A_Create Application",
    end_task: str = "End",
    max_history: int = 15,
) -> List[Event]:
    """Simulate cases through the process graph using next-activity predictions."""
    queue: List[Event] = []
    events: List[Event] = []
    now = datetime.utcnow()

    for idx, ctx in enumerate(cases):
        ts = now + timedelta(seconds=idx)
        heapq.heappush(
            queue,
            Event(
                timestamp=ts,
                case_id=f"Case_{idx+1}",
                task=start_task,
                context=ctx,
            ),
        )

    while queue:
        event = heapq.heappop(queue)
        current = event.task
        next_nodes = list(process_graph.get(current, []))
        event.next_nodes = next_nodes or None

        if current == end_task or not next_nodes:
            events.append(event)
            continue

        prob = "Only Option"
        predicted = None
        if current.startswith("DP"):
            if len(next_nodes) == 1:
                predicted = next_nodes
                prob = 1.0
            else:
                chosen, prob = decision_function_advanced(
                    current,
                    event.history,
                    next_nodes,
                    models,
                    decision_point_map,
                    process_graph,
                    max_history=max_history,
                )
                predicted = [chosen]
                next_nodes = [chosen]

        event.predicted = predicted
        event.probability = prob if current.startswith("DP") else "Only Option"
        events.append(event)

        new_history = event.history + [event.to_history_entry()]
        for nxt in next_nodes:
            heapq.heappush(
                queue,
                Event(
                    timestamp=event.timestamp + timedelta(seconds=1),
                    case_id=event.case_id,
                    task=nxt,
                    context=event.context,
                    history=new_history,
                    probability=prob if current.startswith("DP") else "Only Option",
                ),
            )

    return events


def events_to_dataframe(events: List[Event]) -> pd.DataFrame:
    """Convert simulated events to a dataframe."""
    rows = []
    for ev in events:
        row = {
            "case_id": ev.case_id,
            "task": ev.task,
            "timestamp": ev.timestamp,
            "probability": ev.probability,
        }
        if ev.predicted is not None:
            row["predicted"] = ev.predicted
        if ev.next_nodes is not None:
            row["next_nodes"] = ev.next_nodes
        row.update(ev.context)
        rows.append(row)
    return pd.DataFrame(rows)

