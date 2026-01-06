"""
Data preprocessing for BPIC17 simplified model.

Filters event log to only start/complete lifecycle transitions and adds END tokens.
"""

import pandas as pd
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import pm4py
    from pm4py.objects.log.obj import EventLog
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    logger.warning("pm4py not available. XES file loading will not work.")


def load_and_filter_bpic17(
    log_path: Optional[str] = None,
    lifecycle_filter: list = None
) -> pd.DataFrame:
    """
    Load BPIC 2017 event log and filter to specified lifecycle transitions.
    
    Args:
        log_path: Path to BPIC 2017 XES file. If None, uses default path.
        lifecycle_filter: List of lifecycle transitions to keep. 
                         Default: ["start", "complete"]
    
    Returns:
        Filtered DataFrame with event log data.
    """
    if lifecycle_filter is None:
        lifecycle_filter = ["start", "complete"]
    
    if log_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        log_path = os.path.join(project_root, "Dataset", "BPI Challenge 2017.xes")
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Event log not found: {log_path}")
    
    if not PM4PY_AVAILABLE:
        raise ImportError("pm4py is required to load XES files. Install with: pip install pm4py")
    
    logger.info(f"Loading event log from: {log_path}")
    event_log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(event_log)
    
    logger.info(f"Loaded {len(df):,} events from {df['case:concept:name'].nunique():,} cases")
    
    if "lifecycle:transition" not in df.columns:
        logger.warning("No lifecycle:transition column found. Adding default 'complete'.")
        df["lifecycle:transition"] = "complete"
    
    df["lifecycle:transition"] = df["lifecycle:transition"].str.lower().str.strip()
    
    original_count = len(df)
    df = df[df["lifecycle:transition"].isin(lifecycle_filter)].copy()
    filtered_count = len(df)
    
    logger.info(
        f"Filtered to {lifecycle_filter} lifecycle transitions: "
        f"{filtered_count:,} events ({filtered_count/original_count*100:.1f}% of original)"
    )
    
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    
    df = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    
    return df


def add_end_tokens(df: pd.DataFrame, end_token: str = "END") -> pd.DataFrame:
    """
    Add END tokens at the end of each trace.
    
    Args:
        df: Event log DataFrame with case:concept:name column.
        end_token: Token to use for trace endings. Default: "END"
    
    Returns:
        DataFrame with END tokens added at trace endings.
    """
    end_rows = []
    
    for case_id, group in df.groupby("case:concept:name"):
        last_event = group.iloc[-1]
        
        end_row = last_event.copy()
        end_row["concept:name"] = end_token
        end_row["lifecycle:transition"] = "complete"
        
        if "org:resource" in end_row.index:
            end_row["org:resource"] = "System"
        
        if "time:timestamp" in end_row.index:
            end_row["time:timestamp"] = last_event["time:timestamp"] + pd.Timedelta(seconds=1)
        
        end_rows.append(end_row)
    
    if end_rows:
        end_df = pd.DataFrame(end_rows)
        df_with_ends = pd.concat([df, end_df], ignore_index=True)
        df_with_ends = df_with_ends.sort_values(
            ["case:concept:name", "time:timestamp"]
        ).reset_index(drop=True)
        
        logger.info(f"Added {len(end_rows):,} END tokens to traces")
        return df_with_ends
    
    return df


