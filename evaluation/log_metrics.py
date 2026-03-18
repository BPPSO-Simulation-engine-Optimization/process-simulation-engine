"""
Shared log loading and metric computation for evaluation and termination comparison.

Used by log_comparison notebooks and integration/run_termination_comparison.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLS = {"case:concept:name", "concept:name", "time:timestamp"}

W_COST, W_LOW_PRODUCTIVITY, W_LOW_VALUE = 0.25, 0.25, 0.20
W_LOW_CONTRIBUTION, W_CRITICALITY, W_CYCLE_IMPACT = 0.15, 0.10, 0.05


def load_log(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if str(path).endswith(".csv"):
        df = pd.read_csv(path)
    else:
        import pm4py
        log = pm4py.read_xes(str(path))
        df = pm4py.convert_to_dataframe(log)
    df.columns = [c.lower() for c in df.columns]
    rename_map = {
        "case:concept:name": "case:concept:name",
        "concept:name": "concept:name",
        "time:timestamp": "time:timestamp",
        "org:resource": "org:resource",
        "lifecycle:transition": "lifecycle:transition",
    }
    df = df.rename(columns={c.lower(): c for c in rename_map})
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, format="ISO8601")
    return df


def extract_activity_durations(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    has_lifecycle = "lifecycle:transition" in df.columns
    if not has_lifecycle:
        return pd.DataFrame(), False

    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    lc = df_sorted["lifecycle:transition"].str.lower()
    starts = df_sorted[lc == "start"].copy()
    completes = df_sorted[lc == "complete"].copy()
    if completes.empty:
        return pd.DataFrame(), False

    merge_cols = ["case:concept:name", "concept:name"]
    if "org:resource" in df.columns:
        merge_cols.append("org:resource")
    activities_with_start = set(starts["concept:name"].unique())

    for _df in [starts, completes]:
        _df["_occ"] = _df.groupby(merge_cols).cumcount()
    full_starts = starts[starts["concept:name"].isin(activities_with_start)]
    full_completes = completes[completes["concept:name"].isin(activities_with_start)]

    df_full = pd.DataFrame()
    if not full_starts.empty and not full_completes.empty:
        df_full = pd.merge(
            full_starts.rename(columns={"time:timestamp": "start_time"})[merge_cols + ["_occ", "start_time"]],
            full_completes.rename(columns={"time:timestamp": "complete_time"})[merge_cols + ["_occ", "complete_time"]],
            on=merge_cols + ["_occ"],
            how="inner",
        )
        df_full["duration_hours"] = (df_full["complete_time"] - df_full["start_time"]).dt.total_seconds() / 3600
        df_full = df_full.drop(columns=["_occ"])

    complete_only = completes[~completes["concept:name"].isin(activities_with_start)]
    df_comp_only = pd.DataFrame()
    if not complete_only.empty:
        df_sorted["_prev_time"] = df_sorted.groupby("case:concept:name")["time:timestamp"].shift(1)
        comp_rows = df_sorted.loc[(lc == "complete") & (~df_sorted["concept:name"].isin(activities_with_start))].copy()
        comp_rows = comp_rows[comp_rows["_prev_time"].notna()]
        if not comp_rows.empty:
            comp_rows["duration_hours"] = (comp_rows["time:timestamp"] - comp_rows["_prev_time"]).dt.total_seconds() / 3600
            comp_rows = comp_rows.rename(columns={"time:timestamp": "complete_time"})
            comp_rows["start_time"] = comp_rows["_prev_time"]
            out_cols = merge_cols + ["start_time", "complete_time", "duration_hours"]
            df_comp_only = comp_rows[out_cols].copy()

    df_dur = pd.concat([df_full, df_comp_only], ignore_index=True)
    df_dur = df_dur[df_dur["duration_hours"] > 0].reset_index(drop=True)
    return df_dur, not df_dur.empty


def compute_case_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.sort_values("time:timestamp").groupby("case:concept:name")
    case_start = grp["time:timestamp"].min().rename("case_start")
    case_end = grp["time:timestamp"].max().rename("case_end")
    n_events = grp.size().rename("n_events")
    df_cases = pd.concat([case_start, case_end, n_events], axis=1)
    df_cases["cycle_time_days"] = (df_cases["case_end"] - df_cases["case_start"]).dt.total_seconds() / 86_400
    if "org:resource" in df.columns:
        def count_handovers(sub):
            r = sub.sort_values("time:timestamp")["org:resource"].dropna().tolist()
            return sum(1 for a, b in zip(r, r[1:]) if a != b)
        df_cases["handovers"] = grp.apply(count_handovers)
    else:
        df_cases["handovers"] = np.nan
    return df_cases.reset_index()


def compute_resource_metrics(df_durations: pd.DataFrame, has_durations: bool) -> pd.DataFrame | None:
    if not has_durations or "org:resource" not in df_durations.columns:
        return None
    df_res = (
        df_durations.groupby("org:resource")["duration_hours"]
        .sum()
        .reset_index()
        .rename(columns={"org:resource": "resource", "duration_hours": "busy_hours"})
        .sort_values("busy_hours", ascending=False)
        .reset_index(drop=True)
    )
    df_res["work_share"] = df_res["busy_hours"] / df_res["busy_hours"].sum()
    return df_res


def _wrf(df_res: pd.DataFrame) -> float:
    shares = df_res["work_share"].values
    weights = df_res["busy_hours"].values / df_res["busy_hours"].sum()
    return float(np.sum(weights * np.abs(shares - shares.mean())))


def gini_coefficient(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    v = v[v >= 0]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return np.nan
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * v).sum()) / (n * v.sum()) - (n + 1) / n)


def compute_resource_features(df_raw, df_durations, df_cases, has_durations, cost_per_fte):
    if "org:resource" not in df_raw.columns or not has_durations:
        return None
    df_complete = df_raw[df_raw["lifecycle:transition"].str.lower() == "complete"].copy() if "lifecycle:transition" in df_raw.columns else df_raw.copy()
    busy = df_durations.groupby("org:resource")["duration_hours"].sum().reset_index().rename(columns={"org:resource": "resource", "duration_hours": "busy_hours"})
    busy["fte_cost"] = busy["busy_hours"] * cost_per_fte
    busy["workload_share"] = busy["busy_hours"] / busy["busy_hours"].sum()
    event_counts = df_complete.groupby("org:resource").size().reset_index(name="completed_events_count").rename(columns={"org:resource": "resource"})
    case_counts = df_complete.groupby("org:resource")["case:concept:name"].nunique().reset_index().rename(columns={"org:resource": "resource", "case:concept:name": "distinct_cases_count"})
    case_ct = df_cases[["case:concept:name", "cycle_time_days"]].copy()
    resource_cases = df_complete[["org:resource", "case:concept:name"]].drop_duplicates().rename(columns={"org:resource": "resource"})
    resource_ct = resource_cases.merge(case_ct, on="case:concept:name", how="left").groupby("resource")["cycle_time_days"].mean().reset_index().rename(columns={"cycle_time_days": "avg_cycle_time_of_cases_involved"})
    global_avg_ct = df_cases["cycle_time_days"].mean()
    rm = busy.merge(event_counts, on="resource", how="left").merge(case_counts, on="resource", how="left").merge(resource_ct, on="resource", how="left")
    rm["productivity"] = rm["completed_events_count"] / rm["busy_hours"]
    rm["contribution_per_cost"] = rm["distinct_cases_count"] / rm["fte_cost"]
    rm["cycle_time_delta"] = rm["avg_cycle_time_of_cases_involved"] - global_avg_ct
    return rm


def percentile_normalize(series: pd.Series, p_low: float = 5, p_high: float = 95) -> pd.Series:
    lo, hi = np.percentile(series.dropna(), p_low), np.percentile(series.dropna(), p_high)
    clipped = series.clip(lower=lo, upper=hi)
    return pd.Series(0.5, index=series.index) if (hi - lo) == 0 else (clipped - lo) / (hi - lo)


def compute_all_metrics(df_raw: pd.DataFrame, cost_per_fte: float) -> dict:
    df_dur, has_dur = extract_activity_durations(df_raw)
    df_cases = compute_case_metrics(df_raw)
    df_res = compute_resource_metrics(df_dur, has_dur)

    total_busy = float(df_res["busy_hours"].sum()) if df_res is not None else np.nan
    fte_cost = total_busy * cost_per_fte if not np.isnan(total_busy) else np.nan
    n = len(df_cases)

    def safe_mean(s):
        return float(s.mean()) if s.notna().any() else np.nan

    rm = compute_resource_features(df_raw, df_dur, df_cases, has_dur, cost_per_fte)
    max_term, mean_term = np.nan, np.nan
    if rm is not None and len(rm) > 0:
        for col, norm in [
            ("fte_cost", "norm_cost"),
            ("productivity", "norm_productivity"),
            ("contribution_per_cost", "norm_value"),
            ("distinct_cases_count", "norm_contribution"),
            ("workload_share", "norm_workload_share"),
            ("cycle_time_delta", "norm_cycle_delta"),
        ]:
            rm[norm] = percentile_normalize(rm[col])
        rm["termination_score"] = (
            W_COST * rm["norm_cost"]
            + W_LOW_PRODUCTIVITY * (1 - rm["norm_productivity"])
            + W_LOW_VALUE * (1 - rm["norm_value"])
            + W_LOW_CONTRIBUTION * (1 - rm["norm_contribution"])
            + W_CRITICALITY * (1 - rm["norm_workload_share"])
            + W_CYCLE_IMPACT * rm["norm_cycle_delta"]
        )
        max_term = float(rm["termination_score"].max())
        mean_term = float(rm["termination_score"].mean())

    return {
        "Avg Cycle Time (days)": float(df_cases["cycle_time_days"].mean()),
        "Median Cycle Time (days)": float(df_cases["cycle_time_days"].median()),
        "Range Cycle Time (days)": float(df_cases["cycle_time_days"].max() - df_cases["cycle_time_days"].min()),
        "Throughput (cases/day)": n / ((df_cases["case_end"].max() - df_cases["case_start"].min()).total_seconds() / 86400) if n > 0 else np.nan,
        "Avg Resource Occupation (share)": float(df_res["work_share"].mean()) if df_res is not None else np.nan,
        "Weighted Resource Fairness": _wrf(df_res) if df_res is not None else np.nan,
        "Workload Gini Coefficient": gini_coefficient(df_res["busy_hours"].values) if df_res is not None else np.nan,
        "Total Busy Hours": total_busy,
        "Total FTE Cost (€)": fte_cost,
        "Cost per Completed Case (€)": fte_cost / n if n > 0 and not np.isnan(fte_cost) else np.nan,
        "Avg Handovers per Case": safe_mean(df_cases["handovers"]),
        "Number of Resources": len(df_res) if df_res is not None else np.nan,
        "Max Termination Score": max_term,
        "Mean Termination Score": mean_term,
    }
