"""
Batch evaluation of simulated event logs across resource allocation strategies.

Discovers all simulated_log.xes under integration/output/, computes resource
allocation quality metrics for each, and produces:
  - evaluation/output/batch_evaluation_results.xlsx  (per_run, per_strategy, vs_baseline)
  - evaluation/output/plots/*.png                    (comparison bar charts)

Usage:
    conda activate pse_env
    python evaluation/batch_evaluate.py
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_ROOT / "integration" / "output"
COST_PER_FTE_HOUR = 50.0
FTE_COMPLETE_ONLY = False
BASELINE_STRATEGY = "greedy_random"
OUTPUT_DIR = REPO_ROOT / "evaluation" / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ═════════════════════════════════════════════════════════════════════════════
# Log discovery
# ═════════════════════════════════════════════════════════════════════════════

def discover_logs(base_dir: Path) -> list[dict]:
    """
    Find all simulated_log.xes under *_10k directories (excluding 0archive).

    Expected layouts:
        greedy_10k/<variant>/run<N>/simulated_log.xes
        batch_10k/run<N>/simulated_log.xes
        drl_10k/run<N>/simulated_log.xes
    """
    logs: list[dict] = []
    for xes in sorted(base_dir.rglob("simulated_log.xes")):
        rel = xes.relative_to(base_dir)
        parts = rel.parts  # e.g. ("greedy_10k", "random", "run1", "simulated_log.xes")

        # Skip archive / non-10k dirs
        if "0archive" in parts or not parts[0].endswith("_10k"):
            continue

        strategy_dir = parts[0]  # e.g. "greedy_10k"
        strategy_base = strategy_dir.replace("_10k", "")  # "greedy", "batch", "drl"

        if len(parts) == 4:
            # greedy_10k/<variant>/run<N>/simulated_log.xes
            variant = parts[1]
            run_str = parts[2]
        elif len(parts) == 3:
            # batch_10k/run<N>/simulated_log.xes
            variant = None
            run_str = parts[1]
        else:
            continue

        run_match = re.match(r"run(\d+)", run_str)
        if not run_match:
            continue

        run_num = int(run_match.group(1))
        strategy_label = f"{strategy_base}_{variant}" if variant else strategy_base

        logs.append({
            "path": xes,
            "strategy": strategy_label,
            "strategy_base": strategy_base,
            "variant": variant or "",
            "run": run_num,
        })

    return logs


# ═════════════════════════════════════════════════════════════════════════════
# Core metric computation  (extracted from evaluation.ipynb / log_comparison.ipynb)
# ═════════════════════════════════════════════════════════════════════════════

def _load_log(path: Path) -> pd.DataFrame:
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
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df


def _extract_resource_active_segments(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Lifecycle-aware resource active segments: start|resume → complete|withdraw|ate_abort."""
    required = {"case:concept:name", "time:timestamp", "org:resource", "lifecycle:transition"}
    if not required.issubset(df_raw.columns):
        return pd.DataFrame()

    df = df_raw.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df["_lc"] = df["lifecycle:transition"].astype(str).str.lower()

    start_transitions = {"start", "resume"}
    terminal_transitions = {"complete", "withdraw", "ate_abort"}

    rows: list[dict] = []
    for key, sub in df.groupby(["case:concept:name", "concept:name"], dropna=False):
        case_id, activity = key
        active_since: dict[str, pd.Timestamp] = {}

        for _, rec in sub.iterrows():
            resource = rec.get("org:resource")
            if pd.isna(resource):
                continue
            resource = str(resource)
            ts = rec["time:timestamp"]
            lc = rec["_lc"]

            if lc in start_transitions:
                if resource not in active_since:
                    active_since[resource] = ts
                continue

            if lc == "suspend" and resource in active_since:
                active_since.pop(resource)
                continue

            if lc in terminal_transitions and resource in active_since:
                start_ts = active_since.pop(resource)
                if FTE_COMPLETE_ONLY and lc != "complete":
                    continue
                if ts > start_ts:
                    rows.append({
                        "case:concept:name": case_id,
                        "activity": activity,
                        "resource": resource,
                        "start_time": start_ts,
                        "complete_time": ts,
                        "duration_hours": (ts - start_ts).total_seconds() / 3600.0,
                        "terminal_transition": lc,
                    })

    seg = pd.DataFrame(rows)
    if seg.empty:
        return seg
    return seg[seg["duration_hours"] > 0].reset_index(drop=True)


def _gini_coefficient(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    v = v[v >= 0]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return np.nan
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * v).sum()) / (n * v.sum()) - (n + 1) / n)


def _wrf(busy_hours: np.ndarray) -> float:
    total = busy_hours.sum()
    if total == 0:
        return np.nan
    shares = busy_hours / total
    mean_share = shares.mean()
    weights = shares  # busy_i / total = share_i
    return float(np.sum(weights * np.abs(shares - mean_share)))


def compute_metrics(log_path: Path) -> dict:
    """Compute all evaluation metrics for a single simulated log."""
    df_raw = _load_log(log_path)

    # ── Basic counts ──────────────────────────────────────────────────────
    num_cases = df_raw["case:concept:name"].nunique()
    num_events = len(df_raw)
    num_resources = (
        df_raw["org:resource"].nunique()
        if "org:resource" in df_raw.columns
        else np.nan
    )

    # ── Case-level metrics ────────────────────────────────────────────────
    grp = df_raw.sort_values("time:timestamp").groupby("case:concept:name")
    case_start = grp["time:timestamp"].min()
    case_end = grp["time:timestamp"].max()
    n_events_per_case = grp.size()

    cycle_times = (case_end - case_start).dt.total_seconds() / 86_400
    cycle_time_mean = float(cycle_times.mean())
    cycle_time_median = float(cycle_times.median())
    cycle_time_p90 = float(np.percentile(cycle_times.dropna(), 90))

    span_days = (case_end.max() - case_start.min()).total_seconds() / 86_400
    throughput_per_day = float(num_cases / span_days) if span_days > 0 else np.nan

    events_per_case_mean = float(n_events_per_case.mean())

    # Handovers
    if "org:resource" in df_raw.columns:
        def _count_handovers(sub: pd.DataFrame) -> int:
            r = sub.sort_values("time:timestamp")["org:resource"].dropna().tolist()
            return sum(1 for a, b in zip(r, r[1:]) if a != b)

        handovers = grp.apply(_count_handovers)
        handovers_mean = float(handovers.mean())
    else:
        handovers_mean = np.nan

    # ── Resource-level metrics ────────────────────────────────────────────
    seg = _extract_resource_active_segments(df_raw)

    if seg.empty:
        return {
            "num_cases": num_cases,
            "num_events": num_events,
            "num_resources": num_resources,
            "cycle_time_mean": cycle_time_mean,
            "cycle_time_median": cycle_time_median,
            "cycle_time_p90": cycle_time_p90,
            "throughput_per_day": throughput_per_day,
            "handovers_mean": handovers_mean,
            "events_per_case_mean": events_per_case_mean,
            "total_busy_hours": np.nan,
            "total_fte_cost": np.nan,
            "cost_per_case": np.nan,
            "gini": np.nan,
            "wrf": np.nan,
            "productivity_mean": np.nan,
            "contribution_per_cost_mean": np.nan,
        }

    # Per-resource busy hours
    res_busy = (
        seg.groupby("resource")["duration_hours"]
        .sum()
        .reset_index()
        .rename(columns={"duration_hours": "busy_hours"})
    )
    busy_arr = res_busy["busy_hours"].values
    total_busy_hours = float(busy_arr.sum())
    total_fte_cost = total_busy_hours * COST_PER_FTE_HOUR
    cost_per_case = total_fte_cost / num_cases if num_cases > 0 else np.nan

    gini = _gini_coefficient(busy_arr)
    wrf = _wrf(busy_arr)

    # Productivity: completed_events / busy_hours per resource
    if "lifecycle:transition" in df_raw.columns:
        complete_events = df_raw[
            df_raw["lifecycle:transition"].astype(str).str.lower() == "complete"
        ]
    else:
        complete_events = df_raw

    event_counts = (
        complete_events.groupby("org:resource")
        .size()
        .reset_index(name="completed_events")
        .rename(columns={"org:resource": "resource"})
    )
    case_counts = (
        complete_events.groupby("org:resource")["case:concept:name"]
        .nunique()
        .reset_index()
        .rename(columns={"org:resource": "resource", "case:concept:name": "distinct_cases"})
    )

    res_features = (
        res_busy
        .merge(event_counts, on="resource", how="left")
        .merge(case_counts, on="resource", how="left")
        .fillna({"completed_events": 0, "distinct_cases": 0})
    )
    res_features["fte_cost"] = res_features["busy_hours"] * COST_PER_FTE_HOUR
    res_features["productivity"] = np.where(
        res_features["busy_hours"] > 0,
        res_features["completed_events"] / res_features["busy_hours"],
        0.0,
    )
    res_features["contribution_per_cost"] = np.where(
        res_features["fte_cost"] > 0,
        res_features["distinct_cases"] / res_features["fte_cost"],
        0.0,
    )

    productivity_mean = float(res_features["productivity"].mean())
    contribution_per_cost_mean = float(res_features["contribution_per_cost"].mean())

    return {
        "num_cases": num_cases,
        "num_events": num_events,
        "num_resources": num_resources,
        "cycle_time_mean": cycle_time_mean,
        "cycle_time_median": cycle_time_median,
        "cycle_time_p90": cycle_time_p90,
        "throughput_per_day": throughput_per_day,
        "handovers_mean": handovers_mean,
        "events_per_case_mean": events_per_case_mean,
        "total_busy_hours": total_busy_hours,
        "total_fte_cost": total_fte_cost,
        "cost_per_case": cost_per_case,
        "gini": gini,
        "wrf": wrf,
        "productivity_mean": productivity_mean,
        "contribution_per_cost_mean": contribution_per_cost_mean,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═════════════════════════════════════════════════════════════════════════════

METRIC_COLS = [
    "num_cases", "num_events", "num_resources",
    "cycle_time_mean", "cycle_time_median", "cycle_time_p90",
    "throughput_per_day", "handovers_mean", "events_per_case_mean",
    "total_busy_hours", "total_fte_cost", "cost_per_case",
    "gini", "wrf", "productivity_mean", "contribution_per_cost_mean",
]


def build_per_strategy(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run results into per-strategy mean ± std."""
    rows: list[dict] = []
    for strategy, grp in df_runs.groupby("strategy"):
        row: dict = {"strategy": strategy, "n_runs": len(grp)}
        for col in METRIC_COLS:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_vs_baseline(df_strategy: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Compare each non-baseline strategy against the baseline (mean values)."""
    baseline_row = df_strategy[df_strategy["strategy"] == baseline]
    if baseline_row.empty:
        print(f"  Warning: baseline '{baseline}' not found in strategies. Skipping vs_baseline.")
        return pd.DataFrame()
    baseline_row = baseline_row.iloc[0]

    rows: list[dict] = []
    for _, row in df_strategy.iterrows():
        if row["strategy"] == baseline:
            continue
        entry: dict = {"strategy": row["strategy"]}
        for col in METRIC_COLS:
            mean_col = f"{col}_mean"
            b_val = baseline_row[mean_col]
            s_val = row[mean_col]
            entry[f"{col}_value"] = s_val
            entry[f"{col}_baseline"] = b_val
            entry[f"{col}_diff"] = s_val - b_val if not (np.isnan(s_val) or np.isnan(b_val)) else np.nan
            if not np.isnan(b_val) and b_val != 0 and not np.isnan(s_val):
                entry[f"{col}_pct_change"] = (s_val - b_val) / abs(b_val) * 100
            else:
                entry[f"{col}_pct_change"] = np.nan
        rows.append(entry)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

STRATEGY_COLORS = {
    "greedy_random": "#5b8db8",
    "greedy_round_robin": "#7bb274",
    "greedy_shortest_queue": "#e8a838",
    "batch": "#d65f5f",
    "drl": "#9b72b0",
}

STRATEGY_ORDER = [
    "greedy_random", "greedy_round_robin", "greedy_shortest_queue", "batch", "drl",
]


def _bar_chart(
    df_strategy: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    lower_is_better: bool = True,
) -> None:
    """Grouped bar chart with error bars (std across runs)."""
    # Sort strategies to consistent order
    order = [s for s in STRATEGY_ORDER if s in df_strategy["strategy"].values]
    df_plot = df_strategy.set_index("strategy").loc[order].reset_index()

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_plot))
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in df_plot["strategy"]]

    bars = ax.bar(
        x,
        df_plot[mean_col],
        yerr=df_plot[std_col],
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    # Annotate bars with values
    for bar, val in zip(bars, df_plot[mean_col]):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + df_plot[std_col].max() * 0.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Highlight best strategy
    valid = df_plot[mean_col].dropna()
    if not valid.empty:
        best_idx = valid.idxmin() if lower_is_better else valid.idxmax()
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in df_plot["strategy"]],
        fontsize=9,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_plots(df_strategy: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("gini", "Gini Coefficient", "Workload Gini Coefficient by Strategy", "gini_by_strategy.png", True),
        ("cost_per_case", "Cost per Case (€)", "FTE Cost per Case by Strategy", "cost_per_case_by_strategy.png", True),
        ("cycle_time_mean", "Mean Cycle Time (days)", "Mean Cycle Time by Strategy", "cycle_time_by_strategy.png", True),
        ("throughput_per_day", "Throughput (cases/day)", "Throughput by Strategy", "throughput_by_strategy.png", False),
        ("wrf", "WRF", "Weighted Resource Fairness by Strategy", "wrf_by_strategy.png", True),
        ("handovers_mean", "Mean Handovers per Case", "Handovers by Strategy", "handovers_by_strategy.png", True),
        ("total_fte_cost", "Total FTE Cost (€)", "Total FTE Cost by Strategy", "total_fte_cost_by_strategy.png", True),
        ("total_busy_hours", "Total Busy Hours", "Total Busy Hours by Strategy", "total_busy_hours_by_strategy.png", True),
    ]

    for metric, ylabel, title, filename, lower_is_better in plot_specs:
        _bar_chart(df_strategy, metric, ylabel, title, plots_dir / filename, lower_is_better)
        print(f"  Plot: {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("Batch Evaluation — Resource Allocation Strategies")
    print("=" * 70)
    print(f"Base dir          : {BASE_DIR}")
    print(f"Cost per FTE hour : €{COST_PER_FTE_HOUR:.2f}")
    print(f"FTE complete only : {FTE_COMPLETE_ONLY}")
    print(f"Baseline strategy : {BASELINE_STRATEGY}")
    print()

    # ── Discover logs ─────────────────────────────────────────────────────
    logs = discover_logs(BASE_DIR)
    if not logs:
        print("No simulated logs found. Exiting.")
        return

    print(f"Discovered {len(logs)} logs:")
    for entry in logs:
        print(f"  {entry['strategy']} run{entry['run']} → {entry['path'].relative_to(REPO_ROOT)}")
    print()

    # ── Compute metrics ───────────────────────────────────────────────────
    results: list[dict] = []
    for i, entry in enumerate(logs, 1):
        label = f"{entry['strategy']} run{entry['run']}"
        print(f"[{i}/{len(logs)}] Computing metrics for {label} ...")
        metrics = compute_metrics(entry["path"])
        metrics["strategy"] = entry["strategy"]
        metrics["variant"] = entry["variant"]
        metrics["run"] = entry["run"]
        results.append(metrics)

    # ── Build DataFrames ──────────────────────────────────────────────────
    df_runs = pd.DataFrame(results)
    # Reorder columns: identifiers first
    id_cols = ["strategy", "variant", "run"]
    df_runs = df_runs[id_cols + METRIC_COLS]

    df_strategy = build_per_strategy(df_runs)
    df_vs_baseline = build_vs_baseline(df_strategy, BASELINE_STRATEGY)

    # ── Print summary ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Per-Strategy Summary (mean ± std)")
    print("=" * 70)
    display_cols = ["strategy", "n_runs"]
    for col in ["cycle_time_mean", "cost_per_case", "gini", "wrf", "throughput_per_day"]:
        display_cols.extend([f"{col}_mean", f"{col}_std"])
    print(df_strategy[display_cols].to_string(index=False))
    print()

    # ── Export Excel ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUTPUT_DIR / "batch_evaluation_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_runs.to_excel(writer, sheet_name="per_run", index=False)
        df_strategy.to_excel(writer, sheet_name="per_strategy", index=False)
        if not df_vs_baseline.empty:
            df_vs_baseline.to_excel(writer, sheet_name="vs_baseline", index=False)
    print(f"Excel exported: {xlsx_path.relative_to(REPO_ROOT)}")

    # ── Generate plots ────────────────────────────────────────────────────
    print("\nGenerating plots:")
    generate_plots(df_strategy, PLOTS_DIR)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
