"""
Compare Throughput (relative to arrived cases) across multiple simulated models/strategies.

The window ends at the time when the *first K completed cases* have been finished
(i.e., the K-th completion time).

Metric:
  throughput_per_day = K / window_days
  arrived_cases = number of distinct cases whose first event is <= window_end
  score = throughput_per_day / arrived_cases

Lower/higher selection:
  Higher score means more completions per day relative to how many cases arrived.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    col_title: str
    rel_dir: str  # relative to the models root (contains run*/simulated_log.csv)


MODEL_SPECS: list[ModelSpec] = [
    # Map folder names to the column headers used in the repo's allocation tables.
    ModelSpec("R-RMA", "greedy_10k/random"),
    ModelSpec("R-RRA", "greedy_10k/round_robin"),
    ModelSpec("R-SHQ", "greedy_10k/shortest_queue"),
    ModelSpec("1-Batch-1", "batch_10k"),
    ModelSpec("DRL", "drl_10k"),
]


def parse_timestamp(ts: str) -> datetime:
    # Example: "2016-01-05 04:53:43.758122+00:00"
    # datetime.fromisoformat supports the space between date and time.
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # Fallback for a potential "Z" suffix.
        ts2 = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts2)
    # Ensure tz-aware for safe subtraction
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute_throughput_relative_first_k_completed_cases(csv_path: Path, k: int = 100) -> float:
    """
    Compute makespan for the first K completed cases.
    - case_start_all: min timestamp across ALL events of the case
    - case_end_complete: max timestamp across events with lifecycle:transition == "complete"
    """
    case_start_all: dict[str, datetime] = {}  # earliest event per case
    case_end_complete: dict[str, datetime] = {}  # latest "complete" event per case

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get("case:concept:name")
            ts = row.get("time:timestamp")
            if not case_id or not ts:
                continue

            dt = parse_timestamp(ts)

            prev_start = case_start_all.get(case_id)
            if prev_start is None or dt < prev_start:
                case_start_all[case_id] = dt

            lc = row.get("lifecycle:transition", "")
            if isinstance(lc, str) and lc.lower() == "complete":
                prev_end = case_end_complete.get(case_id)
                if prev_end is None or dt > prev_end:
                    case_end_complete[case_id] = dt

    completed = [
        (cid, end_dt)
        for cid, end_dt in case_end_complete.items()
        if cid in case_start_all
    ]
    if not completed:
        return float("nan")

    completed.sort(key=lambda x: x[1])  # by completion time
    if len(completed) < k:
        return float("nan")

    selected = completed[:k]

    window_end = max(end_dt for _cid, end_dt in selected)  # equals the k-th completion time
    window_start = min(case_start_all.values())
    window_days = (window_end - window_start).total_seconds() / 86400.0
    if window_days <= 0:
        return float("nan")

    throughput_per_day = k / window_days
    arrived_cases = sum(1 for cid, start_dt in case_start_all.items() if start_dt <= window_end)
    if arrived_cases <= 0:
        return float("nan")

    return throughput_per_day / arrived_cases


def collect_run_csvs(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []

    run_dirs = []
    for p in model_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.lower().startswith("run"):
            continue
        run_dirs.append(p)

    def natural_key(path: Path) -> int:
        # run1 -> 1
        s = path.name.lower().replace("run", "").strip()
        try:
            return int(s)
        except ValueError:
            return 10**9

    run_dirs.sort(key=natural_key)
    csv_paths = []
    for rd in run_dirs:
        csv_path = rd / "simulated_log.csv"
        if csv_path.exists():
            csv_paths.append(csv_path)
    return csv_paths


def fmt_mean_pm_std(mean: float, std: float, decimals: int = 2) -> str:
    if math.isnan(mean) or math.isnan(std):
        return "$nan$"
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a Makespan comparison table across multiple simulated models/strategies.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("/Users/laurensohl/Downloads/simulated_ELs"),
        help="Root folder containing the model strategy subfolders (e.g., greedy_10k/random/run1/simulated_log.csv).",
    )
    parser.add_argument("--k", type=int, default=100, help="First K completed cases to consider.")
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("evaluation/output/model_comparison/makespan_first100_table.tex"),
        help="Output LaTeX file (written relative to repo root).",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "Throughput (per day, normalized by arrived cases) comparison on simulated logs "
            "(mean $\\pm$ std over replications; first 100 completed cases, window ends at the 100th completion). "
            "Best value (highest score) in bold."
        ),
    )
    parser.add_argument("--label", type=str, default="tab:throughput-perday-arrived-first100")
    args = parser.parse_args()

    # Resolve output relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    out_tex_path = args.out_tex if args.out_tex.is_absolute() else (repo_root / args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    per_model_values: dict[str, list[float]] = {}
    per_model_runs: dict[str, list[Path]] = {}

    for spec in MODEL_SPECS:
        model_dir = args.models_root / spec.rel_dir
        csv_paths = collect_run_csvs(model_dir)
        per_model_runs[spec.col_title] = csv_paths
        vals: list[float] = []
        for p in csv_paths:
            vals.append(compute_throughput_relative_first_k_completed_cases(p, k=args.k))
        per_model_values[spec.col_title] = vals

    # Compute mean/std
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in per_model_values:
        vals = [v for v in per_model_values[col] if not math.isnan(v)]
        if not vals:
            means[col] = float("nan")
            stds[col] = float("nan")
            continue
        means[col] = float(sum(vals) / len(vals))
        if len(vals) >= 2:
            stds[col] = float(statistics.stdev(vals))  # sample std (ddof=1)
        else:
            stds[col] = 0.0

    # Higher score is better
    best_col = max((c for c in means if not math.isnan(means[c])), key=lambda c: means[c], default=None)

    def cell_for(col: str) -> str:
        base = fmt_mean_pm_std(means[col], stds[col], decimals=2)
        if best_col is not None and col == best_col and not math.isnan(means[col]):
            # Turn $mean \pm std$ into $\mathbf{mean \pm std}$.
            inner = f"{means[col]:.2f} \\pm {stds[col]:.2f}"
            return f"$\\mathbf{{{inner}}}$"
        return base

    lines: list[str] = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(f"\\caption{{{args.caption}}}")
    lines.append(f"\\label{{{args.label}}}")
    lines.append(r"\begin{tabular}{l r r r r r}")
    lines.append(r"\toprule")
    header_cells = " & ".join([r"\textbf{Metric}"] + [fr"\textbf{{{s.col_title}}}" for s in MODEL_SPECS])
    lines.append(header_cells + r" \\")
    lines.append(r"\midrule")

    row_label = "Throughput (cases/day) / arrived cases"
    row = row_label + " & " + " & ".join(cell_for(s.col_title) for s in MODEL_SPECS) + r" \\"
    lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")  # newline

    out_tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_tex_path}")


if __name__ == "__main__":
    main()

