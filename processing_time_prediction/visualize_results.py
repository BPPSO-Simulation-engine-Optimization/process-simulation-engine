"""
Visualisierung der Metriken aus den activity_results.
Liest alle metrics.txt Dateien und erstellt Balkendiagramme für jede Metrik.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ─── Konfiguration ───────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "XGBoost_evaluation_output", "activity_results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_LABELS = {
    "mae_hours": "MAE (Stunden)",
    "rmse_hours": "RMSE (Stunden)",
    "mae_log": "MAE (Log Scale)",
    "rmse_log": "RMSE (Log-Skala)",
}


# ─── Metriken aus allen Unterordnern einlesen ────────────────────────────────
def parse_metrics(results_dir: str) -> pd.DataFrame:
    """Liest alle metrics.txt Dateien und gibt ein DataFrame zurück."""
    rows = []
    for activity_name in sorted(os.listdir(results_dir)):
        activity_dir = os.path.join(results_dir, activity_name)
        metrics_file = os.path.join(activity_dir, "metrics.txt")
        if not os.path.isfile(metrics_file):
            continue

        metrics = {"activity": activity_name}
        with open(metrics_file, "r") as f:
            for line in f:
                match = re.match(r"(\w+):\s+([\d.eE+-]+)", line.strip())
                if match:
                    metrics[match.group(1)] = float(match.group(2))
        if len(metrics) > 1:  # mindestens eine Metrik gefunden
            rows.append(metrics)

    return pd.DataFrame(rows)


# ─── Einzelne Balkendiagramme pro Metrik ──────────────────────────────────────
def plot_single_metric(df: pd.DataFrame, metric: str, label: str, output_dir: str):
    """Erstellt ein horizontales Balkendiagramm für eine einzelne Metrik."""
    sorted_df = df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_df) * 0.4)))

    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(sorted_df)))
    bars = ax.barh(sorted_df["activity"], sorted_df[metric], color=colors, edgecolor="white", linewidth=0.5)

    # Werte an die Balken schreiben
    max_val = sorted_df[metric].max()
    for bar, val in zip(bars, sorted_df[metric]):
        offset = max_val * 0.01
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    # Größere Schriftgröße für mae_log (größer als die Zahlen fontsize=8)
    if metric == "mae_log":
        xlabel_fontsize = 24
        title_fontsize = 26
        ylabel_fontsize = 13
    else:
        xlabel_fontsize = 12
        title_fontsize = 14
        ylabel_fontsize = 9

    ax.set_xlabel(label, fontsize=xlabel_fontsize)
    ax.set_title(f"{label} per Activity", fontsize=title_fontsize, fontweight="bold")
    ax.tick_params(axis="y", labelsize=ylabel_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{metric}.png")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Gespeichert: {filename}")


# ─── Übersichts-Dashboard (2×2) ──────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, output_dir: str):
    """Erstellt ein 2×2 Dashboard mit allen vier Metriken."""
    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(20, max(10, len(df) * 0.35)))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        sorted_df = df.sort_values(metric, ascending=True)
        colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(sorted_df)))
        ax.barh(sorted_df["activity"], sorted_df[metric], color=colors, edgecolor="white", linewidth=0.5)
        # Größere Schriftgröße für mae_log (größer als die Zahlen fontsize=8)
        if metric == "mae_log":
            ax.set_xlabel(METRIC_LABELS[metric], fontsize=20)
            ax.set_title(METRIC_LABELS[metric], fontsize=22, fontweight="bold")
        else:
            ax.set_xlabel(METRIC_LABELS[metric], fontsize=10)
            ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Metriken-Übersicht aller Aktivitäten", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    filename = os.path.join(output_dir, "dashboard_overview.png")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Gespeichert: {filename}")


# ─── Top-N Vergleich (schlechteste Aktivitäten) ──────────────────────────────
def plot_top_n(df: pd.DataFrame, output_dir: str, n: int = 10):
    """Zeigt die Top-N Aktivitäten mit den höchsten Fehlern."""
    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        top = df.nlargest(n, metric)
        top_sorted = top.sort_values(metric, ascending=True)
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_sorted)))
        bars = ax.barh(top_sorted["activity"], top_sorted[metric], color=colors, edgecolor="white", linewidth=0.5)

        max_val = top_sorted[metric].max()
        for bar, val in zip(bars, top_sorted[metric]):
            ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        # Größere Schriftgröße für mae_log (größer als die Zahlen fontsize=8)
        if metric == "mae_log":
            ax.set_xlabel(METRIC_LABELS[metric], fontsize=20)
            ax.set_title(f"Top {n} Highest {METRIC_LABELS[metric]}", fontsize=22, fontweight="bold")
        else:
            ax.set_xlabel(METRIC_LABELS[metric], fontsize=10)
            ax.set_title(f"Top {n} höchste {METRIC_LABELS[metric]}", fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Top {n} Aktivitäten mit höchstem Fehler", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"top_{n}_worst.png")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Gespeichert: {filename}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Visualisierung der Activity-Metriken")
    print("=" * 60)

    # Metriken einlesen
    df = parse_metrics(RESULTS_DIR)
    print(f"\n{len(df)} Aktivitäten gefunden.\n")
    print(df.to_string(index=False))
    print()

    # Einzelne Plots pro Metrik
    print("Erstelle Einzeldiagramme pro Metrik...")
    for metric, label in METRIC_LABELS.items():
        if metric in df.columns:
            plot_single_metric(df, metric, label, OUTPUT_DIR)

    # Dashboard
    print("\nErstelle Dashboard-Übersicht...")
    plot_dashboard(df, OUTPUT_DIR)

    # Top-N schlechteste
    print("\nErstelle Top-10 Fehler-Übersicht...")
    plot_top_n(df, OUTPUT_DIR, n=10)

    print(f"\n{'=' * 60}")
    print(f"Alle Plots gespeichert unter: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
