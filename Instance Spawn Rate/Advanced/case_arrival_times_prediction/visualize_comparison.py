"""
Erstellt Visualisierungen zum Vergleich von Ground Truth und Prediction.
"""
import os
os.environ["THREADPOOLCTL_DISABLE"] = "1"
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Import modules (relative imports)
from .preprocessing import DailySequence
from .metrics import flatten_days
from .runner import interarrival_stats_intraday_only

def extract_case_arrivals_from_log(df):
    """Extrahiert die ersten Event-Timestamps pro Case."""
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed', utc=True)
    first_events = df.groupby('case:concept:name')['time:timestamp'].first()
    arrivals = sorted(first_events.tolist())
    return arrivals

def convert_to_daily_sequence(arrivals):
    """Konvertiert Arrivals in DailySequence."""
    if len(arrivals) == 0:
        return []
    df = pd.DataFrame({'timestamp': arrivals})
    df['date'] = df['timestamp'].dt.date
    daily_sequence = []
    for date, group in df.groupby('date'):
        day_arrivals = group['timestamp'].tolist()
        daily_sequence.append(day_arrivals)
    return daily_sequence

def create_comparison_plots(arrivals_gt, arrivals_pred, output_dir: Path):
    """Erstellt Vergleichs-Diagramme."""
    print("Erstelle Vergleichs-Diagramme...")
    
    # Konvertiere zu DailySequence
    D_gt = convert_to_daily_sequence(arrivals_gt)
    D_pred = convert_to_daily_sequence(arrivals_pred)
    
    # Erstelle Output-Verzeichnis
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Tägliche Volumen-Vergleich (Zeitreihe)
    print("  1. Tägliche Volumen-Zeitreihe...")
    volumes_gt = [len(day) for day in D_gt]
    volumes_pred = [len(day) for day in D_pred]
    
    # Erstelle Datums-Arrays
    dates_gt = []
    dates_pred = []
    for i, day in enumerate(D_gt):
        if len(day) > 0:
            dates_gt.append(pd.to_datetime(day[0]).date())
    
    for i, day in enumerate(D_pred):
        if len(day) > 0:
            dates_pred.append(pd.to_datetime(day[0]).date())
    
    # Pad auf gleiche Länge
    max_len = max(len(dates_gt), len(dates_pred))
    if len(dates_gt) < max_len:
        dates_gt.extend([dates_gt[-1] + pd.Timedelta(days=i+1) for i in range(max_len - len(dates_gt))])
        volumes_gt.extend([0] * (max_len - len(volumes_gt)))
    if len(dates_pred) < max_len:
        dates_pred.extend([dates_pred[-1] + pd.Timedelta(days=i+1) for i in range(max_len - len(dates_pred))])
        volumes_pred.extend([0] * (max_len - len(volumes_pred)))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates_gt[:len(volumes_gt)], volumes_gt, label='Ground Truth', linewidth=2, alpha=0.8)
    ax.plot(dates_pred[:len(volumes_pred)], volumes_pred, label='Prediction', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_xlabel('Datum', fontsize=12)
    ax.set_ylabel('Anzahl Cases pro Tag', fontsize=12)
    ax.set_title('Tägliche Volumen-Vergleich: Ground Truth vs Prediction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates_gt)//10)))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'daily_volume_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Interarrival-Zeit Verteilung (nur Histogramm)
    print("  2. Interarrival-Zeit Verteilung...")
    interarrivals_gt = []
    interarrivals_pred = []
    
    for day in D_gt:
        if len(day) >= 2:
            timestamps = pd.to_datetime(day)
            arr = np.array(sorted(timestamps), dtype="datetime64[ns]")
            diffs = np.diff(arr).astype("timedelta64[s]").astype(float) / 3600.0  # Stunden
            diffs = diffs[diffs > 0]
            interarrivals_gt.extend(diffs.tolist())
    
    for day in D_pred:
        if len(day) >= 2:
            timestamps = pd.to_datetime(day)
            arr = np.array(sorted(timestamps), dtype="datetime64[ns]")
            diffs = np.diff(arr).astype("timedelta64[s]").astype(float) / 3600.0  # Stunden
            diffs = diffs[diffs > 0]
            interarrivals_pred.extend(diffs.tolist())
    
    # Nur Histogramm
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bins = np.logspace(-2, 2, 50)  # Logarithmische Bins
    ax.hist(interarrivals_gt, bins=bins, alpha=0.6, label='Ground Truth', density=True, color='blue', edgecolor='black', linewidth=0.5)
    ax.hist(interarrivals_pred, bins=bins, alpha=0.6, label='Prediction', density=True, color='red', edgecolor='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Interarrival Time (hours)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')
    ax.set_title('Interarrival Time Distribution: Ground Truth vs Prediction', fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'interarrival_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Kumulative Arrival-Verteilung
    print("  3. Kumulative Arrival-Verteilung...")
    # Normalisiere auf gleichen Zeitraum
    start_time = min(arrivals_gt[0], arrivals_pred[0])
    end_time = max(arrivals_gt[-1], arrivals_pred[-1])
    
    def get_cumulative_distribution(arrivals, start, end):
        arrivals_sorted = sorted([pd.to_datetime(a) for a in arrivals])
        time_delta = (end - start).total_seconds() / 3600.0  # Stunden
        bins = np.linspace(0, time_delta, 100)
        hours = [(pd.to_datetime(a) - start).total_seconds() / 3600.0 for a in arrivals_sorted]
        counts, _ = np.histogram(hours, bins=bins)
        cumulative = np.cumsum(counts)
        if len(cumulative) > 0 and cumulative[-1] > 0:
            cumulative = cumulative / cumulative[-1]  # Normalisiere
        return bins[:-1], cumulative
    
    bins_gt, cum_gt = get_cumulative_distribution(arrivals_gt, start_time, end_time)
    bins_pred, cum_pred = get_cumulative_distribution(arrivals_pred, start_time, end_time)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bins_gt, cum_gt, label='Ground Truth', linewidth=2.5, alpha=0.8)
    ax.plot(bins_pred, cum_pred, label='Prediction', linewidth=2.5, alpha=0.8, linestyle='--')
    ax.set_xlabel('Zeit seit Start (Stunden)', fontsize=12)
    ax.set_ylabel('Kumulative Verteilung (normalisiert)', fontsize=12)
    ax.set_title('Kumulative Arrival-Verteilung: Ground Truth vs Prediction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Statistik-Vergleich (Bar Chart) - ohne std
    print("  4. Statistik-Vergleich...")
    stats_gt = interarrival_stats_intraday_only(D_gt, unit="seconds")
    stats_pred = interarrival_stats_intraday_only(D_pred, unit="seconds")
    
    if stats_gt and stats_pred:
        metrics = ['q05', 'q25', 'q50', 'q75', 'q95', 'mean']  # Mit Q05, Q95 und Mean
        labels = ['Q05', 'Q25', 'Median', 'Q75', 'Q95', 'Mean']
        # Werte sind bereits in Sekunden, konvertiere zu Stunden für Anzeige
        gt_values = [stats_gt[m] / 3600.0 for m in metrics]  # Konvertiere zu Stunden
        pred_values = [stats_pred[m] / 3600.0 for m in metrics]
        
        # Debug: Zeige Werte in Sekunden (wie in Terminal)
        print(f"\n  Debug - Ground Truth (Sekunden):")
        for m, l in zip(metrics, labels):
            print(f"    {l}: {stats_gt[m]:.4f}")
        print(f"  Debug - Prediction (Sekunden):")
        for m, l in zip(metrics, labels):
            print(f"    {l}: {stats_pred[m]:.4f}")
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars1 = ax.bar(x - width/2, gt_values, width, label='Ground Truth', alpha=0.8, color='blue', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, pred_values, width, label='Prediction', alpha=0.8, color='red', edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Statistics', fontsize=16, fontweight='bold')
        ax.set_ylabel('Value (hours)', fontsize=16, fontweight='bold')
        ax.set_title('Interarrival Statistics Comparison', fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.legend(fontsize=14, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', which='major', labelsize=14)
        
        # Füge Werte über den Bars hinzu
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'statistics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Volumen-Statistik-Vergleich
    print("  5. Volumen-Statistik-Vergleich...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    volume_stats = {
        'Mittelwert': [np.mean(volumes_gt), np.mean(volumes_pred)],
        'Median': [np.median(volumes_gt), np.median(volumes_pred)],
        'Std': [np.std(volumes_gt), np.std(volumes_pred)],
        'Min': [np.min(volumes_gt), np.min(volumes_pred)],
        'Max': [np.max(volumes_gt), np.max(volumes_pred)]
    }
    
    x = np.arange(len(volume_stats))
    width = 0.35
    
    gt_vols = [volume_stats[k][0] for k in volume_stats.keys()]
    pred_vols = [volume_stats[k][1] for k in volume_stats.keys()]
    
    bars1 = ax.bar(x - width/2, gt_vols, width, label='Ground Truth', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, pred_vols, width, label='Prediction', alpha=0.8, color='red')
    
    ax.set_xlabel('Statistik', fontsize=12)
    ax.set_ylabel('Anzahl Cases', fontsize=12)
    ax.set_title('Tägliche Volumen-Statistiken Vergleich', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(volume_stats.keys())
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Füge Werte über den Bars hinzu
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'volume_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Alle Diagramme gespeichert in: {output_dir}")

# Main
base_dir = Path("/teamspace/studios/this_studio")
ground_truth_path = base_dir / "process-simulation-engine/integration/output/ground_truth_log.csv"
model_path = base_dir / "process-simulation-engine/Instance Spawn Rate/Advanced/case_arrival_times_prediction/models/case_arrival_model.pkl"
output_dir = base_dir / "process-simulation-engine/Instance Spawn Rate/Advanced/case_arrival_times_prediction/comparison_plots"

print("="*80)
print("VISUALISIERUNG: GROUND TRUTH vs PREDICTION")
print("="*80)

# Lade Ground Truth
print("\n[1/3] Lade Ground Truth...")
df_gt = pd.read_csv(str(ground_truth_path))
arrivals_gt = extract_case_arrivals_from_log(df_gt)
print(f"   Ground Truth Cases: {len(arrivals_gt)}")

# Lade Modell und generiere Prediction
print("\n[2/3] Generiere Prediction...")
if not model_path.exists():
    print(f"   ✗ Modell nicht gefunden: {model_path}")
    print("   Bitte zuerst compare_prediction.py ausführen!")
    sys.exit(1)

from .config import SimulationConfig
from .pipeline import CaseInterarrivalPipeline
import pickle

with open(str(model_path), "rb") as f:
    pipe = pickle.load(f)

# Simuliere 1000 Cases
start_date = arrivals_gt[0]
target_cases = 1000

arrivals_pred = []
current_days = 50
max_iterations = 20
iteration = 0

while len(arrivals_pred) < target_cases and iteration < max_iterations:
    D_sim = pipe.simulate_days(N_hat=current_days, start_date=start_date)
    arrivals_pred = flatten_days(D_sim)
    
    if len(arrivals_pred) < target_cases:
        cases_per_day = len(arrivals_pred) / current_days if current_days > 0 else 1
        needed_days = int((target_cases - len(arrivals_pred)) / cases_per_day) + 10
        current_days = needed_days
    iteration += 1

arrivals_pred = arrivals_pred[:target_cases]
print(f"   Prediction Cases: {len(arrivals_pred)}")

# Erstelle Diagramme
print("\n[3/3] Erstelle Vergleichs-Diagramme...")
create_comparison_plots(arrivals_gt, arrivals_pred, output_dir)

print("\n" + "="*80)
print("VISUALISIERUNG ABGESCHLOSSEN")
print("="*80)
