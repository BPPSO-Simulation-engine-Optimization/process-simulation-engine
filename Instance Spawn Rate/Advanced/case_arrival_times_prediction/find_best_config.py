"""
Iterativ verschiedene Config-Kombinationen testen und die beste finden.
"""
import os
os.environ["THREADPOOLCTL_DISABLE"] = "1"
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from typing import Dict, List, Tuple

# Import modules (relative imports)
from .config import SimulationConfig
from .pipeline import CaseInterarrivalPipeline
from .preprocessing import DailySequence
from .metrics import flatten_days, sqrt_cadd, cadd_distance
from .runner import interarrival_stats_intraday_only
import pickle

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

def evaluate_config(df_gt, arrivals_gt, config: SimulationConfig, config_id: str) -> Dict:
    """Evaluiert eine Config-Kombination und gibt Metriken zurück."""
    print(f"\n{'='*80}")
    print(f"Teste Config: {config_id}")
    print(f"{'='*80}")
    print(f"  window_size={config.window_size}, kmax={config.kmax}, L={config.L}")
    print(f"  dbscan_eps={config.dbscan_eps}, dbscan_min_samples={config.dbscan_min_samples}")
    
    try:
        # Trainiere Modell
        pipe = CaseInterarrivalPipeline(config)
        pipe.fit(df_gt)
        
        # Simuliere 1000 Cases
        D_gt = convert_to_daily_sequence(arrivals_gt)
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
        D_pred = convert_to_daily_sequence(arrivals_pred)
        
        # Berechne Metriken
        cadd_score = cadd_distance(arrivals_gt, arrivals_pred)
        sqrt_cadd_score = sqrt_cadd(D_gt, D_pred)
        
        # Volumen-Vergleich
        volumes_gt = [len(day) for day in D_gt]
        volumes_pred = [len(day) for day in D_pred]
        volume_diff = abs(np.mean(volumes_pred) - np.mean(volumes_gt))
        volume_diff_pct = (volume_diff / np.mean(volumes_gt) * 100) if np.mean(volumes_gt) > 0 else 0
        
        # Interarrival Statistiken
        stats_gt = interarrival_stats_intraday_only(D_gt, unit="seconds")
        stats_pred = interarrival_stats_intraday_only(D_pred, unit="seconds")
        
        interarrival_diff = 0.0
        if stats_gt and stats_pred:
            # Vergleiche Mittelwerte
            interarrival_diff = abs(stats_pred['mean'] - stats_gt['mean'])
        
        # Gesamt-Score (niedriger ist besser)
        # Kombiniere CADD, Volumen-Differenz und Interarrival-Differenz
        total_score = sqrt_cadd_score + (volume_diff_pct / 10.0) + (interarrival_diff / 100.0)
        
        result = {
            'config_id': config_id,
            'sqrt_cadd': sqrt_cadd_score,
            'cadd': cadd_score,
            'volume_diff_pct': volume_diff_pct,
            'interarrival_diff': interarrival_diff,
            'total_score': total_score,
            'window_size': config.window_size,
            'kmax': config.kmax,
            'L': config.L,
            'dbscan_eps': config.dbscan_eps,
            'dbscan_min_samples': config.dbscan_min_samples,
            'success': True
        }
        
        print(f"  ✓ sqrt_CADD: {sqrt_cadd_score:.4f}")
        print(f"  ✓ Volume Diff: {volume_diff_pct:.2f}%")
        print(f"  ✓ Total Score: {total_score:.4f}")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
        return {
            'config_id': config_id,
            'success': False,
            'error': str(e)
        }

# Main
base_dir = Path("/teamspace/studios/this_studio")
ground_truth_path = base_dir / "process-simulation-engine/integration/output/ground_truth_log.csv"

print("="*80)
print("ITERATIVE CONFIG-OPTIMIERUNG")
print("="*80)

# Lade Ground Truth
print("\n[1/3] Lade Ground Truth...")
df_gt = pd.read_csv(str(ground_truth_path))
arrivals_gt = extract_case_arrivals_from_log(df_gt)
print(f"   Ground Truth Cases: {len(arrivals_gt)}")

# Definiere Parameter-Räume für Grid Search
param_grid = {
    'window_size': [10, 14, 18],
    'kmax': [5, 6, 8],
    'L': [5, 6, 8],
    'dbscan_eps': [0.6, 0.7, 0.8],
    'dbscan_min_samples': [2, 3]
}

print(f"\n[2/3] Teste {len(list(itertools.product(*param_grid.values())))} Config-Kombinationen...")

# Teste alle Kombinationen
results = []
config_num = 0

for window_size, kmax, L, dbscan_eps, dbscan_min_samples in itertools.product(
    param_grid['window_size'],
    param_grid['kmax'],
    param_grid['L'],
    param_grid['dbscan_eps'],
    param_grid['dbscan_min_samples']
):
    config_num += 1
    config_id = f"Config_{config_num:03d}"
    
    config = SimulationConfig(
        train_ratio=0.8,
        window_size=window_size,
        kmax=kmax,
        z_values=(1.0, 0.8, 0.6, 0.4, 0.2, 0.1),
        L=L,
        kernel="gaussian",
        min_samples_kde=3,
        bandwidth_k_values=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        bandwidth_val_ratio=0.3,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        verbose=False,
        random_state=42
    )
    
    result = evaluate_config(df_gt, arrivals_gt, config, config_id)
    results.append(result)
    
    # Zeige Fortschritt
    if config_num % 10 == 0:
        successful = [r for r in results if r.get('success', False)]
        if successful:
            best_so_far = min(successful, key=lambda x: x.get('total_score', float('inf')))
            print(f"\n  Fortschritt: {config_num} Configs getestet")
            print(f"  Bester Score bisher: {best_so_far['total_score']:.4f} ({best_so_far['config_id']})")

print(f"\n[3/3] Auswertung...")

# Filtere erfolgreiche Ergebnisse
successful_results = [r for r in results if r.get('success', False)]

if not successful_results:
    print("  ✗ Keine erfolgreichen Configs gefunden!")
    sys.exit(1)

# Sortiere nach total_score (niedriger ist besser)
successful_results.sort(key=lambda x: x.get('total_score', float('inf')))

print(f"\n{'='*80}")
print("TOP 10 BESTE CONFIGS:")
print(f"{'='*80}")
print(f"{'Rank':<6} {'Config':<12} {'√CADD':<10} {'Vol Diff %':<12} {'Total Score':<12} {'Parameter':<50}")
print("-" * 80)

for i, result in enumerate(successful_results[:10], 1):
    params = f"w={result['window_size']}, k={result['kmax']}, L={result['L']}, eps={result['dbscan_eps']}, min={result['dbscan_min_samples']}"
    print(f"{i:<6} {result['config_id']:<12} {result['sqrt_cadd']:<10.4f} {result['volume_diff_pct']:<12.2f} {result['total_score']:<12.4f} {params}")

# Bester Config
best_config = successful_results[0]
print(f"\n{'='*80}")
print("BESTE CONFIG:")
print(f"{'='*80}")
print(f"Config ID: {best_config['config_id']}")
print(f"Total Score: {best_config['total_score']:.4f}")
print(f"√CADD: {best_config['sqrt_cadd']:.4f}")
print(f"Volume Diff: {best_config['volume_diff_pct']:.2f}%")
print(f"\nParameter:")
print(f"  window_size: {best_config['window_size']}")
print(f"  kmax: {best_config['kmax']}")
print(f"  L: {best_config['L']}")
print(f"  dbscan_eps: {best_config['dbscan_eps']}")
print(f"  dbscan_min_samples: {best_config['dbscan_min_samples']}")

# Speichere Ergebnisse
results_df = pd.DataFrame(successful_results)
results_path = base_dir / "process-simulation-engine/Instance Spawn Rate/Advanced/case_arrival_times_prediction/config_optimization_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✓ Ergebnisse gespeichert in: {results_path}")

print("\n" + "="*80)
print("OPTIMIERUNG ABGESCHLOSSEN")
print("="*80)
