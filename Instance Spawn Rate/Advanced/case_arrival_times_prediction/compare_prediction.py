"""
Vergleicht eine lange Case Arrival Times Prediction mit Ground Truth.
"""
import os
os.environ["THREADPOOLCTL_DISABLE"] = "1"
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import modules (relative imports)
from .runner import run, interarrival_stats_intraday_only
from .preprocessing import DailySequence
from .metrics import flatten_days, sqrt_cadd, cadd_distance
from .config import SimulationConfig
from .pipeline import CaseInterarrivalPipeline
import pickle

def extract_case_arrivals_from_log(df):
    """Extrahiert die ersten Event-Timestamps pro Case."""
    # Verwende format='mixed' um verschiedene Timestamp-Formate zu handhaben
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

# Main - use absolute paths
script_dir = Path(__file__).parent.absolute()
# Base directory is /teamspace/studios/this_studio
base_dir = Path("/teamspace/studios/this_studio")
ground_truth_path = base_dir / "process-simulation-engine/integration/output/ground_truth_log.csv"
model_path = script_dir / "models/case_arrival_model.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CASE ARRIVAL TIMES PREDICTION vs GROUND TRUTH VERGLEICH")
print("="*80)

print("\n[1/4] Lade Ground Truth Log...")
df_gt = pd.read_csv(str(ground_truth_path))
print(f"   Ground Truth Events: {len(df_gt)}")

print("\n[2/4] Extrahiere Case Arrival Times aus Ground Truth...")
# Stelle sicher, dass wir ALLE Cases verwenden - keine Begrenzung
arrivals_gt = extract_case_arrivals_from_log(df_gt)
print(f"   Ground Truth Case Arrivals: {len(arrivals_gt)} (ALLE Cases werden verwendet)")
print(f"   Zeitraum: {arrivals_gt[0]} bis {arrivals_gt[-1]}")
print(f"   ✓ Keine Begrenzung - alle {len(arrivals_gt)} Ground Truth Cases werden für Vergleich verwendet")

D_gt = convert_to_daily_sequence(arrivals_gt)
n_days_gt = len(D_gt)
print(f"   Anzahl Tage: {n_days_gt}")

start_date = arrivals_gt[0]

# Ziel: 1000 Cases simulieren (entspricht Ground Truth)
target_cases = 1000
print(f"\n[3/4] Generiere Prediction für {target_cases} Cases...")
print(f"   Start-Datum: {start_date}")
print(f"   Ground Truth: {len(arrivals_gt)} Cases werden für Vergleich verwendet")

retrain = not model_path.exists()
if retrain:
    print("   Trainiere neues Modell...")
else:
    print(f"   Lade Modell von {model_path}...")

# Modell trainieren oder laden mit BESTEN Config-Parametern (aus Optimierung)
if retrain:
    # Beste Config basierend auf iterativer Optimierung:
    # Total Score: 4.0919, √CADD: 2.4835, Volume Diff: 7.14%
    cfg = SimulationConfig(
        train_ratio=0.8,  # 80% Training, 20% Test
        
        # Global Segmentation - optimierte Parameter
        window_size=10,  # Beste: 10 (aus Optimierung)
        kmax=5,  # Beste: 5 (aus Optimierung)
        z_values=(1.0, 0.8, 0.6, 0.4, 0.2, 0.1),  # Standard Sensitivitäts-Levels
        
        # Intraday Bins - optimierte Parameter
        L=5,  # Beste: 5 (aus Optimierung)
        
        # KDE - Standard-Parameter
        kernel="gaussian",
        min_samples_kde=2,  # Standard
        bandwidth_k_values=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),  # Standard
        bandwidth_val_ratio=0.3,
        
        # DBSCAN - optimierte Parameter
        dbscan_eps=0.6,  # Beste: 0.6 (aus Optimierung)
        dbscan_min_samples=2,  # Beste: 2 (aus Optimierung)
        
        verbose=False,  # Weniger Output
        random_state=42
    )
    print("   Verwende BESTE Config-Parameter (aus Optimierung):")
    print(f"      window_size={cfg.window_size}, kmax={cfg.kmax}, L={cfg.L}")
    print(f"      dbscan_eps={cfg.dbscan_eps}, dbscan_min_samples={cfg.dbscan_min_samples}")
    print(f"      Erwartete Qualität: √CADD ≈ 2.48, Volume Diff ≈ 7.14%")
    pipe = CaseInterarrivalPipeline(cfg)
    pipe.fit(df_gt)
    with open(str(model_path), "wb") as f:
        pickle.dump(pipe, f)
else:
    with open(str(model_path), "rb") as f:
        pipe = pickle.load(f)

# Simuliere so lange Tage, bis wir 1000 Cases haben
print(f"   Simuliere Tage bis {target_cases} Cases erreicht sind...")
arrivals_pred = []
current_days = 50  # Starte mit 50 Tagen
max_iterations = 20  # Maximal 20 Iterationen
iteration = 0

while len(arrivals_pred) < target_cases and iteration < max_iterations:
    D_sim = pipe.simulate_days(N_hat=current_days, start_date=start_date)
    arrivals_pred = flatten_days(D_sim)
    print(f"   Iteration {iteration + 1}: {current_days} Tage simuliert → {len(arrivals_pred)} Cases")
    
    if len(arrivals_pred) < target_cases:
        # Schätze benötigte Tage basierend auf aktueller Rate
        cases_per_day = len(arrivals_pred) / current_days if current_days > 0 else 1
        needed_days = int((target_cases - len(arrivals_pred)) / cases_per_day) + 10  # +10 als Puffer
        current_days = needed_days
    iteration += 1

# Begrenze auf genau 1000 Cases
arrivals_pred = arrivals_pred[:target_cases]
print(f"   Final: {len(arrivals_pred)} Cases simuliert")

print(f"   Prediction Case Arrivals: {len(arrivals_pred)}")
print(f"   Zeitraum: {arrivals_pred[0]} bis {arrivals_pred[-1]}")

D_pred = convert_to_daily_sequence(arrivals_pred)
n_days_pred = len(D_pred)
print(f"   Anzahl Tage: {n_days_pred}")

print(f"\n[4/4] Vergleiche Prediction mit Ground Truth...")

print("\n--- CADD Metrik (Cumulative Arrival Distribution Distance) ---")
cadd_score = cadd_distance(arrivals_gt, arrivals_pred)
sqrt_cadd_score = sqrt_cadd(D_gt, D_pred)
print(f"CADD: {cadd_score:.4f}")
print(f"√CADD: {sqrt_cadd_score:.4f}")
print("(Niedrigere Werte = bessere Übereinstimmung)")

print("\n--- Interarrival Statistiken (intraday only) ---")
stats_gt = interarrival_stats_intraday_only(D_gt, unit="seconds")
stats_pred = interarrival_stats_intraday_only(D_pred, unit="seconds")

if stats_gt and stats_pred:
    print(f"\n{'Metrik':<20} {'Ground Truth':<20} {'Prediction':<20} {'Differenz':<20}")
    print("-" * 80)
    for key in ['mean', 'std', 'q05', 'q25', 'q50', 'q75', 'q95']:
        if key in stats_gt and key in stats_pred:
            true_val = stats_gt[key]
            sim_val = stats_pred[key]
            diff = sim_val - true_val
            diff_pct = (diff / true_val * 100) if true_val != 0 else 0
            print(f"{key:<20} {true_val:<20.4f} {sim_val:<20.4f} {diff:+.4f} ({diff_pct:+.2f}%)")

print("\n--- Tägliche Volumen-Statistiken ---")
volumes_gt = [len(day) for day in D_gt]
volumes_pred = [len(day) for day in D_pred]

print(f"Ground Truth:")
print(f"  Mittelwert: {np.mean(volumes_gt):.2f} Cases/Tag")
print(f"  Median: {np.median(volumes_gt):.2f} Cases/Tag")
print(f"  Std: {np.std(volumes_gt):.2f}")
print(f"  Min: {np.min(volumes_gt)}, Max: {np.max(volumes_gt)}")

print(f"\nPrediction:")
print(f"  Mittelwert: {np.mean(volumes_pred):.2f} Cases/Tag")
print(f"  Median: {np.median(volumes_pred):.2f} Cases/Tag")
print(f"  Std: {np.std(volumes_pred):.2f}")
print(f"  Min: {np.min(volumes_pred)}, Max: {np.max(volumes_pred)}")

diff_mean = np.mean(volumes_pred) - np.mean(volumes_gt)
diff_pct = (diff_mean / np.mean(volumes_gt) * 100) if np.mean(volumes_gt) > 0 else 0
print(f"\nDifferenz Mittelwert: {diff_mean:+.2f} ({diff_pct:+.2f}%)")

print("\n--- Gesamt-Statistiken ---")
print(f"Ground Truth:")
print(f"  Gesamt Cases: {len(arrivals_gt)}")
print(f"  Gesamt Tage: {n_days_gt}")
print(f"  Cases/Tag (Ø): {len(arrivals_gt) / n_days_gt:.2f}")

print(f"\nPrediction:")
print(f"  Gesamt Cases: {len(arrivals_pred)}")
print(f"  Gesamt Tage: {n_days_pred}")
print(f"  Cases/Tag (Ø): {len(arrivals_pred) / n_days_pred:.2f}")

diff_total = len(arrivals_pred) - len(arrivals_gt)
diff_total_pct = (diff_total / len(arrivals_gt) * 100) if len(arrivals_gt) > 0 else 0
print(f"\nDifferenz Gesamt: {diff_total:+d} ({diff_total_pct:+.2f}%)")

print("\n" + "="*80)
print("VERGLEICH ABGESCHLOSSEN")
print("="*80)
