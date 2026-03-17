"""
Skript zum Simulieren von 10.000 Cases mit allen Attributen.
"""

from pathlib import Path
import pandas as pd

from simulator import AttributeSimulationEngine


def main():
    print("=" * 60)
    print("Simuliere 10.000 Cases mit allen Attributen")
    print("=" * 60)
    
    # Engine initialisieren (nutzt gespeicherte Modelle)
    print("\nInitialisiere AttributeSimulationEngine...")
    try:
        engine = AttributeSimulationEngine(df=None, seed=42, retrain_models=False)
        print("✓ Engine erfolgreich initialisiert (nutzt gespeicherte Modelle)")
    except Exception as e:
        print(f"✗ Fehler beim Initialisieren: {e}")
        print("\nVersuche mit Trainings-Daten...")
        # Falls keine gespeicherten Modelle vorhanden, müsste man df übergeben
        raise
    
    # 10.000 Cases simulieren
    print(f"\nSimuliere 10.000 Cases...")
    sim_df = engine.simulate_n_cases(10_000, with_offer_attributes=True, progress=True)
    
    # Ergebnisse anzeigen
    print("\n" + "=" * 60)
    print("Ergebnisse:")
    print("=" * 60)
    print(f"Anzahl Cases: {len(sim_df):,}")
    print(f"Anzahl Spalten: {len(sim_df.columns)}")
    print(f"\nSpalten: {', '.join(sim_df.columns)}")
    
    print("\nErste 10 Zeilen:")
    print(sim_df.head(10))
    
    print("\nStatistiken:")
    print(sim_df.describe())
    
    # Als CSV speichern
    output_path = Path(__file__).parent / "simulated_10k_cases.csv"
    sim_df.to_csv(output_path, index=False)
    print(f"\n✓ Daten gespeichert in: {output_path}")
    print(f"  Dateigröße: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Fertig!")
    print("=" * 60)


if __name__ == "__main__":
    main()
