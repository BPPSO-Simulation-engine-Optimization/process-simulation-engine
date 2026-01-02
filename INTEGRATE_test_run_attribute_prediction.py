import sys
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from case_attribute_prediction.simulator import AttributeSimulationEngine

# Wenn True: Modelle + Verteilungen aus dem XES-Log neu trainieren und persistieren.
# Wenn False: ausschließlich aus gespeicherten Artefakten simulieren (kein XES-Load).
RETRAIN_MODELS = False

# Eventlog wird geladen für Validierung (immer) und Training (nur wenn RETRAIN_MODELS=True).
# Wir brauchen das df auch für die Validierung, auch wenn wir nicht neu trainieren.
if len(sys.argv) > 1:
    EVENT_LOG_PATH = sys.argv[1]
else:
    EVENT_LOG_PATH = "eventlog/eventlog.xes.gz"

try:
    event_log = xes_importer.apply(EVENT_LOG_PATH)
    df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    print(f"Eventlog geladen: {len(df)} Events, {df['case:concept:name'].nunique()} Cases")
except Exception as e:
    print(f"Warnung: Eventlog konnte nicht geladen werden: {e}")
    print("Validierung wird übersprungen. Nur Simulation wird durchgeführt.")
    df = None

engine = AttributeSimulationEngine(df=df, seed=42, retrain_models=RETRAIN_MODELS)

# Sicherstellen, dass engine.df gesetzt ist für Validierung
# (wichtig wenn df später geladen wurde oder RETRAIN_MODELS=False war)
if df is not None:
    engine.df = df

case_rows = []

n_cases = 30000
for _ in range(n_cases):
    # Case-bezogene Attribute (bleiben für alle Events dieses Cases konstant)
    case_attributes = engine.draw_case_attributes()

    # Hardgecoded Startaktivität für den Case
    # WICHTIG: Offer-Attribute werden nur bei "O_Create Offer" generiert
    next_activity = "O_Create Offer"

    # Zieht so lange event_attributes, bis die Next-Activity "END" ist
    while True:
        # Event-bezogene Attribute nach Vorbild des Eventlogs ziehen
        event_attributes = engine.draw_event_attributes(next_activity)

        event_row = {
            **case_attributes,
            **event_attributes,
        }
        case_rows.append(event_row)


        next_activity = "END"

        if next_activity == "END":
            # Case beenden, nächster Schleifendurchlauf startet neuen Case
            engine.end_current_case()
            break

sim_df = pd.DataFrame(case_rows)

# Validierung wird ausgeführt, wenn ein Trainings-df vorhanden ist.
# Auch ohne RETRAIN_MODELS=True können wir validieren, solange df geladen wurde.
if df is not None:
    val = engine.validate(
        sim_df,
        first_withdrawal=True,
    )

    print(df.columns)
else:
    print("\n(Warnung: Kein Eventlog geladen, Validierung übersprungen)")

cols = [
    "case:RequestedAmount",
    "OfferedAmount",
    "NumberOfTerms",
    "MonthlyCost",
]

print(sim_df[cols].head())

import numpy as np

# Nur relevante Zeilen (robust)
d = sim_df[
    (sim_df["OfferedAmount"] > 0) &
    (sim_df["NumberOfTerms"] > 0) &
    (sim_df["MonthlyCost"] > 0)
].copy()

# Gesamtzahlung
d["total_paid"] = d["NumberOfTerms"] * d["MonthlyCost"]

# Prozentuale Mehrzahlung
d["overpayment_pct"] = (
    (d["total_paid"] - d["OfferedAmount"])
    / d["OfferedAmount"]
) * 100

# Durchschnitt
avg_overpayment_pct = d["overpayment_pct"].mean()

print(f"Durchschnittliche Mehrzahlung: {avg_overpayment_pct:.2f} %")

# Loan goal gut, aplpication type gut, rquested amount gut, Offered amount gut, CreditScore in Abh. zu case:LoanGoal und ApplicationType gut. gut
## TODO:
# CreditScore korrigieren
# Datapreparation korrigiere
# Verbessern von No of terms und monthlycosts. 51% mehr overpay statt 18% im original df

print(sim_df.head()["concept:name"])
print(sim_df["case:concept:name"].unique())

print(sim_df["case:RequestedAmount"].unique())

# Test: Prüfe ob FirstWithdrawalAmount <= OfferedAmount
print("\n=== TEST: FirstWithdrawalAmount vs OfferedAmount ===")
sim_df_test = sim_df[
    sim_df["FirstWithdrawalAmount"].notna() & 
    sim_df["OfferedAmount"].notna() &
    (sim_df["FirstWithdrawalAmount"] >= 0) &
    (sim_df["OfferedAmount"] > 0)
].copy()

if len(sim_df_test) > 0:
    # Vergleich FirstWithdrawalAmount vs OfferedAmount
    sim_df_test["fwa_lt_offered"] = sim_df_test["FirstWithdrawalAmount"] < sim_df_test["OfferedAmount"]
    sim_df_test["fwa_eq_offered"] = sim_df_test["FirstWithdrawalAmount"] == sim_df_test["OfferedAmount"]
    sim_df_test["fwa_gt_offered"] = sim_df_test["FirstWithdrawalAmount"] > sim_df_test["OfferedAmount"]
    sim_df_test["fwa_le_offered"] = sim_df_test["FirstWithdrawalAmount"] <= sim_df_test["OfferedAmount"]
    
    total = len(sim_df_test)
    lt_count = sim_df_test["fwa_lt_offered"].sum()
    eq_count = sim_df_test["fwa_eq_offered"].sum()
    gt_count = sim_df_test["fwa_gt_offered"].sum()
    le_count = sim_df_test["fwa_le_offered"].sum()
    
    print(f"Gesamt Fälle mit gültigen Werten: {total}")
    print(f"FirstWithdrawalAmount < OfferedAmount: {lt_count} ({lt_count/total*100:.2f}%)")
    print(f"FirstWithdrawalAmount == OfferedAmount: {eq_count} ({eq_count/total*100:.2f}%)")
    print(f"FirstWithdrawalAmount <= OfferedAmount: {le_count} ({le_count/total*100:.2f}%)")
    print(f"FirstWithdrawalAmount > OfferedAmount: {gt_count} ({gt_count/total*100:.2f}%) ⚠️")
    
    if gt_count > 0:
        print(f"\n❌ FEHLER: {gt_count} Fälle haben FirstWithdrawalAmount > OfferedAmount (UNGÜLTIG!)")
        print("\nBeispiele (FirstWithdrawalAmount > OfferedAmount):")
        problematic = sim_df_test[sim_df_test["fwa_gt_offered"]][
            ["case:LoanGoal", "case:ApplicationType", "FirstWithdrawalAmount", "OfferedAmount"]
        ].head(10)
        print(problematic)
        
        # Berechne Differenz für problematische Fälle
        problematic["difference"] = problematic["FirstWithdrawalAmount"] - problematic["OfferedAmount"]
        problematic["pct_over"] = (problematic["difference"] / problematic["OfferedAmount"]) * 100
        print("\nÜberschreitung:")
        print(problematic[["FirstWithdrawalAmount", "OfferedAmount", "difference", "pct_over"]])
    else:
        print("\n✅ Alle Fälle sind gültig (FirstWithdrawalAmount <= OfferedAmount)")
    
    # Statistik: Durchschnittliches Verhältnis (Prozent)
    sim_df_test["pct_of_offered"] = (sim_df_test["FirstWithdrawalAmount"] / sim_df_test["OfferedAmount"]) * 100
    print(f"\nStatistik (FirstWithdrawalAmount als % von OfferedAmount):")
    print(f"  Durchschnitt: {sim_df_test['pct_of_offered'].mean():.2f}%")
    print(f"  Median: {sim_df_test['pct_of_offered'].median():.2f}%")
    print(f"  Min: {sim_df_test['pct_of_offered'].min():.2f}%")
    print(f"  Max: {sim_df_test['pct_of_offered'].max():.2f}%")
    
    # Verteilung in Kategorien
    def categorize_pct(pct):
        if pct <= 5:
            return "0-5%"
        elif pct < 95:
            return "5-95%"
        else:
            return "95-100%"
    
    sim_df_test["pct_category"] = sim_df_test["pct_of_offered"].apply(categorize_pct)
    cat_counts = sim_df_test["pct_category"].value_counts(normalize=True) * 100
    print(f"\nVerteilung der Abhebungsprozent:")
    for cat in ["0-5%", "5-95%", "95-100%"]:
        count = cat_counts.get(cat, 0)
        print(f"  {cat}: {count:.2f}%")
else:
    print("⚠️  Keine gültigen Daten für Vergleich vorhanden")



