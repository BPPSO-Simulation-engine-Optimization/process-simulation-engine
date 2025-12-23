import sys
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from case_attribute_prediction.simulator import AttributeSimulationEngine

# Wenn True: Modelle + Verteilungen aus dem XES-Log neu trainieren und persistieren.
# Wenn False: ausschließlich aus gespeicherten Artefakten simulieren (kein XES-Load).
RETRAIN_MODELS = False

# Eventlog wird nur geladen für das Training der Modelle.
if RETRAIN_MODELS:
    if len(sys.argv) > 1:
        EVENT_LOG_PATH = sys.argv[1]
    else:
        EVENT_LOG_PATH = "eventlog.xes.gz"

    event_log = xes_importer.apply(EVENT_LOG_PATH)
    df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
else:
    df = None

engine = AttributeSimulationEngine(df=df, seed=42, retrain_models=RETRAIN_MODELS)

case_rows = []

n_cases = 100
for _ in range(n_cases):
    # Case-bezogene Attribute (bleiben für alle Events dieses Cases konstant)
    case_attributes = engine.draw_case_attributes()

    # Hardgecoded Startaktivität für den Case
    next_activity = "O_Withdrawal"

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

# Validierung nur, wenn ein Trainings-df vorhanden ist (RETRAIN_MODELS=True).
# In reiner Simulationsphase wird ausschließlich vorhergesagt.
if RETRAIN_MODELS:
    # monthly_cost: bool = False,
    # credit_score: bool = False,
    # offered_amount: bool = False,
    # number_of_terms_dist = False,
    # number_of_terms_mae = False,
    # first_withdrawal: bool = False,
    # selected: bool = False,
    # accepted: bool = False,
    print("\n=== VALIDATION: MONTHLY_COST ===")
    val = engine.validate(
        sim_df,
        offered_amount=True
    )
    print(val["offered_amount"])

    print("\n=== VALIDATION: Number of Terms ===")
    val = engine.validate(
        sim_df,
        number_of_terms_dist=True,
        number_of_terms_mae=True,
    )

    print(val["number_of_terms_dist"])
    print(val["number_of_terms_mae_overall"])
    print(val["number_of_terms_mae_baseline"])

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

# Loan goal gut, aplpication type gut, rquested amount gut (Annahme), Offered amount gut,
## TODO:
# rquested amount validieren und creditScore validieren
# testen, ob Fälle gibt mit requested amount < offeredamount
# CreditScore korrigieren
# Datapreparation korrigiere
# Verbessern von No of terms und monthlycosts. 51% mehr overpay statt 18% im original df
# FirstWithdrawal amount fixen!!

print(sim_df.head()["concept:name"])
print(sim_df["case:concept:name"].unique())



