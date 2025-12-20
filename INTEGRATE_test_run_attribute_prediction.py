import pandas as pd
from case_attribute_prediction.simulator import AttributeSimulationEngine

df = pd.read_csv("event_log.csv")

engine = AttributeSimulationEngine(df=df, seed=42)

case_rows = []

n_cases = 30000
for _ in range(n_cases):
    engine.start_new_case()
    # Trigger: damit CreditScore/OfferedAmount/Terms/MonthlyCost/... gezogen werden
    engine.draw_case_and_event_attributes("O_Create Offer")
    case_rows.append(engine.finalize_case_row())

sim_df = pd.DataFrame(case_rows)

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

print(sim_df.head(50))



