class CaseAttributePredictionClass:

    def __init__(self):
        pass

    def predict(self):
        # TODO: Modell bauen
        # TODO: Predicten

        # --- Realistische Dummy-Werte aus deinen Spalten ---
        case_loan_goal = "Home improvement"     # aus case:LoanGoal
        case_application_type = "New credit"    # aus case:ApplicationType
        requested_amount = 20000                # aus case:RequestedAmount

        return case_loan_goal, case_application_type, requested_amount
