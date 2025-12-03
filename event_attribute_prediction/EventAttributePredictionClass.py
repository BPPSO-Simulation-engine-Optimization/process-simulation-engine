from datetime import datetime, timedelta

class EventAttributePredictionClass:

    def __init__(self):
        pass

    def predict(self):
        # TODO: Modell bauen
        # TODO: Predicten

        predicted_action = "Created"                    # Spalte: Action
        predicted_resource = "User_17"                  # Spalte: org:resource
        predicted_origin = "Application"                # Spalte: EventOrigin
        predicted_lifecycle = "complete"                # Spalte: lifecycle:transition

        withdrawal_amount = 20000.0                     # Spalte: FirstWithdrawalAmount
        number_of_terms = 72                            # Spalte: NumberOfTerms
        accepted_flag = True                            # Spalte: Accepted
        monthly_cost = 498.29                           # Spalte: MonthlyCost
        is_selected = True                              # Spalte: Selected
        credit_score = 850                              # Spalte: CreditScore
        offered_amount = 15000.0                        # Spalte: OfferedAmount

        return (
            predicted_action,
            predicted_resource,
            predicted_origin,
            predicted_lifecycle,
            withdrawal_amount,
            number_of_terms,
            accepted_flag,
            monthly_cost,
            is_selected,
            credit_score,
            offered_amount
        )
