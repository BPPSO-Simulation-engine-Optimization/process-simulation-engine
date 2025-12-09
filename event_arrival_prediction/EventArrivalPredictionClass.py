from datetime import datetime, timedelta

class EventArrivalPredictionClass:

    def __init__(self):
        pass

    def predict(self):
        # TODO: Modell bauen
        # TODO: Predicten

        timestamp_in_40_minutes = datetime.now() + timedelta(minutes=40)
        return timestamp_in_40_minutes
