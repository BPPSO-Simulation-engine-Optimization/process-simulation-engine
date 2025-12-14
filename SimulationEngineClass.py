from typing import List, Optional
import pandas as pd
from enum import Enum
import pm4py
import os
import uuid
import random
from datetime import timedelta
from DataPreparationClass import DataPreparationClass
from EventDataClass import LogEvent, Action, ConceptName, EventOrigin, LifecycleTransition
from case_arrival_times_prediction.CaseArrivalTimePredictionClass import CaseArrivalTimePredictionClass
from case_attribute_prediction.CaseAttributePredictionClass import CaseAttributePredictionClass
from case_next_activity_prediction.CaseNextActivityPredictionClass import CaseNextActivityPredictionClass
from event_arrival_prediction.EventArrivalPredictionClass import EventArrivalPredictionClass
from event_attribute_prediction.EventAttributePredictionClass import EventAttributePredictionClass
from processing_time_prediction.ProcessingTimePredictionClass import ProcessingTimePredictionClass


class SimulationEngineClass:

    currentSequence: List[LogEvent]
    dataLogDf: pd.DataFrame = None
    originalDataLogDf: pd.DataFrame = None  # Keep original data for processing time extraction
    dataIsPrepared: bool = False
    
    isCaseEnded = False

    def __init__(self, dataLogDf: pd.DataFrame = None, processing_time_method: str = "ml"):
        self.dataLogDf = dataLogDf
        self.originalDataLogDf = dataLogDf.copy() if dataLogDf is not None else None

        self.case_arrival_time_predictor = CaseArrivalTimePredictionClass()
        self.case_attribute_predictor = CaseAttributePredictionClass()
        self.next_activity_predictor = CaseNextActivityPredictionClass()
        self.event_arrival_predictor = EventArrivalPredictionClass()
        self.event_attribute_predictor = EventAttributePredictionClass()
        
        # Processing time predictor will be initialized after data is loaded
        self.processing_time_predictor: Optional[ProcessingTimePredictionClass] = None
        self.processing_time_method = processing_time_method

    def generate_case_id(self):
        return f"Application_{random.randint(10000000, 1999999999)}"

    def generate_event_id(self):
        return f"Event_{uuid.uuid4().hex[:12]}"

    def generate_offer_id(self):
        return f"Offer_{random.randint(100000000, 1999999999)}"


    # Simulation
    def simulate(self, numberSimulatedCases: int) -> None:

        SIM_LOG_PATH = "simulated_log.csv"

        # Datei löschen, wenn vorhanden
        if os.path.exists(SIM_LOG_PATH):
            os.remove(SIM_LOG_PATH)
            print("🗑️  Alte simulated_log.csv gelöscht.")

        if self.dataLogDf is None:
            raise ValueError("❌ Kein Log geladen! load_or_prepare_log() zuerst aufrufen.")

        # Datapreparation
        if not self.dataIsPrepared:
            prep = DataPreparationClass(self.dataLogDf)
            self.dataLogDf = prep.prepare_data()
            self.dataIsPrepared = True
        
        # Initialize processing time predictor if not already initialized
        if self.processing_time_predictor is None:
            print("Initializing processing time predictor...")
            # Use original data (before preparation) for processing time prediction
            # as we need raw timestamps and lifecycle transitions
            data_for_extraction = self.originalDataLogDf if self.originalDataLogDf is not None else self.dataLogDf
            self.processing_time_predictor = ProcessingTimePredictionClass(
                data_for_extraction, method=self.processing_time_method
            )

        #Simulation
        for i in range(numberSimulatedCases):

            # Case_Id erstellen
            case_id = self.generate_case_id()

            # Case-Arrival-Time
            simulated_timestamp = self.case_arrival_time_predictor.predict()

            # Draw-Case-Attributes
            case_loan_goal, case_application_type, requested_amount = self.case_attribute_predictor.predict()
            
            # Track previous event for processing time prediction
            previous_event: Optional[LogEvent] = None

            while not self.isCaseEnded:

                # Event_Id erstellen
                generated_event_id = self.generate_event_id()

                # Predict Next Activity
                predicted_next_activity, isCaseEnded = self.next_activity_predictor.predict()

                # Draw Event Attributes
                (
                    predicted_action, predicted_resource, predicted_origin, predicted_lifecycle,
                    withdrawal_amount, number_of_terms, accepted_flag,
                    monthly_cost, is_selected, credit_score, offered_amount
                ) = self.event_attribute_predictor.predict()

                # Predict Processing Time between previous and current event
                if previous_event is not None and self.processing_time_predictor is not None:
                    # Get previous event's activity and lifecycle
                    # Handle both Enum and string types
                    prev_activity = previous_event.concept_name.value if isinstance(previous_event.concept_name, Enum) else str(previous_event.concept_name)
                    prev_lifecycle = previous_event.lifecycle.value if isinstance(previous_event.lifecycle, Enum) else str(previous_event.lifecycle)
                    
                    # Get current event's activity and lifecycle
                    curr_activity = predicted_next_activity.value if isinstance(predicted_next_activity, Enum) else str(predicted_next_activity)
                    curr_lifecycle = predicted_lifecycle.value if isinstance(predicted_lifecycle, Enum) else str(predicted_lifecycle)
                    
                    # Prepare context for ML model
                    context = {
                        'resource_1': previous_event.resource,
                        'resource_2': predicted_resource,
                        'case:RequestedAmount': requested_amount,
                        'CreditScore': credit_score,
                        'hour': simulated_timestamp.hour,
                        'weekday': simulated_timestamp.weekday(),
                        'month': simulated_timestamp.month,
                    }
                    
                    # Predict processing time
                    processing_time_seconds = self.processing_time_predictor.predict(
                        prev_activity, prev_lifecycle,
                        curr_activity, curr_lifecycle,
                        context
                    )
                    
                    # Add processing time to timestamp
                    simulated_timestamp = simulated_timestamp + timedelta(seconds=processing_time_seconds)
                else:
                    # First event in case - use event arrival predictor
                    simulated_timestamp = self.event_arrival_predictor.predict()

                # Offer-ID erzeugen
                generated_offer_id = self.generate_offer_id()

                # Validieren der simulierten Event-Daten:
                eventData = LogEvent(
                    action=predicted_action,            
                    resource=predicted_resource,
                    concept_name=predicted_next_activity, 
                    event_origin=predicted_origin,
                    event_id=generated_event_id,
                    lifecycle=predicted_lifecycle,
                    timestamp=simulated_timestamp,

                    loan_goal=case_loan_goal,
                    application_type=case_application_type,

                    case_concept_name=case_id,
                    requested_amount=requested_amount,
                    first_withdrawal_amount=withdrawal_amount,
                    number_of_terms=number_of_terms,
                    accepted=accepted_flag,
                    monthly_cost=monthly_cost,
                    selected=is_selected,
                    credit_score=credit_score,

                    offered_amount=offered_amount,
                    offer_id=generated_offer_id
                )


                # Schreiben ins SimLog:
                self.write_event_into_sim_log(eventData)
                
                # Update previous event for next iteration
                previous_event = eventData

                #Damit endet
                self.isCaseEnded = True


            self.isCaseEnded = False

        
        print("Simulation ended!")

    # Datensatz laden
    def load_or_prepare_log(self, PATH_TO_XES):
        SIMULATION_LOG_PATH = "event_log.csv"

        # Daten laden, wenn schon mal geschrieben
        if os.path.exists(SIMULATION_LOG_PATH):
            df = pd.read_csv(SIMULATION_LOG_PATH)
            self.dataLogDf = df 
            self.originalDataLogDf = df.copy()  # Keep original for processing time extraction
            return df

        # Log neu einlesen
        print("📥 Lese XES Log zum ersten Mal ein (kann dauern)...")
        log = pm4py.read_xes(PATH_TO_XES, return_legacy_log_object=True)
        df = pm4py.convert_to_dataframe(log)

        self.dataLogDf = df 
        self.originalDataLogDf = df.copy()  # Keep original for processing time extraction

        print("✨ Speichere vorbereiteten Log für zukünftige Läufe...")
        df.to_csv(SIMULATION_LOG_PATH, index=False)

        return df

    def write_event_into_sim_log(self, data: LogEvent) -> None:
        """
        Schreibt ein einzelnes simuliertes Event an simulated_log.csv.
        Erstellt die Datei, falls sie nicht existiert.
        """
        SIM_LOG_PATH = "simulated_log.csv"

        # LogEvent → Dictionary
        row_dict = data.to_dict()

        # Dictionary → DataFrame (einzelne Zeile)
        df_row = pd.DataFrame([row_dict])

        # Prüfen, ob die Datei bereits existiert
        file_exists = os.path.exists(SIM_LOG_PATH)

        # Append-Modus (a), Header nur schreiben, wenn Datei neu
        df_row.to_csv(
            SIM_LOG_PATH,
            mode="a",
            header=not file_exists,
            index=False
        )



if __name__ == "__main__":

    engine = SimulationEngineClass()
    engine.load_or_prepare_log("/Users/laurensohl/Downloads/BPI Challenge 2017.xes")
    engine.simulate(5)

    print(engine.dataLogDf)
