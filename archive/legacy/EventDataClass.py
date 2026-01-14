from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Any
from datetime import datetime
import re
from typing import Type, TypeVar
from enum import Enum

E = TypeVar("E", bound=Enum)

def coerce_enum(value: Any, enum_cls: Type[E], field_name: str) -> E:
    """
    Nimmt entweder schon ein Enum oder einen String und gibt ein Enum zurück.
    Wirft eine gut lesbare Exception, wenn der Wert ungültig ist.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        raise ValueError(f"Ungültiger Wert für {field_name}: {value!r}. "
                         f"Erlaubt sind: {[e.value for e in enum_cls]}")

#Enums
class Action(Enum):
    CREATED = "Created"
    STATECHANGE = "statechange"
    DELETED = "Deleted"
    OBTAINED = "Obtained"
    RELEASED = "Released"

class ConceptName(Enum):
    A_CREATE_APPLICATION = "A_Create Application"
    A_SUBMITTED = "A_Submitted"
    W_HANDLE_LEADS = "W_Handle leads"
    W_COMPLETE_APPLICATION = "W_Complete application"
    A_CONCEPT = "A_Concept"
    A_ACCEPTED = "A_Accepted"
    O_CREATE_OFFER = "O_Create Offer"
    O_CREATED = "O_Created"
    O_SENT_MAIL_ONLINE = "O_Sent (mail and online)"
    W_CALL_AFTER_OFFERS = "W_Call after offers"
    A_COMPLETE = "A_Complete"
    W_VALIDATE_APPLICATION = "W_Validate application"
    A_VALIDATING = "A_Validating"
    O_RETURNED = "O_Returned"
    W_CALL_INCOMPLETE_FILES = "W_Call incomplete files"
    A_INCOMPLETE = "A_Incomplete"
    O_ACCEPTED = "O_Accepted"
    A_PENDING = "A_Pending"
    A_DENIED = "A_Denied"
    O_REFUSED = "O_Refused"
    O_CANCELLED = "O_Cancelled"
    A_CANCELLED = "A_Cancelled"
    O_SENT_ONLINE = "O_Sent (online only)"
    W_ASSESS_POTENTIAL_FRAUD = "W_Assess potential fraud"
    W_PERSONAL_LOAN_COLLECTION = "W_Personal Loan collection"
    W_SHORTENED_COMPLETION = "W_Shortened completion "

class EventOrigin(Enum):
    APPLICATION = "Application"
    WORKFLOW = "Workflow"
    OFFER = "Offer"

class LifecycleTransition(Enum):
    COMPLETE = "complete"
    SCHEDULE = "schedule"
    WITHDRAW = "withdraw"
    START = "start"
    SUSPEND = "suspend"
    ATE_ABORT = "ate_abort"
    RESUME = "resume"

class ApplicationType(Enum):
    NEW_CREDIT = "New credit"
    LIMIT_RAISE = "Limit raise"

class LoanGoal(Enum):
    EXISTING_LOAN_TAKEOVER = "Existing loan takeover"
    HOME_IMPROVEMENT = "Home improvement"
    CAR = "Car"
    OTHER = "Other, see explanation"
    REMAINING_DEBT_HOME = "Remaining debt home"
    NOT_SPECIFIED = "Not speficied"
    UNKNOWN = "Unknown"
    CARAVAN = "Caravan / Camper"
    TAX_PAYMENTS = "Tax payments"
    EXTRA_SPENDING_LIMIT = "Extra spending limit"
    MOTORCYCLE = "Motorcycle"
    BOAT = "Boat"
    BUSINESS_GOAL = "Business goal"
    DEBT_RESTRUCTURING = "Debt restructuring"


# Numerische Validierung

USER_PATTERN = re.compile(r"^User_\d+$")

def validate_user(user: str) -> str:
    if not USER_PATTERN.match(user):
        raise ValueError(f"Ungültiger User-Wert: {user}")
    return user


def validate_amount(value: Optional[float], field_name: str):
    if value is None:
        return
    if value < 0:
        raise ValueError(f"{field_name} darf nicht negativ sein")
    return value

def validate_terms(value: Optional[int]):
    if value is None:
        return
    if value <= 0:
        raise ValueError("NumberOfTerms muss positiv sein")
    if value > 200:
        raise ValueError("NumberOfTerms unrealistisch (>200)")
    return value

def validate_credit_score(value: Optional[float]):
    if value is None:
        return
    if not (0 <= value <= 1200):
        raise ValueError("CreditScore muss zwischen 0 und 1200 liegen")
    return value


@dataclass
class LogEvent:
    action: Action
    resource: str
    concept_name: ConceptName
    event_origin: EventOrigin
    event_id: str
    lifecycle: LifecycleTransition
    timestamp: datetime

    loan_goal: Optional[LoanGoal] = None
    application_type: Optional[ApplicationType] = None
    case_concept_name: Optional[str] = None
    requested_amount: Optional[float] = None
    first_withdrawal_amount: Optional[float] = None
    number_of_terms: Optional[int] = None
    accepted: Optional[bool] = None
    monthly_cost: Optional[float] = None
    selected: Optional[bool] = None
    credit_score: Optional[float] = None

    offered_amount: Optional[float] = None
    offer_id: Optional[str] = None

    def __post_init__(self):
        self.action = coerce_enum(self.action, Action, "action")
        self.concept_name = coerce_enum(self.concept_name, ConceptName, "concept_name")
        self.event_origin = coerce_enum(self.event_origin, EventOrigin, "event_origin")
        self.lifecycle = coerce_enum(self.lifecycle, LifecycleTransition, "lifecycle")

        if self.loan_goal is not None:
            self.loan_goal = coerce_enum(self.loan_goal, LoanGoal, "loan_goal")
        if self.application_type is not None:
            self.application_type = coerce_enum(self.application_type, ApplicationType, "application_type")

        self.resource = validate_user(self.resource)
        # für case_id / event_id / offer_id könntest du analog Pattern-Checks machen

        self.requested_amount = validate_amount(self.requested_amount, "RequestedAmount")
        self.offered_amount = validate_amount(self.offered_amount, "OfferedAmount")
        self.monthly_cost = validate_amount(self.monthly_cost, "MonthlyCost")
        self.first_withdrawal_amount = validate_amount(self.first_withdrawal_amount, "FirstWithdrawalAmount")

        self.number_of_terms = validate_terms(self.number_of_terms)
        self.credit_score = validate_credit_score(self.credit_score)

        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp muss ein datetime Objekt sein!")

    @staticmethod
    def from_df_row(row: Any) -> "LogEvent":
        return LogEvent(
            action=row["Action"],                        # String, wird in __post_init__ zu Action
            resource=row["org:resource"],
            concept_name=row["concept:name"],           # String
            event_origin=row["EventOrigin"],            # String
            event_id=row["EventID"],
            lifecycle=row["lifecycle:transition"],      # String
            timestamp=row["time:timestamp"],

            loan_goal=row["case:LoanGoal"],             # String oder NaN -> None
            application_type=row["case:ApplicationType"],
            case_concept_name=row["case:concept:name"],
            requested_amount=row["case:RequestedAmount"],
            first_withdrawal_amount=row["FirstWithdrawalAmount"],
            number_of_terms=row["NumberOfTerms"],
            accepted=row["Accepted"],
            monthly_cost=row["MonthlyCost"],
            selected=row["Selected"],
            credit_score=row["CreditScore"],

            offered_amount=row["OfferedAmount"],
            offer_id=row["OfferID"]
        )
    
    def to_dict(self):
        """
        Konvertiert das LogEvent in ein Dict für CSV-Ausgabe.
        Enums werden zu ihren .value Strings konvertiert.
        """
        result = {}
        for field_name, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[field_name] = value.value   # ❤️ Klartext statt Enum
            else:
                result[field_name] = value
        return result
