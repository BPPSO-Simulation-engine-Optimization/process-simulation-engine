from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .credit_score import CreditScorePredictor
from .selected import SelectedPredictor
from .accepted import AcceptedPredictor
from .offered_amount import OfferedAmountPredictor
from .number_of_terms import NumberOfTermsPredictor
from .monthly_cost import MonthlyCostPredictor
from .first_withdrawal_amount import FirstWithdrawalAmountPredictor


@dataclass
class PredictorRegistry:
    credit_score: CreditScorePredictor
    selected: SelectedPredictor
    accepted: AcceptedPredictor
    offered_amount: OfferedAmountPredictor
    number_of_terms: NumberOfTermsPredictor
    monthly_cost: MonthlyCostPredictor
    first_withdrawal_amount: FirstWithdrawalAmountPredictor

    def fit_all(self, df: pd.DataFrame) -> "PredictorRegistry":
        self.credit_score.fit(df)
        self.selected.fit(df)
        self.accepted.fit(df)
        self.offered_amount.fit(df)
        self.number_of_terms.fit(df)
        # monthly_cost kann entweder Artefakt-basiert sein (ohne fit) oder optional fitten:
        self.monthly_cost.fit(df)
        self.first_withdrawal_amount.fit(df)
        return self


def build_default_registry(seed: int = 42) -> PredictorRegistry:
    return PredictorRegistry(
        credit_score=CreditScorePredictor(seed=seed),
        selected=SelectedPredictor(seed=seed),
        accepted=AcceptedPredictor(seed=seed),
        offered_amount=OfferedAmountPredictor(seed=seed),
        number_of_terms=NumberOfTermsPredictor(seed=seed),
        monthly_cost=MonthlyCostPredictor(seed=seed),
        first_withdrawal_amount=FirstWithdrawalAmountPredictor(seed=seed),
    )
