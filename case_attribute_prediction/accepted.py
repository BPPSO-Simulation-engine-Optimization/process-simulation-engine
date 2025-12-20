from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col


class AcceptedPredictor(AttributePredictorBase):
    name = "Accepted"

    def fit(self, df: pd.DataFrame, accepted_col: str = "Accepted") -> "AcceptedPredictor":
        self.rng = np.random.default_rng(self.seed)

        acc_col = resolve_col(df, accepted_col)
        cols = [acc_col, "MonthlyCost", "CreditScore"]
        case_tbl = to_case_level(df, cols).dropna()

        base_rate = float(case_tbl[acc_col].mean())
        self.model = {"base_rate": base_rate}
        return self

    def predict_proba(self, monthly_cost: float, credit_score: float) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

        p = float(m["base_rate"])
        p += 0.001 * (float(credit_score) - 650.0)
        p -= 0.00001 * float(monthly_cost)
        p = float(np.clip(p, 0.01, 0.99))
        return p

    def predict(self, monthly_cost: float, credit_score: float) -> bool:
        p = self.predict_proba(monthly_cost, credit_score)
        return bool(self.rng.random() < p)

    def validate_binary(
        self,
        df: pd.DataFrame,
        sim_df: pd.DataFrame,
        col: str = "Accepted",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
    ) -> pd.DataFrame:
        col_o = resolve_col(df, col)
        col_s = resolve_col(sim_df, col)

        orig = to_case_level(df, list(group_cols) + [col_o]).copy()
        sim = sim_df[list(group_cols) + [col_s]].copy()

        rows = []
        og = orig.groupby(list(group_cols))
        sg = sim.groupby(list(group_cols))
        keys = set(og.groups.keys()) | set(sg.groups.keys())

        for k in keys:
            o = og.get_group(k)[col_o] if k in og.groups else None
            s = sg.get_group(k)[col_s] if k in sg.groups else None
            if o is None or s is None:
                continue

            o = o.dropna()
            s = s.dropna()
            if len(o) == 0 or len(s) == 0:
                continue

            rows.append({
                group_cols[0]: k[0],
                group_cols[1]: k[1],
                "orig_n": int(len(o)),
                "sim_n": int(len(s)),
                "orig_rate": float(o.mean()),
                "sim_rate": float(s.mean()),
                "abs_diff": float(abs(o.mean() - s.mean())),
            })

        return pd.DataFrame(rows).sort_values("orig_n", ascending=False).reset_index(drop=True)

    def validate(self, df: pd.DataFrame, sim_df: pd.DataFrame, col: str = "Accepted") -> pd.DataFrame:
        return self.validate_binary(df=df, sim_df=sim_df, col=col)
