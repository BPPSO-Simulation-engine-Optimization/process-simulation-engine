from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .metrics import ks_statistic_1d, wasserstein_approx_1d
from .utils import to_case_level, resolve_col


class CreditScorePredictor(AttributePredictorBase):
    name = "CreditScore"

    def fit(self, df: pd.DataFrame, credit_score_col: str = "CreditScore") -> "CreditScorePredictor":
        self.rng = np.random.default_rng(self.seed)

        cs_col = resolve_col(df, credit_score_col)
        cols = ["case:LoanGoal", "case:ApplicationType", cs_col]
        case_tbl = to_case_level(df, cols)

        case_tbl[cs_col] = pd.to_numeric(case_tbl[cs_col], errors="coerce")
        case_tbl = case_tbl.dropna(subset=[cs_col])

        cs_global = case_tbl[cs_col].to_numpy()

        cs_by_pair = {
            (lg, at): sub[cs_col].to_numpy()
            for (lg, at), sub in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType"])
        }
        cs_by_lg = {lg: sub[cs_col].to_numpy() for lg, sub in case_tbl.groupby("case:LoanGoal")}
        cs_by_at = {at: sub[cs_col].to_numpy() for at, sub in case_tbl.groupby("case:ApplicationType")}

        self.model = {
            "global": cs_global,
            "by_pair": cs_by_pair,
            "by_lg": cs_by_lg,
            "by_at": cs_by_at,
        }
        return self

    def predict(self, loan_goal: str, application_type: str) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

        arr = m["by_pair"].get((loan_goal, application_type))
        if arr is None or len(arr) == 0:
            arr = m["by_lg"].get(loan_goal)
        if arr is None or len(arr) == 0:
            arr = m["by_at"].get(application_type)
        if arr is None or len(arr) == 0:
            arr = m["global"]

        return float(self.rng.choice(arr))

    def validate(
        self,
        df: pd.DataFrame,
        sim_df: pd.DataFrame,
        original_cs_col: str = "CreditScore",
        simulated_cs_col: str = "CreditScore",
        per_group: bool = True,
    ):
        o_cs = resolve_col(df, original_cs_col)
        s_cs = resolve_col(sim_df, simulated_cs_col)

        orig = to_case_level(df, ["case:LoanGoal", "case:ApplicationType", o_cs]).copy()
        sim = sim_df[["case:LoanGoal", "case:ApplicationType", s_cs]].copy()

        orig[o_cs] = pd.to_numeric(orig[o_cs], errors="coerce")
        sim[s_cs] = pd.to_numeric(sim[s_cs], errors="coerce")
        orig = orig.dropna(subset=[o_cs])
        sim = sim.dropna(subset=[s_cs])

        def summary(x):
            x = np.asarray(x, dtype=float)
            return {
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "p10": float(np.quantile(x, 0.10)),
                "p50": float(np.quantile(x, 0.50)),
                "p90": float(np.quantile(x, 0.90)),
            }

        x_all = orig[o_cs].to_numpy()
        y_all = sim[s_cs].to_numpy()

        overall = {
            **{f"orig_{k}": v for k, v in summary(x_all).items()},
            **{f"sim_{k}": v for k, v in summary(y_all).items()},
            "ks": ks_statistic_1d(x_all, y_all),
            "wasserstein": wasserstein_approx_1d(x_all, y_all),
        }
        overall_df = pd.DataFrame([overall])

        if not per_group:
            return overall_df, None

        rows = []
        og = orig.groupby(["case:LoanGoal", "case:ApplicationType"])
        sg = sim.groupby(["case:LoanGoal", "case:ApplicationType"])
        keys = set(og.groups.keys()) | set(sg.groups.keys())

        for k in keys:
            o = og.get_group(k)[o_cs].to_numpy() if k in og.groups else np.array([])
            s = sg.get_group(k)[s_cs].to_numpy() if k in sg.groups else np.array([])
            if len(o) == 0 or len(s) == 0:
                continue
            rows.append({
                "case:LoanGoal": k[0],
                "case:ApplicationType": k[1],
                "orig_n": len(o),
                "sim_n": len(s),
                "ks": ks_statistic_1d(o, s),
                "wasserstein": wasserstein_approx_1d(o, s),
            })

        per_group_df = pd.DataFrame(rows).sort_values("orig_n", ascending=False).reset_index(drop=True)
        return overall_df, per_group_df
