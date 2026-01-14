from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col, credit_score_bin


class SelectedPredictor(AttributePredictorBase):
    name = "Selected"

    def fit(
        self,
        df: pd.DataFrame,
        selected_col: str = "Selected",
        tau: float = 200.0,  # Shrinkage wie im Notebook
    ) -> "SelectedPredictor":
        self.rng = np.random.default_rng(self.seed)

        sel_col = resolve_col(df, selected_col)
        cols = ["case:LoanGoal", "case:ApplicationType", "case:RequestedAmount", "CreditScore", sel_col]
        case_tbl = to_case_level(df, cols).copy()

        case_tbl[sel_col] = case_tbl[sel_col].astype(bool)

        grp = case_tbl.groupby(["case:LoanGoal", "case:ApplicationType"])
        sel_rate = grp[sel_col].mean()
        sel_n = grp[sel_col].size()
        global_rate = float(case_tbl[sel_col].mean())

        self.model = {
            "by_pair": sel_rate.to_dict(),      # (loan_goal, app_type) -> p
            "n_by_pair": sel_n.to_dict(),       # (loan_goal, app_type) -> n
            "global": global_rate,
            "tau": float(tau),
        }
        return self

    def predict_proba(self, loan_goal: str, application_type: str, credit_score: float) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

        seg = (loan_goal, application_type)

        p_global = m["global"]
        p_seg = m["by_pair"].get(seg)
        n_seg = m.get("n_by_pair", {}).get(seg, 0)
        tau = float(m.get("tau", 200.0))

        # Shrinkage wie Notebook:
        if p_seg is None:
            base_p = p_global
        else:
            base_p = (n_seg / (n_seg + tau)) * p_seg + (tau / (n_seg + tau)) * p_global

        # Logit
        logit_p = np.log(base_p / (1 - base_p))

        # CreditScore-Effekt (wie Notebook)
        beta = 0.03 / 50 if application_type == "Limit raise" else 0.015 / 50
        logit_p += beta * (float(credit_score) - 650.0)

        # Fix für Limit raise (wie Notebook)
        if application_type == "Limit raise":
            logit_p += 0.45

        p = 1.0 / (1.0 + np.exp(-logit_p))
        return float(np.clip(p, 0.0, 1.0))

    def predict(self, loan_goal: str, application_type: str, credit_score: float) -> bool:
        p = self.predict_proba(loan_goal, application_type, credit_score)
        return bool(self.rng.random() < p)

    def validate_binary_by_score(
        self,
        df: pd.DataFrame,
        sim_df: pd.DataFrame,
        score_col: str = "CreditScore",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
        score_bins=(0, 500, 600, 700, 800, 900, 1000),
        col: str = "Selected",
        print_results: bool = True,
    ) -> pd.DataFrame:
        col_o = resolve_col(df, col)
        col_s = resolve_col(sim_df, col)
        sc_o = resolve_col(df, score_col)
        sc_s = resolve_col(sim_df, score_col)

        orig = to_case_level(df, list(group_cols) + [sc_o, col_o]).copy()
        sim = sim_df[list(group_cols) + [sc_s, col_s]].copy()

        labels = [f"{score_bins[i]}–{score_bins[i+1]-1}" for i in range(len(score_bins)-1)]
        orig["_score_bin"] = pd.cut(orig[sc_o], bins=score_bins, labels=labels)
        sim["_score_bin"] = pd.cut(sim[sc_s], bins=score_bins, labels=labels)

        grp_cols = list(group_cols) + ["_score_bin"]
        orig_grp = orig.groupby(grp_cols)
        sim_grp = sim.groupby(grp_cols)

        rows = []
        for k, o in orig_grp:
            if k not in sim_grp.groups:
                continue
            s = sim_grp.get_group(k)

            rows.append({
                "case:LoanGoal": k[0],
                "case:ApplicationType": k[1],
                "CreditScore_bin": k[2],
                "orig_rate": float(o[col_o].mean()),
                "sim_rate": float(s[col_s].mean()),
                "abs_diff": float(abs(o[col_o].mean() - s[col_s].mean())),
                "orig_n": int(len(o)),
                "sim_n": int(len(s)),
            })

        result_df = (
            pd.DataFrame(rows)
            .sort_values(["case:LoanGoal", "case:ApplicationType", "CreditScore_bin"])
            .reset_index(drop=True)
        )
        
        if print_results:
            print("\n=== VALIDATION: Selected ===")
            print(result_df.head(30))
            if len(result_df) > 30:
                print(f"... ({len(result_df) - 30} weitere Zeilen)")
        
        return result_df

    def validate(self, df: pd.DataFrame, sim_df: pd.DataFrame, col: str = "Selected", print_results: bool = True) -> pd.DataFrame:
        # identisch zur Notebook-Logik validate_binary_by_score
        return self.validate_binary_by_score(df=df, sim_df=sim_df, col=col, print_results=print_results)
