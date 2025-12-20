from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col
from .metrics import ks_statistic_1d, wasserstein_approx_1d


class NumberOfTermsPredictor(AttributePredictorBase):
    name = "NumberOfTerms"

    def __init__(self, seed: int = 42, sklearn_model=None):
        """
        Zwei Betriebsmodi:
          A) Ohne sklearn_model: resample aus empirischer Verteilung (init_number_of_terms_predictor)
          B) Mit sklearn_model: nutzt final_model.predict(X) wie in Ihrem Notebook
        """
        super().__init__(seed=seed)
        self.sklearn_model = sklearn_model

    def fit(
        self,
        df: pd.DataFrame,
        terms_col: str = "NumberOfTerms",
    ) -> "NumberOfTermsPredictor":
        self.rng = np.random.default_rng(self.seed)

        t_col = resolve_col(df, terms_col)
        cols = ["case:LoanGoal", "case:ApplicationType", t_col]
        case_tbl = to_case_level(df, cols).dropna().copy()

        case_tbl[t_col] = pd.to_numeric(case_tbl[t_col], errors="coerce")
        case_tbl = case_tbl.dropna(subset=[t_col])

        dist = {
            k: v[t_col].to_numpy()
            for k, v in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType"])
        }

        self.model = {
            "by_pair": dist,
            "global": case_tbl[t_col].to_numpy(),
            "terms_col": t_col,
        }
        return self

    def predict(
        self,
        offered_amount: float,
        credit_score: float,
        loan_goal: str,
        application_type: str,
    ) -> int:
        """
        Wenn sklearn_model gesetzt ist: exakt Notebook-Pattern (X DataFrame -> model.predict)
        Sonst: segment-basiertes Resampling aus empirischer Terms-Verteilung.
        """
        if self.sklearn_model is not None:
            X = pd.DataFrame([{
                "OfferedAmount": float(offered_amount),
                "CreditScore": float(credit_score),
                "case:LoanGoal": loan_goal,
                "case:ApplicationType": application_type,
            }])
            return int(self.sklearn_model.predict(X)[0])

        self._require_fitted()
        m = self.model
        assert m is not None

        arr = m["by_pair"].get((loan_goal, application_type))
        if arr is None or len(arr) == 0:
            arr = m["global"]

        val = float(self.rng.choice(arr)) if len(arr) else 0.0
        return int(round(val))

    def validate_predict_no_of_terms_mae(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        test_size: float = 0.2,
        k_top_terms: int = 10,
        case_col: str = "case:concept:name",
        offered_col: str = "OfferedAmount",
        credit_col: str = "CreditScore",
        target_col: str = "NumberOfTerms",
    ):
        """
        1:1 Notebook-Logik validate_predict_no_of_terms_mae (leicht konsolidiert).
        """
        if case_col in df.columns:
            d = (df.groupby(case_col)[[offered_col, credit_col, target_col]].first().reset_index(drop=True))
        else:
            d = df[[offered_col, credit_col, target_col]].copy()

        for c in (offered_col, credit_col, target_col):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[offered_col, credit_col, target_col]).copy()
        d[target_col] = d[target_col].round().astype(int)

        rng = np.random.default_rng(seed)
        n = len(d)
        idx = rng.permutation(n)
        n_test = max(1, int(round(test_size * n)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        train = d.iloc[train_idx].copy()
        test = d.iloc[test_idx].copy()

        top_terms = (
            train[target_col].value_counts().head(k_top_terms).index.to_numpy(dtype=int)
        )

        y_true = test[target_col].to_numpy(dtype=int)

        # Wir rufen self.predict(...) auf; LoanGoal/ApplicationType fehlen hier im Notebook-Validator
        # => Wenn Sie diese Konditionierung brauchen, müssen Sie es im Validator-Call ergänzen.
        y_pred = np.array([
            int(self.predict(off, cs, loan_goal="__NA__", application_type="__NA__"))
            for off, cs in zip(test[offered_col].to_numpy(), test[credit_col].to_numpy())
        ], dtype=int)

        y_pred = np.clip(y_pred, train[target_col].min(), train[target_col].max())

        err = y_pred - y_true
        ae = np.abs(err)

        overall = {
            "n_test": int(len(y_true)),
            "mae_months": float(ae.mean()),
            "median_ae_months": float(np.median(ae)),
            "within_1_month": float((ae <= 1).mean()),
            "within_3_months": float((ae <= 3).mean()),
            "within_6_months": float((ae <= 6).mean()),
            "bias_months_mean(pred-true)": float(err.mean()),
            "top_terms_used": top_terms.tolist(),
        }

        rep = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "err": err, "ae": ae})
        by_month = (
            rep.groupby("y_true", as_index=False)
               .agg(
                   n=("ae", "size"),
                   mae_months=("ae", "mean"),
                   median_ae_months=("ae", "median"),
                   bias_months_mean=("err", "mean"),
                   within_3_months=("ae", lambda s: float((s <= 3).mean())),
                   within_6_months=("ae", lambda s: float((s <= 6).mean())),
               )
               .sort_values(["n", "y_true"], ascending=[False, True])
               .reset_index(drop=True)
        )

        mode_term = int(train[target_col].mode().iloc[0])
        ae_base = np.abs(y_true - mode_term)
        baseline = {
            "mode_term": mode_term,
            "baseline_mae_months": float(ae_base.mean()),
            "delta_mae_vs_baseline": float(ae_base.mean() - ae.mean()),
        }

        return overall, by_month, baseline, rep

    def validate(
        self,
        df: pd.DataFrame,
        sim_df: pd.DataFrame,
        original_col="NumberOfTerms",
        simulated_col="NumberOfTerms",
    ) -> pd.DataFrame:
        o = resolve_col(df, original_col)
        s = resolve_col(sim_df, simulated_col)

        orig = (
            to_case_level(df, ["case:concept:name", o])[o]
            .dropna()
            .astype(int)
            .to_numpy()
        )
        sim = (
            sim_df[s]
            .dropna()
            .astype(int)
            .to_numpy()
        )

        return pd.DataFrame([{
            "ks": ks_statistic_1d(orig, sim),
            "wasserstein": wasserstein_approx_1d(orig, sim),
            "orig_mean": float(np.mean(orig)) if len(orig) else np.nan,
            "sim_mean": float(np.mean(sim)) if len(sim) else np.nan,
        }])