from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col
from .metrics import ks_statistic_1d, wasserstein_approx_1d


class MonthlyCostPredictor(AttributePredictorBase):
    name = "MonthlyCost"

    def __init__(self, seed: int = 42, artifact: dict | None = None, iqr_k: float = 1.5):
        super().__init__(seed=seed)
        self.model = artifact
        self.iqr_k = float(iqr_k)

    def fit(self, df: pd.DataFrame) -> "MonthlyCostPredictor":
        """
        Trainiert das MonthlyCost-Modell intern (wie Notebook),
        wenn noch kein Artefakt gesetzt ist.
        """
        # Wenn Modell schon gesetzt wurde, nichts tun (optional, aber praktisch)
        if self.model is not None:
            return self

        # --- Case-level Tabelle ---
        required = [
            "case:concept:name",
            "OfferedAmount",
            "NumberOfTerms",
            "CreditScore",
            "MonthlyCost",
            "case:ApplicationType",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Fehlende Spalten für MonthlyCostPredictor.fit(): {missing}")

        d = (
            df.groupby("case:concept:name")[required[1:]]
              .first()
              .reset_index(drop=True)
        )

        for c in ["OfferedAmount", "NumberOfTerms", "CreditScore", "MonthlyCost"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        d = d.dropna(subset=["OfferedAmount", "NumberOfTerms", "CreditScore", "MonthlyCost"]).copy()
        d["NumberOfTerms"] = d["NumberOfTerms"].round().astype(int)

        # --- Design Matrix: Numerik + OHE (drop_first) ---
        num_cols = ["OfferedAmount", "NumberOfTerms", "CreditScore"]
        X_num = d[num_cols].copy()

        X_cat = pd.get_dummies(
            d[["case:ApplicationType"]],
            drop_first=True,
            dtype=float,
        )
        cat_cols = list(X_cat.columns)

        X = pd.concat([X_num, X_cat], axis=1)
        y = d["MonthlyCost"].to_numpy(dtype=float)

        # --- IQR Outlier Removal (wie Notebook, global) ---
        k = self.iqr_k
        mask = pd.Series(True, index=X.index)

        for c in num_cols:
            q1 = X[c].quantile(0.25)
            q3 = X[c].quantile(0.75)
            iqr = q3 - q1
            lb = q1 - k * iqr
            ub = q3 + k * iqr
            mask &= (X[c] >= lb) & (X[c] <= ub)

        q1y = np.quantile(y, 0.25)
        q3y = np.quantile(y, 0.75)
        iqry = q3y - q1y
        lby = q1y - k * iqry
        uby = q3y + k * iqry
        mask &= (y >= lby) & (y <= uby)

        X_clean = X.loc[mask].copy()
        y_clean = y[mask.to_numpy()]

        # --- Scaling nur auf Numerik ---
        scaler = StandardScaler()
        X_scaled = X_clean.copy()
        X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])

        # --- Linear Fit ---
        reg = LinearRegression()
        reg.fit(X_scaled, y_clean)

        # Bias-Korrektur (in-sample)
        y_pred = reg.predict(X_scaled)
        bias = float(np.mean(y_pred - y_clean))

        # --- Artefakt speichern (genau wie predict() erwartet) ---
        self.model = {
            "coef": reg.coef_,                 # np.ndarray
            "intercept": float(reg.intercept_),
            "bias": float(bias),
            "scaler_mean": scaler.mean_,       # np.ndarray (len=3)
            "scaler_scale": scaler.scale_,     # np.ndarray (len=3)
            "num_cols": num_cols,              # optional (Doku)
            "cat_cols": cat_cols,              # wichtig für predict()
        }

        return self

    def set_artifact(self, artifact: dict) -> "MonthlyCostPredictor":
        self.model = artifact
        return self

    def predict(
        self,
        offered_amount: float,
        number_of_terms: int,
        credit_score: float,
        application_type: str,
    ) -> float:
        self._require_fitted()
        model = self.model
        assert model is not None

        x_num = np.array([float(offered_amount), int(number_of_terms), float(credit_score)], dtype=float)
        x_num_scaled = (x_num - model["scaler_mean"]) / model["scaler_scale"]

        x_cat = np.zeros(len(model["cat_cols"]), dtype=float)
        for i, c in enumerate(model["cat_cols"]):
            if c == f"case:ApplicationType_{application_type}":
                x_cat[i] = 1.0

        x = np.concatenate([x_num_scaled, x_cat])
        y_hat = float(np.dot(x, model["coef"]) + model["intercept"] - model["bias"])
        return float(max(y_hat, 0.0))

    def validate(self, df: pd.DataFrame, sim_df: pd.DataFrame, original_col="MonthlyCost", simulated_col="MonthlyCost", print_results: bool = True):
        o_col = resolve_col(df, original_col)
        s_col = resolve_col(sim_df, simulated_col)

        orig = to_case_level(df, ["case:LoanGoal", "case:ApplicationType", o_col]).dropna()
        sim = sim_df[["case:LoanGoal", "case:ApplicationType", s_col]].dropna()

        x = pd.to_numeric(orig[o_col], errors="coerce").dropna().to_numpy()
        y = pd.to_numeric(sim[s_col], errors="coerce").dropna().to_numpy()

        overall = {
            "ks": ks_statistic_1d(x, y),
            "wasserstein": wasserstein_approx_1d(x, y),
            "orig_mean": float(np.mean(x)) if len(x) else np.nan,
            "sim_mean": float(np.mean(y)) if len(y) else np.nan,
        }
        result_df = pd.DataFrame([overall])
        
        if print_results:
            print("\n=== VALIDATION: Monthly Cost ===")
            print(result_df)
        
        return result_df
