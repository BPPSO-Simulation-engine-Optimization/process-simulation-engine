from __future__ import annotations

import numpy as np
import pandas as pd
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col


class AcceptedPredictor(AttributePredictorBase):
    name = "Accepted"

    def fit(self, df: pd.DataFrame, accepted_col: str = "Accepted") -> "AcceptedPredictor":
        self.rng = np.random.default_rng(self.seed)

        acc_col = resolve_col(df, accepted_col)
        cols = [acc_col, "MonthlyCost", "CreditScore"]
        case_tbl = to_case_level(df, cols).dropna()

        if len(case_tbl) == 0:
            raise ValueError("Keine gültigen Daten für Accepted Predictor gefunden.")

        # Verwende logistische Regression falls verfügbar, sonst verbesserte lineare Formel
        use_lr = False
        if HAS_SKLEARN and len(case_tbl) > 10:
            try:
                X = case_tbl[["CreditScore", "MonthlyCost"]].values
                y = case_tbl[acc_col].values.astype(int)
                
                # Standardisiere Features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Trainiere logistische Regression
                lr_model = LogisticRegression(random_state=self.seed, max_iter=1000)
                lr_model.fit(X_scaled, y)
                
                # Speichere auch base_rate als Fallback
                base_rate = float(case_tbl[acc_col].mean())
                self.model = {
                    "base_rate": base_rate,
                    "use_lr": True,
                    "lr_model": lr_model,    # serialisierbar über pickle
                    "scaler": scaler,        # serialisierbar über pickle
                    "credit_score_mean": float(case_tbl["CreditScore"].mean()),
                    "credit_score_std": float(case_tbl["CreditScore"].std()),
                    "monthly_cost_mean": float(case_tbl["MonthlyCost"].mean()),
                    "monthly_cost_std": float(case_tbl["MonthlyCost"].std()),
                }
                use_lr = True
            except Exception as e:
                # Fallback auf verbesserte lineare Formel
                print(f"Warnung: Logistische Regression fehlgeschlagen ({e}), verwende verbesserte lineare Formel.")
                use_lr = False
        
        if not use_lr:
            # Verbesserte lineare Formel basierend auf tatsächlichen Daten
            base_rate = float(case_tbl[acc_col].mean())
            credit_score_mean = float(case_tbl["CreditScore"].mean())
            credit_score_std = float(case_tbl["CreditScore"].std())
            monthly_cost_mean = float(case_tbl["MonthlyCost"].mean())
            monthly_cost_std = float(case_tbl["MonthlyCost"].std())
            
            # Berechne Koeffizienten basierend auf Korrelationen
            # CreditScore hat moderate positive Korrelation (~0.2)
            # MonthlyCost hat sehr schwache Korrelation (~0.002)
            credit_score_coef = 0.0005  # Reduziert von 0.001
            monthly_cost_coef = -0.000005  # Reduziert von -0.00001
            
            self.model = {
                "base_rate": base_rate,
                "use_lr": False,
                "credit_score_mean": credit_score_mean,
                "credit_score_std": credit_score_std,
                "credit_score_coef": credit_score_coef,
                "monthly_cost_mean": monthly_cost_mean,
                "monthly_cost_std": monthly_cost_std,
                "monthly_cost_coef": monthly_cost_coef,
            }
        
        return self

    def predict_proba(self, monthly_cost: float, credit_score: float) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

        if m.get("use_lr", False) and "lr_model" in m:
            # Verwende logistische Regression (aus model dict)
            X = np.array([[credit_score, monthly_cost]])
            X_scaled = m["scaler"].transform(X)
            p = float(m["lr_model"].predict_proba(X_scaled)[0, 1])
        else:
            # Verbesserte lineare Formel
            base_rate = m["base_rate"]
            
            # Normalisiere Features (z-score)
            credit_score_norm = (float(credit_score) - m["credit_score_mean"]) / max(m["credit_score_std"], 1.0)
            monthly_cost_norm = (float(monthly_cost) - m["monthly_cost_mean"]) / max(m["monthly_cost_std"], 1.0)
            
            # Angepasste Formel mit normalisierten Features
            p = base_rate
            p += m["credit_score_coef"] * credit_score_norm * 100  # Skaliere zurück
            p += m["monthly_cost_coef"] * monthly_cost_norm * 100
            
            # Clip auf sinnvollen Bereich
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
        print_results: bool = True,
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

        result_df = pd.DataFrame(rows).sort_values("orig_n", ascending=False).reset_index(drop=True)
        
        if print_results:
            print("\n=== VALIDATION: Accepted ===")
            print(result_df.head(30))
            if len(result_df) > 30:
                print(f"... ({len(result_df) - 30} weitere Zeilen)")
        
        return result_df

    def validate(self, df: pd.DataFrame, sim_df: pd.DataFrame, col: str = "Accepted", print_results: bool = True) -> pd.DataFrame:
        return self.validate_binary(df=df, sim_df=sim_df, col=col, print_results=print_results)
