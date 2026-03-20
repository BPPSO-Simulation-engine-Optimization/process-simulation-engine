from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

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
        offered_col: str = "OfferedAmount",
        credit_col: str = "CreditScore",
    ) -> "NumberOfTermsPredictor":
        self.rng = np.random.default_rng(self.seed)

        t_col = resolve_col(df, terms_col)
        o_col = resolve_col(df, offered_col)
        c_col = resolve_col(df, credit_col)
        
        # Case-level extrahieren mit allen relevanten Features
        cols = ["case:LoanGoal", "case:ApplicationType", t_col, o_col, c_col]
        case_tbl = to_case_level(df, cols).copy()

        case_tbl[t_col] = pd.to_numeric(case_tbl[t_col], errors="coerce")
        case_tbl[o_col] = pd.to_numeric(case_tbl[o_col], errors="coerce")
        case_tbl[c_col] = pd.to_numeric(case_tbl[c_col], errors="coerce")
        case_tbl = case_tbl.dropna(subset=[t_col, o_col, c_col])
        case_tbl = case_tbl[(case_tbl[o_col] > 0) & (case_tbl[c_col] >= 0)]

        # OfferedAmount in Quantil-Bins aufteilen (0-25%, 25-50%, 50-75%, 75-100%)
        if len(case_tbl) > 0:
            offered_quantiles = [0, 0.25, 0.5, 0.75, 1.0]
            offered_bins = case_tbl[o_col].quantile(offered_quantiles).tolist()
            # Sicherstellen, dass bins eindeutig sind
            offered_bins = sorted(list(set(offered_bins)))
            if len(offered_bins) < 2:
                offered_bins = [case_tbl[o_col].min(), case_tbl[o_col].max()]
            case_tbl["_offered_bin"] = pd.cut(
                case_tbl[o_col], 
                bins=offered_bins, 
                labels=False, 
                include_lowest=True,
                duplicates='drop'
            )
            case_tbl["_offered_bin"] = case_tbl["_offered_bin"].fillna(0).astype(int)

            # CreditScore in Standard-Bins aufteilen (wie in anderen Predictors)
            credit_bins = [0, 600, 650, 700, 750, 1000]
            case_tbl["_credit_bin"] = pd.cut(
                case_tbl[c_col],
                bins=credit_bins,
                labels=[0, 1, 2, 3, 4],  # <600, 600-649, 650-699, 700-749, 750-999
                include_lowest=True,
                right=False
            )
            case_tbl["_credit_bin"] = case_tbl["_credit_bin"].fillna(0).astype(int)

            # Mehrstufige Segmentierung: zuerst detailliert, dann immer allgemeiner
            # Level 1: (LoanGoal, ApplicationType, OfferedBin, CreditBin)
            dist_by_4 = {
                k: v[t_col].to_numpy()
                for k, v in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType", "_offered_bin", "_credit_bin"])
            }

            # Level 2: (LoanGoal, ApplicationType, OfferedBin)
            dist_by_3 = {
                k: v[t_col].to_numpy()
                for k, v in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType", "_offered_bin"])
            }

            # Level 3: (LoanGoal, ApplicationType) - Original
            dist_by_2 = {
                k: v[t_col].to_numpy()
                for k, v in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType"])
            }

            # Level 4: Nur nach LoanGoal
            dist_by_1 = {
                str(k): v[t_col].to_numpy()
                for k, v in case_tbl.groupby("case:LoanGoal")
            }

            self.model = {
                "by_4": dist_by_4,  # (LoanGoal, AppType, OfferedBin, CreditBin)
                "by_3": dist_by_3,  # (LoanGoal, AppType, OfferedBin)
                "by_2": dist_by_2,  # (LoanGoal, AppType)
                "by_1": dist_by_1,  # LoanGoal
                "global": case_tbl[t_col].to_numpy(),
                "terms_col": t_col,
                "offered_bins": offered_bins,
                "credit_bins": credit_bins,
            }
        else:
            # Fallback falls keine Daten
            self.model = {
                "by_4": {},
                "by_3": {},
                "by_2": {},
                "by_1": {},
                "global": np.array([]),
                "terms_col": t_col,
                "offered_bins": [0, 1],
                "credit_bins": [0, 600, 650, 700, 750, 1000],
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
        Sonst: mehrstufiges segment-basiertes Resampling aus empirischer Terms-Verteilung.
        Nutzt OfferedAmount und CreditScore für bessere Segmentierung.
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

        # Rückwärtskompatibilität: Prüfe ob altes oder neues Modellformat
        is_old_format = "by_pair" in m
        is_new_format = "by_4" in m

        if is_old_format:
            # Altes Format: nutze nur (LoanGoal, ApplicationType)
            arr = m["by_pair"].get((loan_goal, application_type))
            if arr is None or len(arr) == 0:
                arr = m["global"]
            val = float(self.rng.choice(arr)) if len(arr) else 0.0
            return int(round(val))

        # Neues Format: mehrstufige Segmentierung
        # Binning für OfferedAmount
        offered_bins = m.get("offered_bins", [0, float('inf')])
        offered_bin = 0
        if len(offered_bins) > 1:
            offered_val = float(offered_amount)
            # Finde das richtige Bin
            for i in range(len(offered_bins) - 1):
                if offered_val <= offered_bins[i + 1]:
                    offered_bin = i
                    break
            else:
                # Fallback: letztes Bin
                offered_bin = len(offered_bins) - 2

        # Binning für CreditScore
        credit_bins_list = m.get("credit_bins", [0, 600, 650, 700, 750, 1000])
        credit_bin = 0
        for i in range(len(credit_bins_list) - 1):
            if credit_bins_list[i] <= float(credit_score) < credit_bins_list[i + 1]:
                credit_bin = i
                break
        if float(credit_score) >= credit_bins_list[-1]:
            credit_bin = len(credit_bins_list) - 2

        # Mehrstufige Suche: versuche immer spezifischer zu werden
        # Level 1: (LoanGoal, AppType, OfferedBin, CreditBin)
        key_4 = (loan_goal, application_type, offered_bin, credit_bin)
        arr = m.get("by_4", {}).get(key_4)
        
        # Level 2: (LoanGoal, AppType, OfferedBin)
        if arr is None or len(arr) == 0:
            key_3 = (loan_goal, application_type, offered_bin)
            arr = m.get("by_3", {}).get(key_3)
        
        # Level 3: (LoanGoal, AppType) - Original
        if arr is None or len(arr) == 0:
            key_2 = (loan_goal, application_type)
            arr = m.get("by_2", {}).get(key_2)
        
        # Level 4: Nur LoanGoal
        if arr is None or len(arr) == 0:
            arr = m.get("by_1", {}).get(str(loan_goal))
        
        # Fallback: globale Verteilung
        if arr is None or len(arr) == 0:
            arr = m.get("global", np.array([]))

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
        per_group: bool = True,
        print_results: bool = True,
    ):
        """
        Validiert simulierte NumberOfTerms gegen Original-Daten.
        
        Returns:
            overall_df: pd.DataFrame mit Overall-Statistiken
            per_group_df: pd.DataFrame mit Statistiken pro Gruppe (LoanGoal, ApplicationType) oder None
        """
        o = resolve_col(df, original_col)
        s = resolve_col(sim_df, simulated_col)

        # Case-level extrahieren mit Gruppierungs-Spalten
        orig = to_case_level(df, ["case:LoanGoal", "case:ApplicationType", "case:concept:name", o]).copy()
        sim = sim_df[["case:LoanGoal", "case:ApplicationType", s]].copy()

        orig[o] = pd.to_numeric(orig[o], errors="coerce")
        sim[s] = pd.to_numeric(sim[s], errors="coerce")
        orig = orig.dropna(subset=[o])
        sim = sim.dropna(subset=[s])
        orig[o] = orig[o].astype(int)
        sim[s] = sim[s].astype(int)

        def summary(x):
            x = np.asarray(x, dtype=float)
            if len(x) == 0:
                return {
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "p10": np.nan,
                    "p50": np.nan,
                    "p90": np.nan,
                }
            return {
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "p10": float(np.quantile(x, 0.10)),
                "p50": float(np.quantile(x, 0.50)),
                "p90": float(np.quantile(x, 0.90)),
            }

        x_all = orig[o].to_numpy()
        y_all = sim[s].to_numpy()

        overall = {
            **{f"orig_{k}": v for k, v in summary(x_all).items()},
            **{f"sim_{k}": v for k, v in summary(y_all).items()},
        }
        
        if len(x_all) > 0 and len(y_all) > 0:
            overall["ks"] = ks_statistic_1d(x_all, y_all)
            overall["wasserstein"] = wasserstein_approx_1d(x_all, y_all)
            
            # Zusätzliche Vergleichs-Metriken für NumberOfTerms
            overall["mean_diff"] = float(np.mean(y_all) - np.mean(x_all))
            overall["mean_diff_pct"] = float((np.mean(y_all) - np.mean(x_all)) / np.mean(x_all) * 100) if np.mean(x_all) > 0 else np.nan
            
            # Median-Differenz
            overall["median_diff"] = float(np.median(y_all) - np.median(x_all))
            
            # Häufigste Werte (Modalwerte)
            try:
                orig_mode_result = stats.mode(x_all, keepdims=False)
                sim_mode_result = stats.mode(y_all, keepdims=False)
                orig_mode = float(orig_mode_result.mode) if orig_mode_result.count > 0 else np.nan
                sim_mode = float(sim_mode_result.mode) if sim_mode_result.count > 0 else np.nan
                overall["orig_mode"] = orig_mode
                overall["sim_mode"] = sim_mode
                overall["mode_match"] = float(orig_mode == sim_mode) if (not np.isnan(orig_mode) and not np.isnan(sim_mode)) else np.nan
            except:
                overall["orig_mode"] = np.nan
                overall["sim_mode"] = np.nan
                overall["mode_match"] = np.nan
            
            # Anteil der Fälle innerhalb bestimmter Bereiche (basierend auf Original-Verteilung)
            x_mean = np.mean(x_all)
            x_std = np.std(x_all)
            
            # Innerhalb 1 Standardabweichung
            within_1std_orig = np.sum((x_all >= x_mean - x_std) & (x_all <= x_mean + x_std)) / len(x_all)
            within_1std_sim = np.sum((y_all >= x_mean - x_std) & (y_all <= x_mean + x_std)) / len(y_all) if len(y_all) > 0 else np.nan
            overall["orig_within_1std"] = float(within_1std_orig)
            overall["sim_within_1std"] = float(within_1std_sim)
            
            # Anteil mit häufigsten 5 Werten
            top_5_values = pd.Series(x_all).value_counts().head(5).index.to_numpy()
            orig_in_top5 = np.sum(np.isin(x_all, top_5_values)) / len(x_all)
            sim_in_top5 = np.sum(np.isin(y_all, top_5_values)) / len(y_all) if len(y_all) > 0 else np.nan
            overall["orig_in_top5_freq"] = float(orig_in_top5)
            overall["sim_in_top5_freq"] = float(sim_in_top5)
            
        else:
            overall["ks"] = np.nan
            overall["wasserstein"] = np.nan
            overall["mean_diff"] = np.nan
            overall["mean_diff_pct"] = np.nan
            overall["median_diff"] = np.nan
            overall["orig_mode"] = np.nan
            overall["sim_mode"] = np.nan
            overall["mode_match"] = np.nan
            overall["orig_within_1std"] = np.nan
            overall["sim_within_1std"] = np.nan
            overall["orig_in_top5_freq"] = np.nan
            overall["sim_in_top5_freq"] = np.nan
            
        overall_df = pd.DataFrame([overall])

        if not per_group:
            return overall_df, None

        rows = []
        og = orig.groupby(["case:LoanGoal", "case:ApplicationType"])
        sg = sim.groupby(["case:LoanGoal", "case:ApplicationType"])
        keys = set(og.groups.keys()) | set(sg.groups.keys())

        for k in keys:
            o_vals = og.get_group(k)[o].to_numpy() if k in og.groups else np.array([], dtype=int)
            s_vals = sg.get_group(k)[s].to_numpy() if k in sg.groups else np.array([], dtype=int)
            
            if len(o_vals) == 0 or len(s_vals) == 0:
                continue

            row = {
                "case:LoanGoal": k[0],
                "case:ApplicationType": k[1],
                **{f"orig_{k}": v for k, v in summary(o_vals).items()},
                **{f"sim_{k}": v for k, v in summary(s_vals).items()},
            }
            
            if len(o_vals) > 0 and len(s_vals) > 0:
                row["ks"] = ks_statistic_1d(o_vals, s_vals)
                row["wasserstein"] = wasserstein_approx_1d(o_vals, s_vals)
            else:
                row["ks"] = np.nan
                row["wasserstein"] = np.nan
                
            rows.append(row)

        per_group_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        
        if print_results:
            print("\n=== VALIDATION: Number of Terms (Distribution) ===")
            print("\n--- Overall Statistics ---")
            print(overall_df)
            
            if per_group_df is not None and len(per_group_df) > 0:
                print("\n--- Per Group Statistics ---")
                print(per_group_df.head(20))
                if len(per_group_df) > 20:
                    print(f"... ({len(per_group_df) - 20} weitere Gruppen)")
        
        return overall_df, per_group_df