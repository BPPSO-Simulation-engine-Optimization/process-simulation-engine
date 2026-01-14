from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .metrics import ks_statistic_1d, wasserstein_approx_1d
from .utils import to_case_level, resolve_col, credit_score_bin


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
        credit_score_bins: tuple = (0, 600, 650, 700, 750, 1000),
        credit_score_labels: tuple = ("<600", "600-649", "650-699", "700-749", "750-999"),
        print_results: bool = True,
    ):
        """
        Validiert simulierte CreditScores gegen Original-Daten.
        
        CreditScore-spezifische Validierung:
        - Verteilung nach CreditScore-Bins (Risiko-Kategorien)
        - Anteil der Fälle in jedem Bin
        - Vergleich von Min/Max/Perzentilen
        - Gruppiert nach LoanGoal und ApplicationType
        
        Args:
            df: Original DataFrame
            sim_df: Simuliertes DataFrame
            original_cs_col: Spaltenname für CreditScore im Original
            simulated_cs_col: Spaltenname für CreditScore in Simulation
            per_group: Ob auch per-Group Statistiken berechnet werden sollen
            credit_score_bins: Bin-Grenzen für CreditScore (Standard: Bank-Bins)
            credit_score_labels: Labels für die Bins
        
        Returns:
            overall_df: pd.DataFrame mit Overall-Statistiken
            per_group_df: pd.DataFrame mit Statistiken pro Gruppe oder None
            bin_distribution_df: pd.DataFrame mit Verteilung über CreditScore-Bins
        """
        o_cs = resolve_col(df, original_cs_col)
        s_cs = resolve_col(sim_df, simulated_cs_col)

        orig = to_case_level(df, ["case:LoanGoal", "case:ApplicationType", o_cs]).copy()
        sim = sim_df[["case:LoanGoal", "case:ApplicationType", s_cs]].copy()

        orig[o_cs] = pd.to_numeric(orig[o_cs], errors="coerce")
        sim[s_cs] = pd.to_numeric(sim[s_cs], errors="coerce")
        orig = orig.dropna(subset=[o_cs])
        sim = sim.dropna(subset=[s_cs])

        # Validierung: CreditScore sollte zwischen 0 und 1200 liegen
        orig = orig[(orig[o_cs] >= 0) & (orig[o_cs] <= 1200)]
        sim = sim[(sim[s_cs] >= 0) & (sim[s_cs] <= 1200)]

        def summary(x):
            x = np.asarray(x, dtype=float)
            if len(x) == 0:
                return {
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "p5": np.nan,
                    "p10": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "p90": np.nan,
                    "p95": np.nan,
                }
            return {
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "p5": float(np.quantile(x, 0.05)),
                "p10": float(np.quantile(x, 0.10)),
                "p25": float(np.quantile(x, 0.25)),
                "p50": float(np.quantile(x, 0.50)),
                "p75": float(np.quantile(x, 0.75)),
                "p90": float(np.quantile(x, 0.90)),
                "p95": float(np.quantile(x, 0.95)),
            }

        x_all = orig[o_cs].to_numpy()
        y_all = sim[s_cs].to_numpy()

        # Basis-Statistiken
        overall = {
            **{f"orig_{k}": v for k, v in summary(x_all).items()},
            **{f"sim_{k}": v for k, v in summary(y_all).items()},
        }
        
        if len(x_all) > 0 and len(y_all) > 0:
            overall["ks"] = ks_statistic_1d(x_all, y_all)
            overall["wasserstein"] = wasserstein_approx_1d(x_all, y_all)
            
            # Mean-Differenz
            overall["mean_diff"] = float(np.mean(y_all) - np.mean(x_all))
            overall["mean_diff_pct"] = float((np.mean(y_all) - np.mean(x_all)) / np.mean(x_all) * 100) if np.mean(x_all) > 0 else np.nan
            
            # Median-Differenz
            overall["median_diff"] = float(np.median(y_all) - np.median(x_all))
            
            # Anteil außerhalb des erwarteten Bereichs (600-1200 ist typisch für Bank-Scores)
            overall["orig_outside_600_1200"] = float(np.sum((x_all < 600) | (x_all > 1200)) / len(x_all)) if len(x_all) > 0 else np.nan
            overall["sim_outside_600_1200"] = float(np.sum((y_all < 600) | (y_all > 1200)) / len(y_all)) if len(y_all) > 0 else np.nan
            
            # Anteil in "guten" Scores (>= 700)
            overall["orig_good_score_pct"] = float(np.sum(x_all >= 700) / len(x_all)) if len(x_all) > 0 else np.nan
            overall["sim_good_score_pct"] = float(np.sum(y_all >= 700) / len(y_all)) if len(y_all) > 0 else np.nan
            
            # Anteil in "sehr guten" Scores (>= 750)
            overall["orig_excellent_score_pct"] = float(np.sum(x_all >= 750) / len(x_all)) if len(x_all) > 0 else np.nan
            overall["sim_excellent_score_pct"] = float(np.sum(y_all >= 750) / len(y_all)) if len(y_all) > 0 else np.nan
            
            # Anteil in "schlechten" Scores (< 650)
            overall["orig_poor_score_pct"] = float(np.sum(x_all < 650) / len(x_all)) if len(x_all) > 0 else np.nan
            overall["sim_poor_score_pct"] = float(np.sum(y_all < 650) / len(y_all)) if len(y_all) > 0 else np.nan
        else:
            overall["ks"] = np.nan
            overall["wasserstein"] = np.nan
            overall["mean_diff"] = np.nan
            overall["mean_diff_pct"] = np.nan
            overall["median_diff"] = np.nan
            overall["orig_outside_600_1200"] = np.nan
            overall["sim_outside_600_1200"] = np.nan
            overall["orig_good_score_pct"] = np.nan
            overall["sim_good_score_pct"] = np.nan
            overall["orig_excellent_score_pct"] = np.nan
            overall["sim_excellent_score_pct"] = np.nan
            overall["orig_poor_score_pct"] = np.nan
            overall["sim_poor_score_pct"] = np.nan
            
        overall_df = pd.DataFrame([overall])

        # CreditScore-Bin-Verteilung
        orig_bins = credit_score_bin(orig[o_cs], bins=credit_score_bins, labels=credit_score_labels)
        sim_bins = credit_score_bin(sim[s_cs], bins=credit_score_bins, labels=credit_score_labels)
        
        orig_bin_counts = orig_bins.value_counts(normalize=True).sort_index()
        sim_bin_counts = sim_bins.value_counts(normalize=True).sort_index()
        
        bin_df_rows = []
        all_bins = sorted(set(orig_bin_counts.index) | set(sim_bin_counts.index))
        for bin_label in all_bins:
            orig_pct = orig_bin_counts.get(bin_label, 0.0)
            sim_pct = sim_bin_counts.get(bin_label, 0.0)
            orig_count = (orig_bins == bin_label).sum()
            sim_count = (sim_bins == bin_label).sum()
            
            bin_df_rows.append({
                "credit_score_bin": bin_label,
                "orig_count": int(orig_count),
                "sim_count": int(sim_count),
                "orig_pct": float(orig_pct) * 100,
                "sim_pct": float(sim_pct) * 100,
                "pct_diff": float(sim_pct - orig_pct) * 100,
            })
        bin_distribution_df = pd.DataFrame(bin_df_rows)

        if not per_group:
            return overall_df, None, bin_distribution_df

        rows = []
        og = orig.groupby(["case:LoanGoal", "case:ApplicationType"])
        sg = sim.groupby(["case:LoanGoal", "case:ApplicationType"])
        keys = set(og.groups.keys()) | set(sg.groups.keys())

        for k in keys:
            o = og.get_group(k)[o_cs].to_numpy() if k in og.groups else np.array([])
            s = sg.get_group(k)[s_cs].to_numpy() if k in sg.groups else np.array([])
            if len(o) == 0 or len(s) == 0:
                continue
            
            group_stats = {
                "case:LoanGoal": k[0],
                "case:ApplicationType": k[1],
                **{f"orig_{k}": v for k, v in summary(o).items()},
                **{f"sim_{k}": v for k, v in summary(s).items()},
            }
            
            if len(o) > 0 and len(s) > 0:
                group_stats["ks"] = ks_statistic_1d(o, s)
                group_stats["wasserstein"] = wasserstein_approx_1d(o, s)
                group_stats["mean_diff"] = float(np.mean(s) - np.mean(o))
                
                # Bin-Verteilung für diese Gruppe
                o_bins = credit_score_bin(pd.Series(o), bins=credit_score_bins, labels=credit_score_labels)
                s_bins = credit_score_bin(pd.Series(s), bins=credit_score_bins, labels=credit_score_labels)
                
                for bin_label in credit_score_labels:
                    o_pct = (o_bins == bin_label).mean() * 100
                    s_pct = (s_bins == bin_label).mean() * 100
                    group_stats[f"orig_bin_{bin_label}_pct"] = float(o_pct)
                    group_stats[f"sim_bin_{bin_label}_pct"] = float(s_pct)
                    group_stats[f"bin_{bin_label}_diff"] = float(s_pct - o_pct)
            else:
                group_stats["ks"] = np.nan
                group_stats["wasserstein"] = np.nan
                group_stats["mean_diff"] = np.nan
                for bin_label in credit_score_labels:
                    group_stats[f"orig_bin_{bin_label}_pct"] = np.nan
                    group_stats[f"sim_bin_{bin_label}_pct"] = np.nan
                    group_stats[f"bin_{bin_label}_diff"] = np.nan
            
            rows.append(group_stats)

        per_group_df = pd.DataFrame(rows).sort_values("orig_n", ascending=False).reset_index(drop=True)
        
        if print_results:
            print("\n=== VALIDATION: Credit Score ===")
            print("\n--- Overall Statistics ---")
            print(overall_df)
            
            if bin_distribution_df is not None and len(bin_distribution_df) > 0:
                print("\n--- Credit Score Bin Distribution ---")
                print(bin_distribution_df)
            
            if per_group_df is not None and len(per_group_df) > 0:
                print("\n--- Per Group Statistics ---")
                cols_to_show = ["case:LoanGoal", "case:ApplicationType", "orig_n", "sim_n", 
                                "orig_mean", "sim_mean", "mean_diff", "ks", "wasserstein"]
                available_cols = [c for c in cols_to_show if c in per_group_df.columns]
                print(per_group_df[available_cols].head(15))
                if len(per_group_df) > 15:
                    print(f"... ({len(per_group_df) - 15} weitere Gruppen)")
        
        return overall_df, per_group_df, bin_distribution_df
