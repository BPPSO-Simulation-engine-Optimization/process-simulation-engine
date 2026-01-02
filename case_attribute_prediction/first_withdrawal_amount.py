from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col, detect_rounding_step
from .metrics import ks_statistic_1d, wasserstein_approx_1d


class FirstWithdrawalAmountPredictor(AttributePredictorBase):
    name = "FirstWithdrawalAmount"

    def __init__(self, seed: int = 42, apply_rounding: bool = False):
        super().__init__(seed=seed)
        self.apply_rounding = bool(apply_rounding)

    def fit(
        self,
        df: pd.DataFrame,
        fwa_col: str = "FirstWithdrawalAmount",
        offered_col: str = "OfferedAmount",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
    ) -> "FirstWithdrawalAmountPredictor":
        """
        Neue Logik:
          firstwithdrawal_pct = (FirstWithdrawalAmount / OfferedAmount) * 100
          Segmentierung nach LoanGoal und ApplicationType (ohne CreditScore)
          Resampling aus empirischer Verteilung der prozentualen Werte
        """
        self.rng = np.random.default_rng(self.seed)

        fwa_c = resolve_col(df, fwa_col)
        off_c = resolve_col(df, offered_col)

        cols = list(group_cols) + [off_c, fwa_c]
        case_tbl = to_case_level(df, cols).copy()

        # numeric
        case_tbl[off_c] = pd.to_numeric(case_tbl[off_c], errors="coerce")
        case_tbl[fwa_c] = pd.to_numeric(case_tbl[fwa_c], errors="coerce")

        # Nur gültige Daten behalten
        m = (
            case_tbl[off_c].notna() & (case_tbl[off_c] > 0) &
            case_tbl[fwa_c].notna() & (case_tbl[fwa_c] >= 0)
        )
        d = case_tbl.loc[m].copy()

        # Berechne prozentuale Werte: (FWA / OfferedAmount) * 100
        d["_fwa_pct"] = (d[fwa_c] / d[off_c]) * 100.0
        # Entferne unplausible Werte (z.B. > 100%)
        d = d[(d["_fwa_pct"] >= 0) & (d["_fwa_pct"] <= 100)]
        
        # Kategorisierung für bimodale Verteilung:
        # - "zero_low": 0-5% (sehr niedrig)
        # - "medium": 5-95% (mittel)
        # - "high_full": 95-100% (hoch/vollständig)
        d["_category"] = pd.cut(
            d["_fwa_pct"],
            bins=[0, 5, 95, 100],
            labels=["zero_low", "medium", "high_full"],
            include_lowest=True,
            right=True
        )

        # Segmentierung nach LoanGoal und ApplicationType (OHNE CreditScore)
        dist_by_2 = {}  # (LoanGoal, ApplicationType) -> Array
        dist_by_1 = {}  # LoanGoal -> Array
        global_dist = d["_fwa_pct"].to_numpy()

        # Kategorien-Wahrscheinlichkeiten (für besseres Sampling der bimodalen Verteilung)
        cat_probs_by_2 = {}  # (LoanGoal, ApplicationType) -> dict mit Kategorien-Probs
        cat_probs_by_1 = {}  # LoanGoal -> dict mit Kategorien-Probs

        # Level 1: (LoanGoal, ApplicationType)
        for key, group in d.groupby([group_cols[0], group_cols[1]]):
            dist_by_2[key] = group["_fwa_pct"].to_numpy()
            cat_counts = group["_category"].value_counts(normalize=True)
            cat_probs_by_2[key] = {
                cat: float(cat_counts.get(cat, 0.0)) 
                for cat in ["zero_low", "medium", "high_full"]
            }
            total = sum(cat_probs_by_2[key].values())
            if total == 0:
                cat_probs_by_2[key] = {"zero_low": 0.33, "medium": 0.34, "high_full": 0.33}
            else:
                # Normalisiere auf 1.0
                for cat in cat_probs_by_2[key]:
                    cat_probs_by_2[key][cat] = cat_probs_by_2[key][cat] / total

        # Level 2: LoanGoal
        for key, group in d.groupby(group_cols[0]):
            dist_by_1[str(key)] = group["_fwa_pct"].to_numpy()
            cat_counts = group["_category"].value_counts(normalize=True)
            cat_probs_by_1[str(key)] = {
                cat: float(cat_counts.get(cat, 0.0)) 
                for cat in ["zero_low", "medium", "high_full"]
            }
            total = sum(cat_probs_by_1[str(key)].values())
            if total == 0:
                cat_probs_by_1[str(key)] = {"zero_low": 0.33, "medium": 0.34, "high_full": 0.33}
            else:
                for cat in cat_probs_by_1[str(key)]:
                    cat_probs_by_1[str(key)][cat] = cat_probs_by_1[str(key)][cat] / total

        # Globale Kategorien-Wahrscheinlichkeiten
        global_cat_counts = d["_category"].value_counts(normalize=True)
        global_cat_probs = {
            cat: float(global_cat_counts.get(cat, 0.0)) 
            for cat in ["zero_low", "medium", "high_full"]
        }
        total = sum(global_cat_probs.values())
        if total == 0:
            global_cat_probs = {"zero_low": 0.33, "medium": 0.34, "high_full": 0.33}
        else:
            for cat in global_cat_probs:
                global_cat_probs[cat] = global_cat_probs[cat] / total

        # Separate Verteilungen pro Kategorie (für besseres Sampling)
        # Level 1: (LoanGoal, ApplicationType, Category)
        dist_by_2_cat = {}
        for key, group in d.groupby([group_cols[0], group_cols[1], "_category"]):
            dist_by_2_cat[key] = group["_fwa_pct"].to_numpy()

        # Level 2: (LoanGoal, Category)
        dist_by_1_cat = {}
        for key, group in d.groupby([group_cols[0], "_category"]):
            # key ist ein Tupel (loan_goal, category)
            dist_by_1_cat[key] = group["_fwa_pct"].to_numpy()

        # Global pro Kategorie
        global_dist_by_cat = {}
        for cat, group in d.groupby("_category"):
            global_dist_by_cat[cat] = group["_fwa_pct"].to_numpy()

        rounding_step = detect_rounding_step(d[fwa_c].to_numpy()) if self.apply_rounding else None

        self.model = {
            "by_2": dist_by_2,  # (LoanGoal, ApplicationType) -> Array
            "by_1": dist_by_1,  # LoanGoal -> Array
            "global": global_dist,
            "cat_probs_by_2": cat_probs_by_2,  # Kategorien-Wahrscheinlichkeiten
            "cat_probs_by_1": cat_probs_by_1,
            "global_cat_probs": global_cat_probs,
            "dist_by_2_cat": dist_by_2_cat,  # Verteilungen pro Kategorie: (LoanGoal, ApplicationType, Category)
            "dist_by_1_cat": dist_by_1_cat,  # Verteilungen pro Kategorie: (LoanGoal, Category)
            "global_dist_by_cat": global_dist_by_cat,
            "rounding_step": rounding_step,
        }
        return self

    def predict(
        self,
        loan_goal: str,
        application_type: str,
        credit_score: float,  # Wird nicht mehr verwendet, aber für Rückwärtskompatibilität behalten
        requested_amount: float,  # Wird nicht mehr verwendet, aber für Rückwärtskompatibilität behalten
        offered_amount: float | None = None,  # WICHTIG: wird benötigt!
        mode: str = "sample",                 # "sample" | "mean"
        seed: int | None = None,
    ) -> float:
        """
        Vorhersage basierend auf prozentualer Verteilung.
        Zieht aus der Verteilung von (FirstWithdrawalAmount / OfferedAmount) * 100
        und multipliziert mit OfferedAmount.
        
        Unterstützt sowohl neues Format (by_3, by_2, by_1) als auch altes Format (mu_by_pair, etc.).
        """
        self._require_fitted()
        m = self.model
        assert m is not None

        # Rückwärtskompatibilität: Prüfe ob altes oder neues Modellformat
        is_old_format = "mu_by_pair" in m
        is_new_format = "by_3" in m

        # ALTES FORMAT: Nutze die alte Log-Ratio-Logik
        if is_old_format:
            req = float(requested_amount)
            if not np.isfinite(req) or req <= 0:
                return 0.0

            seg = (loan_goal, application_type)
            mu_seg = m["mu_by_pair"].get(seg)
            sig_seg = m["sigma_by_pair"].get(seg)
            n_seg = float(m["n_by_pair"].get(seg, 0))
            tau = float(m.get("tau", 200.0))

            global_mu = float(m["global_mu"])
            global_sigma = float(m["global_sigma"])

            # Shrinkage
            if mu_seg is None or sig_seg is None:
                mu = global_mu
                sigma = global_sigma
            else:
                w = n_seg / (n_seg + tau)
                mu = float(w * mu_seg + (1.0 - w) * global_mu)
                sigma = float(w * sig_seg + (1.0 - w) * global_sigma)

            # CreditScore-Effekt
            mu_adj = mu + float(m["beta_cs"]) * ((float(credit_score) - 650.0) / 50.0)

            if mode == "mean":
                ratio_mean = float(np.exp(mu_adj + 0.5 * (sigma ** 2)))
                ratio = float(np.clip(ratio_mean, float(m["eps"]), 1.0))
            else:
                rng = self.rng if seed is None else np.random.default_rng(int(seed))
                ratio = float(rng.lognormal(mean=mu_adj, sigma=max(1e-9, sigma)))
                ratio = float(np.clip(ratio, float(m["eps"]), 1.0))

            fwa = ratio * req

            # Optional: Rounding
            step = m.get("rounding_step")
            if step is not None and np.isfinite(step) and step > 0:
                fwa = round(fwa / step) * step

            fwa = float(np.clip(fwa, 0.0, req))

            # Optional: auf offered_amount clippen
            if offered_amount is not None and np.isfinite(offered_amount):
                fwa = float(min(fwa, float(offered_amount)))

            return fwa

        # NEUES FORMAT: Prozentuale Verteilung mit Kategorien-basiertem Sampling
        # (OHNE CreditScore)
        # OfferedAmount ist jetzt erforderlich
        if offered_amount is None or not np.isfinite(offered_amount) or offered_amount <= 0:
            return 0.0

        offered = float(offered_amount)
        rng = self.rng if seed is None else np.random.default_rng(int(seed))

        # SCHRITT 1: Bestimme Kategorie basierend auf Wahrscheinlichkeiten
        # Mehrstufige Suche nach Kategorien-Wahrscheinlichkeiten (ohne CreditScore)
        key_2 = (loan_goal, application_type)
        cat_probs = m.get("cat_probs_by_2", {}).get(key_2)

        if cat_probs is None:
            cat_probs = m.get("cat_probs_by_1", {}).get(str(loan_goal))

        if cat_probs is None:
            cat_probs = m.get("global_cat_probs", {"zero_low": 0.33, "medium": 0.34, "high_full": 0.33})

        # Ziehe Kategorie basierend auf Wahrscheinlichkeiten
        categories = list(cat_probs.keys())
        probs = [cat_probs[cat] for cat in categories]
        # Normalisiere (falls nicht genau 1.0)
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(categories)] * len(categories)
        
        selected_category = rng.choice(categories, p=probs)

        # SCHRITT 2: Ziehe Wert aus der Verteilung der gewählten Kategorie
        # Suche nach Verteilung für diese Kategorie (mehrstufig, ohne CreditScore)
        arr = None

        # Level 1: (LoanGoal, ApplicationType, Category)
        key_2_cat = (loan_goal, application_type, selected_category)
        arr = m.get("dist_by_2_cat", {}).get(key_2_cat)

        # Level 2: (LoanGoal, Category)
        if arr is None or len(arr) == 0:
            # dist_by_1_cat verwendet Tupel direkt aus groupby: (loan_goal, category)
            for key in m.get("dist_by_1_cat", {}):
                if len(key) == 2 and key[0] == loan_goal and key[1] == selected_category:
                    arr = m["dist_by_1_cat"][key]
                    break

        # Fallback: globale Verteilung für diese Kategorie
        if arr is None or len(arr) == 0:
            arr = m.get("global_dist_by_cat", {}).get(selected_category)

        # Wenn immer noch nichts, nutze Kategorie-spezifische Defaults
        if arr is None or len(arr) == 0:
            if selected_category == "zero_low":
                # Sample zwischen 0-5%
                arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            elif selected_category == "high_full":
                # Sample zwischen 95-100%
                arr = np.array([95.0, 96.0, 97.0, 98.0, 99.0, 100.0])
            else:  # medium
                # Sample zwischen 5-95%
                arr = m.get("global", np.array([50.0]))  # Fallback zu global

        # Ziehe aus der Verteilung
        if mode == "mean":
            fwa_pct = float(np.mean(arr))
        else:
            fwa_pct = float(rng.choice(arr))

        # Konvertiere Prozentwert zurück zu Betrag
        fwa = (fwa_pct / 100.0) * offered

        # Optional: Rounding
        step = m.get("rounding_step")
        if step is not None and np.isfinite(step) and step > 0:
            fwa = round(fwa / step) * step

        # Clipping: nicht mehr als OfferedAmount, nicht weniger als 0
        fwa = float(np.clip(fwa, 0.0, offered))

        return fwa

    def validate(
        self,
        df: pd.DataFrame,
        sim_df: pd.DataFrame,
        col: str = "FirstWithdrawalAmount",
        offered_col: str = "OfferedAmount",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
        case_level: bool = True,
        print_results: bool = True,
    ) -> pd.DataFrame:
        """
        Validiert FirstWithdrawalAmount nach LoanGoal und ApplicationType.
        Validiert sowohl absolute Beträge als auch prozentuale Werte und Kategorien-Verteilung.
        """
        col_o = resolve_col(df, col)
        col_s = resolve_col(sim_df, col)
        off_col_o = resolve_col(df, offered_col)
        off_col_s = resolve_col(sim_df, offered_col)

        if case_level:
            orig = to_case_level(df, list(group_cols) + [col_o, off_col_o]).copy()
        else:
            orig = df[list(group_cols) + [col_o, off_col_o]].copy()

        sim = sim_df[list(group_cols) + [col_s, off_col_s]].copy()

        # Numerische Konvertierung
        for x, cc, oc in [(orig, col_o, off_col_o), (sim, col_s, off_col_s)]:
            x[cc] = pd.to_numeric(x[cc], errors="coerce")
            x[oc] = pd.to_numeric(x[oc], errors="coerce")

        # Berechne prozentuale Werte
        orig["_fwa_pct"] = (orig[col_o] / orig[off_col_o]) * 100.0
        sim["_fwa_pct"] = (sim[col_s] / sim[off_col_s]) * 100.0

        # Entferne unplausible Werte
        orig = orig[
            (orig["_fwa_pct"] >= 0) & (orig["_fwa_pct"] <= 100) & 
            orig[col_o].notna() & orig[off_col_o].notna() & (orig[off_col_o] > 0)
        ].copy()
        sim = sim[
            (sim["_fwa_pct"] >= 0) & (sim["_fwa_pct"] <= 100) & 
            sim[col_s].notna() & sim[off_col_s].notna() & (sim[off_col_s] > 0)
        ].copy()

        # Kategorisierung
        def categorize_pct(pct):
            if pct <= 5:
                return "zero_low"
            elif pct < 95:
                return "medium"
            else:
                return "high_full"

        orig["_category"] = orig["_fwa_pct"].apply(categorize_pct)
        sim["_category"] = sim["_fwa_pct"].apply(categorize_pct)

        rows = []
        og = orig.groupby(list(group_cols))
        sg = sim.groupby(list(group_cols))

        for k, o in og:
            if k not in sg.groups:
                continue
            s = sg.get_group(k)

            # Absolute Beträge
            oa = o[col_o].dropna().to_numpy(dtype=float)
            sa = s[col_s].dropna().to_numpy(dtype=float)

            # Prozentuale Werte
            oa_pct = o["_fwa_pct"].dropna().to_numpy(dtype=float)
            sa_pct = s["_fwa_pct"].dropna().to_numpy(dtype=float)

            if oa.size == 0 or sa.size == 0:
                continue

            def stats(a: np.ndarray) -> dict:
                if a.size == 0:
                    return {
                        "n": 0,
                        "mean": np.nan,
                        "std": np.nan,
                        "p10": np.nan,
                        "p50": np.nan,
                        "p90": np.nan,
                    }
                return {
                    "n": int(a.size),
                    "mean": float(np.mean(a)),
                    "std": float(np.std(a, ddof=0)),
                    "p10": float(np.quantile(a, 0.10)),
                    "p50": float(np.quantile(a, 0.50)),
                    "p90": float(np.quantile(a, 0.90)),
                }

            os = stats(oa)
            ss = stats(sa)
            os_pct = stats(oa_pct)
            ss_pct = stats(sa_pct)

            # Kategorien-Verteilung (Prozent der Fälle in jeder Kategorie)
            orig_cat_counts = o["_category"].value_counts(normalize=True) * 100
            sim_cat_counts = s["_category"].value_counts(normalize=True) * 100

            row = {
                "case:LoanGoal": k[0],
                "case:ApplicationType": k[1],
                "orig_n": os["n"],
                "sim_n": ss["n"],
                # Absolute Beträge
                "orig_mean": os["mean"],
                "sim_mean": ss["mean"],
                "orig_std": os["std"],
                "sim_std": ss["std"],
                "orig_p10": os["p10"],
                "sim_p10": ss["p10"],
                "orig_p50": os["p50"],
                "sim_p50": ss["p50"],
                "orig_p90": os["p90"],
                "sim_p90": ss["p90"],
                "ks": ks_statistic_1d(oa, sa),
                "wasserstein": wasserstein_approx_1d(oa, sa),
                # Prozentuale Werte
                "orig_pct_mean": os_pct["mean"],
                "sim_pct_mean": ss_pct["mean"],
                "orig_pct_median": os_pct["p50"],
                "sim_pct_median": ss_pct["p50"],
                "ks_pct": ks_statistic_1d(oa_pct, sa_pct) if oa_pct.size > 0 and sa_pct.size > 0 else np.nan,
                "wasserstein_pct": wasserstein_approx_1d(oa_pct, sa_pct) if oa_pct.size > 0 and sa_pct.size > 0 else np.nan,
                # Kategorien-Verteilung (Prozent der Fälle)
                "orig_zero_low_pct": float(orig_cat_counts.get("zero_low", 0.0)),
                "sim_zero_low_pct": float(sim_cat_counts.get("zero_low", 0.0)),
                "diff_zero_low_pct": float(sim_cat_counts.get("zero_low", 0.0) - orig_cat_counts.get("zero_low", 0.0)),
                "orig_medium_pct": float(orig_cat_counts.get("medium", 0.0)),
                "sim_medium_pct": float(sim_cat_counts.get("medium", 0.0)),
                "diff_medium_pct": float(sim_cat_counts.get("medium", 0.0) - orig_cat_counts.get("medium", 0.0)),
                "orig_high_full_pct": float(orig_cat_counts.get("high_full", 0.0)),
                "sim_high_full_pct": float(sim_cat_counts.get("high_full", 0.0)),
                "diff_high_full_pct": float(sim_cat_counts.get("high_full", 0.0) - orig_cat_counts.get("high_full", 0.0)),
            }

            rows.append(row)

        result_df = pd.DataFrame(rows).sort_values("orig_n", ascending=False).reset_index(drop=True)
        
        if print_results:
            print("\n=== VALIDATION: First Withdrawal Amount ===")
            
            # Overall Statistiken
            print("\n--- Overall Statistics (Absolute Beträge) ---")
            overall_stats = {
                "orig_n": int(orig[col_o].notna().sum()),
                "sim_n": int(sim[col_s].notna().sum()),
                "orig_mean": float(orig[col_o].mean()),
                "sim_mean": float(sim[col_s].mean()),
                "orig_median": float(orig[col_o].median()),
                "sim_median": float(sim[col_s].median()),
            }
            for key, val in overall_stats.items():
                print(f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}")
            
            # Overall Prozentuale Werte
            print("\n--- Overall Statistics (Prozentual) ---")
            orig_pct_all = orig["_fwa_pct"].dropna().to_numpy()
            sim_pct_all = sim["_fwa_pct"].dropna().to_numpy()
            overall_pct_stats = {
                "orig_pct_mean": float(orig["_fwa_pct"].mean()),
                "sim_pct_mean": float(sim["_fwa_pct"].mean()),
                "orig_pct_median": float(orig["_fwa_pct"].median()),
                "sim_pct_median": float(sim["_fwa_pct"].median()),
            }
            for key, val in overall_pct_stats.items():
                print(f"{key}: {val:.2f}%")
            if len(orig_pct_all) > 0 and len(sim_pct_all) > 0:
                overall_ks_pct = ks_statistic_1d(orig_pct_all, sim_pct_all)
                overall_wasserstein_pct = wasserstein_approx_1d(orig_pct_all, sim_pct_all)
                print(f"KS-Statistik (prozentual): {overall_ks_pct:.4f}")
                print(f"Wasserstein-Distanz (prozentual): {overall_wasserstein_pct:.4f}")
            
            # Overall Kategorien-Verteilung
            print("\n--- Overall Category Distribution (% of cases) ---")
            orig_overall_cat = orig["_category"].value_counts(normalize=True) * 100
            sim_overall_cat = sim["_category"].value_counts(normalize=True) * 100
            for cat in ["zero_low", "medium", "high_full"]:
                orig_pct = float(orig_overall_cat.get(cat, 0.0))
                sim_pct = float(sim_overall_cat.get(cat, 0.0))
                diff = sim_pct - orig_pct
                print(f"{cat}: Orig={orig_pct:.2f}%, Sim={sim_pct:.2f}%, Diff={diff:+.2f}%")
            
            # Per Group Statistiken - Prozentuale Verteilung (FirstWithdrawalAmount/OfferedAmount * 100)
            # mit KS-Statistik zwischen Original und Simulation
            print("\n--- Per Group Statistics: Prozentuale Verteilung (FirstWithdrawalAmount/OfferedAmount * 100) ---")
            print("KS-Statistik misst Unterschied zwischen Original- und Simulations-Verteilung der Prozentsätze")
            cols_pct = [
                "case:LoanGoal", "case:ApplicationType", "orig_n", "sim_n",
                "orig_pct_mean", "sim_pct_mean", "orig_pct_median", "sim_pct_median",
                "ks_pct", "wasserstein_pct",
                "orig_zero_low_pct", "sim_zero_low_pct", "diff_zero_low_pct",
                "orig_medium_pct", "sim_medium_pct", "diff_medium_pct",
                "orig_high_full_pct", "sim_high_full_pct", "diff_high_full_pct",
            ]
            available_pct_cols = [c for c in cols_pct if c in result_df.columns]
            pct_df = result_df[available_pct_cols].head(20).copy()
            # Formatierung für bessere Lesbarkeit
            if "orig_pct_mean" in pct_df.columns:
                pct_df["orig_pct_mean"] = pct_df["orig_pct_mean"].apply(lambda x: f"{x:.2f}%" if not np.isnan(x) else "N/A")
            if "sim_pct_mean" in pct_df.columns:
                pct_df["sim_pct_mean"] = pct_df["sim_pct_mean"].apply(lambda x: f"{x:.2f}%" if not np.isnan(x) else "N/A")
            if "orig_pct_median" in pct_df.columns:
                pct_df["orig_pct_median"] = pct_df["orig_pct_median"].apply(lambda x: f"{x:.2f}%" if not np.isnan(x) else "N/A")
            if "sim_pct_median" in pct_df.columns:
                pct_df["sim_pct_median"] = pct_df["sim_pct_median"].apply(lambda x: f"{x:.2f}%" if not np.isnan(x) else "N/A")
            if "ks_pct" in pct_df.columns:
                pct_df["ks_pct"] = pct_df["ks_pct"].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
            if "wasserstein_pct" in pct_df.columns:
                pct_df["wasserstein_pct"] = pct_df["wasserstein_pct"].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
            print(pct_df.to_string())
            if len(result_df) > 20:
                print(f"\n... ({len(result_df) - 20} weitere Gruppen)")
            
            # Zusammenfassung der KS-Statistik über alle Gruppen
            print("\n--- Zusammenfassung KS-Statistik (prozentuale Verteilung) ---")
            ks_pct_values = result_df["ks_pct"].dropna()
            if len(ks_pct_values) > 0:
                print(f"Durchschnittliche KS-Statistik: {ks_pct_values.mean():.4f}")
                print(f"Median KS-Statistik: {ks_pct_values.median():.4f}")
                print(f"Min KS-Statistik: {ks_pct_values.min():.4f}")
                print(f"Max KS-Statistik: {ks_pct_values.max():.4f}")
                print(f"Anzahl Gruppen mit KS-Statistik < 0.1: {(ks_pct_values < 0.1).sum()} ({(ks_pct_values < 0.1).sum()/len(ks_pct_values)*100:.1f}%)")
                print(f"Anzahl Gruppen mit KS-Statistik < 0.2: {(ks_pct_values < 0.2).sum()} ({(ks_pct_values < 0.2).sum()/len(ks_pct_values)*100:.1f}%)")
            
            # Per Group Statistiken - Absolute Beträge mit KS-Statistik
            print("\n--- Per Group Statistics: Absolute Beträge mit KS-Statistik (Top 20) ---")
            cols_abs = [
                "case:LoanGoal", "case:ApplicationType", "orig_n", "sim_n",
                "orig_mean", "sim_mean", "orig_median", "sim_median",
                "ks", "wasserstein",
            ]
            available_abs_cols = [c for c in cols_abs if c in result_df.columns]
            print(result_df[available_abs_cols].head(20).to_string())
            if len(result_df) > 20:
                print(f"\n... ({len(result_df) - 20} weitere Gruppen)")
        
        return result_df
