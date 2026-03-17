from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit, expit

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col, detect_rounding_step
from .metrics import ks_statistic_1d, wasserstein_approx_1d


class FirstWithdrawalAmountPredictor(AttributePredictorBase):
    """
    Mixture-Modell für FirstWithdrawalAmount:

      FWA_pct = (FirstWithdrawalAmount / OfferedAmount) * 100

      Verteilung = P(zero)  * δ(0)          ← kein Auszahlungsauftakt  (≤ ZERO_THRESHOLD)
                 + P(full)  * δ(100)         ← volle Auszahlung         (≥ FULL_THRESHOLD)
                 + P(part.) * KDE_logit(x)   ← Teilauszahlung (1–99 %)

    Segmentierung nach (LoanGoal, ApplicationType), Fallback auf LoanGoal → global.

    Kernverbesserungen gegenüber dem alten Ansatz:
      1. Explizite Massenatome bei 0 % und 100 % statt breiter 0–5 %/95–100 %-Bins.
      2. Logit-KDE für den kontinuierlichen Teilbereich (bounded interpolation).
      3. Gruppenspezifische Wahrscheinlichkeiten (z. B. „Limit raise" → p_full ≈ 0).
    """

    name = "FirstWithdrawalAmount"

    # Klassenkonstanten – werden auch im Modell-Dict gespeichert
    ZERO_THRESHOLD: float = 1.0   # FWA_pct ≤ 1 %  → „zero"
    FULL_THRESHOLD: float = 99.0  # FWA_pct ≥ 99 % → „full"
    KDE_MIN_SAMPLES: int = 8      # Mindest-Stichprobe für Logit-KDE

    def __init__(self, seed: int = 42, apply_rounding: bool = False):
        super().__init__(seed=seed)
        self.apply_rounding = bool(apply_rounding)

    # ─── interne Hilfsmethoden ────────────────────────────────────────────────

    def _build_segment(
        self,
        arr_pct: np.ndarray,
        zero_thr: float,
        full_thr: float,
    ) -> dict | None:
        """
        Baut ein Mixture-Segment aus einem Array von FWA-Prozentwerten (0–100).

        Rückgabe-Dict:
            p_zero       – Anteil ≤ zero_thr
            p_full       – Anteil ≥ full_thr
            p_partial    – Anteil (zero_thr, full_thr)
            partial_vals – Rohe Prozentwerte im partiellen Bereich
            kde_logit    – scipy gaussian_kde auf logit-transformierten Werten (oder None)
            logit_vals   – gespeicherte logit-Werte für Bandbreiten-Reproduzierbarkeit
            bw_factor    – KDE-Bandbreitenfaktor (Scott)
        """
        arr_pct = np.asarray(arr_pct, dtype=float)
        n = len(arr_pct)
        if n == 0:
            return None

        mask_zero = arr_pct <= zero_thr
        mask_full = arr_pct >= full_thr
        mask_part = ~mask_zero & ~mask_full

        p_zero = float(mask_zero.sum() / n)
        p_full = float(mask_full.sum() / n)
        p_part = float(mask_part.sum() / n)

        partial_vals = arr_pct[mask_part].copy()

        # Logit-KDE für den partiellen Bereich (bounded distribution → logit-transform)
        kde_logit = None
        logit_vals = np.array([], dtype=float)
        bw_factor = np.nan

        if len(partial_vals) >= self.KDE_MIN_SAMPLES:
            try:
                range_w = full_thr - zero_thr
                eps = 1e-6
                normalized = (partial_vals - zero_thr) / range_w
                normalized = np.clip(normalized, eps, 1.0 - eps)
                logit_vals = logit(normalized)
                kde_logit = stats.gaussian_kde(logit_vals)  # Scott's Regel
                bw_factor = float(kde_logit.factor)
            except Exception:
                kde_logit = None
                logit_vals = np.array([], dtype=float)
                bw_factor = np.nan

        return {
            "p_zero": p_zero,
            "p_full": p_full,
            "p_partial": p_part,
            "partial_vals": partial_vals,
            "kde_logit": kde_logit,
            "logit_vals": logit_vals,
            "bw_factor": bw_factor,
        }

    def _sample_from_segment(
        self,
        seg: dict,
        rng: np.random.Generator,
        zero_thr: float,
        full_thr: float,
        mode: str = "sample",
    ) -> float:
        """
        Zieht einen FWA-Prozentwert (0–100) aus dem gegebenen Segment.

        mode="sample" → stochastische Ziehung
        mode="mean"   → Erwartungswert
        """
        p_zero = seg["p_zero"]
        p_full = seg["p_full"]
        p_part = seg["p_partial"]
        partial_vals = seg["partial_vals"]
        kde_logit = seg.get("kde_logit")

        if mode == "mean":
            part_mean = float(np.mean(partial_vals)) if len(partial_vals) > 0 else (zero_thr + full_thr) / 2
            return float(p_zero * 0.0 + p_full * 100.0 + p_part * part_mean)

        # ── Kategorie ziehen ─────────────────────────────────────────────────
        cats = ["zero", "full", "partial"]
        probs = np.array([p_zero, p_full, p_part], dtype=float)

        # Sonderfall: p_partial > 0 aber keine Werte → partial auf full schieben
        if p_part > 0 and len(partial_vals) == 0:
            probs[1] += probs[2]
            probs[2] = 0.0

        total = probs.sum()
        if total <= 0:
            return 0.0
        probs /= total

        cat = str(rng.choice(cats, p=probs))

        if cat == "zero":
            return 0.0
        if cat == "full":
            return 100.0

        # ── Partial: Logit-KDE oder Bootstrap ────────────────────────────────
        if kde_logit is not None:
            range_w = full_thr - zero_thr
            MAX_ATTEMPTS = 30
            for _ in range(MAX_ATTEMPTS):
                seed_i = int(rng.integers(0, 2 ** 31))
                logit_s = float(kde_logit.resample(1, seed=seed_i)[0, 0])
                pct = float(expit(logit_s)) * range_w + zero_thr
                if zero_thr < pct < full_thr:
                    return float(pct)
            # Fallback bei gescheitertem Rejection Sampling

        if len(partial_vals) > 0:
            return float(rng.choice(partial_vals))
        return float((zero_thr + full_thr) / 2)

    # ─── öffentliche API ──────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        fwa_col: str = "FirstWithdrawalAmount",
        offered_col: str = "OfferedAmount",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
    ) -> "FirstWithdrawalAmountPredictor":
        """
        Fittet das Mixture-KDE-Modell.

        Ablauf:
          1. FWA_pct = (FWA / OfferedAmount) × 100 auf Case-Ebene berechnen.
          2. Für jedes Segment (by_2, by_1, global) ein Mixture-Segment bauen.
          3. Modell als Dict persistieren (Format-Tag: "mixture_kde_v2").
        """
        self.rng = np.random.default_rng(self.seed)

        fwa_c = resolve_col(df, fwa_col)
        off_c = resolve_col(df, offered_col)

        cols = list(group_cols) + [off_c, fwa_c]
        case_tbl = to_case_level(df, cols).copy()

        case_tbl[off_c] = pd.to_numeric(case_tbl[off_c], errors="coerce")
        case_tbl[fwa_c] = pd.to_numeric(case_tbl[fwa_c], errors="coerce")

        mask = (
            case_tbl[off_c].notna() & (case_tbl[off_c] > 0) &
            case_tbl[fwa_c].notna() & (case_tbl[fwa_c] >= 0)
        )
        d = case_tbl.loc[mask].copy()

        d["_fwa_pct"] = (d[fwa_c] / d[off_c]) * 100.0
        d = d[(d["_fwa_pct"] >= 0.0) & (d["_fwa_pct"] <= 100.0)].copy()

        zt = self.ZERO_THRESHOLD
        ft = self.FULL_THRESHOLD

        # ── Level-2-Segmente: (LoanGoal, ApplicationType) ────────────────────
        segs_by_2: dict = {}
        for key, grp in d.groupby([group_cols[0], group_cols[1]]):
            seg = self._build_segment(grp["_fwa_pct"].to_numpy(), zt, ft)
            if seg is not None:
                segs_by_2[key] = seg

        # ── Level-1-Segmente: LoanGoal ────────────────────────────────────────
        segs_by_1: dict = {}
        for key, grp in d.groupby(group_cols[0]):
            seg = self._build_segment(grp["_fwa_pct"].to_numpy(), zt, ft)
            if seg is not None:
                segs_by_1[str(key)] = seg

        # ── Globales Segment ──────────────────────────────────────────────────
        global_seg = self._build_segment(d["_fwa_pct"].to_numpy(), zt, ft)

        rounding_step = detect_rounding_step(d[fwa_c].to_numpy()) if self.apply_rounding else None

        self.model = {
            "format": "mixture_kde_v2",
            "ZERO_THRESHOLD": zt,
            "FULL_THRESHOLD": ft,
            "segs_by_2": segs_by_2,
            "segs_by_1": segs_by_1,
            "global_seg": global_seg,
            "rounding_step": rounding_step,
        }
        return self

    def predict(
        self,
        loan_goal: str,
        application_type: str,
        credit_score: float,          # ← für Rückwärtskompatibilität, wird nicht mehr genutzt
        requested_amount: float = None,  # ← für Rückwärtskompatibilität, wird nicht mehr genutzt
        offered_amount: float | None = None,
        mode: str = "sample",
        seed: int | None = None,
    ) -> float:
        """
        Sagt FirstWithdrawalAmount vorher.

        Unterstützt drei Modell-Formate (Rückwärtskompatibilität):
          - "mixture_kde_v2"  → neues Mixture-KDE-Modell (dieser Code)
          - altes Prozent-Bootstrap-Format (no "format" key)
          - ältestes Log-Ratio-Format ("mu_by_pair" key)
        """
        self._require_fitted()
        m = self.model
        assert m is not None

        # ── ÄLTESTES FORMAT: Log-Ratio ────────────────────────────────────────
        if "mu_by_pair" in m:
            req = float(requested_amount) if requested_amount is not None else 0.0
            if not np.isfinite(req) or req <= 0:
                return 0.0

            seg = (loan_goal, application_type)
            mu_seg = m["mu_by_pair"].get(seg)
            sig_seg = m["sigma_by_pair"].get(seg)
            n_seg = float(m["n_by_pair"].get(seg, 0))
            tau = float(m.get("tau", 200.0))

            global_mu = float(m["global_mu"])
            global_sigma = float(m["global_sigma"])

            if mu_seg is None or sig_seg is None:
                mu, sigma = global_mu, global_sigma
            else:
                w = n_seg / (n_seg + tau)
                mu = float(w * mu_seg + (1.0 - w) * global_mu)
                sigma = float(w * sig_seg + (1.0 - w) * global_sigma)

            mu_adj = mu + float(m["beta_cs"]) * ((float(credit_score) - 650.0) / 50.0)

            if mode == "mean":
                ratio = float(np.clip(np.exp(mu_adj + 0.5 * sigma ** 2), float(m["eps"]), 1.0))
            else:
                rng = self.rng if seed is None else np.random.default_rng(int(seed))
                ratio = float(np.clip(rng.lognormal(mean=mu_adj, sigma=max(1e-9, sigma)), float(m["eps"]), 1.0))

            fwa = ratio * req
            step = m.get("rounding_step")
            if step is not None and np.isfinite(step) and step > 0:
                fwa = round(fwa / step) * step
            fwa = float(np.clip(fwa, 0.0, req))
            if offered_amount is not None and np.isfinite(offered_amount):
                fwa = float(min(fwa, float(offered_amount)))
            return fwa

        # ── NEUES MIXTURE-KDE-FORMAT ──────────────────────────────────────────
        if m.get("format") == "mixture_kde_v2":
            if offered_amount is None or not np.isfinite(float(offered_amount)) or float(offered_amount) <= 0:
                return 0.0

            offered = float(offered_amount)
            rng = self.rng if seed is None else np.random.default_rng(int(seed))

            zt = float(m.get("ZERO_THRESHOLD", self.ZERO_THRESHOLD))
            ft = float(m.get("FULL_THRESHOLD", self.FULL_THRESHOLD))

            # Segment-Hierarchie: by_2 → by_1 → global
            key_2 = (str(loan_goal), str(application_type))
            seg = m["segs_by_2"].get(key_2)
            if seg is None:
                seg = m["segs_by_1"].get(str(loan_goal))
            if seg is None:
                seg = m.get("global_seg")
            if seg is None:
                return 0.0

            fwa_pct = self._sample_from_segment(seg, rng, zt, ft, mode=mode)

            # Prozentwert → Absolute Betrag
            if fwa_pct >= 99.9:
                fwa = offered          # exakt OfferedAmount
            elif fwa_pct <= 0.1:
                fwa = 0.0
            else:
                fwa = (fwa_pct / 100.0) * offered

            # Optionales Rounding
            step = m.get("rounding_step")
            if step is not None and np.isfinite(step) and step > 0:
                fwa = round(fwa / step) * step

            return float(np.clip(fwa, 0.0, offered))

        # ── ZWISCHENFORMAT: Prozent-Bootstrap (altes "by_2/by_1/global"-Format) ─
        if offered_amount is None or not np.isfinite(float(offered_amount)) or float(offered_amount) <= 0:
            return 0.0

        offered = float(offered_amount)
        rng = self.rng if seed is None else np.random.default_rng(int(seed))

        key_2 = (loan_goal, application_type)
        cat_probs = m.get("cat_probs_by_2", {}).get(key_2)
        if cat_probs is None:
            cat_probs = m.get("cat_probs_by_1", {}).get(str(loan_goal))
        if cat_probs is None:
            cat_probs = m.get("global_cat_probs", {"zero_low": 0.33, "medium": 0.34, "high_full": 0.33})

        categories = list(cat_probs.keys())
        probs = [cat_probs[cat] for cat in categories]
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(categories)] * len(categories)

        selected_category = str(rng.choice(categories, p=probs))

        arr = None
        key_2_cat = (loan_goal, application_type, selected_category)
        arr = m.get("dist_by_2_cat", {}).get(key_2_cat)

        if arr is None or len(arr) == 0:
            for key in m.get("dist_by_1_cat", {}):
                if len(key) == 2 and key[0] == loan_goal and key[1] == selected_category:
                    arr = m["dist_by_1_cat"][key]
                    break

        if arr is None or len(arr) == 0:
            arr = m.get("global_dist_by_cat", {}).get(selected_category)

        if arr is None or len(arr) == 0:
            if selected_category == "zero_low":
                arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            elif selected_category == "high_full":
                arr = np.array([95.0, 96.0, 97.0, 98.0, 99.0, 100.0])
            else:
                arr = m.get("global", np.array([50.0]))

        fwa_pct = float(np.mean(arr)) if mode == "mean" else float(rng.choice(arr))
        fwa = (fwa_pct / 100.0) * offered

        step = m.get("rounding_step")
        if step is not None and np.isfinite(step) and step > 0:
            fwa = round(fwa / step) * step

        return float(np.clip(fwa, 0.0, offered))

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
        Validiert FirstWithdrawalAmount gegen simulierte Werte.
        Berichtet absolute Beträge, Prozentwerte und Kategorien-Verteilung.
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

        for x, cc, oc in [(orig, col_o, off_col_o), (sim, col_s, off_col_s)]:
            x[cc] = pd.to_numeric(x[cc], errors="coerce")
            x[oc] = pd.to_numeric(x[oc], errors="coerce")

        orig["_fwa_pct"] = (orig[col_o] / orig[off_col_o]) * 100.0
        sim["_fwa_pct"]  = (sim[col_s]  / sim[off_col_s])  * 100.0

        orig = orig[
            orig["_fwa_pct"].between(0, 100) &
            orig[col_o].notna() & orig[off_col_o].notna() & (orig[off_col_o] > 0)
        ].copy()
        sim = sim[
            sim["_fwa_pct"].between(0, 100) &
            sim[col_s].notna() & sim[off_col_s].notna() & (sim[off_col_s] > 0)
        ].copy()

        def _cat(pct):
            if pct <= 1:
                return "zero"
            elif pct >= 99:
                return "full"
            return "partial"

        orig["_cat"] = orig["_fwa_pct"].apply(_cat)
        sim["_cat"]  = sim["_fwa_pct"].apply(_cat)

        def _stats(a: np.ndarray) -> dict:
            if a.size == 0:
                return {"n": 0, "mean": np.nan, "std": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}
            return {
                "n":   int(a.size),
                "mean": float(np.mean(a)),
                "std":  float(np.std(a, ddof=0)),
                "p10":  float(np.quantile(a, 0.10)),
                "p50":  float(np.quantile(a, 0.50)),
                "p90":  float(np.quantile(a, 0.90)),
            }

        rows = []
        og = orig.groupby(list(group_cols))
        sg = sim.groupby(list(group_cols))

        for k, o in og:
            if k not in sg.groups:
                continue
            s = sg.get_group(k)

            oa     = o[col_o].dropna().to_numpy(dtype=float)
            sa     = s[col_s].dropna().to_numpy(dtype=float)
            oa_pct = o["_fwa_pct"].dropna().to_numpy(dtype=float)
            sa_pct = s["_fwa_pct"].dropna().to_numpy(dtype=float)

            if oa.size == 0 or sa.size == 0:
                continue

            os_ = _stats(oa);   ss_ = _stats(sa)
            osp = _stats(oa_pct); ssp = _stats(sa_pct)

            orig_cat = o["_cat"].value_counts(normalize=True) * 100
            sim_cat  = s["_cat"].value_counts(normalize=True) * 100

            rows.append({
                "case:LoanGoal":        k[0],
                "case:ApplicationType": k[1],
                "orig_n":   os_["n"],
                "sim_n":    ss_["n"],
                "orig_mean": os_["mean"],  "sim_mean": ss_["mean"],
                "orig_p50":  os_["p50"],   "sim_p50":  ss_["p50"],
                "orig_p90":  os_["p90"],   "sim_p90":  ss_["p90"],
                "ks":         ks_statistic_1d(oa,     sa),
                "ks_pct":     ks_statistic_1d(oa_pct, sa_pct) if oa_pct.size > 0 and sa_pct.size > 0 else np.nan,
                "wasserstein": wasserstein_approx_1d(oa, sa),
                "orig_zero_pct":    float(orig_cat.get("zero",    0.0)),
                "sim_zero_pct":     float(sim_cat.get("zero",     0.0)),
                "diff_zero_pct":    float(sim_cat.get("zero",     0.0) - orig_cat.get("zero",    0.0)),
                "orig_full_pct":    float(orig_cat.get("full",    0.0)),
                "sim_full_pct":     float(sim_cat.get("full",     0.0)),
                "diff_full_pct":    float(sim_cat.get("full",     0.0) - orig_cat.get("full",    0.0)),
                "orig_partial_pct": float(orig_cat.get("partial", 0.0)),
                "sim_partial_pct":  float(sim_cat.get("partial",  0.0)),
                "diff_partial_pct": float(sim_cat.get("partial",  0.0) - orig_cat.get("partial", 0.0)),
                "orig_pct_mean":   osp["mean"], "sim_pct_mean":   ssp["mean"],
                "orig_pct_median": osp["p50"],  "sim_pct_median": ssp["p50"],
            })

        result_df = pd.DataFrame(rows).sort_values("orig_n", ascending=False).reset_index(drop=True)

        if print_results:
            print("\n=== VALIDATION: First Withdrawal Amount ===")

            orig_pct_all = orig["_fwa_pct"].dropna().to_numpy()
            sim_pct_all  = sim["_fwa_pct"].dropna().to_numpy()

            print("\n--- Overall Statistics ---")
            for label, arr in [("orig", orig[col_o].dropna().to_numpy()), ("sim", sim[col_s].dropna().to_numpy())]:
                print(f"  {label}: n={len(arr)}, mean={np.mean(arr):.0f}, median={np.median(arr):.0f}")

            print("\n--- Overall Category Distribution (% of cases) ---")
            orig_cat_all = orig["_cat"].value_counts(normalize=True) * 100
            sim_cat_all  = sim["_cat"].value_counts(normalize=True) * 100
            for cat in ["zero", "partial", "full"]:
                o_p = float(orig_cat_all.get(cat, 0.0))
                s_p = float(sim_cat_all.get(cat,  0.0))
                print(f"  {cat:10s}: Orig={o_p:.1f}%  Sim={s_p:.1f}%  Diff={s_p - o_p:+.1f}%")

            if len(orig_pct_all) > 0 and len(sim_pct_all) > 0:
                print(f"\n  KS (pct, overall): {ks_statistic_1d(orig_pct_all, sim_pct_all):.4f}")

            print("\n--- Per Group (KS on % values, sorted by n) ---")
            cols_show = [
                "case:LoanGoal", "case:ApplicationType", "orig_n", "sim_n",
                "orig_pct_mean", "sim_pct_mean", "ks_pct",
                "orig_zero_pct", "sim_zero_pct", "diff_zero_pct",
                "orig_full_pct", "sim_full_pct", "diff_full_pct",
            ]
            avail = [c for c in cols_show if c in result_df.columns]
            print(result_df[avail].head(20).to_string(index=False))
            if len(result_df) > 20:
                print(f"... ({len(result_df) - 20} weitere Gruppen)")

            print("\n--- KS Zusammenfassung (% values) ---")
            ks_vals = result_df["ks_pct"].dropna()
            if len(ks_vals) > 0:
                print(f"  Ø KS: {ks_vals.mean():.4f}   Median: {ks_vals.median():.4f}"
                      f"   Max: {ks_vals.max():.4f}")
                print(f"  Gruppen KS < 0.10: {(ks_vals < 0.10).sum()}/{len(ks_vals)}")
                print(f"  Gruppen KS < 0.20: {(ks_vals < 0.20).sum()}/{len(ks_vals)}")

        return result_df
