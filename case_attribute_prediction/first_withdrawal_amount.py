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
        requested_col: str = "case:RequestedAmount",
        score_col: str = "CreditScore",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
        eps: float = 1e-9,
        tau: float = 200.0,
    ) -> "FirstWithdrawalAmountPredictor":
        """
        Notebook-Logik:
          ratio = FWA / Requested
          log(ratio) ~ Normal(mu, sigma), mu_adjust = mu + beta * (CS-650)/50
          plus Segment-Parameter + Shrinkage via tau (hier gespeichert, in predict genutzt).
        """
        self.rng = np.random.default_rng(self.seed)

        fwa_c = resolve_col(df, fwa_col)
        req_c = resolve_col(df, requested_col)
        sc_c = resolve_col(df, score_col)

        cols = list(group_cols) + [req_c, sc_c, fwa_c]
        case_tbl = to_case_level(df, cols).copy()

        # numeric
        case_tbl[req_c] = pd.to_numeric(case_tbl[req_c], errors="coerce")
        case_tbl[sc_c] = pd.to_numeric(case_tbl[sc_c], errors="coerce")
        case_tbl[fwa_c] = pd.to_numeric(case_tbl[fwa_c], errors="coerce")

        m = (
            case_tbl[req_c].notna() & (case_tbl[req_c] > 0) &
            case_tbl[fwa_c].notna() & (case_tbl[fwa_c] >= 0) &
            case_tbl[sc_c].notna()
        )
        d = case_tbl.loc[m].copy()

        ratio = (d[fwa_c] / (d[req_c] + eps)).clip(lower=eps, upper=1.0)
        d["_log_ratio"] = np.log(ratio)

        global_mu = float(d["_log_ratio"].mean())
        global_sigma = float(d["_log_ratio"].std(ddof=0) if len(d) else 0.5)

        grp = d.groupby(list(group_cols))
        seg_mu = grp["_log_ratio"].mean().to_dict()
        seg_sigma = grp["_log_ratio"].std(ddof=0).fillna(global_sigma).to_dict()
        seg_n = grp.size().to_dict()

        x = (d[sc_c] - 650.0) / 50.0
        y = d["_log_ratio"] - global_mu
        beta = float(np.nan_to_num(np.cov(x, y, ddof=0)[0, 1] / (np.var(x, ddof=0) + eps)))

        rounding_step = detect_rounding_step(d[fwa_c].to_numpy()) if self.apply_rounding else None

        self.model = {
            "mu_by_pair": seg_mu,
            "sigma_by_pair": seg_sigma,
            "n_by_pair": seg_n,
            "global_mu": global_mu,
            "global_sigma": global_sigma,
            "beta_cs": beta,
            "tau": float(tau),
            "eps": float(eps),
            "rounding_step": rounding_step,
        }
        return self

    def predict(
        self,
        loan_goal: str,
        application_type: str,
        credit_score: float,
        requested_amount: float,
        offered_amount: float | None = None,  # optional (für Clip, falls gewünscht)
        mode: str = "sample",                 # "sample" | "mean"
        seed: int | None = None,
    ) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

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

        # Shrinkage (stabil; tau identisch zur Modell-Intention)
        if mu_seg is None or sig_seg is None:
            mu = global_mu
            sigma = global_sigma
        else:
            w = n_seg / (n_seg + tau)
            mu = float(w * mu_seg + (1.0 - w) * global_mu)
            sigma = float(w * sig_seg + (1.0 - w) * global_sigma)

        # CreditScore-Effekt (wie Notebook)
        mu_adj = mu + float(m["beta_cs"]) * ((float(credit_score) - 650.0) / 50.0)

        if mode == "mean":
            # E[lognormal] = exp(mu + 0.5*sigma^2) im ratio-space; dann clamp
            ratio_mean = float(np.exp(mu_adj + 0.5 * (sigma ** 2)))
            ratio = float(np.clip(ratio_mean, float(m["eps"]), 1.0))
        else:
            rng = self.rng if seed is None else np.random.default_rng(int(seed))
            ratio = float(rng.lognormal(mean=mu_adj, sigma=max(1e-9, sigma)))
            ratio = float(np.clip(ratio, float(m["eps"]), 1.0))

        fwa = ratio * req

        # optional rounding wie im Sampler-Notebook
        step = m.get("rounding_step")
        if step is not None and np.isfinite(step) and step > 0:
            fwa = round(fwa / step) * step

        fwa = float(np.clip(fwa, 0.0, req))

        # optional: zusätzlich auf offered_amount clippen, wenn das in Ihrer Domäne gilt
        if offered_amount is not None and np.isfinite(offered_amount):
            fwa = float(min(fwa, float(offered_amount)))

        return fwa

    def validate(
        self,
        df: pd.DataFrame,
        sim_df: pd.DataFrame,
        col: str = "FirstWithdrawalAmount",
        score_col: str = "CreditScore",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
        score_bins=(0, 500, 600, 650, 700, 750, 1000),
        case_level: bool = True,
    ) -> pd.DataFrame:
        col_o = resolve_col(df, col)
        col_s = resolve_col(sim_df, col)
        sc_o = resolve_col(df, score_col)
        sc_s = resolve_col(sim_df, score_col)

        if case_level:
            orig = to_case_level(df, list(group_cols) + [sc_o, col_o]).copy()
        else:
            orig = df[list(group_cols) + [sc_o, col_o]].copy()

        sim = sim_df[list(group_cols) + [sc_s, col_s]].copy()

        for x, sc, cc in [(orig, sc_o, col_o), (sim, sc_s, col_s)]:
            x[sc] = pd.to_numeric(x[sc], errors="coerce")
            x[cc] = pd.to_numeric(x[cc], errors="coerce")

        labels = [f"{score_bins[i]}–{score_bins[i+1]-1}" for i in range(len(score_bins)-1)]
        orig["_score_bin"] = pd.cut(orig[sc_o], bins=score_bins, labels=labels)
        sim["_score_bin"] = pd.cut(sim[sc_s], bins=score_bins, labels=labels)

        rows = []
        grp_cols = list(group_cols) + ["_score_bin"]
        og = orig.groupby(grp_cols)
        sg = sim.groupby(grp_cols)

        for k, o in og:
            if k not in sg.groups:
                continue
            s = sg.get_group(k)

            oa = o[col_o].dropna().to_numpy(dtype=float)
            sa = s[col_s].dropna().to_numpy(dtype=float)
            if oa.size == 0 or sa.size == 0:
                continue

            def stats(a: np.ndarray) -> dict:
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

            rows.append({
                "case:LoanGoal": k[0],
                "case:ApplicationType": k[1],
                "CreditScore_bin": k[2],
                "orig_n": os["n"],
                "sim_n": ss["n"],
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
            })

        return pd.DataFrame(rows).sort_values("orig_n", ascending=False).reset_index(drop=True)
