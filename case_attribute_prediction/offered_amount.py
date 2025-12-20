from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col, credit_score_bin
from .metrics import ks_statistic_1d, wasserstein_approx_1d


class OfferedAmountPredictor(AttributePredictorBase):
    name = "OfferedAmount"

    def fit(self, df: pd.DataFrame, offered_col: str = "OfferedAmount") -> "OfferedAmountPredictor":
        self.rng = np.random.default_rng(self.seed)

        off_col = resolve_col(df, offered_col)
        cols = ["case:LoanGoal", "case:ApplicationType", "case:RequestedAmount", off_col]
        case_tbl = to_case_level(df, cols).copy().dropna()

        case_tbl["case:RequestedAmount"] = pd.to_numeric(case_tbl["case:RequestedAmount"], errors="coerce")
        case_tbl[off_col] = pd.to_numeric(case_tbl[off_col], errors="coerce")
        case_tbl = case_tbl.dropna(subset=["case:RequestedAmount", off_col])
        case_tbl = case_tbl[case_tbl["case:RequestedAmount"] > 0]

        case_tbl["_ratio_"] = (case_tbl[off_col] / case_tbl["case:RequestedAmount"]).clip(0, 1)

        ratios = {
            k: v["_ratio_"].to_numpy()
            for k, v in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType"])
        }

        self.model = {
            "by_pair": ratios,
            "global": case_tbl["_ratio_"].to_numpy(),
        }
        return self

    def predict(
        self,
        loan_goal: str,
        application_type: str,
        requested_amount: float,
        mode: str = "sample",   # "sample" | "mean" | "median"
        n_draws: int = 1,
        seed: int | None = None,
    ) -> float:
        """
        Notebook-Logik: Offered/Requested Ratio resamplen (segment -> global), ratio in [0,1].
        """
        self._require_fitted()
        m = self.model
        assert m is not None

        req = float(requested_amount)
        if not np.isfinite(req) or req <= 0:
            return float("nan")

        arr = m["by_pair"].get((loan_goal, application_type))
        if arr is None or len(arr) == 0:
            arr = m["global"]

        if mode == "mean":
            r = float(np.mean(arr)) if len(arr) else 0.0
            return float(np.clip(r * req, 0.0, req))

        rng = self.rng if seed is None else np.random.default_rng(int(seed))
        n = max(1, int(n_draws))

        draws = rng.choice(arr, size=n, replace=True) if len(arr) else np.zeros(n, dtype=float)
        draws = np.clip(draws.astype(float), 0.0, 1.0)
        offered = np.clip(draws * req, 0.0, req)

        if mode == "median":
            return float(np.median(offered))
        return float(offered[0])

    def validate(
        self,
        df: pd.DataFrame,
        sim_df: pd.DataFrame,
        original_col: str = "OfferedAmount",
        simulated_col: str = "OfferedAmount",
        score_col: str = "CreditScore",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
        bin_col: str = "CreditScore_bin",
        score_bins=(0, 600, 650, 700, 750, 1000),
        score_labels=("<600", "600-649", "650-699", "700-749", "750-999"),
        quantiles=(0.10, 0.50, 0.90),
        include_overall: bool = True,
    ) -> pd.DataFrame:
        # 1) Resolve
        o_amt = resolve_col(df, original_col)
        s_amt = resolve_col(sim_df, simulated_col)
        o_sc = resolve_col(df, score_col)
        s_sc = resolve_col(sim_df, score_col)

        o_gc = [resolve_col(df, c) for c in group_cols]
        s_gc = [resolve_col(sim_df, c) for c in group_cols]

        # 2) Case-level
        orig = to_case_level(df, o_gc + [o_sc, o_amt]).copy()
        sim = to_case_level(sim_df, s_gc + [s_sc, s_amt]).copy()

        # 3) numeric
        orig[o_amt] = pd.to_numeric(orig[o_amt], errors="coerce")
        sim[s_amt] = pd.to_numeric(sim[s_amt], errors="coerce")
        orig[o_sc] = pd.to_numeric(orig[o_sc], errors="coerce")
        sim[s_sc] = pd.to_numeric(sim[s_sc], errors="coerce")
        orig = orig.dropna(subset=o_gc + [o_sc, o_amt])
        sim = sim.dropna(subset=s_gc + [s_sc, s_amt])

        # 4) binning
        orig[bin_col] = pd.cut(orig[o_sc], bins=score_bins, labels=score_labels, right=False, include_lowest=True)
        sim[bin_col] = pd.cut(sim[s_sc], bins=score_bins, labels=score_labels, right=False, include_lowest=True)

        # 5) group -> arrays
        orig_groups = {k: g[o_amt].to_numpy(dtype=float) for k, g in orig.groupby(o_gc + [bin_col], dropna=False)}
        sim_groups = {k: g[s_amt].to_numpy(dtype=float) for k, g in sim.groupby(s_gc + [bin_col], dropna=False)}
        all_keys = set(orig_groups.keys()) | set(sim_groups.keys())

        def _q(arr: np.ndarray, q: float) -> float:
            return float(np.quantile(arr, q)) if arr.size else float("nan")

        def _stats_pair(o: np.ndarray, s: np.ndarray) -> dict:
            o = np.asarray(o, dtype=float)
            s = np.asarray(s, dtype=float)

            out = {
                "orig_n": int(o.size),
                "sim_n": int(s.size),
                "orig_mean": float(np.mean(o)) if o.size else np.nan,
                "sim_mean": float(np.mean(s)) if s.size else np.nan,
                "orig_std": float(np.std(o, ddof=1)) if o.size > 1 else (0.0 if o.size == 1 else np.nan),
                "sim_std": float(np.std(s, ddof=1)) if s.size > 1 else (0.0 if s.size == 1 else np.nan),
            }
            for qv in quantiles:
                qname = f"p{int(round(qv * 100)):02d}"
                out[f"orig_{qname}"] = _q(o, qv)
                out[f"sim_{qname}"] = _q(s, qv)

            if o.size and s.size:
                out["ks"] = float(ks_statistic_1d(o, s))
                out["wasserstein"] = float(wasserstein_approx_1d(o, s))
            else:
                out["ks"] = np.nan
                out["wasserstein"] = np.nan
            return out

        rows = []
        if include_overall:
            rows.append({
                group_cols[0]: "__OVERALL__",
                group_cols[1]: "__OVERALL__",
                bin_col: "__OVERALL__",
                **_stats_pair(orig[o_amt].to_numpy(), sim[s_amt].to_numpy()),
            })

        def _sort_key(k):
            return tuple("" if v is None else str(v) for v in k)

        for k in sorted(all_keys, key=_sort_key):
            o = orig_groups.get(k, np.asarray([]))
            s = sim_groups.get(k, np.asarray([]))
            loan_goal, app_type, cs_bin = k
            rows.append({
                group_cols[0]: loan_goal,
                group_cols[1]: app_type,
                bin_col: str(cs_bin),
                **_stats_pair(o, s),
            })

                # -------------------------------------------------
        # 6) Weighted KS (sim_n * ks)
        # -------------------------------------------------
        df_out = pd.DataFrame(rows)

        # nur echte Gruppen (kein OVERALL, keine NaNs)
        mask = (
            df_out["sim_n"].notna()
            & df_out["ks"].notna()
            & (df_out[group_cols[0]] != "__OVERALL__")
        )

        if mask.any():
            ks_weighted_sum = float((df_out.loc[mask, "sim_n"] * df_out.loc[mask, "ks"]).sum())
            sim_n_sum = int(df_out.loc[mask, "sim_n"].sum())
            ks_weighted_mean = ks_weighted_sum / sim_n_sum if sim_n_sum > 0 else np.nan
        else:
            ks_weighted_sum = np.nan
            ks_weighted_mean = np.nan

        rows.append({
            group_cols[0]: "__KS_WEIGHTED__",
            group_cols[1]: "__SUM__",
            bin_col: "__ALL__",
            "orig_n": np.nan,
            "sim_n": sim_n_sum if mask.any() else np.nan,
            "orig_mean": np.nan,
            "sim_mean": np.nan,
            "orig_std": np.nan,
            "sim_std": np.nan,
            "ks": ks_weighted_sum,
            "wasserstein": np.nan,
        })

        rows.append({
            group_cols[0]: "__KS_WEIGHTED__",
            group_cols[1]: "__MEAN__",
            bin_col: "__ALL__",
            "orig_n": np.nan,
            "sim_n": sim_n_sum if mask.any() else np.nan,
            "orig_mean": np.nan,
            "sim_mean": np.nan,
            "orig_std": np.nan,
            "sim_std": np.nan,
            "ks": ks_weighted_mean,
            "wasserstein": np.nan,
        })

        return pd.DataFrame(rows)
