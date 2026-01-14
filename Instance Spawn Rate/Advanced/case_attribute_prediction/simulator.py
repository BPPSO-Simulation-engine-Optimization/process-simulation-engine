from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Callable, Optional, Any, Dict
import numpy as np
import pandas as pd

from .registry import (
    build_default_registry,
    PredictorRegistry,
    registry_models_to_dict,
    load_models_into_registry,
)
from .utils import resolve_col, to_case_level
from .metrics import ks_statistic_1d, wasserstein_approx_1d


def _compute_case_sampler_artifact(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    loan_goal_col: str = "case:LoanGoal",
    app_type_col: str = "case:ApplicationType",
    requested_col: str = "case:RequestedAmount",
) -> dict:
    """
    Berechnet alle benötigten Verteilungen für die Case-Sampler und gibt sie
    als serialisierbares Artefakt zurück.
    """
    for c in (loan_goal_col, app_type_col, requested_col):
        if c not in df.columns:
            raise KeyError(f"Fehlende Spalte im Input df: '{c}'")

    case_tbl = df.groupby(case_col)[[loan_goal_col, app_type_col, requested_col]].first().copy()
    case_tbl[requested_col] = pd.to_numeric(case_tbl[requested_col], errors="coerce")
    case_tbl = case_tbl.dropna(subset=[loan_goal_col, app_type_col, requested_col])
    case_tbl = case_tbl[case_tbl[requested_col] > 0]

    # P(LoanGoal)
    lg_p = case_tbl[loan_goal_col].value_counts(normalize=True)
    lg_vals = lg_p.index.to_numpy()
    lg_probs = lg_p.to_numpy()

    # P(AppType | LoanGoal)
    at_marg = case_tbl[app_type_col].value_counts(normalize=True)
    at_vals = at_marg.index.to_numpy()
    at_probs = at_marg.to_numpy()

    at_cond: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lg, sub in case_tbl.groupby(loan_goal_col):
        p = sub[app_type_col].value_counts(normalize=True)
        at_cond[str(lg)] = (p.index.to_numpy(), p.to_numpy())

    # RequestedAmount | (lg, at) + Fallbacks
    req_global = case_tbl[requested_col].to_numpy(dtype=float)
    req_by_pair = {
        (str(lg), str(at)): sub[requested_col].to_numpy(dtype=float)
        for (lg, at), sub in case_tbl.groupby([loan_goal_col, app_type_col])
    }
    req_by_lg = {
        str(lg): sub[requested_col].to_numpy(dtype=float)
        for lg, sub in case_tbl.groupby(loan_goal_col)
    }

    return {
        "lg_vals": lg_vals,
        "lg_probs": lg_probs,
        "at_vals": at_vals,
        "at_probs": at_probs,
        "at_cond": at_cond,
        "req_global": req_global,
        "req_by_pair": req_by_pair,
        "req_by_lg": req_by_lg,
    }


def _build_case_samplers_from_artifact(
    artifact: dict,
    seed: int = 42,
) -> tuple[
    Callable[[], str],
    Callable[[str], str],
    Callable[[str, str], float],
]:
    """
    Baut die Case-Sampler-Funktionen aus einem zuvor berechneten Artefakt.
    """
    rng = np.random.default_rng(seed)

    lg_vals = artifact["lg_vals"]
    lg_probs = artifact["lg_probs"]
    at_vals = artifact["at_vals"]
    at_probs = artifact["at_probs"]
    at_cond = artifact["at_cond"]
    req_global = artifact["req_global"]
    req_by_pair = artifact["req_by_pair"]
    req_by_lg = artifact["req_by_lg"]

    def draw_lg() -> str:
        return str(rng.choice(lg_vals, p=lg_probs))

    def draw_at(loan_goal: str) -> str:
        vals_probs = at_cond.get(str(loan_goal))
        if vals_probs is None:
            return str(rng.choice(at_vals, p=at_probs))
        vals, probs = vals_probs
        return str(rng.choice(vals, p=probs))

    def draw_requested_amount(loan_goal: str, application_type: str) -> float:
        arr = req_by_pair.get((str(loan_goal), str(application_type)))
        if arr is None or len(arr) == 0:
            arr = req_by_lg.get(str(loan_goal))
        if arr is None or len(arr) == 0:
            arr = req_global
        return float(rng.choice(arr))

    return draw_lg, draw_at, draw_requested_amount


def _build_case_samplers(
    df: pd.DataFrame,
    seed: int = 42,
    case_col: str = "case:concept:name",
    loan_goal_col: str = "case:LoanGoal",
    app_type_col: str = "case:ApplicationType",
    requested_col: str = "case:RequestedAmount",
) -> tuple[
    Callable[[], str],
    Callable[[str], str],
    Callable[[str, str], float],
]:
    """
    Convenience-Wrapper: berechnet das Artefakt aus dem df und baut daraus
    die Sampler-Funktionen.
    """
    artifact = _compute_case_sampler_artifact(
        df,
        case_col=case_col,
        loan_goal_col=loan_goal_col,
        app_type_col=app_type_col,
        requested_col=requested_col,
    )
    return _build_case_samplers_from_artifact(artifact, seed=seed)


def _compute_monthly_rate_artifact(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    monthly_col: str = "MonthlyCost",
    offered_col: str = "OfferedAmount",
) -> dict:
    """
    Berechnet die Verteilung der Rate = MonthlyCost/OfferedAmount auf Case-Level.
    """
    mc = resolve_col(df, monthly_col)
    oa = resolve_col(df, offered_col)

    case_tbl = df.groupby(case_col)[[mc, oa]].first().copy()
    case_tbl[mc] = pd.to_numeric(case_tbl[mc], errors="coerce")
    case_tbl[oa] = pd.to_numeric(case_tbl[oa], errors="coerce")
    case_tbl = case_tbl.dropna(subset=[mc, oa])
    case_tbl = case_tbl[(case_tbl[oa] > 0) & (case_tbl[mc] >= 0)]

    rate = (case_tbl[mc] / case_tbl[oa]).to_numpy(dtype=float)
    rate = rate[np.isfinite(rate)]
    if rate.size == 0:
        rate = np.array([0.01], dtype=float)

    return {"rate": rate}


def _build_monthly_rate_sampler_from_artifact(
    artifact: dict,
    seed: int = 42,
) -> Callable[[float], float]:
    """
    Baut den Monthly-Rate-Sampler aus einem zuvor berechneten Artefakt.
    """
    rng = np.random.default_rng(seed)
    rate = np.asarray(artifact["rate"], dtype=float)

    def draw_monthly(offered_amount: float) -> float:
        r = float(rng.choice(rate))
        return float(max(r * float(offered_amount), 0.0))

    return draw_monthly


def _build_monthly_rate_sampler(
    df: pd.DataFrame,
    seed: int = 42,
    case_col: str = "case:concept:name",
    monthly_col: str = "MonthlyCost",
    offered_col: str = "OfferedAmount",
) -> Callable[[float], float]:
    """
    Convenience-Wrapper: berechnet das Artefakt aus dem df und baut daraus
    den Monthly-Rate-Sampler.
    """
    artifact = _compute_monthly_rate_artifact(
        df,
        case_col=case_col,
        monthly_col=monthly_col,
        offered_col=offered_col,
    )
    return _build_monthly_rate_sampler_from_artifact(artifact, seed=seed)


@dataclass
class CaseState:
    case_id: str
    loan_goal: str
    application_type: str
    requested_amount: float

    # Offer-abhängige Attribute (werden einmalig gezogen)
    credit_score: float = np.nan
    offered_amount: float = np.nan
    first_withdrawal_amount: float = np.nan
    number_of_terms: float = np.nan
    monthly_cost: float = np.nan
    selected: Optional[bool] = None
    accepted: Optional[bool] = None


class AttributeSimulationEngine:
    """
    Einmal initialisieren (fit + sampler), danach pro Event:
      simulated_event_data = draw_case_and_event_attributes(next_activity)

    Kerneigenschaft:
      - Attribute werden pro Case nur einmal gezogen (Cache im CaseState)
      - Offer-abhängige Attribute werden erst gezogen, wenn die Aktivität das erfordert
        (default: "O_Create Offer")
    """

    def __init__(
        self,
        df: pd.DataFrame | None,
        seed: int = 42,
        monthly_artifact: dict | None = None,
        offer_create_activity: str = "O_Create Offer",
        model_store_path: str | Path | None = None,
        retrain_models: bool = False,
    ):
        """
        df kann None sein, wenn ausschließlich aus gespeicherten Artefakten
        (Modelle + Verteilungen) simuliert werden soll.
        """
        self.df = df
        self.seed = int(seed)
        self.offer_create_activity = str(offer_create_activity)

        # Pfad für Modellartefakte (Pickle); default neben dieser Datei
        self.model_store_path = Path(model_store_path) if model_store_path is not None else Path(__file__).with_name("attribute_models.pkl")
        # Pfad für Verteilungs-Artefakte (Case-Sampler + Monthly-Rate)
        self.dist_store_path = Path(__file__).with_name("attribute_distributions.pkl")
        self.retrain_models = bool(retrain_models)

        # Registry + Fit (einmal)
        self.registry: PredictorRegistry = build_default_registry(seed=self.seed)
        self._init_models(df)

        # MonthlyCost: Artefakt oder fallback rate sampler
        self.monthly_artifact = monthly_artifact
        self.monthly_rate_sampler = None

        if monthly_artifact is not None:
            self.registry.monthly_cost.set_artifact(monthly_artifact)
        # Fallback-Rate-Sampler oder Laden aus Artefakt
        # sowie Case-Sampler:
        if df is not None:
            # Sampler direkt aus df bauen und Artefakt persistieren
            self.monthly_rate_sampler = _build_monthly_rate_sampler(df, seed=self.seed)
            self.draw_lg, self.draw_at, self.draw_requested_amount = _build_case_samplers(df, seed=self.seed)

            dist_artifact = {
                "case": _compute_case_sampler_artifact(df),
                "monthly_rate": _compute_monthly_rate_artifact(df),
            }
            try:
                with self.dist_store_path.open("wb") as f:
                    pickle.dump(dist_artifact, f)
            except Exception:
                # Persistence-Fehler sollen die Simulation nicht verhindern
                pass
        else:
            # df ist None: wir erwarten vortrainierte Verteilungs-Artefakte
            if not self.dist_store_path.exists():
                raise ValueError(
                    f"Kein df übergeben und Verteilungs-Artefakt '{self.dist_store_path}' nicht gefunden. "
                    f"Bitte zuerst einmal mit retrain_models=True und df trainieren."
                )
            with self.dist_store_path.open("rb") as f:
                dist_artifact = pickle.load(f)

            case_art = dist_artifact.get("case")
            monthly_art = dist_artifact.get("monthly_rate")
            if case_art is None or monthly_art is None:
                raise ValueError(
                    f"Verteilungs-Artefakt '{self.dist_store_path}' ist unvollständig oder korrupt."
                )

            self.draw_lg, self.draw_at, self.draw_requested_amount = _build_case_samplers_from_artifact(
                case_art,
                seed=self.seed,
            )

            if monthly_artifact is None:
                self.monthly_rate_sampler = _build_monthly_rate_sampler_from_artifact(
                    monthly_art,
                    seed=self.seed,
                )

        # interner Zustand
        self._case_counter = 0
        self._active_case: CaseState | None = None

    def _init_models(self, df: pd.DataFrame | None):
        """
        Modelle entweder laden oder neu fitten + persistieren.
        """
        models_loaded = False
        if not self.retrain_models and self.model_store_path.exists():
            try:
                with self.model_store_path.open("rb") as f:
                    models = pickle.load(f)
                load_models_into_registry(self.registry, models)
                models_loaded = True
            except Exception:
                # Fallback auf Neu-Training, wenn Laden fehlschlägt
                models_loaded = False

        if not models_loaded:
            # Fit einmalig – benötigt ein df
            if df is None:
                raise ValueError(
                    "Kein df übergeben, aber Modelle müssen neu trainiert werden. "
                    "Bitte ein Trainings-DataFrame übergeben oder retrain_models=False setzen."
                )
            self.registry.credit_score.fit(df)
            self.registry.offered_amount.fit(df)
            self.registry.first_withdrawal_amount.fit(df)
            self.registry.number_of_terms.fit(df)
            self.registry.selected.fit(df)
            self.registry.accepted.fit(df)
            self.registry.monthly_cost.fit(df)
            # persistieren
            try:
                with self.model_store_path.open("wb") as f:
                    pickle.dump(registry_models_to_dict(self.registry), f)
            except Exception:
                # Persistence failures sollen Simulation nicht stoppen
                pass

    def start_new_case(self, case_id: str | None = None) -> CaseState:
        """
        Zieht Basis-Case-Attribute (LoanGoal, ApplicationType, RequestedAmount) genau einmal.
        """
        self._case_counter += 1
        if case_id is None:
            case_id = f"SIM_{self._case_counter:07d}"

        lg = self.draw_lg()
        at = self.draw_at(lg)
        req = self.draw_requested_amount(lg, at)

        self._active_case = CaseState(
            case_id=str(case_id),
            loan_goal=str(lg),
            application_type=str(at),
            requested_amount=float(req),
        )
        return self._active_case

    def _ensure_case(self):
        if self._active_case is None:
            self.start_new_case()

    def end_current_case(self) -> None:
        """
        Beendet den aktuell aktiven Case.
        Beim nächsten draw_case_attributes()/start_new_case()
        wird automatisch ein neuer Case begonnen.
        """
        self._active_case = None

    def _draw_offer_dependent_attributes_once(self):
        """
        Zieht CreditScore/OfferedAmount/FWA/Terms/Monthly/Selected/Accepted nur,
        wenn noch nicht gezogen.
        """
        assert self._active_case is not None
        cs = self._active_case

        if np.isfinite(cs.offered_amount):
            # bereits gezogen
            return

        # deterministischer Case-Seed für Reproduzierbarkeit
        case_seed = self.seed + self._case_counter

        cs.credit_score = float(self.registry.credit_score.predict(cs.loan_goal, cs.application_type))

        cs.offered_amount = float(
            self.registry.offered_amount.predict(
                cs.loan_goal, cs.application_type, cs.requested_amount,
                mode="sample", n_draws=1, seed=case_seed
            )
        )

        cs.first_withdrawal_amount = float(
            self.registry.first_withdrawal_amount.predict(
                cs.loan_goal, cs.application_type, cs.credit_score,
                requested_amount=cs.requested_amount,
                offered_amount=cs.offered_amount,
                mode="sample",
                seed=case_seed,
            )
        )

        cs.number_of_terms = int(
            self.registry.number_of_terms.predict(
                offered_amount=cs.offered_amount,
                credit_score=cs.credit_score,
                loan_goal=cs.loan_goal,
                application_type=cs.application_type,
            )
        )

        if self.monthly_artifact is not None:
            cs.monthly_cost = float(
                self.registry.monthly_cost.predict(
                    offered_amount=cs.offered_amount,
                    number_of_terms=int(cs.number_of_terms),
                    credit_score=cs.credit_score,
                    application_type=cs.application_type,
                )
            )
        else:
            assert self.monthly_rate_sampler is not None
            cs.monthly_cost = float(self.monthly_rate_sampler(cs.offered_amount))

        cs.selected = bool(self.registry.selected.predict(cs.loan_goal, cs.application_type, cs.credit_score))
        cs.accepted = bool(self.registry.accepted.predict(cs.monthly_cost, cs.credit_score))

    def populate_offer_attributes(self, case_state: CaseState) -> None:
        """
        Populiert Offer-abhängige Attribute direkt auf dem übergebenen CaseState.
        
        Diese Methode ist für die DES-Engine gedacht, die viele Cases parallel
        verarbeitet. Im Gegensatz zu _draw_offer_dependent_attributes_once()
        verwendet sie NICHT den internen _active_case Pointer, sondern operiert
        direkt auf dem übergebenen CaseState-Objekt.
        
        Args:
            case_state: Das CaseState-Objekt, auf dem die Attribute gesetzt werden.
        """
        cs = case_state

        # Bereits gezogen -> überspringen
        if np.isfinite(cs.offered_amount):
            return

        # Deterministischer Seed aus case_id für Reproduzierbarkeit
        case_seed = self.seed + hash(cs.case_id) % 100000

        cs.credit_score = float(self.registry.credit_score.predict(cs.loan_goal, cs.application_type))

        cs.offered_amount = float(
            self.registry.offered_amount.predict(
                cs.loan_goal, cs.application_type, cs.requested_amount,
                mode="sample", n_draws=1, seed=case_seed
            )
        )

        cs.first_withdrawal_amount = float(
            self.registry.first_withdrawal_amount.predict(
                cs.loan_goal, cs.application_type, cs.credit_score,
                requested_amount=cs.requested_amount,
                offered_amount=cs.offered_amount,
                mode="sample",
                seed=case_seed,
            )
        )

        cs.number_of_terms = int(
            self.registry.number_of_terms.predict(
                offered_amount=cs.offered_amount,
                credit_score=cs.credit_score,
                loan_goal=cs.loan_goal,
                application_type=cs.application_type,
            )
        )

        if self.monthly_artifact is not None:
            cs.monthly_cost = float(
                self.registry.monthly_cost.predict(
                    offered_amount=cs.offered_amount,
                    number_of_terms=int(cs.number_of_terms),
                    credit_score=cs.credit_score,
                    application_type=cs.application_type,
                )
            )
        else:
            assert self.monthly_rate_sampler is not None
            cs.monthly_cost = float(self.monthly_rate_sampler(cs.offered_amount))

        cs.selected = bool(self.registry.selected.predict(cs.loan_goal, cs.application_type, cs.credit_score))
        cs.accepted = bool(self.registry.accepted.predict(cs.monthly_cost, cs.credit_score))

    def _validate_requested_amount(
        self,
        df: pd.DataFrame,
        sim_df: pd.DataFrame,
        requested_col: str = "case:RequestedAmount",
        per_group: bool = True,
        print_results: bool = True,
    ):
        """
        Validiert simulierte RequestedAmount gegen Original-Daten.
        Gruppiert nach LoanGoal und ApplicationType.
        """
        o_col = resolve_col(df, requested_col)
        s_col = resolve_col(sim_df, requested_col)

        orig = to_case_level(df, ["case:LoanGoal", "case:ApplicationType", "case:concept:name", o_col]).copy()
        sim = sim_df[["case:LoanGoal", "case:ApplicationType", s_col]].copy()

        orig[o_col] = pd.to_numeric(orig[o_col], errors="coerce")
        sim[s_col] = pd.to_numeric(sim[s_col], errors="coerce")
        orig = orig.dropna(subset=[o_col])
        sim = sim.dropna(subset=[s_col])
        orig = orig[orig[o_col] > 0]
        sim = sim[sim[s_col] > 0]

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

        x_all = orig[o_col].to_numpy()
        y_all = sim[s_col].to_numpy()

        overall = {
            **{f"orig_{k}": v for k, v in summary(x_all).items()},
            **{f"sim_{k}": v for k, v in summary(y_all).items()},
        }

        if len(x_all) > 0 and len(y_all) > 0:
            overall["ks"] = ks_statistic_1d(x_all, y_all)
            overall["wasserstein"] = wasserstein_approx_1d(x_all, y_all)
            overall["mean_diff"] = float(np.mean(y_all) - np.mean(x_all))
            overall["mean_diff_pct"] = float((np.mean(y_all) - np.mean(x_all)) / np.mean(x_all) * 100) if np.mean(x_all) > 0 else np.nan
            overall["median_diff"] = float(np.median(y_all) - np.median(x_all))
        else:
            overall["ks"] = np.nan
            overall["wasserstein"] = np.nan
            overall["mean_diff"] = np.nan
            overall["mean_diff_pct"] = np.nan
            overall["median_diff"] = np.nan

        overall_df = pd.DataFrame([overall])

        if not per_group:
            return overall_df, None

        rows = []
        og = orig.groupby(["case:LoanGoal", "case:ApplicationType"])
        sg = sim.groupby(["case:LoanGoal", "case:ApplicationType"])
        keys = set(og.groups.keys()) | set(sg.groups.keys())

        for k in keys:
            o_vals = og.get_group(k)[o_col].to_numpy() if k in og.groups else np.array([])
            s_vals = sg.get_group(k)[s_col].to_numpy() if k in sg.groups else np.array([])

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
                row["mean_diff"] = float(np.mean(s_vals) - np.mean(o_vals))
                row["mean_diff_pct"] = float((np.mean(s_vals) - np.mean(o_vals)) / np.mean(o_vals) * 100) if np.mean(o_vals) > 0 else np.nan
            else:
                row["ks"] = np.nan
                row["wasserstein"] = np.nan
                row["mean_diff"] = np.nan
                row["mean_diff_pct"] = np.nan

            rows.append(row)

        per_group_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        
        if print_results:
            print("\n=== VALIDATION: Requested Amount ===")
            print("\n--- Overall Statistics ---")
            print(overall_df)
            
            if per_group_df is not None and len(per_group_df) > 0:
                print("\n--- Per Group Statistics (LoanGoal, ApplicationType) ---")
                cols_to_show = ["case:LoanGoal", "case:ApplicationType", "orig_n", "sim_n", 
                                "orig_mean", "sim_mean", "orig_p50", "sim_p50", 
                                "mean_diff", "mean_diff_pct", "ks", "wasserstein"]
                available_cols = [c for c in cols_to_show if c in per_group_df.columns]
                print(per_group_df[available_cols].head(20))
                if len(per_group_df) > 20:
                    print(f"... ({len(per_group_df) - 20} weitere Gruppen)")
        
        return overall_df, per_group_df

    def validate(
        self,
        sim_df: pd.DataFrame,
        *,
        monthly_cost: bool = False,
        credit_score: bool = False,
        offered_amount: bool = False,
        requested_amount: bool = False,
        number_of_terms_dist = False,
        number_of_terms_mae = False,
        first_withdrawal: bool = False,
        selected: bool = False,
        accepted: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Führt gezielte Validierungen gegen das Original-df durch.
        Nutzt ausschließlich die validate()-Methoden der Predictor.
        """
        results: dict[str, pd.DataFrame] = {}

        if number_of_terms_dist:
            overall, per_group = self.registry.number_of_terms.validate(
                self.df,
                sim_df,
                original_col="NumberOfTerms",
                simulated_col="NumberOfTerms",
                per_group=True,
                print_results=True,
            )
            results["number_of_terms_dist"] = overall
            if per_group is not None and len(per_group) > 0:
                results["number_of_terms_dist_per_group"] = per_group

        if number_of_terms_mae:
            overall, by_month, baseline, _ = (
                self.registry.number_of_terms.validate_predict_no_of_terms_mae(
                    self.df,
                    seed=self.seed,
                )
            )
            results["number_of_terms_mae_overall"] = pd.DataFrame([overall])
            results["number_of_terms_mae_by_month"] = by_month
            results["number_of_terms_mae_baseline"] = pd.DataFrame([baseline])
            
            print("\n=== VALIDATION: Number of Terms (MAE) ===")
            print("\n--- MAE Overall ---")
            print(results["number_of_terms_mae_overall"])
            print("\n--- MAE Baseline ---")
            print(results["number_of_terms_mae_baseline"])


        if credit_score:
            overall, per_group, bin_dist = self.registry.credit_score.validate(
                self.df,
                sim_df,
                original_cs_col="CreditScore",
                simulated_cs_col="CreditScore",
                per_group=True,
                print_results=True,
            )
            results["credit_score_overall"] = overall
            if per_group is not None:
                results["credit_score_per_group"] = per_group
            if bin_dist is not None and len(bin_dist) > 0:
                results["credit_score_bin_distribution"] = bin_dist

        if offered_amount:
            results["offered_amount"] = self.registry.offered_amount.validate(
                self.df,
                sim_df,
                original_col="OfferedAmount",
                simulated_col="OfferedAmount",
                score_col="CreditScore",
                print_results=True,
            )

        if first_withdrawal:
            results["first_withdrawal_amount"] = self.registry.first_withdrawal_amount.validate(
                self.df,
                sim_df,
                col="FirstWithdrawalAmount",
                print_results=True,
            )

        if selected:
            results["selected"] = self.registry.selected.validate(
                self.df,
                sim_df,
                col="Selected",
                print_results=True,
            )

        if accepted:
            results["accepted"] = self.registry.accepted.validate(
                self.df,
                sim_df,
                col="Accepted",
                print_results=True,
            )

        if monthly_cost:
            results["monthly_cost"] = self.registry.monthly_cost.validate(
                self.df,
                sim_df,
                original_col="MonthlyCost",
                simulated_col="MonthlyCost",
                print_results=True,
            )

        if requested_amount:
            overall, per_group = self._validate_requested_amount(
                self.df,
                sim_df,
                requested_col="case:RequestedAmount",
                per_group=True,
                print_results=True,
            )
            results["requested_amount_overall"] = overall
            if per_group is not None and len(per_group) > 0:
                results["requested_amount_per_group"] = per_group

        return results


    def draw_case_attributes(
        self,
        *,
        case_id: str | None = None,
        include_case_prefix_cols: bool = True,
    ) -> dict[str, Any]:
        """
        Liefert nur die Case-bezogenen Attribute (ohne Event-spezifische Felder).
        Diese Werte bleiben für alle Events eines Cases konstant.
        """
        if self._active_case is None:
            self.start_new_case(case_id=case_id)
        elif case_id is not None and self._active_case.case_id != case_id:
            # Falls der Caller explizit Case-ID wechselt -> neuen Case starten
            self.start_new_case(case_id=case_id)

        assert self._active_case is not None
        cs = self._active_case

        out: dict[str, Any] = {
            "case:concept:name": cs.case_id,
        }

        if include_case_prefix_cols:
            out.update({
                "case:LoanGoal": cs.loan_goal,
                "case:ApplicationType": cs.application_type,
                "case:RequestedAmount": cs.requested_amount,
            })
        else:
            out.update({
                "LoanGoal": cs.loan_goal,
                "ApplicationType": cs.application_type,
                "RequestedAmount": cs.requested_amount,
            })

        return out

    def draw_event_attributes(
        self,
        next_activity: str,
        *,
        extra_event_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Liefert nur die Event-bezogenen Attribute:
        - setzt bei O_Create Offer (offer_create_activity) die Offer-abhängigen Attribute,
        - für andere Aktivitäten bleiben diese Attribute NaN/None (entspricht \"kein Offer-Event\").
        """
        self._ensure_case()
        assert self._active_case is not None
        cs = self._active_case

        # Nur bei Offer-Creation wird das Bündel gezogen
        if str(next_activity) == self.offer_create_activity:
            self._draw_offer_dependent_attributes_once()

        out: dict[str, Any] = {
            "concept:name": str(next_activity),
            "CreditScore": cs.credit_score,
            "OfferedAmount": cs.offered_amount,
            "FirstWithdrawalAmount": cs.first_withdrawal_amount,
            "NumberOfTerms": cs.number_of_terms,
            "MonthlyCost": cs.monthly_cost,
            "Selected": cs.selected,
            "Accepted": cs.accepted,
        }

        if extra_event_fields:
            out.update(dict(extra_event_fields))

        return out

    def draw_case_and_event_attributes(
        self,
        next_activity: str,
        *,
        case_id: str | None = None,
        include_case_prefix_cols: bool = True,
        extra_event_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Convenience-Wrapper für Legacy-Code:
        kombiniert draw_case_attributes() und draw_event_attributes().

        Wird perspektivisch durch die getrennten Methoden ersetzt.
        """
        case_attrs = self.draw_case_attributes(
            case_id=case_id,
            include_case_prefix_cols=include_case_prefix_cols,
        )
        event_attrs = self.draw_event_attributes(
            next_activity,
            extra_event_fields=extra_event_fields,
        )
        out: dict[str, Any] = {}
        out.update(case_attrs)
        out.update(event_attrs)
        return out

    def finalize_case_row(self) -> dict[str, Any]:
        """
        Liefert eine Case-Level-Zeile. Wenn Offer-Attribute noch nie gezogen wurden,
        bleiben sie NaN/None.
        """
        self._ensure_case()
        assert self._active_case is not None
        cs = self._active_case
        return {
            "case:concept:name": cs.case_id,
            "case:LoanGoal": cs.loan_goal,
            "case:ApplicationType": cs.application_type,
            "case:RequestedAmount": cs.requested_amount,
            "CreditScore": cs.credit_score,
            "OfferedAmount": cs.offered_amount,
            "FirstWithdrawalAmount": cs.first_withdrawal_amount,
            "NumberOfTerms": cs.number_of_terms,
            "MonthlyCost": cs.monthly_cost,
            "Selected": cs.selected,
            "Accepted": cs.accepted,
        }
