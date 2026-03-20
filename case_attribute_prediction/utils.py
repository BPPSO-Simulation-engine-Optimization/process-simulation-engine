from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import pandas as pd


def resolve_col(frame: pd.DataFrame, col: str) -> str:
    """
    Unterstützt case:-Prefix-Varianten:
      - "CreditScore" vs "case:CreditScore"
    """
    if col in frame.columns:
        return col
    if f"case:{col}" in frame.columns:
        return f"case:{col}"
    if col.startswith("case:"):
        plain = col.replace("case:", "", 1)
        if plain in frame.columns:
            return plain
    raise KeyError(f"Spalte '{col}' nicht gefunden. Verfügbare Spalten: {list(frame.columns)[:30]} ...")


def to_case_level(frame: pd.DataFrame, cols: Sequence[str], case_id_col: str = "case:concept:name") -> pd.DataFrame:
    """
    Case-Level wie im Notebook: groupby(case).first()
    """
    if case_id_col in frame.columns:
        return frame.groupby(case_id_col)[list(cols)].first()
    return frame[list(cols)].copy()


def nonempty_arr(x) -> bool:
    if x is None:
        return False
    try:
        return np.asarray(x).size > 0
    except Exception:
        return True


def first_nonempty(*candidates):
    for c in candidates:
        if nonempty_arr(c):
            return c
    return None


def credit_score_bin(
    s: pd.Series,
    bins=(0, 600, 650, 700, 750, 1000),
    labels=("<600", "600-649", "650-699", "700-749", "750-999"),
    unknown_label="UNKNOWN",
) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    b = pd.cut(s_num, bins=bins, labels=labels, right=False, include_lowest=True)
    if isinstance(b.dtype, pd.CategoricalDtype):
        b = b.cat.add_categories([unknown_label]).fillna(unknown_label)
    else:
        b = b.fillna(unknown_label)
    return b


def infer_first_withdrawal_col(df: pd.DataFrame) -> str:
    candidates = [
        "case:FirstWithdrawalAmount",
        "FirstWithdrawalAmount",
        "case:firstWithdrawalAmount",
        "firstWithdrawalAmount",
        "case:FirstWithdrawalamount",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "Keine Spalte für FirstWithdrawalAmount gefunden. "
        "Bitte Spaltennamen prüfen oder FWA-Spalte explizit übergeben."
    )


def detect_rounding_step(values: np.ndarray) -> float | None:
    """
    Heuristik aus dem Notebook (robust): typischste Schrittweite bei vielen Unique-Werten.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 200:
        return None

    u = np.unique(v)
    if u.size < 200:
        return None

    u = np.sort(u)
    diffs = np.diff(u)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.size < 200:
        return None

    med = float(np.median(diffs))
    nice = np.array([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000], dtype=float)
    step = float(nice[np.argmin(np.abs(nice - med))])
    return step if step > 0 else None
