from __future__ import annotations

from .registry import PredictorRegistry, build_default_registry

# Optional: Backwards-Kompatibilität – eine globale Registry
_DEFAULT_REGISTRY: PredictorRegistry | None = None


def init_all(df, seed: int = 42) -> PredictorRegistry:
    """
    Ein Einstiegspunkt wie im Notebook: init_* in einem Call.
    """
    global _DEFAULT_REGISTRY
    reg = build_default_registry(seed=seed).fit_all(df)
    _DEFAULT_REGISTRY = reg
    return reg


def get_registry() -> PredictorRegistry:
    if _DEFAULT_REGISTRY is None:
        raise RuntimeError("Registry nicht initialisiert. Bitte init_all(df) aufrufen.")
    return _DEFAULT_REGISTRY
