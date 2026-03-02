"""
DRL-based resource allocation using MaskablePPO.

Re-exports key classes for convenience:
    from resources.drl_allocation import DRLAllocationPolicy, DRLStateBuilder
"""

from resources.drl_allocation.state import DRLStateBuilder
from resources.drl_allocation.policy import (
    DRLAllocationPolicy,
    DRLDecisionRequest,
    DRLDecisionResponse,
    InteractiveBatchPolicy,
)
from resources.drl_allocation.env import ResourceAllocationEnv

__all__ = [
    "DRLStateBuilder",
    "DRLAllocationPolicy",
    "DRLDecisionRequest",
    "DRLDecisionResponse",
    "InteractiveBatchPolicy",
    "ResourceAllocationEnv",
]
