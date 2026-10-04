"""Compatibility imports for material-balance constraint classes.

New code should import from :mod:`src.material_balances.constraints` or from
the package root. This module remains so existing notebooks keep working.
"""

from .constraints import (
    ComponentFlowRatio,
    ComponentFlowValue,
    CompositionRatio,
    FlowRatio,
    Ratio,
)

__all__ = [
    "Ratio",
    "FlowRatio",
    "ComponentFlowRatio",
    "ComponentFlowValue",
    "CompositionRatio",
]
