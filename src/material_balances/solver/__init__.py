"""Linear-system building blocks used by the material-balance façade."""

from .linear_solver import solve_nonnegative
from .system import ComponentFlowSystem

__all__ = ["ComponentFlowSystem", "solve_nonnegative"]
