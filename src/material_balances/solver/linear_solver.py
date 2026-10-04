"""Numerical solution of assembled material-balance systems."""

import numpy as np
from scipy.optimize import linprog

from .system import ComponentFlowSystem


def solve_nonnegative(system: ComponentFlowSystem, tolerance: float) -> np.ndarray:
    """Solve a uniquely determined equality system with nonnegative flows."""
    result = linprog(
        c=np.zeros(len(system.variable_refs)),
        A_eq=system.matrix,
        b_eq=system.targets,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        detail = (
            "constraints are infeasible with nonnegative component flows"
            if result.status == 2
            else result.message
        )
        raise ValueError(detail)
    if not system.is_uniquely_determined:
        raise ValueError("constraints allow more than one component-flow solution")
    residuals = system.matrix @ result.x - system.targets
    for residual, target, label in zip(residuals, system.targets, system.labels):
        allowed_error = (
            tolerance
            if label.startswith("material balance")
            else tolerance * max(1.0, abs(target))
        )
        if abs(residual) > allowed_error:
            raise ValueError(
                f"{label} residual {residual:.6g} exceeds tolerance {allowed_error:.6g}."
            )
    return result.x
