"""Representation and structural analysis of assembled linear systems."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComponentFlowSystem:
    """The variables and equality constraints for a material-balance problem."""

    variable_refs: list
    matrix: np.ndarray
    targets: np.ndarray
    labels: list[str]

    @property
    def normalized_matrix(self) -> np.ndarray:
        """Scale rows before rank checks so flow magnitudes do not dominate."""
        row_scales = np.maximum(np.max(np.abs(self.matrix), axis=1), 1.0)
        return self.matrix / row_scales[:, np.newaxis]

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.normalized_matrix))

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.variable_refs) - self.rank

    @property
    def is_uniquely_determined(self) -> bool:
        return self.rank == len(self.variable_refs)
