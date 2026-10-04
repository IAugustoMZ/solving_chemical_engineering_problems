"""Chemical-component domain model."""

import numpy as np


class Component:
    """A chemical species, optionally with a molar mass."""

    def __init__(self, name: str, molar_mass: float = None) -> None:
        if molar_mass is not None and (not np.isfinite(molar_mass) or molar_mass <= 0):
            raise ValueError("molar_mass must be a finite positive value.")
        self.name = name
        self.molar_mass = molar_mass
