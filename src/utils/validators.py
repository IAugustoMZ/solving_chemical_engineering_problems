import numpy as np
import math

def check_fraction(value: float, name: str = "Fraction") -> None:
    """
    Checks if a fraction is between 0 and 1 inclusive.
    """
    if not (0 <= value <= 1):
        raise ValueError(f"{name} must be higher or equal to 0 and lower or equal to 1")

def check_fractions_sum(fractions: list, name: str = "Fractions") -> None:
    """
    Checks if an array or list of fractions sums up to 1.
    """
    total = sum(fractions)
    if not math.isclose(total, 1.0, rel_tol=1e-5):
        raise ValueError(f"The {name} must sum up to 1 (got {total})")

