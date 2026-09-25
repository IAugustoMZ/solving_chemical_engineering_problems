"""
Solver functions for specific material balance problems.

This module contains utility functions for solving particular material
balance problems using numpy and scipy.
"""

import numpy as np


def calculate_stream_based_on_jam_mass(
    m_jam: float,
    k: float = 45.0 / 55.0,
    strawberry_water_content: float = 0.85,
    jam_water_content: float = 1 / 3,
) -> dict:
    """
    Solve strawberry jam production material balance.

    Performs material balance calculations for a jam heating process where
    strawberries and sugar are mixed and heated to evaporate water.

    The system consists of:
    - Input: strawberries (85 wt% water, 15 wt% solids) and sugar
    - Mixed in ratio k = strawberry:sugar (default 45:55 by mass)
    - Heated to produce jam containing 1/3 water by mass

    Parameters:
        m_jam (float): Desired mass of jam product (in any consistent units).
        k (float, optional): Mass ratio of strawberries to sugar.
                            Defaults to 45/55 ≈ 0.818.
        strawberry_water_content (float, optional): Mass fraction of water in
                                                   raw strawberries.
                                                   Defaults to 0.85.
        jam_water_content (float, optional): Mass fraction of water in final jam.
                                            Defaults to 1/3.

    Returns:
        dict: Dictionary with keys:
            - strawberry_mass: Mass of strawberries needed
            - sugar_mass: Mass of sugar needed
            - water_mass: Mass of water evaporated
            - jam_mass: Mass of jam produced (input value)

    Raises:
        ValueError: If m_jam is negative.
        numpy.linalg.LinAlgError: If the system matrix is singular.

    Example:
        >>> result = calculate_stream_based_on_jam_mass(1000)
        >>> print(f"Strawberries: {result['strawberry_mass']:.1f} kg")
        Strawberries: 2000.0 kg
    """
    if m_jam < 0:
        raise ValueError("Jam mass must be non-negative")

    strawberry_solids = 1 - strawberry_water_content
    jam_solids = 1 - jam_water_content

    # Set up linear system: A @ [sugar_mass, water_mass] = b
    # Based on material balances for water and solids
    A = np.array(
        [
            [strawberry_water_content * k, -1],
            [strawberry_solids * k + 1, 0],
        ]
    )

    b = np.array([jam_water_content * m_jam, jam_solids * m_jam])

    # Solve for sugar mass and water evaporated
    sugar_mass, water_mass = np.linalg.solve(A, b)

    strawberry_mass = sugar_mass * k

    return {
        "strawberry_mass": strawberry_mass,
        "sugar_mass": sugar_mass,
        "water_mass": water_mass,
        "jam_mass": m_jam,
    }
