"""
Stream class for representing process streams in material balance problems.

This module provides the Stream class, which represents a stream of material
flowing through a process unit with specified flow rate, composition, and components.
"""

import numpy as np


class Stream:
    """
    Represents a process stream with flow rate and composition.

    A Stream is a fundamental entity in material balance calculations, representing
    a flow of material at a specific point in a process. Each stream has a flow rate,
    composition (mass or mole fractions), and list of components.

    The Stream class automatically validates composition data and can calculate
    missing composition values when all but one component's fraction is specified.

    Attributes:
        name (str): Identifier for the stream (e.g., "Feed", "Product").
        flow_rate (float or None): Volumetric, mass, or molar flow rate. Can be None
                                   if unknown and to be solved for.
        flow_type (str): Type of flow rate (e.g., "mole", "mass", "volume").
        components (list): List of Component objects present in the stream.
        composition (dict[str, float or None]): Mole/mass fractions for each component.
                                                Can contain one None value for unknown.
        direction (str): Direction of flow ("input" or "output").
    """

    def __init__(
        self,
        name: str,
        flow_rate: float,
        flow_type: str,
        components: list,
        composition: dict,
        direction: str = "input",
    ) -> None:
        """
        Initialize a Stream with flow rate and composition data.

        Validates that the number of components matches composition values and
        that compositions sum to 1.0 (if all specified). Automatically calculates
        the complementary composition value if exactly one is None.

        Parameters:
            name (str): Stream identifier or tag.
            flow_rate (float): Flow rate value. Can be None if unknown.
            flow_type (str): Type of flow rate (e.g., "mole", "mass").
            components (list): List of Component objects present in the stream.
            composition (dict): Dictionary mapping component names to fractions.
                               Can have one None value for unknown composition.
            direction (str, optional): Flow direction, "input" or "output".
                                      Defaults to "input".

        Raises:
            ValueError: If number of components doesn't match composition values,
                       or if composition values don't sum to ~1.0 when all specified.

        Example:
            >>> water = Component("Water")
            >>> acetone = Component("Acetone")
            >>> stream = Stream(
            ...     name="evaporator_feed",
            ...     flow_rate=10.0,
            ...     flow_type="mole",
            ...     components=[water, acetone],
            ...     composition={"Water": 0.65, "Acetone": None}
            ... )
            >>> stream.composition["Acetone"]
            0.35
        """
        self.name = name
        self.flow_rate = flow_rate
        self.flow_type = flow_type
        self.components = components
        self.composition = composition
        self.direction = direction

        self._validate_components_composition()
        self._validate_composition_sum()
        self.calculate_complementary_composition()

    def _validate_components_composition(self) -> None:
        """
        Validate that number of components matches number of composition values.

        Raises:
            ValueError: If number of components doesn't match number of composition values.
        """
        if len(self.components) != len(self.composition):
            raise ValueError(
                "Number of components must match the number of composition values."
            )

    def _validate_composition_sum(self) -> None:
        """
        Validate that specified composition values sum to 1.0.

        Only validates if all composition values are specified (no None values).

        Raises:
            ValueError: If specified compositions don't sum to 1.0.
        """
        if all(value is not None for value in self.composition.values()):
            if not np.isclose(sum(self.composition.values()), 1.0):
                raise ValueError("Composition values must sum to 1.")

    def calculate_complementary_composition(self) -> None:
        """
        Calculate missing composition value if exactly one is None.

        If exactly one component's composition is None, calculate it as
        1.0 minus the sum of all other known composition values.

        Raises:
            ValueError: If more than one composition value is missing.
        """
        none_count = sum(1 for v in self.composition.values() if v is None)

        if none_count > 1:
            raise ValueError(
                "Cannot calculate complementary composition: "
                "more than one composition value is missing."
            )

        if none_count == 1:
            known_sum = sum(v for v in self.composition.values() if v is not None)
            missing_component = next(
                key for key, value in self.composition.items() if value is None
            )
            self.composition[missing_component] = 1.0 - known_sum
