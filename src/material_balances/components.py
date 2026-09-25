"""
Component class for representing chemical species in material balance problems.

This module provides the Component class, which represents a chemical component
or species (e.g., water, acetone, sugar) that participates in material balance
calculations.
"""


class Component:
    """
    Represents a chemical component or species in a process stream.

    A Component is a fundamental entity in material balance calculations,
    representing individual chemical species (e.g., H2O, C2H5OH, C12H22O11).
    Components are referenced by name and used in composition calculations
    for streams and material balances for process units.

    Attributes:
        name (str): The name or identifier of the component (e.g., "Water",
                    "Acetone", "Sugar").

    Example:
        >>> water = Component("Water")
        >>> water.name
        'Water'
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a Component with a given name.

        Parameters:
            name (str): The name or identifier of the chemical component.

        Example:
            >>> acetone = Component("Acetone")
        """
        self.name = name
