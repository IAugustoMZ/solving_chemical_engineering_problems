"""
Material balances module for chemical engineering problems.

This module provides classes and functions for modeling and solving
material balance problems in chemical engineering processes.

Main components:
    - Component: Represents a chemical component/species
    - Stream: Represents a process stream with composition
    - ProcessUnit: Represents a process unit with inlet/outlet streams
    - Solvers: Utility functions for solving specific material balance problems
"""

from .components import Component
from .stream import Stream
from .process_unit import ProcessUnit
from .solvers import calculate_stream_based_on_jam_mass

__all__ = [
    "Component",
    "Stream",
    "ProcessUnit",
    "calculate_stream_based_on_jam_mass",
]

__version__ = "0.1.0"
