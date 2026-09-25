"""
Material balances module for chemical engineering problems.

This module provides classes and functions for modeling and solving
material balance problems in chemical engineering processes.

Main components:
    - Component: Represents a chemical component/species
    - Stream: Represents a process stream with composition
    - ProcessUnit: Represents a process unit with inlet/outlet streams
    - Ratio constraints: Algebraic constraints that reduce degrees of freedom
    - Solvers: Utility functions for solving specific material balance problems
"""

from .components import Component
from .process_unit import ProcessUnit
from .ratio_constraints import ComponentFlowRatio, CompositionRatio, FlowRatio, Ratio
from .stream import Stream

__all__ = [
    "Component",
    "Stream",
    "ProcessUnit",
    "Ratio",
    "FlowRatio",
    "ComponentFlowRatio",
    "CompositionRatio",
]

__version__ = "0.1.0"
